
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Abstract linear algebra library.
2: 
3: This module defines a class hierarchy that implements a kind of "lazy"
4: matrix representation, called the ``LinearOperator``. It can be used to do
5: linear algebra with extremely large sparse or structured matrices, without
6: representing those explicitly in memory. Such matrices can be added,
7: multiplied, transposed, etc.
8: 
9: As a motivating example, suppose you want have a matrix where almost all of
10: the elements have the value one. The standard sparse matrix representation
11: skips the storage of zeros, but not ones. By contrast, a LinearOperator is
12: able to represent such matrices efficiently. First, we need a compact way to
13: represent an all-ones matrix::
14: 
15:     >>> import numpy as np
16:     >>> class Ones(LinearOperator):
17:     ...     def __init__(self, shape):
18:     ...         super(Ones, self).__init__(dtype=None, shape=shape)
19:     ...     def _matvec(self, x):
20:     ...         return np.repeat(x.sum(), self.shape[0])
21: 
22: Instances of this class emulate ``np.ones(shape)``, but using a constant
23: amount of storage, independent of ``shape``. The ``_matvec`` method specifies
24: how this linear operator multiplies with (operates on) a vector. We can now
25: add this operator to a sparse matrix that stores only offsets from one::
26: 
27:     >>> from scipy.sparse import csr_matrix
28:     >>> offsets = csr_matrix([[1, 0, 2], [0, -1, 0], [0, 0, 3]])
29:     >>> A = aslinearoperator(offsets) + Ones(offsets.shape)
30:     >>> A.dot([1, 2, 3])
31:     array([13,  4, 15])
32: 
33: The result is the same as that given by its dense, explicitly-stored
34: counterpart::
35: 
36:     >>> (np.ones(A.shape, A.dtype) + offsets.toarray()).dot([1, 2, 3])
37:     array([13,  4, 15])
38: 
39: Several algorithms in the ``scipy.sparse`` library are able to operate on
40: ``LinearOperator`` instances.
41: '''
42: 
43: from __future__ import division, print_function, absolute_import
44: 
45: import numpy as np
46: 
47: from scipy.sparse import isspmatrix
48: from scipy.sparse.sputils import isshape, isintlike
49: 
50: __all__ = ['LinearOperator', 'aslinearoperator']
51: 
52: 
53: class LinearOperator(object):
54:     '''Common interface for performing matrix vector products
55: 
56:     Many iterative methods (e.g. cg, gmres) do not need to know the
57:     individual entries of a matrix to solve a linear system A*x=b.
58:     Such solvers only require the computation of matrix vector
59:     products, A*v where v is a dense vector.  This class serves as
60:     an abstract interface between iterative solvers and matrix-like
61:     objects.
62: 
63:     To construct a concrete LinearOperator, either pass appropriate
64:     callables to the constructor of this class, or subclass it.
65: 
66:     A subclass must implement either one of the methods ``_matvec``
67:     and ``_matmat``, and the attributes/properties ``shape`` (pair of
68:     integers) and ``dtype`` (may be None). It may call the ``__init__``
69:     on this class to have these attributes validated. Implementing
70:     ``_matvec`` automatically implements ``_matmat`` (using a naive
71:     algorithm) and vice-versa.
72: 
73:     Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
74:     to implement the Hermitian adjoint (conjugate transpose). As with
75:     ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
76:     ``_adjoint`` implements the other automatically. Implementing
77:     ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
78:     backwards compatibility.
79: 
80:     Parameters
81:     ----------
82:     shape : tuple
83:         Matrix dimensions (M,N).
84:     matvec : callable f(v)
85:         Returns returns A * v.
86:     rmatvec : callable f(v)
87:         Returns A^H * v, where A^H is the conjugate transpose of A.
88:     matmat : callable f(V)
89:         Returns A * V, where V is a dense matrix with dimensions (N,K).
90:     dtype : dtype
91:         Data type of the matrix.
92: 
93:     Attributes
94:     ----------
95:     args : tuple
96:         For linear operators describing products etc. of other linear
97:         operators, the operands of the binary operation.
98: 
99:     See Also
100:     --------
101:     aslinearoperator : Construct LinearOperators
102: 
103:     Notes
104:     -----
105:     The user-defined matvec() function must properly handle the case
106:     where v has shape (N,) as well as the (N,1) case.  The shape of
107:     the return type is handled internally by LinearOperator.
108: 
109:     LinearOperator instances can also be multiplied, added with each
110:     other and exponentiated, all lazily: the result of these operations
111:     is always a new, composite LinearOperator, that defers linear
112:     operations to the original operators and combines the results.
113: 
114:     Examples
115:     --------
116:     >>> import numpy as np
117:     >>> from scipy.sparse.linalg import LinearOperator
118:     >>> def mv(v):
119:     ...     return np.array([2*v[0], 3*v[1]])
120:     ...
121:     >>> A = LinearOperator((2,2), matvec=mv)
122:     >>> A
123:     <2x2 _CustomLinearOperator with dtype=float64>
124:     >>> A.matvec(np.ones(2))
125:     array([ 2.,  3.])
126:     >>> A * np.ones(2)
127:     array([ 2.,  3.])
128: 
129:     '''
130:     def __new__(cls, *args, **kwargs):
131:         if cls is LinearOperator:
132:             # Operate as _CustomLinearOperator factory.
133:             return _CustomLinearOperator(*args, **kwargs)
134:         else:
135:             obj = super(LinearOperator, cls).__new__(cls)
136: 
137:             if (type(obj)._matvec == LinearOperator._matvec
138:                     and type(obj)._matmat == LinearOperator._matmat):
139:                 raise TypeError("LinearOperator subclass should implement"
140:                                 " at least one of _matvec and _matmat.")
141: 
142:             return obj
143: 
144:     def __init__(self, dtype, shape):
145:         '''Initialize this LinearOperator.
146: 
147:         To be called by subclasses. ``dtype`` may be None; ``shape`` should
148:         be convertible to a length-2 tuple.
149:         '''
150:         if dtype is not None:
151:             dtype = np.dtype(dtype)
152: 
153:         shape = tuple(shape)
154:         if not isshape(shape):
155:             raise ValueError("invalid shape %r (must be 2-d)" % (shape,))
156: 
157:         self.dtype = dtype
158:         self.shape = shape
159: 
160:     def _init_dtype(self):
161:         '''Called from subclasses at the end of the __init__ routine.
162:         '''
163:         if self.dtype is None:
164:             v = np.zeros(self.shape[-1])
165:             self.dtype = np.asarray(self.matvec(v)).dtype
166: 
167:     def _matmat(self, X):
168:         '''Default matrix-matrix multiplication handler.
169: 
170:         Falls back on the user-defined _matvec method, so defining that will
171:         define matrix multiplication (though in a very suboptimal way).
172:         '''
173: 
174:         return np.hstack([self.matvec(col.reshape(-1,1)) for col in X.T])
175: 
176:     def _matvec(self, x):
177:         '''Default matrix-vector multiplication handler.
178: 
179:         If self is a linear operator of shape (M, N), then this method will
180:         be called on a shape (N,) or (N, 1) ndarray, and should return a
181:         shape (M,) or (M, 1) ndarray.
182: 
183:         This default implementation falls back on _matmat, so defining that
184:         will define matrix-vector multiplication as well.
185:         '''
186:         return self.matmat(x.reshape(-1, 1))
187: 
188:     def matvec(self, x):
189:         '''Matrix-vector multiplication.
190: 
191:         Performs the operation y=A*x where A is an MxN linear
192:         operator and x is a column vector or 1-d array.
193: 
194:         Parameters
195:         ----------
196:         x : {matrix, ndarray}
197:             An array with shape (N,) or (N,1).
198: 
199:         Returns
200:         -------
201:         y : {matrix, ndarray}
202:             A matrix or ndarray with shape (M,) or (M,1) depending
203:             on the type and shape of the x argument.
204: 
205:         Notes
206:         -----
207:         This matvec wraps the user-specified matvec routine or overridden
208:         _matvec method to ensure that y has the correct shape and type.
209: 
210:         '''
211: 
212:         x = np.asanyarray(x)
213: 
214:         M,N = self.shape
215: 
216:         if x.shape != (N,) and x.shape != (N,1):
217:             raise ValueError('dimension mismatch')
218: 
219:         y = self._matvec(x)
220: 
221:         if isinstance(x, np.matrix):
222:             y = np.asmatrix(y)
223:         else:
224:             y = np.asarray(y)
225: 
226:         if x.ndim == 1:
227:             y = y.reshape(M)
228:         elif x.ndim == 2:
229:             y = y.reshape(M,1)
230:         else:
231:             raise ValueError('invalid shape returned by user-defined matvec()')
232: 
233:         return y
234: 
235:     def rmatvec(self, x):
236:         '''Adjoint matrix-vector multiplication.
237: 
238:         Performs the operation y = A^H * x where A is an MxN linear
239:         operator and x is a column vector or 1-d array.
240: 
241:         Parameters
242:         ----------
243:         x : {matrix, ndarray}
244:             An array with shape (M,) or (M,1).
245: 
246:         Returns
247:         -------
248:         y : {matrix, ndarray}
249:             A matrix or ndarray with shape (N,) or (N,1) depending
250:             on the type and shape of the x argument.
251: 
252:         Notes
253:         -----
254:         This rmatvec wraps the user-specified rmatvec routine or overridden
255:         _rmatvec method to ensure that y has the correct shape and type.
256: 
257:         '''
258: 
259:         x = np.asanyarray(x)
260: 
261:         M,N = self.shape
262: 
263:         if x.shape != (M,) and x.shape != (M,1):
264:             raise ValueError('dimension mismatch')
265: 
266:         y = self._rmatvec(x)
267: 
268:         if isinstance(x, np.matrix):
269:             y = np.asmatrix(y)
270:         else:
271:             y = np.asarray(y)
272: 
273:         if x.ndim == 1:
274:             y = y.reshape(N)
275:         elif x.ndim == 2:
276:             y = y.reshape(N,1)
277:         else:
278:             raise ValueError('invalid shape returned by user-defined rmatvec()')
279: 
280:         return y
281: 
282:     def _rmatvec(self, x):
283:         '''Default implementation of _rmatvec; defers to adjoint.'''
284:         if type(self)._adjoint == LinearOperator._adjoint:
285:             # _adjoint not overridden, prevent infinite recursion
286:             raise NotImplementedError
287:         else:
288:             return self.H.matvec(x)
289: 
290:     def matmat(self, X):
291:         '''Matrix-matrix multiplication.
292: 
293:         Performs the operation y=A*X where A is an MxN linear
294:         operator and X dense N*K matrix or ndarray.
295: 
296:         Parameters
297:         ----------
298:         X : {matrix, ndarray}
299:             An array with shape (N,K).
300: 
301:         Returns
302:         -------
303:         Y : {matrix, ndarray}
304:             A matrix or ndarray with shape (M,K) depending on
305:             the type of the X argument.
306: 
307:         Notes
308:         -----
309:         This matmat wraps any user-specified matmat routine or overridden
310:         _matmat method to ensure that y has the correct type.
311: 
312:         '''
313: 
314:         X = np.asanyarray(X)
315: 
316:         if X.ndim != 2:
317:             raise ValueError('expected 2-d ndarray or matrix, not %d-d'
318:                              % X.ndim)
319: 
320:         M,N = self.shape
321: 
322:         if X.shape[0] != N:
323:             raise ValueError('dimension mismatch: %r, %r'
324:                              % (self.shape, X.shape))
325: 
326:         Y = self._matmat(X)
327: 
328:         if isinstance(Y, np.matrix):
329:             Y = np.asmatrix(Y)
330: 
331:         return Y
332: 
333:     def __call__(self, x):
334:         return self*x
335: 
336:     def __mul__(self, x):
337:         return self.dot(x)
338: 
339:     def dot(self, x):
340:         '''Matrix-matrix or matrix-vector multiplication.
341: 
342:         Parameters
343:         ----------
344:         x : array_like
345:             1-d or 2-d array, representing a vector or matrix.
346: 
347:         Returns
348:         -------
349:         Ax : array
350:             1-d or 2-d array (depending on the shape of x) that represents
351:             the result of applying this linear operator on x.
352: 
353:         '''
354:         if isinstance(x, LinearOperator):
355:             return _ProductLinearOperator(self, x)
356:         elif np.isscalar(x):
357:             return _ScaledLinearOperator(self, x)
358:         else:
359:             x = np.asarray(x)
360: 
361:             if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
362:                 return self.matvec(x)
363:             elif x.ndim == 2:
364:                 return self.matmat(x)
365:             else:
366:                 raise ValueError('expected 1-d or 2-d array or matrix, got %r'
367:                                  % x)
368: 
369:     def __matmul__(self, other):
370:         if np.isscalar(other):
371:             raise ValueError("Scalar operands are not allowed, "
372:                              "use '*' instead")
373:         return self.__mul__(other)
374: 
375:     def __rmatmul__(self, other):
376:         if np.isscalar(other):
377:             raise ValueError("Scalar operands are not allowed, "
378:                              "use '*' instead")
379:         return self.__rmul__(other)
380: 
381:     def __rmul__(self, x):
382:         if np.isscalar(x):
383:             return _ScaledLinearOperator(self, x)
384:         else:
385:             return NotImplemented
386: 
387:     def __pow__(self, p):
388:         if np.isscalar(p):
389:             return _PowerLinearOperator(self, p)
390:         else:
391:             return NotImplemented
392: 
393:     def __add__(self, x):
394:         if isinstance(x, LinearOperator):
395:             return _SumLinearOperator(self, x)
396:         else:
397:             return NotImplemented
398: 
399:     def __neg__(self):
400:         return _ScaledLinearOperator(self, -1)
401: 
402:     def __sub__(self, x):
403:         return self.__add__(-x)
404: 
405:     def __repr__(self):
406:         M,N = self.shape
407:         if self.dtype is None:
408:             dt = 'unspecified dtype'
409:         else:
410:             dt = 'dtype=' + str(self.dtype)
411: 
412:         return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)
413: 
414:     def adjoint(self):
415:         '''Hermitian adjoint.
416: 
417:         Returns the Hermitian adjoint of self, aka the Hermitian
418:         conjugate or Hermitian transpose. For a complex matrix, the
419:         Hermitian adjoint is equal to the conjugate transpose.
420: 
421:         Can be abbreviated self.H instead of self.adjoint().
422: 
423:         Returns
424:         -------
425:         A_H : LinearOperator
426:             Hermitian adjoint of self.
427:         '''
428:         return self._adjoint()
429: 
430:     H = property(adjoint)
431: 
432:     def transpose(self):
433:         '''Transpose this linear operator.
434: 
435:         Returns a LinearOperator that represents the transpose of this one.
436:         Can be abbreviated self.T instead of self.transpose().
437:         '''
438:         return self._transpose()
439: 
440:     T = property(transpose)
441: 
442:     def _adjoint(self):
443:         '''Default implementation of _adjoint; defers to rmatvec.'''
444:         shape = (self.shape[1], self.shape[0])
445:         return _CustomLinearOperator(shape, matvec=self.rmatvec,
446:                                      rmatvec=self.matvec,
447:                                      dtype=self.dtype)
448: 
449: 
450: class _CustomLinearOperator(LinearOperator):
451:     '''Linear operator defined in terms of user-specified operations.'''
452: 
453:     def __init__(self, shape, matvec, rmatvec=None, matmat=None, dtype=None):
454:         super(_CustomLinearOperator, self).__init__(dtype, shape)
455: 
456:         self.args = ()
457: 
458:         self.__matvec_impl = matvec
459:         self.__rmatvec_impl = rmatvec
460:         self.__matmat_impl = matmat
461: 
462:         self._init_dtype()
463: 
464:     def _matmat(self, X):
465:         if self.__matmat_impl is not None:
466:             return self.__matmat_impl(X)
467:         else:
468:             return super(_CustomLinearOperator, self)._matmat(X)
469: 
470:     def _matvec(self, x):
471:         return self.__matvec_impl(x)
472: 
473:     def _rmatvec(self, x):
474:         func = self.__rmatvec_impl
475:         if func is None:
476:             raise NotImplementedError("rmatvec is not defined")
477:         return self.__rmatvec_impl(x)
478: 
479:     def _adjoint(self):
480:         return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
481:                                      matvec=self.__rmatvec_impl,
482:                                      rmatvec=self.__matvec_impl,
483:                                      dtype=self.dtype)
484: 
485: 
486: def _get_dtype(operators, dtypes=None):
487:     if dtypes is None:
488:         dtypes = []
489:     for obj in operators:
490:         if obj is not None and hasattr(obj, 'dtype'):
491:             dtypes.append(obj.dtype)
492:     return np.find_common_type(dtypes, [])
493: 
494: 
495: class _SumLinearOperator(LinearOperator):
496:     def __init__(self, A, B):
497:         if not isinstance(A, LinearOperator) or \
498:                 not isinstance(B, LinearOperator):
499:             raise ValueError('both operands have to be a LinearOperator')
500:         if A.shape != B.shape:
501:             raise ValueError('cannot add %r and %r: shape mismatch'
502:                              % (A, B))
503:         self.args = (A, B)
504:         super(_SumLinearOperator, self).__init__(_get_dtype([A, B]), A.shape)
505: 
506:     def _matvec(self, x):
507:         return self.args[0].matvec(x) + self.args[1].matvec(x)
508: 
509:     def _rmatvec(self, x):
510:         return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)
511: 
512:     def _matmat(self, x):
513:         return self.args[0].matmat(x) + self.args[1].matmat(x)
514: 
515:     def _adjoint(self):
516:         A, B = self.args
517:         return A.H + B.H
518: 
519: 
520: class _ProductLinearOperator(LinearOperator):
521:     def __init__(self, A, B):
522:         if not isinstance(A, LinearOperator) or \
523:                 not isinstance(B, LinearOperator):
524:             raise ValueError('both operands have to be a LinearOperator')
525:         if A.shape[1] != B.shape[0]:
526:             raise ValueError('cannot multiply %r and %r: shape mismatch'
527:                              % (A, B))
528:         super(_ProductLinearOperator, self).__init__(_get_dtype([A, B]),
529:                                                      (A.shape[0], B.shape[1]))
530:         self.args = (A, B)
531: 
532:     def _matvec(self, x):
533:         return self.args[0].matvec(self.args[1].matvec(x))
534: 
535:     def _rmatvec(self, x):
536:         return self.args[1].rmatvec(self.args[0].rmatvec(x))
537: 
538:     def _matmat(self, x):
539:         return self.args[0].matmat(self.args[1].matmat(x))
540: 
541:     def _adjoint(self):
542:         A, B = self.args
543:         return B.H * A.H
544: 
545: 
546: class _ScaledLinearOperator(LinearOperator):
547:     def __init__(self, A, alpha):
548:         if not isinstance(A, LinearOperator):
549:             raise ValueError('LinearOperator expected as A')
550:         if not np.isscalar(alpha):
551:             raise ValueError('scalar expected as alpha')
552:         dtype = _get_dtype([A], [type(alpha)])
553:         super(_ScaledLinearOperator, self).__init__(dtype, A.shape)
554:         self.args = (A, alpha)
555: 
556:     def _matvec(self, x):
557:         return self.args[1] * self.args[0].matvec(x)
558: 
559:     def _rmatvec(self, x):
560:         return np.conj(self.args[1]) * self.args[0].rmatvec(x)
561: 
562:     def _matmat(self, x):
563:         return self.args[1] * self.args[0].matmat(x)
564: 
565:     def _adjoint(self):
566:         A, alpha = self.args
567:         return A.H * alpha
568: 
569: 
570: class _PowerLinearOperator(LinearOperator):
571:     def __init__(self, A, p):
572:         if not isinstance(A, LinearOperator):
573:             raise ValueError('LinearOperator expected as A')
574:         if A.shape[0] != A.shape[1]:
575:             raise ValueError('square LinearOperator expected, got %r' % A)
576:         if not isintlike(p) or p < 0:
577:             raise ValueError('non-negative integer expected as p')
578: 
579:         super(_PowerLinearOperator, self).__init__(_get_dtype([A]), A.shape)
580:         self.args = (A, p)
581: 
582:     def _power(self, fun, x):
583:         res = np.array(x, copy=True)
584:         for i in range(self.args[1]):
585:             res = fun(res)
586:         return res
587: 
588:     def _matvec(self, x):
589:         return self._power(self.args[0].matvec, x)
590: 
591:     def _rmatvec(self, x):
592:         return self._power(self.args[0].rmatvec, x)
593: 
594:     def _matmat(self, x):
595:         return self._power(self.args[0].matmat, x)
596: 
597:     def _adjoint(self):
598:         A, p = self.args
599:         return A.H ** p
600: 
601: 
602: class MatrixLinearOperator(LinearOperator):
603:     def __init__(self, A):
604:         super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
605:         self.A = A
606:         self.__adj = None
607:         self.args = (A,)
608: 
609:     def _matmat(self, X):
610:         return self.A.dot(X)
611: 
612:     def _adjoint(self):
613:         if self.__adj is None:
614:             self.__adj = _AdjointMatrixOperator(self)
615:         return self.__adj
616: 
617: 
618: class _AdjointMatrixOperator(MatrixLinearOperator):
619:     def __init__(self, adjoint):
620:         self.A = adjoint.A.T.conj()
621:         self.__adjoint = adjoint
622:         self.args = (adjoint,)
623:         self.shape = adjoint.shape[1], adjoint.shape[0]
624: 
625:     @property
626:     def dtype(self):
627:         return self.__adjoint.dtype
628: 
629:     def _adjoint(self):
630:         return self.__adjoint
631: 
632: 
633: class IdentityOperator(LinearOperator):
634:     def __init__(self, shape, dtype=None):
635:         super(IdentityOperator, self).__init__(dtype, shape)
636: 
637:     def _matvec(self, x):
638:         return x
639: 
640:     def _rmatvec(self, x):
641:         return x
642: 
643:     def _matmat(self, x):
644:         return x
645: 
646:     def _adjoint(self):
647:         return self
648: 
649: 
650: def aslinearoperator(A):
651:     '''Return A as a LinearOperator.
652: 
653:     'A' may be any of the following types:
654:      - ndarray
655:      - matrix
656:      - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
657:      - LinearOperator
658:      - An object with .shape and .matvec attributes
659: 
660:     See the LinearOperator documentation for additional information.
661: 
662:     Examples
663:     --------
664:     >>> from scipy.sparse.linalg import aslinearoperator
665:     >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
666:     >>> aslinearoperator(M)
667:     <2x3 MatrixLinearOperator with dtype=int32>
668: 
669:     '''
670:     if isinstance(A, LinearOperator):
671:         return A
672: 
673:     elif isinstance(A, np.ndarray) or isinstance(A, np.matrix):
674:         if A.ndim > 2:
675:             raise ValueError('array must have ndim <= 2')
676:         A = np.atleast_2d(np.asarray(A))
677:         return MatrixLinearOperator(A)
678: 
679:     elif isspmatrix(A):
680:         return MatrixLinearOperator(A)
681: 
682:     else:
683:         if hasattr(A, 'shape') and hasattr(A, 'matvec'):
684:             rmatvec = None
685:             dtype = None
686: 
687:             if hasattr(A, 'rmatvec'):
688:                 rmatvec = A.rmatvec
689:             if hasattr(A, 'dtype'):
690:                 dtype = A.dtype
691:             return LinearOperator(A.shape, A.matvec,
692:                                   rmatvec=rmatvec, dtype=dtype)
693: 
694:         else:
695:             raise TypeError('type not understood')
696: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_384921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', 'Abstract linear algebra library.\n\nThis module defines a class hierarchy that implements a kind of "lazy"\nmatrix representation, called the ``LinearOperator``. It can be used to do\nlinear algebra with extremely large sparse or structured matrices, without\nrepresenting those explicitly in memory. Such matrices can be added,\nmultiplied, transposed, etc.\n\nAs a motivating example, suppose you want have a matrix where almost all of\nthe elements have the value one. The standard sparse matrix representation\nskips the storage of zeros, but not ones. By contrast, a LinearOperator is\nable to represent such matrices efficiently. First, we need a compact way to\nrepresent an all-ones matrix::\n\n    >>> import numpy as np\n    >>> class Ones(LinearOperator):\n    ...     def __init__(self, shape):\n    ...         super(Ones, self).__init__(dtype=None, shape=shape)\n    ...     def _matvec(self, x):\n    ...         return np.repeat(x.sum(), self.shape[0])\n\nInstances of this class emulate ``np.ones(shape)``, but using a constant\namount of storage, independent of ``shape``. The ``_matvec`` method specifies\nhow this linear operator multiplies with (operates on) a vector. We can now\nadd this operator to a sparse matrix that stores only offsets from one::\n\n    >>> from scipy.sparse import csr_matrix\n    >>> offsets = csr_matrix([[1, 0, 2], [0, -1, 0], [0, 0, 3]])\n    >>> A = aslinearoperator(offsets) + Ones(offsets.shape)\n    >>> A.dot([1, 2, 3])\n    array([13,  4, 15])\n\nThe result is the same as that given by its dense, explicitly-stored\ncounterpart::\n\n    >>> (np.ones(A.shape, A.dtype) + offsets.toarray()).dot([1, 2, 3])\n    array([13,  4, 15])\n\nSeveral algorithms in the ``scipy.sparse`` library are able to operate on\n``LinearOperator`` instances.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'import numpy' statement (line 45)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_384922 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy')

if (type(import_384922) is not StypyTypeError):

    if (import_384922 != 'pyd_module'):
        __import__(import_384922)
        sys_modules_384923 = sys.modules[import_384922]
        import_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'np', sys_modules_384923.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy', import_384922)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 0))

# 'from scipy.sparse import isspmatrix' statement (line 47)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_384924 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'scipy.sparse')

if (type(import_384924) is not StypyTypeError):

    if (import_384924 != 'pyd_module'):
        __import__(import_384924)
        sys_modules_384925 = sys.modules[import_384924]
        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'scipy.sparse', sys_modules_384925.module_type_store, module_type_store, ['isspmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 47, 0), __file__, sys_modules_384925, sys_modules_384925.module_type_store, module_type_store)
    else:
        from scipy.sparse import isspmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'scipy.sparse', None, module_type_store, ['isspmatrix'], [isspmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'scipy.sparse', import_384924)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'from scipy.sparse.sputils import isshape, isintlike' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
import_384926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.sputils')

if (type(import_384926) is not StypyTypeError):

    if (import_384926 != 'pyd_module'):
        __import__(import_384926)
        sys_modules_384927 = sys.modules[import_384926]
        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.sputils', sys_modules_384927.module_type_store, module_type_store, ['isshape', 'isintlike'])
        nest_module(stypy.reporting.localization.Localization(__file__, 48, 0), __file__, sys_modules_384927, sys_modules_384927.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import isshape, isintlike

        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.sputils', None, module_type_store, ['isshape', 'isintlike'], [isshape, isintlike])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'scipy.sparse.sputils', import_384926)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')


# Assigning a List to a Name (line 50):

# Assigning a List to a Name (line 50):
__all__ = ['LinearOperator', 'aslinearoperator']
module_type_store.set_exportable_members(['LinearOperator', 'aslinearoperator'])

# Obtaining an instance of the builtin type 'list' (line 50)
list_384928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 50)
# Adding element type (line 50)
str_384929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'str', 'LinearOperator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_384928, str_384929)
# Adding element type (line 50)
str_384930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'str', 'aslinearoperator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_384928, str_384930)

# Assigning a type to the variable '__all__' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '__all__', list_384928)
# Declaration of the 'LinearOperator' class

class LinearOperator(object, ):
    str_384931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', 'Common interface for performing matrix vector products\n\n    Many iterative methods (e.g. cg, gmres) do not need to know the\n    individual entries of a matrix to solve a linear system A*x=b.\n    Such solvers only require the computation of matrix vector\n    products, A*v where v is a dense vector.  This class serves as\n    an abstract interface between iterative solvers and matrix-like\n    objects.\n\n    To construct a concrete LinearOperator, either pass appropriate\n    callables to the constructor of this class, or subclass it.\n\n    A subclass must implement either one of the methods ``_matvec``\n    and ``_matmat``, and the attributes/properties ``shape`` (pair of\n    integers) and ``dtype`` (may be None). It may call the ``__init__``\n    on this class to have these attributes validated. Implementing\n    ``_matvec`` automatically implements ``_matmat`` (using a naive\n    algorithm) and vice-versa.\n\n    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``\n    to implement the Hermitian adjoint (conjugate transpose). As with\n    ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or\n    ``_adjoint`` implements the other automatically. Implementing\n    ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for\n    backwards compatibility.\n\n    Parameters\n    ----------\n    shape : tuple\n        Matrix dimensions (M,N).\n    matvec : callable f(v)\n        Returns returns A * v.\n    rmatvec : callable f(v)\n        Returns A^H * v, where A^H is the conjugate transpose of A.\n    matmat : callable f(V)\n        Returns A * V, where V is a dense matrix with dimensions (N,K).\n    dtype : dtype\n        Data type of the matrix.\n\n    Attributes\n    ----------\n    args : tuple\n        For linear operators describing products etc. of other linear\n        operators, the operands of the binary operation.\n\n    See Also\n    --------\n    aslinearoperator : Construct LinearOperators\n\n    Notes\n    -----\n    The user-defined matvec() function must properly handle the case\n    where v has shape (N,) as well as the (N,1) case.  The shape of\n    the return type is handled internally by LinearOperator.\n\n    LinearOperator instances can also be multiplied, added with each\n    other and exponentiated, all lazily: the result of these operations\n    is always a new, composite LinearOperator, that defers linear\n    operations to the original operators and combines the results.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse.linalg import LinearOperator\n    >>> def mv(v):\n    ...     return np.array([2*v[0], 3*v[1]])\n    ...\n    >>> A = LinearOperator((2,2), matvec=mv)\n    >>> A\n    <2x2 _CustomLinearOperator with dtype=float64>\n    >>> A.matvec(np.ones(2))\n    array([ 2.,  3.])\n    >>> A * np.ones(2)\n    array([ 2.,  3.])\n\n    ')

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__new__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__new__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__new__')
        LinearOperator.__new__.__dict__.__setitem__('stypy_param_names_list', [])
        LinearOperator.__new__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        LinearOperator.__new__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        LinearOperator.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__new__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__new__', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        
        # Getting the type of 'cls' (line 131)
        cls_384932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'cls')
        # Getting the type of 'LinearOperator' (line 131)
        LinearOperator_384933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'LinearOperator')
        # Applying the binary operator 'is' (line 131)
        result_is__384934 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 11), 'is', cls_384932, LinearOperator_384933)
        
        # Testing the type of an if condition (line 131)
        if_condition_384935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 8), result_is__384934)
        # Assigning a type to the variable 'if_condition_384935' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'if_condition_384935', if_condition_384935)
        # SSA begins for if statement (line 131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _CustomLinearOperator(...): (line 133)
        # Getting the type of 'args' (line 133)
        args_384937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'args', False)
        # Processing the call keyword arguments (line 133)
        # Getting the type of 'kwargs' (line 133)
        kwargs_384938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 50), 'kwargs', False)
        kwargs_384939 = {'kwargs_384938': kwargs_384938}
        # Getting the type of '_CustomLinearOperator' (line 133)
        _CustomLinearOperator_384936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), '_CustomLinearOperator', False)
        # Calling _CustomLinearOperator(args, kwargs) (line 133)
        _CustomLinearOperator_call_result_384940 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), _CustomLinearOperator_384936, *[args_384937], **kwargs_384939)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'stypy_return_type', _CustomLinearOperator_call_result_384940)
        # SSA branch for the else part of an if statement (line 131)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to __new__(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'cls' (line 135)
        cls_384947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 53), 'cls', False)
        # Processing the call keyword arguments (line 135)
        kwargs_384948 = {}
        
        # Call to super(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'LinearOperator' (line 135)
        LinearOperator_384942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'LinearOperator', False)
        # Getting the type of 'cls' (line 135)
        cls_384943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), 'cls', False)
        # Processing the call keyword arguments (line 135)
        kwargs_384944 = {}
        # Getting the type of 'super' (line 135)
        super_384941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'super', False)
        # Calling super(args, kwargs) (line 135)
        super_call_result_384945 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), super_384941, *[LinearOperator_384942, cls_384943], **kwargs_384944)
        
        # Obtaining the member '__new__' of a type (line 135)
        new___384946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 18), super_call_result_384945, '__new__')
        # Calling __new__(args, kwargs) (line 135)
        new___call_result_384949 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), new___384946, *[cls_384947], **kwargs_384948)
        
        # Assigning a type to the variable 'obj' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'obj', new___call_result_384949)
        
        
        # Evaluating a boolean operation
        
        
        # Call to type(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'obj' (line 137)
        obj_384951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'obj', False)
        # Processing the call keyword arguments (line 137)
        kwargs_384952 = {}
        # Getting the type of 'type' (line 137)
        type_384950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'type', False)
        # Calling type(args, kwargs) (line 137)
        type_call_result_384953 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), type_384950, *[obj_384951], **kwargs_384952)
        
        # Obtaining the member '_matvec' of a type (line 137)
        _matvec_384954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), type_call_result_384953, '_matvec')
        # Getting the type of 'LinearOperator' (line 137)
        LinearOperator_384955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'LinearOperator')
        # Obtaining the member '_matvec' of a type (line 137)
        _matvec_384956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 37), LinearOperator_384955, '_matvec')
        # Applying the binary operator '==' (line 137)
        result_eq_384957 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 16), '==', _matvec_384954, _matvec_384956)
        
        
        
        # Call to type(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'obj' (line 138)
        obj_384959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'obj', False)
        # Processing the call keyword arguments (line 138)
        kwargs_384960 = {}
        # Getting the type of 'type' (line 138)
        type_384958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'type', False)
        # Calling type(args, kwargs) (line 138)
        type_call_result_384961 = invoke(stypy.reporting.localization.Localization(__file__, 138, 24), type_384958, *[obj_384959], **kwargs_384960)
        
        # Obtaining the member '_matmat' of a type (line 138)
        _matmat_384962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 24), type_call_result_384961, '_matmat')
        # Getting the type of 'LinearOperator' (line 138)
        LinearOperator_384963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 45), 'LinearOperator')
        # Obtaining the member '_matmat' of a type (line 138)
        _matmat_384964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 45), LinearOperator_384963, '_matmat')
        # Applying the binary operator '==' (line 138)
        result_eq_384965 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 24), '==', _matmat_384962, _matmat_384964)
        
        # Applying the binary operator 'and' (line 137)
        result_and_keyword_384966 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 16), 'and', result_eq_384957, result_eq_384965)
        
        # Testing the type of an if condition (line 137)
        if_condition_384967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 12), result_and_keyword_384966)
        # Assigning a type to the variable 'if_condition_384967' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'if_condition_384967', if_condition_384967)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 139)
        # Processing the call arguments (line 139)
        str_384969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 32), 'str', 'LinearOperator subclass should implement at least one of _matvec and _matmat.')
        # Processing the call keyword arguments (line 139)
        kwargs_384970 = {}
        # Getting the type of 'TypeError' (line 139)
        TypeError_384968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 139)
        TypeError_call_result_384971 = invoke(stypy.reporting.localization.Localization(__file__, 139, 22), TypeError_384968, *[str_384969], **kwargs_384970)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 16), TypeError_call_result_384971, 'raise parameter', BaseException)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj' (line 142)
        obj_384972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'stypy_return_type', obj_384972)
        # SSA join for if statement (line 131)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_384973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_384973)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_384973


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__init__', ['dtype', 'shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dtype', 'shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_384974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', 'Initialize this LinearOperator.\n\n        To be called by subclasses. ``dtype`` may be None; ``shape`` should\n        be convertible to a length-2 tuple.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 150)
        # Getting the type of 'dtype' (line 150)
        dtype_384975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'dtype')
        # Getting the type of 'None' (line 150)
        None_384976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'None')
        
        (may_be_384977, more_types_in_union_384978) = may_not_be_none(dtype_384975, None_384976)

        if may_be_384977:

            if more_types_in_union_384978:
                # Runtime conditional SSA (line 150)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 151):
            
            # Assigning a Call to a Name (line 151):
            
            # Call to dtype(...): (line 151)
            # Processing the call arguments (line 151)
            # Getting the type of 'dtype' (line 151)
            dtype_384981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'dtype', False)
            # Processing the call keyword arguments (line 151)
            kwargs_384982 = {}
            # Getting the type of 'np' (line 151)
            np_384979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'np', False)
            # Obtaining the member 'dtype' of a type (line 151)
            dtype_384980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 20), np_384979, 'dtype')
            # Calling dtype(args, kwargs) (line 151)
            dtype_call_result_384983 = invoke(stypy.reporting.localization.Localization(__file__, 151, 20), dtype_384980, *[dtype_384981], **kwargs_384982)
            
            # Assigning a type to the variable 'dtype' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'dtype', dtype_call_result_384983)

            if more_types_in_union_384978:
                # SSA join for if statement (line 150)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to tuple(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'shape' (line 153)
        shape_384985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 22), 'shape', False)
        # Processing the call keyword arguments (line 153)
        kwargs_384986 = {}
        # Getting the type of 'tuple' (line 153)
        tuple_384984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 153)
        tuple_call_result_384987 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), tuple_384984, *[shape_384985], **kwargs_384986)
        
        # Assigning a type to the variable 'shape' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'shape', tuple_call_result_384987)
        
        
        
        # Call to isshape(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'shape' (line 154)
        shape_384989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'shape', False)
        # Processing the call keyword arguments (line 154)
        kwargs_384990 = {}
        # Getting the type of 'isshape' (line 154)
        isshape_384988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'isshape', False)
        # Calling isshape(args, kwargs) (line 154)
        isshape_call_result_384991 = invoke(stypy.reporting.localization.Localization(__file__, 154, 15), isshape_384988, *[shape_384989], **kwargs_384990)
        
        # Applying the 'not' unary operator (line 154)
        result_not__384992 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 11), 'not', isshape_call_result_384991)
        
        # Testing the type of an if condition (line 154)
        if_condition_384993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), result_not__384992)
        # Assigning a type to the variable 'if_condition_384993' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'if_condition_384993', if_condition_384993)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 155)
        # Processing the call arguments (line 155)
        str_384995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 29), 'str', 'invalid shape %r (must be 2-d)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 155)
        tuple_384996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 155)
        # Adding element type (line 155)
        # Getting the type of 'shape' (line 155)
        shape_384997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 65), 'shape', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 65), tuple_384996, shape_384997)
        
        # Applying the binary operator '%' (line 155)
        result_mod_384998 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 29), '%', str_384995, tuple_384996)
        
        # Processing the call keyword arguments (line 155)
        kwargs_384999 = {}
        # Getting the type of 'ValueError' (line 155)
        ValueError_384994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 155)
        ValueError_call_result_385000 = invoke(stypy.reporting.localization.Localization(__file__, 155, 18), ValueError_384994, *[result_mod_384998], **kwargs_384999)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 12), ValueError_call_result_385000, 'raise parameter', BaseException)
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 157):
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'dtype' (line 157)
        dtype_385001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'dtype')
        # Getting the type of 'self' (line 157)
        self_385002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_385002, 'dtype', dtype_385001)
        
        # Assigning a Name to a Attribute (line 158):
        
        # Assigning a Name to a Attribute (line 158):
        # Getting the type of 'shape' (line 158)
        shape_385003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'shape')
        # Getting the type of 'self' (line 158)
        self_385004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self')
        # Setting the type of the member 'shape' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_385004, 'shape', shape_385003)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _init_dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_dtype'
        module_type_store = module_type_store.open_function_context('_init_dtype', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_function_name', 'LinearOperator._init_dtype')
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_param_names_list', [])
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator._init_dtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator._init_dtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_dtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_dtype(...)' code ##################

        str_385005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', 'Called from subclasses at the end of the __init__ routine.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 163)
        # Getting the type of 'self' (line 163)
        self_385006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 163)
        dtype_385007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), self_385006, 'dtype')
        # Getting the type of 'None' (line 163)
        None_385008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'None')
        
        (may_be_385009, more_types_in_union_385010) = may_be_none(dtype_385007, None_385008)

        if may_be_385009:

            if more_types_in_union_385010:
                # Runtime conditional SSA (line 163)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 164):
            
            # Assigning a Call to a Name (line 164):
            
            # Call to zeros(...): (line 164)
            # Processing the call arguments (line 164)
            
            # Obtaining the type of the subscript
            int_385013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 36), 'int')
            # Getting the type of 'self' (line 164)
            self_385014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'self', False)
            # Obtaining the member 'shape' of a type (line 164)
            shape_385015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), self_385014, 'shape')
            # Obtaining the member '__getitem__' of a type (line 164)
            getitem___385016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), shape_385015, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 164)
            subscript_call_result_385017 = invoke(stypy.reporting.localization.Localization(__file__, 164, 25), getitem___385016, int_385013)
            
            # Processing the call keyword arguments (line 164)
            kwargs_385018 = {}
            # Getting the type of 'np' (line 164)
            np_385011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'np', False)
            # Obtaining the member 'zeros' of a type (line 164)
            zeros_385012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), np_385011, 'zeros')
            # Calling zeros(args, kwargs) (line 164)
            zeros_call_result_385019 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), zeros_385012, *[subscript_call_result_385017], **kwargs_385018)
            
            # Assigning a type to the variable 'v' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'v', zeros_call_result_385019)
            
            # Assigning a Attribute to a Attribute (line 165):
            
            # Assigning a Attribute to a Attribute (line 165):
            
            # Call to asarray(...): (line 165)
            # Processing the call arguments (line 165)
            
            # Call to matvec(...): (line 165)
            # Processing the call arguments (line 165)
            # Getting the type of 'v' (line 165)
            v_385024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 48), 'v', False)
            # Processing the call keyword arguments (line 165)
            kwargs_385025 = {}
            # Getting the type of 'self' (line 165)
            self_385022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'self', False)
            # Obtaining the member 'matvec' of a type (line 165)
            matvec_385023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 36), self_385022, 'matvec')
            # Calling matvec(args, kwargs) (line 165)
            matvec_call_result_385026 = invoke(stypy.reporting.localization.Localization(__file__, 165, 36), matvec_385023, *[v_385024], **kwargs_385025)
            
            # Processing the call keyword arguments (line 165)
            kwargs_385027 = {}
            # Getting the type of 'np' (line 165)
            np_385020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'np', False)
            # Obtaining the member 'asarray' of a type (line 165)
            asarray_385021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), np_385020, 'asarray')
            # Calling asarray(args, kwargs) (line 165)
            asarray_call_result_385028 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), asarray_385021, *[matvec_call_result_385026], **kwargs_385027)
            
            # Obtaining the member 'dtype' of a type (line 165)
            dtype_385029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), asarray_call_result_385028, 'dtype')
            # Getting the type of 'self' (line 165)
            self_385030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'self')
            # Setting the type of the member 'dtype' of a type (line 165)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), self_385030, 'dtype', dtype_385029)

            if more_types_in_union_385010:
                # SSA join for if statement (line 163)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_init_dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_385031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385031)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_dtype'
        return stypy_return_type_385031


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator._matmat.__dict__.__setitem__('stypy_function_name', 'LinearOperator._matmat')
        LinearOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        LinearOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator._matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        str_385032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, (-1)), 'str', 'Default matrix-matrix multiplication handler.\n\n        Falls back on the user-defined _matvec method, so defining that will\n        define matrix multiplication (though in a very suboptimal way).\n        ')
        
        # Call to hstack(...): (line 174)
        # Processing the call arguments (line 174)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'X' (line 174)
        X_385045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 68), 'X', False)
        # Obtaining the member 'T' of a type (line 174)
        T_385046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 68), X_385045, 'T')
        comprehension_385047 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 26), T_385046)
        # Assigning a type to the variable 'col' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'col', comprehension_385047)
        
        # Call to matvec(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Call to reshape(...): (line 174)
        # Processing the call arguments (line 174)
        int_385039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 50), 'int')
        int_385040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 53), 'int')
        # Processing the call keyword arguments (line 174)
        kwargs_385041 = {}
        # Getting the type of 'col' (line 174)
        col_385037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'col', False)
        # Obtaining the member 'reshape' of a type (line 174)
        reshape_385038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), col_385037, 'reshape')
        # Calling reshape(args, kwargs) (line 174)
        reshape_call_result_385042 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), reshape_385038, *[int_385039, int_385040], **kwargs_385041)
        
        # Processing the call keyword arguments (line 174)
        kwargs_385043 = {}
        # Getting the type of 'self' (line 174)
        self_385035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'self', False)
        # Obtaining the member 'matvec' of a type (line 174)
        matvec_385036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), self_385035, 'matvec')
        # Calling matvec(args, kwargs) (line 174)
        matvec_call_result_385044 = invoke(stypy.reporting.localization.Localization(__file__, 174, 26), matvec_385036, *[reshape_call_result_385042], **kwargs_385043)
        
        list_385048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 26), list_385048, matvec_call_result_385044)
        # Processing the call keyword arguments (line 174)
        kwargs_385049 = {}
        # Getting the type of 'np' (line 174)
        np_385033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 174)
        hstack_385034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 15), np_385033, 'hstack')
        # Calling hstack(args, kwargs) (line 174)
        hstack_call_result_385050 = invoke(stypy.reporting.localization.Localization(__file__, 174, 15), hstack_385034, *[list_385048], **kwargs_385049)
        
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', hstack_call_result_385050)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_385051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385051)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_385051


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator._matvec.__dict__.__setitem__('stypy_function_name', 'LinearOperator._matvec')
        LinearOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        str_385052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, (-1)), 'str', 'Default matrix-vector multiplication handler.\n\n        If self is a linear operator of shape (M, N), then this method will\n        be called on a shape (N,) or (N, 1) ndarray, and should return a\n        shape (M,) or (M, 1) ndarray.\n\n        This default implementation falls back on _matmat, so defining that\n        will define matrix-vector multiplication as well.\n        ')
        
        # Call to matmat(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to reshape(...): (line 186)
        # Processing the call arguments (line 186)
        int_385057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 37), 'int')
        int_385058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 41), 'int')
        # Processing the call keyword arguments (line 186)
        kwargs_385059 = {}
        # Getting the type of 'x' (line 186)
        x_385055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'x', False)
        # Obtaining the member 'reshape' of a type (line 186)
        reshape_385056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 27), x_385055, 'reshape')
        # Calling reshape(args, kwargs) (line 186)
        reshape_call_result_385060 = invoke(stypy.reporting.localization.Localization(__file__, 186, 27), reshape_385056, *[int_385057, int_385058], **kwargs_385059)
        
        # Processing the call keyword arguments (line 186)
        kwargs_385061 = {}
        # Getting the type of 'self' (line 186)
        self_385053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'self', False)
        # Obtaining the member 'matmat' of a type (line 186)
        matmat_385054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 15), self_385053, 'matmat')
        # Calling matmat(args, kwargs) (line 186)
        matmat_call_result_385062 = invoke(stypy.reporting.localization.Localization(__file__, 186, 15), matmat_385054, *[reshape_call_result_385060], **kwargs_385061)
        
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', matmat_call_result_385062)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_385063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_385063


    @norecursion
    def matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matvec'
        module_type_store = module_type_store.open_function_context('matvec', 188, 4, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.matvec.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.matvec.__dict__.__setitem__('stypy_function_name', 'LinearOperator.matvec')
        LinearOperator.matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matvec(...)' code ##################

        str_385064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', 'Matrix-vector multiplication.\n\n        Performs the operation y=A*x where A is an MxN linear\n        operator and x is a column vector or 1-d array.\n\n        Parameters\n        ----------\n        x : {matrix, ndarray}\n            An array with shape (N,) or (N,1).\n\n        Returns\n        -------\n        y : {matrix, ndarray}\n            A matrix or ndarray with shape (M,) or (M,1) depending\n            on the type and shape of the x argument.\n\n        Notes\n        -----\n        This matvec wraps the user-specified matvec routine or overridden\n        _matvec method to ensure that y has the correct shape and type.\n\n        ')
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to asanyarray(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'x' (line 212)
        x_385067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 26), 'x', False)
        # Processing the call keyword arguments (line 212)
        kwargs_385068 = {}
        # Getting the type of 'np' (line 212)
        np_385065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'np', False)
        # Obtaining the member 'asanyarray' of a type (line 212)
        asanyarray_385066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), np_385065, 'asanyarray')
        # Calling asanyarray(args, kwargs) (line 212)
        asanyarray_call_result_385069 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), asanyarray_385066, *[x_385067], **kwargs_385068)
        
        # Assigning a type to the variable 'x' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'x', asanyarray_call_result_385069)
        
        # Assigning a Attribute to a Tuple (line 214):
        
        # Assigning a Subscript to a Name (line 214):
        
        # Obtaining the type of the subscript
        int_385070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
        # Getting the type of 'self' (line 214)
        self_385071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 14), 'self')
        # Obtaining the member 'shape' of a type (line 214)
        shape_385072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 14), self_385071, 'shape')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___385073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), shape_385072, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_385074 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___385073, int_385070)
        
        # Assigning a type to the variable 'tuple_var_assignment_384905' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_384905', subscript_call_result_385074)
        
        # Assigning a Subscript to a Name (line 214):
        
        # Obtaining the type of the subscript
        int_385075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
        # Getting the type of 'self' (line 214)
        self_385076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 14), 'self')
        # Obtaining the member 'shape' of a type (line 214)
        shape_385077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 14), self_385076, 'shape')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___385078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), shape_385077, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_385079 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___385078, int_385075)
        
        # Assigning a type to the variable 'tuple_var_assignment_384906' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_384906', subscript_call_result_385079)
        
        # Assigning a Name to a Name (line 214):
        # Getting the type of 'tuple_var_assignment_384905' (line 214)
        tuple_var_assignment_384905_385080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_384905')
        # Assigning a type to the variable 'M' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'M', tuple_var_assignment_384905_385080)
        
        # Assigning a Name to a Name (line 214):
        # Getting the type of 'tuple_var_assignment_384906' (line 214)
        tuple_var_assignment_384906_385081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_384906')
        # Assigning a type to the variable 'N' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 10), 'N', tuple_var_assignment_384906_385081)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 216)
        x_385082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 'x')
        # Obtaining the member 'shape' of a type (line 216)
        shape_385083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 11), x_385082, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 216)
        tuple_385084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 216)
        # Adding element type (line 216)
        # Getting the type of 'N' (line 216)
        N_385085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 23), tuple_385084, N_385085)
        
        # Applying the binary operator '!=' (line 216)
        result_ne_385086 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 11), '!=', shape_385083, tuple_385084)
        
        
        # Getting the type of 'x' (line 216)
        x_385087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'x')
        # Obtaining the member 'shape' of a type (line 216)
        shape_385088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 31), x_385087, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 216)
        tuple_385089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 216)
        # Adding element type (line 216)
        # Getting the type of 'N' (line 216)
        N_385090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 43), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 43), tuple_385089, N_385090)
        # Adding element type (line 216)
        int_385091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 43), tuple_385089, int_385091)
        
        # Applying the binary operator '!=' (line 216)
        result_ne_385092 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 31), '!=', shape_385088, tuple_385089)
        
        # Applying the binary operator 'and' (line 216)
        result_and_keyword_385093 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 11), 'and', result_ne_385086, result_ne_385092)
        
        # Testing the type of an if condition (line 216)
        if_condition_385094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 8), result_and_keyword_385093)
        # Assigning a type to the variable 'if_condition_385094' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'if_condition_385094', if_condition_385094)
        # SSA begins for if statement (line 216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 217)
        # Processing the call arguments (line 217)
        str_385096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'str', 'dimension mismatch')
        # Processing the call keyword arguments (line 217)
        kwargs_385097 = {}
        # Getting the type of 'ValueError' (line 217)
        ValueError_385095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 217)
        ValueError_call_result_385098 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), ValueError_385095, *[str_385096], **kwargs_385097)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 217, 12), ValueError_call_result_385098, 'raise parameter', BaseException)
        # SSA join for if statement (line 216)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to _matvec(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'x' (line 219)
        x_385101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'x', False)
        # Processing the call keyword arguments (line 219)
        kwargs_385102 = {}
        # Getting the type of 'self' (line 219)
        self_385099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'self', False)
        # Obtaining the member '_matvec' of a type (line 219)
        _matvec_385100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), self_385099, '_matvec')
        # Calling _matvec(args, kwargs) (line 219)
        _matvec_call_result_385103 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), _matvec_385100, *[x_385101], **kwargs_385102)
        
        # Assigning a type to the variable 'y' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'y', _matvec_call_result_385103)
        
        
        # Call to isinstance(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'x' (line 221)
        x_385105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'x', False)
        # Getting the type of 'np' (line 221)
        np_385106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'np', False)
        # Obtaining the member 'matrix' of a type (line 221)
        matrix_385107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 25), np_385106, 'matrix')
        # Processing the call keyword arguments (line 221)
        kwargs_385108 = {}
        # Getting the type of 'isinstance' (line 221)
        isinstance_385104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 221)
        isinstance_call_result_385109 = invoke(stypy.reporting.localization.Localization(__file__, 221, 11), isinstance_385104, *[x_385105, matrix_385107], **kwargs_385108)
        
        # Testing the type of an if condition (line 221)
        if_condition_385110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), isinstance_call_result_385109)
        # Assigning a type to the variable 'if_condition_385110' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_385110', if_condition_385110)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to asmatrix(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'y' (line 222)
        y_385113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'y', False)
        # Processing the call keyword arguments (line 222)
        kwargs_385114 = {}
        # Getting the type of 'np' (line 222)
        np_385111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 222)
        asmatrix_385112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 16), np_385111, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 222)
        asmatrix_call_result_385115 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), asmatrix_385112, *[y_385113], **kwargs_385114)
        
        # Assigning a type to the variable 'y' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'y', asmatrix_call_result_385115)
        # SSA branch for the else part of an if statement (line 221)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to asarray(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'y' (line 224)
        y_385118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'y', False)
        # Processing the call keyword arguments (line 224)
        kwargs_385119 = {}
        # Getting the type of 'np' (line 224)
        np_385116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 224)
        asarray_385117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), np_385116, 'asarray')
        # Calling asarray(args, kwargs) (line 224)
        asarray_call_result_385120 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), asarray_385117, *[y_385118], **kwargs_385119)
        
        # Assigning a type to the variable 'y' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'y', asarray_call_result_385120)
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'x' (line 226)
        x_385121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'x')
        # Obtaining the member 'ndim' of a type (line 226)
        ndim_385122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 11), x_385121, 'ndim')
        int_385123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 21), 'int')
        # Applying the binary operator '==' (line 226)
        result_eq_385124 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 11), '==', ndim_385122, int_385123)
        
        # Testing the type of an if condition (line 226)
        if_condition_385125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), result_eq_385124)
        # Assigning a type to the variable 'if_condition_385125' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_385125', if_condition_385125)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to reshape(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'M' (line 227)
        M_385128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'M', False)
        # Processing the call keyword arguments (line 227)
        kwargs_385129 = {}
        # Getting the type of 'y' (line 227)
        y_385126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'y', False)
        # Obtaining the member 'reshape' of a type (line 227)
        reshape_385127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), y_385126, 'reshape')
        # Calling reshape(args, kwargs) (line 227)
        reshape_call_result_385130 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), reshape_385127, *[M_385128], **kwargs_385129)
        
        # Assigning a type to the variable 'y' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'y', reshape_call_result_385130)
        # SSA branch for the else part of an if statement (line 226)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'x' (line 228)
        x_385131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 13), 'x')
        # Obtaining the member 'ndim' of a type (line 228)
        ndim_385132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 13), x_385131, 'ndim')
        int_385133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 23), 'int')
        # Applying the binary operator '==' (line 228)
        result_eq_385134 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 13), '==', ndim_385132, int_385133)
        
        # Testing the type of an if condition (line 228)
        if_condition_385135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 13), result_eq_385134)
        # Assigning a type to the variable 'if_condition_385135' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 13), 'if_condition_385135', if_condition_385135)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to reshape(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'M' (line 229)
        M_385138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 26), 'M', False)
        int_385139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 28), 'int')
        # Processing the call keyword arguments (line 229)
        kwargs_385140 = {}
        # Getting the type of 'y' (line 229)
        y_385136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'y', False)
        # Obtaining the member 'reshape' of a type (line 229)
        reshape_385137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), y_385136, 'reshape')
        # Calling reshape(args, kwargs) (line 229)
        reshape_call_result_385141 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), reshape_385137, *[M_385138, int_385139], **kwargs_385140)
        
        # Assigning a type to the variable 'y' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'y', reshape_call_result_385141)
        # SSA branch for the else part of an if statement (line 228)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 231)
        # Processing the call arguments (line 231)
        str_385143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 29), 'str', 'invalid shape returned by user-defined matvec()')
        # Processing the call keyword arguments (line 231)
        kwargs_385144 = {}
        # Getting the type of 'ValueError' (line 231)
        ValueError_385142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 231)
        ValueError_call_result_385145 = invoke(stypy.reporting.localization.Localization(__file__, 231, 18), ValueError_385142, *[str_385143], **kwargs_385144)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 231, 12), ValueError_call_result_385145, 'raise parameter', BaseException)
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 233)
        y_385146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'stypy_return_type', y_385146)
        
        # ################# End of 'matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 188)
        stypy_return_type_385147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385147)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matvec'
        return stypy_return_type_385147


    @norecursion
    def rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rmatvec'
        module_type_store = module_type_store.open_function_context('rmatvec', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_function_name', 'LinearOperator.rmatvec')
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rmatvec(...)' code ##################

        str_385148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'str', 'Adjoint matrix-vector multiplication.\n\n        Performs the operation y = A^H * x where A is an MxN linear\n        operator and x is a column vector or 1-d array.\n\n        Parameters\n        ----------\n        x : {matrix, ndarray}\n            An array with shape (M,) or (M,1).\n\n        Returns\n        -------\n        y : {matrix, ndarray}\n            A matrix or ndarray with shape (N,) or (N,1) depending\n            on the type and shape of the x argument.\n\n        Notes\n        -----\n        This rmatvec wraps the user-specified rmatvec routine or overridden\n        _rmatvec method to ensure that y has the correct shape and type.\n\n        ')
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to asanyarray(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'x' (line 259)
        x_385151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'x', False)
        # Processing the call keyword arguments (line 259)
        kwargs_385152 = {}
        # Getting the type of 'np' (line 259)
        np_385149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'np', False)
        # Obtaining the member 'asanyarray' of a type (line 259)
        asanyarray_385150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), np_385149, 'asanyarray')
        # Calling asanyarray(args, kwargs) (line 259)
        asanyarray_call_result_385153 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), asanyarray_385150, *[x_385151], **kwargs_385152)
        
        # Assigning a type to the variable 'x' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'x', asanyarray_call_result_385153)
        
        # Assigning a Attribute to a Tuple (line 261):
        
        # Assigning a Subscript to a Name (line 261):
        
        # Obtaining the type of the subscript
        int_385154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'int')
        # Getting the type of 'self' (line 261)
        self_385155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 14), 'self')
        # Obtaining the member 'shape' of a type (line 261)
        shape_385156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 14), self_385155, 'shape')
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___385157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), shape_385156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_385158 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), getitem___385157, int_385154)
        
        # Assigning a type to the variable 'tuple_var_assignment_384907' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_384907', subscript_call_result_385158)
        
        # Assigning a Subscript to a Name (line 261):
        
        # Obtaining the type of the subscript
        int_385159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'int')
        # Getting the type of 'self' (line 261)
        self_385160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 14), 'self')
        # Obtaining the member 'shape' of a type (line 261)
        shape_385161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 14), self_385160, 'shape')
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___385162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), shape_385161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_385163 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), getitem___385162, int_385159)
        
        # Assigning a type to the variable 'tuple_var_assignment_384908' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_384908', subscript_call_result_385163)
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'tuple_var_assignment_384907' (line 261)
        tuple_var_assignment_384907_385164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_384907')
        # Assigning a type to the variable 'M' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'M', tuple_var_assignment_384907_385164)
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'tuple_var_assignment_384908' (line 261)
        tuple_var_assignment_384908_385165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_384908')
        # Assigning a type to the variable 'N' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 10), 'N', tuple_var_assignment_384908_385165)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 263)
        x_385166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'x')
        # Obtaining the member 'shape' of a type (line 263)
        shape_385167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 11), x_385166, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 263)
        tuple_385168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 263)
        # Adding element type (line 263)
        # Getting the type of 'M' (line 263)
        M_385169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'M')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 23), tuple_385168, M_385169)
        
        # Applying the binary operator '!=' (line 263)
        result_ne_385170 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 11), '!=', shape_385167, tuple_385168)
        
        
        # Getting the type of 'x' (line 263)
        x_385171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 31), 'x')
        # Obtaining the member 'shape' of a type (line 263)
        shape_385172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 31), x_385171, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 263)
        tuple_385173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 263)
        # Adding element type (line 263)
        # Getting the type of 'M' (line 263)
        M_385174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 43), 'M')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 43), tuple_385173, M_385174)
        # Adding element type (line 263)
        int_385175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 43), tuple_385173, int_385175)
        
        # Applying the binary operator '!=' (line 263)
        result_ne_385176 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 31), '!=', shape_385172, tuple_385173)
        
        # Applying the binary operator 'and' (line 263)
        result_and_keyword_385177 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 11), 'and', result_ne_385170, result_ne_385176)
        
        # Testing the type of an if condition (line 263)
        if_condition_385178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 8), result_and_keyword_385177)
        # Assigning a type to the variable 'if_condition_385178' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'if_condition_385178', if_condition_385178)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 264)
        # Processing the call arguments (line 264)
        str_385180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 29), 'str', 'dimension mismatch')
        # Processing the call keyword arguments (line 264)
        kwargs_385181 = {}
        # Getting the type of 'ValueError' (line 264)
        ValueError_385179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 264)
        ValueError_call_result_385182 = invoke(stypy.reporting.localization.Localization(__file__, 264, 18), ValueError_385179, *[str_385180], **kwargs_385181)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 264, 12), ValueError_call_result_385182, 'raise parameter', BaseException)
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to _rmatvec(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'x' (line 266)
        x_385185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'x', False)
        # Processing the call keyword arguments (line 266)
        kwargs_385186 = {}
        # Getting the type of 'self' (line 266)
        self_385183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self', False)
        # Obtaining the member '_rmatvec' of a type (line 266)
        _rmatvec_385184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_385183, '_rmatvec')
        # Calling _rmatvec(args, kwargs) (line 266)
        _rmatvec_call_result_385187 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), _rmatvec_385184, *[x_385185], **kwargs_385186)
        
        # Assigning a type to the variable 'y' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'y', _rmatvec_call_result_385187)
        
        
        # Call to isinstance(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'x' (line 268)
        x_385189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'x', False)
        # Getting the type of 'np' (line 268)
        np_385190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 25), 'np', False)
        # Obtaining the member 'matrix' of a type (line 268)
        matrix_385191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 25), np_385190, 'matrix')
        # Processing the call keyword arguments (line 268)
        kwargs_385192 = {}
        # Getting the type of 'isinstance' (line 268)
        isinstance_385188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 268)
        isinstance_call_result_385193 = invoke(stypy.reporting.localization.Localization(__file__, 268, 11), isinstance_385188, *[x_385189, matrix_385191], **kwargs_385192)
        
        # Testing the type of an if condition (line 268)
        if_condition_385194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), isinstance_call_result_385193)
        # Assigning a type to the variable 'if_condition_385194' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_385194', if_condition_385194)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to asmatrix(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'y' (line 269)
        y_385197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'y', False)
        # Processing the call keyword arguments (line 269)
        kwargs_385198 = {}
        # Getting the type of 'np' (line 269)
        np_385195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 269)
        asmatrix_385196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), np_385195, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 269)
        asmatrix_call_result_385199 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), asmatrix_385196, *[y_385197], **kwargs_385198)
        
        # Assigning a type to the variable 'y' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'y', asmatrix_call_result_385199)
        # SSA branch for the else part of an if statement (line 268)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 271):
        
        # Assigning a Call to a Name (line 271):
        
        # Call to asarray(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'y' (line 271)
        y_385202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 27), 'y', False)
        # Processing the call keyword arguments (line 271)
        kwargs_385203 = {}
        # Getting the type of 'np' (line 271)
        np_385200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 271)
        asarray_385201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), np_385200, 'asarray')
        # Calling asarray(args, kwargs) (line 271)
        asarray_call_result_385204 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), asarray_385201, *[y_385202], **kwargs_385203)
        
        # Assigning a type to the variable 'y' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'y', asarray_call_result_385204)
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'x' (line 273)
        x_385205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'x')
        # Obtaining the member 'ndim' of a type (line 273)
        ndim_385206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 11), x_385205, 'ndim')
        int_385207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'int')
        # Applying the binary operator '==' (line 273)
        result_eq_385208 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), '==', ndim_385206, int_385207)
        
        # Testing the type of an if condition (line 273)
        if_condition_385209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_385208)
        # Assigning a type to the variable 'if_condition_385209' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_385209', if_condition_385209)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to reshape(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'N' (line 274)
        N_385212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 26), 'N', False)
        # Processing the call keyword arguments (line 274)
        kwargs_385213 = {}
        # Getting the type of 'y' (line 274)
        y_385210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'y', False)
        # Obtaining the member 'reshape' of a type (line 274)
        reshape_385211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), y_385210, 'reshape')
        # Calling reshape(args, kwargs) (line 274)
        reshape_call_result_385214 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), reshape_385211, *[N_385212], **kwargs_385213)
        
        # Assigning a type to the variable 'y' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'y', reshape_call_result_385214)
        # SSA branch for the else part of an if statement (line 273)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'x' (line 275)
        x_385215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'x')
        # Obtaining the member 'ndim' of a type (line 275)
        ndim_385216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 13), x_385215, 'ndim')
        int_385217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 23), 'int')
        # Applying the binary operator '==' (line 275)
        result_eq_385218 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 13), '==', ndim_385216, int_385217)
        
        # Testing the type of an if condition (line 275)
        if_condition_385219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 13), result_eq_385218)
        # Assigning a type to the variable 'if_condition_385219' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'if_condition_385219', if_condition_385219)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to reshape(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'N' (line 276)
        N_385222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'N', False)
        int_385223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'int')
        # Processing the call keyword arguments (line 276)
        kwargs_385224 = {}
        # Getting the type of 'y' (line 276)
        y_385220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'y', False)
        # Obtaining the member 'reshape' of a type (line 276)
        reshape_385221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), y_385220, 'reshape')
        # Calling reshape(args, kwargs) (line 276)
        reshape_call_result_385225 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), reshape_385221, *[N_385222, int_385223], **kwargs_385224)
        
        # Assigning a type to the variable 'y' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'y', reshape_call_result_385225)
        # SSA branch for the else part of an if statement (line 275)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 278)
        # Processing the call arguments (line 278)
        str_385227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 29), 'str', 'invalid shape returned by user-defined rmatvec()')
        # Processing the call keyword arguments (line 278)
        kwargs_385228 = {}
        # Getting the type of 'ValueError' (line 278)
        ValueError_385226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 278)
        ValueError_call_result_385229 = invoke(stypy.reporting.localization.Localization(__file__, 278, 18), ValueError_385226, *[str_385227], **kwargs_385228)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 278, 12), ValueError_call_result_385229, 'raise parameter', BaseException)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'y' (line 280)
        y_385230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', y_385230)
        
        # ################# End of 'rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_385231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385231)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rmatvec'
        return stypy_return_type_385231


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_function_name', 'LinearOperator._rmatvec')
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        str_385232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 8), 'str', 'Default implementation of _rmatvec; defers to adjoint.')
        
        
        
        # Call to type(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'self' (line 284)
        self_385234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'self', False)
        # Processing the call keyword arguments (line 284)
        kwargs_385235 = {}
        # Getting the type of 'type' (line 284)
        type_385233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'type', False)
        # Calling type(args, kwargs) (line 284)
        type_call_result_385236 = invoke(stypy.reporting.localization.Localization(__file__, 284, 11), type_385233, *[self_385234], **kwargs_385235)
        
        # Obtaining the member '_adjoint' of a type (line 284)
        _adjoint_385237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 11), type_call_result_385236, '_adjoint')
        # Getting the type of 'LinearOperator' (line 284)
        LinearOperator_385238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 'LinearOperator')
        # Obtaining the member '_adjoint' of a type (line 284)
        _adjoint_385239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 34), LinearOperator_385238, '_adjoint')
        # Applying the binary operator '==' (line 284)
        result_eq_385240 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), '==', _adjoint_385237, _adjoint_385239)
        
        # Testing the type of an if condition (line 284)
        if_condition_385241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_eq_385240)
        # Assigning a type to the variable 'if_condition_385241' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_385241', if_condition_385241)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplementedError' (line 286)
        NotImplementedError_385242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 286, 12), NotImplementedError_385242, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 284)
        module_type_store.open_ssa_branch('else')
        
        # Call to matvec(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'x' (line 288)
        x_385246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 33), 'x', False)
        # Processing the call keyword arguments (line 288)
        kwargs_385247 = {}
        # Getting the type of 'self' (line 288)
        self_385243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'self', False)
        # Obtaining the member 'H' of a type (line 288)
        H_385244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 19), self_385243, 'H')
        # Obtaining the member 'matvec' of a type (line 288)
        matvec_385245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 19), H_385244, 'matvec')
        # Calling matvec(args, kwargs) (line 288)
        matvec_call_result_385248 = invoke(stypy.reporting.localization.Localization(__file__, 288, 19), matvec_385245, *[x_385246], **kwargs_385247)
        
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'stypy_return_type', matvec_call_result_385248)
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_385249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385249)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_385249


    @norecursion
    def matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'matmat'
        module_type_store = module_type_store.open_function_context('matmat', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.matmat.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.matmat.__dict__.__setitem__('stypy_function_name', 'LinearOperator.matmat')
        LinearOperator.matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        LinearOperator.matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'matmat(...)' code ##################

        str_385250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, (-1)), 'str', 'Matrix-matrix multiplication.\n\n        Performs the operation y=A*X where A is an MxN linear\n        operator and X dense N*K matrix or ndarray.\n\n        Parameters\n        ----------\n        X : {matrix, ndarray}\n            An array with shape (N,K).\n\n        Returns\n        -------\n        Y : {matrix, ndarray}\n            A matrix or ndarray with shape (M,K) depending on\n            the type of the X argument.\n\n        Notes\n        -----\n        This matmat wraps any user-specified matmat routine or overridden\n        _matmat method to ensure that y has the correct type.\n\n        ')
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to asanyarray(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'X' (line 314)
        X_385253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'X', False)
        # Processing the call keyword arguments (line 314)
        kwargs_385254 = {}
        # Getting the type of 'np' (line 314)
        np_385251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'np', False)
        # Obtaining the member 'asanyarray' of a type (line 314)
        asanyarray_385252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), np_385251, 'asanyarray')
        # Calling asanyarray(args, kwargs) (line 314)
        asanyarray_call_result_385255 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), asanyarray_385252, *[X_385253], **kwargs_385254)
        
        # Assigning a type to the variable 'X' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'X', asanyarray_call_result_385255)
        
        
        # Getting the type of 'X' (line 316)
        X_385256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 'X')
        # Obtaining the member 'ndim' of a type (line 316)
        ndim_385257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 11), X_385256, 'ndim')
        int_385258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 21), 'int')
        # Applying the binary operator '!=' (line 316)
        result_ne_385259 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 11), '!=', ndim_385257, int_385258)
        
        # Testing the type of an if condition (line 316)
        if_condition_385260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 8), result_ne_385259)
        # Assigning a type to the variable 'if_condition_385260' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'if_condition_385260', if_condition_385260)
        # SSA begins for if statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 317)
        # Processing the call arguments (line 317)
        str_385262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 29), 'str', 'expected 2-d ndarray or matrix, not %d-d')
        # Getting the type of 'X' (line 318)
        X_385263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 31), 'X', False)
        # Obtaining the member 'ndim' of a type (line 318)
        ndim_385264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 31), X_385263, 'ndim')
        # Applying the binary operator '%' (line 317)
        result_mod_385265 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 29), '%', str_385262, ndim_385264)
        
        # Processing the call keyword arguments (line 317)
        kwargs_385266 = {}
        # Getting the type of 'ValueError' (line 317)
        ValueError_385261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 317)
        ValueError_call_result_385267 = invoke(stypy.reporting.localization.Localization(__file__, 317, 18), ValueError_385261, *[result_mod_385265], **kwargs_385266)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 317, 12), ValueError_call_result_385267, 'raise parameter', BaseException)
        # SSA join for if statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 320):
        
        # Assigning a Subscript to a Name (line 320):
        
        # Obtaining the type of the subscript
        int_385268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 8), 'int')
        # Getting the type of 'self' (line 320)
        self_385269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 14), 'self')
        # Obtaining the member 'shape' of a type (line 320)
        shape_385270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 14), self_385269, 'shape')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___385271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), shape_385270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_385272 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), getitem___385271, int_385268)
        
        # Assigning a type to the variable 'tuple_var_assignment_384909' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_384909', subscript_call_result_385272)
        
        # Assigning a Subscript to a Name (line 320):
        
        # Obtaining the type of the subscript
        int_385273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 8), 'int')
        # Getting the type of 'self' (line 320)
        self_385274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 14), 'self')
        # Obtaining the member 'shape' of a type (line 320)
        shape_385275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 14), self_385274, 'shape')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___385276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), shape_385275, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_385277 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), getitem___385276, int_385273)
        
        # Assigning a type to the variable 'tuple_var_assignment_384910' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_384910', subscript_call_result_385277)
        
        # Assigning a Name to a Name (line 320):
        # Getting the type of 'tuple_var_assignment_384909' (line 320)
        tuple_var_assignment_384909_385278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_384909')
        # Assigning a type to the variable 'M' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'M', tuple_var_assignment_384909_385278)
        
        # Assigning a Name to a Name (line 320):
        # Getting the type of 'tuple_var_assignment_384910' (line 320)
        tuple_var_assignment_384910_385279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_384910')
        # Assigning a type to the variable 'N' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 10), 'N', tuple_var_assignment_384910_385279)
        
        
        
        # Obtaining the type of the subscript
        int_385280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 19), 'int')
        # Getting the type of 'X' (line 322)
        X_385281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'X')
        # Obtaining the member 'shape' of a type (line 322)
        shape_385282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 11), X_385281, 'shape')
        # Obtaining the member '__getitem__' of a type (line 322)
        getitem___385283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 11), shape_385282, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 322)
        subscript_call_result_385284 = invoke(stypy.reporting.localization.Localization(__file__, 322, 11), getitem___385283, int_385280)
        
        # Getting the type of 'N' (line 322)
        N_385285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'N')
        # Applying the binary operator '!=' (line 322)
        result_ne_385286 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 11), '!=', subscript_call_result_385284, N_385285)
        
        # Testing the type of an if condition (line 322)
        if_condition_385287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 8), result_ne_385286)
        # Assigning a type to the variable 'if_condition_385287' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'if_condition_385287', if_condition_385287)
        # SSA begins for if statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 323)
        # Processing the call arguments (line 323)
        str_385289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 29), 'str', 'dimension mismatch: %r, %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 324)
        tuple_385290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 324)
        # Adding element type (line 324)
        # Getting the type of 'self' (line 324)
        self_385291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 32), 'self', False)
        # Obtaining the member 'shape' of a type (line 324)
        shape_385292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 32), self_385291, 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 32), tuple_385290, shape_385292)
        # Adding element type (line 324)
        # Getting the type of 'X' (line 324)
        X_385293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 44), 'X', False)
        # Obtaining the member 'shape' of a type (line 324)
        shape_385294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 44), X_385293, 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 32), tuple_385290, shape_385294)
        
        # Applying the binary operator '%' (line 323)
        result_mod_385295 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 29), '%', str_385289, tuple_385290)
        
        # Processing the call keyword arguments (line 323)
        kwargs_385296 = {}
        # Getting the type of 'ValueError' (line 323)
        ValueError_385288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 323)
        ValueError_call_result_385297 = invoke(stypy.reporting.localization.Localization(__file__, 323, 18), ValueError_385288, *[result_mod_385295], **kwargs_385296)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 323, 12), ValueError_call_result_385297, 'raise parameter', BaseException)
        # SSA join for if statement (line 322)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to _matmat(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'X' (line 326)
        X_385300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 25), 'X', False)
        # Processing the call keyword arguments (line 326)
        kwargs_385301 = {}
        # Getting the type of 'self' (line 326)
        self_385298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'self', False)
        # Obtaining the member '_matmat' of a type (line 326)
        _matmat_385299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), self_385298, '_matmat')
        # Calling _matmat(args, kwargs) (line 326)
        _matmat_call_result_385302 = invoke(stypy.reporting.localization.Localization(__file__, 326, 12), _matmat_385299, *[X_385300], **kwargs_385301)
        
        # Assigning a type to the variable 'Y' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'Y', _matmat_call_result_385302)
        
        
        # Call to isinstance(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'Y' (line 328)
        Y_385304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 22), 'Y', False)
        # Getting the type of 'np' (line 328)
        np_385305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'np', False)
        # Obtaining the member 'matrix' of a type (line 328)
        matrix_385306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 25), np_385305, 'matrix')
        # Processing the call keyword arguments (line 328)
        kwargs_385307 = {}
        # Getting the type of 'isinstance' (line 328)
        isinstance_385303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 328)
        isinstance_call_result_385308 = invoke(stypy.reporting.localization.Localization(__file__, 328, 11), isinstance_385303, *[Y_385304, matrix_385306], **kwargs_385307)
        
        # Testing the type of an if condition (line 328)
        if_condition_385309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 8), isinstance_call_result_385308)
        # Assigning a type to the variable 'if_condition_385309' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'if_condition_385309', if_condition_385309)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Call to asmatrix(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'Y' (line 329)
        Y_385312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 28), 'Y', False)
        # Processing the call keyword arguments (line 329)
        kwargs_385313 = {}
        # Getting the type of 'np' (line 329)
        np_385310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 329)
        asmatrix_385311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), np_385310, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 329)
        asmatrix_call_result_385314 = invoke(stypy.reporting.localization.Localization(__file__, 329, 16), asmatrix_385311, *[Y_385312], **kwargs_385313)
        
        # Assigning a type to the variable 'Y' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'Y', asmatrix_call_result_385314)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'Y' (line 331)
        Y_385315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 15), 'Y')
        # Assigning a type to the variable 'stypy_return_type' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'stypy_return_type', Y_385315)
        
        # ################# End of 'matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_385316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'matmat'
        return stypy_return_type_385316


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__call__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__call__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__call__')
        LinearOperator.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__call__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        # Getting the type of 'self' (line 334)
        self_385317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'self')
        # Getting the type of 'x' (line 334)
        x_385318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'x')
        # Applying the binary operator '*' (line 334)
        result_mul_385319 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 15), '*', self_385317, x_385318)
        
        # Assigning a type to the variable 'stypy_return_type' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'stypy_return_type', result_mul_385319)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_385320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385320)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_385320


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__mul__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__mul__')
        LinearOperator.__mul__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__mul__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        # Call to dot(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'x' (line 337)
        x_385323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'x', False)
        # Processing the call keyword arguments (line 337)
        kwargs_385324 = {}
        # Getting the type of 'self' (line 337)
        self_385321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'self', False)
        # Obtaining the member 'dot' of a type (line 337)
        dot_385322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 15), self_385321, 'dot')
        # Calling dot(args, kwargs) (line 337)
        dot_call_result_385325 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), dot_385322, *[x_385323], **kwargs_385324)
        
        # Assigning a type to the variable 'stypy_return_type' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type', dot_call_result_385325)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_385326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_385326


    @norecursion
    def dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dot'
        module_type_store = module_type_store.open_function_context('dot', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.dot.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.dot.__dict__.__setitem__('stypy_function_name', 'LinearOperator.dot')
        LinearOperator.dot.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.dot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.dot.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.dot', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dot', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dot(...)' code ##################

        str_385327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, (-1)), 'str', 'Matrix-matrix or matrix-vector multiplication.\n\n        Parameters\n        ----------\n        x : array_like\n            1-d or 2-d array, representing a vector or matrix.\n\n        Returns\n        -------\n        Ax : array\n            1-d or 2-d array (depending on the shape of x) that represents\n            the result of applying this linear operator on x.\n\n        ')
        
        
        # Call to isinstance(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'x' (line 354)
        x_385329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'x', False)
        # Getting the type of 'LinearOperator' (line 354)
        LinearOperator_385330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 25), 'LinearOperator', False)
        # Processing the call keyword arguments (line 354)
        kwargs_385331 = {}
        # Getting the type of 'isinstance' (line 354)
        isinstance_385328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 354)
        isinstance_call_result_385332 = invoke(stypy.reporting.localization.Localization(__file__, 354, 11), isinstance_385328, *[x_385329, LinearOperator_385330], **kwargs_385331)
        
        # Testing the type of an if condition (line 354)
        if_condition_385333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 8), isinstance_call_result_385332)
        # Assigning a type to the variable 'if_condition_385333' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'if_condition_385333', if_condition_385333)
        # SSA begins for if statement (line 354)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _ProductLinearOperator(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'self' (line 355)
        self_385335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 42), 'self', False)
        # Getting the type of 'x' (line 355)
        x_385336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'x', False)
        # Processing the call keyword arguments (line 355)
        kwargs_385337 = {}
        # Getting the type of '_ProductLinearOperator' (line 355)
        _ProductLinearOperator_385334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), '_ProductLinearOperator', False)
        # Calling _ProductLinearOperator(args, kwargs) (line 355)
        _ProductLinearOperator_call_result_385338 = invoke(stypy.reporting.localization.Localization(__file__, 355, 19), _ProductLinearOperator_385334, *[self_385335, x_385336], **kwargs_385337)
        
        # Assigning a type to the variable 'stypy_return_type' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'stypy_return_type', _ProductLinearOperator_call_result_385338)
        # SSA branch for the else part of an if statement (line 354)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isscalar(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'x' (line 356)
        x_385341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 25), 'x', False)
        # Processing the call keyword arguments (line 356)
        kwargs_385342 = {}
        # Getting the type of 'np' (line 356)
        np_385339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 356)
        isscalar_385340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 13), np_385339, 'isscalar')
        # Calling isscalar(args, kwargs) (line 356)
        isscalar_call_result_385343 = invoke(stypy.reporting.localization.Localization(__file__, 356, 13), isscalar_385340, *[x_385341], **kwargs_385342)
        
        # Testing the type of an if condition (line 356)
        if_condition_385344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 13), isscalar_call_result_385343)
        # Assigning a type to the variable 'if_condition_385344' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'if_condition_385344', if_condition_385344)
        # SSA begins for if statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _ScaledLinearOperator(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'self' (line 357)
        self_385346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 41), 'self', False)
        # Getting the type of 'x' (line 357)
        x_385347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 47), 'x', False)
        # Processing the call keyword arguments (line 357)
        kwargs_385348 = {}
        # Getting the type of '_ScaledLinearOperator' (line 357)
        _ScaledLinearOperator_385345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), '_ScaledLinearOperator', False)
        # Calling _ScaledLinearOperator(args, kwargs) (line 357)
        _ScaledLinearOperator_call_result_385349 = invoke(stypy.reporting.localization.Localization(__file__, 357, 19), _ScaledLinearOperator_385345, *[self_385346, x_385347], **kwargs_385348)
        
        # Assigning a type to the variable 'stypy_return_type' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'stypy_return_type', _ScaledLinearOperator_call_result_385349)
        # SSA branch for the else part of an if statement (line 356)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 359):
        
        # Assigning a Call to a Name (line 359):
        
        # Call to asarray(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'x' (line 359)
        x_385352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'x', False)
        # Processing the call keyword arguments (line 359)
        kwargs_385353 = {}
        # Getting the type of 'np' (line 359)
        np_385350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 359)
        asarray_385351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), np_385350, 'asarray')
        # Calling asarray(args, kwargs) (line 359)
        asarray_call_result_385354 = invoke(stypy.reporting.localization.Localization(__file__, 359, 16), asarray_385351, *[x_385352], **kwargs_385353)
        
        # Assigning a type to the variable 'x' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'x', asarray_call_result_385354)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 361)
        x_385355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'x')
        # Obtaining the member 'ndim' of a type (line 361)
        ndim_385356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 15), x_385355, 'ndim')
        int_385357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 25), 'int')
        # Applying the binary operator '==' (line 361)
        result_eq_385358 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 15), '==', ndim_385356, int_385357)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 361)
        x_385359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'x')
        # Obtaining the member 'ndim' of a type (line 361)
        ndim_385360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 30), x_385359, 'ndim')
        int_385361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 40), 'int')
        # Applying the binary operator '==' (line 361)
        result_eq_385362 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 30), '==', ndim_385360, int_385361)
        
        
        
        # Obtaining the type of the subscript
        int_385363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 54), 'int')
        # Getting the type of 'x' (line 361)
        x_385364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 46), 'x')
        # Obtaining the member 'shape' of a type (line 361)
        shape_385365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 46), x_385364, 'shape')
        # Obtaining the member '__getitem__' of a type (line 361)
        getitem___385366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 46), shape_385365, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 361)
        subscript_call_result_385367 = invoke(stypy.reporting.localization.Localization(__file__, 361, 46), getitem___385366, int_385363)
        
        int_385368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 60), 'int')
        # Applying the binary operator '==' (line 361)
        result_eq_385369 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 46), '==', subscript_call_result_385367, int_385368)
        
        # Applying the binary operator 'and' (line 361)
        result_and_keyword_385370 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 30), 'and', result_eq_385362, result_eq_385369)
        
        # Applying the binary operator 'or' (line 361)
        result_or_keyword_385371 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 15), 'or', result_eq_385358, result_and_keyword_385370)
        
        # Testing the type of an if condition (line 361)
        if_condition_385372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 12), result_or_keyword_385371)
        # Assigning a type to the variable 'if_condition_385372' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'if_condition_385372', if_condition_385372)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to matvec(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'x' (line 362)
        x_385375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'x', False)
        # Processing the call keyword arguments (line 362)
        kwargs_385376 = {}
        # Getting the type of 'self' (line 362)
        self_385373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'self', False)
        # Obtaining the member 'matvec' of a type (line 362)
        matvec_385374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 23), self_385373, 'matvec')
        # Calling matvec(args, kwargs) (line 362)
        matvec_call_result_385377 = invoke(stypy.reporting.localization.Localization(__file__, 362, 23), matvec_385374, *[x_385375], **kwargs_385376)
        
        # Assigning a type to the variable 'stypy_return_type' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'stypy_return_type', matvec_call_result_385377)
        # SSA branch for the else part of an if statement (line 361)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'x' (line 363)
        x_385378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 17), 'x')
        # Obtaining the member 'ndim' of a type (line 363)
        ndim_385379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 17), x_385378, 'ndim')
        int_385380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 27), 'int')
        # Applying the binary operator '==' (line 363)
        result_eq_385381 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 17), '==', ndim_385379, int_385380)
        
        # Testing the type of an if condition (line 363)
        if_condition_385382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 17), result_eq_385381)
        # Assigning a type to the variable 'if_condition_385382' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 17), 'if_condition_385382', if_condition_385382)
        # SSA begins for if statement (line 363)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to matmat(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'x' (line 364)
        x_385385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 35), 'x', False)
        # Processing the call keyword arguments (line 364)
        kwargs_385386 = {}
        # Getting the type of 'self' (line 364)
        self_385383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'self', False)
        # Obtaining the member 'matmat' of a type (line 364)
        matmat_385384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 23), self_385383, 'matmat')
        # Calling matmat(args, kwargs) (line 364)
        matmat_call_result_385387 = invoke(stypy.reporting.localization.Localization(__file__, 364, 23), matmat_385384, *[x_385385], **kwargs_385386)
        
        # Assigning a type to the variable 'stypy_return_type' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'stypy_return_type', matmat_call_result_385387)
        # SSA branch for the else part of an if statement (line 363)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 366)
        # Processing the call arguments (line 366)
        str_385389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 33), 'str', 'expected 1-d or 2-d array or matrix, got %r')
        # Getting the type of 'x' (line 367)
        x_385390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 35), 'x', False)
        # Applying the binary operator '%' (line 366)
        result_mod_385391 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 33), '%', str_385389, x_385390)
        
        # Processing the call keyword arguments (line 366)
        kwargs_385392 = {}
        # Getting the type of 'ValueError' (line 366)
        ValueError_385388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 366)
        ValueError_call_result_385393 = invoke(stypy.reporting.localization.Localization(__file__, 366, 22), ValueError_385388, *[result_mod_385391], **kwargs_385392)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 366, 16), ValueError_call_result_385393, 'raise parameter', BaseException)
        # SSA join for if statement (line 363)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 356)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 354)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dot' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_385394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385394)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dot'
        return stypy_return_type_385394


    @norecursion
    def __matmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__matmul__'
        module_type_store = module_type_store.open_function_context('__matmul__', 369, 4, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__matmul__')
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__matmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__matmul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__matmul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__matmul__(...)' code ##################

        
        
        # Call to isscalar(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'other' (line 370)
        other_385397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'other', False)
        # Processing the call keyword arguments (line 370)
        kwargs_385398 = {}
        # Getting the type of 'np' (line 370)
        np_385395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 370)
        isscalar_385396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 11), np_385395, 'isscalar')
        # Calling isscalar(args, kwargs) (line 370)
        isscalar_call_result_385399 = invoke(stypy.reporting.localization.Localization(__file__, 370, 11), isscalar_385396, *[other_385397], **kwargs_385398)
        
        # Testing the type of an if condition (line 370)
        if_condition_385400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 8), isscalar_call_result_385399)
        # Assigning a type to the variable 'if_condition_385400' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'if_condition_385400', if_condition_385400)
        # SSA begins for if statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 371)
        # Processing the call arguments (line 371)
        str_385402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 29), 'str', "Scalar operands are not allowed, use '*' instead")
        # Processing the call keyword arguments (line 371)
        kwargs_385403 = {}
        # Getting the type of 'ValueError' (line 371)
        ValueError_385401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 371)
        ValueError_call_result_385404 = invoke(stypy.reporting.localization.Localization(__file__, 371, 18), ValueError_385401, *[str_385402], **kwargs_385403)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 371, 12), ValueError_call_result_385404, 'raise parameter', BaseException)
        # SSA join for if statement (line 370)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __mul__(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'other' (line 373)
        other_385407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 28), 'other', False)
        # Processing the call keyword arguments (line 373)
        kwargs_385408 = {}
        # Getting the type of 'self' (line 373)
        self_385405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'self', False)
        # Obtaining the member '__mul__' of a type (line 373)
        mul___385406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), self_385405, '__mul__')
        # Calling __mul__(args, kwargs) (line 373)
        mul___call_result_385409 = invoke(stypy.reporting.localization.Localization(__file__, 373, 15), mul___385406, *[other_385407], **kwargs_385408)
        
        # Assigning a type to the variable 'stypy_return_type' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'stypy_return_type', mul___call_result_385409)
        
        # ################# End of '__matmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__matmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_385410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__matmul__'
        return stypy_return_type_385410


    @norecursion
    def __rmatmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmatmul__'
        module_type_store = module_type_store.open_function_context('__rmatmul__', 375, 4, False)
        # Assigning a type to the variable 'self' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__rmatmul__')
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__rmatmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__rmatmul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmatmul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmatmul__(...)' code ##################

        
        
        # Call to isscalar(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'other' (line 376)
        other_385413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 23), 'other', False)
        # Processing the call keyword arguments (line 376)
        kwargs_385414 = {}
        # Getting the type of 'np' (line 376)
        np_385411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 11), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 376)
        isscalar_385412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 11), np_385411, 'isscalar')
        # Calling isscalar(args, kwargs) (line 376)
        isscalar_call_result_385415 = invoke(stypy.reporting.localization.Localization(__file__, 376, 11), isscalar_385412, *[other_385413], **kwargs_385414)
        
        # Testing the type of an if condition (line 376)
        if_condition_385416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 8), isscalar_call_result_385415)
        # Assigning a type to the variable 'if_condition_385416' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'if_condition_385416', if_condition_385416)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 377)
        # Processing the call arguments (line 377)
        str_385418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 29), 'str', "Scalar operands are not allowed, use '*' instead")
        # Processing the call keyword arguments (line 377)
        kwargs_385419 = {}
        # Getting the type of 'ValueError' (line 377)
        ValueError_385417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 377)
        ValueError_call_result_385420 = invoke(stypy.reporting.localization.Localization(__file__, 377, 18), ValueError_385417, *[str_385418], **kwargs_385419)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 377, 12), ValueError_call_result_385420, 'raise parameter', BaseException)
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __rmul__(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'other' (line 379)
        other_385423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 29), 'other', False)
        # Processing the call keyword arguments (line 379)
        kwargs_385424 = {}
        # Getting the type of 'self' (line 379)
        self_385421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'self', False)
        # Obtaining the member '__rmul__' of a type (line 379)
        rmul___385422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), self_385421, '__rmul__')
        # Calling __rmul__(args, kwargs) (line 379)
        rmul___call_result_385425 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), rmul___385422, *[other_385423], **kwargs_385424)
        
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', rmul___call_result_385425)
        
        # ################# End of '__rmatmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmatmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 375)
        stypy_return_type_385426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmatmul__'
        return stypy_return_type_385426


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__rmul__')
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__rmul__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmul__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmul__(...)' code ##################

        
        
        # Call to isscalar(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'x' (line 382)
        x_385429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 23), 'x', False)
        # Processing the call keyword arguments (line 382)
        kwargs_385430 = {}
        # Getting the type of 'np' (line 382)
        np_385427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 11), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 382)
        isscalar_385428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 11), np_385427, 'isscalar')
        # Calling isscalar(args, kwargs) (line 382)
        isscalar_call_result_385431 = invoke(stypy.reporting.localization.Localization(__file__, 382, 11), isscalar_385428, *[x_385429], **kwargs_385430)
        
        # Testing the type of an if condition (line 382)
        if_condition_385432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 8), isscalar_call_result_385431)
        # Assigning a type to the variable 'if_condition_385432' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'if_condition_385432', if_condition_385432)
        # SSA begins for if statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _ScaledLinearOperator(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'self' (line 383)
        self_385434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 41), 'self', False)
        # Getting the type of 'x' (line 383)
        x_385435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 47), 'x', False)
        # Processing the call keyword arguments (line 383)
        kwargs_385436 = {}
        # Getting the type of '_ScaledLinearOperator' (line 383)
        _ScaledLinearOperator_385433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), '_ScaledLinearOperator', False)
        # Calling _ScaledLinearOperator(args, kwargs) (line 383)
        _ScaledLinearOperator_call_result_385437 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), _ScaledLinearOperator_385433, *[self_385434, x_385435], **kwargs_385436)
        
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'stypy_return_type', _ScaledLinearOperator_call_result_385437)
        # SSA branch for the else part of an if statement (line 382)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 385)
        NotImplemented_385438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'stypy_return_type', NotImplemented_385438)
        # SSA join for if statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_385439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385439)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_385439


    @norecursion
    def __pow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pow__'
        module_type_store = module_type_store.open_function_context('__pow__', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__pow__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__pow__')
        LinearOperator.__pow__.__dict__.__setitem__('stypy_param_names_list', ['p'])
        LinearOperator.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__pow__', ['p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pow__', localization, ['p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pow__(...)' code ##################

        
        
        # Call to isscalar(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'p' (line 388)
        p_385442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'p', False)
        # Processing the call keyword arguments (line 388)
        kwargs_385443 = {}
        # Getting the type of 'np' (line 388)
        np_385440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 388)
        isscalar_385441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 11), np_385440, 'isscalar')
        # Calling isscalar(args, kwargs) (line 388)
        isscalar_call_result_385444 = invoke(stypy.reporting.localization.Localization(__file__, 388, 11), isscalar_385441, *[p_385442], **kwargs_385443)
        
        # Testing the type of an if condition (line 388)
        if_condition_385445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 8), isscalar_call_result_385444)
        # Assigning a type to the variable 'if_condition_385445' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'if_condition_385445', if_condition_385445)
        # SSA begins for if statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _PowerLinearOperator(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'self' (line 389)
        self_385447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 40), 'self', False)
        # Getting the type of 'p' (line 389)
        p_385448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 46), 'p', False)
        # Processing the call keyword arguments (line 389)
        kwargs_385449 = {}
        # Getting the type of '_PowerLinearOperator' (line 389)
        _PowerLinearOperator_385446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), '_PowerLinearOperator', False)
        # Calling _PowerLinearOperator(args, kwargs) (line 389)
        _PowerLinearOperator_call_result_385450 = invoke(stypy.reporting.localization.Localization(__file__, 389, 19), _PowerLinearOperator_385446, *[self_385447, p_385448], **kwargs_385449)
        
        # Assigning a type to the variable 'stypy_return_type' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'stypy_return_type', _PowerLinearOperator_call_result_385450)
        # SSA branch for the else part of an if statement (line 388)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 391)
        NotImplemented_385451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'stypy_return_type', NotImplemented_385451)
        # SSA join for if statement (line 388)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__pow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pow__' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_385452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pow__'
        return stypy_return_type_385452


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__add__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__add__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__add__')
        LinearOperator.__add__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__add__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        
        # Call to isinstance(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'x' (line 394)
        x_385454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 22), 'x', False)
        # Getting the type of 'LinearOperator' (line 394)
        LinearOperator_385455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 25), 'LinearOperator', False)
        # Processing the call keyword arguments (line 394)
        kwargs_385456 = {}
        # Getting the type of 'isinstance' (line 394)
        isinstance_385453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 394)
        isinstance_call_result_385457 = invoke(stypy.reporting.localization.Localization(__file__, 394, 11), isinstance_385453, *[x_385454, LinearOperator_385455], **kwargs_385456)
        
        # Testing the type of an if condition (line 394)
        if_condition_385458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 8), isinstance_call_result_385457)
        # Assigning a type to the variable 'if_condition_385458' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'if_condition_385458', if_condition_385458)
        # SSA begins for if statement (line 394)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _SumLinearOperator(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'self' (line 395)
        self_385460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 38), 'self', False)
        # Getting the type of 'x' (line 395)
        x_385461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 44), 'x', False)
        # Processing the call keyword arguments (line 395)
        kwargs_385462 = {}
        # Getting the type of '_SumLinearOperator' (line 395)
        _SumLinearOperator_385459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), '_SumLinearOperator', False)
        # Calling _SumLinearOperator(args, kwargs) (line 395)
        _SumLinearOperator_call_result_385463 = invoke(stypy.reporting.localization.Localization(__file__, 395, 19), _SumLinearOperator_385459, *[self_385460, x_385461], **kwargs_385462)
        
        # Assigning a type to the variable 'stypy_return_type' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'stypy_return_type', _SumLinearOperator_call_result_385463)
        # SSA branch for the else part of an if statement (line 394)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 397)
        NotImplemented_385464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'stypy_return_type', NotImplemented_385464)
        # SSA join for if statement (line 394)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_385465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385465)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_385465


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 399, 4, False)
        # Assigning a type to the variable 'self' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__neg__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__neg__')
        LinearOperator.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        LinearOperator.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__neg__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__neg__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__neg__(...)' code ##################

        
        # Call to _ScaledLinearOperator(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'self' (line 400)
        self_385467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 37), 'self', False)
        int_385468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 43), 'int')
        # Processing the call keyword arguments (line 400)
        kwargs_385469 = {}
        # Getting the type of '_ScaledLinearOperator' (line 400)
        _ScaledLinearOperator_385466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), '_ScaledLinearOperator', False)
        # Calling _ScaledLinearOperator(args, kwargs) (line 400)
        _ScaledLinearOperator_call_result_385470 = invoke(stypy.reporting.localization.Localization(__file__, 400, 15), _ScaledLinearOperator_385466, *[self_385467, int_385468], **kwargs_385469)
        
        # Assigning a type to the variable 'stypy_return_type' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'stypy_return_type', _ScaledLinearOperator_call_result_385470)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_385471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_385471


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.__sub__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.__sub__')
        LinearOperator.__sub__.__dict__.__setitem__('stypy_param_names_list', ['x'])
        LinearOperator.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.__sub__', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        
        # Call to __add__(...): (line 403)
        # Processing the call arguments (line 403)
        
        # Getting the type of 'x' (line 403)
        x_385474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 29), 'x', False)
        # Applying the 'usub' unary operator (line 403)
        result___neg___385475 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 28), 'usub', x_385474)
        
        # Processing the call keyword arguments (line 403)
        kwargs_385476 = {}
        # Getting the type of 'self' (line 403)
        self_385472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'self', False)
        # Obtaining the member '__add__' of a type (line 403)
        add___385473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 15), self_385472, '__add__')
        # Calling __add__(args, kwargs) (line 403)
        add___call_result_385477 = invoke(stypy.reporting.localization.Localization(__file__, 403, 15), add___385473, *[result___neg___385475], **kwargs_385476)
        
        # Assigning a type to the variable 'stypy_return_type' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'stypy_return_type', add___call_result_385477)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_385478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_385478


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 405, 4, False)
        # Assigning a type to the variable 'self' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'LinearOperator.stypy__repr__')
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Tuple (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_385479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 8), 'int')
        # Getting the type of 'self' (line 406)
        self_385480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 14), 'self')
        # Obtaining the member 'shape' of a type (line 406)
        shape_385481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 14), self_385480, 'shape')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___385482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), shape_385481, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_385483 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), getitem___385482, int_385479)
        
        # Assigning a type to the variable 'tuple_var_assignment_384911' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_384911', subscript_call_result_385483)
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_385484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 8), 'int')
        # Getting the type of 'self' (line 406)
        self_385485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 14), 'self')
        # Obtaining the member 'shape' of a type (line 406)
        shape_385486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 14), self_385485, 'shape')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___385487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), shape_385486, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_385488 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), getitem___385487, int_385484)
        
        # Assigning a type to the variable 'tuple_var_assignment_384912' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_384912', subscript_call_result_385488)
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_384911' (line 406)
        tuple_var_assignment_384911_385489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_384911')
        # Assigning a type to the variable 'M' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'M', tuple_var_assignment_384911_385489)
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_384912' (line 406)
        tuple_var_assignment_384912_385490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_384912')
        # Assigning a type to the variable 'N' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 10), 'N', tuple_var_assignment_384912_385490)
        
        # Type idiom detected: calculating its left and rigth part (line 407)
        # Getting the type of 'self' (line 407)
        self_385491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 407)
        dtype_385492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 11), self_385491, 'dtype')
        # Getting the type of 'None' (line 407)
        None_385493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'None')
        
        (may_be_385494, more_types_in_union_385495) = may_be_none(dtype_385492, None_385493)

        if may_be_385494:

            if more_types_in_union_385495:
                # Runtime conditional SSA (line 407)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 408):
            
            # Assigning a Str to a Name (line 408):
            str_385496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 17), 'str', 'unspecified dtype')
            # Assigning a type to the variable 'dt' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'dt', str_385496)

            if more_types_in_union_385495:
                # Runtime conditional SSA for else branch (line 407)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_385494) or more_types_in_union_385495):
            
            # Assigning a BinOp to a Name (line 410):
            
            # Assigning a BinOp to a Name (line 410):
            str_385497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 17), 'str', 'dtype=')
            
            # Call to str(...): (line 410)
            # Processing the call arguments (line 410)
            # Getting the type of 'self' (line 410)
            self_385499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 32), 'self', False)
            # Obtaining the member 'dtype' of a type (line 410)
            dtype_385500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 32), self_385499, 'dtype')
            # Processing the call keyword arguments (line 410)
            kwargs_385501 = {}
            # Getting the type of 'str' (line 410)
            str_385498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 28), 'str', False)
            # Calling str(args, kwargs) (line 410)
            str_call_result_385502 = invoke(stypy.reporting.localization.Localization(__file__, 410, 28), str_385498, *[dtype_385500], **kwargs_385501)
            
            # Applying the binary operator '+' (line 410)
            result_add_385503 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 17), '+', str_385497, str_call_result_385502)
            
            # Assigning a type to the variable 'dt' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'dt', result_add_385503)

            if (may_be_385494 and more_types_in_union_385495):
                # SSA join for if statement (line 407)
                module_type_store = module_type_store.join_ssa_context()


        
        str_385504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 15), 'str', '<%dx%d %s with %s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 412)
        tuple_385505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 412)
        # Adding element type (line 412)
        # Getting the type of 'M' (line 412)
        M_385506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 39), 'M')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 39), tuple_385505, M_385506)
        # Adding element type (line 412)
        # Getting the type of 'N' (line 412)
        N_385507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 42), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 39), tuple_385505, N_385507)
        # Adding element type (line 412)
        # Getting the type of 'self' (line 412)
        self_385508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 45), 'self')
        # Obtaining the member '__class__' of a type (line 412)
        class___385509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 45), self_385508, '__class__')
        # Obtaining the member '__name__' of a type (line 412)
        name___385510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 45), class___385509, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 39), tuple_385505, name___385510)
        # Adding element type (line 412)
        # Getting the type of 'dt' (line 412)
        dt_385511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 70), 'dt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 39), tuple_385505, dt_385511)
        
        # Applying the binary operator '%' (line 412)
        result_mod_385512 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 15), '%', str_385504, tuple_385505)
        
        # Assigning a type to the variable 'stypy_return_type' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'stypy_return_type', result_mod_385512)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 405)
        stypy_return_type_385513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_385513


    @norecursion
    def adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'adjoint'
        module_type_store = module_type_store.open_function_context('adjoint', 414, 4, False)
        # Assigning a type to the variable 'self' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.adjoint.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_function_name', 'LinearOperator.adjoint')
        LinearOperator.adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        LinearOperator.adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'adjoint(...)' code ##################

        str_385514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, (-1)), 'str', 'Hermitian adjoint.\n\n        Returns the Hermitian adjoint of self, aka the Hermitian\n        conjugate or Hermitian transpose. For a complex matrix, the\n        Hermitian adjoint is equal to the conjugate transpose.\n\n        Can be abbreviated self.H instead of self.adjoint().\n\n        Returns\n        -------\n        A_H : LinearOperator\n            Hermitian adjoint of self.\n        ')
        
        # Call to _adjoint(...): (line 428)
        # Processing the call keyword arguments (line 428)
        kwargs_385517 = {}
        # Getting the type of 'self' (line 428)
        self_385515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'self', False)
        # Obtaining the member '_adjoint' of a type (line 428)
        _adjoint_385516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 15), self_385515, '_adjoint')
        # Calling _adjoint(args, kwargs) (line 428)
        _adjoint_call_result_385518 = invoke(stypy.reporting.localization.Localization(__file__, 428, 15), _adjoint_385516, *[], **kwargs_385517)
        
        # Assigning a type to the variable 'stypy_return_type' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'stypy_return_type', _adjoint_call_result_385518)
        
        # ################# End of 'adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 414)
        stypy_return_type_385519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385519)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'adjoint'
        return stypy_return_type_385519

    
    # Assigning a Call to a Name (line 430):

    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 432, 4, False)
        # Assigning a type to the variable 'self' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator.transpose.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator.transpose.__dict__.__setitem__('stypy_function_name', 'LinearOperator.transpose')
        LinearOperator.transpose.__dict__.__setitem__('stypy_param_names_list', [])
        LinearOperator.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator.transpose.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator.transpose', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transpose', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transpose(...)' code ##################

        str_385520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, (-1)), 'str', 'Transpose this linear operator.\n\n        Returns a LinearOperator that represents the transpose of this one.\n        Can be abbreviated self.T instead of self.transpose().\n        ')
        
        # Call to _transpose(...): (line 438)
        # Processing the call keyword arguments (line 438)
        kwargs_385523 = {}
        # Getting the type of 'self' (line 438)
        self_385521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'self', False)
        # Obtaining the member '_transpose' of a type (line 438)
        _transpose_385522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 15), self_385521, '_transpose')
        # Calling _transpose(args, kwargs) (line 438)
        _transpose_call_result_385524 = invoke(stypy.reporting.localization.Localization(__file__, 438, 15), _transpose_385522, *[], **kwargs_385523)
        
        # Assigning a type to the variable 'stypy_return_type' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'stypy_return_type', _transpose_call_result_385524)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 432)
        stypy_return_type_385525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385525)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_385525

    
    # Assigning a Call to a Name (line 440):

    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LinearOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_function_name', 'LinearOperator._adjoint')
        LinearOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        LinearOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LinearOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinearOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        str_385526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'str', 'Default implementation of _adjoint; defers to rmatvec.')
        
        # Assigning a Tuple to a Name (line 444):
        
        # Assigning a Tuple to a Name (line 444):
        
        # Obtaining an instance of the builtin type 'tuple' (line 444)
        tuple_385527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 444)
        # Adding element type (line 444)
        
        # Obtaining the type of the subscript
        int_385528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 28), 'int')
        # Getting the type of 'self' (line 444)
        self_385529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 17), 'self')
        # Obtaining the member 'shape' of a type (line 444)
        shape_385530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 17), self_385529, 'shape')
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___385531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 17), shape_385530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_385532 = invoke(stypy.reporting.localization.Localization(__file__, 444, 17), getitem___385531, int_385528)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 17), tuple_385527, subscript_call_result_385532)
        # Adding element type (line 444)
        
        # Obtaining the type of the subscript
        int_385533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 43), 'int')
        # Getting the type of 'self' (line 444)
        self_385534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 32), 'self')
        # Obtaining the member 'shape' of a type (line 444)
        shape_385535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 32), self_385534, 'shape')
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___385536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 32), shape_385535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_385537 = invoke(stypy.reporting.localization.Localization(__file__, 444, 32), getitem___385536, int_385533)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 17), tuple_385527, subscript_call_result_385537)
        
        # Assigning a type to the variable 'shape' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'shape', tuple_385527)
        
        # Call to _CustomLinearOperator(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'shape' (line 445)
        shape_385539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 37), 'shape', False)
        # Processing the call keyword arguments (line 445)
        # Getting the type of 'self' (line 445)
        self_385540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 51), 'self', False)
        # Obtaining the member 'rmatvec' of a type (line 445)
        rmatvec_385541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 51), self_385540, 'rmatvec')
        keyword_385542 = rmatvec_385541
        # Getting the type of 'self' (line 446)
        self_385543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 45), 'self', False)
        # Obtaining the member 'matvec' of a type (line 446)
        matvec_385544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 45), self_385543, 'matvec')
        keyword_385545 = matvec_385544
        # Getting the type of 'self' (line 447)
        self_385546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 447)
        dtype_385547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 43), self_385546, 'dtype')
        keyword_385548 = dtype_385547
        kwargs_385549 = {'dtype': keyword_385548, 'rmatvec': keyword_385545, 'matvec': keyword_385542}
        # Getting the type of '_CustomLinearOperator' (line 445)
        _CustomLinearOperator_385538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 15), '_CustomLinearOperator', False)
        # Calling _CustomLinearOperator(args, kwargs) (line 445)
        _CustomLinearOperator_call_result_385550 = invoke(stypy.reporting.localization.Localization(__file__, 445, 15), _CustomLinearOperator_385538, *[shape_385539], **kwargs_385549)
        
        # Assigning a type to the variable 'stypy_return_type' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'stypy_return_type', _CustomLinearOperator_call_result_385550)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_385551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_385551


# Assigning a type to the variable 'LinearOperator' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'LinearOperator', LinearOperator)

# Assigning a Call to a Name (line 430):

# Call to property(...): (line 430)
# Processing the call arguments (line 430)
# Getting the type of 'LinearOperator'
LinearOperator_385553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LinearOperator', False)
# Obtaining the member 'adjoint' of a type
adjoint_385554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LinearOperator_385553, 'adjoint')
# Processing the call keyword arguments (line 430)
kwargs_385555 = {}
# Getting the type of 'property' (line 430)
property_385552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'property', False)
# Calling property(args, kwargs) (line 430)
property_call_result_385556 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), property_385552, *[adjoint_385554], **kwargs_385555)

# Getting the type of 'LinearOperator'
LinearOperator_385557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LinearOperator')
# Setting the type of the member 'H' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LinearOperator_385557, 'H', property_call_result_385556)

# Assigning a Call to a Name (line 440):

# Call to property(...): (line 440)
# Processing the call arguments (line 440)
# Getting the type of 'LinearOperator'
LinearOperator_385559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LinearOperator', False)
# Obtaining the member 'transpose' of a type
transpose_385560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LinearOperator_385559, 'transpose')
# Processing the call keyword arguments (line 440)
kwargs_385561 = {}
# Getting the type of 'property' (line 440)
property_385558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'property', False)
# Calling property(args, kwargs) (line 440)
property_call_result_385562 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), property_385558, *[transpose_385560], **kwargs_385561)

# Getting the type of 'LinearOperator'
LinearOperator_385563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LinearOperator')
# Setting the type of the member 'T' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LinearOperator_385563, 'T', property_call_result_385562)
# Declaration of the '_CustomLinearOperator' class
# Getting the type of 'LinearOperator' (line 450)
LinearOperator_385564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 28), 'LinearOperator')

class _CustomLinearOperator(LinearOperator_385564, ):
    str_385565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 4), 'str', 'Linear operator defined in terms of user-specified operations.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 453)
        None_385566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 46), 'None')
        # Getting the type of 'None' (line 453)
        None_385567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 59), 'None')
        # Getting the type of 'None' (line 453)
        None_385568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 71), 'None')
        defaults = [None_385566, None_385567, None_385568]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 453, 4, False)
        # Assigning a type to the variable 'self' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CustomLinearOperator.__init__', ['shape', 'matvec', 'rmatvec', 'matmat', 'dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['shape', 'matvec', 'rmatvec', 'matmat', 'dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'dtype' (line 454)
        dtype_385575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 52), 'dtype', False)
        # Getting the type of 'shape' (line 454)
        shape_385576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 59), 'shape', False)
        # Processing the call keyword arguments (line 454)
        kwargs_385577 = {}
        
        # Call to super(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of '_CustomLinearOperator' (line 454)
        _CustomLinearOperator_385570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 14), '_CustomLinearOperator', False)
        # Getting the type of 'self' (line 454)
        self_385571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 37), 'self', False)
        # Processing the call keyword arguments (line 454)
        kwargs_385572 = {}
        # Getting the type of 'super' (line 454)
        super_385569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'super', False)
        # Calling super(args, kwargs) (line 454)
        super_call_result_385573 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), super_385569, *[_CustomLinearOperator_385570, self_385571], **kwargs_385572)
        
        # Obtaining the member '__init__' of a type (line 454)
        init___385574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), super_call_result_385573, '__init__')
        # Calling __init__(args, kwargs) (line 454)
        init___call_result_385578 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), init___385574, *[dtype_385575, shape_385576], **kwargs_385577)
        
        
        # Assigning a Tuple to a Attribute (line 456):
        
        # Assigning a Tuple to a Attribute (line 456):
        
        # Obtaining an instance of the builtin type 'tuple' (line 456)
        tuple_385579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 456)
        
        # Getting the type of 'self' (line 456)
        self_385580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'self')
        # Setting the type of the member 'args' of a type (line 456)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), self_385580, 'args', tuple_385579)
        
        # Assigning a Name to a Attribute (line 458):
        
        # Assigning a Name to a Attribute (line 458):
        # Getting the type of 'matvec' (line 458)
        matvec_385581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 29), 'matvec')
        # Getting the type of 'self' (line 458)
        self_385582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'self')
        # Setting the type of the member '__matvec_impl' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), self_385582, '__matvec_impl', matvec_385581)
        
        # Assigning a Name to a Attribute (line 459):
        
        # Assigning a Name to a Attribute (line 459):
        # Getting the type of 'rmatvec' (line 459)
        rmatvec_385583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 30), 'rmatvec')
        # Getting the type of 'self' (line 459)
        self_385584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'self')
        # Setting the type of the member '__rmatvec_impl' of a type (line 459)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), self_385584, '__rmatvec_impl', rmatvec_385583)
        
        # Assigning a Name to a Attribute (line 460):
        
        # Assigning a Name to a Attribute (line 460):
        # Getting the type of 'matmat' (line 460)
        matmat_385585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 29), 'matmat')
        # Getting the type of 'self' (line 460)
        self_385586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'self')
        # Setting the type of the member '__matmat_impl' of a type (line 460)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), self_385586, '__matmat_impl', matmat_385585)
        
        # Call to _init_dtype(...): (line 462)
        # Processing the call keyword arguments (line 462)
        kwargs_385589 = {}
        # Getting the type of 'self' (line 462)
        self_385587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'self', False)
        # Obtaining the member '_init_dtype' of a type (line 462)
        _init_dtype_385588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), self_385587, '_init_dtype')
        # Calling _init_dtype(args, kwargs) (line 462)
        _init_dtype_call_result_385590 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), _init_dtype_385588, *[], **kwargs_385589)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 464, 4, False)
        # Assigning a type to the variable 'self' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_function_name', '_CustomLinearOperator._matmat')
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _CustomLinearOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CustomLinearOperator._matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        
        # Getting the type of 'self' (line 465)
        self_385591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'self')
        # Obtaining the member '__matmat_impl' of a type (line 465)
        matmat_impl_385592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 11), self_385591, '__matmat_impl')
        # Getting the type of 'None' (line 465)
        None_385593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 37), 'None')
        # Applying the binary operator 'isnot' (line 465)
        result_is_not_385594 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 11), 'isnot', matmat_impl_385592, None_385593)
        
        # Testing the type of an if condition (line 465)
        if_condition_385595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 8), result_is_not_385594)
        # Assigning a type to the variable 'if_condition_385595' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'if_condition_385595', if_condition_385595)
        # SSA begins for if statement (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __matmat_impl(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'X' (line 466)
        X_385598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 38), 'X', False)
        # Processing the call keyword arguments (line 466)
        kwargs_385599 = {}
        # Getting the type of 'self' (line 466)
        self_385596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 19), 'self', False)
        # Obtaining the member '__matmat_impl' of a type (line 466)
        matmat_impl_385597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 19), self_385596, '__matmat_impl')
        # Calling __matmat_impl(args, kwargs) (line 466)
        matmat_impl_call_result_385600 = invoke(stypy.reporting.localization.Localization(__file__, 466, 19), matmat_impl_385597, *[X_385598], **kwargs_385599)
        
        # Assigning a type to the variable 'stypy_return_type' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'stypy_return_type', matmat_impl_call_result_385600)
        # SSA branch for the else part of an if statement (line 465)
        module_type_store.open_ssa_branch('else')
        
        # Call to _matmat(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'X' (line 468)
        X_385607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 62), 'X', False)
        # Processing the call keyword arguments (line 468)
        kwargs_385608 = {}
        
        # Call to super(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of '_CustomLinearOperator' (line 468)
        _CustomLinearOperator_385602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 25), '_CustomLinearOperator', False)
        # Getting the type of 'self' (line 468)
        self_385603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 48), 'self', False)
        # Processing the call keyword arguments (line 468)
        kwargs_385604 = {}
        # Getting the type of 'super' (line 468)
        super_385601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 19), 'super', False)
        # Calling super(args, kwargs) (line 468)
        super_call_result_385605 = invoke(stypy.reporting.localization.Localization(__file__, 468, 19), super_385601, *[_CustomLinearOperator_385602, self_385603], **kwargs_385604)
        
        # Obtaining the member '_matmat' of a type (line 468)
        _matmat_385606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 19), super_call_result_385605, '_matmat')
        # Calling _matmat(args, kwargs) (line 468)
        _matmat_call_result_385609 = invoke(stypy.reporting.localization.Localization(__file__, 468, 19), _matmat_385606, *[X_385607], **kwargs_385608)
        
        # Assigning a type to the variable 'stypy_return_type' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'stypy_return_type', _matmat_call_result_385609)
        # SSA join for if statement (line 465)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_385610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_385610


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 470, 4, False)
        # Assigning a type to the variable 'self' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_function_name', '_CustomLinearOperator._matvec')
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _CustomLinearOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CustomLinearOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        # Call to __matvec_impl(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'x' (line 471)
        x_385613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 34), 'x', False)
        # Processing the call keyword arguments (line 471)
        kwargs_385614 = {}
        # Getting the type of 'self' (line 471)
        self_385611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 15), 'self', False)
        # Obtaining the member '__matvec_impl' of a type (line 471)
        matvec_impl_385612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 15), self_385611, '__matvec_impl')
        # Calling __matvec_impl(args, kwargs) (line 471)
        matvec_impl_call_result_385615 = invoke(stypy.reporting.localization.Localization(__file__, 471, 15), matvec_impl_385612, *[x_385613], **kwargs_385614)
        
        # Assigning a type to the variable 'stypy_return_type' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'stypy_return_type', matvec_impl_call_result_385615)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 470)
        stypy_return_type_385616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385616)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_385616


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 473, 4, False)
        # Assigning a type to the variable 'self' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_function_name', '_CustomLinearOperator._rmatvec')
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _CustomLinearOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CustomLinearOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Assigning a Attribute to a Name (line 474):
        
        # Assigning a Attribute to a Name (line 474):
        # Getting the type of 'self' (line 474)
        self_385617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'self')
        # Obtaining the member '__rmatvec_impl' of a type (line 474)
        rmatvec_impl_385618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 15), self_385617, '__rmatvec_impl')
        # Assigning a type to the variable 'func' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'func', rmatvec_impl_385618)
        
        # Type idiom detected: calculating its left and rigth part (line 475)
        # Getting the type of 'func' (line 475)
        func_385619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 11), 'func')
        # Getting the type of 'None' (line 475)
        None_385620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'None')
        
        (may_be_385621, more_types_in_union_385622) = may_be_none(func_385619, None_385620)

        if may_be_385621:

            if more_types_in_union_385622:
                # Runtime conditional SSA (line 475)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to NotImplementedError(...): (line 476)
            # Processing the call arguments (line 476)
            str_385624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 38), 'str', 'rmatvec is not defined')
            # Processing the call keyword arguments (line 476)
            kwargs_385625 = {}
            # Getting the type of 'NotImplementedError' (line 476)
            NotImplementedError_385623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 18), 'NotImplementedError', False)
            # Calling NotImplementedError(args, kwargs) (line 476)
            NotImplementedError_call_result_385626 = invoke(stypy.reporting.localization.Localization(__file__, 476, 18), NotImplementedError_385623, *[str_385624], **kwargs_385625)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 476, 12), NotImplementedError_call_result_385626, 'raise parameter', BaseException)

            if more_types_in_union_385622:
                # SSA join for if statement (line 475)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to __rmatvec_impl(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'x' (line 477)
        x_385629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 35), 'x', False)
        # Processing the call keyword arguments (line 477)
        kwargs_385630 = {}
        # Getting the type of 'self' (line 477)
        self_385627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'self', False)
        # Obtaining the member '__rmatvec_impl' of a type (line 477)
        rmatvec_impl_385628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 15), self_385627, '__rmatvec_impl')
        # Calling __rmatvec_impl(args, kwargs) (line 477)
        rmatvec_impl_call_result_385631 = invoke(stypy.reporting.localization.Localization(__file__, 477, 15), rmatvec_impl_385628, *[x_385629], **kwargs_385630)
        
        # Assigning a type to the variable 'stypy_return_type' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'stypy_return_type', rmatvec_impl_call_result_385631)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 473)
        stypy_return_type_385632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_385632


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 479, 4, False)
        # Assigning a type to the variable 'self' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_function_name', '_CustomLinearOperator._adjoint')
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _CustomLinearOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_CustomLinearOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        
        # Call to _CustomLinearOperator(...): (line 480)
        # Processing the call keyword arguments (line 480)
        
        # Obtaining an instance of the builtin type 'tuple' (line 480)
        tuple_385634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 480)
        # Adding element type (line 480)
        
        # Obtaining the type of the subscript
        int_385635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 55), 'int')
        # Getting the type of 'self' (line 480)
        self_385636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 44), 'self', False)
        # Obtaining the member 'shape' of a type (line 480)
        shape_385637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 44), self_385636, 'shape')
        # Obtaining the member '__getitem__' of a type (line 480)
        getitem___385638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 44), shape_385637, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 480)
        subscript_call_result_385639 = invoke(stypy.reporting.localization.Localization(__file__, 480, 44), getitem___385638, int_385635)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 44), tuple_385634, subscript_call_result_385639)
        # Adding element type (line 480)
        
        # Obtaining the type of the subscript
        int_385640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 70), 'int')
        # Getting the type of 'self' (line 480)
        self_385641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 59), 'self', False)
        # Obtaining the member 'shape' of a type (line 480)
        shape_385642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 59), self_385641, 'shape')
        # Obtaining the member '__getitem__' of a type (line 480)
        getitem___385643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 59), shape_385642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 480)
        subscript_call_result_385644 = invoke(stypy.reporting.localization.Localization(__file__, 480, 59), getitem___385643, int_385640)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 44), tuple_385634, subscript_call_result_385644)
        
        keyword_385645 = tuple_385634
        # Getting the type of 'self' (line 481)
        self_385646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 44), 'self', False)
        # Obtaining the member '__rmatvec_impl' of a type (line 481)
        rmatvec_impl_385647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 44), self_385646, '__rmatvec_impl')
        keyword_385648 = rmatvec_impl_385647
        # Getting the type of 'self' (line 482)
        self_385649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 45), 'self', False)
        # Obtaining the member '__matvec_impl' of a type (line 482)
        matvec_impl_385650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 45), self_385649, '__matvec_impl')
        keyword_385651 = matvec_impl_385650
        # Getting the type of 'self' (line 483)
        self_385652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 483)
        dtype_385653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 43), self_385652, 'dtype')
        keyword_385654 = dtype_385653
        kwargs_385655 = {'dtype': keyword_385654, 'shape': keyword_385645, 'rmatvec': keyword_385651, 'matvec': keyword_385648}
        # Getting the type of '_CustomLinearOperator' (line 480)
        _CustomLinearOperator_385633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), '_CustomLinearOperator', False)
        # Calling _CustomLinearOperator(args, kwargs) (line 480)
        _CustomLinearOperator_call_result_385656 = invoke(stypy.reporting.localization.Localization(__file__, 480, 15), _CustomLinearOperator_385633, *[], **kwargs_385655)
        
        # Assigning a type to the variable 'stypy_return_type' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'stypy_return_type', _CustomLinearOperator_call_result_385656)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 479)
        stypy_return_type_385657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385657)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_385657


# Assigning a type to the variable '_CustomLinearOperator' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), '_CustomLinearOperator', _CustomLinearOperator)

@norecursion
def _get_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 486)
    None_385658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 33), 'None')
    defaults = [None_385658]
    # Create a new context for function '_get_dtype'
    module_type_store = module_type_store.open_function_context('_get_dtype', 486, 0, False)
    
    # Passed parameters checking function
    _get_dtype.stypy_localization = localization
    _get_dtype.stypy_type_of_self = None
    _get_dtype.stypy_type_store = module_type_store
    _get_dtype.stypy_function_name = '_get_dtype'
    _get_dtype.stypy_param_names_list = ['operators', 'dtypes']
    _get_dtype.stypy_varargs_param_name = None
    _get_dtype.stypy_kwargs_param_name = None
    _get_dtype.stypy_call_defaults = defaults
    _get_dtype.stypy_call_varargs = varargs
    _get_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_dtype', ['operators', 'dtypes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_dtype', localization, ['operators', 'dtypes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_dtype(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 487)
    # Getting the type of 'dtypes' (line 487)
    dtypes_385659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 7), 'dtypes')
    # Getting the type of 'None' (line 487)
    None_385660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 17), 'None')
    
    (may_be_385661, more_types_in_union_385662) = may_be_none(dtypes_385659, None_385660)

    if may_be_385661:

        if more_types_in_union_385662:
            # Runtime conditional SSA (line 487)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 488):
        
        # Assigning a List to a Name (line 488):
        
        # Obtaining an instance of the builtin type 'list' (line 488)
        list_385663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 488)
        
        # Assigning a type to the variable 'dtypes' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'dtypes', list_385663)

        if more_types_in_union_385662:
            # SSA join for if statement (line 487)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'operators' (line 489)
    operators_385664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'operators')
    # Testing the type of a for loop iterable (line 489)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 489, 4), operators_385664)
    # Getting the type of the for loop variable (line 489)
    for_loop_var_385665 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 489, 4), operators_385664)
    # Assigning a type to the variable 'obj' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'obj', for_loop_var_385665)
    # SSA begins for a for statement (line 489)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'obj' (line 490)
    obj_385666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'obj')
    # Getting the type of 'None' (line 490)
    None_385667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 22), 'None')
    # Applying the binary operator 'isnot' (line 490)
    result_is_not_385668 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), 'isnot', obj_385666, None_385667)
    
    
    # Call to hasattr(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'obj' (line 490)
    obj_385670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 39), 'obj', False)
    str_385671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 44), 'str', 'dtype')
    # Processing the call keyword arguments (line 490)
    kwargs_385672 = {}
    # Getting the type of 'hasattr' (line 490)
    hasattr_385669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 31), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 490)
    hasattr_call_result_385673 = invoke(stypy.reporting.localization.Localization(__file__, 490, 31), hasattr_385669, *[obj_385670, str_385671], **kwargs_385672)
    
    # Applying the binary operator 'and' (line 490)
    result_and_keyword_385674 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), 'and', result_is_not_385668, hasattr_call_result_385673)
    
    # Testing the type of an if condition (line 490)
    if_condition_385675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_and_keyword_385674)
    # Assigning a type to the variable 'if_condition_385675' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_385675', if_condition_385675)
    # SSA begins for if statement (line 490)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'obj' (line 491)
    obj_385678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 26), 'obj', False)
    # Obtaining the member 'dtype' of a type (line 491)
    dtype_385679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 26), obj_385678, 'dtype')
    # Processing the call keyword arguments (line 491)
    kwargs_385680 = {}
    # Getting the type of 'dtypes' (line 491)
    dtypes_385676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'dtypes', False)
    # Obtaining the member 'append' of a type (line 491)
    append_385677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 12), dtypes_385676, 'append')
    # Calling append(args, kwargs) (line 491)
    append_call_result_385681 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), append_385677, *[dtype_385679], **kwargs_385680)
    
    # SSA join for if statement (line 490)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to find_common_type(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'dtypes' (line 492)
    dtypes_385684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 31), 'dtypes', False)
    
    # Obtaining an instance of the builtin type 'list' (line 492)
    list_385685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 492)
    
    # Processing the call keyword arguments (line 492)
    kwargs_385686 = {}
    # Getting the type of 'np' (line 492)
    np_385682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 11), 'np', False)
    # Obtaining the member 'find_common_type' of a type (line 492)
    find_common_type_385683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 11), np_385682, 'find_common_type')
    # Calling find_common_type(args, kwargs) (line 492)
    find_common_type_call_result_385687 = invoke(stypy.reporting.localization.Localization(__file__, 492, 11), find_common_type_385683, *[dtypes_385684, list_385685], **kwargs_385686)
    
    # Assigning a type to the variable 'stypy_return_type' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type', find_common_type_call_result_385687)
    
    # ################# End of '_get_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 486)
    stypy_return_type_385688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_385688)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_dtype'
    return stypy_return_type_385688

# Assigning a type to the variable '_get_dtype' (line 486)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 0), '_get_dtype', _get_dtype)
# Declaration of the '_SumLinearOperator' class
# Getting the type of 'LinearOperator' (line 495)
LinearOperator_385689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 25), 'LinearOperator')

class _SumLinearOperator(LinearOperator_385689, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 496, 4, False)
        # Assigning a type to the variable 'self' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SumLinearOperator.__init__', ['A', 'B'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'B'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'A' (line 497)
        A_385691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 26), 'A', False)
        # Getting the type of 'LinearOperator' (line 497)
        LinearOperator_385692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 29), 'LinearOperator', False)
        # Processing the call keyword arguments (line 497)
        kwargs_385693 = {}
        # Getting the type of 'isinstance' (line 497)
        isinstance_385690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 497)
        isinstance_call_result_385694 = invoke(stypy.reporting.localization.Localization(__file__, 497, 15), isinstance_385690, *[A_385691, LinearOperator_385692], **kwargs_385693)
        
        # Applying the 'not' unary operator (line 497)
        result_not__385695 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 11), 'not', isinstance_call_result_385694)
        
        
        
        # Call to isinstance(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'B' (line 498)
        B_385697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'B', False)
        # Getting the type of 'LinearOperator' (line 498)
        LinearOperator_385698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 34), 'LinearOperator', False)
        # Processing the call keyword arguments (line 498)
        kwargs_385699 = {}
        # Getting the type of 'isinstance' (line 498)
        isinstance_385696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 498)
        isinstance_call_result_385700 = invoke(stypy.reporting.localization.Localization(__file__, 498, 20), isinstance_385696, *[B_385697, LinearOperator_385698], **kwargs_385699)
        
        # Applying the 'not' unary operator (line 498)
        result_not__385701 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 16), 'not', isinstance_call_result_385700)
        
        # Applying the binary operator 'or' (line 497)
        result_or_keyword_385702 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 11), 'or', result_not__385695, result_not__385701)
        
        # Testing the type of an if condition (line 497)
        if_condition_385703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 8), result_or_keyword_385702)
        # Assigning a type to the variable 'if_condition_385703' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'if_condition_385703', if_condition_385703)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 499)
        # Processing the call arguments (line 499)
        str_385705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 29), 'str', 'both operands have to be a LinearOperator')
        # Processing the call keyword arguments (line 499)
        kwargs_385706 = {}
        # Getting the type of 'ValueError' (line 499)
        ValueError_385704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 499)
        ValueError_call_result_385707 = invoke(stypy.reporting.localization.Localization(__file__, 499, 18), ValueError_385704, *[str_385705], **kwargs_385706)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 499, 12), ValueError_call_result_385707, 'raise parameter', BaseException)
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'A' (line 500)
        A_385708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 11), 'A')
        # Obtaining the member 'shape' of a type (line 500)
        shape_385709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 11), A_385708, 'shape')
        # Getting the type of 'B' (line 500)
        B_385710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 22), 'B')
        # Obtaining the member 'shape' of a type (line 500)
        shape_385711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 22), B_385710, 'shape')
        # Applying the binary operator '!=' (line 500)
        result_ne_385712 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 11), '!=', shape_385709, shape_385711)
        
        # Testing the type of an if condition (line 500)
        if_condition_385713 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 8), result_ne_385712)
        # Assigning a type to the variable 'if_condition_385713' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'if_condition_385713', if_condition_385713)
        # SSA begins for if statement (line 500)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 501)
        # Processing the call arguments (line 501)
        str_385715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 29), 'str', 'cannot add %r and %r: shape mismatch')
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_385716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        # Getting the type of 'A' (line 502)
        A_385717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), tuple_385716, A_385717)
        # Adding element type (line 502)
        # Getting the type of 'B' (line 502)
        B_385718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 35), 'B', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 32), tuple_385716, B_385718)
        
        # Applying the binary operator '%' (line 501)
        result_mod_385719 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 29), '%', str_385715, tuple_385716)
        
        # Processing the call keyword arguments (line 501)
        kwargs_385720 = {}
        # Getting the type of 'ValueError' (line 501)
        ValueError_385714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 501)
        ValueError_call_result_385721 = invoke(stypy.reporting.localization.Localization(__file__, 501, 18), ValueError_385714, *[result_mod_385719], **kwargs_385720)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 501, 12), ValueError_call_result_385721, 'raise parameter', BaseException)
        # SSA join for if statement (line 500)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Attribute (line 503):
        
        # Assigning a Tuple to a Attribute (line 503):
        
        # Obtaining an instance of the builtin type 'tuple' (line 503)
        tuple_385722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 503)
        # Adding element type (line 503)
        # Getting the type of 'A' (line 503)
        A_385723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 21), 'A')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), tuple_385722, A_385723)
        # Adding element type (line 503)
        # Getting the type of 'B' (line 503)
        B_385724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), tuple_385722, B_385724)
        
        # Getting the type of 'self' (line 503)
        self_385725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'self')
        # Setting the type of the member 'args' of a type (line 503)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), self_385725, 'args', tuple_385722)
        
        # Call to __init__(...): (line 504)
        # Processing the call arguments (line 504)
        
        # Call to _get_dtype(...): (line 504)
        # Processing the call arguments (line 504)
        
        # Obtaining an instance of the builtin type 'list' (line 504)
        list_385733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 504)
        # Adding element type (line 504)
        # Getting the type of 'A' (line 504)
        A_385734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 61), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 60), list_385733, A_385734)
        # Adding element type (line 504)
        # Getting the type of 'B' (line 504)
        B_385735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 64), 'B', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 60), list_385733, B_385735)
        
        # Processing the call keyword arguments (line 504)
        kwargs_385736 = {}
        # Getting the type of '_get_dtype' (line 504)
        _get_dtype_385732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 49), '_get_dtype', False)
        # Calling _get_dtype(args, kwargs) (line 504)
        _get_dtype_call_result_385737 = invoke(stypy.reporting.localization.Localization(__file__, 504, 49), _get_dtype_385732, *[list_385733], **kwargs_385736)
        
        # Getting the type of 'A' (line 504)
        A_385738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 69), 'A', False)
        # Obtaining the member 'shape' of a type (line 504)
        shape_385739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 69), A_385738, 'shape')
        # Processing the call keyword arguments (line 504)
        kwargs_385740 = {}
        
        # Call to super(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of '_SumLinearOperator' (line 504)
        _SumLinearOperator_385727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 14), '_SumLinearOperator', False)
        # Getting the type of 'self' (line 504)
        self_385728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 34), 'self', False)
        # Processing the call keyword arguments (line 504)
        kwargs_385729 = {}
        # Getting the type of 'super' (line 504)
        super_385726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'super', False)
        # Calling super(args, kwargs) (line 504)
        super_call_result_385730 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), super_385726, *[_SumLinearOperator_385727, self_385728], **kwargs_385729)
        
        # Obtaining the member '__init__' of a type (line 504)
        init___385731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 8), super_call_result_385730, '__init__')
        # Calling __init__(args, kwargs) (line 504)
        init___call_result_385741 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), init___385731, *[_get_dtype_call_result_385737, shape_385739], **kwargs_385740)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 506, 4, False)
        # Assigning a type to the variable 'self' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_function_name', '_SumLinearOperator._matvec')
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SumLinearOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SumLinearOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        # Call to matvec(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'x' (line 507)
        x_385748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 35), 'x', False)
        # Processing the call keyword arguments (line 507)
        kwargs_385749 = {}
        
        # Obtaining the type of the subscript
        int_385742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 25), 'int')
        # Getting the type of 'self' (line 507)
        self_385743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 15), 'self', False)
        # Obtaining the member 'args' of a type (line 507)
        args_385744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 15), self_385743, 'args')
        # Obtaining the member '__getitem__' of a type (line 507)
        getitem___385745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 15), args_385744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 507)
        subscript_call_result_385746 = invoke(stypy.reporting.localization.Localization(__file__, 507, 15), getitem___385745, int_385742)
        
        # Obtaining the member 'matvec' of a type (line 507)
        matvec_385747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 15), subscript_call_result_385746, 'matvec')
        # Calling matvec(args, kwargs) (line 507)
        matvec_call_result_385750 = invoke(stypy.reporting.localization.Localization(__file__, 507, 15), matvec_385747, *[x_385748], **kwargs_385749)
        
        
        # Call to matvec(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'x' (line 507)
        x_385757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 60), 'x', False)
        # Processing the call keyword arguments (line 507)
        kwargs_385758 = {}
        
        # Obtaining the type of the subscript
        int_385751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 50), 'int')
        # Getting the type of 'self' (line 507)
        self_385752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 40), 'self', False)
        # Obtaining the member 'args' of a type (line 507)
        args_385753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 40), self_385752, 'args')
        # Obtaining the member '__getitem__' of a type (line 507)
        getitem___385754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 40), args_385753, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 507)
        subscript_call_result_385755 = invoke(stypy.reporting.localization.Localization(__file__, 507, 40), getitem___385754, int_385751)
        
        # Obtaining the member 'matvec' of a type (line 507)
        matvec_385756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 40), subscript_call_result_385755, 'matvec')
        # Calling matvec(args, kwargs) (line 507)
        matvec_call_result_385759 = invoke(stypy.reporting.localization.Localization(__file__, 507, 40), matvec_385756, *[x_385757], **kwargs_385758)
        
        # Applying the binary operator '+' (line 507)
        result_add_385760 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 15), '+', matvec_call_result_385750, matvec_call_result_385759)
        
        # Assigning a type to the variable 'stypy_return_type' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'stypy_return_type', result_add_385760)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_385761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385761)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_385761


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 509, 4, False)
        # Assigning a type to the variable 'self' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_function_name', '_SumLinearOperator._rmatvec')
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SumLinearOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SumLinearOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Call to rmatvec(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'x' (line 510)
        x_385768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 36), 'x', False)
        # Processing the call keyword arguments (line 510)
        kwargs_385769 = {}
        
        # Obtaining the type of the subscript
        int_385762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 25), 'int')
        # Getting the type of 'self' (line 510)
        self_385763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), 'self', False)
        # Obtaining the member 'args' of a type (line 510)
        args_385764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 15), self_385763, 'args')
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___385765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 15), args_385764, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_385766 = invoke(stypy.reporting.localization.Localization(__file__, 510, 15), getitem___385765, int_385762)
        
        # Obtaining the member 'rmatvec' of a type (line 510)
        rmatvec_385767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 15), subscript_call_result_385766, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 510)
        rmatvec_call_result_385770 = invoke(stypy.reporting.localization.Localization(__file__, 510, 15), rmatvec_385767, *[x_385768], **kwargs_385769)
        
        
        # Call to rmatvec(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'x' (line 510)
        x_385777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 62), 'x', False)
        # Processing the call keyword arguments (line 510)
        kwargs_385778 = {}
        
        # Obtaining the type of the subscript
        int_385771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 51), 'int')
        # Getting the type of 'self' (line 510)
        self_385772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 41), 'self', False)
        # Obtaining the member 'args' of a type (line 510)
        args_385773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 41), self_385772, 'args')
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___385774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 41), args_385773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_385775 = invoke(stypy.reporting.localization.Localization(__file__, 510, 41), getitem___385774, int_385771)
        
        # Obtaining the member 'rmatvec' of a type (line 510)
        rmatvec_385776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 41), subscript_call_result_385775, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 510)
        rmatvec_call_result_385779 = invoke(stypy.reporting.localization.Localization(__file__, 510, 41), rmatvec_385776, *[x_385777], **kwargs_385778)
        
        # Applying the binary operator '+' (line 510)
        result_add_385780 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 15), '+', rmatvec_call_result_385770, rmatvec_call_result_385779)
        
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', result_add_385780)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 509)
        stypy_return_type_385781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_385781


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 512, 4, False)
        # Assigning a type to the variable 'self' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_function_name', '_SumLinearOperator._matmat')
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SumLinearOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SumLinearOperator._matmat', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        # Call to matmat(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'x' (line 513)
        x_385788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 35), 'x', False)
        # Processing the call keyword arguments (line 513)
        kwargs_385789 = {}
        
        # Obtaining the type of the subscript
        int_385782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 25), 'int')
        # Getting the type of 'self' (line 513)
        self_385783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'self', False)
        # Obtaining the member 'args' of a type (line 513)
        args_385784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 15), self_385783, 'args')
        # Obtaining the member '__getitem__' of a type (line 513)
        getitem___385785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 15), args_385784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 513)
        subscript_call_result_385786 = invoke(stypy.reporting.localization.Localization(__file__, 513, 15), getitem___385785, int_385782)
        
        # Obtaining the member 'matmat' of a type (line 513)
        matmat_385787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 15), subscript_call_result_385786, 'matmat')
        # Calling matmat(args, kwargs) (line 513)
        matmat_call_result_385790 = invoke(stypy.reporting.localization.Localization(__file__, 513, 15), matmat_385787, *[x_385788], **kwargs_385789)
        
        
        # Call to matmat(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'x' (line 513)
        x_385797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 60), 'x', False)
        # Processing the call keyword arguments (line 513)
        kwargs_385798 = {}
        
        # Obtaining the type of the subscript
        int_385791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 50), 'int')
        # Getting the type of 'self' (line 513)
        self_385792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 40), 'self', False)
        # Obtaining the member 'args' of a type (line 513)
        args_385793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 40), self_385792, 'args')
        # Obtaining the member '__getitem__' of a type (line 513)
        getitem___385794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 40), args_385793, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 513)
        subscript_call_result_385795 = invoke(stypy.reporting.localization.Localization(__file__, 513, 40), getitem___385794, int_385791)
        
        # Obtaining the member 'matmat' of a type (line 513)
        matmat_385796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 40), subscript_call_result_385795, 'matmat')
        # Calling matmat(args, kwargs) (line 513)
        matmat_call_result_385799 = invoke(stypy.reporting.localization.Localization(__file__, 513, 40), matmat_385796, *[x_385797], **kwargs_385798)
        
        # Applying the binary operator '+' (line 513)
        result_add_385800 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 15), '+', matmat_call_result_385790, matmat_call_result_385799)
        
        # Assigning a type to the variable 'stypy_return_type' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'stypy_return_type', result_add_385800)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 512)
        stypy_return_type_385801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_385801


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_function_name', '_SumLinearOperator._adjoint')
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _SumLinearOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_SumLinearOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 516):
        
        # Assigning a Subscript to a Name (line 516):
        
        # Obtaining the type of the subscript
        int_385802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 8), 'int')
        # Getting the type of 'self' (line 516)
        self_385803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'self')
        # Obtaining the member 'args' of a type (line 516)
        args_385804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 15), self_385803, 'args')
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___385805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), args_385804, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_385806 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), getitem___385805, int_385802)
        
        # Assigning a type to the variable 'tuple_var_assignment_384913' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_384913', subscript_call_result_385806)
        
        # Assigning a Subscript to a Name (line 516):
        
        # Obtaining the type of the subscript
        int_385807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 8), 'int')
        # Getting the type of 'self' (line 516)
        self_385808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'self')
        # Obtaining the member 'args' of a type (line 516)
        args_385809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 15), self_385808, 'args')
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___385810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), args_385809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_385811 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), getitem___385810, int_385807)
        
        # Assigning a type to the variable 'tuple_var_assignment_384914' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_384914', subscript_call_result_385811)
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'tuple_var_assignment_384913' (line 516)
        tuple_var_assignment_384913_385812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_384913')
        # Assigning a type to the variable 'A' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'A', tuple_var_assignment_384913_385812)
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'tuple_var_assignment_384914' (line 516)
        tuple_var_assignment_384914_385813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_384914')
        # Assigning a type to the variable 'B' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'B', tuple_var_assignment_384914_385813)
        # Getting the type of 'A' (line 517)
        A_385814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 15), 'A')
        # Obtaining the member 'H' of a type (line 517)
        H_385815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 15), A_385814, 'H')
        # Getting the type of 'B' (line 517)
        B_385816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 21), 'B')
        # Obtaining the member 'H' of a type (line 517)
        H_385817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 21), B_385816, 'H')
        # Applying the binary operator '+' (line 517)
        result_add_385818 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 15), '+', H_385815, H_385817)
        
        # Assigning a type to the variable 'stypy_return_type' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'stypy_return_type', result_add_385818)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_385819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385819)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_385819


# Assigning a type to the variable '_SumLinearOperator' (line 495)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 0), '_SumLinearOperator', _SumLinearOperator)
# Declaration of the '_ProductLinearOperator' class
# Getting the type of 'LinearOperator' (line 520)
LinearOperator_385820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 29), 'LinearOperator')

class _ProductLinearOperator(LinearOperator_385820, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 521, 4, False)
        # Assigning a type to the variable 'self' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ProductLinearOperator.__init__', ['A', 'B'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'B'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'A' (line 522)
        A_385822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 26), 'A', False)
        # Getting the type of 'LinearOperator' (line 522)
        LinearOperator_385823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 29), 'LinearOperator', False)
        # Processing the call keyword arguments (line 522)
        kwargs_385824 = {}
        # Getting the type of 'isinstance' (line 522)
        isinstance_385821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 522)
        isinstance_call_result_385825 = invoke(stypy.reporting.localization.Localization(__file__, 522, 15), isinstance_385821, *[A_385822, LinearOperator_385823], **kwargs_385824)
        
        # Applying the 'not' unary operator (line 522)
        result_not__385826 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 11), 'not', isinstance_call_result_385825)
        
        
        
        # Call to isinstance(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'B' (line 523)
        B_385828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 31), 'B', False)
        # Getting the type of 'LinearOperator' (line 523)
        LinearOperator_385829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 34), 'LinearOperator', False)
        # Processing the call keyword arguments (line 523)
        kwargs_385830 = {}
        # Getting the type of 'isinstance' (line 523)
        isinstance_385827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 523)
        isinstance_call_result_385831 = invoke(stypy.reporting.localization.Localization(__file__, 523, 20), isinstance_385827, *[B_385828, LinearOperator_385829], **kwargs_385830)
        
        # Applying the 'not' unary operator (line 523)
        result_not__385832 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 16), 'not', isinstance_call_result_385831)
        
        # Applying the binary operator 'or' (line 522)
        result_or_keyword_385833 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 11), 'or', result_not__385826, result_not__385832)
        
        # Testing the type of an if condition (line 522)
        if_condition_385834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 8), result_or_keyword_385833)
        # Assigning a type to the variable 'if_condition_385834' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'if_condition_385834', if_condition_385834)
        # SSA begins for if statement (line 522)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 524)
        # Processing the call arguments (line 524)
        str_385836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 29), 'str', 'both operands have to be a LinearOperator')
        # Processing the call keyword arguments (line 524)
        kwargs_385837 = {}
        # Getting the type of 'ValueError' (line 524)
        ValueError_385835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 524)
        ValueError_call_result_385838 = invoke(stypy.reporting.localization.Localization(__file__, 524, 18), ValueError_385835, *[str_385836], **kwargs_385837)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 524, 12), ValueError_call_result_385838, 'raise parameter', BaseException)
        # SSA join for if statement (line 522)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_385839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 19), 'int')
        # Getting the type of 'A' (line 525)
        A_385840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 11), 'A')
        # Obtaining the member 'shape' of a type (line 525)
        shape_385841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 11), A_385840, 'shape')
        # Obtaining the member '__getitem__' of a type (line 525)
        getitem___385842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 11), shape_385841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 525)
        subscript_call_result_385843 = invoke(stypy.reporting.localization.Localization(__file__, 525, 11), getitem___385842, int_385839)
        
        
        # Obtaining the type of the subscript
        int_385844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 33), 'int')
        # Getting the type of 'B' (line 525)
        B_385845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'B')
        # Obtaining the member 'shape' of a type (line 525)
        shape_385846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 25), B_385845, 'shape')
        # Obtaining the member '__getitem__' of a type (line 525)
        getitem___385847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 25), shape_385846, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 525)
        subscript_call_result_385848 = invoke(stypy.reporting.localization.Localization(__file__, 525, 25), getitem___385847, int_385844)
        
        # Applying the binary operator '!=' (line 525)
        result_ne_385849 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 11), '!=', subscript_call_result_385843, subscript_call_result_385848)
        
        # Testing the type of an if condition (line 525)
        if_condition_385850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 8), result_ne_385849)
        # Assigning a type to the variable 'if_condition_385850' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'if_condition_385850', if_condition_385850)
        # SSA begins for if statement (line 525)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 526)
        # Processing the call arguments (line 526)
        str_385852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 29), 'str', 'cannot multiply %r and %r: shape mismatch')
        
        # Obtaining an instance of the builtin type 'tuple' (line 527)
        tuple_385853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 527)
        # Adding element type (line 527)
        # Getting the type of 'A' (line 527)
        A_385854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 32), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 32), tuple_385853, A_385854)
        # Adding element type (line 527)
        # Getting the type of 'B' (line 527)
        B_385855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 35), 'B', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 32), tuple_385853, B_385855)
        
        # Applying the binary operator '%' (line 526)
        result_mod_385856 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 29), '%', str_385852, tuple_385853)
        
        # Processing the call keyword arguments (line 526)
        kwargs_385857 = {}
        # Getting the type of 'ValueError' (line 526)
        ValueError_385851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 526)
        ValueError_call_result_385858 = invoke(stypy.reporting.localization.Localization(__file__, 526, 18), ValueError_385851, *[result_mod_385856], **kwargs_385857)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 526, 12), ValueError_call_result_385858, 'raise parameter', BaseException)
        # SSA join for if statement (line 525)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __init__(...): (line 528)
        # Processing the call arguments (line 528)
        
        # Call to _get_dtype(...): (line 528)
        # Processing the call arguments (line 528)
        
        # Obtaining an instance of the builtin type 'list' (line 528)
        list_385866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 528)
        # Adding element type (line 528)
        # Getting the type of 'A' (line 528)
        A_385867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 65), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 64), list_385866, A_385867)
        # Adding element type (line 528)
        # Getting the type of 'B' (line 528)
        B_385868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 68), 'B', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 64), list_385866, B_385868)
        
        # Processing the call keyword arguments (line 528)
        kwargs_385869 = {}
        # Getting the type of '_get_dtype' (line 528)
        _get_dtype_385865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 53), '_get_dtype', False)
        # Calling _get_dtype(args, kwargs) (line 528)
        _get_dtype_call_result_385870 = invoke(stypy.reporting.localization.Localization(__file__, 528, 53), _get_dtype_385865, *[list_385866], **kwargs_385869)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 529)
        tuple_385871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 529)
        # Adding element type (line 529)
        
        # Obtaining the type of the subscript
        int_385872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 62), 'int')
        # Getting the type of 'A' (line 529)
        A_385873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 54), 'A', False)
        # Obtaining the member 'shape' of a type (line 529)
        shape_385874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 54), A_385873, 'shape')
        # Obtaining the member '__getitem__' of a type (line 529)
        getitem___385875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 54), shape_385874, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 529)
        subscript_call_result_385876 = invoke(stypy.reporting.localization.Localization(__file__, 529, 54), getitem___385875, int_385872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 54), tuple_385871, subscript_call_result_385876)
        # Adding element type (line 529)
        
        # Obtaining the type of the subscript
        int_385877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 74), 'int')
        # Getting the type of 'B' (line 529)
        B_385878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 66), 'B', False)
        # Obtaining the member 'shape' of a type (line 529)
        shape_385879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 66), B_385878, 'shape')
        # Obtaining the member '__getitem__' of a type (line 529)
        getitem___385880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 66), shape_385879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 529)
        subscript_call_result_385881 = invoke(stypy.reporting.localization.Localization(__file__, 529, 66), getitem___385880, int_385877)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 54), tuple_385871, subscript_call_result_385881)
        
        # Processing the call keyword arguments (line 528)
        kwargs_385882 = {}
        
        # Call to super(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of '_ProductLinearOperator' (line 528)
        _ProductLinearOperator_385860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 14), '_ProductLinearOperator', False)
        # Getting the type of 'self' (line 528)
        self_385861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 38), 'self', False)
        # Processing the call keyword arguments (line 528)
        kwargs_385862 = {}
        # Getting the type of 'super' (line 528)
        super_385859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'super', False)
        # Calling super(args, kwargs) (line 528)
        super_call_result_385863 = invoke(stypy.reporting.localization.Localization(__file__, 528, 8), super_385859, *[_ProductLinearOperator_385860, self_385861], **kwargs_385862)
        
        # Obtaining the member '__init__' of a type (line 528)
        init___385864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 8), super_call_result_385863, '__init__')
        # Calling __init__(args, kwargs) (line 528)
        init___call_result_385883 = invoke(stypy.reporting.localization.Localization(__file__, 528, 8), init___385864, *[_get_dtype_call_result_385870, tuple_385871], **kwargs_385882)
        
        
        # Assigning a Tuple to a Attribute (line 530):
        
        # Assigning a Tuple to a Attribute (line 530):
        
        # Obtaining an instance of the builtin type 'tuple' (line 530)
        tuple_385884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 530)
        # Adding element type (line 530)
        # Getting the type of 'A' (line 530)
        A_385885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 21), 'A')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 21), tuple_385884, A_385885)
        # Adding element type (line 530)
        # Getting the type of 'B' (line 530)
        B_385886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 24), 'B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 21), tuple_385884, B_385886)
        
        # Getting the type of 'self' (line 530)
        self_385887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'self')
        # Setting the type of the member 'args' of a type (line 530)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 8), self_385887, 'args', tuple_385884)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 532, 4, False)
        # Assigning a type to the variable 'self' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_function_name', '_ProductLinearOperator._matvec')
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ProductLinearOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ProductLinearOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        # Call to matvec(...): (line 533)
        # Processing the call arguments (line 533)
        
        # Call to matvec(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'x' (line 533)
        x_385900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 55), 'x', False)
        # Processing the call keyword arguments (line 533)
        kwargs_385901 = {}
        
        # Obtaining the type of the subscript
        int_385894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 45), 'int')
        # Getting the type of 'self' (line 533)
        self_385895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 35), 'self', False)
        # Obtaining the member 'args' of a type (line 533)
        args_385896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 35), self_385895, 'args')
        # Obtaining the member '__getitem__' of a type (line 533)
        getitem___385897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 35), args_385896, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 533)
        subscript_call_result_385898 = invoke(stypy.reporting.localization.Localization(__file__, 533, 35), getitem___385897, int_385894)
        
        # Obtaining the member 'matvec' of a type (line 533)
        matvec_385899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 35), subscript_call_result_385898, 'matvec')
        # Calling matvec(args, kwargs) (line 533)
        matvec_call_result_385902 = invoke(stypy.reporting.localization.Localization(__file__, 533, 35), matvec_385899, *[x_385900], **kwargs_385901)
        
        # Processing the call keyword arguments (line 533)
        kwargs_385903 = {}
        
        # Obtaining the type of the subscript
        int_385888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 25), 'int')
        # Getting the type of 'self' (line 533)
        self_385889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'self', False)
        # Obtaining the member 'args' of a type (line 533)
        args_385890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 15), self_385889, 'args')
        # Obtaining the member '__getitem__' of a type (line 533)
        getitem___385891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 15), args_385890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 533)
        subscript_call_result_385892 = invoke(stypy.reporting.localization.Localization(__file__, 533, 15), getitem___385891, int_385888)
        
        # Obtaining the member 'matvec' of a type (line 533)
        matvec_385893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 15), subscript_call_result_385892, 'matvec')
        # Calling matvec(args, kwargs) (line 533)
        matvec_call_result_385904 = invoke(stypy.reporting.localization.Localization(__file__, 533, 15), matvec_385893, *[matvec_call_result_385902], **kwargs_385903)
        
        # Assigning a type to the variable 'stypy_return_type' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'stypy_return_type', matvec_call_result_385904)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 532)
        stypy_return_type_385905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_385905


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 535, 4, False)
        # Assigning a type to the variable 'self' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_function_name', '_ProductLinearOperator._rmatvec')
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ProductLinearOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ProductLinearOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Call to rmatvec(...): (line 536)
        # Processing the call arguments (line 536)
        
        # Call to rmatvec(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'x' (line 536)
        x_385918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 57), 'x', False)
        # Processing the call keyword arguments (line 536)
        kwargs_385919 = {}
        
        # Obtaining the type of the subscript
        int_385912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 46), 'int')
        # Getting the type of 'self' (line 536)
        self_385913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 36), 'self', False)
        # Obtaining the member 'args' of a type (line 536)
        args_385914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 36), self_385913, 'args')
        # Obtaining the member '__getitem__' of a type (line 536)
        getitem___385915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 36), args_385914, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 536)
        subscript_call_result_385916 = invoke(stypy.reporting.localization.Localization(__file__, 536, 36), getitem___385915, int_385912)
        
        # Obtaining the member 'rmatvec' of a type (line 536)
        rmatvec_385917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 36), subscript_call_result_385916, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 536)
        rmatvec_call_result_385920 = invoke(stypy.reporting.localization.Localization(__file__, 536, 36), rmatvec_385917, *[x_385918], **kwargs_385919)
        
        # Processing the call keyword arguments (line 536)
        kwargs_385921 = {}
        
        # Obtaining the type of the subscript
        int_385906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 25), 'int')
        # Getting the type of 'self' (line 536)
        self_385907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 15), 'self', False)
        # Obtaining the member 'args' of a type (line 536)
        args_385908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 15), self_385907, 'args')
        # Obtaining the member '__getitem__' of a type (line 536)
        getitem___385909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 15), args_385908, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 536)
        subscript_call_result_385910 = invoke(stypy.reporting.localization.Localization(__file__, 536, 15), getitem___385909, int_385906)
        
        # Obtaining the member 'rmatvec' of a type (line 536)
        rmatvec_385911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 15), subscript_call_result_385910, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 536)
        rmatvec_call_result_385922 = invoke(stypy.reporting.localization.Localization(__file__, 536, 15), rmatvec_385911, *[rmatvec_call_result_385920], **kwargs_385921)
        
        # Assigning a type to the variable 'stypy_return_type' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'stypy_return_type', rmatvec_call_result_385922)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_385923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385923)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_385923


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 538, 4, False)
        # Assigning a type to the variable 'self' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_function_name', '_ProductLinearOperator._matmat')
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ProductLinearOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ProductLinearOperator._matmat', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        # Call to matmat(...): (line 539)
        # Processing the call arguments (line 539)
        
        # Call to matmat(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'x' (line 539)
        x_385936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 55), 'x', False)
        # Processing the call keyword arguments (line 539)
        kwargs_385937 = {}
        
        # Obtaining the type of the subscript
        int_385930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 45), 'int')
        # Getting the type of 'self' (line 539)
        self_385931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 35), 'self', False)
        # Obtaining the member 'args' of a type (line 539)
        args_385932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 35), self_385931, 'args')
        # Obtaining the member '__getitem__' of a type (line 539)
        getitem___385933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 35), args_385932, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 539)
        subscript_call_result_385934 = invoke(stypy.reporting.localization.Localization(__file__, 539, 35), getitem___385933, int_385930)
        
        # Obtaining the member 'matmat' of a type (line 539)
        matmat_385935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 35), subscript_call_result_385934, 'matmat')
        # Calling matmat(args, kwargs) (line 539)
        matmat_call_result_385938 = invoke(stypy.reporting.localization.Localization(__file__, 539, 35), matmat_385935, *[x_385936], **kwargs_385937)
        
        # Processing the call keyword arguments (line 539)
        kwargs_385939 = {}
        
        # Obtaining the type of the subscript
        int_385924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 25), 'int')
        # Getting the type of 'self' (line 539)
        self_385925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'self', False)
        # Obtaining the member 'args' of a type (line 539)
        args_385926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 15), self_385925, 'args')
        # Obtaining the member '__getitem__' of a type (line 539)
        getitem___385927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 15), args_385926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 539)
        subscript_call_result_385928 = invoke(stypy.reporting.localization.Localization(__file__, 539, 15), getitem___385927, int_385924)
        
        # Obtaining the member 'matmat' of a type (line 539)
        matmat_385929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 15), subscript_call_result_385928, 'matmat')
        # Calling matmat(args, kwargs) (line 539)
        matmat_call_result_385940 = invoke(stypy.reporting.localization.Localization(__file__, 539, 15), matmat_385929, *[matmat_call_result_385938], **kwargs_385939)
        
        # Assigning a type to the variable 'stypy_return_type' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'stypy_return_type', matmat_call_result_385940)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 538)
        stypy_return_type_385941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385941)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_385941


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 541, 4, False)
        # Assigning a type to the variable 'self' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_function_name', '_ProductLinearOperator._adjoint')
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ProductLinearOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ProductLinearOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 542):
        
        # Assigning a Subscript to a Name (line 542):
        
        # Obtaining the type of the subscript
        int_385942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 8), 'int')
        # Getting the type of 'self' (line 542)
        self_385943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 15), 'self')
        # Obtaining the member 'args' of a type (line 542)
        args_385944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 15), self_385943, 'args')
        # Obtaining the member '__getitem__' of a type (line 542)
        getitem___385945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 8), args_385944, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 542)
        subscript_call_result_385946 = invoke(stypy.reporting.localization.Localization(__file__, 542, 8), getitem___385945, int_385942)
        
        # Assigning a type to the variable 'tuple_var_assignment_384915' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'tuple_var_assignment_384915', subscript_call_result_385946)
        
        # Assigning a Subscript to a Name (line 542):
        
        # Obtaining the type of the subscript
        int_385947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 8), 'int')
        # Getting the type of 'self' (line 542)
        self_385948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 15), 'self')
        # Obtaining the member 'args' of a type (line 542)
        args_385949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 15), self_385948, 'args')
        # Obtaining the member '__getitem__' of a type (line 542)
        getitem___385950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 8), args_385949, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 542)
        subscript_call_result_385951 = invoke(stypy.reporting.localization.Localization(__file__, 542, 8), getitem___385950, int_385947)
        
        # Assigning a type to the variable 'tuple_var_assignment_384916' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'tuple_var_assignment_384916', subscript_call_result_385951)
        
        # Assigning a Name to a Name (line 542):
        # Getting the type of 'tuple_var_assignment_384915' (line 542)
        tuple_var_assignment_384915_385952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'tuple_var_assignment_384915')
        # Assigning a type to the variable 'A' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'A', tuple_var_assignment_384915_385952)
        
        # Assigning a Name to a Name (line 542):
        # Getting the type of 'tuple_var_assignment_384916' (line 542)
        tuple_var_assignment_384916_385953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'tuple_var_assignment_384916')
        # Assigning a type to the variable 'B' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 11), 'B', tuple_var_assignment_384916_385953)
        # Getting the type of 'B' (line 543)
        B_385954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 15), 'B')
        # Obtaining the member 'H' of a type (line 543)
        H_385955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 15), B_385954, 'H')
        # Getting the type of 'A' (line 543)
        A_385956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 21), 'A')
        # Obtaining the member 'H' of a type (line 543)
        H_385957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 21), A_385956, 'H')
        # Applying the binary operator '*' (line 543)
        result_mul_385958 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 15), '*', H_385955, H_385957)
        
        # Assigning a type to the variable 'stypy_return_type' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'stypy_return_type', result_mul_385958)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 541)
        stypy_return_type_385959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_385959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_385959


# Assigning a type to the variable '_ProductLinearOperator' (line 520)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 0), '_ProductLinearOperator', _ProductLinearOperator)
# Declaration of the '_ScaledLinearOperator' class
# Getting the type of 'LinearOperator' (line 546)
LinearOperator_385960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 28), 'LinearOperator')

class _ScaledLinearOperator(LinearOperator_385960, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 547, 4, False)
        # Assigning a type to the variable 'self' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ScaledLinearOperator.__init__', ['A', 'alpha'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'alpha'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'A' (line 548)
        A_385962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 26), 'A', False)
        # Getting the type of 'LinearOperator' (line 548)
        LinearOperator_385963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 29), 'LinearOperator', False)
        # Processing the call keyword arguments (line 548)
        kwargs_385964 = {}
        # Getting the type of 'isinstance' (line 548)
        isinstance_385961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 548)
        isinstance_call_result_385965 = invoke(stypy.reporting.localization.Localization(__file__, 548, 15), isinstance_385961, *[A_385962, LinearOperator_385963], **kwargs_385964)
        
        # Applying the 'not' unary operator (line 548)
        result_not__385966 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 11), 'not', isinstance_call_result_385965)
        
        # Testing the type of an if condition (line 548)
        if_condition_385967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 8), result_not__385966)
        # Assigning a type to the variable 'if_condition_385967' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'if_condition_385967', if_condition_385967)
        # SSA begins for if statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 549)
        # Processing the call arguments (line 549)
        str_385969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 29), 'str', 'LinearOperator expected as A')
        # Processing the call keyword arguments (line 549)
        kwargs_385970 = {}
        # Getting the type of 'ValueError' (line 549)
        ValueError_385968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 549)
        ValueError_call_result_385971 = invoke(stypy.reporting.localization.Localization(__file__, 549, 18), ValueError_385968, *[str_385969], **kwargs_385970)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 549, 12), ValueError_call_result_385971, 'raise parameter', BaseException)
        # SSA join for if statement (line 548)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isscalar(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'alpha' (line 550)
        alpha_385974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 27), 'alpha', False)
        # Processing the call keyword arguments (line 550)
        kwargs_385975 = {}
        # Getting the type of 'np' (line 550)
        np_385972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'np', False)
        # Obtaining the member 'isscalar' of a type (line 550)
        isscalar_385973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 15), np_385972, 'isscalar')
        # Calling isscalar(args, kwargs) (line 550)
        isscalar_call_result_385976 = invoke(stypy.reporting.localization.Localization(__file__, 550, 15), isscalar_385973, *[alpha_385974], **kwargs_385975)
        
        # Applying the 'not' unary operator (line 550)
        result_not__385977 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 11), 'not', isscalar_call_result_385976)
        
        # Testing the type of an if condition (line 550)
        if_condition_385978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 8), result_not__385977)
        # Assigning a type to the variable 'if_condition_385978' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'if_condition_385978', if_condition_385978)
        # SSA begins for if statement (line 550)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 551)
        # Processing the call arguments (line 551)
        str_385980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 29), 'str', 'scalar expected as alpha')
        # Processing the call keyword arguments (line 551)
        kwargs_385981 = {}
        # Getting the type of 'ValueError' (line 551)
        ValueError_385979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 551)
        ValueError_call_result_385982 = invoke(stypy.reporting.localization.Localization(__file__, 551, 18), ValueError_385979, *[str_385980], **kwargs_385981)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 551, 12), ValueError_call_result_385982, 'raise parameter', BaseException)
        # SSA join for if statement (line 550)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 552):
        
        # Assigning a Call to a Name (line 552):
        
        # Call to _get_dtype(...): (line 552)
        # Processing the call arguments (line 552)
        
        # Obtaining an instance of the builtin type 'list' (line 552)
        list_385984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 552)
        # Adding element type (line 552)
        # Getting the type of 'A' (line 552)
        A_385985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 28), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 27), list_385984, A_385985)
        
        
        # Obtaining an instance of the builtin type 'list' (line 552)
        list_385986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 552)
        # Adding element type (line 552)
        
        # Call to type(...): (line 552)
        # Processing the call arguments (line 552)
        # Getting the type of 'alpha' (line 552)
        alpha_385988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 38), 'alpha', False)
        # Processing the call keyword arguments (line 552)
        kwargs_385989 = {}
        # Getting the type of 'type' (line 552)
        type_385987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 33), 'type', False)
        # Calling type(args, kwargs) (line 552)
        type_call_result_385990 = invoke(stypy.reporting.localization.Localization(__file__, 552, 33), type_385987, *[alpha_385988], **kwargs_385989)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 32), list_385986, type_call_result_385990)
        
        # Processing the call keyword arguments (line 552)
        kwargs_385991 = {}
        # Getting the type of '_get_dtype' (line 552)
        _get_dtype_385983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), '_get_dtype', False)
        # Calling _get_dtype(args, kwargs) (line 552)
        _get_dtype_call_result_385992 = invoke(stypy.reporting.localization.Localization(__file__, 552, 16), _get_dtype_385983, *[list_385984, list_385986], **kwargs_385991)
        
        # Assigning a type to the variable 'dtype' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'dtype', _get_dtype_call_result_385992)
        
        # Call to __init__(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'dtype' (line 553)
        dtype_385999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 52), 'dtype', False)
        # Getting the type of 'A' (line 553)
        A_386000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 59), 'A', False)
        # Obtaining the member 'shape' of a type (line 553)
        shape_386001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 59), A_386000, 'shape')
        # Processing the call keyword arguments (line 553)
        kwargs_386002 = {}
        
        # Call to super(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of '_ScaledLinearOperator' (line 553)
        _ScaledLinearOperator_385994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 14), '_ScaledLinearOperator', False)
        # Getting the type of 'self' (line 553)
        self_385995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 37), 'self', False)
        # Processing the call keyword arguments (line 553)
        kwargs_385996 = {}
        # Getting the type of 'super' (line 553)
        super_385993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'super', False)
        # Calling super(args, kwargs) (line 553)
        super_call_result_385997 = invoke(stypy.reporting.localization.Localization(__file__, 553, 8), super_385993, *[_ScaledLinearOperator_385994, self_385995], **kwargs_385996)
        
        # Obtaining the member '__init__' of a type (line 553)
        init___385998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 8), super_call_result_385997, '__init__')
        # Calling __init__(args, kwargs) (line 553)
        init___call_result_386003 = invoke(stypy.reporting.localization.Localization(__file__, 553, 8), init___385998, *[dtype_385999, shape_386001], **kwargs_386002)
        
        
        # Assigning a Tuple to a Attribute (line 554):
        
        # Assigning a Tuple to a Attribute (line 554):
        
        # Obtaining an instance of the builtin type 'tuple' (line 554)
        tuple_386004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 554)
        # Adding element type (line 554)
        # Getting the type of 'A' (line 554)
        A_386005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'A')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 21), tuple_386004, A_386005)
        # Adding element type (line 554)
        # Getting the type of 'alpha' (line 554)
        alpha_386006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 24), 'alpha')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 554, 21), tuple_386004, alpha_386006)
        
        # Getting the type of 'self' (line 554)
        self_386007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'self')
        # Setting the type of the member 'args' of a type (line 554)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 8), self_386007, 'args', tuple_386004)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 556, 4, False)
        # Assigning a type to the variable 'self' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_function_name', '_ScaledLinearOperator._matvec')
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ScaledLinearOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ScaledLinearOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        # Obtaining the type of the subscript
        int_386008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 25), 'int')
        # Getting the type of 'self' (line 557)
        self_386009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 15), 'self')
        # Obtaining the member 'args' of a type (line 557)
        args_386010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 15), self_386009, 'args')
        # Obtaining the member '__getitem__' of a type (line 557)
        getitem___386011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 15), args_386010, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 557)
        subscript_call_result_386012 = invoke(stypy.reporting.localization.Localization(__file__, 557, 15), getitem___386011, int_386008)
        
        
        # Call to matvec(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'x' (line 557)
        x_386019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 50), 'x', False)
        # Processing the call keyword arguments (line 557)
        kwargs_386020 = {}
        
        # Obtaining the type of the subscript
        int_386013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 40), 'int')
        # Getting the type of 'self' (line 557)
        self_386014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 30), 'self', False)
        # Obtaining the member 'args' of a type (line 557)
        args_386015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 30), self_386014, 'args')
        # Obtaining the member '__getitem__' of a type (line 557)
        getitem___386016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 30), args_386015, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 557)
        subscript_call_result_386017 = invoke(stypy.reporting.localization.Localization(__file__, 557, 30), getitem___386016, int_386013)
        
        # Obtaining the member 'matvec' of a type (line 557)
        matvec_386018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 30), subscript_call_result_386017, 'matvec')
        # Calling matvec(args, kwargs) (line 557)
        matvec_call_result_386021 = invoke(stypy.reporting.localization.Localization(__file__, 557, 30), matvec_386018, *[x_386019], **kwargs_386020)
        
        # Applying the binary operator '*' (line 557)
        result_mul_386022 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 15), '*', subscript_call_result_386012, matvec_call_result_386021)
        
        # Assigning a type to the variable 'stypy_return_type' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'stypy_return_type', result_mul_386022)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 556)
        stypy_return_type_386023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386023)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_386023


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 559, 4, False)
        # Assigning a type to the variable 'self' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_function_name', '_ScaledLinearOperator._rmatvec')
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ScaledLinearOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ScaledLinearOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Call to conj(...): (line 560)
        # Processing the call arguments (line 560)
        
        # Obtaining the type of the subscript
        int_386026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 33), 'int')
        # Getting the type of 'self' (line 560)
        self_386027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 23), 'self', False)
        # Obtaining the member 'args' of a type (line 560)
        args_386028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 23), self_386027, 'args')
        # Obtaining the member '__getitem__' of a type (line 560)
        getitem___386029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 23), args_386028, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 560)
        subscript_call_result_386030 = invoke(stypy.reporting.localization.Localization(__file__, 560, 23), getitem___386029, int_386026)
        
        # Processing the call keyword arguments (line 560)
        kwargs_386031 = {}
        # Getting the type of 'np' (line 560)
        np_386024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'np', False)
        # Obtaining the member 'conj' of a type (line 560)
        conj_386025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 15), np_386024, 'conj')
        # Calling conj(args, kwargs) (line 560)
        conj_call_result_386032 = invoke(stypy.reporting.localization.Localization(__file__, 560, 15), conj_386025, *[subscript_call_result_386030], **kwargs_386031)
        
        
        # Call to rmatvec(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'x' (line 560)
        x_386039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 60), 'x', False)
        # Processing the call keyword arguments (line 560)
        kwargs_386040 = {}
        
        # Obtaining the type of the subscript
        int_386033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 49), 'int')
        # Getting the type of 'self' (line 560)
        self_386034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 39), 'self', False)
        # Obtaining the member 'args' of a type (line 560)
        args_386035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 39), self_386034, 'args')
        # Obtaining the member '__getitem__' of a type (line 560)
        getitem___386036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 39), args_386035, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 560)
        subscript_call_result_386037 = invoke(stypy.reporting.localization.Localization(__file__, 560, 39), getitem___386036, int_386033)
        
        # Obtaining the member 'rmatvec' of a type (line 560)
        rmatvec_386038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 39), subscript_call_result_386037, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 560)
        rmatvec_call_result_386041 = invoke(stypy.reporting.localization.Localization(__file__, 560, 39), rmatvec_386038, *[x_386039], **kwargs_386040)
        
        # Applying the binary operator '*' (line 560)
        result_mul_386042 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 15), '*', conj_call_result_386032, rmatvec_call_result_386041)
        
        # Assigning a type to the variable 'stypy_return_type' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'stypy_return_type', result_mul_386042)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 559)
        stypy_return_type_386043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386043)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_386043


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 562, 4, False)
        # Assigning a type to the variable 'self' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_function_name', '_ScaledLinearOperator._matmat')
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ScaledLinearOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ScaledLinearOperator._matmat', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        # Obtaining the type of the subscript
        int_386044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 25), 'int')
        # Getting the type of 'self' (line 563)
        self_386045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'self')
        # Obtaining the member 'args' of a type (line 563)
        args_386046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 15), self_386045, 'args')
        # Obtaining the member '__getitem__' of a type (line 563)
        getitem___386047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 15), args_386046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 563)
        subscript_call_result_386048 = invoke(stypy.reporting.localization.Localization(__file__, 563, 15), getitem___386047, int_386044)
        
        
        # Call to matmat(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'x' (line 563)
        x_386055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 50), 'x', False)
        # Processing the call keyword arguments (line 563)
        kwargs_386056 = {}
        
        # Obtaining the type of the subscript
        int_386049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 40), 'int')
        # Getting the type of 'self' (line 563)
        self_386050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 30), 'self', False)
        # Obtaining the member 'args' of a type (line 563)
        args_386051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 30), self_386050, 'args')
        # Obtaining the member '__getitem__' of a type (line 563)
        getitem___386052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 30), args_386051, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 563)
        subscript_call_result_386053 = invoke(stypy.reporting.localization.Localization(__file__, 563, 30), getitem___386052, int_386049)
        
        # Obtaining the member 'matmat' of a type (line 563)
        matmat_386054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 30), subscript_call_result_386053, 'matmat')
        # Calling matmat(args, kwargs) (line 563)
        matmat_call_result_386057 = invoke(stypy.reporting.localization.Localization(__file__, 563, 30), matmat_386054, *[x_386055], **kwargs_386056)
        
        # Applying the binary operator '*' (line 563)
        result_mul_386058 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 15), '*', subscript_call_result_386048, matmat_call_result_386057)
        
        # Assigning a type to the variable 'stypy_return_type' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'stypy_return_type', result_mul_386058)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 562)
        stypy_return_type_386059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_386059


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 565, 4, False)
        # Assigning a type to the variable 'self' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_function_name', '_ScaledLinearOperator._adjoint')
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ScaledLinearOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ScaledLinearOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 566):
        
        # Assigning a Subscript to a Name (line 566):
        
        # Obtaining the type of the subscript
        int_386060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 8), 'int')
        # Getting the type of 'self' (line 566)
        self_386061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'self')
        # Obtaining the member 'args' of a type (line 566)
        args_386062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 19), self_386061, 'args')
        # Obtaining the member '__getitem__' of a type (line 566)
        getitem___386063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), args_386062, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 566)
        subscript_call_result_386064 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), getitem___386063, int_386060)
        
        # Assigning a type to the variable 'tuple_var_assignment_384917' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_384917', subscript_call_result_386064)
        
        # Assigning a Subscript to a Name (line 566):
        
        # Obtaining the type of the subscript
        int_386065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 8), 'int')
        # Getting the type of 'self' (line 566)
        self_386066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'self')
        # Obtaining the member 'args' of a type (line 566)
        args_386067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 19), self_386066, 'args')
        # Obtaining the member '__getitem__' of a type (line 566)
        getitem___386068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), args_386067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 566)
        subscript_call_result_386069 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), getitem___386068, int_386065)
        
        # Assigning a type to the variable 'tuple_var_assignment_384918' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_384918', subscript_call_result_386069)
        
        # Assigning a Name to a Name (line 566):
        # Getting the type of 'tuple_var_assignment_384917' (line 566)
        tuple_var_assignment_384917_386070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_384917')
        # Assigning a type to the variable 'A' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'A', tuple_var_assignment_384917_386070)
        
        # Assigning a Name to a Name (line 566):
        # Getting the type of 'tuple_var_assignment_384918' (line 566)
        tuple_var_assignment_384918_386071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'tuple_var_assignment_384918')
        # Assigning a type to the variable 'alpha' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 11), 'alpha', tuple_var_assignment_384918_386071)
        # Getting the type of 'A' (line 567)
        A_386072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), 'A')
        # Obtaining the member 'H' of a type (line 567)
        H_386073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 15), A_386072, 'H')
        # Getting the type of 'alpha' (line 567)
        alpha_386074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'alpha')
        # Applying the binary operator '*' (line 567)
        result_mul_386075 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 15), '*', H_386073, alpha_386074)
        
        # Assigning a type to the variable 'stypy_return_type' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'stypy_return_type', result_mul_386075)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 565)
        stypy_return_type_386076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_386076


# Assigning a type to the variable '_ScaledLinearOperator' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), '_ScaledLinearOperator', _ScaledLinearOperator)
# Declaration of the '_PowerLinearOperator' class
# Getting the type of 'LinearOperator' (line 570)
LinearOperator_386077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 27), 'LinearOperator')

class _PowerLinearOperator(LinearOperator_386077, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 571, 4, False)
        # Assigning a type to the variable 'self' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PowerLinearOperator.__init__', ['A', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'A' (line 572)
        A_386079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 26), 'A', False)
        # Getting the type of 'LinearOperator' (line 572)
        LinearOperator_386080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 29), 'LinearOperator', False)
        # Processing the call keyword arguments (line 572)
        kwargs_386081 = {}
        # Getting the type of 'isinstance' (line 572)
        isinstance_386078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 572)
        isinstance_call_result_386082 = invoke(stypy.reporting.localization.Localization(__file__, 572, 15), isinstance_386078, *[A_386079, LinearOperator_386080], **kwargs_386081)
        
        # Applying the 'not' unary operator (line 572)
        result_not__386083 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 11), 'not', isinstance_call_result_386082)
        
        # Testing the type of an if condition (line 572)
        if_condition_386084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 572, 8), result_not__386083)
        # Assigning a type to the variable 'if_condition_386084' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'if_condition_386084', if_condition_386084)
        # SSA begins for if statement (line 572)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 573)
        # Processing the call arguments (line 573)
        str_386086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 29), 'str', 'LinearOperator expected as A')
        # Processing the call keyword arguments (line 573)
        kwargs_386087 = {}
        # Getting the type of 'ValueError' (line 573)
        ValueError_386085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 573)
        ValueError_call_result_386088 = invoke(stypy.reporting.localization.Localization(__file__, 573, 18), ValueError_386085, *[str_386086], **kwargs_386087)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 573, 12), ValueError_call_result_386088, 'raise parameter', BaseException)
        # SSA join for if statement (line 572)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_386089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 19), 'int')
        # Getting the type of 'A' (line 574)
        A_386090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), 'A')
        # Obtaining the member 'shape' of a type (line 574)
        shape_386091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 11), A_386090, 'shape')
        # Obtaining the member '__getitem__' of a type (line 574)
        getitem___386092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 11), shape_386091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 574)
        subscript_call_result_386093 = invoke(stypy.reporting.localization.Localization(__file__, 574, 11), getitem___386092, int_386089)
        
        
        # Obtaining the type of the subscript
        int_386094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 33), 'int')
        # Getting the type of 'A' (line 574)
        A_386095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 25), 'A')
        # Obtaining the member 'shape' of a type (line 574)
        shape_386096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 25), A_386095, 'shape')
        # Obtaining the member '__getitem__' of a type (line 574)
        getitem___386097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 25), shape_386096, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 574)
        subscript_call_result_386098 = invoke(stypy.reporting.localization.Localization(__file__, 574, 25), getitem___386097, int_386094)
        
        # Applying the binary operator '!=' (line 574)
        result_ne_386099 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), '!=', subscript_call_result_386093, subscript_call_result_386098)
        
        # Testing the type of an if condition (line 574)
        if_condition_386100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 8), result_ne_386099)
        # Assigning a type to the variable 'if_condition_386100' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'if_condition_386100', if_condition_386100)
        # SSA begins for if statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 575)
        # Processing the call arguments (line 575)
        str_386102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 29), 'str', 'square LinearOperator expected, got %r')
        # Getting the type of 'A' (line 575)
        A_386103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 72), 'A', False)
        # Applying the binary operator '%' (line 575)
        result_mod_386104 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 29), '%', str_386102, A_386103)
        
        # Processing the call keyword arguments (line 575)
        kwargs_386105 = {}
        # Getting the type of 'ValueError' (line 575)
        ValueError_386101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 575)
        ValueError_call_result_386106 = invoke(stypy.reporting.localization.Localization(__file__, 575, 18), ValueError_386101, *[result_mod_386104], **kwargs_386105)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 575, 12), ValueError_call_result_386106, 'raise parameter', BaseException)
        # SSA join for if statement (line 574)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to isintlike(...): (line 576)
        # Processing the call arguments (line 576)
        # Getting the type of 'p' (line 576)
        p_386108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 25), 'p', False)
        # Processing the call keyword arguments (line 576)
        kwargs_386109 = {}
        # Getting the type of 'isintlike' (line 576)
        isintlike_386107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 15), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 576)
        isintlike_call_result_386110 = invoke(stypy.reporting.localization.Localization(__file__, 576, 15), isintlike_386107, *[p_386108], **kwargs_386109)
        
        # Applying the 'not' unary operator (line 576)
        result_not__386111 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 11), 'not', isintlike_call_result_386110)
        
        
        # Getting the type of 'p' (line 576)
        p_386112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 31), 'p')
        int_386113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 35), 'int')
        # Applying the binary operator '<' (line 576)
        result_lt_386114 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 31), '<', p_386112, int_386113)
        
        # Applying the binary operator 'or' (line 576)
        result_or_keyword_386115 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 11), 'or', result_not__386111, result_lt_386114)
        
        # Testing the type of an if condition (line 576)
        if_condition_386116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 8), result_or_keyword_386115)
        # Assigning a type to the variable 'if_condition_386116' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'if_condition_386116', if_condition_386116)
        # SSA begins for if statement (line 576)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 577)
        # Processing the call arguments (line 577)
        str_386118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 29), 'str', 'non-negative integer expected as p')
        # Processing the call keyword arguments (line 577)
        kwargs_386119 = {}
        # Getting the type of 'ValueError' (line 577)
        ValueError_386117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 577)
        ValueError_call_result_386120 = invoke(stypy.reporting.localization.Localization(__file__, 577, 18), ValueError_386117, *[str_386118], **kwargs_386119)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 577, 12), ValueError_call_result_386120, 'raise parameter', BaseException)
        # SSA join for if statement (line 576)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __init__(...): (line 579)
        # Processing the call arguments (line 579)
        
        # Call to _get_dtype(...): (line 579)
        # Processing the call arguments (line 579)
        
        # Obtaining an instance of the builtin type 'list' (line 579)
        list_386128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 579)
        # Adding element type (line 579)
        # Getting the type of 'A' (line 579)
        A_386129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 63), 'A', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 62), list_386128, A_386129)
        
        # Processing the call keyword arguments (line 579)
        kwargs_386130 = {}
        # Getting the type of '_get_dtype' (line 579)
        _get_dtype_386127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 51), '_get_dtype', False)
        # Calling _get_dtype(args, kwargs) (line 579)
        _get_dtype_call_result_386131 = invoke(stypy.reporting.localization.Localization(__file__, 579, 51), _get_dtype_386127, *[list_386128], **kwargs_386130)
        
        # Getting the type of 'A' (line 579)
        A_386132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 68), 'A', False)
        # Obtaining the member 'shape' of a type (line 579)
        shape_386133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 68), A_386132, 'shape')
        # Processing the call keyword arguments (line 579)
        kwargs_386134 = {}
        
        # Call to super(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of '_PowerLinearOperator' (line 579)
        _PowerLinearOperator_386122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 14), '_PowerLinearOperator', False)
        # Getting the type of 'self' (line 579)
        self_386123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 36), 'self', False)
        # Processing the call keyword arguments (line 579)
        kwargs_386124 = {}
        # Getting the type of 'super' (line 579)
        super_386121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'super', False)
        # Calling super(args, kwargs) (line 579)
        super_call_result_386125 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), super_386121, *[_PowerLinearOperator_386122, self_386123], **kwargs_386124)
        
        # Obtaining the member '__init__' of a type (line 579)
        init___386126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), super_call_result_386125, '__init__')
        # Calling __init__(args, kwargs) (line 579)
        init___call_result_386135 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), init___386126, *[_get_dtype_call_result_386131, shape_386133], **kwargs_386134)
        
        
        # Assigning a Tuple to a Attribute (line 580):
        
        # Assigning a Tuple to a Attribute (line 580):
        
        # Obtaining an instance of the builtin type 'tuple' (line 580)
        tuple_386136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 580)
        # Adding element type (line 580)
        # Getting the type of 'A' (line 580)
        A_386137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 21), 'A')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 21), tuple_386136, A_386137)
        # Adding element type (line 580)
        # Getting the type of 'p' (line 580)
        p_386138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 24), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 21), tuple_386136, p_386138)
        
        # Getting the type of 'self' (line 580)
        self_386139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'self')
        # Setting the type of the member 'args' of a type (line 580)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 8), self_386139, 'args', tuple_386136)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _power(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_power'
        module_type_store = module_type_store.open_function_context('_power', 582, 4, False)
        # Assigning a type to the variable 'self' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_localization', localization)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_type_store', module_type_store)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_function_name', '_PowerLinearOperator._power')
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_param_names_list', ['fun', 'x'])
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_varargs_param_name', None)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_call_defaults', defaults)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_call_varargs', varargs)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _PowerLinearOperator._power.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PowerLinearOperator._power', ['fun', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_power', localization, ['fun', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_power(...)' code ##################

        
        # Assigning a Call to a Name (line 583):
        
        # Assigning a Call to a Name (line 583):
        
        # Call to array(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'x' (line 583)
        x_386142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 23), 'x', False)
        # Processing the call keyword arguments (line 583)
        # Getting the type of 'True' (line 583)
        True_386143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 31), 'True', False)
        keyword_386144 = True_386143
        kwargs_386145 = {'copy': keyword_386144}
        # Getting the type of 'np' (line 583)
        np_386140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 14), 'np', False)
        # Obtaining the member 'array' of a type (line 583)
        array_386141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 14), np_386140, 'array')
        # Calling array(args, kwargs) (line 583)
        array_call_result_386146 = invoke(stypy.reporting.localization.Localization(__file__, 583, 14), array_386141, *[x_386142], **kwargs_386145)
        
        # Assigning a type to the variable 'res' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'res', array_call_result_386146)
        
        
        # Call to range(...): (line 584)
        # Processing the call arguments (line 584)
        
        # Obtaining the type of the subscript
        int_386148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 33), 'int')
        # Getting the type of 'self' (line 584)
        self_386149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 23), 'self', False)
        # Obtaining the member 'args' of a type (line 584)
        args_386150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 23), self_386149, 'args')
        # Obtaining the member '__getitem__' of a type (line 584)
        getitem___386151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 23), args_386150, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 584)
        subscript_call_result_386152 = invoke(stypy.reporting.localization.Localization(__file__, 584, 23), getitem___386151, int_386148)
        
        # Processing the call keyword arguments (line 584)
        kwargs_386153 = {}
        # Getting the type of 'range' (line 584)
        range_386147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 17), 'range', False)
        # Calling range(args, kwargs) (line 584)
        range_call_result_386154 = invoke(stypy.reporting.localization.Localization(__file__, 584, 17), range_386147, *[subscript_call_result_386152], **kwargs_386153)
        
        # Testing the type of a for loop iterable (line 584)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 584, 8), range_call_result_386154)
        # Getting the type of the for loop variable (line 584)
        for_loop_var_386155 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 584, 8), range_call_result_386154)
        # Assigning a type to the variable 'i' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'i', for_loop_var_386155)
        # SSA begins for a for statement (line 584)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 585):
        
        # Assigning a Call to a Name (line 585):
        
        # Call to fun(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'res' (line 585)
        res_386157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 22), 'res', False)
        # Processing the call keyword arguments (line 585)
        kwargs_386158 = {}
        # Getting the type of 'fun' (line 585)
        fun_386156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 18), 'fun', False)
        # Calling fun(args, kwargs) (line 585)
        fun_call_result_386159 = invoke(stypy.reporting.localization.Localization(__file__, 585, 18), fun_386156, *[res_386157], **kwargs_386158)
        
        # Assigning a type to the variable 'res' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'res', fun_call_result_386159)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'res' (line 586)
        res_386160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'stypy_return_type', res_386160)
        
        # ################# End of '_power(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_power' in the type store
        # Getting the type of 'stypy_return_type' (line 582)
        stypy_return_type_386161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386161)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_power'
        return stypy_return_type_386161


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 588, 4, False)
        # Assigning a type to the variable 'self' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_function_name', '_PowerLinearOperator._matvec')
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _PowerLinearOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PowerLinearOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        
        # Call to _power(...): (line 589)
        # Processing the call arguments (line 589)
        
        # Obtaining the type of the subscript
        int_386164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 37), 'int')
        # Getting the type of 'self' (line 589)
        self_386165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 27), 'self', False)
        # Obtaining the member 'args' of a type (line 589)
        args_386166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 27), self_386165, 'args')
        # Obtaining the member '__getitem__' of a type (line 589)
        getitem___386167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 27), args_386166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 589)
        subscript_call_result_386168 = invoke(stypy.reporting.localization.Localization(__file__, 589, 27), getitem___386167, int_386164)
        
        # Obtaining the member 'matvec' of a type (line 589)
        matvec_386169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 27), subscript_call_result_386168, 'matvec')
        # Getting the type of 'x' (line 589)
        x_386170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 48), 'x', False)
        # Processing the call keyword arguments (line 589)
        kwargs_386171 = {}
        # Getting the type of 'self' (line 589)
        self_386162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 15), 'self', False)
        # Obtaining the member '_power' of a type (line 589)
        _power_386163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 15), self_386162, '_power')
        # Calling _power(args, kwargs) (line 589)
        _power_call_result_386172 = invoke(stypy.reporting.localization.Localization(__file__, 589, 15), _power_386163, *[matvec_386169, x_386170], **kwargs_386171)
        
        # Assigning a type to the variable 'stypy_return_type' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'stypy_return_type', _power_call_result_386172)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 588)
        stypy_return_type_386173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_386173


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 591, 4, False)
        # Assigning a type to the variable 'self' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_function_name', '_PowerLinearOperator._rmatvec')
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _PowerLinearOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PowerLinearOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        
        # Call to _power(...): (line 592)
        # Processing the call arguments (line 592)
        
        # Obtaining the type of the subscript
        int_386176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 37), 'int')
        # Getting the type of 'self' (line 592)
        self_386177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 27), 'self', False)
        # Obtaining the member 'args' of a type (line 592)
        args_386178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 27), self_386177, 'args')
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___386179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 27), args_386178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_386180 = invoke(stypy.reporting.localization.Localization(__file__, 592, 27), getitem___386179, int_386176)
        
        # Obtaining the member 'rmatvec' of a type (line 592)
        rmatvec_386181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 27), subscript_call_result_386180, 'rmatvec')
        # Getting the type of 'x' (line 592)
        x_386182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 49), 'x', False)
        # Processing the call keyword arguments (line 592)
        kwargs_386183 = {}
        # Getting the type of 'self' (line 592)
        self_386174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'self', False)
        # Obtaining the member '_power' of a type (line 592)
        _power_386175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 15), self_386174, '_power')
        # Calling _power(args, kwargs) (line 592)
        _power_call_result_386184 = invoke(stypy.reporting.localization.Localization(__file__, 592, 15), _power_386175, *[rmatvec_386181, x_386182], **kwargs_386183)
        
        # Assigning a type to the variable 'stypy_return_type' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'stypy_return_type', _power_call_result_386184)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 591)
        stypy_return_type_386185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386185)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_386185


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 594, 4, False)
        # Assigning a type to the variable 'self' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_function_name', '_PowerLinearOperator._matmat')
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['x'])
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _PowerLinearOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PowerLinearOperator._matmat', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        # Call to _power(...): (line 595)
        # Processing the call arguments (line 595)
        
        # Obtaining the type of the subscript
        int_386188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 37), 'int')
        # Getting the type of 'self' (line 595)
        self_386189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 27), 'self', False)
        # Obtaining the member 'args' of a type (line 595)
        args_386190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 27), self_386189, 'args')
        # Obtaining the member '__getitem__' of a type (line 595)
        getitem___386191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 27), args_386190, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 595)
        subscript_call_result_386192 = invoke(stypy.reporting.localization.Localization(__file__, 595, 27), getitem___386191, int_386188)
        
        # Obtaining the member 'matmat' of a type (line 595)
        matmat_386193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 27), subscript_call_result_386192, 'matmat')
        # Getting the type of 'x' (line 595)
        x_386194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 48), 'x', False)
        # Processing the call keyword arguments (line 595)
        kwargs_386195 = {}
        # Getting the type of 'self' (line 595)
        self_386186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 15), 'self', False)
        # Obtaining the member '_power' of a type (line 595)
        _power_386187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 15), self_386186, '_power')
        # Calling _power(args, kwargs) (line 595)
        _power_call_result_386196 = invoke(stypy.reporting.localization.Localization(__file__, 595, 15), _power_386187, *[matmat_386193, x_386194], **kwargs_386195)
        
        # Assigning a type to the variable 'stypy_return_type' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'stypy_return_type', _power_call_result_386196)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 594)
        stypy_return_type_386197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_386197


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 597, 4, False)
        # Assigning a type to the variable 'self' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_function_name', '_PowerLinearOperator._adjoint')
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _PowerLinearOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_PowerLinearOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 598):
        
        # Assigning a Subscript to a Name (line 598):
        
        # Obtaining the type of the subscript
        int_386198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 8), 'int')
        # Getting the type of 'self' (line 598)
        self_386199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'self')
        # Obtaining the member 'args' of a type (line 598)
        args_386200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 15), self_386199, 'args')
        # Obtaining the member '__getitem__' of a type (line 598)
        getitem___386201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 8), args_386200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 598)
        subscript_call_result_386202 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), getitem___386201, int_386198)
        
        # Assigning a type to the variable 'tuple_var_assignment_384919' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_384919', subscript_call_result_386202)
        
        # Assigning a Subscript to a Name (line 598):
        
        # Obtaining the type of the subscript
        int_386203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 8), 'int')
        # Getting the type of 'self' (line 598)
        self_386204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'self')
        # Obtaining the member 'args' of a type (line 598)
        args_386205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 15), self_386204, 'args')
        # Obtaining the member '__getitem__' of a type (line 598)
        getitem___386206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 8), args_386205, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 598)
        subscript_call_result_386207 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), getitem___386206, int_386203)
        
        # Assigning a type to the variable 'tuple_var_assignment_384920' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_384920', subscript_call_result_386207)
        
        # Assigning a Name to a Name (line 598):
        # Getting the type of 'tuple_var_assignment_384919' (line 598)
        tuple_var_assignment_384919_386208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_384919')
        # Assigning a type to the variable 'A' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'A', tuple_var_assignment_384919_386208)
        
        # Assigning a Name to a Name (line 598):
        # Getting the type of 'tuple_var_assignment_384920' (line 598)
        tuple_var_assignment_384920_386209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'tuple_var_assignment_384920')
        # Assigning a type to the variable 'p' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 11), 'p', tuple_var_assignment_384920_386209)
        # Getting the type of 'A' (line 599)
        A_386210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 15), 'A')
        # Obtaining the member 'H' of a type (line 599)
        H_386211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 15), A_386210, 'H')
        # Getting the type of 'p' (line 599)
        p_386212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 22), 'p')
        # Applying the binary operator '**' (line 599)
        result_pow_386213 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 15), '**', H_386211, p_386212)
        
        # Assigning a type to the variable 'stypy_return_type' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'stypy_return_type', result_pow_386213)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 597)
        stypy_return_type_386214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_386214


# Assigning a type to the variable '_PowerLinearOperator' (line 570)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 0), '_PowerLinearOperator', _PowerLinearOperator)
# Declaration of the 'MatrixLinearOperator' class
# Getting the type of 'LinearOperator' (line 602)
LinearOperator_386215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 27), 'LinearOperator')

class MatrixLinearOperator(LinearOperator_386215, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 603, 4, False)
        # Assigning a type to the variable 'self' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixLinearOperator.__init__', ['A'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['A'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 604)
        # Processing the call arguments (line 604)
        # Getting the type of 'A' (line 604)
        A_386222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 51), 'A', False)
        # Obtaining the member 'dtype' of a type (line 604)
        dtype_386223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 51), A_386222, 'dtype')
        # Getting the type of 'A' (line 604)
        A_386224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 60), 'A', False)
        # Obtaining the member 'shape' of a type (line 604)
        shape_386225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 60), A_386224, 'shape')
        # Processing the call keyword arguments (line 604)
        kwargs_386226 = {}
        
        # Call to super(...): (line 604)
        # Processing the call arguments (line 604)
        # Getting the type of 'MatrixLinearOperator' (line 604)
        MatrixLinearOperator_386217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 14), 'MatrixLinearOperator', False)
        # Getting the type of 'self' (line 604)
        self_386218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 36), 'self', False)
        # Processing the call keyword arguments (line 604)
        kwargs_386219 = {}
        # Getting the type of 'super' (line 604)
        super_386216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'super', False)
        # Calling super(args, kwargs) (line 604)
        super_call_result_386220 = invoke(stypy.reporting.localization.Localization(__file__, 604, 8), super_386216, *[MatrixLinearOperator_386217, self_386218], **kwargs_386219)
        
        # Obtaining the member '__init__' of a type (line 604)
        init___386221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 8), super_call_result_386220, '__init__')
        # Calling __init__(args, kwargs) (line 604)
        init___call_result_386227 = invoke(stypy.reporting.localization.Localization(__file__, 604, 8), init___386221, *[dtype_386223, shape_386225], **kwargs_386226)
        
        
        # Assigning a Name to a Attribute (line 605):
        
        # Assigning a Name to a Attribute (line 605):
        # Getting the type of 'A' (line 605)
        A_386228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 17), 'A')
        # Getting the type of 'self' (line 605)
        self_386229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'self')
        # Setting the type of the member 'A' of a type (line 605)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 8), self_386229, 'A', A_386228)
        
        # Assigning a Name to a Attribute (line 606):
        
        # Assigning a Name to a Attribute (line 606):
        # Getting the type of 'None' (line 606)
        None_386230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 21), 'None')
        # Getting the type of 'self' (line 606)
        self_386231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'self')
        # Setting the type of the member '__adj' of a type (line 606)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 8), self_386231, '__adj', None_386230)
        
        # Assigning a Tuple to a Attribute (line 607):
        
        # Assigning a Tuple to a Attribute (line 607):
        
        # Obtaining an instance of the builtin type 'tuple' (line 607)
        tuple_386232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 607)
        # Adding element type (line 607)
        # Getting the type of 'A' (line 607)
        A_386233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 21), 'A')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 21), tuple_386232, A_386233)
        
        # Getting the type of 'self' (line 607)
        self_386234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'self')
        # Setting the type of the member 'args' of a type (line 607)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 8), self_386234, 'args', tuple_386232)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 609, 4, False)
        # Assigning a type to the variable 'self' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_function_name', 'MatrixLinearOperator._matmat')
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['X'])
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixLinearOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixLinearOperator._matmat', ['X'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['X'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        
        # Call to dot(...): (line 610)
        # Processing the call arguments (line 610)
        # Getting the type of 'X' (line 610)
        X_386238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 26), 'X', False)
        # Processing the call keyword arguments (line 610)
        kwargs_386239 = {}
        # Getting the type of 'self' (line 610)
        self_386235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 15), 'self', False)
        # Obtaining the member 'A' of a type (line 610)
        A_386236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 15), self_386235, 'A')
        # Obtaining the member 'dot' of a type (line 610)
        dot_386237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 15), A_386236, 'dot')
        # Calling dot(args, kwargs) (line 610)
        dot_call_result_386240 = invoke(stypy.reporting.localization.Localization(__file__, 610, 15), dot_386237, *[X_386238], **kwargs_386239)
        
        # Assigning a type to the variable 'stypy_return_type' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'stypy_return_type', dot_call_result_386240)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 609)
        stypy_return_type_386241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386241)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_386241


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 612, 4, False)
        # Assigning a type to the variable 'self' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_function_name', 'MatrixLinearOperator._adjoint')
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatrixLinearOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatrixLinearOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 613)
        # Getting the type of 'self' (line 613)
        self_386242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 11), 'self')
        # Obtaining the member '__adj' of a type (line 613)
        adj_386243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 11), self_386242, '__adj')
        # Getting the type of 'None' (line 613)
        None_386244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 25), 'None')
        
        (may_be_386245, more_types_in_union_386246) = may_be_none(adj_386243, None_386244)

        if may_be_386245:

            if more_types_in_union_386246:
                # Runtime conditional SSA (line 613)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 614):
            
            # Assigning a Call to a Attribute (line 614):
            
            # Call to _AdjointMatrixOperator(...): (line 614)
            # Processing the call arguments (line 614)
            # Getting the type of 'self' (line 614)
            self_386248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 48), 'self', False)
            # Processing the call keyword arguments (line 614)
            kwargs_386249 = {}
            # Getting the type of '_AdjointMatrixOperator' (line 614)
            _AdjointMatrixOperator_386247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 25), '_AdjointMatrixOperator', False)
            # Calling _AdjointMatrixOperator(args, kwargs) (line 614)
            _AdjointMatrixOperator_call_result_386250 = invoke(stypy.reporting.localization.Localization(__file__, 614, 25), _AdjointMatrixOperator_386247, *[self_386248], **kwargs_386249)
            
            # Getting the type of 'self' (line 614)
            self_386251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'self')
            # Setting the type of the member '__adj' of a type (line 614)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 12), self_386251, '__adj', _AdjointMatrixOperator_call_result_386250)

            if more_types_in_union_386246:
                # SSA join for if statement (line 613)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 615)
        self_386252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 15), 'self')
        # Obtaining the member '__adj' of a type (line 615)
        adj_386253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 15), self_386252, '__adj')
        # Assigning a type to the variable 'stypy_return_type' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'stypy_return_type', adj_386253)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 612)
        stypy_return_type_386254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386254)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_386254


# Assigning a type to the variable 'MatrixLinearOperator' (line 602)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'MatrixLinearOperator', MatrixLinearOperator)
# Declaration of the '_AdjointMatrixOperator' class
# Getting the type of 'MatrixLinearOperator' (line 618)
MatrixLinearOperator_386255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 29), 'MatrixLinearOperator')

class _AdjointMatrixOperator(MatrixLinearOperator_386255, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 619, 4, False)
        # Assigning a type to the variable 'self' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_AdjointMatrixOperator.__init__', ['adjoint'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['adjoint'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 620):
        
        # Assigning a Call to a Attribute (line 620):
        
        # Call to conj(...): (line 620)
        # Processing the call keyword arguments (line 620)
        kwargs_386260 = {}
        # Getting the type of 'adjoint' (line 620)
        adjoint_386256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 17), 'adjoint', False)
        # Obtaining the member 'A' of a type (line 620)
        A_386257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 17), adjoint_386256, 'A')
        # Obtaining the member 'T' of a type (line 620)
        T_386258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 17), A_386257, 'T')
        # Obtaining the member 'conj' of a type (line 620)
        conj_386259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 17), T_386258, 'conj')
        # Calling conj(args, kwargs) (line 620)
        conj_call_result_386261 = invoke(stypy.reporting.localization.Localization(__file__, 620, 17), conj_386259, *[], **kwargs_386260)
        
        # Getting the type of 'self' (line 620)
        self_386262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'self')
        # Setting the type of the member 'A' of a type (line 620)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 8), self_386262, 'A', conj_call_result_386261)
        
        # Assigning a Name to a Attribute (line 621):
        
        # Assigning a Name to a Attribute (line 621):
        # Getting the type of 'adjoint' (line 621)
        adjoint_386263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 25), 'adjoint')
        # Getting the type of 'self' (line 621)
        self_386264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'self')
        # Setting the type of the member '__adjoint' of a type (line 621)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 8), self_386264, '__adjoint', adjoint_386263)
        
        # Assigning a Tuple to a Attribute (line 622):
        
        # Assigning a Tuple to a Attribute (line 622):
        
        # Obtaining an instance of the builtin type 'tuple' (line 622)
        tuple_386265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 622)
        # Adding element type (line 622)
        # Getting the type of 'adjoint' (line 622)
        adjoint_386266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 21), 'adjoint')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 21), tuple_386265, adjoint_386266)
        
        # Getting the type of 'self' (line 622)
        self_386267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'self')
        # Setting the type of the member 'args' of a type (line 622)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 8), self_386267, 'args', tuple_386265)
        
        # Assigning a Tuple to a Attribute (line 623):
        
        # Assigning a Tuple to a Attribute (line 623):
        
        # Obtaining an instance of the builtin type 'tuple' (line 623)
        tuple_386268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 623)
        # Adding element type (line 623)
        
        # Obtaining the type of the subscript
        int_386269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 35), 'int')
        # Getting the type of 'adjoint' (line 623)
        adjoint_386270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 21), 'adjoint')
        # Obtaining the member 'shape' of a type (line 623)
        shape_386271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 21), adjoint_386270, 'shape')
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___386272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 21), shape_386271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_386273 = invoke(stypy.reporting.localization.Localization(__file__, 623, 21), getitem___386272, int_386269)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 21), tuple_386268, subscript_call_result_386273)
        # Adding element type (line 623)
        
        # Obtaining the type of the subscript
        int_386274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 53), 'int')
        # Getting the type of 'adjoint' (line 623)
        adjoint_386275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 39), 'adjoint')
        # Obtaining the member 'shape' of a type (line 623)
        shape_386276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 39), adjoint_386275, 'shape')
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___386277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 39), shape_386276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_386278 = invoke(stypy.reporting.localization.Localization(__file__, 623, 39), getitem___386277, int_386274)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 21), tuple_386268, subscript_call_result_386278)
        
        # Getting the type of 'self' (line 623)
        self_386279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'self')
        # Setting the type of the member 'shape' of a type (line 623)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 8), self_386279, 'shape', tuple_386268)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dtype'
        module_type_store = module_type_store.open_function_context('dtype', 625, 4, False)
        # Assigning a type to the variable 'self' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_localization', localization)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_function_name', '_AdjointMatrixOperator.dtype')
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_param_names_list', [])
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _AdjointMatrixOperator.dtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_AdjointMatrixOperator.dtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dtype(...)' code ##################

        # Getting the type of 'self' (line 627)
        self_386280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 15), 'self')
        # Obtaining the member '__adjoint' of a type (line 627)
        adjoint_386281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 15), self_386280, '__adjoint')
        # Obtaining the member 'dtype' of a type (line 627)
        dtype_386282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 15), adjoint_386281, 'dtype')
        # Assigning a type to the variable 'stypy_return_type' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'stypy_return_type', dtype_386282)
        
        # ################# End of 'dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 625)
        stypy_return_type_386283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dtype'
        return stypy_return_type_386283


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 629, 4, False)
        # Assigning a type to the variable 'self' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_function_name', '_AdjointMatrixOperator._adjoint')
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _AdjointMatrixOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_AdjointMatrixOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        # Getting the type of 'self' (line 630)
        self_386284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 15), 'self')
        # Obtaining the member '__adjoint' of a type (line 630)
        adjoint_386285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 15), self_386284, '__adjoint')
        # Assigning a type to the variable 'stypy_return_type' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'stypy_return_type', adjoint_386285)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 629)
        stypy_return_type_386286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_386286


# Assigning a type to the variable '_AdjointMatrixOperator' (line 618)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), '_AdjointMatrixOperator', _AdjointMatrixOperator)
# Declaration of the 'IdentityOperator' class
# Getting the type of 'LinearOperator' (line 633)
LinearOperator_386287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 23), 'LinearOperator')

class IdentityOperator(LinearOperator_386287, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 634)
        None_386288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 36), 'None')
        defaults = [None_386288]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 634, 4, False)
        # Assigning a type to the variable 'self' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdentityOperator.__init__', ['shape', 'dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['shape', 'dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'dtype' (line 635)
        dtype_386295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 47), 'dtype', False)
        # Getting the type of 'shape' (line 635)
        shape_386296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 54), 'shape', False)
        # Processing the call keyword arguments (line 635)
        kwargs_386297 = {}
        
        # Call to super(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'IdentityOperator' (line 635)
        IdentityOperator_386290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 14), 'IdentityOperator', False)
        # Getting the type of 'self' (line 635)
        self_386291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 32), 'self', False)
        # Processing the call keyword arguments (line 635)
        kwargs_386292 = {}
        # Getting the type of 'super' (line 635)
        super_386289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'super', False)
        # Calling super(args, kwargs) (line 635)
        super_call_result_386293 = invoke(stypy.reporting.localization.Localization(__file__, 635, 8), super_386289, *[IdentityOperator_386290, self_386291], **kwargs_386292)
        
        # Obtaining the member '__init__' of a type (line 635)
        init___386294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 8), super_call_result_386293, '__init__')
        # Calling __init__(args, kwargs) (line 635)
        init___call_result_386298 = invoke(stypy.reporting.localization.Localization(__file__, 635, 8), init___386294, *[dtype_386295, shape_386296], **kwargs_386297)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _matvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matvec'
        module_type_store = module_type_store.open_function_context('_matvec', 637, 4, False)
        # Assigning a type to the variable 'self' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IdentityOperator._matvec.__dict__.__setitem__('stypy_localization', localization)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_function_name', 'IdentityOperator._matvec')
        IdentityOperator._matvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        IdentityOperator._matvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IdentityOperator._matvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdentityOperator._matvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matvec(...)' code ##################

        # Getting the type of 'x' (line 638)
        x_386299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'stypy_return_type', x_386299)
        
        # ################# End of '_matvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matvec' in the type store
        # Getting the type of 'stypy_return_type' (line 637)
        stypy_return_type_386300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matvec'
        return stypy_return_type_386300


    @norecursion
    def _rmatvec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rmatvec'
        module_type_store = module_type_store.open_function_context('_rmatvec', 640, 4, False)
        # Assigning a type to the variable 'self' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_localization', localization)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_type_store', module_type_store)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_function_name', 'IdentityOperator._rmatvec')
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_param_names_list', ['x'])
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_varargs_param_name', None)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_call_defaults', defaults)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_call_varargs', varargs)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IdentityOperator._rmatvec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdentityOperator._rmatvec', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rmatvec', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rmatvec(...)' code ##################

        # Getting the type of 'x' (line 641)
        x_386301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'stypy_return_type', x_386301)
        
        # ################# End of '_rmatvec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rmatvec' in the type store
        # Getting the type of 'stypy_return_type' (line 640)
        stypy_return_type_386302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386302)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rmatvec'
        return stypy_return_type_386302


    @norecursion
    def _matmat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_matmat'
        module_type_store = module_type_store.open_function_context('_matmat', 643, 4, False)
        # Assigning a type to the variable 'self' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IdentityOperator._matmat.__dict__.__setitem__('stypy_localization', localization)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_type_store', module_type_store)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_function_name', 'IdentityOperator._matmat')
        IdentityOperator._matmat.__dict__.__setitem__('stypy_param_names_list', ['x'])
        IdentityOperator._matmat.__dict__.__setitem__('stypy_varargs_param_name', None)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_call_defaults', defaults)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_call_varargs', varargs)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IdentityOperator._matmat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdentityOperator._matmat', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_matmat', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_matmat(...)' code ##################

        # Getting the type of 'x' (line 644)
        x_386303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'stypy_return_type', x_386303)
        
        # ################# End of '_matmat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_matmat' in the type store
        # Getting the type of 'stypy_return_type' (line 643)
        stypy_return_type_386304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386304)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_matmat'
        return stypy_return_type_386304


    @norecursion
    def _adjoint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_adjoint'
        module_type_store = module_type_store.open_function_context('_adjoint', 646, 4, False)
        # Assigning a type to the variable 'self' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_localization', localization)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_type_store', module_type_store)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_function_name', 'IdentityOperator._adjoint')
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_param_names_list', [])
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_varargs_param_name', None)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_call_defaults', defaults)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_call_varargs', varargs)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IdentityOperator._adjoint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IdentityOperator._adjoint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_adjoint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_adjoint(...)' code ##################

        # Getting the type of 'self' (line 647)
        self_386305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'stypy_return_type', self_386305)
        
        # ################# End of '_adjoint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_adjoint' in the type store
        # Getting the type of 'stypy_return_type' (line 646)
        stypy_return_type_386306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_386306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_adjoint'
        return stypy_return_type_386306


# Assigning a type to the variable 'IdentityOperator' (line 633)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 0), 'IdentityOperator', IdentityOperator)

@norecursion
def aslinearoperator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'aslinearoperator'
    module_type_store = module_type_store.open_function_context('aslinearoperator', 650, 0, False)
    
    # Passed parameters checking function
    aslinearoperator.stypy_localization = localization
    aslinearoperator.stypy_type_of_self = None
    aslinearoperator.stypy_type_store = module_type_store
    aslinearoperator.stypy_function_name = 'aslinearoperator'
    aslinearoperator.stypy_param_names_list = ['A']
    aslinearoperator.stypy_varargs_param_name = None
    aslinearoperator.stypy_kwargs_param_name = None
    aslinearoperator.stypy_call_defaults = defaults
    aslinearoperator.stypy_call_varargs = varargs
    aslinearoperator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'aslinearoperator', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'aslinearoperator', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'aslinearoperator(...)' code ##################

    str_386307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, (-1)), 'str', "Return A as a LinearOperator.\n\n    'A' may be any of the following types:\n     - ndarray\n     - matrix\n     - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)\n     - LinearOperator\n     - An object with .shape and .matvec attributes\n\n    See the LinearOperator documentation for additional information.\n\n    Examples\n    --------\n    >>> from scipy.sparse.linalg import aslinearoperator\n    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)\n    >>> aslinearoperator(M)\n    <2x3 MatrixLinearOperator with dtype=int32>\n\n    ")
    
    
    # Call to isinstance(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'A' (line 670)
    A_386309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 18), 'A', False)
    # Getting the type of 'LinearOperator' (line 670)
    LinearOperator_386310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 21), 'LinearOperator', False)
    # Processing the call keyword arguments (line 670)
    kwargs_386311 = {}
    # Getting the type of 'isinstance' (line 670)
    isinstance_386308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 670)
    isinstance_call_result_386312 = invoke(stypy.reporting.localization.Localization(__file__, 670, 7), isinstance_386308, *[A_386309, LinearOperator_386310], **kwargs_386311)
    
    # Testing the type of an if condition (line 670)
    if_condition_386313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 4), isinstance_call_result_386312)
    # Assigning a type to the variable 'if_condition_386313' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'if_condition_386313', if_condition_386313)
    # SSA begins for if statement (line 670)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'A' (line 671)
    A_386314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 15), 'A')
    # Assigning a type to the variable 'stypy_return_type' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'stypy_return_type', A_386314)
    # SSA branch for the else part of an if statement (line 670)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 673)
    # Processing the call arguments (line 673)
    # Getting the type of 'A' (line 673)
    A_386316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 20), 'A', False)
    # Getting the type of 'np' (line 673)
    np_386317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 23), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 673)
    ndarray_386318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 23), np_386317, 'ndarray')
    # Processing the call keyword arguments (line 673)
    kwargs_386319 = {}
    # Getting the type of 'isinstance' (line 673)
    isinstance_386315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 673)
    isinstance_call_result_386320 = invoke(stypy.reporting.localization.Localization(__file__, 673, 9), isinstance_386315, *[A_386316, ndarray_386318], **kwargs_386319)
    
    
    # Call to isinstance(...): (line 673)
    # Processing the call arguments (line 673)
    # Getting the type of 'A' (line 673)
    A_386322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 49), 'A', False)
    # Getting the type of 'np' (line 673)
    np_386323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 52), 'np', False)
    # Obtaining the member 'matrix' of a type (line 673)
    matrix_386324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 52), np_386323, 'matrix')
    # Processing the call keyword arguments (line 673)
    kwargs_386325 = {}
    # Getting the type of 'isinstance' (line 673)
    isinstance_386321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 38), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 673)
    isinstance_call_result_386326 = invoke(stypy.reporting.localization.Localization(__file__, 673, 38), isinstance_386321, *[A_386322, matrix_386324], **kwargs_386325)
    
    # Applying the binary operator 'or' (line 673)
    result_or_keyword_386327 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 9), 'or', isinstance_call_result_386320, isinstance_call_result_386326)
    
    # Testing the type of an if condition (line 673)
    if_condition_386328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 673, 9), result_or_keyword_386327)
    # Assigning a type to the variable 'if_condition_386328' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 9), 'if_condition_386328', if_condition_386328)
    # SSA begins for if statement (line 673)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'A' (line 674)
    A_386329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 11), 'A')
    # Obtaining the member 'ndim' of a type (line 674)
    ndim_386330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 11), A_386329, 'ndim')
    int_386331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 20), 'int')
    # Applying the binary operator '>' (line 674)
    result_gt_386332 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 11), '>', ndim_386330, int_386331)
    
    # Testing the type of an if condition (line 674)
    if_condition_386333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 8), result_gt_386332)
    # Assigning a type to the variable 'if_condition_386333' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'if_condition_386333', if_condition_386333)
    # SSA begins for if statement (line 674)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 675)
    # Processing the call arguments (line 675)
    str_386335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 29), 'str', 'array must have ndim <= 2')
    # Processing the call keyword arguments (line 675)
    kwargs_386336 = {}
    # Getting the type of 'ValueError' (line 675)
    ValueError_386334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 675)
    ValueError_call_result_386337 = invoke(stypy.reporting.localization.Localization(__file__, 675, 18), ValueError_386334, *[str_386335], **kwargs_386336)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 675, 12), ValueError_call_result_386337, 'raise parameter', BaseException)
    # SSA join for if statement (line 674)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 676):
    
    # Assigning a Call to a Name (line 676):
    
    # Call to atleast_2d(...): (line 676)
    # Processing the call arguments (line 676)
    
    # Call to asarray(...): (line 676)
    # Processing the call arguments (line 676)
    # Getting the type of 'A' (line 676)
    A_386342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 37), 'A', False)
    # Processing the call keyword arguments (line 676)
    kwargs_386343 = {}
    # Getting the type of 'np' (line 676)
    np_386340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 26), 'np', False)
    # Obtaining the member 'asarray' of a type (line 676)
    asarray_386341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 26), np_386340, 'asarray')
    # Calling asarray(args, kwargs) (line 676)
    asarray_call_result_386344 = invoke(stypy.reporting.localization.Localization(__file__, 676, 26), asarray_386341, *[A_386342], **kwargs_386343)
    
    # Processing the call keyword arguments (line 676)
    kwargs_386345 = {}
    # Getting the type of 'np' (line 676)
    np_386338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 676)
    atleast_2d_386339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 12), np_386338, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 676)
    atleast_2d_call_result_386346 = invoke(stypy.reporting.localization.Localization(__file__, 676, 12), atleast_2d_386339, *[asarray_call_result_386344], **kwargs_386345)
    
    # Assigning a type to the variable 'A' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'A', atleast_2d_call_result_386346)
    
    # Call to MatrixLinearOperator(...): (line 677)
    # Processing the call arguments (line 677)
    # Getting the type of 'A' (line 677)
    A_386348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 36), 'A', False)
    # Processing the call keyword arguments (line 677)
    kwargs_386349 = {}
    # Getting the type of 'MatrixLinearOperator' (line 677)
    MatrixLinearOperator_386347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 15), 'MatrixLinearOperator', False)
    # Calling MatrixLinearOperator(args, kwargs) (line 677)
    MatrixLinearOperator_call_result_386350 = invoke(stypy.reporting.localization.Localization(__file__, 677, 15), MatrixLinearOperator_386347, *[A_386348], **kwargs_386349)
    
    # Assigning a type to the variable 'stypy_return_type' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'stypy_return_type', MatrixLinearOperator_call_result_386350)
    # SSA branch for the else part of an if statement (line 673)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isspmatrix(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'A' (line 679)
    A_386352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 20), 'A', False)
    # Processing the call keyword arguments (line 679)
    kwargs_386353 = {}
    # Getting the type of 'isspmatrix' (line 679)
    isspmatrix_386351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 9), 'isspmatrix', False)
    # Calling isspmatrix(args, kwargs) (line 679)
    isspmatrix_call_result_386354 = invoke(stypy.reporting.localization.Localization(__file__, 679, 9), isspmatrix_386351, *[A_386352], **kwargs_386353)
    
    # Testing the type of an if condition (line 679)
    if_condition_386355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 679, 9), isspmatrix_call_result_386354)
    # Assigning a type to the variable 'if_condition_386355' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 9), 'if_condition_386355', if_condition_386355)
    # SSA begins for if statement (line 679)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to MatrixLinearOperator(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of 'A' (line 680)
    A_386357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 36), 'A', False)
    # Processing the call keyword arguments (line 680)
    kwargs_386358 = {}
    # Getting the type of 'MatrixLinearOperator' (line 680)
    MatrixLinearOperator_386356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 15), 'MatrixLinearOperator', False)
    # Calling MatrixLinearOperator(args, kwargs) (line 680)
    MatrixLinearOperator_call_result_386359 = invoke(stypy.reporting.localization.Localization(__file__, 680, 15), MatrixLinearOperator_386356, *[A_386357], **kwargs_386358)
    
    # Assigning a type to the variable 'stypy_return_type' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'stypy_return_type', MatrixLinearOperator_call_result_386359)
    # SSA branch for the else part of an if statement (line 679)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'A' (line 683)
    A_386361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 19), 'A', False)
    str_386362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 22), 'str', 'shape')
    # Processing the call keyword arguments (line 683)
    kwargs_386363 = {}
    # Getting the type of 'hasattr' (line 683)
    hasattr_386360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 11), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 683)
    hasattr_call_result_386364 = invoke(stypy.reporting.localization.Localization(__file__, 683, 11), hasattr_386360, *[A_386361, str_386362], **kwargs_386363)
    
    
    # Call to hasattr(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'A' (line 683)
    A_386366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 43), 'A', False)
    str_386367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 46), 'str', 'matvec')
    # Processing the call keyword arguments (line 683)
    kwargs_386368 = {}
    # Getting the type of 'hasattr' (line 683)
    hasattr_386365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 35), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 683)
    hasattr_call_result_386369 = invoke(stypy.reporting.localization.Localization(__file__, 683, 35), hasattr_386365, *[A_386366, str_386367], **kwargs_386368)
    
    # Applying the binary operator 'and' (line 683)
    result_and_keyword_386370 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 11), 'and', hasattr_call_result_386364, hasattr_call_result_386369)
    
    # Testing the type of an if condition (line 683)
    if_condition_386371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 8), result_and_keyword_386370)
    # Assigning a type to the variable 'if_condition_386371' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'if_condition_386371', if_condition_386371)
    # SSA begins for if statement (line 683)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 684):
    
    # Assigning a Name to a Name (line 684):
    # Getting the type of 'None' (line 684)
    None_386372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 22), 'None')
    # Assigning a type to the variable 'rmatvec' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'rmatvec', None_386372)
    
    # Assigning a Name to a Name (line 685):
    
    # Assigning a Name to a Name (line 685):
    # Getting the type of 'None' (line 685)
    None_386373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 20), 'None')
    # Assigning a type to the variable 'dtype' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 12), 'dtype', None_386373)
    
    # Type idiom detected: calculating its left and rigth part (line 687)
    str_386374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 26), 'str', 'rmatvec')
    # Getting the type of 'A' (line 687)
    A_386375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 23), 'A')
    
    (may_be_386376, more_types_in_union_386377) = may_provide_member(str_386374, A_386375)

    if may_be_386376:

        if more_types_in_union_386377:
            # Runtime conditional SSA (line 687)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'A' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 12), 'A', remove_not_member_provider_from_union(A_386375, 'rmatvec'))
        
        # Assigning a Attribute to a Name (line 688):
        
        # Assigning a Attribute to a Name (line 688):
        # Getting the type of 'A' (line 688)
        A_386378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 26), 'A')
        # Obtaining the member 'rmatvec' of a type (line 688)
        rmatvec_386379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 26), A_386378, 'rmatvec')
        # Assigning a type to the variable 'rmatvec' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'rmatvec', rmatvec_386379)

        if more_types_in_union_386377:
            # SSA join for if statement (line 687)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 689)
    str_386380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 26), 'str', 'dtype')
    # Getting the type of 'A' (line 689)
    A_386381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 23), 'A')
    
    (may_be_386382, more_types_in_union_386383) = may_provide_member(str_386380, A_386381)

    if may_be_386382:

        if more_types_in_union_386383:
            # Runtime conditional SSA (line 689)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'A' (line 689)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 12), 'A', remove_not_member_provider_from_union(A_386381, 'dtype'))
        
        # Assigning a Attribute to a Name (line 690):
        
        # Assigning a Attribute to a Name (line 690):
        # Getting the type of 'A' (line 690)
        A_386384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 24), 'A')
        # Obtaining the member 'dtype' of a type (line 690)
        dtype_386385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 24), A_386384, 'dtype')
        # Assigning a type to the variable 'dtype' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'dtype', dtype_386385)

        if more_types_in_union_386383:
            # SSA join for if statement (line 689)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to LinearOperator(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'A' (line 691)
    A_386387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 34), 'A', False)
    # Obtaining the member 'shape' of a type (line 691)
    shape_386388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 34), A_386387, 'shape')
    # Getting the type of 'A' (line 691)
    A_386389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 43), 'A', False)
    # Obtaining the member 'matvec' of a type (line 691)
    matvec_386390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 43), A_386389, 'matvec')
    # Processing the call keyword arguments (line 691)
    # Getting the type of 'rmatvec' (line 692)
    rmatvec_386391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 42), 'rmatvec', False)
    keyword_386392 = rmatvec_386391
    # Getting the type of 'dtype' (line 692)
    dtype_386393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 57), 'dtype', False)
    keyword_386394 = dtype_386393
    kwargs_386395 = {'dtype': keyword_386394, 'rmatvec': keyword_386392}
    # Getting the type of 'LinearOperator' (line 691)
    LinearOperator_386386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 19), 'LinearOperator', False)
    # Calling LinearOperator(args, kwargs) (line 691)
    LinearOperator_call_result_386396 = invoke(stypy.reporting.localization.Localization(__file__, 691, 19), LinearOperator_386386, *[shape_386388, matvec_386390], **kwargs_386395)
    
    # Assigning a type to the variable 'stypy_return_type' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'stypy_return_type', LinearOperator_call_result_386396)
    # SSA branch for the else part of an if statement (line 683)
    module_type_store.open_ssa_branch('else')
    
    # Call to TypeError(...): (line 695)
    # Processing the call arguments (line 695)
    str_386398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 28), 'str', 'type not understood')
    # Processing the call keyword arguments (line 695)
    kwargs_386399 = {}
    # Getting the type of 'TypeError' (line 695)
    TypeError_386397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 695)
    TypeError_call_result_386400 = invoke(stypy.reporting.localization.Localization(__file__, 695, 18), TypeError_386397, *[str_386398], **kwargs_386399)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 695, 12), TypeError_call_result_386400, 'raise parameter', BaseException)
    # SSA join for if statement (line 683)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 679)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 673)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 670)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'aslinearoperator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'aslinearoperator' in the type store
    # Getting the type of 'stypy_return_type' (line 650)
    stypy_return_type_386401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_386401)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'aslinearoperator'
    return stypy_return_type_386401

# Assigning a type to the variable 'aslinearoperator' (line 650)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 0), 'aslinearoperator', aslinearoperator)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
