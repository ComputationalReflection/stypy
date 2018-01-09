
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Base class for sparse matrices'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import sys
5: 
6: import numpy as np
7: 
8: from scipy._lib.six import xrange
9: from scipy._lib._numpy_compat import broadcast_to
10: from .sputils import (isdense, isscalarlike, isintlike,
11:                       get_sum_dtype, validateaxis)
12: 
13: __all__ = ['spmatrix', 'isspmatrix', 'issparse',
14:            'SparseWarning', 'SparseEfficiencyWarning']
15: 
16: 
17: class SparseWarning(Warning):
18:     pass
19: 
20: 
21: class SparseFormatWarning(SparseWarning):
22:     pass
23: 
24: 
25: class SparseEfficiencyWarning(SparseWarning):
26:     pass
27: 
28: 
29: # The formats that we might potentially understand.
30: _formats = {'csc': [0, "Compressed Sparse Column"],
31:             'csr': [1, "Compressed Sparse Row"],
32:             'dok': [2, "Dictionary Of Keys"],
33:             'lil': [3, "LInked List"],
34:             'dod': [4, "Dictionary of Dictionaries"],
35:             'sss': [5, "Symmetric Sparse Skyline"],
36:             'coo': [6, "COOrdinate"],
37:             'lba': [7, "Linpack BAnded"],
38:             'egd': [8, "Ellpack-itpack Generalized Diagonal"],
39:             'dia': [9, "DIAgonal"],
40:             'bsr': [10, "Block Sparse Row"],
41:             'msr': [11, "Modified compressed Sparse Row"],
42:             'bsc': [12, "Block Sparse Column"],
43:             'msc': [13, "Modified compressed Sparse Column"],
44:             'ssk': [14, "Symmetric SKyline"],
45:             'nsk': [15, "Nonsymmetric SKyline"],
46:             'jad': [16, "JAgged Diagonal"],
47:             'uss': [17, "Unsymmetric Sparse Skyline"],
48:             'vbr': [18, "Variable Block Row"],
49:             'und': [19, "Undefined"]
50:             }
51: 
52: 
53: # These univariate ufuncs preserve zeros.
54: _ufuncs_with_fixed_point_at_zero = frozenset([
55:         np.sin, np.tan, np.arcsin, np.arctan, np.sinh, np.tanh, np.arcsinh,
56:         np.arctanh, np.rint, np.sign, np.expm1, np.log1p, np.deg2rad,
57:         np.rad2deg, np.floor, np.ceil, np.trunc, np.sqrt])
58: 
59: 
60: MAXPRINT = 50
61: 
62: 
63: class spmatrix(object):
64:     ''' This class provides a base class for all sparse matrices.  It
65:     cannot be instantiated.  Most of the work is provided by subclasses.
66:     '''
67: 
68:     __array_priority__ = 10.1
69:     ndim = 2
70: 
71:     def __init__(self, maxprint=MAXPRINT):
72:         self._shape = None
73:         if self.__class__.__name__ == 'spmatrix':
74:             raise ValueError("This class is not intended"
75:                              " to be instantiated directly.")
76:         self.maxprint = maxprint
77: 
78:     def set_shape(self, shape):
79:         '''See `reshape`.'''
80:         shape = tuple(shape)
81: 
82:         if len(shape) != 2:
83:             raise ValueError("Only two-dimensional sparse "
84:                              "arrays are supported.")
85:         try:
86:             shape = int(shape[0]), int(shape[1])  # floats, other weirdness
87:         except:
88:             raise TypeError('invalid shape')
89: 
90:         if not (shape[0] >= 0 and shape[1] >= 0):
91:             raise ValueError('invalid shape')
92: 
93:         if (self._shape != shape) and (self._shape is not None):
94:             try:
95:                 self = self.reshape(shape)
96:             except NotImplementedError:
97:                 raise NotImplementedError("Reshaping not implemented for %s." %
98:                                           self.__class__.__name__)
99:         self._shape = shape
100: 
101:     def get_shape(self):
102:         '''Get shape of a matrix.'''
103:         return self._shape
104: 
105:     shape = property(fget=get_shape, fset=set_shape)
106: 
107:     def reshape(self, shape, order='C'):
108:         '''
109:         Gives a new shape to a sparse matrix without changing its data.
110: 
111:         Parameters
112:         ----------
113:         shape : length-2 tuple of ints
114:             The new shape should be compatible with the original shape.
115:         order : 'C', optional
116:             This argument is in the signature *solely* for NumPy
117:             compatibility reasons. Do not pass in anything except
118:             for the default value, as this argument is not used.
119: 
120:         Returns
121:         -------
122:         reshaped_matrix : `self` with the new dimensions of `shape`
123: 
124:         See Also
125:         --------
126:         np.matrix.reshape : NumPy's implementation of 'reshape' for matrices
127:         '''
128:         raise NotImplementedError("Reshaping not implemented for %s." %
129:                                   self.__class__.__name__)
130: 
131:     def astype(self, dtype, casting='unsafe', copy=True):
132:         '''Cast the matrix elements to a specified type.
133: 
134:         Parameters
135:         ----------
136:         dtype : string or numpy dtype
137:             Typecode or data-type to which to cast the data.
138:         casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
139:             Controls what kind of data casting may occur.
140:             Defaults to 'unsafe' for backwards compatibility.
141:             'no' means the data types should not be cast at all.
142:             'equiv' means only byte-order changes are allowed.
143:             'safe' means only casts which can preserve values are allowed.
144:             'same_kind' means only safe casts or casts within a kind,
145:             like float64 to float32, are allowed.
146:             'unsafe' means any data conversions may be done.
147:         copy : bool, optional
148:             If `copy` is `False`, the result might share some memory with this
149:             matrix. If `copy` is `True`, it is guaranteed that the result and
150:             this matrix do not share any memory.
151:         '''
152: 
153:         dtype = np.dtype(dtype)
154:         if self.dtype != dtype:
155:             return self.tocsr().astype(
156:                 dtype, casting=casting, copy=copy).asformat(self.format)
157:         elif copy:
158:             return self.copy()
159:         else:
160:             return self
161: 
162:     def asfptype(self):
163:         '''Upcast matrix to a floating point format (if necessary)'''
164: 
165:         fp_types = ['f', 'd', 'F', 'D']
166: 
167:         if self.dtype.char in fp_types:
168:             return self
169:         else:
170:             for fp_type in fp_types:
171:                 if self.dtype <= np.dtype(fp_type):
172:                     return self.astype(fp_type)
173: 
174:             raise TypeError('cannot upcast [%s] to a floating '
175:                             'point format' % self.dtype.name)
176: 
177:     def __iter__(self):
178:         for r in xrange(self.shape[0]):
179:             yield self[r, :]
180: 
181:     def getmaxprint(self):
182:         '''Maximum number of elements to display when printed.'''
183:         return self.maxprint
184: 
185:     def count_nonzero(self):
186:         '''Number of non-zero entries, equivalent to
187: 
188:         np.count_nonzero(a.toarray())
189: 
190:         Unlike getnnz() and the nnz property, which return the number of stored
191:         entries (the length of the data attribute), this method counts the
192:         actual number of non-zero entries in data.
193:         '''
194:         raise NotImplementedError("count_nonzero not implemented for %s." %
195:                                   self.__class__.__name__)
196: 
197:     def getnnz(self, axis=None):
198:         '''Number of stored values, including explicit zeros.
199: 
200:         Parameters
201:         ----------
202:         axis : None, 0, or 1
203:             Select between the number of values across the whole matrix, in
204:             each column, or in each row.
205: 
206:         See also
207:         --------
208:         count_nonzero : Number of non-zero entries
209:         '''
210:         raise NotImplementedError("getnnz not implemented for %s." %
211:                                   self.__class__.__name__)
212: 
213:     @property
214:     def nnz(self):
215:         '''Number of stored values, including explicit zeros.
216: 
217:         See also
218:         --------
219:         count_nonzero : Number of non-zero entries
220:         '''
221:         return self.getnnz()
222: 
223:     def getformat(self):
224:         '''Format of a matrix representation as a string.'''
225:         return getattr(self, 'format', 'und')
226: 
227:     def __repr__(self):
228:         _, format_name = _formats[self.getformat()]
229:         return "<%dx%d sparse matrix of type '%s'\n" \
230:                "\twith %d stored elements in %s format>" % \
231:                (self.shape + (self.dtype.type, self.nnz, format_name))
232: 
233:     def __str__(self):
234:         maxprint = self.getmaxprint()
235: 
236:         A = self.tocoo()
237: 
238:         # helper function, outputs "(i,j)  v"
239:         def tostr(row, col, data):
240:             triples = zip(list(zip(row, col)), data)
241:             return '\n'.join([('  %s\t%s' % t) for t in triples])
242: 
243:         if self.nnz > maxprint:
244:             half = maxprint // 2
245:             out = tostr(A.row[:half], A.col[:half], A.data[:half])
246:             out += "\n  :\t:\n"
247:             half = maxprint - maxprint//2
248:             out += tostr(A.row[-half:], A.col[-half:], A.data[-half:])
249:         else:
250:             out = tostr(A.row, A.col, A.data)
251: 
252:         return out
253: 
254:     def __bool__(self):  # Simple -- other ideas?
255:         if self.shape == (1, 1):
256:             return self.nnz != 0
257:         else:
258:             raise ValueError("The truth value of an array with more than one "
259:                              "element is ambiguous. Use a.any() or a.all().")
260:     __nonzero__ = __bool__
261: 
262:     # What should len(sparse) return? For consistency with dense matrices,
263:     # perhaps it should be the number of rows?  But for some uses the number of
264:     # non-zeros is more important.  For now, raise an exception!
265:     def __len__(self):
266:         raise TypeError("sparse matrix length is ambiguous; use getnnz()"
267:                         " or shape[0]")
268: 
269:     def asformat(self, format):
270:         '''Return this matrix in a given sparse format
271: 
272:         Parameters
273:         ----------
274:         format : {string, None}
275:             desired sparse matrix format
276:                 - None for no format conversion
277:                 - "csr" for csr_matrix format
278:                 - "csc" for csc_matrix format
279:                 - "lil" for lil_matrix format
280:                 - "dok" for dok_matrix format and so on
281: 
282:         '''
283: 
284:         if format is None or format == self.format:
285:             return self
286:         else:
287:             return getattr(self, 'to' + format)()
288: 
289:     ###################################################################
290:     #  NOTE: All arithmetic operations use csr_matrix by default.
291:     # Therefore a new sparse matrix format just needs to define a
292:     # .tocsr() method to provide arithmetic support.  Any of these
293:     # methods can be overridden for efficiency.
294:     ####################################################################
295: 
296:     def multiply(self, other):
297:         '''Point-wise multiplication by another matrix
298:         '''
299:         return self.tocsr().multiply(other)
300: 
301:     def maximum(self, other):
302:         '''Element-wise maximum between this and another matrix.'''
303:         return self.tocsr().maximum(other)
304: 
305:     def minimum(self, other):
306:         '''Element-wise minimum between this and another matrix.'''
307:         return self.tocsr().minimum(other)
308: 
309:     def dot(self, other):
310:         '''Ordinary dot product
311: 
312:         Examples
313:         --------
314:         >>> import numpy as np
315:         >>> from scipy.sparse import csr_matrix
316:         >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
317:         >>> v = np.array([1, 0, -1])
318:         >>> A.dot(v)
319:         array([ 1, -3, -1], dtype=int64)
320: 
321:         '''
322:         return self * other
323: 
324:     def power(self, n, dtype=None):
325:         '''Element-wise power.'''
326:         return self.tocsr().power(n, dtype=dtype)
327: 
328:     def __eq__(self, other):
329:         return self.tocsr().__eq__(other)
330: 
331:     def __ne__(self, other):
332:         return self.tocsr().__ne__(other)
333: 
334:     def __lt__(self, other):
335:         return self.tocsr().__lt__(other)
336: 
337:     def __gt__(self, other):
338:         return self.tocsr().__gt__(other)
339: 
340:     def __le__(self, other):
341:         return self.tocsr().__le__(other)
342: 
343:     def __ge__(self, other):
344:         return self.tocsr().__ge__(other)
345: 
346:     def __abs__(self):
347:         return abs(self.tocsr())
348: 
349:     def _add_sparse(self, other):
350:         return self.tocsr()._add_sparse(other)
351: 
352:     def _add_dense(self, other):
353:         return self.tocoo()._add_dense(other)
354: 
355:     def _sub_sparse(self, other):
356:         return self.tocsr()._sub_sparse(other)
357: 
358:     def _sub_dense(self, other):
359:         return self.todense() - other
360: 
361:     def _rsub_dense(self, other):
362:         # note: this can't be replaced by other + (-self) for unsigned types
363:         return other - self.todense()
364: 
365:     def __add__(self, other):  # self + other
366:         if isscalarlike(other):
367:             if other == 0:
368:                 return self.copy()
369:             # Now we would add this scalar to every element.
370:             raise NotImplementedError('adding a nonzero scalar to a '
371:                                       'sparse matrix is not supported')
372:         elif isspmatrix(other):
373:             if other.shape != self.shape:
374:                 raise ValueError("inconsistent shapes")
375:             return self._add_sparse(other)
376:         elif isdense(other):
377:             other = broadcast_to(other, self.shape)
378:             return self._add_dense(other)
379:         else:
380:             return NotImplemented
381: 
382:     def __radd__(self,other):  # other + self
383:         return self.__add__(other)
384: 
385:     def __sub__(self, other):  # self - other
386:         if isscalarlike(other):
387:             if other == 0:
388:                 return self.copy()
389:             raise NotImplementedError('subtracting a nonzero scalar from a '
390:                                       'sparse matrix is not supported')
391:         elif isspmatrix(other):
392:             if other.shape != self.shape:
393:                 raise ValueError("inconsistent shapes")
394:             return self._sub_sparse(other)
395:         elif isdense(other):
396:             other = broadcast_to(other, self.shape)
397:             return self._sub_dense(other)
398:         else:
399:             return NotImplemented
400: 
401:     def __rsub__(self,other):  # other - self
402:         if isscalarlike(other):
403:             if other == 0:
404:                 return -self.copy()
405:             raise NotImplementedError('subtracting a sparse matrix from a '
406:                                       'nonzero scalar is not supported')
407:         elif isdense(other):
408:             other = broadcast_to(other, self.shape)
409:             return self._rsub_dense(other)
410:         else:
411:             return NotImplemented
412: 
413:     def __mul__(self, other):
414:         '''interpret other and call one of the following
415: 
416:         self._mul_scalar()
417:         self._mul_vector()
418:         self._mul_multivector()
419:         self._mul_sparse_matrix()
420:         '''
421: 
422:         M, N = self.shape
423: 
424:         if other.__class__ is np.ndarray:
425:             # Fast path for the most common case
426:             if other.shape == (N,):
427:                 return self._mul_vector(other)
428:             elif other.shape == (N, 1):
429:                 return self._mul_vector(other.ravel()).reshape(M, 1)
430:             elif other.ndim == 2 and other.shape[0] == N:
431:                 return self._mul_multivector(other)
432: 
433:         if isscalarlike(other):
434:             # scalar value
435:             return self._mul_scalar(other)
436: 
437:         if issparse(other):
438:             if self.shape[1] != other.shape[0]:
439:                 raise ValueError('dimension mismatch')
440:             return self._mul_sparse_matrix(other)
441: 
442:         # If it's a list or whatever, treat it like a matrix
443:         other_a = np.asanyarray(other)
444: 
445:         if other_a.ndim == 0 and other_a.dtype == np.object_:
446:             # Not interpretable as an array; return NotImplemented so that
447:             # other's __rmul__ can kick in if that's implemented.
448:             return NotImplemented
449: 
450:         try:
451:             other.shape
452:         except AttributeError:
453:             other = other_a
454: 
455:         if other.ndim == 1 or other.ndim == 2 and other.shape[1] == 1:
456:             # dense row or column vector
457:             if other.shape != (N,) and other.shape != (N, 1):
458:                 raise ValueError('dimension mismatch')
459: 
460:             result = self._mul_vector(np.ravel(other))
461: 
462:             if isinstance(other, np.matrix):
463:                 result = np.asmatrix(result)
464: 
465:             if other.ndim == 2 and other.shape[1] == 1:
466:                 # If 'other' was an (nx1) column vector, reshape the result
467:                 result = result.reshape(-1, 1)
468: 
469:             return result
470: 
471:         elif other.ndim == 2:
472:             ##
473:             # dense 2D array or matrix ("multivector")
474: 
475:             if other.shape[0] != self.shape[1]:
476:                 raise ValueError('dimension mismatch')
477: 
478:             result = self._mul_multivector(np.asarray(other))
479: 
480:             if isinstance(other, np.matrix):
481:                 result = np.asmatrix(result)
482: 
483:             return result
484: 
485:         else:
486:             raise ValueError('could not interpret dimensions')
487: 
488:     # by default, use CSR for __mul__ handlers
489:     def _mul_scalar(self, other):
490:         return self.tocsr()._mul_scalar(other)
491: 
492:     def _mul_vector(self, other):
493:         return self.tocsr()._mul_vector(other)
494: 
495:     def _mul_multivector(self, other):
496:         return self.tocsr()._mul_multivector(other)
497: 
498:     def _mul_sparse_matrix(self, other):
499:         return self.tocsr()._mul_sparse_matrix(other)
500: 
501:     def __rmul__(self, other):  # other * self
502:         if isscalarlike(other):
503:             return self.__mul__(other)
504:         else:
505:             # Don't use asarray unless we have to
506:             try:
507:                 tr = other.transpose()
508:             except AttributeError:
509:                 tr = np.asarray(other).transpose()
510:             return (self.transpose() * tr).transpose()
511: 
512:     #####################################
513:     # matmul (@) operator (Python 3.5+) #
514:     #####################################
515: 
516:     def __matmul__(self, other):
517:         if isscalarlike(other):
518:             raise ValueError("Scalar operands are not allowed, "
519:                              "use '*' instead")
520:         return self.__mul__(other)
521: 
522:     def __rmatmul__(self, other):
523:         if isscalarlike(other):
524:             raise ValueError("Scalar operands are not allowed, "
525:                              "use '*' instead")
526:         return self.__rmul__(other)
527: 
528:     ####################
529:     # Other Arithmetic #
530:     ####################
531: 
532:     def _divide(self, other, true_divide=False, rdivide=False):
533:         if isscalarlike(other):
534:             if rdivide:
535:                 if true_divide:
536:                     return np.true_divide(other, self.todense())
537:                 else:
538:                     return np.divide(other, self.todense())
539: 
540:             if true_divide and np.can_cast(self.dtype, np.float_):
541:                 return self.astype(np.float_)._mul_scalar(1./other)
542:             else:
543:                 r = self._mul_scalar(1./other)
544: 
545:                 scalar_dtype = np.asarray(other).dtype
546:                 if (np.issubdtype(self.dtype, np.integer) and
547:                         np.issubdtype(scalar_dtype, np.integer)):
548:                     return r.astype(self.dtype)
549:                 else:
550:                     return r
551: 
552:         elif isdense(other):
553:             if not rdivide:
554:                 if true_divide:
555:                     return np.true_divide(self.todense(), other)
556:                 else:
557:                     return np.divide(self.todense(), other)
558:             else:
559:                 if true_divide:
560:                     return np.true_divide(other, self.todense())
561:                 else:
562:                     return np.divide(other, self.todense())
563:         elif isspmatrix(other):
564:             if rdivide:
565:                 return other._divide(self, true_divide, rdivide=False)
566: 
567:             self_csr = self.tocsr()
568:             if true_divide and np.can_cast(self.dtype, np.float_):
569:                 return self_csr.astype(np.float_)._divide_sparse(other)
570:             else:
571:                 return self_csr._divide_sparse(other)
572:         else:
573:             return NotImplemented
574: 
575:     def __truediv__(self, other):
576:         return self._divide(other, true_divide=True)
577: 
578:     def __div__(self, other):
579:         # Always do true division
580:         return self._divide(other, true_divide=True)
581: 
582:     def __rtruediv__(self, other):
583:         # Implementing this as the inverse would be too magical -- bail out
584:         return NotImplemented
585: 
586:     def __rdiv__(self, other):
587:         # Implementing this as the inverse would be too magical -- bail out
588:         return NotImplemented
589: 
590:     def __neg__(self):
591:         return -self.tocsr()
592: 
593:     def __iadd__(self, other):
594:         return NotImplemented
595: 
596:     def __isub__(self, other):
597:         return NotImplemented
598: 
599:     def __imul__(self, other):
600:         return NotImplemented
601: 
602:     def __idiv__(self, other):
603:         return self.__itruediv__(other)
604: 
605:     def __itruediv__(self, other):
606:         return NotImplemented
607: 
608:     def __pow__(self, other):
609:         if self.shape[0] != self.shape[1]:
610:             raise TypeError('matrix is not square')
611: 
612:         if isintlike(other):
613:             other = int(other)
614:             if other < 0:
615:                 raise ValueError('exponent must be >= 0')
616: 
617:             if other == 0:
618:                 from .construct import eye
619:                 return eye(self.shape[0], dtype=self.dtype)
620:             elif other == 1:
621:                 return self.copy()
622:             else:
623:                 tmp = self.__pow__(other//2)
624:                 if (other % 2):
625:                     return self * tmp * tmp
626:                 else:
627:                     return tmp * tmp
628:         elif isscalarlike(other):
629:             raise ValueError('exponent must be an integer')
630:         else:
631:             return NotImplemented
632: 
633:     def __getattr__(self, attr):
634:         if attr == 'A':
635:             return self.toarray()
636:         elif attr == 'T':
637:             return self.transpose()
638:         elif attr == 'H':
639:             return self.getH()
640:         elif attr == 'real':
641:             return self._real()
642:         elif attr == 'imag':
643:             return self._imag()
644:         elif attr == 'size':
645:             return self.getnnz()
646:         else:
647:             raise AttributeError(attr + " not found")
648: 
649:     def transpose(self, axes=None, copy=False):
650:         '''
651:         Reverses the dimensions of the sparse matrix.
652: 
653:         Parameters
654:         ----------
655:         axes : None, optional
656:             This argument is in the signature *solely* for NumPy
657:             compatibility reasons. Do not pass in anything except
658:             for the default value.
659:         copy : bool, optional
660:             Indicates whether or not attributes of `self` should be
661:             copied whenever possible. The degree to which attributes
662:             are copied varies depending on the type of sparse matrix
663:             being used.
664: 
665:         Returns
666:         -------
667:         p : `self` with the dimensions reversed.
668: 
669:         See Also
670:         --------
671:         np.matrix.transpose : NumPy's implementation of 'transpose'
672:                               for matrices
673:         '''
674:         return self.tocsr().transpose(axes=axes, copy=copy)
675: 
676:     def conj(self):
677:         '''Element-wise complex conjugation.
678: 
679:         If the matrix is of non-complex data type, then this method does
680:         nothing and the data is not copied.
681:         '''
682:         return self.tocsr().conj()
683: 
684:     def conjugate(self):
685:         return self.conj()
686: 
687:     conjugate.__doc__ = conj.__doc__
688: 
689:     # Renamed conjtranspose() -> getH() for compatibility with dense matrices
690:     def getH(self):
691:         '''Return the Hermitian transpose of this matrix.
692: 
693:         See Also
694:         --------
695:         np.matrix.getH : NumPy's implementation of `getH` for matrices
696:         '''
697:         return self.transpose().conj()
698: 
699:     def _real(self):
700:         return self.tocsr()._real()
701: 
702:     def _imag(self):
703:         return self.tocsr()._imag()
704: 
705:     def nonzero(self):
706:         '''nonzero indices
707: 
708:         Returns a tuple of arrays (row,col) containing the indices
709:         of the non-zero elements of the matrix.
710: 
711:         Examples
712:         --------
713:         >>> from scipy.sparse import csr_matrix
714:         >>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
715:         >>> A.nonzero()
716:         (array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))
717: 
718:         '''
719: 
720:         # convert to COOrdinate format
721:         A = self.tocoo()
722:         nz_mask = A.data != 0
723:         return (A.row[nz_mask], A.col[nz_mask])
724: 
725:     def getcol(self, j):
726:         '''Returns a copy of column j of the matrix, as an (m x 1) sparse
727:         matrix (column vector).
728:         '''
729:         # Spmatrix subclasses should override this method for efficiency.
730:         # Post-multiply by a (n x 1) column vector 'a' containing all zeros
731:         # except for a_j = 1
732:         from .csc import csc_matrix
733:         n = self.shape[1]
734:         if j < 0:
735:             j += n
736:         if j < 0 or j >= n:
737:             raise IndexError("index out of bounds")
738:         col_selector = csc_matrix(([1], [[j], [0]]),
739:                                   shape=(n, 1), dtype=self.dtype)
740:         return self * col_selector
741: 
742:     def getrow(self, i):
743:         '''Returns a copy of row i of the matrix, as a (1 x n) sparse
744:         matrix (row vector).
745:         '''
746:         # Spmatrix subclasses should override this method for efficiency.
747:         # Pre-multiply by a (1 x m) row vector 'a' containing all zeros
748:         # except for a_i = 1
749:         from .csr import csr_matrix
750:         m = self.shape[0]
751:         if i < 0:
752:             i += m
753:         if i < 0 or i >= m:
754:             raise IndexError("index out of bounds")
755:         row_selector = csr_matrix(([1], [[0], [i]]),
756:                                   shape=(1, m), dtype=self.dtype)
757:         return row_selector * self
758: 
759:     # def __array__(self):
760:     #    return self.toarray()
761: 
762:     def todense(self, order=None, out=None):
763:         '''
764:         Return a dense matrix representation of this matrix.
765: 
766:         Parameters
767:         ----------
768:         order : {'C', 'F'}, optional
769:             Whether to store multi-dimensional data in C (row-major)
770:             or Fortran (column-major) order in memory. The default
771:             is 'None', indicating the NumPy default of C-ordered.
772:             Cannot be specified in conjunction with the `out`
773:             argument.
774: 
775:         out : ndarray, 2-dimensional, optional
776:             If specified, uses this array (or `numpy.matrix`) as the
777:             output buffer instead of allocating a new array to
778:             return. The provided array must have the same shape and
779:             dtype as the sparse matrix on which you are calling the
780:             method.
781: 
782:         Returns
783:         -------
784:         arr : numpy.matrix, 2-dimensional
785:             A NumPy matrix object with the same shape and containing
786:             the same data represented by the sparse matrix, with the
787:             requested memory order. If `out` was passed and was an
788:             array (rather than a `numpy.matrix`), it will be filled
789:             with the appropriate values and returned wrapped in a
790:             `numpy.matrix` object that shares the same memory.
791:         '''
792:         return np.asmatrix(self.toarray(order=order, out=out))
793: 
794:     def toarray(self, order=None, out=None):
795:         '''
796:         Return a dense ndarray representation of this matrix.
797: 
798:         Parameters
799:         ----------
800:         order : {'C', 'F'}, optional
801:             Whether to store multi-dimensional data in C (row-major)
802:             or Fortran (column-major) order in memory. The default
803:             is 'None', indicating the NumPy default of C-ordered.
804:             Cannot be specified in conjunction with the `out`
805:             argument.
806: 
807:         out : ndarray, 2-dimensional, optional
808:             If specified, uses this array as the output buffer
809:             instead of allocating a new array to return. The provided
810:             array must have the same shape and dtype as the sparse
811:             matrix on which you are calling the method. For most
812:             sparse types, `out` is required to be memory contiguous
813:             (either C or Fortran ordered).
814: 
815:         Returns
816:         -------
817:         arr : ndarray, 2-dimensional
818:             An array with the same shape and containing the same
819:             data represented by the sparse matrix, with the requested
820:             memory order. If `out` was passed, the same object is
821:             returned after being modified in-place to contain the
822:             appropriate values.
823:         '''
824:         return self.tocoo(copy=False).toarray(order=order, out=out)
825: 
826:     # Any sparse matrix format deriving from spmatrix must define one of
827:     # tocsr or tocoo. The other conversion methods may be implemented for
828:     # efficiency, but are not required.
829:     def tocsr(self, copy=False):
830:         '''Convert this matrix to Compressed Sparse Row format.
831: 
832:         With copy=False, the data/indices may be shared between this matrix and
833:         the resultant csr_matrix.
834:         '''
835:         return self.tocoo(copy=copy).tocsr(copy=False)
836: 
837:     def todok(self, copy=False):
838:         '''Convert this matrix to Dictionary Of Keys format.
839: 
840:         With copy=False, the data/indices may be shared between this matrix and
841:         the resultant dok_matrix.
842:         '''
843:         return self.tocoo(copy=copy).todok(copy=False)
844: 
845:     def tocoo(self, copy=False):
846:         '''Convert this matrix to COOrdinate format.
847: 
848:         With copy=False, the data/indices may be shared between this matrix and
849:         the resultant coo_matrix.
850:         '''
851:         return self.tocsr(copy=False).tocoo(copy=copy)
852: 
853:     def tolil(self, copy=False):
854:         '''Convert this matrix to LInked List format.
855: 
856:         With copy=False, the data/indices may be shared between this matrix and
857:         the resultant lil_matrix.
858:         '''
859:         return self.tocsr(copy=False).tolil(copy=copy)
860: 
861:     def todia(self, copy=False):
862:         '''Convert this matrix to sparse DIAgonal format.
863: 
864:         With copy=False, the data/indices may be shared between this matrix and
865:         the resultant dia_matrix.
866:         '''
867:         return self.tocoo(copy=copy).todia(copy=False)
868: 
869:     def tobsr(self, blocksize=None, copy=False):
870:         '''Convert this matrix to Block Sparse Row format.
871: 
872:         With copy=False, the data/indices may be shared between this matrix and
873:         the resultant bsr_matrix.
874: 
875:         When blocksize=(R, C) is provided, it will be used for construction of
876:         the bsr_matrix.
877:         '''
878:         return self.tocsr(copy=False).tobsr(blocksize=blocksize, copy=copy)
879: 
880:     def tocsc(self, copy=False):
881:         '''Convert this matrix to Compressed Sparse Column format.
882: 
883:         With copy=False, the data/indices may be shared between this matrix and
884:         the resultant csc_matrix.
885:         '''
886:         return self.tocsr(copy=copy).tocsc(copy=False)
887: 
888:     def copy(self):
889:         '''Returns a copy of this matrix.
890: 
891:         No data/indices will be shared between the returned value and current
892:         matrix.
893:         '''
894:         return self.__class__(self, copy=True)
895: 
896:     def sum(self, axis=None, dtype=None, out=None):
897:         '''
898:         Sum the matrix elements over a given axis.
899: 
900:         Parameters
901:         ----------
902:         axis : {-2, -1, 0, 1, None} optional
903:             Axis along which the sum is computed. The default is to
904:             compute the sum of all the matrix elements, returning a scalar
905:             (i.e. `axis` = `None`).
906:         dtype : dtype, optional
907:             The type of the returned matrix and of the accumulator in which
908:             the elements are summed.  The dtype of `a` is used by default
909:             unless `a` has an integer dtype of less precision than the default
910:             platform integer.  In that case, if `a` is signed then the platform
911:             integer is used while if `a` is unsigned then an unsigned integer
912:             of the same precision as the platform integer is used.
913: 
914:             .. versionadded: 0.18.0
915: 
916:         out : np.matrix, optional
917:             Alternative output matrix in which to place the result. It must
918:             have the same shape as the expected output, but the type of the
919:             output values will be cast if necessary.
920: 
921:             .. versionadded: 0.18.0
922: 
923:         Returns
924:         -------
925:         sum_along_axis : np.matrix
926:             A matrix with the same shape as `self`, with the specified
927:             axis removed.
928: 
929:         See Also
930:         --------
931:         np.matrix.sum : NumPy's implementation of 'sum' for matrices
932: 
933:         '''
934:         validateaxis(axis)
935: 
936:         # We use multiplication by a matrix of ones to achieve this.
937:         # For some sparse matrix formats more efficient methods are
938:         # possible -- these should override this function.
939:         m, n = self.shape
940: 
941:         # Mimic numpy's casting.
942:         res_dtype = get_sum_dtype(self.dtype)
943: 
944:         if axis is None:
945:             # sum over rows and columns
946:             return (self * np.asmatrix(np.ones(
947:                 (n, 1), dtype=res_dtype))).sum(
948:                 dtype=dtype, out=out)
949: 
950:         if axis < 0:
951:             axis += 2
952: 
953:         # axis = 0 or 1 now
954:         if axis == 0:
955:             # sum over columns
956:             ret = np.asmatrix(np.ones(
957:                 (1, m), dtype=res_dtype)) * self
958:         else:
959:             # sum over rows
960:             ret = self * np.asmatrix(
961:                 np.ones((n, 1), dtype=res_dtype))
962: 
963:         if out is not None and out.shape != ret.shape:
964:             raise ValueError("dimensions do not match")
965: 
966:         return ret.sum(axis=(), dtype=dtype, out=out)
967: 
968:     def mean(self, axis=None, dtype=None, out=None):
969:         '''
970:         Compute the arithmetic mean along the specified axis.
971: 
972:         Returns the average of the matrix elements. The average is taken
973:         over all elements in the matrix by default, otherwise over the
974:         specified axis. `float64` intermediate and return values are used
975:         for integer inputs.
976: 
977:         Parameters
978:         ----------
979:         axis : {-2, -1, 0, 1, None} optional
980:             Axis along which the mean is computed. The default is to compute
981:             the mean of all elements in the matrix (i.e. `axis` = `None`).
982:         dtype : data-type, optional
983:             Type to use in computing the mean. For integer inputs, the default
984:             is `float64`; for floating point inputs, it is the same as the
985:             input dtype.
986: 
987:             .. versionadded: 0.18.0
988: 
989:         out : np.matrix, optional
990:             Alternative output matrix in which to place the result. It must
991:             have the same shape as the expected output, but the type of the
992:             output values will be cast if necessary.
993: 
994:             .. versionadded: 0.18.0
995: 
996:         Returns
997:         -------
998:         m : np.matrix
999: 
1000:         See Also
1001:         --------
1002:         np.matrix.mean : NumPy's implementation of 'mean' for matrices
1003: 
1004:         '''
1005:         def _is_integral(dtype):
1006:             return (np.issubdtype(dtype, np.integer) or
1007:                     np.issubdtype(dtype, np.bool_))
1008: 
1009:         validateaxis(axis)
1010: 
1011:         res_dtype = self.dtype.type
1012:         integral = _is_integral(self.dtype)
1013: 
1014:         # output dtype
1015:         if dtype is None:
1016:             if integral:
1017:                 res_dtype = np.float64
1018:         else:
1019:             res_dtype = np.dtype(dtype).type
1020: 
1021:         # intermediate dtype for summation
1022:         inter_dtype = np.float64 if integral else res_dtype
1023:         inter_self = self.astype(inter_dtype)
1024: 
1025:         if axis is None:
1026:             return (inter_self / np.array(
1027:                 self.shape[0] * self.shape[1]))\
1028:                 .sum(dtype=res_dtype, out=out)
1029: 
1030:         if axis < 0:
1031:             axis += 2
1032: 
1033:         # axis = 0 or 1 now
1034:         if axis == 0:
1035:             return (inter_self * (1.0 / self.shape[0])).sum(
1036:                 axis=0, dtype=res_dtype, out=out)
1037:         else:
1038:             return (inter_self * (1.0 / self.shape[1])).sum(
1039:                 axis=1, dtype=res_dtype, out=out)
1040: 
1041:     def diagonal(self, k=0):
1042:         '''Returns the k-th diagonal of the matrix.
1043: 
1044:         Parameters
1045:         ----------
1046:         k : int, optional
1047:             Which diagonal to set, corresponding to elements a[i, i+k].
1048:             Default: 0 (the main diagonal).
1049: 
1050:             .. versionadded: 1.0
1051: 
1052:         See also
1053:         --------
1054:         numpy.diagonal : Equivalent numpy function.
1055: 
1056:         Examples
1057:         --------
1058:         >>> from scipy.sparse import csr_matrix
1059:         >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
1060:         >>> A.diagonal()
1061:         array([1, 0, 5])
1062:         >>> A.diagonal(k=1)
1063:         array([2, 3])
1064:         '''
1065:         return self.tocsr().diagonal(k=k)
1066: 
1067:     def setdiag(self, values, k=0):
1068:         '''
1069:         Set diagonal or off-diagonal elements of the array.
1070: 
1071:         Parameters
1072:         ----------
1073:         values : array_like
1074:             New values of the diagonal elements.
1075: 
1076:             Values may have any length.  If the diagonal is longer than values,
1077:             then the remaining diagonal entries will not be set.  If values if
1078:             longer than the diagonal, then the remaining values are ignored.
1079: 
1080:             If a scalar value is given, all of the diagonal is set to it.
1081: 
1082:         k : int, optional
1083:             Which off-diagonal to set, corresponding to elements a[i,i+k].
1084:             Default: 0 (the main diagonal).
1085: 
1086:         '''
1087:         M, N = self.shape
1088:         if (k > 0 and k >= N) or (k < 0 and -k >= M):
1089:             raise ValueError("k exceeds matrix dimensions")
1090:         self._setdiag(np.asarray(values), k)
1091: 
1092:     def _setdiag(self, values, k):
1093:         M, N = self.shape
1094:         if k < 0:
1095:             if values.ndim == 0:
1096:                 # broadcast
1097:                 max_index = min(M+k, N)
1098:                 for i in xrange(max_index):
1099:                     self[i - k, i] = values
1100:             else:
1101:                 max_index = min(M+k, N, len(values))
1102:                 if max_index <= 0:
1103:                     return
1104:                 for i, v in enumerate(values[:max_index]):
1105:                     self[i - k, i] = v
1106:         else:
1107:             if values.ndim == 0:
1108:                 # broadcast
1109:                 max_index = min(M, N-k)
1110:                 for i in xrange(max_index):
1111:                     self[i, i + k] = values
1112:             else:
1113:                 max_index = min(M, N-k, len(values))
1114:                 if max_index <= 0:
1115:                     return
1116:                 for i, v in enumerate(values[:max_index]):
1117:                     self[i, i + k] = v
1118: 
1119:     def _process_toarray_args(self, order, out):
1120:         if out is not None:
1121:             if order is not None:
1122:                 raise ValueError('order cannot be specified if out '
1123:                                  'is not None')
1124:             if out.shape != self.shape or out.dtype != self.dtype:
1125:                 raise ValueError('out array must be same dtype and shape as '
1126:                                  'sparse matrix')
1127:             out[...] = 0.
1128:             return out
1129:         else:
1130:             return np.zeros(self.shape, dtype=self.dtype, order=order)
1131: 
1132:     def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
1133:         '''Method for compatibility with NumPy's ufuncs and dot
1134:         functions.
1135:         '''
1136: 
1137:         if any(not isinstance(x, spmatrix) and np.asarray(x).dtype == object
1138:                for x in inputs):
1139:             # preserve previous behavior with object arrays
1140:             with_self = list(inputs)
1141:             with_self[pos] = np.asarray(self, dtype=object)
1142:             return getattr(func, method)(*with_self, **kwargs)
1143: 
1144:         out = kwargs.pop('out', None)
1145:         if method != '__call__' or kwargs:
1146:             return NotImplemented
1147: 
1148:         without_self = list(inputs)
1149:         del without_self[pos]
1150:         without_self = tuple(without_self)
1151: 
1152:         if func is np.multiply:
1153:             result = self.multiply(*without_self)
1154:         elif func is np.add:
1155:             result = self.__add__(*without_self)
1156:         elif func is np.dot:
1157:             if pos == 0:
1158:                 result = self.__mul__(inputs[1])
1159:             else:
1160:                 result = self.__rmul__(inputs[0])
1161:         elif func is np.subtract:
1162:             if pos == 0:
1163:                 result = self.__sub__(inputs[1])
1164:             else:
1165:                 result = self.__rsub__(inputs[0])
1166:         elif func is np.divide:
1167:             true_divide = (sys.version_info[0] >= 3)
1168:             rdivide = (pos == 1)
1169:             result = self._divide(*without_self,
1170:                                   true_divide=true_divide,
1171:                                   rdivide=rdivide)
1172:         elif func is np.true_divide:
1173:             rdivide = (pos == 1)
1174:             result = self._divide(*without_self,
1175:                                   true_divide=True,
1176:                                   rdivide=rdivide)
1177:         elif func is np.maximum:
1178:             result = self.maximum(*without_self)
1179:         elif func is np.minimum:
1180:             result = self.minimum(*without_self)
1181:         elif func is np.absolute:
1182:             result = abs(self)
1183:         elif func in _ufuncs_with_fixed_point_at_zero:
1184:             func_name = func.__name__
1185:             if hasattr(self, func_name):
1186:                 result = getattr(self, func_name)()
1187:             else:
1188:                 result = getattr(self.tocsr(), func_name)()
1189:         else:
1190:             return NotImplemented
1191: 
1192:         if out is not None:
1193:             if not isinstance(out, spmatrix) and isinstance(result, spmatrix):
1194:                 out[...] = result.todense()
1195:             else:
1196:                 out[...] = result
1197:             result = out
1198: 
1199:         return result
1200: 
1201: 
1202: def isspmatrix(x):
1203:     '''Is x of a sparse matrix type?
1204: 
1205:     Parameters
1206:     ----------
1207:     x
1208:         object to check for being a sparse matrix
1209: 
1210:     Returns
1211:     -------
1212:     bool
1213:         True if x is a sparse matrix, False otherwise
1214: 
1215:     Notes
1216:     -----
1217:     issparse and isspmatrix are aliases for the same function.
1218: 
1219:     Examples
1220:     --------
1221:     >>> from scipy.sparse import csr_matrix, isspmatrix
1222:     >>> isspmatrix(csr_matrix([[5]]))
1223:     True
1224: 
1225:     >>> from scipy.sparse import isspmatrix
1226:     >>> isspmatrix(5)
1227:     False
1228:     '''
1229:     return isinstance(x, spmatrix)
1230: 
1231: issparse = isspmatrix
1232: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_356184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Base class for sparse matrices')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_356185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_356185) is not StypyTypeError):

    if (import_356185 != 'pyd_module'):
        __import__(import_356185)
        sys_modules_356186 = sys.modules[import_356185]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_356186.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_356185)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib.six import xrange' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_356187 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six')

if (type(import_356187) is not StypyTypeError):

    if (import_356187 != 'pyd_module'):
        __import__(import_356187)
        sys_modules_356188 = sys.modules[import_356187]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', sys_modules_356188.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_356188, sys_modules_356188.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', import_356187)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy._lib._numpy_compat import broadcast_to' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_356189 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat')

if (type(import_356189) is not StypyTypeError):

    if (import_356189 != 'pyd_module'):
        __import__(import_356189)
        sys_modules_356190 = sys.modules[import_356189]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat', sys_modules_356190.module_type_store, module_type_store, ['broadcast_to'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_356190, sys_modules_356190.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import broadcast_to

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['broadcast_to'], [broadcast_to])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy._lib._numpy_compat', import_356189)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.sparse.sputils import isdense, isscalarlike, isintlike, get_sum_dtype, validateaxis' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_356191 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.sputils')

if (type(import_356191) is not StypyTypeError):

    if (import_356191 != 'pyd_module'):
        __import__(import_356191)
        sys_modules_356192 = sys.modules[import_356191]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.sputils', sys_modules_356192.module_type_store, module_type_store, ['isdense', 'isscalarlike', 'isintlike', 'get_sum_dtype', 'validateaxis'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_356192, sys_modules_356192.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import isdense, isscalarlike, isintlike, get_sum_dtype, validateaxis

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.sputils', None, module_type_store, ['isdense', 'isscalarlike', 'isintlike', 'get_sum_dtype', 'validateaxis'], [isdense, isscalarlike, isintlike, get_sum_dtype, validateaxis])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.sparse.sputils', import_356191)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['spmatrix', 'isspmatrix', 'issparse', 'SparseWarning', 'SparseEfficiencyWarning']
module_type_store.set_exportable_members(['spmatrix', 'isspmatrix', 'issparse', 'SparseWarning', 'SparseEfficiencyWarning'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_356193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_356194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'spmatrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_356193, str_356194)
# Adding element type (line 13)
str_356195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'isspmatrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_356193, str_356195)
# Adding element type (line 13)
str_356196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 37), 'str', 'issparse')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_356193, str_356196)
# Adding element type (line 13)
str_356197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'SparseWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_356193, str_356197)
# Adding element type (line 13)
str_356198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'str', 'SparseEfficiencyWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_356193, str_356198)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_356193)
# Declaration of the 'SparseWarning' class
# Getting the type of 'Warning' (line 17)
Warning_356199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'Warning')

class SparseWarning(Warning_356199, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SparseWarning' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'SparseWarning', SparseWarning)
# Declaration of the 'SparseFormatWarning' class
# Getting the type of 'SparseWarning' (line 21)
SparseWarning_356200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'SparseWarning')

class SparseFormatWarning(SparseWarning_356200, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 0, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseFormatWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SparseFormatWarning' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'SparseFormatWarning', SparseFormatWarning)
# Declaration of the 'SparseEfficiencyWarning' class
# Getting the type of 'SparseWarning' (line 25)
SparseWarning_356201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'SparseWarning')

class SparseEfficiencyWarning(SparseWarning_356201, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SparseEfficiencyWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SparseEfficiencyWarning' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'SparseEfficiencyWarning', SparseEfficiencyWarning)

# Assigning a Dict to a Name (line 30):

# Assigning a Dict to a Name (line 30):

# Obtaining an instance of the builtin type 'dict' (line 30)
dict_356202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 30)
# Adding element type (key, value) (line 30)
str_356203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'str', 'csc')

# Obtaining an instance of the builtin type 'list' (line 30)
list_356204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
int_356205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 19), list_356204, int_356205)
# Adding element type (line 30)
str_356206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'str', 'Compressed Sparse Column')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 19), list_356204, str_356206)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356203, list_356204))
# Adding element type (key, value) (line 30)
str_356207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'str', 'csr')

# Obtaining an instance of the builtin type 'list' (line 31)
list_356208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
int_356209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_356208, int_356209)
# Adding element type (line 31)
str_356210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'str', 'Compressed Sparse Row')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_356208, str_356210)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356207, list_356208))
# Adding element type (key, value) (line 30)
str_356211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'str', 'dok')

# Obtaining an instance of the builtin type 'list' (line 32)
list_356212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
int_356213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 19), list_356212, int_356213)
# Adding element type (line 32)
str_356214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', 'Dictionary Of Keys')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 19), list_356212, str_356214)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356211, list_356212))
# Adding element type (key, value) (line 30)
str_356215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 12), 'str', 'lil')

# Obtaining an instance of the builtin type 'list' (line 33)
list_356216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
int_356217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), list_356216, int_356217)
# Adding element type (line 33)
str_356218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'str', 'LInked List')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 19), list_356216, str_356218)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356215, list_356216))
# Adding element type (key, value) (line 30)
str_356219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'str', 'dod')

# Obtaining an instance of the builtin type 'list' (line 34)
list_356220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
int_356221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), list_356220, int_356221)
# Adding element type (line 34)
str_356222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'str', 'Dictionary of Dictionaries')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), list_356220, str_356222)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356219, list_356220))
# Adding element type (key, value) (line 30)
str_356223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 12), 'str', 'sss')

# Obtaining an instance of the builtin type 'list' (line 35)
list_356224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)
# Adding element type (line 35)
int_356225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 19), list_356224, int_356225)
# Adding element type (line 35)
str_356226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', 'Symmetric Sparse Skyline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 19), list_356224, str_356226)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356223, list_356224))
# Adding element type (key, value) (line 30)
str_356227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 12), 'str', 'coo')

# Obtaining an instance of the builtin type 'list' (line 36)
list_356228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
int_356229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 19), list_356228, int_356229)
# Adding element type (line 36)
str_356230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'COOrdinate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 19), list_356228, str_356230)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356227, list_356228))
# Adding element type (key, value) (line 30)
str_356231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'str', 'lba')

# Obtaining an instance of the builtin type 'list' (line 37)
list_356232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
int_356233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), list_356232, int_356233)
# Adding element type (line 37)
str_356234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'str', 'Linpack BAnded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), list_356232, str_356234)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356231, list_356232))
# Adding element type (key, value) (line 30)
str_356235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 12), 'str', 'egd')

# Obtaining an instance of the builtin type 'list' (line 38)
list_356236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
int_356237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 19), list_356236, int_356237)
# Adding element type (line 38)
str_356238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'str', 'Ellpack-itpack Generalized Diagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 19), list_356236, str_356238)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356235, list_356236))
# Adding element type (key, value) (line 30)
str_356239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 12), 'str', 'dia')

# Obtaining an instance of the builtin type 'list' (line 39)
list_356240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)
int_356241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 19), list_356240, int_356241)
# Adding element type (line 39)
str_356242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'str', 'DIAgonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 19), list_356240, str_356242)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356239, list_356240))
# Adding element type (key, value) (line 30)
str_356243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 12), 'str', 'bsr')

# Obtaining an instance of the builtin type 'list' (line 40)
list_356244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
int_356245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_356244, int_356245)
# Adding element type (line 40)
str_356246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'str', 'Block Sparse Row')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_356244, str_356246)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356243, list_356244))
# Adding element type (key, value) (line 30)
str_356247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 12), 'str', 'msr')

# Obtaining an instance of the builtin type 'list' (line 41)
list_356248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
int_356249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 19), list_356248, int_356249)
# Adding element type (line 41)
str_356250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'str', 'Modified compressed Sparse Row')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 19), list_356248, str_356250)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356247, list_356248))
# Adding element type (key, value) (line 30)
str_356251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'str', 'bsc')

# Obtaining an instance of the builtin type 'list' (line 42)
list_356252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)
int_356253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_356252, int_356253)
# Adding element type (line 42)
str_356254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'str', 'Block Sparse Column')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_356252, str_356254)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356251, list_356252))
# Adding element type (key, value) (line 30)
str_356255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 12), 'str', 'msc')

# Obtaining an instance of the builtin type 'list' (line 43)
list_356256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
int_356257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_356256, int_356257)
# Adding element type (line 43)
str_356258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'str', 'Modified compressed Sparse Column')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_356256, str_356258)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356255, list_356256))
# Adding element type (key, value) (line 30)
str_356259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'str', 'ssk')

# Obtaining an instance of the builtin type 'list' (line 44)
list_356260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 44)
# Adding element type (line 44)
int_356261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), list_356260, int_356261)
# Adding element type (line 44)
str_356262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'str', 'Symmetric SKyline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), list_356260, str_356262)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356259, list_356260))
# Adding element type (key, value) (line 30)
str_356263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'str', 'nsk')

# Obtaining an instance of the builtin type 'list' (line 45)
list_356264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 45)
# Adding element type (line 45)
int_356265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), list_356264, int_356265)
# Adding element type (line 45)
str_356266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'str', 'Nonsymmetric SKyline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), list_356264, str_356266)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356263, list_356264))
# Adding element type (key, value) (line 30)
str_356267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'str', 'jad')

# Obtaining an instance of the builtin type 'list' (line 46)
list_356268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 46)
# Adding element type (line 46)
int_356269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_356268, int_356269)
# Adding element type (line 46)
str_356270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 24), 'str', 'JAgged Diagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_356268, str_356270)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356267, list_356268))
# Adding element type (key, value) (line 30)
str_356271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'str', 'uss')

# Obtaining an instance of the builtin type 'list' (line 47)
list_356272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
int_356273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), list_356272, int_356273)
# Adding element type (line 47)
str_356274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'str', 'Unsymmetric Sparse Skyline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), list_356272, str_356274)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356271, list_356272))
# Adding element type (key, value) (line 30)
str_356275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 12), 'str', 'vbr')

# Obtaining an instance of the builtin type 'list' (line 48)
list_356276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 48)
# Adding element type (line 48)
int_356277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 19), list_356276, int_356277)
# Adding element type (line 48)
str_356278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 24), 'str', 'Variable Block Row')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 19), list_356276, str_356278)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356275, list_356276))
# Adding element type (key, value) (line 30)
str_356279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 12), 'str', 'und')

# Obtaining an instance of the builtin type 'list' (line 49)
list_356280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)
int_356281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_356280, int_356281)
# Adding element type (line 49)
str_356282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'str', 'Undefined')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_356280, str_356282)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), dict_356202, (str_356279, list_356280))

# Assigning a type to the variable '_formats' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_formats', dict_356202)

# Assigning a Call to a Name (line 54):

# Assigning a Call to a Name (line 54):

# Call to frozenset(...): (line 54)
# Processing the call arguments (line 54)

# Obtaining an instance of the builtin type 'list' (line 54)
list_356284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 45), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
# Getting the type of 'np' (line 55)
np_356285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'np', False)
# Obtaining the member 'sin' of a type (line 55)
sin_356286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), np_356285, 'sin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, sin_356286)
# Adding element type (line 54)
# Getting the type of 'np' (line 55)
np_356287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'np', False)
# Obtaining the member 'tan' of a type (line 55)
tan_356288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), np_356287, 'tan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, tan_356288)
# Adding element type (line 54)
# Getting the type of 'np' (line 55)
np_356289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'np', False)
# Obtaining the member 'arcsin' of a type (line 55)
arcsin_356290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), np_356289, 'arcsin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, arcsin_356290)
# Adding element type (line 54)
# Getting the type of 'np' (line 55)
np_356291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'np', False)
# Obtaining the member 'arctan' of a type (line 55)
arctan_356292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 35), np_356291, 'arctan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, arctan_356292)
# Adding element type (line 54)
# Getting the type of 'np' (line 55)
np_356293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'np', False)
# Obtaining the member 'sinh' of a type (line 55)
sinh_356294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 46), np_356293, 'sinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, sinh_356294)
# Adding element type (line 54)
# Getting the type of 'np' (line 55)
np_356295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 55), 'np', False)
# Obtaining the member 'tanh' of a type (line 55)
tanh_356296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 55), np_356295, 'tanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, tanh_356296)
# Adding element type (line 54)
# Getting the type of 'np' (line 55)
np_356297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 64), 'np', False)
# Obtaining the member 'arcsinh' of a type (line 55)
arcsinh_356298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 64), np_356297, 'arcsinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, arcsinh_356298)
# Adding element type (line 54)
# Getting the type of 'np' (line 56)
np_356299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'np', False)
# Obtaining the member 'arctanh' of a type (line 56)
arctanh_356300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), np_356299, 'arctanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, arctanh_356300)
# Adding element type (line 54)
# Getting the type of 'np' (line 56)
np_356301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'np', False)
# Obtaining the member 'rint' of a type (line 56)
rint_356302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 20), np_356301, 'rint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, rint_356302)
# Adding element type (line 54)
# Getting the type of 'np' (line 56)
np_356303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'np', False)
# Obtaining the member 'sign' of a type (line 56)
sign_356304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), np_356303, 'sign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, sign_356304)
# Adding element type (line 54)
# Getting the type of 'np' (line 56)
np_356305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 38), 'np', False)
# Obtaining the member 'expm1' of a type (line 56)
expm1_356306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 38), np_356305, 'expm1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, expm1_356306)
# Adding element type (line 54)
# Getting the type of 'np' (line 56)
np_356307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 48), 'np', False)
# Obtaining the member 'log1p' of a type (line 56)
log1p_356308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 48), np_356307, 'log1p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, log1p_356308)
# Adding element type (line 54)
# Getting the type of 'np' (line 56)
np_356309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 58), 'np', False)
# Obtaining the member 'deg2rad' of a type (line 56)
deg2rad_356310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 58), np_356309, 'deg2rad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, deg2rad_356310)
# Adding element type (line 54)
# Getting the type of 'np' (line 57)
np_356311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'np', False)
# Obtaining the member 'rad2deg' of a type (line 57)
rad2deg_356312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), np_356311, 'rad2deg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, rad2deg_356312)
# Adding element type (line 54)
# Getting the type of 'np' (line 57)
np_356313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'np', False)
# Obtaining the member 'floor' of a type (line 57)
floor_356314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 20), np_356313, 'floor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, floor_356314)
# Adding element type (line 54)
# Getting the type of 'np' (line 57)
np_356315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'np', False)
# Obtaining the member 'ceil' of a type (line 57)
ceil_356316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 30), np_356315, 'ceil')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, ceil_356316)
# Adding element type (line 54)
# Getting the type of 'np' (line 57)
np_356317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 39), 'np', False)
# Obtaining the member 'trunc' of a type (line 57)
trunc_356318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 39), np_356317, 'trunc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, trunc_356318)
# Adding element type (line 54)
# Getting the type of 'np' (line 57)
np_356319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 49), 'np', False)
# Obtaining the member 'sqrt' of a type (line 57)
sqrt_356320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 49), np_356319, 'sqrt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 45), list_356284, sqrt_356320)

# Processing the call keyword arguments (line 54)
kwargs_356321 = {}
# Getting the type of 'frozenset' (line 54)
frozenset_356283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 35), 'frozenset', False)
# Calling frozenset(args, kwargs) (line 54)
frozenset_call_result_356322 = invoke(stypy.reporting.localization.Localization(__file__, 54, 35), frozenset_356283, *[list_356284], **kwargs_356321)

# Assigning a type to the variable '_ufuncs_with_fixed_point_at_zero' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), '_ufuncs_with_fixed_point_at_zero', frozenset_call_result_356322)

# Assigning a Num to a Name (line 60):

# Assigning a Num to a Name (line 60):
int_356323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 11), 'int')
# Assigning a type to the variable 'MAXPRINT' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'MAXPRINT', int_356323)
# Declaration of the 'spmatrix' class

class spmatrix(object, ):
    str_356324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', ' This class provides a base class for all sparse matrices.  It\n    cannot be instantiated.  Most of the work is provided by subclasses.\n    ')
    
    # Assigning a Num to a Name (line 68):
    
    # Assigning a Num to a Name (line 69):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'MAXPRINT' (line 71)
        MAXPRINT_356325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'MAXPRINT')
        defaults = [MAXPRINT_356325]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__init__', ['maxprint'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['maxprint'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 72):
        
        # Assigning a Name to a Attribute (line 72):
        # Getting the type of 'None' (line 72)
        None_356326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'None')
        # Getting the type of 'self' (line 72)
        self_356327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member '_shape' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_356327, '_shape', None_356326)
        
        
        # Getting the type of 'self' (line 73)
        self_356328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'self')
        # Obtaining the member '__class__' of a type (line 73)
        class___356329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), self_356328, '__class__')
        # Obtaining the member '__name__' of a type (line 73)
        name___356330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), class___356329, '__name__')
        str_356331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 38), 'str', 'spmatrix')
        # Applying the binary operator '==' (line 73)
        result_eq_356332 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '==', name___356330, str_356331)
        
        # Testing the type of an if condition (line 73)
        if_condition_356333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_356332)
        # Assigning a type to the variable 'if_condition_356333' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_356333', if_condition_356333)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 74)
        # Processing the call arguments (line 74)
        str_356335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'str', 'This class is not intended to be instantiated directly.')
        # Processing the call keyword arguments (line 74)
        kwargs_356336 = {}
        # Getting the type of 'ValueError' (line 74)
        ValueError_356334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 74)
        ValueError_call_result_356337 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), ValueError_356334, *[str_356335], **kwargs_356336)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 74, 12), ValueError_call_result_356337, 'raise parameter', BaseException)
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 76):
        
        # Assigning a Name to a Attribute (line 76):
        # Getting the type of 'maxprint' (line 76)
        maxprint_356338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'maxprint')
        # Getting the type of 'self' (line 76)
        self_356339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'maxprint' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_356339, 'maxprint', maxprint_356338)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_shape'
        module_type_store = module_type_store.open_function_context('set_shape', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.set_shape.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.set_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.set_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.set_shape.__dict__.__setitem__('stypy_function_name', 'spmatrix.set_shape')
        spmatrix.set_shape.__dict__.__setitem__('stypy_param_names_list', ['shape'])
        spmatrix.set_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.set_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.set_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.set_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.set_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.set_shape.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.set_shape', ['shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_shape', localization, ['shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_shape(...)' code ##################

        str_356340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'str', 'See `reshape`.')
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to tuple(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'shape' (line 80)
        shape_356342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'shape', False)
        # Processing the call keyword arguments (line 80)
        kwargs_356343 = {}
        # Getting the type of 'tuple' (line 80)
        tuple_356341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 80)
        tuple_call_result_356344 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), tuple_356341, *[shape_356342], **kwargs_356343)
        
        # Assigning a type to the variable 'shape' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'shape', tuple_call_result_356344)
        
        
        
        # Call to len(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'shape' (line 82)
        shape_356346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'shape', False)
        # Processing the call keyword arguments (line 82)
        kwargs_356347 = {}
        # Getting the type of 'len' (line 82)
        len_356345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'len', False)
        # Calling len(args, kwargs) (line 82)
        len_call_result_356348 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), len_356345, *[shape_356346], **kwargs_356347)
        
        int_356349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'int')
        # Applying the binary operator '!=' (line 82)
        result_ne_356350 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 11), '!=', len_call_result_356348, int_356349)
        
        # Testing the type of an if condition (line 82)
        if_condition_356351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_ne_356350)
        # Assigning a type to the variable 'if_condition_356351' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_356351', if_condition_356351)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 83)
        # Processing the call arguments (line 83)
        str_356353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'str', 'Only two-dimensional sparse arrays are supported.')
        # Processing the call keyword arguments (line 83)
        kwargs_356354 = {}
        # Getting the type of 'ValueError' (line 83)
        ValueError_356352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 83)
        ValueError_call_result_356355 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), ValueError_356352, *[str_356353], **kwargs_356354)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 83, 12), ValueError_call_result_356355, 'raise parameter', BaseException)
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Tuple to a Name (line 86):
        
        # Assigning a Tuple to a Name (line 86):
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_356356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        
        # Call to int(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining the type of the subscript
        int_356358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
        # Getting the type of 'shape' (line 86)
        shape_356359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'shape', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___356360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), shape_356359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_356361 = invoke(stypy.reporting.localization.Localization(__file__, 86, 24), getitem___356360, int_356358)
        
        # Processing the call keyword arguments (line 86)
        kwargs_356362 = {}
        # Getting the type of 'int' (line 86)
        int_356357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'int', False)
        # Calling int(args, kwargs) (line 86)
        int_call_result_356363 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), int_356357, *[subscript_call_result_356361], **kwargs_356362)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_356356, int_call_result_356363)
        # Adding element type (line 86)
        
        # Call to int(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining the type of the subscript
        int_356365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 45), 'int')
        # Getting the type of 'shape' (line 86)
        shape_356366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'shape', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___356367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 39), shape_356366, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_356368 = invoke(stypy.reporting.localization.Localization(__file__, 86, 39), getitem___356367, int_356365)
        
        # Processing the call keyword arguments (line 86)
        kwargs_356369 = {}
        # Getting the type of 'int' (line 86)
        int_356364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'int', False)
        # Calling int(args, kwargs) (line 86)
        int_call_result_356370 = invoke(stypy.reporting.localization.Localization(__file__, 86, 35), int_356364, *[subscript_call_result_356368], **kwargs_356369)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 20), tuple_356356, int_call_result_356370)
        
        # Assigning a type to the variable 'shape' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'shape', tuple_356356)
        # SSA branch for the except part of a try statement (line 85)
        # SSA branch for the except '<any exception>' branch of a try statement (line 85)
        module_type_store.open_ssa_branch('except')
        
        # Call to TypeError(...): (line 88)
        # Processing the call arguments (line 88)
        str_356372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'str', 'invalid shape')
        # Processing the call keyword arguments (line 88)
        kwargs_356373 = {}
        # Getting the type of 'TypeError' (line 88)
        TypeError_356371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 88)
        TypeError_call_result_356374 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), TypeError_356371, *[str_356372], **kwargs_356373)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 12), TypeError_call_result_356374, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_356375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'int')
        # Getting the type of 'shape' (line 90)
        shape_356376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'shape')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___356377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 16), shape_356376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_356378 = invoke(stypy.reporting.localization.Localization(__file__, 90, 16), getitem___356377, int_356375)
        
        int_356379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'int')
        # Applying the binary operator '>=' (line 90)
        result_ge_356380 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 16), '>=', subscript_call_result_356378, int_356379)
        
        
        
        # Obtaining the type of the subscript
        int_356381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 40), 'int')
        # Getting the type of 'shape' (line 90)
        shape_356382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'shape')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___356383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 34), shape_356382, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_356384 = invoke(stypy.reporting.localization.Localization(__file__, 90, 34), getitem___356383, int_356381)
        
        int_356385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 46), 'int')
        # Applying the binary operator '>=' (line 90)
        result_ge_356386 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '>=', subscript_call_result_356384, int_356385)
        
        # Applying the binary operator 'and' (line 90)
        result_and_keyword_356387 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 16), 'and', result_ge_356380, result_ge_356386)
        
        # Applying the 'not' unary operator (line 90)
        result_not__356388 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), 'not', result_and_keyword_356387)
        
        # Testing the type of an if condition (line 90)
        if_condition_356389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 8), result_not__356388)
        # Assigning a type to the variable 'if_condition_356389' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'if_condition_356389', if_condition_356389)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 91)
        # Processing the call arguments (line 91)
        str_356391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'str', 'invalid shape')
        # Processing the call keyword arguments (line 91)
        kwargs_356392 = {}
        # Getting the type of 'ValueError' (line 91)
        ValueError_356390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 91)
        ValueError_call_result_356393 = invoke(stypy.reporting.localization.Localization(__file__, 91, 18), ValueError_356390, *[str_356391], **kwargs_356392)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 91, 12), ValueError_call_result_356393, 'raise parameter', BaseException)
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 93)
        self_356394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self')
        # Obtaining the member '_shape' of a type (line 93)
        _shape_356395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_356394, '_shape')
        # Getting the type of 'shape' (line 93)
        shape_356396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'shape')
        # Applying the binary operator '!=' (line 93)
        result_ne_356397 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '!=', _shape_356395, shape_356396)
        
        
        # Getting the type of 'self' (line 93)
        self_356398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 39), 'self')
        # Obtaining the member '_shape' of a type (line 93)
        _shape_356399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 39), self_356398, '_shape')
        # Getting the type of 'None' (line 93)
        None_356400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 58), 'None')
        # Applying the binary operator 'isnot' (line 93)
        result_is_not_356401 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 39), 'isnot', _shape_356399, None_356400)
        
        # Applying the binary operator 'and' (line 93)
        result_and_keyword_356402 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), 'and', result_ne_356397, result_is_not_356401)
        
        # Testing the type of an if condition (line 93)
        if_condition_356403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 8), result_and_keyword_356402)
        # Assigning a type to the variable 'if_condition_356403' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'if_condition_356403', if_condition_356403)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to reshape(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'shape' (line 95)
        shape_356406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'shape', False)
        # Processing the call keyword arguments (line 95)
        kwargs_356407 = {}
        # Getting the type of 'self' (line 95)
        self_356404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'self', False)
        # Obtaining the member 'reshape' of a type (line 95)
        reshape_356405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), self_356404, 'reshape')
        # Calling reshape(args, kwargs) (line 95)
        reshape_call_result_356408 = invoke(stypy.reporting.localization.Localization(__file__, 95, 23), reshape_356405, *[shape_356406], **kwargs_356407)
        
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'self', reshape_call_result_356408)
        # SSA branch for the except part of a try statement (line 94)
        # SSA branch for the except 'NotImplementedError' branch of a try statement (line 94)
        module_type_store.open_ssa_branch('except')
        
        # Call to NotImplementedError(...): (line 97)
        # Processing the call arguments (line 97)
        str_356410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'str', 'Reshaping not implemented for %s.')
        # Getting the type of 'self' (line 98)
        self_356411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'self', False)
        # Obtaining the member '__class__' of a type (line 98)
        class___356412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 42), self_356411, '__class__')
        # Obtaining the member '__name__' of a type (line 98)
        name___356413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 42), class___356412, '__name__')
        # Applying the binary operator '%' (line 97)
        result_mod_356414 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 42), '%', str_356410, name___356413)
        
        # Processing the call keyword arguments (line 97)
        kwargs_356415 = {}
        # Getting the type of 'NotImplementedError' (line 97)
        NotImplementedError_356409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 97)
        NotImplementedError_call_result_356416 = invoke(stypy.reporting.localization.Localization(__file__, 97, 22), NotImplementedError_356409, *[result_mod_356414], **kwargs_356415)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 97, 16), NotImplementedError_call_result_356416, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 99):
        
        # Assigning a Name to a Attribute (line 99):
        # Getting the type of 'shape' (line 99)
        shape_356417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'shape')
        # Getting the type of 'self' (line 99)
        self_356418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member '_shape' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_356418, '_shape', shape_356417)
        
        # ################# End of 'set_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_356419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_shape'
        return stypy_return_type_356419


    @norecursion
    def get_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_shape'
        module_type_store = module_type_store.open_function_context('get_shape', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.get_shape.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.get_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.get_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.get_shape.__dict__.__setitem__('stypy_function_name', 'spmatrix.get_shape')
        spmatrix.get_shape.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.get_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.get_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.get_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.get_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.get_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.get_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.get_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_shape(...)' code ##################

        str_356420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'str', 'Get shape of a matrix.')
        # Getting the type of 'self' (line 103)
        self_356421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'self')
        # Obtaining the member '_shape' of a type (line 103)
        _shape_356422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), self_356421, '_shape')
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', _shape_356422)
        
        # ################# End of 'get_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_356423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_shape'
        return stypy_return_type_356423

    
    # Assigning a Call to a Name (line 105):

    @norecursion
    def reshape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_356424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 35), 'str', 'C')
        defaults = [str_356424]
        # Create a new context for function 'reshape'
        module_type_store = module_type_store.open_function_context('reshape', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.reshape.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.reshape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.reshape.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.reshape.__dict__.__setitem__('stypy_function_name', 'spmatrix.reshape')
        spmatrix.reshape.__dict__.__setitem__('stypy_param_names_list', ['shape', 'order'])
        spmatrix.reshape.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.reshape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.reshape.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.reshape.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.reshape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.reshape.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.reshape', ['shape', 'order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reshape', localization, ['shape', 'order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reshape(...)' code ##################

        str_356425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'str', "\n        Gives a new shape to a sparse matrix without changing its data.\n\n        Parameters\n        ----------\n        shape : length-2 tuple of ints\n            The new shape should be compatible with the original shape.\n        order : 'C', optional\n            This argument is in the signature *solely* for NumPy\n            compatibility reasons. Do not pass in anything except\n            for the default value, as this argument is not used.\n\n        Returns\n        -------\n        reshaped_matrix : `self` with the new dimensions of `shape`\n\n        See Also\n        --------\n        np.matrix.reshape : NumPy's implementation of 'reshape' for matrices\n        ")
        
        # Call to NotImplementedError(...): (line 128)
        # Processing the call arguments (line 128)
        str_356427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'str', 'Reshaping not implemented for %s.')
        # Getting the type of 'self' (line 129)
        self_356428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 34), 'self', False)
        # Obtaining the member '__class__' of a type (line 129)
        class___356429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 34), self_356428, '__class__')
        # Obtaining the member '__name__' of a type (line 129)
        name___356430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 34), class___356429, '__name__')
        # Applying the binary operator '%' (line 128)
        result_mod_356431 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 34), '%', str_356427, name___356430)
        
        # Processing the call keyword arguments (line 128)
        kwargs_356432 = {}
        # Getting the type of 'NotImplementedError' (line 128)
        NotImplementedError_356426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 128)
        NotImplementedError_call_result_356433 = invoke(stypy.reporting.localization.Localization(__file__, 128, 14), NotImplementedError_356426, *[result_mod_356431], **kwargs_356432)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 8), NotImplementedError_call_result_356433, 'raise parameter', BaseException)
        
        # ################# End of 'reshape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reshape' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_356434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356434)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reshape'
        return stypy_return_type_356434


    @norecursion
    def astype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_356435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 36), 'str', 'unsafe')
        # Getting the type of 'True' (line 131)
        True_356436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 51), 'True')
        defaults = [str_356435, True_356436]
        # Create a new context for function 'astype'
        module_type_store = module_type_store.open_function_context('astype', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.astype.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.astype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.astype.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.astype.__dict__.__setitem__('stypy_function_name', 'spmatrix.astype')
        spmatrix.astype.__dict__.__setitem__('stypy_param_names_list', ['dtype', 'casting', 'copy'])
        spmatrix.astype.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.astype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.astype.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.astype.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.astype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.astype.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.astype', ['dtype', 'casting', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'astype', localization, ['dtype', 'casting', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'astype(...)' code ##################

        str_356437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', "Cast the matrix elements to a specified type.\n\n        Parameters\n        ----------\n        dtype : string or numpy dtype\n            Typecode or data-type to which to cast the data.\n        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n            Controls what kind of data casting may occur.\n            Defaults to 'unsafe' for backwards compatibility.\n            'no' means the data types should not be cast at all.\n            'equiv' means only byte-order changes are allowed.\n            'safe' means only casts which can preserve values are allowed.\n            'same_kind' means only safe casts or casts within a kind,\n            like float64 to float32, are allowed.\n            'unsafe' means any data conversions may be done.\n        copy : bool, optional\n            If `copy` is `False`, the result might share some memory with this\n            matrix. If `copy` is `True`, it is guaranteed that the result and\n            this matrix do not share any memory.\n        ")
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to dtype(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'dtype' (line 153)
        dtype_356440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'dtype', False)
        # Processing the call keyword arguments (line 153)
        kwargs_356441 = {}
        # Getting the type of 'np' (line 153)
        np_356438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'np', False)
        # Obtaining the member 'dtype' of a type (line 153)
        dtype_356439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), np_356438, 'dtype')
        # Calling dtype(args, kwargs) (line 153)
        dtype_call_result_356442 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), dtype_356439, *[dtype_356440], **kwargs_356441)
        
        # Assigning a type to the variable 'dtype' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'dtype', dtype_call_result_356442)
        
        
        # Getting the type of 'self' (line 154)
        self_356443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 154)
        dtype_356444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 11), self_356443, 'dtype')
        # Getting the type of 'dtype' (line 154)
        dtype_356445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'dtype')
        # Applying the binary operator '!=' (line 154)
        result_ne_356446 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 11), '!=', dtype_356444, dtype_356445)
        
        # Testing the type of an if condition (line 154)
        if_condition_356447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), result_ne_356446)
        # Assigning a type to the variable 'if_condition_356447' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'if_condition_356447', if_condition_356447)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to asformat(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'self' (line 156)
        self_356461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 60), 'self', False)
        # Obtaining the member 'format' of a type (line 156)
        format_356462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 60), self_356461, 'format')
        # Processing the call keyword arguments (line 155)
        kwargs_356463 = {}
        
        # Call to astype(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'dtype' (line 156)
        dtype_356453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'dtype', False)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'casting' (line 156)
        casting_356454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'casting', False)
        keyword_356455 = casting_356454
        # Getting the type of 'copy' (line 156)
        copy_356456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'copy', False)
        keyword_356457 = copy_356456
        kwargs_356458 = {'copy': keyword_356457, 'casting': keyword_356455}
        
        # Call to tocsr(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_356450 = {}
        # Getting the type of 'self' (line 155)
        self_356448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 155)
        tocsr_356449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), self_356448, 'tocsr')
        # Calling tocsr(args, kwargs) (line 155)
        tocsr_call_result_356451 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), tocsr_356449, *[], **kwargs_356450)
        
        # Obtaining the member 'astype' of a type (line 155)
        astype_356452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), tocsr_call_result_356451, 'astype')
        # Calling astype(args, kwargs) (line 155)
        astype_call_result_356459 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), astype_356452, *[dtype_356453], **kwargs_356458)
        
        # Obtaining the member 'asformat' of a type (line 155)
        asformat_356460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), astype_call_result_356459, 'asformat')
        # Calling asformat(args, kwargs) (line 155)
        asformat_call_result_356464 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), asformat_356460, *[format_356462], **kwargs_356463)
        
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'stypy_return_type', asformat_call_result_356464)
        # SSA branch for the else part of an if statement (line 154)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'copy' (line 157)
        copy_356465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'copy')
        # Testing the type of an if condition (line 157)
        if_condition_356466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 13), copy_356465)
        # Assigning a type to the variable 'if_condition_356466' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'if_condition_356466', if_condition_356466)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_356469 = {}
        # Getting the type of 'self' (line 158)
        self_356467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 158)
        copy_356468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 19), self_356467, 'copy')
        # Calling copy(args, kwargs) (line 158)
        copy_call_result_356470 = invoke(stypy.reporting.localization.Localization(__file__, 158, 19), copy_356468, *[], **kwargs_356469)
        
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'stypy_return_type', copy_call_result_356470)
        # SSA branch for the else part of an if statement (line 157)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 160)
        self_356471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type', self_356471)
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'astype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'astype' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_356472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356472)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'astype'
        return stypy_return_type_356472


    @norecursion
    def asfptype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'asfptype'
        module_type_store = module_type_store.open_function_context('asfptype', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.asfptype.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.asfptype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.asfptype.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.asfptype.__dict__.__setitem__('stypy_function_name', 'spmatrix.asfptype')
        spmatrix.asfptype.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.asfptype.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.asfptype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.asfptype.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.asfptype.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.asfptype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.asfptype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.asfptype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'asfptype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'asfptype(...)' code ##################

        str_356473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 8), 'str', 'Upcast matrix to a floating point format (if necessary)')
        
        # Assigning a List to a Name (line 165):
        
        # Assigning a List to a Name (line 165):
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_356474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        str_356475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'str', 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 19), list_356474, str_356475)
        # Adding element type (line 165)
        str_356476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'str', 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 19), list_356474, str_356476)
        # Adding element type (line 165)
        str_356477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 30), 'str', 'F')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 19), list_356474, str_356477)
        # Adding element type (line 165)
        str_356478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 35), 'str', 'D')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 19), list_356474, str_356478)
        
        # Assigning a type to the variable 'fp_types' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'fp_types', list_356474)
        
        
        # Getting the type of 'self' (line 167)
        self_356479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 167)
        dtype_356480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), self_356479, 'dtype')
        # Obtaining the member 'char' of a type (line 167)
        char_356481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), dtype_356480, 'char')
        # Getting the type of 'fp_types' (line 167)
        fp_types_356482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'fp_types')
        # Applying the binary operator 'in' (line 167)
        result_contains_356483 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), 'in', char_356481, fp_types_356482)
        
        # Testing the type of an if condition (line 167)
        if_condition_356484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), result_contains_356483)
        # Assigning a type to the variable 'if_condition_356484' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_356484', if_condition_356484)
        # SSA begins for if statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 168)
        self_356485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'stypy_return_type', self_356485)
        # SSA branch for the else part of an if statement (line 167)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'fp_types' (line 170)
        fp_types_356486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'fp_types')
        # Testing the type of a for loop iterable (line 170)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 170, 12), fp_types_356486)
        # Getting the type of the for loop variable (line 170)
        for_loop_var_356487 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 170, 12), fp_types_356486)
        # Assigning a type to the variable 'fp_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'fp_type', for_loop_var_356487)
        # SSA begins for a for statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'self' (line 171)
        self_356488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 19), 'self')
        # Obtaining the member 'dtype' of a type (line 171)
        dtype_356489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 19), self_356488, 'dtype')
        
        # Call to dtype(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'fp_type' (line 171)
        fp_type_356492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'fp_type', False)
        # Processing the call keyword arguments (line 171)
        kwargs_356493 = {}
        # Getting the type of 'np' (line 171)
        np_356490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'np', False)
        # Obtaining the member 'dtype' of a type (line 171)
        dtype_356491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 33), np_356490, 'dtype')
        # Calling dtype(args, kwargs) (line 171)
        dtype_call_result_356494 = invoke(stypy.reporting.localization.Localization(__file__, 171, 33), dtype_356491, *[fp_type_356492], **kwargs_356493)
        
        # Applying the binary operator '<=' (line 171)
        result_le_356495 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 19), '<=', dtype_356489, dtype_call_result_356494)
        
        # Testing the type of an if condition (line 171)
        if_condition_356496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 16), result_le_356495)
        # Assigning a type to the variable 'if_condition_356496' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'if_condition_356496', if_condition_356496)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to astype(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'fp_type' (line 172)
        fp_type_356499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 39), 'fp_type', False)
        # Processing the call keyword arguments (line 172)
        kwargs_356500 = {}
        # Getting the type of 'self' (line 172)
        self_356497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'self', False)
        # Obtaining the member 'astype' of a type (line 172)
        astype_356498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 27), self_356497, 'astype')
        # Calling astype(args, kwargs) (line 172)
        astype_call_result_356501 = invoke(stypy.reporting.localization.Localization(__file__, 172, 27), astype_356498, *[fp_type_356499], **kwargs_356500)
        
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'stypy_return_type', astype_call_result_356501)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to TypeError(...): (line 174)
        # Processing the call arguments (line 174)
        str_356503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 28), 'str', 'cannot upcast [%s] to a floating point format')
        # Getting the type of 'self' (line 175)
        self_356504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 45), 'self', False)
        # Obtaining the member 'dtype' of a type (line 175)
        dtype_356505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 45), self_356504, 'dtype')
        # Obtaining the member 'name' of a type (line 175)
        name_356506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 45), dtype_356505, 'name')
        # Applying the binary operator '%' (line 174)
        result_mod_356507 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 28), '%', str_356503, name_356506)
        
        # Processing the call keyword arguments (line 174)
        kwargs_356508 = {}
        # Getting the type of 'TypeError' (line 174)
        TypeError_356502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 174)
        TypeError_call_result_356509 = invoke(stypy.reporting.localization.Localization(__file__, 174, 18), TypeError_356502, *[result_mod_356507], **kwargs_356508)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 174, 12), TypeError_call_result_356509, 'raise parameter', BaseException)
        # SSA join for if statement (line 167)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'asfptype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'asfptype' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_356510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356510)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'asfptype'
        return stypy_return_type_356510


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__iter__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__iter__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__iter__')
        spmatrix.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        
        # Call to xrange(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining the type of the subscript
        int_356512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 35), 'int')
        # Getting the type of 'self' (line 178)
        self_356513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'self', False)
        # Obtaining the member 'shape' of a type (line 178)
        shape_356514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 24), self_356513, 'shape')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___356515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 24), shape_356514, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_356516 = invoke(stypy.reporting.localization.Localization(__file__, 178, 24), getitem___356515, int_356512)
        
        # Processing the call keyword arguments (line 178)
        kwargs_356517 = {}
        # Getting the type of 'xrange' (line 178)
        xrange_356511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 178)
        xrange_call_result_356518 = invoke(stypy.reporting.localization.Localization(__file__, 178, 17), xrange_356511, *[subscript_call_result_356516], **kwargs_356517)
        
        # Testing the type of a for loop iterable (line 178)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 178, 8), xrange_call_result_356518)
        # Getting the type of the for loop variable (line 178)
        for_loop_var_356519 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 178, 8), xrange_call_result_356518)
        # Assigning a type to the variable 'r' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'r', for_loop_var_356519)
        # SSA begins for a for statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        
        # Obtaining the type of the subscript
        # Getting the type of 'r' (line 179)
        r_356520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 'r')
        slice_356521 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 179, 18), None, None, None)
        # Getting the type of 'self' (line 179)
        self_356522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'self')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___356523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 18), self_356522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_356524 = invoke(stypy.reporting.localization.Localization(__file__, 179, 18), getitem___356523, (r_356520, slice_356521))
        
        GeneratorType_356525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 12), GeneratorType_356525, subscript_call_result_356524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'stypy_return_type', GeneratorType_356525)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_356526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_356526


    @norecursion
    def getmaxprint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getmaxprint'
        module_type_store = module_type_store.open_function_context('getmaxprint', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_function_name', 'spmatrix.getmaxprint')
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.getmaxprint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.getmaxprint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getmaxprint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getmaxprint(...)' code ##################

        str_356527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'str', 'Maximum number of elements to display when printed.')
        # Getting the type of 'self' (line 183)
        self_356528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'self')
        # Obtaining the member 'maxprint' of a type (line 183)
        maxprint_356529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 15), self_356528, 'maxprint')
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', maxprint_356529)
        
        # ################# End of 'getmaxprint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getmaxprint' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_356530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356530)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getmaxprint'
        return stypy_return_type_356530


    @norecursion
    def count_nonzero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'count_nonzero'
        module_type_store = module_type_store.open_function_context('count_nonzero', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_function_name', 'spmatrix.count_nonzero')
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.count_nonzero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.count_nonzero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'count_nonzero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'count_nonzero(...)' code ##################

        str_356531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', 'Number of non-zero entries, equivalent to\n\n        np.count_nonzero(a.toarray())\n\n        Unlike getnnz() and the nnz property, which return the number of stored\n        entries (the length of the data attribute), this method counts the\n        actual number of non-zero entries in data.\n        ')
        
        # Call to NotImplementedError(...): (line 194)
        # Processing the call arguments (line 194)
        str_356533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 34), 'str', 'count_nonzero not implemented for %s.')
        # Getting the type of 'self' (line 195)
        self_356534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 34), 'self', False)
        # Obtaining the member '__class__' of a type (line 195)
        class___356535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 34), self_356534, '__class__')
        # Obtaining the member '__name__' of a type (line 195)
        name___356536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 34), class___356535, '__name__')
        # Applying the binary operator '%' (line 194)
        result_mod_356537 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 34), '%', str_356533, name___356536)
        
        # Processing the call keyword arguments (line 194)
        kwargs_356538 = {}
        # Getting the type of 'NotImplementedError' (line 194)
        NotImplementedError_356532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 194)
        NotImplementedError_call_result_356539 = invoke(stypy.reporting.localization.Localization(__file__, 194, 14), NotImplementedError_356532, *[result_mod_356537], **kwargs_356538)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 194, 8), NotImplementedError_call_result_356539, 'raise parameter', BaseException)
        
        # ################# End of 'count_nonzero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count_nonzero' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_356540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_nonzero'
        return stypy_return_type_356540


    @norecursion
    def getnnz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 197)
        None_356541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'None')
        defaults = [None_356541]
        # Create a new context for function 'getnnz'
        module_type_store = module_type_store.open_function_context('getnnz', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.getnnz.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.getnnz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.getnnz.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.getnnz.__dict__.__setitem__('stypy_function_name', 'spmatrix.getnnz')
        spmatrix.getnnz.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        spmatrix.getnnz.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.getnnz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.getnnz.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.getnnz.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.getnnz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.getnnz.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.getnnz', ['axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getnnz', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getnnz(...)' code ##################

        str_356542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, (-1)), 'str', 'Number of stored values, including explicit zeros.\n\n        Parameters\n        ----------\n        axis : None, 0, or 1\n            Select between the number of values across the whole matrix, in\n            each column, or in each row.\n\n        See also\n        --------\n        count_nonzero : Number of non-zero entries\n        ')
        
        # Call to NotImplementedError(...): (line 210)
        # Processing the call arguments (line 210)
        str_356544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 34), 'str', 'getnnz not implemented for %s.')
        # Getting the type of 'self' (line 211)
        self_356545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 34), 'self', False)
        # Obtaining the member '__class__' of a type (line 211)
        class___356546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 34), self_356545, '__class__')
        # Obtaining the member '__name__' of a type (line 211)
        name___356547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 34), class___356546, '__name__')
        # Applying the binary operator '%' (line 210)
        result_mod_356548 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 34), '%', str_356544, name___356547)
        
        # Processing the call keyword arguments (line 210)
        kwargs_356549 = {}
        # Getting the type of 'NotImplementedError' (line 210)
        NotImplementedError_356543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 210)
        NotImplementedError_call_result_356550 = invoke(stypy.reporting.localization.Localization(__file__, 210, 14), NotImplementedError_356543, *[result_mod_356548], **kwargs_356549)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 210, 8), NotImplementedError_call_result_356550, 'raise parameter', BaseException)
        
        # ################# End of 'getnnz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getnnz' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_356551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getnnz'
        return stypy_return_type_356551


    @norecursion
    def nnz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'nnz'
        module_type_store = module_type_store.open_function_context('nnz', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.nnz.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.nnz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.nnz.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.nnz.__dict__.__setitem__('stypy_function_name', 'spmatrix.nnz')
        spmatrix.nnz.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.nnz.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.nnz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.nnz.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.nnz.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.nnz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.nnz.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.nnz', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'nnz', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'nnz(...)' code ##################

        str_356552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'str', 'Number of stored values, including explicit zeros.\n\n        See also\n        --------\n        count_nonzero : Number of non-zero entries\n        ')
        
        # Call to getnnz(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_356555 = {}
        # Getting the type of 'self' (line 221)
        self_356553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'self', False)
        # Obtaining the member 'getnnz' of a type (line 221)
        getnnz_356554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 15), self_356553, 'getnnz')
        # Calling getnnz(args, kwargs) (line 221)
        getnnz_call_result_356556 = invoke(stypy.reporting.localization.Localization(__file__, 221, 15), getnnz_356554, *[], **kwargs_356555)
        
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', getnnz_call_result_356556)
        
        # ################# End of 'nnz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'nnz' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_356557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'nnz'
        return stypy_return_type_356557


    @norecursion
    def getformat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getformat'
        module_type_store = module_type_store.open_function_context('getformat', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.getformat.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.getformat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.getformat.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.getformat.__dict__.__setitem__('stypy_function_name', 'spmatrix.getformat')
        spmatrix.getformat.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.getformat.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.getformat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.getformat.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.getformat.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.getformat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.getformat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.getformat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getformat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getformat(...)' code ##################

        str_356558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'str', 'Format of a matrix representation as a string.')
        
        # Call to getattr(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'self' (line 225)
        self_356560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'self', False)
        str_356561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 29), 'str', 'format')
        str_356562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 39), 'str', 'und')
        # Processing the call keyword arguments (line 225)
        kwargs_356563 = {}
        # Getting the type of 'getattr' (line 225)
        getattr_356559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 225)
        getattr_call_result_356564 = invoke(stypy.reporting.localization.Localization(__file__, 225, 15), getattr_356559, *[self_356560, str_356561, str_356562], **kwargs_356563)
        
        # Assigning a type to the variable 'stypy_return_type' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'stypy_return_type', getattr_call_result_356564)
        
        # ################# End of 'getformat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getformat' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_356565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356565)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getformat'
        return stypy_return_type_356565


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'spmatrix.stypy__repr__')
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Subscript to a Tuple (line 228):
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_356566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to getformat(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_356569 = {}
        # Getting the type of 'self' (line 228)
        self_356567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 34), 'self', False)
        # Obtaining the member 'getformat' of a type (line 228)
        getformat_356568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 34), self_356567, 'getformat')
        # Calling getformat(args, kwargs) (line 228)
        getformat_call_result_356570 = invoke(stypy.reporting.localization.Localization(__file__, 228, 34), getformat_356568, *[], **kwargs_356569)
        
        # Getting the type of '_formats' (line 228)
        _formats_356571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), '_formats')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___356572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 25), _formats_356571, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_356573 = invoke(stypy.reporting.localization.Localization(__file__, 228, 25), getitem___356572, getformat_call_result_356570)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___356574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), subscript_call_result_356573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_356575 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___356574, int_356566)
        
        # Assigning a type to the variable 'tuple_var_assignment_356174' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_356174', subscript_call_result_356575)
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_356576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to getformat(...): (line 228)
        # Processing the call keyword arguments (line 228)
        kwargs_356579 = {}
        # Getting the type of 'self' (line 228)
        self_356577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 34), 'self', False)
        # Obtaining the member 'getformat' of a type (line 228)
        getformat_356578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 34), self_356577, 'getformat')
        # Calling getformat(args, kwargs) (line 228)
        getformat_call_result_356580 = invoke(stypy.reporting.localization.Localization(__file__, 228, 34), getformat_356578, *[], **kwargs_356579)
        
        # Getting the type of '_formats' (line 228)
        _formats_356581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), '_formats')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___356582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 25), _formats_356581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_356583 = invoke(stypy.reporting.localization.Localization(__file__, 228, 25), getitem___356582, getformat_call_result_356580)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___356584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), subscript_call_result_356583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_356585 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___356584, int_356576)
        
        # Assigning a type to the variable 'tuple_var_assignment_356175' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_356175', subscript_call_result_356585)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_356174' (line 228)
        tuple_var_assignment_356174_356586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_356174')
        # Assigning a type to the variable '_' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), '_', tuple_var_assignment_356174_356586)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_356175' (line 228)
        tuple_var_assignment_356175_356587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_356175')
        # Assigning a type to the variable 'format_name' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'format_name', tuple_var_assignment_356175_356587)
        str_356588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 15), 'str', "<%dx%d sparse matrix of type '%s'\n\twith %d stored elements in %s format>")
        # Getting the type of 'self' (line 231)
        self_356589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'self')
        # Obtaining the member 'shape' of a type (line 231)
        shape_356590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), self_356589, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 231)
        tuple_356591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 231)
        # Adding element type (line 231)
        # Getting the type of 'self' (line 231)
        self_356592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'self')
        # Obtaining the member 'dtype' of a type (line 231)
        dtype_356593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 30), self_356592, 'dtype')
        # Obtaining the member 'type' of a type (line 231)
        type_356594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 30), dtype_356593, 'type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 30), tuple_356591, type_356594)
        # Adding element type (line 231)
        # Getting the type of 'self' (line 231)
        self_356595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 47), 'self')
        # Obtaining the member 'nnz' of a type (line 231)
        nnz_356596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 47), self_356595, 'nnz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 30), tuple_356591, nnz_356596)
        # Adding element type (line 231)
        # Getting the type of 'format_name' (line 231)
        format_name_356597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 57), 'format_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 30), tuple_356591, format_name_356597)
        
        # Applying the binary operator '+' (line 231)
        result_add_356598 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 16), '+', shape_356590, tuple_356591)
        
        # Applying the binary operator '%' (line 229)
        result_mod_356599 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '%', str_356588, result_add_356598)
        
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', result_mod_356599)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_356600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_356600


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_function_name', 'spmatrix.stypy__str__')
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to getmaxprint(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_356603 = {}
        # Getting the type of 'self' (line 234)
        self_356601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'self', False)
        # Obtaining the member 'getmaxprint' of a type (line 234)
        getmaxprint_356602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 19), self_356601, 'getmaxprint')
        # Calling getmaxprint(args, kwargs) (line 234)
        getmaxprint_call_result_356604 = invoke(stypy.reporting.localization.Localization(__file__, 234, 19), getmaxprint_356602, *[], **kwargs_356603)
        
        # Assigning a type to the variable 'maxprint' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'maxprint', getmaxprint_call_result_356604)
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to tocoo(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_356607 = {}
        # Getting the type of 'self' (line 236)
        self_356605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 236)
        tocoo_356606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), self_356605, 'tocoo')
        # Calling tocoo(args, kwargs) (line 236)
        tocoo_call_result_356608 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), tocoo_356606, *[], **kwargs_356607)
        
        # Assigning a type to the variable 'A' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'A', tocoo_call_result_356608)

        @norecursion
        def tostr(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'tostr'
            module_type_store = module_type_store.open_function_context('tostr', 239, 8, False)
            
            # Passed parameters checking function
            tostr.stypy_localization = localization
            tostr.stypy_type_of_self = None
            tostr.stypy_type_store = module_type_store
            tostr.stypy_function_name = 'tostr'
            tostr.stypy_param_names_list = ['row', 'col', 'data']
            tostr.stypy_varargs_param_name = None
            tostr.stypy_kwargs_param_name = None
            tostr.stypy_call_defaults = defaults
            tostr.stypy_call_varargs = varargs
            tostr.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'tostr', ['row', 'col', 'data'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'tostr', localization, ['row', 'col', 'data'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'tostr(...)' code ##################

            
            # Assigning a Call to a Name (line 240):
            
            # Assigning a Call to a Name (line 240):
            
            # Call to zip(...): (line 240)
            # Processing the call arguments (line 240)
            
            # Call to list(...): (line 240)
            # Processing the call arguments (line 240)
            
            # Call to zip(...): (line 240)
            # Processing the call arguments (line 240)
            # Getting the type of 'row' (line 240)
            row_356612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 35), 'row', False)
            # Getting the type of 'col' (line 240)
            col_356613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 40), 'col', False)
            # Processing the call keyword arguments (line 240)
            kwargs_356614 = {}
            # Getting the type of 'zip' (line 240)
            zip_356611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'zip', False)
            # Calling zip(args, kwargs) (line 240)
            zip_call_result_356615 = invoke(stypy.reporting.localization.Localization(__file__, 240, 31), zip_356611, *[row_356612, col_356613], **kwargs_356614)
            
            # Processing the call keyword arguments (line 240)
            kwargs_356616 = {}
            # Getting the type of 'list' (line 240)
            list_356610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 26), 'list', False)
            # Calling list(args, kwargs) (line 240)
            list_call_result_356617 = invoke(stypy.reporting.localization.Localization(__file__, 240, 26), list_356610, *[zip_call_result_356615], **kwargs_356616)
            
            # Getting the type of 'data' (line 240)
            data_356618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 47), 'data', False)
            # Processing the call keyword arguments (line 240)
            kwargs_356619 = {}
            # Getting the type of 'zip' (line 240)
            zip_356609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'zip', False)
            # Calling zip(args, kwargs) (line 240)
            zip_call_result_356620 = invoke(stypy.reporting.localization.Localization(__file__, 240, 22), zip_356609, *[list_call_result_356617, data_356618], **kwargs_356619)
            
            # Assigning a type to the variable 'triples' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'triples', zip_call_result_356620)
            
            # Call to join(...): (line 241)
            # Processing the call arguments (line 241)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'triples' (line 241)
            triples_356626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 56), 'triples', False)
            comprehension_356627 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 30), triples_356626)
            # Assigning a type to the variable 't' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 't', comprehension_356627)
            str_356623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 31), 'str', '  %s\t%s')
            # Getting the type of 't' (line 241)
            t_356624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 44), 't', False)
            # Applying the binary operator '%' (line 241)
            result_mod_356625 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 31), '%', str_356623, t_356624)
            
            list_356628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 30), list_356628, result_mod_356625)
            # Processing the call keyword arguments (line 241)
            kwargs_356629 = {}
            str_356621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 19), 'str', '\n')
            # Obtaining the member 'join' of a type (line 241)
            join_356622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 19), str_356621, 'join')
            # Calling join(args, kwargs) (line 241)
            join_call_result_356630 = invoke(stypy.reporting.localization.Localization(__file__, 241, 19), join_356622, *[list_356628], **kwargs_356629)
            
            # Assigning a type to the variable 'stypy_return_type' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'stypy_return_type', join_call_result_356630)
            
            # ################# End of 'tostr(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'tostr' in the type store
            # Getting the type of 'stypy_return_type' (line 239)
            stypy_return_type_356631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_356631)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'tostr'
            return stypy_return_type_356631

        # Assigning a type to the variable 'tostr' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tostr', tostr)
        
        
        # Getting the type of 'self' (line 243)
        self_356632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'self')
        # Obtaining the member 'nnz' of a type (line 243)
        nnz_356633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 11), self_356632, 'nnz')
        # Getting the type of 'maxprint' (line 243)
        maxprint_356634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'maxprint')
        # Applying the binary operator '>' (line 243)
        result_gt_356635 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 11), '>', nnz_356633, maxprint_356634)
        
        # Testing the type of an if condition (line 243)
        if_condition_356636 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), result_gt_356635)
        # Assigning a type to the variable 'if_condition_356636' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_356636', if_condition_356636)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 244):
        
        # Assigning a BinOp to a Name (line 244):
        # Getting the type of 'maxprint' (line 244)
        maxprint_356637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'maxprint')
        int_356638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 31), 'int')
        # Applying the binary operator '//' (line 244)
        result_floordiv_356639 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 19), '//', maxprint_356637, int_356638)
        
        # Assigning a type to the variable 'half' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'half', result_floordiv_356639)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to tostr(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Obtaining the type of the subscript
        # Getting the type of 'half' (line 245)
        half_356641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 31), 'half', False)
        slice_356642 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 24), None, half_356641, None)
        # Getting the type of 'A' (line 245)
        A_356643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'A', False)
        # Obtaining the member 'row' of a type (line 245)
        row_356644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 24), A_356643, 'row')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___356645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 24), row_356644, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_356646 = invoke(stypy.reporting.localization.Localization(__file__, 245, 24), getitem___356645, slice_356642)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'half' (line 245)
        half_356647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 45), 'half', False)
        slice_356648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 38), None, half_356647, None)
        # Getting the type of 'A' (line 245)
        A_356649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 38), 'A', False)
        # Obtaining the member 'col' of a type (line 245)
        col_356650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 38), A_356649, 'col')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___356651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 38), col_356650, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_356652 = invoke(stypy.reporting.localization.Localization(__file__, 245, 38), getitem___356651, slice_356648)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'half' (line 245)
        half_356653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 60), 'half', False)
        slice_356654 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 52), None, half_356653, None)
        # Getting the type of 'A' (line 245)
        A_356655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 52), 'A', False)
        # Obtaining the member 'data' of a type (line 245)
        data_356656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 52), A_356655, 'data')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___356657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 52), data_356656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_356658 = invoke(stypy.reporting.localization.Localization(__file__, 245, 52), getitem___356657, slice_356654)
        
        # Processing the call keyword arguments (line 245)
        kwargs_356659 = {}
        # Getting the type of 'tostr' (line 245)
        tostr_356640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 18), 'tostr', False)
        # Calling tostr(args, kwargs) (line 245)
        tostr_call_result_356660 = invoke(stypy.reporting.localization.Localization(__file__, 245, 18), tostr_356640, *[subscript_call_result_356646, subscript_call_result_356652, subscript_call_result_356658], **kwargs_356659)
        
        # Assigning a type to the variable 'out' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'out', tostr_call_result_356660)
        
        # Getting the type of 'out' (line 246)
        out_356661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'out')
        str_356662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'str', '\n  :\t:\n')
        # Applying the binary operator '+=' (line 246)
        result_iadd_356663 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 12), '+=', out_356661, str_356662)
        # Assigning a type to the variable 'out' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'out', result_iadd_356663)
        
        
        # Assigning a BinOp to a Name (line 247):
        
        # Assigning a BinOp to a Name (line 247):
        # Getting the type of 'maxprint' (line 247)
        maxprint_356664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 19), 'maxprint')
        # Getting the type of 'maxprint' (line 247)
        maxprint_356665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 30), 'maxprint')
        int_356666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 40), 'int')
        # Applying the binary operator '//' (line 247)
        result_floordiv_356667 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 30), '//', maxprint_356665, int_356666)
        
        # Applying the binary operator '-' (line 247)
        result_sub_356668 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 19), '-', maxprint_356664, result_floordiv_356667)
        
        # Assigning a type to the variable 'half' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'half', result_sub_356668)
        
        # Getting the type of 'out' (line 248)
        out_356669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'out')
        
        # Call to tostr(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'half' (line 248)
        half_356671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 32), 'half', False)
        # Applying the 'usub' unary operator (line 248)
        result___neg___356672 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 31), 'usub', half_356671)
        
        slice_356673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 248, 25), result___neg___356672, None, None)
        # Getting the type of 'A' (line 248)
        A_356674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'A', False)
        # Obtaining the member 'row' of a type (line 248)
        row_356675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 25), A_356674, 'row')
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___356676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 25), row_356675, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_356677 = invoke(stypy.reporting.localization.Localization(__file__, 248, 25), getitem___356676, slice_356673)
        
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'half' (line 248)
        half_356678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'half', False)
        # Applying the 'usub' unary operator (line 248)
        result___neg___356679 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 46), 'usub', half_356678)
        
        slice_356680 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 248, 40), result___neg___356679, None, None)
        # Getting the type of 'A' (line 248)
        A_356681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 40), 'A', False)
        # Obtaining the member 'col' of a type (line 248)
        col_356682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 40), A_356681, 'col')
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___356683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 40), col_356682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_356684 = invoke(stypy.reporting.localization.Localization(__file__, 248, 40), getitem___356683, slice_356680)
        
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'half' (line 248)
        half_356685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 63), 'half', False)
        # Applying the 'usub' unary operator (line 248)
        result___neg___356686 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 62), 'usub', half_356685)
        
        slice_356687 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 248, 55), result___neg___356686, None, None)
        # Getting the type of 'A' (line 248)
        A_356688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 55), 'A', False)
        # Obtaining the member 'data' of a type (line 248)
        data_356689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 55), A_356688, 'data')
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___356690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 55), data_356689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_356691 = invoke(stypy.reporting.localization.Localization(__file__, 248, 55), getitem___356690, slice_356687)
        
        # Processing the call keyword arguments (line 248)
        kwargs_356692 = {}
        # Getting the type of 'tostr' (line 248)
        tostr_356670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'tostr', False)
        # Calling tostr(args, kwargs) (line 248)
        tostr_call_result_356693 = invoke(stypy.reporting.localization.Localization(__file__, 248, 19), tostr_356670, *[subscript_call_result_356677, subscript_call_result_356684, subscript_call_result_356691], **kwargs_356692)
        
        # Applying the binary operator '+=' (line 248)
        result_iadd_356694 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 12), '+=', out_356669, tostr_call_result_356693)
        # Assigning a type to the variable 'out' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'out', result_iadd_356694)
        
        # SSA branch for the else part of an if statement (line 243)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to tostr(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'A' (line 250)
        A_356696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'A', False)
        # Obtaining the member 'row' of a type (line 250)
        row_356697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 24), A_356696, 'row')
        # Getting the type of 'A' (line 250)
        A_356698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 31), 'A', False)
        # Obtaining the member 'col' of a type (line 250)
        col_356699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 31), A_356698, 'col')
        # Getting the type of 'A' (line 250)
        A_356700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 38), 'A', False)
        # Obtaining the member 'data' of a type (line 250)
        data_356701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 38), A_356700, 'data')
        # Processing the call keyword arguments (line 250)
        kwargs_356702 = {}
        # Getting the type of 'tostr' (line 250)
        tostr_356695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'tostr', False)
        # Calling tostr(args, kwargs) (line 250)
        tostr_call_result_356703 = invoke(stypy.reporting.localization.Localization(__file__, 250, 18), tostr_356695, *[row_356697, col_356699, data_356701], **kwargs_356702)
        
        # Assigning a type to the variable 'out' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'out', tostr_call_result_356703)
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'out' (line 252)
        out_356704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'stypy_return_type', out_356704)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_356705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_356705


    @norecursion
    def __bool__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__bool__'
        module_type_store = module_type_store.open_function_context('__bool__', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__bool__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__bool__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__bool__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__bool__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__bool__')
        spmatrix.__bool__.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.__bool__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__bool__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__bool__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__bool__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__bool__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__bool__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__bool__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__bool__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__bool__(...)' code ##################

        
        
        # Getting the type of 'self' (line 255)
        self_356706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'self')
        # Obtaining the member 'shape' of a type (line 255)
        shape_356707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 11), self_356706, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 255)
        tuple_356708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 255)
        # Adding element type (line 255)
        int_356709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 26), tuple_356708, int_356709)
        # Adding element type (line 255)
        int_356710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 26), tuple_356708, int_356710)
        
        # Applying the binary operator '==' (line 255)
        result_eq_356711 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 11), '==', shape_356707, tuple_356708)
        
        # Testing the type of an if condition (line 255)
        if_condition_356712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_eq_356711)
        # Assigning a type to the variable 'if_condition_356712' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_356712', if_condition_356712)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 256)
        self_356713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'self')
        # Obtaining the member 'nnz' of a type (line 256)
        nnz_356714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 19), self_356713, 'nnz')
        int_356715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 31), 'int')
        # Applying the binary operator '!=' (line 256)
        result_ne_356716 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 19), '!=', nnz_356714, int_356715)
        
        # Assigning a type to the variable 'stypy_return_type' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'stypy_return_type', result_ne_356716)
        # SSA branch for the else part of an if statement (line 255)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 258)
        # Processing the call arguments (line 258)
        str_356718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'str', 'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().')
        # Processing the call keyword arguments (line 258)
        kwargs_356719 = {}
        # Getting the type of 'ValueError' (line 258)
        ValueError_356717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 258)
        ValueError_call_result_356720 = invoke(stypy.reporting.localization.Localization(__file__, 258, 18), ValueError_356717, *[str_356718], **kwargs_356719)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 258, 12), ValueError_call_result_356720, 'raise parameter', BaseException)
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__bool__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__bool__' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_356721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356721)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__bool__'
        return stypy_return_type_356721

    
    # Assigning a Name to a Name (line 260):

    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__len__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__len__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__len__')
        spmatrix.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        
        # Call to TypeError(...): (line 266)
        # Processing the call arguments (line 266)
        str_356723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 24), 'str', 'sparse matrix length is ambiguous; use getnnz() or shape[0]')
        # Processing the call keyword arguments (line 266)
        kwargs_356724 = {}
        # Getting the type of 'TypeError' (line 266)
        TypeError_356722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 14), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 266)
        TypeError_call_result_356725 = invoke(stypy.reporting.localization.Localization(__file__, 266, 14), TypeError_356722, *[str_356723], **kwargs_356724)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 266, 8), TypeError_call_result_356725, 'raise parameter', BaseException)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 265)
        stypy_return_type_356726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_356726


    @norecursion
    def asformat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'asformat'
        module_type_store = module_type_store.open_function_context('asformat', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.asformat.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.asformat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.asformat.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.asformat.__dict__.__setitem__('stypy_function_name', 'spmatrix.asformat')
        spmatrix.asformat.__dict__.__setitem__('stypy_param_names_list', ['format'])
        spmatrix.asformat.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.asformat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.asformat.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.asformat.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.asformat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.asformat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.asformat', ['format'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'asformat', localization, ['format'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'asformat(...)' code ##################

        str_356727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, (-1)), 'str', 'Return this matrix in a given sparse format\n\n        Parameters\n        ----------\n        format : {string, None}\n            desired sparse matrix format\n                - None for no format conversion\n                - "csr" for csr_matrix format\n                - "csc" for csc_matrix format\n                - "lil" for lil_matrix format\n                - "dok" for dok_matrix format and so on\n\n        ')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'format' (line 284)
        format_356728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'format')
        # Getting the type of 'None' (line 284)
        None_356729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'None')
        # Applying the binary operator 'is' (line 284)
        result_is__356730 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'is', format_356728, None_356729)
        
        
        # Getting the type of 'format' (line 284)
        format_356731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 'format')
        # Getting the type of 'self' (line 284)
        self_356732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 39), 'self')
        # Obtaining the member 'format' of a type (line 284)
        format_356733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 39), self_356732, 'format')
        # Applying the binary operator '==' (line 284)
        result_eq_356734 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 29), '==', format_356731, format_356733)
        
        # Applying the binary operator 'or' (line 284)
        result_or_keyword_356735 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'or', result_is__356730, result_eq_356734)
        
        # Testing the type of an if condition (line 284)
        if_condition_356736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_or_keyword_356735)
        # Assigning a type to the variable 'if_condition_356736' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_356736', if_condition_356736)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 285)
        self_356737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'stypy_return_type', self_356737)
        # SSA branch for the else part of an if statement (line 284)
        module_type_store.open_ssa_branch('else')
        
        # Call to (...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_356745 = {}
        
        # Call to getattr(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'self' (line 287)
        self_356739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 27), 'self', False)
        str_356740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 33), 'str', 'to')
        # Getting the type of 'format' (line 287)
        format_356741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 40), 'format', False)
        # Applying the binary operator '+' (line 287)
        result_add_356742 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 33), '+', str_356740, format_356741)
        
        # Processing the call keyword arguments (line 287)
        kwargs_356743 = {}
        # Getting the type of 'getattr' (line 287)
        getattr_356738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 287)
        getattr_call_result_356744 = invoke(stypy.reporting.localization.Localization(__file__, 287, 19), getattr_356738, *[self_356739, result_add_356742], **kwargs_356743)
        
        # Calling (args, kwargs) (line 287)
        _call_result_356746 = invoke(stypy.reporting.localization.Localization(__file__, 287, 19), getattr_call_result_356744, *[], **kwargs_356745)
        
        # Assigning a type to the variable 'stypy_return_type' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'stypy_return_type', _call_result_356746)
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'asformat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'asformat' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_356747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'asformat'
        return stypy_return_type_356747


    @norecursion
    def multiply(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'multiply'
        module_type_store = module_type_store.open_function_context('multiply', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.multiply.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.multiply.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.multiply.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.multiply.__dict__.__setitem__('stypy_function_name', 'spmatrix.multiply')
        spmatrix.multiply.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.multiply.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.multiply.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.multiply.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.multiply.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.multiply.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.multiply.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.multiply', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'multiply', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'multiply(...)' code ##################

        str_356748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, (-1)), 'str', 'Point-wise multiplication by another matrix\n        ')
        
        # Call to multiply(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'other' (line 299)
        other_356754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 37), 'other', False)
        # Processing the call keyword arguments (line 299)
        kwargs_356755 = {}
        
        # Call to tocsr(...): (line 299)
        # Processing the call keyword arguments (line 299)
        kwargs_356751 = {}
        # Getting the type of 'self' (line 299)
        self_356749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 299)
        tocsr_356750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 15), self_356749, 'tocsr')
        # Calling tocsr(args, kwargs) (line 299)
        tocsr_call_result_356752 = invoke(stypy.reporting.localization.Localization(__file__, 299, 15), tocsr_356750, *[], **kwargs_356751)
        
        # Obtaining the member 'multiply' of a type (line 299)
        multiply_356753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 15), tocsr_call_result_356752, 'multiply')
        # Calling multiply(args, kwargs) (line 299)
        multiply_call_result_356756 = invoke(stypy.reporting.localization.Localization(__file__, 299, 15), multiply_356753, *[other_356754], **kwargs_356755)
        
        # Assigning a type to the variable 'stypy_return_type' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'stypy_return_type', multiply_call_result_356756)
        
        # ################# End of 'multiply(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'multiply' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_356757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356757)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'multiply'
        return stypy_return_type_356757


    @norecursion
    def maximum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'maximum'
        module_type_store = module_type_store.open_function_context('maximum', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.maximum.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.maximum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.maximum.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.maximum.__dict__.__setitem__('stypy_function_name', 'spmatrix.maximum')
        spmatrix.maximum.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.maximum.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.maximum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.maximum.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.maximum.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.maximum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.maximum.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.maximum', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'maximum', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'maximum(...)' code ##################

        str_356758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 8), 'str', 'Element-wise maximum between this and another matrix.')
        
        # Call to maximum(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'other' (line 303)
        other_356764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'other', False)
        # Processing the call keyword arguments (line 303)
        kwargs_356765 = {}
        
        # Call to tocsr(...): (line 303)
        # Processing the call keyword arguments (line 303)
        kwargs_356761 = {}
        # Getting the type of 'self' (line 303)
        self_356759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 303)
        tocsr_356760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 15), self_356759, 'tocsr')
        # Calling tocsr(args, kwargs) (line 303)
        tocsr_call_result_356762 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), tocsr_356760, *[], **kwargs_356761)
        
        # Obtaining the member 'maximum' of a type (line 303)
        maximum_356763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 15), tocsr_call_result_356762, 'maximum')
        # Calling maximum(args, kwargs) (line 303)
        maximum_call_result_356766 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), maximum_356763, *[other_356764], **kwargs_356765)
        
        # Assigning a type to the variable 'stypy_return_type' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'stypy_return_type', maximum_call_result_356766)
        
        # ################# End of 'maximum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'maximum' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_356767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356767)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'maximum'
        return stypy_return_type_356767


    @norecursion
    def minimum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'minimum'
        module_type_store = module_type_store.open_function_context('minimum', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.minimum.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.minimum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.minimum.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.minimum.__dict__.__setitem__('stypy_function_name', 'spmatrix.minimum')
        spmatrix.minimum.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.minimum.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.minimum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.minimum.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.minimum.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.minimum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.minimum.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.minimum', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'minimum', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'minimum(...)' code ##################

        str_356768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 8), 'str', 'Element-wise minimum between this and another matrix.')
        
        # Call to minimum(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'other' (line 307)
        other_356774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 36), 'other', False)
        # Processing the call keyword arguments (line 307)
        kwargs_356775 = {}
        
        # Call to tocsr(...): (line 307)
        # Processing the call keyword arguments (line 307)
        kwargs_356771 = {}
        # Getting the type of 'self' (line 307)
        self_356769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 307)
        tocsr_356770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 15), self_356769, 'tocsr')
        # Calling tocsr(args, kwargs) (line 307)
        tocsr_call_result_356772 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), tocsr_356770, *[], **kwargs_356771)
        
        # Obtaining the member 'minimum' of a type (line 307)
        minimum_356773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 15), tocsr_call_result_356772, 'minimum')
        # Calling minimum(args, kwargs) (line 307)
        minimum_call_result_356776 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), minimum_356773, *[other_356774], **kwargs_356775)
        
        # Assigning a type to the variable 'stypy_return_type' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', minimum_call_result_356776)
        
        # ################# End of 'minimum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'minimum' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_356777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356777)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'minimum'
        return stypy_return_type_356777


    @norecursion
    def dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dot'
        module_type_store = module_type_store.open_function_context('dot', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.dot.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.dot.__dict__.__setitem__('stypy_function_name', 'spmatrix.dot')
        spmatrix.dot.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.dot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.dot.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.dot', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dot', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dot(...)' code ##################

        str_356778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, (-1)), 'str', 'Ordinary dot product\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> from scipy.sparse import csr_matrix\n        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])\n        >>> v = np.array([1, 0, -1])\n        >>> A.dot(v)\n        array([ 1, -3, -1], dtype=int64)\n\n        ')
        # Getting the type of 'self' (line 322)
        self_356779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'self')
        # Getting the type of 'other' (line 322)
        other_356780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 22), 'other')
        # Applying the binary operator '*' (line 322)
        result_mul_356781 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), '*', self_356779, other_356780)
        
        # Assigning a type to the variable 'stypy_return_type' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'stypy_return_type', result_mul_356781)
        
        # ################# End of 'dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dot' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_356782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356782)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dot'
        return stypy_return_type_356782


    @norecursion
    def power(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 324)
        None_356783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 29), 'None')
        defaults = [None_356783]
        # Create a new context for function 'power'
        module_type_store = module_type_store.open_function_context('power', 324, 4, False)
        # Assigning a type to the variable 'self' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.power.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.power.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.power.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.power.__dict__.__setitem__('stypy_function_name', 'spmatrix.power')
        spmatrix.power.__dict__.__setitem__('stypy_param_names_list', ['n', 'dtype'])
        spmatrix.power.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.power.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.power.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.power.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.power.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.power.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.power', ['n', 'dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'power', localization, ['n', 'dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'power(...)' code ##################

        str_356784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 8), 'str', 'Element-wise power.')
        
        # Call to power(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'n' (line 326)
        n_356790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 34), 'n', False)
        # Processing the call keyword arguments (line 326)
        # Getting the type of 'dtype' (line 326)
        dtype_356791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 43), 'dtype', False)
        keyword_356792 = dtype_356791
        kwargs_356793 = {'dtype': keyword_356792}
        
        # Call to tocsr(...): (line 326)
        # Processing the call keyword arguments (line 326)
        kwargs_356787 = {}
        # Getting the type of 'self' (line 326)
        self_356785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 326)
        tocsr_356786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 15), self_356785, 'tocsr')
        # Calling tocsr(args, kwargs) (line 326)
        tocsr_call_result_356788 = invoke(stypy.reporting.localization.Localization(__file__, 326, 15), tocsr_356786, *[], **kwargs_356787)
        
        # Obtaining the member 'power' of a type (line 326)
        power_356789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 15), tocsr_call_result_356788, 'power')
        # Calling power(args, kwargs) (line 326)
        power_call_result_356794 = invoke(stypy.reporting.localization.Localization(__file__, 326, 15), power_356789, *[n_356790], **kwargs_356793)
        
        # Assigning a type to the variable 'stypy_return_type' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'stypy_return_type', power_call_result_356794)
        
        # ################# End of 'power(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'power' in the type store
        # Getting the type of 'stypy_return_type' (line 324)
        stypy_return_type_356795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'power'
        return stypy_return_type_356795


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 328, 4, False)
        # Assigning a type to the variable 'self' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'spmatrix.stypy__eq__')
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Call to __eq__(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'other' (line 329)
        other_356801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 35), 'other', False)
        # Processing the call keyword arguments (line 329)
        kwargs_356802 = {}
        
        # Call to tocsr(...): (line 329)
        # Processing the call keyword arguments (line 329)
        kwargs_356798 = {}
        # Getting the type of 'self' (line 329)
        self_356796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 329)
        tocsr_356797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), self_356796, 'tocsr')
        # Calling tocsr(args, kwargs) (line 329)
        tocsr_call_result_356799 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), tocsr_356797, *[], **kwargs_356798)
        
        # Obtaining the member '__eq__' of a type (line 329)
        eq___356800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), tocsr_call_result_356799, '__eq__')
        # Calling __eq__(args, kwargs) (line 329)
        eq___call_result_356803 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), eq___356800, *[other_356801], **kwargs_356802)
        
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'stypy_return_type', eq___call_result_356803)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_356804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356804)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_356804


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 331, 4, False)
        # Assigning a type to the variable 'self' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__ne__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__ne__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__ne__')
        spmatrix.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        # Call to __ne__(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'other' (line 332)
        other_356810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 35), 'other', False)
        # Processing the call keyword arguments (line 332)
        kwargs_356811 = {}
        
        # Call to tocsr(...): (line 332)
        # Processing the call keyword arguments (line 332)
        kwargs_356807 = {}
        # Getting the type of 'self' (line 332)
        self_356805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 332)
        tocsr_356806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 15), self_356805, 'tocsr')
        # Calling tocsr(args, kwargs) (line 332)
        tocsr_call_result_356808 = invoke(stypy.reporting.localization.Localization(__file__, 332, 15), tocsr_356806, *[], **kwargs_356807)
        
        # Obtaining the member '__ne__' of a type (line 332)
        ne___356809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 15), tocsr_call_result_356808, '__ne__')
        # Calling __ne__(args, kwargs) (line 332)
        ne___call_result_356812 = invoke(stypy.reporting.localization.Localization(__file__, 332, 15), ne___356809, *[other_356810], **kwargs_356811)
        
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'stypy_return_type', ne___call_result_356812)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_356813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356813)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_356813


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__lt__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__lt__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__lt__')
        spmatrix.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__lt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lt__(...)' code ##################

        
        # Call to __lt__(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'other' (line 335)
        other_356819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 35), 'other', False)
        # Processing the call keyword arguments (line 335)
        kwargs_356820 = {}
        
        # Call to tocsr(...): (line 335)
        # Processing the call keyword arguments (line 335)
        kwargs_356816 = {}
        # Getting the type of 'self' (line 335)
        self_356814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 335)
        tocsr_356815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 15), self_356814, 'tocsr')
        # Calling tocsr(args, kwargs) (line 335)
        tocsr_call_result_356817 = invoke(stypy.reporting.localization.Localization(__file__, 335, 15), tocsr_356815, *[], **kwargs_356816)
        
        # Obtaining the member '__lt__' of a type (line 335)
        lt___356818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 15), tocsr_call_result_356817, '__lt__')
        # Calling __lt__(args, kwargs) (line 335)
        lt___call_result_356821 = invoke(stypy.reporting.localization.Localization(__file__, 335, 15), lt___356818, *[other_356819], **kwargs_356820)
        
        # Assigning a type to the variable 'stypy_return_type' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'stypy_return_type', lt___call_result_356821)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_356822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_356822


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 337, 4, False)
        # Assigning a type to the variable 'self' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__gt__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__gt__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__gt__')
        spmatrix.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        
        # Call to __gt__(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'other' (line 338)
        other_356828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 35), 'other', False)
        # Processing the call keyword arguments (line 338)
        kwargs_356829 = {}
        
        # Call to tocsr(...): (line 338)
        # Processing the call keyword arguments (line 338)
        kwargs_356825 = {}
        # Getting the type of 'self' (line 338)
        self_356823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 338)
        tocsr_356824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 15), self_356823, 'tocsr')
        # Calling tocsr(args, kwargs) (line 338)
        tocsr_call_result_356826 = invoke(stypy.reporting.localization.Localization(__file__, 338, 15), tocsr_356824, *[], **kwargs_356825)
        
        # Obtaining the member '__gt__' of a type (line 338)
        gt___356827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 15), tocsr_call_result_356826, '__gt__')
        # Calling __gt__(args, kwargs) (line 338)
        gt___call_result_356830 = invoke(stypy.reporting.localization.Localization(__file__, 338, 15), gt___356827, *[other_356828], **kwargs_356829)
        
        # Assigning a type to the variable 'stypy_return_type' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'stypy_return_type', gt___call_result_356830)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_356831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356831)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_356831


    @norecursion
    def __le__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__le__'
        module_type_store = module_type_store.open_function_context('__le__', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__le__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__le__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__le__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__le__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__le__')
        spmatrix.__le__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__le__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__le__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__le__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__le__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__le__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__le__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__le__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__le__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__le__(...)' code ##################

        
        # Call to __le__(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'other' (line 341)
        other_356837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 35), 'other', False)
        # Processing the call keyword arguments (line 341)
        kwargs_356838 = {}
        
        # Call to tocsr(...): (line 341)
        # Processing the call keyword arguments (line 341)
        kwargs_356834 = {}
        # Getting the type of 'self' (line 341)
        self_356832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 341)
        tocsr_356833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), self_356832, 'tocsr')
        # Calling tocsr(args, kwargs) (line 341)
        tocsr_call_result_356835 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), tocsr_356833, *[], **kwargs_356834)
        
        # Obtaining the member '__le__' of a type (line 341)
        le___356836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), tocsr_call_result_356835, '__le__')
        # Calling __le__(args, kwargs) (line 341)
        le___call_result_356839 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), le___356836, *[other_356837], **kwargs_356838)
        
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'stypy_return_type', le___call_result_356839)
        
        # ################# End of '__le__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__le__' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_356840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__le__'
        return stypy_return_type_356840


    @norecursion
    def __ge__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ge__'
        module_type_store = module_type_store.open_function_context('__ge__', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__ge__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__ge__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__ge__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__ge__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__ge__')
        spmatrix.__ge__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__ge__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__ge__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__ge__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__ge__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__ge__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__ge__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__ge__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ge__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ge__(...)' code ##################

        
        # Call to __ge__(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'other' (line 344)
        other_356846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 35), 'other', False)
        # Processing the call keyword arguments (line 344)
        kwargs_356847 = {}
        
        # Call to tocsr(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_356843 = {}
        # Getting the type of 'self' (line 344)
        self_356841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 344)
        tocsr_356842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 15), self_356841, 'tocsr')
        # Calling tocsr(args, kwargs) (line 344)
        tocsr_call_result_356844 = invoke(stypy.reporting.localization.Localization(__file__, 344, 15), tocsr_356842, *[], **kwargs_356843)
        
        # Obtaining the member '__ge__' of a type (line 344)
        ge___356845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 15), tocsr_call_result_356844, '__ge__')
        # Calling __ge__(args, kwargs) (line 344)
        ge___call_result_356848 = invoke(stypy.reporting.localization.Localization(__file__, 344, 15), ge___356845, *[other_356846], **kwargs_356847)
        
        # Assigning a type to the variable 'stypy_return_type' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'stypy_return_type', ge___call_result_356848)
        
        # ################# End of '__ge__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ge__' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_356849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ge__'
        return stypy_return_type_356849


    @norecursion
    def __abs__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__abs__'
        module_type_store = module_type_store.open_function_context('__abs__', 346, 4, False)
        # Assigning a type to the variable 'self' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__abs__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__abs__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__abs__')
        spmatrix.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__abs__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__abs__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__abs__(...)' code ##################

        
        # Call to abs(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Call to tocsr(...): (line 347)
        # Processing the call keyword arguments (line 347)
        kwargs_356853 = {}
        # Getting the type of 'self' (line 347)
        self_356851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 19), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 347)
        tocsr_356852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 19), self_356851, 'tocsr')
        # Calling tocsr(args, kwargs) (line 347)
        tocsr_call_result_356854 = invoke(stypy.reporting.localization.Localization(__file__, 347, 19), tocsr_356852, *[], **kwargs_356853)
        
        # Processing the call keyword arguments (line 347)
        kwargs_356855 = {}
        # Getting the type of 'abs' (line 347)
        abs_356850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 347)
        abs_call_result_356856 = invoke(stypy.reporting.localization.Localization(__file__, 347, 15), abs_356850, *[tocsr_call_result_356854], **kwargs_356855)
        
        # Assigning a type to the variable 'stypy_return_type' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'stypy_return_type', abs_call_result_356856)
        
        # ################# End of '__abs__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__abs__' in the type store
        # Getting the type of 'stypy_return_type' (line 346)
        stypy_return_type_356857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__abs__'
        return stypy_return_type_356857


    @norecursion
    def _add_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_sparse'
        module_type_store = module_type_store.open_function_context('_add_sparse', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._add_sparse.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_function_name', 'spmatrix._add_sparse')
        spmatrix._add_sparse.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._add_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._add_sparse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._add_sparse', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_sparse', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_sparse(...)' code ##################

        
        # Call to _add_sparse(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'other' (line 350)
        other_356863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 40), 'other', False)
        # Processing the call keyword arguments (line 350)
        kwargs_356864 = {}
        
        # Call to tocsr(...): (line 350)
        # Processing the call keyword arguments (line 350)
        kwargs_356860 = {}
        # Getting the type of 'self' (line 350)
        self_356858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 350)
        tocsr_356859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 15), self_356858, 'tocsr')
        # Calling tocsr(args, kwargs) (line 350)
        tocsr_call_result_356861 = invoke(stypy.reporting.localization.Localization(__file__, 350, 15), tocsr_356859, *[], **kwargs_356860)
        
        # Obtaining the member '_add_sparse' of a type (line 350)
        _add_sparse_356862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 15), tocsr_call_result_356861, '_add_sparse')
        # Calling _add_sparse(args, kwargs) (line 350)
        _add_sparse_call_result_356865 = invoke(stypy.reporting.localization.Localization(__file__, 350, 15), _add_sparse_356862, *[other_356863], **kwargs_356864)
        
        # Assigning a type to the variable 'stypy_return_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type', _add_sparse_call_result_356865)
        
        # ################# End of '_add_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_356866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356866)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_sparse'
        return stypy_return_type_356866


    @norecursion
    def _add_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_dense'
        module_type_store = module_type_store.open_function_context('_add_dense', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._add_dense.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._add_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._add_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._add_dense.__dict__.__setitem__('stypy_function_name', 'spmatrix._add_dense')
        spmatrix._add_dense.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._add_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._add_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._add_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._add_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._add_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._add_dense.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._add_dense', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_dense', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_dense(...)' code ##################

        
        # Call to _add_dense(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'other' (line 353)
        other_356872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 39), 'other', False)
        # Processing the call keyword arguments (line 353)
        kwargs_356873 = {}
        
        # Call to tocoo(...): (line 353)
        # Processing the call keyword arguments (line 353)
        kwargs_356869 = {}
        # Getting the type of 'self' (line 353)
        self_356867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 353)
        tocoo_356868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 15), self_356867, 'tocoo')
        # Calling tocoo(args, kwargs) (line 353)
        tocoo_call_result_356870 = invoke(stypy.reporting.localization.Localization(__file__, 353, 15), tocoo_356868, *[], **kwargs_356869)
        
        # Obtaining the member '_add_dense' of a type (line 353)
        _add_dense_356871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 15), tocoo_call_result_356870, '_add_dense')
        # Calling _add_dense(args, kwargs) (line 353)
        _add_dense_call_result_356874 = invoke(stypy.reporting.localization.Localization(__file__, 353, 15), _add_dense_356871, *[other_356872], **kwargs_356873)
        
        # Assigning a type to the variable 'stypy_return_type' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'stypy_return_type', _add_dense_call_result_356874)
        
        # ################# End of '_add_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_356875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_dense'
        return stypy_return_type_356875


    @norecursion
    def _sub_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sub_sparse'
        module_type_store = module_type_store.open_function_context('_sub_sparse', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_function_name', 'spmatrix._sub_sparse')
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._sub_sparse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._sub_sparse', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sub_sparse', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sub_sparse(...)' code ##################

        
        # Call to _sub_sparse(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'other' (line 356)
        other_356881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 40), 'other', False)
        # Processing the call keyword arguments (line 356)
        kwargs_356882 = {}
        
        # Call to tocsr(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_356878 = {}
        # Getting the type of 'self' (line 356)
        self_356876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 356)
        tocsr_356877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 15), self_356876, 'tocsr')
        # Calling tocsr(args, kwargs) (line 356)
        tocsr_call_result_356879 = invoke(stypy.reporting.localization.Localization(__file__, 356, 15), tocsr_356877, *[], **kwargs_356878)
        
        # Obtaining the member '_sub_sparse' of a type (line 356)
        _sub_sparse_356880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 15), tocsr_call_result_356879, '_sub_sparse')
        # Calling _sub_sparse(args, kwargs) (line 356)
        _sub_sparse_call_result_356883 = invoke(stypy.reporting.localization.Localization(__file__, 356, 15), _sub_sparse_356880, *[other_356881], **kwargs_356882)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type', _sub_sparse_call_result_356883)
        
        # ################# End of '_sub_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sub_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_356884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356884)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sub_sparse'
        return stypy_return_type_356884


    @norecursion
    def _sub_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sub_dense'
        module_type_store = module_type_store.open_function_context('_sub_dense', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._sub_dense.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_function_name', 'spmatrix._sub_dense')
        spmatrix._sub_dense.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._sub_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._sub_dense.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._sub_dense', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sub_dense', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sub_dense(...)' code ##################

        
        # Call to todense(...): (line 359)
        # Processing the call keyword arguments (line 359)
        kwargs_356887 = {}
        # Getting the type of 'self' (line 359)
        self_356885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'self', False)
        # Obtaining the member 'todense' of a type (line 359)
        todense_356886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 15), self_356885, 'todense')
        # Calling todense(args, kwargs) (line 359)
        todense_call_result_356888 = invoke(stypy.reporting.localization.Localization(__file__, 359, 15), todense_356886, *[], **kwargs_356887)
        
        # Getting the type of 'other' (line 359)
        other_356889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'other')
        # Applying the binary operator '-' (line 359)
        result_sub_356890 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 15), '-', todense_call_result_356888, other_356889)
        
        # Assigning a type to the variable 'stypy_return_type' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'stypy_return_type', result_sub_356890)
        
        # ################# End of '_sub_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sub_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 358)
        stypy_return_type_356891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sub_dense'
        return stypy_return_type_356891


    @norecursion
    def _rsub_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rsub_dense'
        module_type_store = module_type_store.open_function_context('_rsub_dense', 361, 4, False)
        # Assigning a type to the variable 'self' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_function_name', 'spmatrix._rsub_dense')
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._rsub_dense.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._rsub_dense', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rsub_dense', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rsub_dense(...)' code ##################

        # Getting the type of 'other' (line 363)
        other_356892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'other')
        
        # Call to todense(...): (line 363)
        # Processing the call keyword arguments (line 363)
        kwargs_356895 = {}
        # Getting the type of 'self' (line 363)
        self_356893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'self', False)
        # Obtaining the member 'todense' of a type (line 363)
        todense_356894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 23), self_356893, 'todense')
        # Calling todense(args, kwargs) (line 363)
        todense_call_result_356896 = invoke(stypy.reporting.localization.Localization(__file__, 363, 23), todense_356894, *[], **kwargs_356895)
        
        # Applying the binary operator '-' (line 363)
        result_sub_356897 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 15), '-', other_356892, todense_call_result_356896)
        
        # Assigning a type to the variable 'stypy_return_type' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'stypy_return_type', result_sub_356897)
        
        # ################# End of '_rsub_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rsub_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 361)
        stypy_return_type_356898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rsub_dense'
        return stypy_return_type_356898


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 365, 4, False)
        # Assigning a type to the variable 'self' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__add__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__add__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__add__')
        spmatrix.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'other' (line 366)
        other_356900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'other', False)
        # Processing the call keyword arguments (line 366)
        kwargs_356901 = {}
        # Getting the type of 'isscalarlike' (line 366)
        isscalarlike_356899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 366)
        isscalarlike_call_result_356902 = invoke(stypy.reporting.localization.Localization(__file__, 366, 11), isscalarlike_356899, *[other_356900], **kwargs_356901)
        
        # Testing the type of an if condition (line 366)
        if_condition_356903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 8), isscalarlike_call_result_356902)
        # Assigning a type to the variable 'if_condition_356903' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'if_condition_356903', if_condition_356903)
        # SSA begins for if statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 367)
        other_356904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'other')
        int_356905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 24), 'int')
        # Applying the binary operator '==' (line 367)
        result_eq_356906 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 15), '==', other_356904, int_356905)
        
        # Testing the type of an if condition (line 367)
        if_condition_356907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 12), result_eq_356906)
        # Assigning a type to the variable 'if_condition_356907' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'if_condition_356907', if_condition_356907)
        # SSA begins for if statement (line 367)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 368)
        # Processing the call keyword arguments (line 368)
        kwargs_356910 = {}
        # Getting the type of 'self' (line 368)
        self_356908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 23), 'self', False)
        # Obtaining the member 'copy' of a type (line 368)
        copy_356909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 23), self_356908, 'copy')
        # Calling copy(args, kwargs) (line 368)
        copy_call_result_356911 = invoke(stypy.reporting.localization.Localization(__file__, 368, 23), copy_356909, *[], **kwargs_356910)
        
        # Assigning a type to the variable 'stypy_return_type' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'stypy_return_type', copy_call_result_356911)
        # SSA join for if statement (line 367)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to NotImplementedError(...): (line 370)
        # Processing the call arguments (line 370)
        str_356913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 38), 'str', 'adding a nonzero scalar to a sparse matrix is not supported')
        # Processing the call keyword arguments (line 370)
        kwargs_356914 = {}
        # Getting the type of 'NotImplementedError' (line 370)
        NotImplementedError_356912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 370)
        NotImplementedError_call_result_356915 = invoke(stypy.reporting.localization.Localization(__file__, 370, 18), NotImplementedError_356912, *[str_356913], **kwargs_356914)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 370, 12), NotImplementedError_call_result_356915, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 366)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'other' (line 372)
        other_356917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'other', False)
        # Processing the call keyword arguments (line 372)
        kwargs_356918 = {}
        # Getting the type of 'isspmatrix' (line 372)
        isspmatrix_356916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 13), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 372)
        isspmatrix_call_result_356919 = invoke(stypy.reporting.localization.Localization(__file__, 372, 13), isspmatrix_356916, *[other_356917], **kwargs_356918)
        
        # Testing the type of an if condition (line 372)
        if_condition_356920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 13), isspmatrix_call_result_356919)
        # Assigning a type to the variable 'if_condition_356920' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 13), 'if_condition_356920', if_condition_356920)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 373)
        other_356921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'other')
        # Obtaining the member 'shape' of a type (line 373)
        shape_356922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), other_356921, 'shape')
        # Getting the type of 'self' (line 373)
        self_356923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 30), 'self')
        # Obtaining the member 'shape' of a type (line 373)
        shape_356924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 30), self_356923, 'shape')
        # Applying the binary operator '!=' (line 373)
        result_ne_356925 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 15), '!=', shape_356922, shape_356924)
        
        # Testing the type of an if condition (line 373)
        if_condition_356926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 12), result_ne_356925)
        # Assigning a type to the variable 'if_condition_356926' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'if_condition_356926', if_condition_356926)
        # SSA begins for if statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 374)
        # Processing the call arguments (line 374)
        str_356928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 33), 'str', 'inconsistent shapes')
        # Processing the call keyword arguments (line 374)
        kwargs_356929 = {}
        # Getting the type of 'ValueError' (line 374)
        ValueError_356927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 374)
        ValueError_call_result_356930 = invoke(stypy.reporting.localization.Localization(__file__, 374, 22), ValueError_356927, *[str_356928], **kwargs_356929)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 374, 16), ValueError_call_result_356930, 'raise parameter', BaseException)
        # SSA join for if statement (line 373)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _add_sparse(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'other' (line 375)
        other_356933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 36), 'other', False)
        # Processing the call keyword arguments (line 375)
        kwargs_356934 = {}
        # Getting the type of 'self' (line 375)
        self_356931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'self', False)
        # Obtaining the member '_add_sparse' of a type (line 375)
        _add_sparse_356932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 19), self_356931, '_add_sparse')
        # Calling _add_sparse(args, kwargs) (line 375)
        _add_sparse_call_result_356935 = invoke(stypy.reporting.localization.Localization(__file__, 375, 19), _add_sparse_356932, *[other_356933], **kwargs_356934)
        
        # Assigning a type to the variable 'stypy_return_type' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'stypy_return_type', _add_sparse_call_result_356935)
        # SSA branch for the else part of an if statement (line 372)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdense(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'other' (line 376)
        other_356937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 21), 'other', False)
        # Processing the call keyword arguments (line 376)
        kwargs_356938 = {}
        # Getting the type of 'isdense' (line 376)
        isdense_356936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 13), 'isdense', False)
        # Calling isdense(args, kwargs) (line 376)
        isdense_call_result_356939 = invoke(stypy.reporting.localization.Localization(__file__, 376, 13), isdense_356936, *[other_356937], **kwargs_356938)
        
        # Testing the type of an if condition (line 376)
        if_condition_356940 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 13), isdense_call_result_356939)
        # Assigning a type to the variable 'if_condition_356940' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 13), 'if_condition_356940', if_condition_356940)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to broadcast_to(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'other' (line 377)
        other_356942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 33), 'other', False)
        # Getting the type of 'self' (line 377)
        self_356943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 40), 'self', False)
        # Obtaining the member 'shape' of a type (line 377)
        shape_356944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 40), self_356943, 'shape')
        # Processing the call keyword arguments (line 377)
        kwargs_356945 = {}
        # Getting the type of 'broadcast_to' (line 377)
        broadcast_to_356941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 20), 'broadcast_to', False)
        # Calling broadcast_to(args, kwargs) (line 377)
        broadcast_to_call_result_356946 = invoke(stypy.reporting.localization.Localization(__file__, 377, 20), broadcast_to_356941, *[other_356942, shape_356944], **kwargs_356945)
        
        # Assigning a type to the variable 'other' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'other', broadcast_to_call_result_356946)
        
        # Call to _add_dense(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'other' (line 378)
        other_356949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 35), 'other', False)
        # Processing the call keyword arguments (line 378)
        kwargs_356950 = {}
        # Getting the type of 'self' (line 378)
        self_356947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 'self', False)
        # Obtaining the member '_add_dense' of a type (line 378)
        _add_dense_356948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 19), self_356947, '_add_dense')
        # Calling _add_dense(args, kwargs) (line 378)
        _add_dense_call_result_356951 = invoke(stypy.reporting.localization.Localization(__file__, 378, 19), _add_dense_356948, *[other_356949], **kwargs_356950)
        
        # Assigning a type to the variable 'stypy_return_type' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'stypy_return_type', _add_dense_call_result_356951)
        # SSA branch for the else part of an if statement (line 376)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 380)
        NotImplemented_356952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'stypy_return_type', NotImplemented_356952)
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 365)
        stypy_return_type_356953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356953)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_356953


    @norecursion
    def __radd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__radd__'
        module_type_store = module_type_store.open_function_context('__radd__', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__radd__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__radd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__radd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__radd__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__radd__')
        spmatrix.__radd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__radd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__radd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__radd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__radd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__radd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__radd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__radd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__radd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__radd__(...)' code ##################

        
        # Call to __add__(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'other' (line 383)
        other_356956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 28), 'other', False)
        # Processing the call keyword arguments (line 383)
        kwargs_356957 = {}
        # Getting the type of 'self' (line 383)
        self_356954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'self', False)
        # Obtaining the member '__add__' of a type (line 383)
        add___356955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 15), self_356954, '__add__')
        # Calling __add__(args, kwargs) (line 383)
        add___call_result_356958 = invoke(stypy.reporting.localization.Localization(__file__, 383, 15), add___356955, *[other_356956], **kwargs_356957)
        
        # Assigning a type to the variable 'stypy_return_type' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'stypy_return_type', add___call_result_356958)
        
        # ################# End of '__radd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__radd__' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_356959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_356959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__radd__'
        return stypy_return_type_356959


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 385, 4, False)
        # Assigning a type to the variable 'self' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__sub__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__sub__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__sub__')
        spmatrix.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__sub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'other' (line 386)
        other_356961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'other', False)
        # Processing the call keyword arguments (line 386)
        kwargs_356962 = {}
        # Getting the type of 'isscalarlike' (line 386)
        isscalarlike_356960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 386)
        isscalarlike_call_result_356963 = invoke(stypy.reporting.localization.Localization(__file__, 386, 11), isscalarlike_356960, *[other_356961], **kwargs_356962)
        
        # Testing the type of an if condition (line 386)
        if_condition_356964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 8), isscalarlike_call_result_356963)
        # Assigning a type to the variable 'if_condition_356964' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'if_condition_356964', if_condition_356964)
        # SSA begins for if statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 387)
        other_356965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 15), 'other')
        int_356966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 24), 'int')
        # Applying the binary operator '==' (line 387)
        result_eq_356967 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 15), '==', other_356965, int_356966)
        
        # Testing the type of an if condition (line 387)
        if_condition_356968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 12), result_eq_356967)
        # Assigning a type to the variable 'if_condition_356968' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'if_condition_356968', if_condition_356968)
        # SSA begins for if statement (line 387)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 388)
        # Processing the call keyword arguments (line 388)
        kwargs_356971 = {}
        # Getting the type of 'self' (line 388)
        self_356969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'self', False)
        # Obtaining the member 'copy' of a type (line 388)
        copy_356970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 23), self_356969, 'copy')
        # Calling copy(args, kwargs) (line 388)
        copy_call_result_356972 = invoke(stypy.reporting.localization.Localization(__file__, 388, 23), copy_356970, *[], **kwargs_356971)
        
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'stypy_return_type', copy_call_result_356972)
        # SSA join for if statement (line 387)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to NotImplementedError(...): (line 389)
        # Processing the call arguments (line 389)
        str_356974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 38), 'str', 'subtracting a nonzero scalar from a sparse matrix is not supported')
        # Processing the call keyword arguments (line 389)
        kwargs_356975 = {}
        # Getting the type of 'NotImplementedError' (line 389)
        NotImplementedError_356973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 389)
        NotImplementedError_call_result_356976 = invoke(stypy.reporting.localization.Localization(__file__, 389, 18), NotImplementedError_356973, *[str_356974], **kwargs_356975)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 389, 12), NotImplementedError_call_result_356976, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 386)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'other' (line 391)
        other_356978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 24), 'other', False)
        # Processing the call keyword arguments (line 391)
        kwargs_356979 = {}
        # Getting the type of 'isspmatrix' (line 391)
        isspmatrix_356977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 13), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 391)
        isspmatrix_call_result_356980 = invoke(stypy.reporting.localization.Localization(__file__, 391, 13), isspmatrix_356977, *[other_356978], **kwargs_356979)
        
        # Testing the type of an if condition (line 391)
        if_condition_356981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 13), isspmatrix_call_result_356980)
        # Assigning a type to the variable 'if_condition_356981' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 13), 'if_condition_356981', if_condition_356981)
        # SSA begins for if statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 392)
        other_356982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'other')
        # Obtaining the member 'shape' of a type (line 392)
        shape_356983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), other_356982, 'shape')
        # Getting the type of 'self' (line 392)
        self_356984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 30), 'self')
        # Obtaining the member 'shape' of a type (line 392)
        shape_356985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 30), self_356984, 'shape')
        # Applying the binary operator '!=' (line 392)
        result_ne_356986 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 15), '!=', shape_356983, shape_356985)
        
        # Testing the type of an if condition (line 392)
        if_condition_356987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 12), result_ne_356986)
        # Assigning a type to the variable 'if_condition_356987' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'if_condition_356987', if_condition_356987)
        # SSA begins for if statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 393)
        # Processing the call arguments (line 393)
        str_356989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 33), 'str', 'inconsistent shapes')
        # Processing the call keyword arguments (line 393)
        kwargs_356990 = {}
        # Getting the type of 'ValueError' (line 393)
        ValueError_356988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 393)
        ValueError_call_result_356991 = invoke(stypy.reporting.localization.Localization(__file__, 393, 22), ValueError_356988, *[str_356989], **kwargs_356990)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 393, 16), ValueError_call_result_356991, 'raise parameter', BaseException)
        # SSA join for if statement (line 392)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _sub_sparse(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'other' (line 394)
        other_356994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 36), 'other', False)
        # Processing the call keyword arguments (line 394)
        kwargs_356995 = {}
        # Getting the type of 'self' (line 394)
        self_356992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'self', False)
        # Obtaining the member '_sub_sparse' of a type (line 394)
        _sub_sparse_356993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), self_356992, '_sub_sparse')
        # Calling _sub_sparse(args, kwargs) (line 394)
        _sub_sparse_call_result_356996 = invoke(stypy.reporting.localization.Localization(__file__, 394, 19), _sub_sparse_356993, *[other_356994], **kwargs_356995)
        
        # Assigning a type to the variable 'stypy_return_type' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'stypy_return_type', _sub_sparse_call_result_356996)
        # SSA branch for the else part of an if statement (line 391)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdense(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'other' (line 395)
        other_356998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'other', False)
        # Processing the call keyword arguments (line 395)
        kwargs_356999 = {}
        # Getting the type of 'isdense' (line 395)
        isdense_356997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'isdense', False)
        # Calling isdense(args, kwargs) (line 395)
        isdense_call_result_357000 = invoke(stypy.reporting.localization.Localization(__file__, 395, 13), isdense_356997, *[other_356998], **kwargs_356999)
        
        # Testing the type of an if condition (line 395)
        if_condition_357001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 13), isdense_call_result_357000)
        # Assigning a type to the variable 'if_condition_357001' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'if_condition_357001', if_condition_357001)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 396):
        
        # Assigning a Call to a Name (line 396):
        
        # Call to broadcast_to(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'other' (line 396)
        other_357003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 33), 'other', False)
        # Getting the type of 'self' (line 396)
        self_357004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 40), 'self', False)
        # Obtaining the member 'shape' of a type (line 396)
        shape_357005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 40), self_357004, 'shape')
        # Processing the call keyword arguments (line 396)
        kwargs_357006 = {}
        # Getting the type of 'broadcast_to' (line 396)
        broadcast_to_357002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'broadcast_to', False)
        # Calling broadcast_to(args, kwargs) (line 396)
        broadcast_to_call_result_357007 = invoke(stypy.reporting.localization.Localization(__file__, 396, 20), broadcast_to_357002, *[other_357003, shape_357005], **kwargs_357006)
        
        # Assigning a type to the variable 'other' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'other', broadcast_to_call_result_357007)
        
        # Call to _sub_dense(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'other' (line 397)
        other_357010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 35), 'other', False)
        # Processing the call keyword arguments (line 397)
        kwargs_357011 = {}
        # Getting the type of 'self' (line 397)
        self_357008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 'self', False)
        # Obtaining the member '_sub_dense' of a type (line 397)
        _sub_dense_357009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 19), self_357008, '_sub_dense')
        # Calling _sub_dense(args, kwargs) (line 397)
        _sub_dense_call_result_357012 = invoke(stypy.reporting.localization.Localization(__file__, 397, 19), _sub_dense_357009, *[other_357010], **kwargs_357011)
        
        # Assigning a type to the variable 'stypy_return_type' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'stypy_return_type', _sub_dense_call_result_357012)
        # SSA branch for the else part of an if statement (line 395)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 399)
        NotImplemented_357013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'stypy_return_type', NotImplemented_357013)
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 391)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 386)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_357014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_357014


    @norecursion
    def __rsub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rsub__'
        module_type_store = module_type_store.open_function_context('__rsub__', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__rsub__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__rsub__')
        spmatrix.__rsub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__rsub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__rsub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__rsub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rsub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rsub__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'other' (line 402)
        other_357016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'other', False)
        # Processing the call keyword arguments (line 402)
        kwargs_357017 = {}
        # Getting the type of 'isscalarlike' (line 402)
        isscalarlike_357015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 402)
        isscalarlike_call_result_357018 = invoke(stypy.reporting.localization.Localization(__file__, 402, 11), isscalarlike_357015, *[other_357016], **kwargs_357017)
        
        # Testing the type of an if condition (line 402)
        if_condition_357019 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), isscalarlike_call_result_357018)
        # Assigning a type to the variable 'if_condition_357019' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_357019', if_condition_357019)
        # SSA begins for if statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 403)
        other_357020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 15), 'other')
        int_357021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 24), 'int')
        # Applying the binary operator '==' (line 403)
        result_eq_357022 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 15), '==', other_357020, int_357021)
        
        # Testing the type of an if condition (line 403)
        if_condition_357023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 12), result_eq_357022)
        # Assigning a type to the variable 'if_condition_357023' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'if_condition_357023', if_condition_357023)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to copy(...): (line 404)
        # Processing the call keyword arguments (line 404)
        kwargs_357026 = {}
        # Getting the type of 'self' (line 404)
        self_357024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'self', False)
        # Obtaining the member 'copy' of a type (line 404)
        copy_357025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 24), self_357024, 'copy')
        # Calling copy(args, kwargs) (line 404)
        copy_call_result_357027 = invoke(stypy.reporting.localization.Localization(__file__, 404, 24), copy_357025, *[], **kwargs_357026)
        
        # Applying the 'usub' unary operator (line 404)
        result___neg___357028 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 23), 'usub', copy_call_result_357027)
        
        # Assigning a type to the variable 'stypy_return_type' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 16), 'stypy_return_type', result___neg___357028)
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to NotImplementedError(...): (line 405)
        # Processing the call arguments (line 405)
        str_357030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 38), 'str', 'subtracting a sparse matrix from a nonzero scalar is not supported')
        # Processing the call keyword arguments (line 405)
        kwargs_357031 = {}
        # Getting the type of 'NotImplementedError' (line 405)
        NotImplementedError_357029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 405)
        NotImplementedError_call_result_357032 = invoke(stypy.reporting.localization.Localization(__file__, 405, 18), NotImplementedError_357029, *[str_357030], **kwargs_357031)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 405, 12), NotImplementedError_call_result_357032, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 402)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdense(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'other' (line 407)
        other_357034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 21), 'other', False)
        # Processing the call keyword arguments (line 407)
        kwargs_357035 = {}
        # Getting the type of 'isdense' (line 407)
        isdense_357033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 13), 'isdense', False)
        # Calling isdense(args, kwargs) (line 407)
        isdense_call_result_357036 = invoke(stypy.reporting.localization.Localization(__file__, 407, 13), isdense_357033, *[other_357034], **kwargs_357035)
        
        # Testing the type of an if condition (line 407)
        if_condition_357037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 13), isdense_call_result_357036)
        # Assigning a type to the variable 'if_condition_357037' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 13), 'if_condition_357037', if_condition_357037)
        # SSA begins for if statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 408):
        
        # Assigning a Call to a Name (line 408):
        
        # Call to broadcast_to(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'other' (line 408)
        other_357039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 33), 'other', False)
        # Getting the type of 'self' (line 408)
        self_357040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'self', False)
        # Obtaining the member 'shape' of a type (line 408)
        shape_357041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 40), self_357040, 'shape')
        # Processing the call keyword arguments (line 408)
        kwargs_357042 = {}
        # Getting the type of 'broadcast_to' (line 408)
        broadcast_to_357038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 20), 'broadcast_to', False)
        # Calling broadcast_to(args, kwargs) (line 408)
        broadcast_to_call_result_357043 = invoke(stypy.reporting.localization.Localization(__file__, 408, 20), broadcast_to_357038, *[other_357039, shape_357041], **kwargs_357042)
        
        # Assigning a type to the variable 'other' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'other', broadcast_to_call_result_357043)
        
        # Call to _rsub_dense(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'other' (line 409)
        other_357046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 36), 'other', False)
        # Processing the call keyword arguments (line 409)
        kwargs_357047 = {}
        # Getting the type of 'self' (line 409)
        self_357044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 19), 'self', False)
        # Obtaining the member '_rsub_dense' of a type (line 409)
        _rsub_dense_357045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 19), self_357044, '_rsub_dense')
        # Calling _rsub_dense(args, kwargs) (line 409)
        _rsub_dense_call_result_357048 = invoke(stypy.reporting.localization.Localization(__file__, 409, 19), _rsub_dense_357045, *[other_357046], **kwargs_357047)
        
        # Assigning a type to the variable 'stypy_return_type' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'stypy_return_type', _rsub_dense_call_result_357048)
        # SSA branch for the else part of an if statement (line 407)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 411)
        NotImplemented_357049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'stypy_return_type', NotImplemented_357049)
        # SSA join for if statement (line 407)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__rsub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rsub__' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_357050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rsub__'
        return stypy_return_type_357050


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__mul__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__mul__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__mul__')
        spmatrix.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        str_357051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, (-1)), 'str', 'interpret other and call one of the following\n\n        self._mul_scalar()\n        self._mul_vector()\n        self._mul_multivector()\n        self._mul_sparse_matrix()\n        ')
        
        # Assigning a Attribute to a Tuple (line 422):
        
        # Assigning a Subscript to a Name (line 422):
        
        # Obtaining the type of the subscript
        int_357052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 8), 'int')
        # Getting the type of 'self' (line 422)
        self_357053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'self')
        # Obtaining the member 'shape' of a type (line 422)
        shape_357054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), self_357053, 'shape')
        # Obtaining the member '__getitem__' of a type (line 422)
        getitem___357055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), shape_357054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 422)
        subscript_call_result_357056 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), getitem___357055, int_357052)
        
        # Assigning a type to the variable 'tuple_var_assignment_356176' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'tuple_var_assignment_356176', subscript_call_result_357056)
        
        # Assigning a Subscript to a Name (line 422):
        
        # Obtaining the type of the subscript
        int_357057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 8), 'int')
        # Getting the type of 'self' (line 422)
        self_357058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'self')
        # Obtaining the member 'shape' of a type (line 422)
        shape_357059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 15), self_357058, 'shape')
        # Obtaining the member '__getitem__' of a type (line 422)
        getitem___357060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), shape_357059, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 422)
        subscript_call_result_357061 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), getitem___357060, int_357057)
        
        # Assigning a type to the variable 'tuple_var_assignment_356177' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'tuple_var_assignment_356177', subscript_call_result_357061)
        
        # Assigning a Name to a Name (line 422):
        # Getting the type of 'tuple_var_assignment_356176' (line 422)
        tuple_var_assignment_356176_357062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'tuple_var_assignment_356176')
        # Assigning a type to the variable 'M' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'M', tuple_var_assignment_356176_357062)
        
        # Assigning a Name to a Name (line 422):
        # Getting the type of 'tuple_var_assignment_356177' (line 422)
        tuple_var_assignment_356177_357063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'tuple_var_assignment_356177')
        # Assigning a type to the variable 'N' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'N', tuple_var_assignment_356177_357063)
        
        
        # Getting the type of 'other' (line 424)
        other_357064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'other')
        # Obtaining the member '__class__' of a type (line 424)
        class___357065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 11), other_357064, '__class__')
        # Getting the type of 'np' (line 424)
        np_357066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'np')
        # Obtaining the member 'ndarray' of a type (line 424)
        ndarray_357067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 30), np_357066, 'ndarray')
        # Applying the binary operator 'is' (line 424)
        result_is__357068 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 11), 'is', class___357065, ndarray_357067)
        
        # Testing the type of an if condition (line 424)
        if_condition_357069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 8), result_is__357068)
        # Assigning a type to the variable 'if_condition_357069' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'if_condition_357069', if_condition_357069)
        # SSA begins for if statement (line 424)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 426)
        other_357070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'other')
        # Obtaining the member 'shape' of a type (line 426)
        shape_357071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 15), other_357070, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 426)
        tuple_357072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 426)
        # Adding element type (line 426)
        # Getting the type of 'N' (line 426)
        N_357073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 31), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 31), tuple_357072, N_357073)
        
        # Applying the binary operator '==' (line 426)
        result_eq_357074 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 15), '==', shape_357071, tuple_357072)
        
        # Testing the type of an if condition (line 426)
        if_condition_357075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 12), result_eq_357074)
        # Assigning a type to the variable 'if_condition_357075' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'if_condition_357075', if_condition_357075)
        # SSA begins for if statement (line 426)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _mul_vector(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'other' (line 427)
        other_357078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 40), 'other', False)
        # Processing the call keyword arguments (line 427)
        kwargs_357079 = {}
        # Getting the type of 'self' (line 427)
        self_357076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 23), 'self', False)
        # Obtaining the member '_mul_vector' of a type (line 427)
        _mul_vector_357077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 23), self_357076, '_mul_vector')
        # Calling _mul_vector(args, kwargs) (line 427)
        _mul_vector_call_result_357080 = invoke(stypy.reporting.localization.Localization(__file__, 427, 23), _mul_vector_357077, *[other_357078], **kwargs_357079)
        
        # Assigning a type to the variable 'stypy_return_type' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'stypy_return_type', _mul_vector_call_result_357080)
        # SSA branch for the else part of an if statement (line 426)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'other' (line 428)
        other_357081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 17), 'other')
        # Obtaining the member 'shape' of a type (line 428)
        shape_357082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 17), other_357081, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 428)
        tuple_357083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 428)
        # Adding element type (line 428)
        # Getting the type of 'N' (line 428)
        N_357084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 33), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 33), tuple_357083, N_357084)
        # Adding element type (line 428)
        int_357085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 33), tuple_357083, int_357085)
        
        # Applying the binary operator '==' (line 428)
        result_eq_357086 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 17), '==', shape_357082, tuple_357083)
        
        # Testing the type of an if condition (line 428)
        if_condition_357087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 17), result_eq_357086)
        # Assigning a type to the variable 'if_condition_357087' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 17), 'if_condition_357087', if_condition_357087)
        # SSA begins for if statement (line 428)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to reshape(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'M' (line 429)
        M_357097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 63), 'M', False)
        int_357098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 66), 'int')
        # Processing the call keyword arguments (line 429)
        kwargs_357099 = {}
        
        # Call to _mul_vector(...): (line 429)
        # Processing the call arguments (line 429)
        
        # Call to ravel(...): (line 429)
        # Processing the call keyword arguments (line 429)
        kwargs_357092 = {}
        # Getting the type of 'other' (line 429)
        other_357090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 40), 'other', False)
        # Obtaining the member 'ravel' of a type (line 429)
        ravel_357091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 40), other_357090, 'ravel')
        # Calling ravel(args, kwargs) (line 429)
        ravel_call_result_357093 = invoke(stypy.reporting.localization.Localization(__file__, 429, 40), ravel_357091, *[], **kwargs_357092)
        
        # Processing the call keyword arguments (line 429)
        kwargs_357094 = {}
        # Getting the type of 'self' (line 429)
        self_357088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'self', False)
        # Obtaining the member '_mul_vector' of a type (line 429)
        _mul_vector_357089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 23), self_357088, '_mul_vector')
        # Calling _mul_vector(args, kwargs) (line 429)
        _mul_vector_call_result_357095 = invoke(stypy.reporting.localization.Localization(__file__, 429, 23), _mul_vector_357089, *[ravel_call_result_357093], **kwargs_357094)
        
        # Obtaining the member 'reshape' of a type (line 429)
        reshape_357096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 23), _mul_vector_call_result_357095, 'reshape')
        # Calling reshape(args, kwargs) (line 429)
        reshape_call_result_357100 = invoke(stypy.reporting.localization.Localization(__file__, 429, 23), reshape_357096, *[M_357097, int_357098], **kwargs_357099)
        
        # Assigning a type to the variable 'stypy_return_type' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'stypy_return_type', reshape_call_result_357100)
        # SSA branch for the else part of an if statement (line 428)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'other' (line 430)
        other_357101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 17), 'other')
        # Obtaining the member 'ndim' of a type (line 430)
        ndim_357102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 17), other_357101, 'ndim')
        int_357103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 31), 'int')
        # Applying the binary operator '==' (line 430)
        result_eq_357104 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 17), '==', ndim_357102, int_357103)
        
        
        
        # Obtaining the type of the subscript
        int_357105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 49), 'int')
        # Getting the type of 'other' (line 430)
        other_357106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 37), 'other')
        # Obtaining the member 'shape' of a type (line 430)
        shape_357107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 37), other_357106, 'shape')
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___357108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 37), shape_357107, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_357109 = invoke(stypy.reporting.localization.Localization(__file__, 430, 37), getitem___357108, int_357105)
        
        # Getting the type of 'N' (line 430)
        N_357110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 55), 'N')
        # Applying the binary operator '==' (line 430)
        result_eq_357111 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 37), '==', subscript_call_result_357109, N_357110)
        
        # Applying the binary operator 'and' (line 430)
        result_and_keyword_357112 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 17), 'and', result_eq_357104, result_eq_357111)
        
        # Testing the type of an if condition (line 430)
        if_condition_357113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 17), result_and_keyword_357112)
        # Assigning a type to the variable 'if_condition_357113' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 17), 'if_condition_357113', if_condition_357113)
        # SSA begins for if statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _mul_multivector(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'other' (line 431)
        other_357116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 45), 'other', False)
        # Processing the call keyword arguments (line 431)
        kwargs_357117 = {}
        # Getting the type of 'self' (line 431)
        self_357114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'self', False)
        # Obtaining the member '_mul_multivector' of a type (line 431)
        _mul_multivector_357115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 23), self_357114, '_mul_multivector')
        # Calling _mul_multivector(args, kwargs) (line 431)
        _mul_multivector_call_result_357118 = invoke(stypy.reporting.localization.Localization(__file__, 431, 23), _mul_multivector_357115, *[other_357116], **kwargs_357117)
        
        # Assigning a type to the variable 'stypy_return_type' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'stypy_return_type', _mul_multivector_call_result_357118)
        # SSA join for if statement (line 430)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 428)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 426)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 424)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isscalarlike(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'other' (line 433)
        other_357120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 24), 'other', False)
        # Processing the call keyword arguments (line 433)
        kwargs_357121 = {}
        # Getting the type of 'isscalarlike' (line 433)
        isscalarlike_357119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 433)
        isscalarlike_call_result_357122 = invoke(stypy.reporting.localization.Localization(__file__, 433, 11), isscalarlike_357119, *[other_357120], **kwargs_357121)
        
        # Testing the type of an if condition (line 433)
        if_condition_357123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 8), isscalarlike_call_result_357122)
        # Assigning a type to the variable 'if_condition_357123' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'if_condition_357123', if_condition_357123)
        # SSA begins for if statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _mul_scalar(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'other' (line 435)
        other_357126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 36), 'other', False)
        # Processing the call keyword arguments (line 435)
        kwargs_357127 = {}
        # Getting the type of 'self' (line 435)
        self_357124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 'self', False)
        # Obtaining the member '_mul_scalar' of a type (line 435)
        _mul_scalar_357125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 19), self_357124, '_mul_scalar')
        # Calling _mul_scalar(args, kwargs) (line 435)
        _mul_scalar_call_result_357128 = invoke(stypy.reporting.localization.Localization(__file__, 435, 19), _mul_scalar_357125, *[other_357126], **kwargs_357127)
        
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'stypy_return_type', _mul_scalar_call_result_357128)
        # SSA join for if statement (line 433)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to issparse(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'other' (line 437)
        other_357130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'other', False)
        # Processing the call keyword arguments (line 437)
        kwargs_357131 = {}
        # Getting the type of 'issparse' (line 437)
        issparse_357129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'issparse', False)
        # Calling issparse(args, kwargs) (line 437)
        issparse_call_result_357132 = invoke(stypy.reporting.localization.Localization(__file__, 437, 11), issparse_357129, *[other_357130], **kwargs_357131)
        
        # Testing the type of an if condition (line 437)
        if_condition_357133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), issparse_call_result_357132)
        # Assigning a type to the variable 'if_condition_357133' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'if_condition_357133', if_condition_357133)
        # SSA begins for if statement (line 437)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_357134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 26), 'int')
        # Getting the type of 'self' (line 438)
        self_357135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'self')
        # Obtaining the member 'shape' of a type (line 438)
        shape_357136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 15), self_357135, 'shape')
        # Obtaining the member '__getitem__' of a type (line 438)
        getitem___357137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 15), shape_357136, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 438)
        subscript_call_result_357138 = invoke(stypy.reporting.localization.Localization(__file__, 438, 15), getitem___357137, int_357134)
        
        
        # Obtaining the type of the subscript
        int_357139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 44), 'int')
        # Getting the type of 'other' (line 438)
        other_357140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'other')
        # Obtaining the member 'shape' of a type (line 438)
        shape_357141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 32), other_357140, 'shape')
        # Obtaining the member '__getitem__' of a type (line 438)
        getitem___357142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 32), shape_357141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 438)
        subscript_call_result_357143 = invoke(stypy.reporting.localization.Localization(__file__, 438, 32), getitem___357142, int_357139)
        
        # Applying the binary operator '!=' (line 438)
        result_ne_357144 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 15), '!=', subscript_call_result_357138, subscript_call_result_357143)
        
        # Testing the type of an if condition (line 438)
        if_condition_357145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 12), result_ne_357144)
        # Assigning a type to the variable 'if_condition_357145' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'if_condition_357145', if_condition_357145)
        # SSA begins for if statement (line 438)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 439)
        # Processing the call arguments (line 439)
        str_357147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 33), 'str', 'dimension mismatch')
        # Processing the call keyword arguments (line 439)
        kwargs_357148 = {}
        # Getting the type of 'ValueError' (line 439)
        ValueError_357146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 439)
        ValueError_call_result_357149 = invoke(stypy.reporting.localization.Localization(__file__, 439, 22), ValueError_357146, *[str_357147], **kwargs_357148)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 439, 16), ValueError_call_result_357149, 'raise parameter', BaseException)
        # SSA join for if statement (line 438)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _mul_sparse_matrix(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'other' (line 440)
        other_357152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 43), 'other', False)
        # Processing the call keyword arguments (line 440)
        kwargs_357153 = {}
        # Getting the type of 'self' (line 440)
        self_357150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), 'self', False)
        # Obtaining the member '_mul_sparse_matrix' of a type (line 440)
        _mul_sparse_matrix_357151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 19), self_357150, '_mul_sparse_matrix')
        # Calling _mul_sparse_matrix(args, kwargs) (line 440)
        _mul_sparse_matrix_call_result_357154 = invoke(stypy.reporting.localization.Localization(__file__, 440, 19), _mul_sparse_matrix_357151, *[other_357152], **kwargs_357153)
        
        # Assigning a type to the variable 'stypy_return_type' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'stypy_return_type', _mul_sparse_matrix_call_result_357154)
        # SSA join for if statement (line 437)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 443):
        
        # Assigning a Call to a Name (line 443):
        
        # Call to asanyarray(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'other' (line 443)
        other_357157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 32), 'other', False)
        # Processing the call keyword arguments (line 443)
        kwargs_357158 = {}
        # Getting the type of 'np' (line 443)
        np_357155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 18), 'np', False)
        # Obtaining the member 'asanyarray' of a type (line 443)
        asanyarray_357156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 18), np_357155, 'asanyarray')
        # Calling asanyarray(args, kwargs) (line 443)
        asanyarray_call_result_357159 = invoke(stypy.reporting.localization.Localization(__file__, 443, 18), asanyarray_357156, *[other_357157], **kwargs_357158)
        
        # Assigning a type to the variable 'other_a' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'other_a', asanyarray_call_result_357159)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'other_a' (line 445)
        other_a_357160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'other_a')
        # Obtaining the member 'ndim' of a type (line 445)
        ndim_357161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 11), other_a_357160, 'ndim')
        int_357162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 27), 'int')
        # Applying the binary operator '==' (line 445)
        result_eq_357163 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 11), '==', ndim_357161, int_357162)
        
        
        # Getting the type of 'other_a' (line 445)
        other_a_357164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 33), 'other_a')
        # Obtaining the member 'dtype' of a type (line 445)
        dtype_357165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 33), other_a_357164, 'dtype')
        # Getting the type of 'np' (line 445)
        np_357166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 50), 'np')
        # Obtaining the member 'object_' of a type (line 445)
        object__357167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 50), np_357166, 'object_')
        # Applying the binary operator '==' (line 445)
        result_eq_357168 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 33), '==', dtype_357165, object__357167)
        
        # Applying the binary operator 'and' (line 445)
        result_and_keyword_357169 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 11), 'and', result_eq_357163, result_eq_357168)
        
        # Testing the type of an if condition (line 445)
        if_condition_357170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), result_and_keyword_357169)
        # Assigning a type to the variable 'if_condition_357170' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_357170', if_condition_357170)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 448)
        NotImplemented_357171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'stypy_return_type', NotImplemented_357171)
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 450)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'other' (line 451)
        other_357172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'other')
        # Obtaining the member 'shape' of a type (line 451)
        shape_357173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), other_357172, 'shape')
        # SSA branch for the except part of a try statement (line 450)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 450)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 453):
        
        # Assigning a Name to a Name (line 453):
        # Getting the type of 'other_a' (line 453)
        other_a_357174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 20), 'other_a')
        # Assigning a type to the variable 'other' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'other', other_a_357174)
        # SSA join for try-except statement (line 450)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'other' (line 455)
        other_357175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 'other')
        # Obtaining the member 'ndim' of a type (line 455)
        ndim_357176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 11), other_357175, 'ndim')
        int_357177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 25), 'int')
        # Applying the binary operator '==' (line 455)
        result_eq_357178 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), '==', ndim_357176, int_357177)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'other' (line 455)
        other_357179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 30), 'other')
        # Obtaining the member 'ndim' of a type (line 455)
        ndim_357180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 30), other_357179, 'ndim')
        int_357181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 44), 'int')
        # Applying the binary operator '==' (line 455)
        result_eq_357182 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 30), '==', ndim_357180, int_357181)
        
        
        
        # Obtaining the type of the subscript
        int_357183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 62), 'int')
        # Getting the type of 'other' (line 455)
        other_357184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 50), 'other')
        # Obtaining the member 'shape' of a type (line 455)
        shape_357185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 50), other_357184, 'shape')
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___357186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 50), shape_357185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_357187 = invoke(stypy.reporting.localization.Localization(__file__, 455, 50), getitem___357186, int_357183)
        
        int_357188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 68), 'int')
        # Applying the binary operator '==' (line 455)
        result_eq_357189 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 50), '==', subscript_call_result_357187, int_357188)
        
        # Applying the binary operator 'and' (line 455)
        result_and_keyword_357190 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 30), 'and', result_eq_357182, result_eq_357189)
        
        # Applying the binary operator 'or' (line 455)
        result_or_keyword_357191 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), 'or', result_eq_357178, result_and_keyword_357190)
        
        # Testing the type of an if condition (line 455)
        if_condition_357192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 8), result_or_keyword_357191)
        # Assigning a type to the variable 'if_condition_357192' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'if_condition_357192', if_condition_357192)
        # SSA begins for if statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'other' (line 457)
        other_357193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'other')
        # Obtaining the member 'shape' of a type (line 457)
        shape_357194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), other_357193, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 457)
        tuple_357195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 457)
        # Adding element type (line 457)
        # Getting the type of 'N' (line 457)
        N_357196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 31), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 31), tuple_357195, N_357196)
        
        # Applying the binary operator '!=' (line 457)
        result_ne_357197 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 15), '!=', shape_357194, tuple_357195)
        
        
        # Getting the type of 'other' (line 457)
        other_357198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 39), 'other')
        # Obtaining the member 'shape' of a type (line 457)
        shape_357199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 39), other_357198, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 457)
        tuple_357200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 457)
        # Adding element type (line 457)
        # Getting the type of 'N' (line 457)
        N_357201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 55), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 55), tuple_357200, N_357201)
        # Adding element type (line 457)
        int_357202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 55), tuple_357200, int_357202)
        
        # Applying the binary operator '!=' (line 457)
        result_ne_357203 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 39), '!=', shape_357199, tuple_357200)
        
        # Applying the binary operator 'and' (line 457)
        result_and_keyword_357204 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 15), 'and', result_ne_357197, result_ne_357203)
        
        # Testing the type of an if condition (line 457)
        if_condition_357205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 12), result_and_keyword_357204)
        # Assigning a type to the variable 'if_condition_357205' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'if_condition_357205', if_condition_357205)
        # SSA begins for if statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 458)
        # Processing the call arguments (line 458)
        str_357207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 33), 'str', 'dimension mismatch')
        # Processing the call keyword arguments (line 458)
        kwargs_357208 = {}
        # Getting the type of 'ValueError' (line 458)
        ValueError_357206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 458)
        ValueError_call_result_357209 = invoke(stypy.reporting.localization.Localization(__file__, 458, 22), ValueError_357206, *[str_357207], **kwargs_357208)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 458, 16), ValueError_call_result_357209, 'raise parameter', BaseException)
        # SSA join for if statement (line 457)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 460):
        
        # Assigning a Call to a Name (line 460):
        
        # Call to _mul_vector(...): (line 460)
        # Processing the call arguments (line 460)
        
        # Call to ravel(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'other' (line 460)
        other_357214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 47), 'other', False)
        # Processing the call keyword arguments (line 460)
        kwargs_357215 = {}
        # Getting the type of 'np' (line 460)
        np_357212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 38), 'np', False)
        # Obtaining the member 'ravel' of a type (line 460)
        ravel_357213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 38), np_357212, 'ravel')
        # Calling ravel(args, kwargs) (line 460)
        ravel_call_result_357216 = invoke(stypy.reporting.localization.Localization(__file__, 460, 38), ravel_357213, *[other_357214], **kwargs_357215)
        
        # Processing the call keyword arguments (line 460)
        kwargs_357217 = {}
        # Getting the type of 'self' (line 460)
        self_357210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 21), 'self', False)
        # Obtaining the member '_mul_vector' of a type (line 460)
        _mul_vector_357211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 21), self_357210, '_mul_vector')
        # Calling _mul_vector(args, kwargs) (line 460)
        _mul_vector_call_result_357218 = invoke(stypy.reporting.localization.Localization(__file__, 460, 21), _mul_vector_357211, *[ravel_call_result_357216], **kwargs_357217)
        
        # Assigning a type to the variable 'result' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'result', _mul_vector_call_result_357218)
        
        
        # Call to isinstance(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'other' (line 462)
        other_357220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 26), 'other', False)
        # Getting the type of 'np' (line 462)
        np_357221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 33), 'np', False)
        # Obtaining the member 'matrix' of a type (line 462)
        matrix_357222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 33), np_357221, 'matrix')
        # Processing the call keyword arguments (line 462)
        kwargs_357223 = {}
        # Getting the type of 'isinstance' (line 462)
        isinstance_357219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 462)
        isinstance_call_result_357224 = invoke(stypy.reporting.localization.Localization(__file__, 462, 15), isinstance_357219, *[other_357220, matrix_357222], **kwargs_357223)
        
        # Testing the type of an if condition (line 462)
        if_condition_357225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 12), isinstance_call_result_357224)
        # Assigning a type to the variable 'if_condition_357225' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'if_condition_357225', if_condition_357225)
        # SSA begins for if statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to asmatrix(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'result' (line 463)
        result_357228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 37), 'result', False)
        # Processing the call keyword arguments (line 463)
        kwargs_357229 = {}
        # Getting the type of 'np' (line 463)
        np_357226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 25), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 463)
        asmatrix_357227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 25), np_357226, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 463)
        asmatrix_call_result_357230 = invoke(stypy.reporting.localization.Localization(__file__, 463, 25), asmatrix_357227, *[result_357228], **kwargs_357229)
        
        # Assigning a type to the variable 'result' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'result', asmatrix_call_result_357230)
        # SSA join for if statement (line 462)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'other' (line 465)
        other_357231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'other')
        # Obtaining the member 'ndim' of a type (line 465)
        ndim_357232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 15), other_357231, 'ndim')
        int_357233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 29), 'int')
        # Applying the binary operator '==' (line 465)
        result_eq_357234 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 15), '==', ndim_357232, int_357233)
        
        
        
        # Obtaining the type of the subscript
        int_357235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 47), 'int')
        # Getting the type of 'other' (line 465)
        other_357236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 35), 'other')
        # Obtaining the member 'shape' of a type (line 465)
        shape_357237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 35), other_357236, 'shape')
        # Obtaining the member '__getitem__' of a type (line 465)
        getitem___357238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 35), shape_357237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 465)
        subscript_call_result_357239 = invoke(stypy.reporting.localization.Localization(__file__, 465, 35), getitem___357238, int_357235)
        
        int_357240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 53), 'int')
        # Applying the binary operator '==' (line 465)
        result_eq_357241 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 35), '==', subscript_call_result_357239, int_357240)
        
        # Applying the binary operator 'and' (line 465)
        result_and_keyword_357242 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 15), 'and', result_eq_357234, result_eq_357241)
        
        # Testing the type of an if condition (line 465)
        if_condition_357243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 12), result_and_keyword_357242)
        # Assigning a type to the variable 'if_condition_357243' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'if_condition_357243', if_condition_357243)
        # SSA begins for if statement (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to reshape(...): (line 467)
        # Processing the call arguments (line 467)
        int_357246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 40), 'int')
        int_357247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 44), 'int')
        # Processing the call keyword arguments (line 467)
        kwargs_357248 = {}
        # Getting the type of 'result' (line 467)
        result_357244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 25), 'result', False)
        # Obtaining the member 'reshape' of a type (line 467)
        reshape_357245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 25), result_357244, 'reshape')
        # Calling reshape(args, kwargs) (line 467)
        reshape_call_result_357249 = invoke(stypy.reporting.localization.Localization(__file__, 467, 25), reshape_357245, *[int_357246, int_357247], **kwargs_357248)
        
        # Assigning a type to the variable 'result' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'result', reshape_call_result_357249)
        # SSA join for if statement (line 465)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 469)
        result_357250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 19), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'stypy_return_type', result_357250)
        # SSA branch for the else part of an if statement (line 455)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'other' (line 471)
        other_357251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 13), 'other')
        # Obtaining the member 'ndim' of a type (line 471)
        ndim_357252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 13), other_357251, 'ndim')
        int_357253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 27), 'int')
        # Applying the binary operator '==' (line 471)
        result_eq_357254 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 13), '==', ndim_357252, int_357253)
        
        # Testing the type of an if condition (line 471)
        if_condition_357255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 13), result_eq_357254)
        # Assigning a type to the variable 'if_condition_357255' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 13), 'if_condition_357255', if_condition_357255)
        # SSA begins for if statement (line 471)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_357256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 27), 'int')
        # Getting the type of 'other' (line 475)
        other_357257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 'other')
        # Obtaining the member 'shape' of a type (line 475)
        shape_357258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 15), other_357257, 'shape')
        # Obtaining the member '__getitem__' of a type (line 475)
        getitem___357259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 15), shape_357258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 475)
        subscript_call_result_357260 = invoke(stypy.reporting.localization.Localization(__file__, 475, 15), getitem___357259, int_357256)
        
        
        # Obtaining the type of the subscript
        int_357261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 44), 'int')
        # Getting the type of 'self' (line 475)
        self_357262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 33), 'self')
        # Obtaining the member 'shape' of a type (line 475)
        shape_357263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 33), self_357262, 'shape')
        # Obtaining the member '__getitem__' of a type (line 475)
        getitem___357264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 33), shape_357263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 475)
        subscript_call_result_357265 = invoke(stypy.reporting.localization.Localization(__file__, 475, 33), getitem___357264, int_357261)
        
        # Applying the binary operator '!=' (line 475)
        result_ne_357266 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 15), '!=', subscript_call_result_357260, subscript_call_result_357265)
        
        # Testing the type of an if condition (line 475)
        if_condition_357267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 12), result_ne_357266)
        # Assigning a type to the variable 'if_condition_357267' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'if_condition_357267', if_condition_357267)
        # SSA begins for if statement (line 475)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 476)
        # Processing the call arguments (line 476)
        str_357269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 33), 'str', 'dimension mismatch')
        # Processing the call keyword arguments (line 476)
        kwargs_357270 = {}
        # Getting the type of 'ValueError' (line 476)
        ValueError_357268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 476)
        ValueError_call_result_357271 = invoke(stypy.reporting.localization.Localization(__file__, 476, 22), ValueError_357268, *[str_357269], **kwargs_357270)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 476, 16), ValueError_call_result_357271, 'raise parameter', BaseException)
        # SSA join for if statement (line 475)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 478):
        
        # Assigning a Call to a Name (line 478):
        
        # Call to _mul_multivector(...): (line 478)
        # Processing the call arguments (line 478)
        
        # Call to asarray(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'other' (line 478)
        other_357276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 54), 'other', False)
        # Processing the call keyword arguments (line 478)
        kwargs_357277 = {}
        # Getting the type of 'np' (line 478)
        np_357274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 43), 'np', False)
        # Obtaining the member 'asarray' of a type (line 478)
        asarray_357275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 43), np_357274, 'asarray')
        # Calling asarray(args, kwargs) (line 478)
        asarray_call_result_357278 = invoke(stypy.reporting.localization.Localization(__file__, 478, 43), asarray_357275, *[other_357276], **kwargs_357277)
        
        # Processing the call keyword arguments (line 478)
        kwargs_357279 = {}
        # Getting the type of 'self' (line 478)
        self_357272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'self', False)
        # Obtaining the member '_mul_multivector' of a type (line 478)
        _mul_multivector_357273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 21), self_357272, '_mul_multivector')
        # Calling _mul_multivector(args, kwargs) (line 478)
        _mul_multivector_call_result_357280 = invoke(stypy.reporting.localization.Localization(__file__, 478, 21), _mul_multivector_357273, *[asarray_call_result_357278], **kwargs_357279)
        
        # Assigning a type to the variable 'result' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'result', _mul_multivector_call_result_357280)
        
        
        # Call to isinstance(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'other' (line 480)
        other_357282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 26), 'other', False)
        # Getting the type of 'np' (line 480)
        np_357283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 33), 'np', False)
        # Obtaining the member 'matrix' of a type (line 480)
        matrix_357284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 33), np_357283, 'matrix')
        # Processing the call keyword arguments (line 480)
        kwargs_357285 = {}
        # Getting the type of 'isinstance' (line 480)
        isinstance_357281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 480)
        isinstance_call_result_357286 = invoke(stypy.reporting.localization.Localization(__file__, 480, 15), isinstance_357281, *[other_357282, matrix_357284], **kwargs_357285)
        
        # Testing the type of an if condition (line 480)
        if_condition_357287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 12), isinstance_call_result_357286)
        # Assigning a type to the variable 'if_condition_357287' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'if_condition_357287', if_condition_357287)
        # SSA begins for if statement (line 480)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 481):
        
        # Assigning a Call to a Name (line 481):
        
        # Call to asmatrix(...): (line 481)
        # Processing the call arguments (line 481)
        # Getting the type of 'result' (line 481)
        result_357290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 37), 'result', False)
        # Processing the call keyword arguments (line 481)
        kwargs_357291 = {}
        # Getting the type of 'np' (line 481)
        np_357288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 25), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 481)
        asmatrix_357289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 25), np_357288, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 481)
        asmatrix_call_result_357292 = invoke(stypy.reporting.localization.Localization(__file__, 481, 25), asmatrix_357289, *[result_357290], **kwargs_357291)
        
        # Assigning a type to the variable 'result' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 16), 'result', asmatrix_call_result_357292)
        # SSA join for if statement (line 480)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 483)
        result_357293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 19), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'stypy_return_type', result_357293)
        # SSA branch for the else part of an if statement (line 471)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 486)
        # Processing the call arguments (line 486)
        str_357295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 29), 'str', 'could not interpret dimensions')
        # Processing the call keyword arguments (line 486)
        kwargs_357296 = {}
        # Getting the type of 'ValueError' (line 486)
        ValueError_357294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 486)
        ValueError_call_result_357297 = invoke(stypy.reporting.localization.Localization(__file__, 486, 18), ValueError_357294, *[str_357295], **kwargs_357296)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 486, 12), ValueError_call_result_357297, 'raise parameter', BaseException)
        # SSA join for if statement (line 471)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 455)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 413)
        stypy_return_type_357298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357298)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_357298


    @norecursion
    def _mul_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_scalar'
        module_type_store = module_type_store.open_function_context('_mul_scalar', 489, 4, False)
        # Assigning a type to the variable 'self' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_function_name', 'spmatrix._mul_scalar')
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._mul_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._mul_scalar', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_scalar', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_scalar(...)' code ##################

        
        # Call to _mul_scalar(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'other' (line 490)
        other_357304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 40), 'other', False)
        # Processing the call keyword arguments (line 490)
        kwargs_357305 = {}
        
        # Call to tocsr(...): (line 490)
        # Processing the call keyword arguments (line 490)
        kwargs_357301 = {}
        # Getting the type of 'self' (line 490)
        self_357299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 490)
        tocsr_357300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 15), self_357299, 'tocsr')
        # Calling tocsr(args, kwargs) (line 490)
        tocsr_call_result_357302 = invoke(stypy.reporting.localization.Localization(__file__, 490, 15), tocsr_357300, *[], **kwargs_357301)
        
        # Obtaining the member '_mul_scalar' of a type (line 490)
        _mul_scalar_357303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 15), tocsr_call_result_357302, '_mul_scalar')
        # Calling _mul_scalar(args, kwargs) (line 490)
        _mul_scalar_call_result_357306 = invoke(stypy.reporting.localization.Localization(__file__, 490, 15), _mul_scalar_357303, *[other_357304], **kwargs_357305)
        
        # Assigning a type to the variable 'stypy_return_type' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'stypy_return_type', _mul_scalar_call_result_357306)
        
        # ################# End of '_mul_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 489)
        stypy_return_type_357307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357307)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_scalar'
        return stypy_return_type_357307


    @norecursion
    def _mul_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_vector'
        module_type_store = module_type_store.open_function_context('_mul_vector', 492, 4, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._mul_vector.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_function_name', 'spmatrix._mul_vector')
        spmatrix._mul_vector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._mul_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._mul_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._mul_vector', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_vector', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_vector(...)' code ##################

        
        # Call to _mul_vector(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'other' (line 493)
        other_357313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 40), 'other', False)
        # Processing the call keyword arguments (line 493)
        kwargs_357314 = {}
        
        # Call to tocsr(...): (line 493)
        # Processing the call keyword arguments (line 493)
        kwargs_357310 = {}
        # Getting the type of 'self' (line 493)
        self_357308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 493)
        tocsr_357309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 15), self_357308, 'tocsr')
        # Calling tocsr(args, kwargs) (line 493)
        tocsr_call_result_357311 = invoke(stypy.reporting.localization.Localization(__file__, 493, 15), tocsr_357309, *[], **kwargs_357310)
        
        # Obtaining the member '_mul_vector' of a type (line 493)
        _mul_vector_357312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 15), tocsr_call_result_357311, '_mul_vector')
        # Calling _mul_vector(args, kwargs) (line 493)
        _mul_vector_call_result_357315 = invoke(stypy.reporting.localization.Localization(__file__, 493, 15), _mul_vector_357312, *[other_357313], **kwargs_357314)
        
        # Assigning a type to the variable 'stypy_return_type' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'stypy_return_type', _mul_vector_call_result_357315)
        
        # ################# End of '_mul_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_357316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_vector'
        return stypy_return_type_357316


    @norecursion
    def _mul_multivector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_multivector'
        module_type_store = module_type_store.open_function_context('_mul_multivector', 495, 4, False)
        # Assigning a type to the variable 'self' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_function_name', 'spmatrix._mul_multivector')
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._mul_multivector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._mul_multivector', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_multivector', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_multivector(...)' code ##################

        
        # Call to _mul_multivector(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'other' (line 496)
        other_357322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 45), 'other', False)
        # Processing the call keyword arguments (line 496)
        kwargs_357323 = {}
        
        # Call to tocsr(...): (line 496)
        # Processing the call keyword arguments (line 496)
        kwargs_357319 = {}
        # Getting the type of 'self' (line 496)
        self_357317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 496)
        tocsr_357318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 15), self_357317, 'tocsr')
        # Calling tocsr(args, kwargs) (line 496)
        tocsr_call_result_357320 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), tocsr_357318, *[], **kwargs_357319)
        
        # Obtaining the member '_mul_multivector' of a type (line 496)
        _mul_multivector_357321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 15), tocsr_call_result_357320, '_mul_multivector')
        # Calling _mul_multivector(args, kwargs) (line 496)
        _mul_multivector_call_result_357324 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), _mul_multivector_357321, *[other_357322], **kwargs_357323)
        
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', _mul_multivector_call_result_357324)
        
        # ################# End of '_mul_multivector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_multivector' in the type store
        # Getting the type of 'stypy_return_type' (line 495)
        stypy_return_type_357325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357325)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_multivector'
        return stypy_return_type_357325


    @norecursion
    def _mul_sparse_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_sparse_matrix'
        module_type_store = module_type_store.open_function_context('_mul_sparse_matrix', 498, 4, False)
        # Assigning a type to the variable 'self' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_function_name', 'spmatrix._mul_sparse_matrix')
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._mul_sparse_matrix.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._mul_sparse_matrix', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_sparse_matrix', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_sparse_matrix(...)' code ##################

        
        # Call to _mul_sparse_matrix(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'other' (line 499)
        other_357331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 47), 'other', False)
        # Processing the call keyword arguments (line 499)
        kwargs_357332 = {}
        
        # Call to tocsr(...): (line 499)
        # Processing the call keyword arguments (line 499)
        kwargs_357328 = {}
        # Getting the type of 'self' (line 499)
        self_357326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 499)
        tocsr_357327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), self_357326, 'tocsr')
        # Calling tocsr(args, kwargs) (line 499)
        tocsr_call_result_357329 = invoke(stypy.reporting.localization.Localization(__file__, 499, 15), tocsr_357327, *[], **kwargs_357328)
        
        # Obtaining the member '_mul_sparse_matrix' of a type (line 499)
        _mul_sparse_matrix_357330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), tocsr_call_result_357329, '_mul_sparse_matrix')
        # Calling _mul_sparse_matrix(args, kwargs) (line 499)
        _mul_sparse_matrix_call_result_357333 = invoke(stypy.reporting.localization.Localization(__file__, 499, 15), _mul_sparse_matrix_357330, *[other_357331], **kwargs_357332)
        
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'stypy_return_type', _mul_sparse_matrix_call_result_357333)
        
        # ################# End of '_mul_sparse_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_sparse_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 498)
        stypy_return_type_357334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357334)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_sparse_matrix'
        return stypy_return_type_357334


    @norecursion
    def __rmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmul__'
        module_type_store = module_type_store.open_function_context('__rmul__', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__rmul__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__rmul__')
        spmatrix.__rmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__rmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__rmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__rmul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmul__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'other' (line 502)
        other_357336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 24), 'other', False)
        # Processing the call keyword arguments (line 502)
        kwargs_357337 = {}
        # Getting the type of 'isscalarlike' (line 502)
        isscalarlike_357335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 502)
        isscalarlike_call_result_357338 = invoke(stypy.reporting.localization.Localization(__file__, 502, 11), isscalarlike_357335, *[other_357336], **kwargs_357337)
        
        # Testing the type of an if condition (line 502)
        if_condition_357339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 8), isscalarlike_call_result_357338)
        # Assigning a type to the variable 'if_condition_357339' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'if_condition_357339', if_condition_357339)
        # SSA begins for if statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __mul__(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'other' (line 503)
        other_357342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 32), 'other', False)
        # Processing the call keyword arguments (line 503)
        kwargs_357343 = {}
        # Getting the type of 'self' (line 503)
        self_357340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 19), 'self', False)
        # Obtaining the member '__mul__' of a type (line 503)
        mul___357341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), self_357340, '__mul__')
        # Calling __mul__(args, kwargs) (line 503)
        mul___call_result_357344 = invoke(stypy.reporting.localization.Localization(__file__, 503, 19), mul___357341, *[other_357342], **kwargs_357343)
        
        # Assigning a type to the variable 'stypy_return_type' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'stypy_return_type', mul___call_result_357344)
        # SSA branch for the else part of an if statement (line 502)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 507):
        
        # Assigning a Call to a Name (line 507):
        
        # Call to transpose(...): (line 507)
        # Processing the call keyword arguments (line 507)
        kwargs_357347 = {}
        # Getting the type of 'other' (line 507)
        other_357345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 21), 'other', False)
        # Obtaining the member 'transpose' of a type (line 507)
        transpose_357346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 21), other_357345, 'transpose')
        # Calling transpose(args, kwargs) (line 507)
        transpose_call_result_357348 = invoke(stypy.reporting.localization.Localization(__file__, 507, 21), transpose_357346, *[], **kwargs_357347)
        
        # Assigning a type to the variable 'tr' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'tr', transpose_call_result_357348)
        # SSA branch for the except part of a try statement (line 506)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 506)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to transpose(...): (line 509)
        # Processing the call keyword arguments (line 509)
        kwargs_357355 = {}
        
        # Call to asarray(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'other' (line 509)
        other_357351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 32), 'other', False)
        # Processing the call keyword arguments (line 509)
        kwargs_357352 = {}
        # Getting the type of 'np' (line 509)
        np_357349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'np', False)
        # Obtaining the member 'asarray' of a type (line 509)
        asarray_357350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 21), np_357349, 'asarray')
        # Calling asarray(args, kwargs) (line 509)
        asarray_call_result_357353 = invoke(stypy.reporting.localization.Localization(__file__, 509, 21), asarray_357350, *[other_357351], **kwargs_357352)
        
        # Obtaining the member 'transpose' of a type (line 509)
        transpose_357354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 21), asarray_call_result_357353, 'transpose')
        # Calling transpose(args, kwargs) (line 509)
        transpose_call_result_357356 = invoke(stypy.reporting.localization.Localization(__file__, 509, 21), transpose_357354, *[], **kwargs_357355)
        
        # Assigning a type to the variable 'tr' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'tr', transpose_call_result_357356)
        # SSA join for try-except statement (line 506)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to transpose(...): (line 510)
        # Processing the call keyword arguments (line 510)
        kwargs_357364 = {}
        
        # Call to transpose(...): (line 510)
        # Processing the call keyword arguments (line 510)
        kwargs_357359 = {}
        # Getting the type of 'self' (line 510)
        self_357357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 20), 'self', False)
        # Obtaining the member 'transpose' of a type (line 510)
        transpose_357358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 20), self_357357, 'transpose')
        # Calling transpose(args, kwargs) (line 510)
        transpose_call_result_357360 = invoke(stypy.reporting.localization.Localization(__file__, 510, 20), transpose_357358, *[], **kwargs_357359)
        
        # Getting the type of 'tr' (line 510)
        tr_357361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 39), 'tr', False)
        # Applying the binary operator '*' (line 510)
        result_mul_357362 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 20), '*', transpose_call_result_357360, tr_357361)
        
        # Obtaining the member 'transpose' of a type (line 510)
        transpose_357363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 20), result_mul_357362, 'transpose')
        # Calling transpose(args, kwargs) (line 510)
        transpose_call_result_357365 = invoke(stypy.reporting.localization.Localization(__file__, 510, 20), transpose_357363, *[], **kwargs_357364)
        
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'stypy_return_type', transpose_call_result_357365)
        # SSA join for if statement (line 502)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__rmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_357366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmul__'
        return stypy_return_type_357366


    @norecursion
    def __matmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__matmul__'
        module_type_store = module_type_store.open_function_context('__matmul__', 516, 4, False)
        # Assigning a type to the variable 'self' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__matmul__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__matmul__')
        spmatrix.__matmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__matmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__matmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__matmul__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isscalarlike(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'other' (line 517)
        other_357368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 24), 'other', False)
        # Processing the call keyword arguments (line 517)
        kwargs_357369 = {}
        # Getting the type of 'isscalarlike' (line 517)
        isscalarlike_357367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 517)
        isscalarlike_call_result_357370 = invoke(stypy.reporting.localization.Localization(__file__, 517, 11), isscalarlike_357367, *[other_357368], **kwargs_357369)
        
        # Testing the type of an if condition (line 517)
        if_condition_357371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 8), isscalarlike_call_result_357370)
        # Assigning a type to the variable 'if_condition_357371' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'if_condition_357371', if_condition_357371)
        # SSA begins for if statement (line 517)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 518)
        # Processing the call arguments (line 518)
        str_357373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 29), 'str', "Scalar operands are not allowed, use '*' instead")
        # Processing the call keyword arguments (line 518)
        kwargs_357374 = {}
        # Getting the type of 'ValueError' (line 518)
        ValueError_357372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 518)
        ValueError_call_result_357375 = invoke(stypy.reporting.localization.Localization(__file__, 518, 18), ValueError_357372, *[str_357373], **kwargs_357374)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 518, 12), ValueError_call_result_357375, 'raise parameter', BaseException)
        # SSA join for if statement (line 517)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __mul__(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'other' (line 520)
        other_357378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 28), 'other', False)
        # Processing the call keyword arguments (line 520)
        kwargs_357379 = {}
        # Getting the type of 'self' (line 520)
        self_357376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), 'self', False)
        # Obtaining the member '__mul__' of a type (line 520)
        mul___357377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 15), self_357376, '__mul__')
        # Calling __mul__(args, kwargs) (line 520)
        mul___call_result_357380 = invoke(stypy.reporting.localization.Localization(__file__, 520, 15), mul___357377, *[other_357378], **kwargs_357379)
        
        # Assigning a type to the variable 'stypy_return_type' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'stypy_return_type', mul___call_result_357380)
        
        # ################# End of '__matmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__matmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 516)
        stypy_return_type_357381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__matmul__'
        return stypy_return_type_357381


    @norecursion
    def __rmatmul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmatmul__'
        module_type_store = module_type_store.open_function_context('__rmatmul__', 522, 4, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__rmatmul__')
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__rmatmul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__rmatmul__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isscalarlike(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'other' (line 523)
        other_357383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'other', False)
        # Processing the call keyword arguments (line 523)
        kwargs_357384 = {}
        # Getting the type of 'isscalarlike' (line 523)
        isscalarlike_357382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 523)
        isscalarlike_call_result_357385 = invoke(stypy.reporting.localization.Localization(__file__, 523, 11), isscalarlike_357382, *[other_357383], **kwargs_357384)
        
        # Testing the type of an if condition (line 523)
        if_condition_357386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 523, 8), isscalarlike_call_result_357385)
        # Assigning a type to the variable 'if_condition_357386' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'if_condition_357386', if_condition_357386)
        # SSA begins for if statement (line 523)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 524)
        # Processing the call arguments (line 524)
        str_357388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 29), 'str', "Scalar operands are not allowed, use '*' instead")
        # Processing the call keyword arguments (line 524)
        kwargs_357389 = {}
        # Getting the type of 'ValueError' (line 524)
        ValueError_357387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 524)
        ValueError_call_result_357390 = invoke(stypy.reporting.localization.Localization(__file__, 524, 18), ValueError_357387, *[str_357388], **kwargs_357389)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 524, 12), ValueError_call_result_357390, 'raise parameter', BaseException)
        # SSA join for if statement (line 523)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __rmul__(...): (line 526)
        # Processing the call arguments (line 526)
        # Getting the type of 'other' (line 526)
        other_357393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 29), 'other', False)
        # Processing the call keyword arguments (line 526)
        kwargs_357394 = {}
        # Getting the type of 'self' (line 526)
        self_357391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'self', False)
        # Obtaining the member '__rmul__' of a type (line 526)
        rmul___357392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), self_357391, '__rmul__')
        # Calling __rmul__(args, kwargs) (line 526)
        rmul___call_result_357395 = invoke(stypy.reporting.localization.Localization(__file__, 526, 15), rmul___357392, *[other_357393], **kwargs_357394)
        
        # Assigning a type to the variable 'stypy_return_type' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'stypy_return_type', rmul___call_result_357395)
        
        # ################# End of '__rmatmul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmatmul__' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_357396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357396)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmatmul__'
        return stypy_return_type_357396


    @norecursion
    def _divide(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 532)
        False_357397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 41), 'False')
        # Getting the type of 'False' (line 532)
        False_357398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 56), 'False')
        defaults = [False_357397, False_357398]
        # Create a new context for function '_divide'
        module_type_store = module_type_store.open_function_context('_divide', 532, 4, False)
        # Assigning a type to the variable 'self' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._divide.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._divide.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._divide.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._divide.__dict__.__setitem__('stypy_function_name', 'spmatrix._divide')
        spmatrix._divide.__dict__.__setitem__('stypy_param_names_list', ['other', 'true_divide', 'rdivide'])
        spmatrix._divide.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._divide.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._divide.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._divide.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._divide.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._divide.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._divide', ['other', 'true_divide', 'rdivide'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_divide', localization, ['other', 'true_divide', 'rdivide'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_divide(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'other' (line 533)
        other_357400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 24), 'other', False)
        # Processing the call keyword arguments (line 533)
        kwargs_357401 = {}
        # Getting the type of 'isscalarlike' (line 533)
        isscalarlike_357399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 533)
        isscalarlike_call_result_357402 = invoke(stypy.reporting.localization.Localization(__file__, 533, 11), isscalarlike_357399, *[other_357400], **kwargs_357401)
        
        # Testing the type of an if condition (line 533)
        if_condition_357403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 533, 8), isscalarlike_call_result_357402)
        # Assigning a type to the variable 'if_condition_357403' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'if_condition_357403', if_condition_357403)
        # SSA begins for if statement (line 533)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'rdivide' (line 534)
        rdivide_357404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 15), 'rdivide')
        # Testing the type of an if condition (line 534)
        if_condition_357405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 12), rdivide_357404)
        # Assigning a type to the variable 'if_condition_357405' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'if_condition_357405', if_condition_357405)
        # SSA begins for if statement (line 534)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'true_divide' (line 535)
        true_divide_357406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 19), 'true_divide')
        # Testing the type of an if condition (line 535)
        if_condition_357407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 16), true_divide_357406)
        # Assigning a type to the variable 'if_condition_357407' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 16), 'if_condition_357407', if_condition_357407)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to true_divide(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'other' (line 536)
        other_357410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 42), 'other', False)
        
        # Call to todense(...): (line 536)
        # Processing the call keyword arguments (line 536)
        kwargs_357413 = {}
        # Getting the type of 'self' (line 536)
        self_357411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 49), 'self', False)
        # Obtaining the member 'todense' of a type (line 536)
        todense_357412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 49), self_357411, 'todense')
        # Calling todense(args, kwargs) (line 536)
        todense_call_result_357414 = invoke(stypy.reporting.localization.Localization(__file__, 536, 49), todense_357412, *[], **kwargs_357413)
        
        # Processing the call keyword arguments (line 536)
        kwargs_357415 = {}
        # Getting the type of 'np' (line 536)
        np_357408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 27), 'np', False)
        # Obtaining the member 'true_divide' of a type (line 536)
        true_divide_357409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 27), np_357408, 'true_divide')
        # Calling true_divide(args, kwargs) (line 536)
        true_divide_call_result_357416 = invoke(stypy.reporting.localization.Localization(__file__, 536, 27), true_divide_357409, *[other_357410, todense_call_result_357414], **kwargs_357415)
        
        # Assigning a type to the variable 'stypy_return_type' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 20), 'stypy_return_type', true_divide_call_result_357416)
        # SSA branch for the else part of an if statement (line 535)
        module_type_store.open_ssa_branch('else')
        
        # Call to divide(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'other' (line 538)
        other_357419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 37), 'other', False)
        
        # Call to todense(...): (line 538)
        # Processing the call keyword arguments (line 538)
        kwargs_357422 = {}
        # Getting the type of 'self' (line 538)
        self_357420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 44), 'self', False)
        # Obtaining the member 'todense' of a type (line 538)
        todense_357421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 44), self_357420, 'todense')
        # Calling todense(args, kwargs) (line 538)
        todense_call_result_357423 = invoke(stypy.reporting.localization.Localization(__file__, 538, 44), todense_357421, *[], **kwargs_357422)
        
        # Processing the call keyword arguments (line 538)
        kwargs_357424 = {}
        # Getting the type of 'np' (line 538)
        np_357417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 27), 'np', False)
        # Obtaining the member 'divide' of a type (line 538)
        divide_357418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 27), np_357417, 'divide')
        # Calling divide(args, kwargs) (line 538)
        divide_call_result_357425 = invoke(stypy.reporting.localization.Localization(__file__, 538, 27), divide_357418, *[other_357419, todense_call_result_357423], **kwargs_357424)
        
        # Assigning a type to the variable 'stypy_return_type' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 20), 'stypy_return_type', divide_call_result_357425)
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 534)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'true_divide' (line 540)
        true_divide_357426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), 'true_divide')
        
        # Call to can_cast(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'self' (line 540)
        self_357429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 540)
        dtype_357430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 43), self_357429, 'dtype')
        # Getting the type of 'np' (line 540)
        np_357431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 55), 'np', False)
        # Obtaining the member 'float_' of a type (line 540)
        float__357432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 55), np_357431, 'float_')
        # Processing the call keyword arguments (line 540)
        kwargs_357433 = {}
        # Getting the type of 'np' (line 540)
        np_357427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 31), 'np', False)
        # Obtaining the member 'can_cast' of a type (line 540)
        can_cast_357428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 31), np_357427, 'can_cast')
        # Calling can_cast(args, kwargs) (line 540)
        can_cast_call_result_357434 = invoke(stypy.reporting.localization.Localization(__file__, 540, 31), can_cast_357428, *[dtype_357430, float__357432], **kwargs_357433)
        
        # Applying the binary operator 'and' (line 540)
        result_and_keyword_357435 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 15), 'and', true_divide_357426, can_cast_call_result_357434)
        
        # Testing the type of an if condition (line 540)
        if_condition_357436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 12), result_and_keyword_357435)
        # Assigning a type to the variable 'if_condition_357436' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'if_condition_357436', if_condition_357436)
        # SSA begins for if statement (line 540)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _mul_scalar(...): (line 541)
        # Processing the call arguments (line 541)
        float_357444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 58), 'float')
        # Getting the type of 'other' (line 541)
        other_357445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 61), 'other', False)
        # Applying the binary operator 'div' (line 541)
        result_div_357446 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 58), 'div', float_357444, other_357445)
        
        # Processing the call keyword arguments (line 541)
        kwargs_357447 = {}
        
        # Call to astype(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'np' (line 541)
        np_357439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 35), 'np', False)
        # Obtaining the member 'float_' of a type (line 541)
        float__357440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 35), np_357439, 'float_')
        # Processing the call keyword arguments (line 541)
        kwargs_357441 = {}
        # Getting the type of 'self' (line 541)
        self_357437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 23), 'self', False)
        # Obtaining the member 'astype' of a type (line 541)
        astype_357438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 23), self_357437, 'astype')
        # Calling astype(args, kwargs) (line 541)
        astype_call_result_357442 = invoke(stypy.reporting.localization.Localization(__file__, 541, 23), astype_357438, *[float__357440], **kwargs_357441)
        
        # Obtaining the member '_mul_scalar' of a type (line 541)
        _mul_scalar_357443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 23), astype_call_result_357442, '_mul_scalar')
        # Calling _mul_scalar(args, kwargs) (line 541)
        _mul_scalar_call_result_357448 = invoke(stypy.reporting.localization.Localization(__file__, 541, 23), _mul_scalar_357443, *[result_div_357446], **kwargs_357447)
        
        # Assigning a type to the variable 'stypy_return_type' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'stypy_return_type', _mul_scalar_call_result_357448)
        # SSA branch for the else part of an if statement (line 540)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 543):
        
        # Assigning a Call to a Name (line 543):
        
        # Call to _mul_scalar(...): (line 543)
        # Processing the call arguments (line 543)
        float_357451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 37), 'float')
        # Getting the type of 'other' (line 543)
        other_357452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 40), 'other', False)
        # Applying the binary operator 'div' (line 543)
        result_div_357453 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 37), 'div', float_357451, other_357452)
        
        # Processing the call keyword arguments (line 543)
        kwargs_357454 = {}
        # Getting the type of 'self' (line 543)
        self_357449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 20), 'self', False)
        # Obtaining the member '_mul_scalar' of a type (line 543)
        _mul_scalar_357450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 20), self_357449, '_mul_scalar')
        # Calling _mul_scalar(args, kwargs) (line 543)
        _mul_scalar_call_result_357455 = invoke(stypy.reporting.localization.Localization(__file__, 543, 20), _mul_scalar_357450, *[result_div_357453], **kwargs_357454)
        
        # Assigning a type to the variable 'r' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 16), 'r', _mul_scalar_call_result_357455)
        
        # Assigning a Attribute to a Name (line 545):
        
        # Assigning a Attribute to a Name (line 545):
        
        # Call to asarray(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'other' (line 545)
        other_357458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 42), 'other', False)
        # Processing the call keyword arguments (line 545)
        kwargs_357459 = {}
        # Getting the type of 'np' (line 545)
        np_357456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 31), 'np', False)
        # Obtaining the member 'asarray' of a type (line 545)
        asarray_357457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 31), np_357456, 'asarray')
        # Calling asarray(args, kwargs) (line 545)
        asarray_call_result_357460 = invoke(stypy.reporting.localization.Localization(__file__, 545, 31), asarray_357457, *[other_357458], **kwargs_357459)
        
        # Obtaining the member 'dtype' of a type (line 545)
        dtype_357461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 31), asarray_call_result_357460, 'dtype')
        # Assigning a type to the variable 'scalar_dtype' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'scalar_dtype', dtype_357461)
        
        
        # Evaluating a boolean operation
        
        # Call to issubdtype(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'self' (line 546)
        self_357464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 34), 'self', False)
        # Obtaining the member 'dtype' of a type (line 546)
        dtype_357465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 34), self_357464, 'dtype')
        # Getting the type of 'np' (line 546)
        np_357466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 46), 'np', False)
        # Obtaining the member 'integer' of a type (line 546)
        integer_357467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 46), np_357466, 'integer')
        # Processing the call keyword arguments (line 546)
        kwargs_357468 = {}
        # Getting the type of 'np' (line 546)
        np_357462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 20), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 546)
        issubdtype_357463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 20), np_357462, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 546)
        issubdtype_call_result_357469 = invoke(stypy.reporting.localization.Localization(__file__, 546, 20), issubdtype_357463, *[dtype_357465, integer_357467], **kwargs_357468)
        
        
        # Call to issubdtype(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'scalar_dtype' (line 547)
        scalar_dtype_357472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 38), 'scalar_dtype', False)
        # Getting the type of 'np' (line 547)
        np_357473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 52), 'np', False)
        # Obtaining the member 'integer' of a type (line 547)
        integer_357474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 52), np_357473, 'integer')
        # Processing the call keyword arguments (line 547)
        kwargs_357475 = {}
        # Getting the type of 'np' (line 547)
        np_357470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 24), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 547)
        issubdtype_357471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 24), np_357470, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 547)
        issubdtype_call_result_357476 = invoke(stypy.reporting.localization.Localization(__file__, 547, 24), issubdtype_357471, *[scalar_dtype_357472, integer_357474], **kwargs_357475)
        
        # Applying the binary operator 'and' (line 546)
        result_and_keyword_357477 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 20), 'and', issubdtype_call_result_357469, issubdtype_call_result_357476)
        
        # Testing the type of an if condition (line 546)
        if_condition_357478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 546, 16), result_and_keyword_357477)
        # Assigning a type to the variable 'if_condition_357478' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'if_condition_357478', if_condition_357478)
        # SSA begins for if statement (line 546)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to astype(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'self' (line 548)
        self_357481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 36), 'self', False)
        # Obtaining the member 'dtype' of a type (line 548)
        dtype_357482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 36), self_357481, 'dtype')
        # Processing the call keyword arguments (line 548)
        kwargs_357483 = {}
        # Getting the type of 'r' (line 548)
        r_357479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 27), 'r', False)
        # Obtaining the member 'astype' of a type (line 548)
        astype_357480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 27), r_357479, 'astype')
        # Calling astype(args, kwargs) (line 548)
        astype_call_result_357484 = invoke(stypy.reporting.localization.Localization(__file__, 548, 27), astype_357480, *[dtype_357482], **kwargs_357483)
        
        # Assigning a type to the variable 'stypy_return_type' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'stypy_return_type', astype_call_result_357484)
        # SSA branch for the else part of an if statement (line 546)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'r' (line 550)
        r_357485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 27), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'stypy_return_type', r_357485)
        # SSA join for if statement (line 546)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 540)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 533)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdense(...): (line 552)
        # Processing the call arguments (line 552)
        # Getting the type of 'other' (line 552)
        other_357487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 21), 'other', False)
        # Processing the call keyword arguments (line 552)
        kwargs_357488 = {}
        # Getting the type of 'isdense' (line 552)
        isdense_357486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 13), 'isdense', False)
        # Calling isdense(args, kwargs) (line 552)
        isdense_call_result_357489 = invoke(stypy.reporting.localization.Localization(__file__, 552, 13), isdense_357486, *[other_357487], **kwargs_357488)
        
        # Testing the type of an if condition (line 552)
        if_condition_357490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 552, 13), isdense_call_result_357489)
        # Assigning a type to the variable 'if_condition_357490' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 13), 'if_condition_357490', if_condition_357490)
        # SSA begins for if statement (line 552)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'rdivide' (line 553)
        rdivide_357491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 19), 'rdivide')
        # Applying the 'not' unary operator (line 553)
        result_not__357492 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 15), 'not', rdivide_357491)
        
        # Testing the type of an if condition (line 553)
        if_condition_357493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 12), result_not__357492)
        # Assigning a type to the variable 'if_condition_357493' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'if_condition_357493', if_condition_357493)
        # SSA begins for if statement (line 553)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'true_divide' (line 554)
        true_divide_357494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'true_divide')
        # Testing the type of an if condition (line 554)
        if_condition_357495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 16), true_divide_357494)
        # Assigning a type to the variable 'if_condition_357495' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'if_condition_357495', if_condition_357495)
        # SSA begins for if statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to true_divide(...): (line 555)
        # Processing the call arguments (line 555)
        
        # Call to todense(...): (line 555)
        # Processing the call keyword arguments (line 555)
        kwargs_357500 = {}
        # Getting the type of 'self' (line 555)
        self_357498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 42), 'self', False)
        # Obtaining the member 'todense' of a type (line 555)
        todense_357499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 42), self_357498, 'todense')
        # Calling todense(args, kwargs) (line 555)
        todense_call_result_357501 = invoke(stypy.reporting.localization.Localization(__file__, 555, 42), todense_357499, *[], **kwargs_357500)
        
        # Getting the type of 'other' (line 555)
        other_357502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 58), 'other', False)
        # Processing the call keyword arguments (line 555)
        kwargs_357503 = {}
        # Getting the type of 'np' (line 555)
        np_357496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 27), 'np', False)
        # Obtaining the member 'true_divide' of a type (line 555)
        true_divide_357497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 27), np_357496, 'true_divide')
        # Calling true_divide(args, kwargs) (line 555)
        true_divide_call_result_357504 = invoke(stypy.reporting.localization.Localization(__file__, 555, 27), true_divide_357497, *[todense_call_result_357501, other_357502], **kwargs_357503)
        
        # Assigning a type to the variable 'stypy_return_type' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 20), 'stypy_return_type', true_divide_call_result_357504)
        # SSA branch for the else part of an if statement (line 554)
        module_type_store.open_ssa_branch('else')
        
        # Call to divide(...): (line 557)
        # Processing the call arguments (line 557)
        
        # Call to todense(...): (line 557)
        # Processing the call keyword arguments (line 557)
        kwargs_357509 = {}
        # Getting the type of 'self' (line 557)
        self_357507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 37), 'self', False)
        # Obtaining the member 'todense' of a type (line 557)
        todense_357508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 37), self_357507, 'todense')
        # Calling todense(args, kwargs) (line 557)
        todense_call_result_357510 = invoke(stypy.reporting.localization.Localization(__file__, 557, 37), todense_357508, *[], **kwargs_357509)
        
        # Getting the type of 'other' (line 557)
        other_357511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 53), 'other', False)
        # Processing the call keyword arguments (line 557)
        kwargs_357512 = {}
        # Getting the type of 'np' (line 557)
        np_357505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 27), 'np', False)
        # Obtaining the member 'divide' of a type (line 557)
        divide_357506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 27), np_357505, 'divide')
        # Calling divide(args, kwargs) (line 557)
        divide_call_result_357513 = invoke(stypy.reporting.localization.Localization(__file__, 557, 27), divide_357506, *[todense_call_result_357510, other_357511], **kwargs_357512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 20), 'stypy_return_type', divide_call_result_357513)
        # SSA join for if statement (line 554)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 553)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'true_divide' (line 559)
        true_divide_357514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 19), 'true_divide')
        # Testing the type of an if condition (line 559)
        if_condition_357515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 16), true_divide_357514)
        # Assigning a type to the variable 'if_condition_357515' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 16), 'if_condition_357515', if_condition_357515)
        # SSA begins for if statement (line 559)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to true_divide(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'other' (line 560)
        other_357518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'other', False)
        
        # Call to todense(...): (line 560)
        # Processing the call keyword arguments (line 560)
        kwargs_357521 = {}
        # Getting the type of 'self' (line 560)
        self_357519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 49), 'self', False)
        # Obtaining the member 'todense' of a type (line 560)
        todense_357520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 49), self_357519, 'todense')
        # Calling todense(args, kwargs) (line 560)
        todense_call_result_357522 = invoke(stypy.reporting.localization.Localization(__file__, 560, 49), todense_357520, *[], **kwargs_357521)
        
        # Processing the call keyword arguments (line 560)
        kwargs_357523 = {}
        # Getting the type of 'np' (line 560)
        np_357516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 27), 'np', False)
        # Obtaining the member 'true_divide' of a type (line 560)
        true_divide_357517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 27), np_357516, 'true_divide')
        # Calling true_divide(args, kwargs) (line 560)
        true_divide_call_result_357524 = invoke(stypy.reporting.localization.Localization(__file__, 560, 27), true_divide_357517, *[other_357518, todense_call_result_357522], **kwargs_357523)
        
        # Assigning a type to the variable 'stypy_return_type' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 20), 'stypy_return_type', true_divide_call_result_357524)
        # SSA branch for the else part of an if statement (line 559)
        module_type_store.open_ssa_branch('else')
        
        # Call to divide(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'other' (line 562)
        other_357527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 37), 'other', False)
        
        # Call to todense(...): (line 562)
        # Processing the call keyword arguments (line 562)
        kwargs_357530 = {}
        # Getting the type of 'self' (line 562)
        self_357528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 44), 'self', False)
        # Obtaining the member 'todense' of a type (line 562)
        todense_357529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 44), self_357528, 'todense')
        # Calling todense(args, kwargs) (line 562)
        todense_call_result_357531 = invoke(stypy.reporting.localization.Localization(__file__, 562, 44), todense_357529, *[], **kwargs_357530)
        
        # Processing the call keyword arguments (line 562)
        kwargs_357532 = {}
        # Getting the type of 'np' (line 562)
        np_357525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 27), 'np', False)
        # Obtaining the member 'divide' of a type (line 562)
        divide_357526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 27), np_357525, 'divide')
        # Calling divide(args, kwargs) (line 562)
        divide_call_result_357533 = invoke(stypy.reporting.localization.Localization(__file__, 562, 27), divide_357526, *[other_357527, todense_call_result_357531], **kwargs_357532)
        
        # Assigning a type to the variable 'stypy_return_type' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'stypy_return_type', divide_call_result_357533)
        # SSA join for if statement (line 559)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 553)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 552)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'other' (line 563)
        other_357535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 24), 'other', False)
        # Processing the call keyword arguments (line 563)
        kwargs_357536 = {}
        # Getting the type of 'isspmatrix' (line 563)
        isspmatrix_357534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 13), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 563)
        isspmatrix_call_result_357537 = invoke(stypy.reporting.localization.Localization(__file__, 563, 13), isspmatrix_357534, *[other_357535], **kwargs_357536)
        
        # Testing the type of an if condition (line 563)
        if_condition_357538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 563, 13), isspmatrix_call_result_357537)
        # Assigning a type to the variable 'if_condition_357538' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 13), 'if_condition_357538', if_condition_357538)
        # SSA begins for if statement (line 563)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'rdivide' (line 564)
        rdivide_357539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'rdivide')
        # Testing the type of an if condition (line 564)
        if_condition_357540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 564, 12), rdivide_357539)
        # Assigning a type to the variable 'if_condition_357540' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'if_condition_357540', if_condition_357540)
        # SSA begins for if statement (line 564)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _divide(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 'self' (line 565)
        self_357543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 37), 'self', False)
        # Getting the type of 'true_divide' (line 565)
        true_divide_357544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 43), 'true_divide', False)
        # Processing the call keyword arguments (line 565)
        # Getting the type of 'False' (line 565)
        False_357545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 64), 'False', False)
        keyword_357546 = False_357545
        kwargs_357547 = {'rdivide': keyword_357546}
        # Getting the type of 'other' (line 565)
        other_357541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 23), 'other', False)
        # Obtaining the member '_divide' of a type (line 565)
        _divide_357542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 23), other_357541, '_divide')
        # Calling _divide(args, kwargs) (line 565)
        _divide_call_result_357548 = invoke(stypy.reporting.localization.Localization(__file__, 565, 23), _divide_357542, *[self_357543, true_divide_357544], **kwargs_357547)
        
        # Assigning a type to the variable 'stypy_return_type' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'stypy_return_type', _divide_call_result_357548)
        # SSA join for if statement (line 564)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 567):
        
        # Assigning a Call to a Name (line 567):
        
        # Call to tocsr(...): (line 567)
        # Processing the call keyword arguments (line 567)
        kwargs_357551 = {}
        # Getting the type of 'self' (line 567)
        self_357549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 567)
        tocsr_357550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 23), self_357549, 'tocsr')
        # Calling tocsr(args, kwargs) (line 567)
        tocsr_call_result_357552 = invoke(stypy.reporting.localization.Localization(__file__, 567, 23), tocsr_357550, *[], **kwargs_357551)
        
        # Assigning a type to the variable 'self_csr' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'self_csr', tocsr_call_result_357552)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'true_divide' (line 568)
        true_divide_357553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'true_divide')
        
        # Call to can_cast(...): (line 568)
        # Processing the call arguments (line 568)
        # Getting the type of 'self' (line 568)
        self_357556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 568)
        dtype_357557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 43), self_357556, 'dtype')
        # Getting the type of 'np' (line 568)
        np_357558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 55), 'np', False)
        # Obtaining the member 'float_' of a type (line 568)
        float__357559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 55), np_357558, 'float_')
        # Processing the call keyword arguments (line 568)
        kwargs_357560 = {}
        # Getting the type of 'np' (line 568)
        np_357554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 31), 'np', False)
        # Obtaining the member 'can_cast' of a type (line 568)
        can_cast_357555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 31), np_357554, 'can_cast')
        # Calling can_cast(args, kwargs) (line 568)
        can_cast_call_result_357561 = invoke(stypy.reporting.localization.Localization(__file__, 568, 31), can_cast_357555, *[dtype_357557, float__357559], **kwargs_357560)
        
        # Applying the binary operator 'and' (line 568)
        result_and_keyword_357562 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 15), 'and', true_divide_357553, can_cast_call_result_357561)
        
        # Testing the type of an if condition (line 568)
        if_condition_357563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 12), result_and_keyword_357562)
        # Assigning a type to the variable 'if_condition_357563' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'if_condition_357563', if_condition_357563)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _divide_sparse(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'other' (line 569)
        other_357571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 65), 'other', False)
        # Processing the call keyword arguments (line 569)
        kwargs_357572 = {}
        
        # Call to astype(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'np' (line 569)
        np_357566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 39), 'np', False)
        # Obtaining the member 'float_' of a type (line 569)
        float__357567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 39), np_357566, 'float_')
        # Processing the call keyword arguments (line 569)
        kwargs_357568 = {}
        # Getting the type of 'self_csr' (line 569)
        self_csr_357564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 'self_csr', False)
        # Obtaining the member 'astype' of a type (line 569)
        astype_357565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 23), self_csr_357564, 'astype')
        # Calling astype(args, kwargs) (line 569)
        astype_call_result_357569 = invoke(stypy.reporting.localization.Localization(__file__, 569, 23), astype_357565, *[float__357567], **kwargs_357568)
        
        # Obtaining the member '_divide_sparse' of a type (line 569)
        _divide_sparse_357570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 23), astype_call_result_357569, '_divide_sparse')
        # Calling _divide_sparse(args, kwargs) (line 569)
        _divide_sparse_call_result_357573 = invoke(stypy.reporting.localization.Localization(__file__, 569, 23), _divide_sparse_357570, *[other_357571], **kwargs_357572)
        
        # Assigning a type to the variable 'stypy_return_type' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), 'stypy_return_type', _divide_sparse_call_result_357573)
        # SSA branch for the else part of an if statement (line 568)
        module_type_store.open_ssa_branch('else')
        
        # Call to _divide_sparse(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'other' (line 571)
        other_357576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 47), 'other', False)
        # Processing the call keyword arguments (line 571)
        kwargs_357577 = {}
        # Getting the type of 'self_csr' (line 571)
        self_csr_357574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 23), 'self_csr', False)
        # Obtaining the member '_divide_sparse' of a type (line 571)
        _divide_sparse_357575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 23), self_csr_357574, '_divide_sparse')
        # Calling _divide_sparse(args, kwargs) (line 571)
        _divide_sparse_call_result_357578 = invoke(stypy.reporting.localization.Localization(__file__, 571, 23), _divide_sparse_357575, *[other_357576], **kwargs_357577)
        
        # Assigning a type to the variable 'stypy_return_type' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'stypy_return_type', _divide_sparse_call_result_357578)
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 563)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 573)
        NotImplemented_357579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'stypy_return_type', NotImplemented_357579)
        # SSA join for if statement (line 563)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 552)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 533)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_divide(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_divide' in the type store
        # Getting the type of 'stypy_return_type' (line 532)
        stypy_return_type_357580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357580)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_divide'
        return stypy_return_type_357580


    @norecursion
    def __truediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__truediv__'
        module_type_store = module_type_store.open_function_context('__truediv__', 575, 4, False)
        # Assigning a type to the variable 'self' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__truediv__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__truediv__')
        spmatrix.__truediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__truediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__truediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__truediv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__truediv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__truediv__(...)' code ##################

        
        # Call to _divide(...): (line 576)
        # Processing the call arguments (line 576)
        # Getting the type of 'other' (line 576)
        other_357583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 28), 'other', False)
        # Processing the call keyword arguments (line 576)
        # Getting the type of 'True' (line 576)
        True_357584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 47), 'True', False)
        keyword_357585 = True_357584
        kwargs_357586 = {'true_divide': keyword_357585}
        # Getting the type of 'self' (line 576)
        self_357581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 15), 'self', False)
        # Obtaining the member '_divide' of a type (line 576)
        _divide_357582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 15), self_357581, '_divide')
        # Calling _divide(args, kwargs) (line 576)
        _divide_call_result_357587 = invoke(stypy.reporting.localization.Localization(__file__, 576, 15), _divide_357582, *[other_357583], **kwargs_357586)
        
        # Assigning a type to the variable 'stypy_return_type' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'stypy_return_type', _divide_call_result_357587)
        
        # ################# End of '__truediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__truediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 575)
        stypy_return_type_357588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__truediv__'
        return stypy_return_type_357588


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 578, 4, False)
        # Assigning a type to the variable 'self' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__div__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__div__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__div__')
        spmatrix.__div__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__div__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__div__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__div__(...)' code ##################

        
        # Call to _divide(...): (line 580)
        # Processing the call arguments (line 580)
        # Getting the type of 'other' (line 580)
        other_357591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 28), 'other', False)
        # Processing the call keyword arguments (line 580)
        # Getting the type of 'True' (line 580)
        True_357592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 47), 'True', False)
        keyword_357593 = True_357592
        kwargs_357594 = {'true_divide': keyword_357593}
        # Getting the type of 'self' (line 580)
        self_357589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'self', False)
        # Obtaining the member '_divide' of a type (line 580)
        _divide_357590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 15), self_357589, '_divide')
        # Calling _divide(args, kwargs) (line 580)
        _divide_call_result_357595 = invoke(stypy.reporting.localization.Localization(__file__, 580, 15), _divide_357590, *[other_357591], **kwargs_357594)
        
        # Assigning a type to the variable 'stypy_return_type' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'stypy_return_type', _divide_call_result_357595)
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 578)
        stypy_return_type_357596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357596)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_357596


    @norecursion
    def __rtruediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rtruediv__'
        module_type_store = module_type_store.open_function_context('__rtruediv__', 582, 4, False)
        # Assigning a type to the variable 'self' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__rtruediv__')
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__rtruediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__rtruediv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rtruediv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rtruediv__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 584)
        NotImplemented_357597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'stypy_return_type', NotImplemented_357597)
        
        # ################# End of '__rtruediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rtruediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 582)
        stypy_return_type_357598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357598)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rtruediv__'
        return stypy_return_type_357598


    @norecursion
    def __rdiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rdiv__'
        module_type_store = module_type_store.open_function_context('__rdiv__', 586, 4, False)
        # Assigning a type to the variable 'self' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__rdiv__')
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__rdiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__rdiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rdiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rdiv__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 588)
        NotImplemented_357599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'stypy_return_type', NotImplemented_357599)
        
        # ################# End of '__rdiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rdiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 586)
        stypy_return_type_357600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rdiv__'
        return stypy_return_type_357600


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 590, 4, False)
        # Assigning a type to the variable 'self' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__neg__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__neg__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__neg__')
        spmatrix.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__neg__', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to tocsr(...): (line 591)
        # Processing the call keyword arguments (line 591)
        kwargs_357603 = {}
        # Getting the type of 'self' (line 591)
        self_357601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 591)
        tocsr_357602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 16), self_357601, 'tocsr')
        # Calling tocsr(args, kwargs) (line 591)
        tocsr_call_result_357604 = invoke(stypy.reporting.localization.Localization(__file__, 591, 16), tocsr_357602, *[], **kwargs_357603)
        
        # Applying the 'usub' unary operator (line 591)
        result___neg___357605 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 15), 'usub', tocsr_call_result_357604)
        
        # Assigning a type to the variable 'stypy_return_type' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'stypy_return_type', result___neg___357605)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 590)
        stypy_return_type_357606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_357606


    @norecursion
    def __iadd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iadd__'
        module_type_store = module_type_store.open_function_context('__iadd__', 593, 4, False)
        # Assigning a type to the variable 'self' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__iadd__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__iadd__')
        spmatrix.__iadd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__iadd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__iadd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__iadd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iadd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iadd__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 594)
        NotImplemented_357607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'stypy_return_type', NotImplemented_357607)
        
        # ################# End of '__iadd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iadd__' in the type store
        # Getting the type of 'stypy_return_type' (line 593)
        stypy_return_type_357608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iadd__'
        return stypy_return_type_357608


    @norecursion
    def __isub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__isub__'
        module_type_store = module_type_store.open_function_context('__isub__', 596, 4, False)
        # Assigning a type to the variable 'self' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__isub__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__isub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__isub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__isub__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__isub__')
        spmatrix.__isub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__isub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__isub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__isub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__isub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__isub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__isub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__isub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__isub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__isub__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 597)
        NotImplemented_357609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'stypy_return_type', NotImplemented_357609)
        
        # ################# End of '__isub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__isub__' in the type store
        # Getting the type of 'stypy_return_type' (line 596)
        stypy_return_type_357610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__isub__'
        return stypy_return_type_357610


    @norecursion
    def __imul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imul__'
        module_type_store = module_type_store.open_function_context('__imul__', 599, 4, False)
        # Assigning a type to the variable 'self' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__imul__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__imul__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__imul__')
        spmatrix.__imul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__imul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__imul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__imul__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 600)
        NotImplemented_357611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'stypy_return_type', NotImplemented_357611)
        
        # ################# End of '__imul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imul__' in the type store
        # Getting the type of 'stypy_return_type' (line 599)
        stypy_return_type_357612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imul__'
        return stypy_return_type_357612


    @norecursion
    def __idiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__idiv__'
        module_type_store = module_type_store.open_function_context('__idiv__', 602, 4, False)
        # Assigning a type to the variable 'self' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__idiv__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__idiv__')
        spmatrix.__idiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__idiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__idiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__idiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__idiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__idiv__(...)' code ##################

        
        # Call to __itruediv__(...): (line 603)
        # Processing the call arguments (line 603)
        # Getting the type of 'other' (line 603)
        other_357615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 33), 'other', False)
        # Processing the call keyword arguments (line 603)
        kwargs_357616 = {}
        # Getting the type of 'self' (line 603)
        self_357613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'self', False)
        # Obtaining the member '__itruediv__' of a type (line 603)
        itruediv___357614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 15), self_357613, '__itruediv__')
        # Calling __itruediv__(args, kwargs) (line 603)
        itruediv___call_result_357617 = invoke(stypy.reporting.localization.Localization(__file__, 603, 15), itruediv___357614, *[other_357615], **kwargs_357616)
        
        # Assigning a type to the variable 'stypy_return_type' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'stypy_return_type', itruediv___call_result_357617)
        
        # ################# End of '__idiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__idiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 602)
        stypy_return_type_357618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__idiv__'
        return stypy_return_type_357618


    @norecursion
    def __itruediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__itruediv__'
        module_type_store = module_type_store.open_function_context('__itruediv__', 605, 4, False)
        # Assigning a type to the variable 'self' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__itruediv__')
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__itruediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__itruediv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__itruediv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__itruediv__(...)' code ##################

        # Getting the type of 'NotImplemented' (line 606)
        NotImplemented_357619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'stypy_return_type', NotImplemented_357619)
        
        # ################# End of '__itruediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__itruediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 605)
        stypy_return_type_357620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357620)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__itruediv__'
        return stypy_return_type_357620


    @norecursion
    def __pow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pow__'
        module_type_store = module_type_store.open_function_context('__pow__', 608, 4, False)
        # Assigning a type to the variable 'self' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__pow__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__pow__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__pow__')
        spmatrix.__pow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        spmatrix.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__pow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pow__(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        int_357621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 22), 'int')
        # Getting the type of 'self' (line 609)
        self_357622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 11), 'self')
        # Obtaining the member 'shape' of a type (line 609)
        shape_357623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 11), self_357622, 'shape')
        # Obtaining the member '__getitem__' of a type (line 609)
        getitem___357624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 11), shape_357623, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 609)
        subscript_call_result_357625 = invoke(stypy.reporting.localization.Localization(__file__, 609, 11), getitem___357624, int_357621)
        
        
        # Obtaining the type of the subscript
        int_357626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 39), 'int')
        # Getting the type of 'self' (line 609)
        self_357627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 28), 'self')
        # Obtaining the member 'shape' of a type (line 609)
        shape_357628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 28), self_357627, 'shape')
        # Obtaining the member '__getitem__' of a type (line 609)
        getitem___357629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 28), shape_357628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 609)
        subscript_call_result_357630 = invoke(stypy.reporting.localization.Localization(__file__, 609, 28), getitem___357629, int_357626)
        
        # Applying the binary operator '!=' (line 609)
        result_ne_357631 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 11), '!=', subscript_call_result_357625, subscript_call_result_357630)
        
        # Testing the type of an if condition (line 609)
        if_condition_357632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 8), result_ne_357631)
        # Assigning a type to the variable 'if_condition_357632' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'if_condition_357632', if_condition_357632)
        # SSA begins for if statement (line 609)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 610)
        # Processing the call arguments (line 610)
        str_357634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 28), 'str', 'matrix is not square')
        # Processing the call keyword arguments (line 610)
        kwargs_357635 = {}
        # Getting the type of 'TypeError' (line 610)
        TypeError_357633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 610)
        TypeError_call_result_357636 = invoke(stypy.reporting.localization.Localization(__file__, 610, 18), TypeError_357633, *[str_357634], **kwargs_357635)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 610, 12), TypeError_call_result_357636, 'raise parameter', BaseException)
        # SSA join for if statement (line 609)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isintlike(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'other' (line 612)
        other_357638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 21), 'other', False)
        # Processing the call keyword arguments (line 612)
        kwargs_357639 = {}
        # Getting the type of 'isintlike' (line 612)
        isintlike_357637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 11), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 612)
        isintlike_call_result_357640 = invoke(stypy.reporting.localization.Localization(__file__, 612, 11), isintlike_357637, *[other_357638], **kwargs_357639)
        
        # Testing the type of an if condition (line 612)
        if_condition_357641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 8), isintlike_call_result_357640)
        # Assigning a type to the variable 'if_condition_357641' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'if_condition_357641', if_condition_357641)
        # SSA begins for if statement (line 612)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 613):
        
        # Assigning a Call to a Name (line 613):
        
        # Call to int(...): (line 613)
        # Processing the call arguments (line 613)
        # Getting the type of 'other' (line 613)
        other_357643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 24), 'other', False)
        # Processing the call keyword arguments (line 613)
        kwargs_357644 = {}
        # Getting the type of 'int' (line 613)
        int_357642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'int', False)
        # Calling int(args, kwargs) (line 613)
        int_call_result_357645 = invoke(stypy.reporting.localization.Localization(__file__, 613, 20), int_357642, *[other_357643], **kwargs_357644)
        
        # Assigning a type to the variable 'other' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'other', int_call_result_357645)
        
        
        # Getting the type of 'other' (line 614)
        other_357646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 15), 'other')
        int_357647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 23), 'int')
        # Applying the binary operator '<' (line 614)
        result_lt_357648 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 15), '<', other_357646, int_357647)
        
        # Testing the type of an if condition (line 614)
        if_condition_357649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 614, 12), result_lt_357648)
        # Assigning a type to the variable 'if_condition_357649' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'if_condition_357649', if_condition_357649)
        # SSA begins for if statement (line 614)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 615)
        # Processing the call arguments (line 615)
        str_357651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 33), 'str', 'exponent must be >= 0')
        # Processing the call keyword arguments (line 615)
        kwargs_357652 = {}
        # Getting the type of 'ValueError' (line 615)
        ValueError_357650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 615)
        ValueError_call_result_357653 = invoke(stypy.reporting.localization.Localization(__file__, 615, 22), ValueError_357650, *[str_357651], **kwargs_357652)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 615, 16), ValueError_call_result_357653, 'raise parameter', BaseException)
        # SSA join for if statement (line 614)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'other' (line 617)
        other_357654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 15), 'other')
        int_357655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 24), 'int')
        # Applying the binary operator '==' (line 617)
        result_eq_357656 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 15), '==', other_357654, int_357655)
        
        # Testing the type of an if condition (line 617)
        if_condition_357657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 12), result_eq_357656)
        # Assigning a type to the variable 'if_condition_357657' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'if_condition_357657', if_condition_357657)
        # SSA begins for if statement (line 617)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 618, 16))
        
        # 'from scipy.sparse.construct import eye' statement (line 618)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_357658 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 618, 16), 'scipy.sparse.construct')

        if (type(import_357658) is not StypyTypeError):

            if (import_357658 != 'pyd_module'):
                __import__(import_357658)
                sys_modules_357659 = sys.modules[import_357658]
                import_from_module(stypy.reporting.localization.Localization(__file__, 618, 16), 'scipy.sparse.construct', sys_modules_357659.module_type_store, module_type_store, ['eye'])
                nest_module(stypy.reporting.localization.Localization(__file__, 618, 16), __file__, sys_modules_357659, sys_modules_357659.module_type_store, module_type_store)
            else:
                from scipy.sparse.construct import eye

                import_from_module(stypy.reporting.localization.Localization(__file__, 618, 16), 'scipy.sparse.construct', None, module_type_store, ['eye'], [eye])

        else:
            # Assigning a type to the variable 'scipy.sparse.construct' (line 618)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'scipy.sparse.construct', import_357658)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Call to eye(...): (line 619)
        # Processing the call arguments (line 619)
        
        # Obtaining the type of the subscript
        int_357661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 38), 'int')
        # Getting the type of 'self' (line 619)
        self_357662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 27), 'self', False)
        # Obtaining the member 'shape' of a type (line 619)
        shape_357663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 27), self_357662, 'shape')
        # Obtaining the member '__getitem__' of a type (line 619)
        getitem___357664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 27), shape_357663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 619)
        subscript_call_result_357665 = invoke(stypy.reporting.localization.Localization(__file__, 619, 27), getitem___357664, int_357661)
        
        # Processing the call keyword arguments (line 619)
        # Getting the type of 'self' (line 619)
        self_357666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 48), 'self', False)
        # Obtaining the member 'dtype' of a type (line 619)
        dtype_357667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 48), self_357666, 'dtype')
        keyword_357668 = dtype_357667
        kwargs_357669 = {'dtype': keyword_357668}
        # Getting the type of 'eye' (line 619)
        eye_357660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 23), 'eye', False)
        # Calling eye(args, kwargs) (line 619)
        eye_call_result_357670 = invoke(stypy.reporting.localization.Localization(__file__, 619, 23), eye_357660, *[subscript_call_result_357665], **kwargs_357669)
        
        # Assigning a type to the variable 'stypy_return_type' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'stypy_return_type', eye_call_result_357670)
        # SSA branch for the else part of an if statement (line 617)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'other' (line 620)
        other_357671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 17), 'other')
        int_357672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 26), 'int')
        # Applying the binary operator '==' (line 620)
        result_eq_357673 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 17), '==', other_357671, int_357672)
        
        # Testing the type of an if condition (line 620)
        if_condition_357674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 17), result_eq_357673)
        # Assigning a type to the variable 'if_condition_357674' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 17), 'if_condition_357674', if_condition_357674)
        # SSA begins for if statement (line 620)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 621)
        # Processing the call keyword arguments (line 621)
        kwargs_357677 = {}
        # Getting the type of 'self' (line 621)
        self_357675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 23), 'self', False)
        # Obtaining the member 'copy' of a type (line 621)
        copy_357676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 23), self_357675, 'copy')
        # Calling copy(args, kwargs) (line 621)
        copy_call_result_357678 = invoke(stypy.reporting.localization.Localization(__file__, 621, 23), copy_357676, *[], **kwargs_357677)
        
        # Assigning a type to the variable 'stypy_return_type' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 16), 'stypy_return_type', copy_call_result_357678)
        # SSA branch for the else part of an if statement (line 620)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 623):
        
        # Assigning a Call to a Name (line 623):
        
        # Call to __pow__(...): (line 623)
        # Processing the call arguments (line 623)
        # Getting the type of 'other' (line 623)
        other_357681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 35), 'other', False)
        int_357682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 42), 'int')
        # Applying the binary operator '//' (line 623)
        result_floordiv_357683 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 35), '//', other_357681, int_357682)
        
        # Processing the call keyword arguments (line 623)
        kwargs_357684 = {}
        # Getting the type of 'self' (line 623)
        self_357679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 22), 'self', False)
        # Obtaining the member '__pow__' of a type (line 623)
        pow___357680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 22), self_357679, '__pow__')
        # Calling __pow__(args, kwargs) (line 623)
        pow___call_result_357685 = invoke(stypy.reporting.localization.Localization(__file__, 623, 22), pow___357680, *[result_floordiv_357683], **kwargs_357684)
        
        # Assigning a type to the variable 'tmp' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'tmp', pow___call_result_357685)
        
        # Getting the type of 'other' (line 624)
        other_357686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 20), 'other')
        int_357687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 28), 'int')
        # Applying the binary operator '%' (line 624)
        result_mod_357688 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 20), '%', other_357686, int_357687)
        
        # Testing the type of an if condition (line 624)
        if_condition_357689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 624, 16), result_mod_357688)
        # Assigning a type to the variable 'if_condition_357689' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 16), 'if_condition_357689', if_condition_357689)
        # SSA begins for if statement (line 624)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 625)
        self_357690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 27), 'self')
        # Getting the type of 'tmp' (line 625)
        tmp_357691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 34), 'tmp')
        # Applying the binary operator '*' (line 625)
        result_mul_357692 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 27), '*', self_357690, tmp_357691)
        
        # Getting the type of 'tmp' (line 625)
        tmp_357693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 40), 'tmp')
        # Applying the binary operator '*' (line 625)
        result_mul_357694 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 38), '*', result_mul_357692, tmp_357693)
        
        # Assigning a type to the variable 'stypy_return_type' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 20), 'stypy_return_type', result_mul_357694)
        # SSA branch for the else part of an if statement (line 624)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'tmp' (line 627)
        tmp_357695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 27), 'tmp')
        # Getting the type of 'tmp' (line 627)
        tmp_357696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 33), 'tmp')
        # Applying the binary operator '*' (line 627)
        result_mul_357697 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 27), '*', tmp_357695, tmp_357696)
        
        # Assigning a type to the variable 'stypy_return_type' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 20), 'stypy_return_type', result_mul_357697)
        # SSA join for if statement (line 624)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 620)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 617)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 612)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isscalarlike(...): (line 628)
        # Processing the call arguments (line 628)
        # Getting the type of 'other' (line 628)
        other_357699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 26), 'other', False)
        # Processing the call keyword arguments (line 628)
        kwargs_357700 = {}
        # Getting the type of 'isscalarlike' (line 628)
        isscalarlike_357698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 628)
        isscalarlike_call_result_357701 = invoke(stypy.reporting.localization.Localization(__file__, 628, 13), isscalarlike_357698, *[other_357699], **kwargs_357700)
        
        # Testing the type of an if condition (line 628)
        if_condition_357702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 13), isscalarlike_call_result_357701)
        # Assigning a type to the variable 'if_condition_357702' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'if_condition_357702', if_condition_357702)
        # SSA begins for if statement (line 628)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 629)
        # Processing the call arguments (line 629)
        str_357704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 29), 'str', 'exponent must be an integer')
        # Processing the call keyword arguments (line 629)
        kwargs_357705 = {}
        # Getting the type of 'ValueError' (line 629)
        ValueError_357703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 629)
        ValueError_call_result_357706 = invoke(stypy.reporting.localization.Localization(__file__, 629, 18), ValueError_357703, *[str_357704], **kwargs_357705)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 629, 12), ValueError_call_result_357706, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 628)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 631)
        NotImplemented_357707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'stypy_return_type', NotImplemented_357707)
        # SSA join for if statement (line 628)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 612)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__pow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pow__' in the type store
        # Getting the type of 'stypy_return_type' (line 608)
        stypy_return_type_357708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357708)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pow__'
        return stypy_return_type_357708


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 633, 4, False)
        # Assigning a type to the variable 'self' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__getattr__')
        spmatrix.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        spmatrix.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        
        # Getting the type of 'attr' (line 634)
        attr_357709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 11), 'attr')
        str_357710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 19), 'str', 'A')
        # Applying the binary operator '==' (line 634)
        result_eq_357711 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 11), '==', attr_357709, str_357710)
        
        # Testing the type of an if condition (line 634)
        if_condition_357712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 634, 8), result_eq_357711)
        # Assigning a type to the variable 'if_condition_357712' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'if_condition_357712', if_condition_357712)
        # SSA begins for if statement (line 634)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to toarray(...): (line 635)
        # Processing the call keyword arguments (line 635)
        kwargs_357715 = {}
        # Getting the type of 'self' (line 635)
        self_357713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 19), 'self', False)
        # Obtaining the member 'toarray' of a type (line 635)
        toarray_357714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 19), self_357713, 'toarray')
        # Calling toarray(args, kwargs) (line 635)
        toarray_call_result_357716 = invoke(stypy.reporting.localization.Localization(__file__, 635, 19), toarray_357714, *[], **kwargs_357715)
        
        # Assigning a type to the variable 'stypy_return_type' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'stypy_return_type', toarray_call_result_357716)
        # SSA branch for the else part of an if statement (line 634)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'attr' (line 636)
        attr_357717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 13), 'attr')
        str_357718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 21), 'str', 'T')
        # Applying the binary operator '==' (line 636)
        result_eq_357719 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 13), '==', attr_357717, str_357718)
        
        # Testing the type of an if condition (line 636)
        if_condition_357720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 636, 13), result_eq_357719)
        # Assigning a type to the variable 'if_condition_357720' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 13), 'if_condition_357720', if_condition_357720)
        # SSA begins for if statement (line 636)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to transpose(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_357723 = {}
        # Getting the type of 'self' (line 637)
        self_357721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 19), 'self', False)
        # Obtaining the member 'transpose' of a type (line 637)
        transpose_357722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 19), self_357721, 'transpose')
        # Calling transpose(args, kwargs) (line 637)
        transpose_call_result_357724 = invoke(stypy.reporting.localization.Localization(__file__, 637, 19), transpose_357722, *[], **kwargs_357723)
        
        # Assigning a type to the variable 'stypy_return_type' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'stypy_return_type', transpose_call_result_357724)
        # SSA branch for the else part of an if statement (line 636)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'attr' (line 638)
        attr_357725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 13), 'attr')
        str_357726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 21), 'str', 'H')
        # Applying the binary operator '==' (line 638)
        result_eq_357727 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 13), '==', attr_357725, str_357726)
        
        # Testing the type of an if condition (line 638)
        if_condition_357728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 638, 13), result_eq_357727)
        # Assigning a type to the variable 'if_condition_357728' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 13), 'if_condition_357728', if_condition_357728)
        # SSA begins for if statement (line 638)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to getH(...): (line 639)
        # Processing the call keyword arguments (line 639)
        kwargs_357731 = {}
        # Getting the type of 'self' (line 639)
        self_357729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 19), 'self', False)
        # Obtaining the member 'getH' of a type (line 639)
        getH_357730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 19), self_357729, 'getH')
        # Calling getH(args, kwargs) (line 639)
        getH_call_result_357732 = invoke(stypy.reporting.localization.Localization(__file__, 639, 19), getH_357730, *[], **kwargs_357731)
        
        # Assigning a type to the variable 'stypy_return_type' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'stypy_return_type', getH_call_result_357732)
        # SSA branch for the else part of an if statement (line 638)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'attr' (line 640)
        attr_357733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 13), 'attr')
        str_357734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 21), 'str', 'real')
        # Applying the binary operator '==' (line 640)
        result_eq_357735 = python_operator(stypy.reporting.localization.Localization(__file__, 640, 13), '==', attr_357733, str_357734)
        
        # Testing the type of an if condition (line 640)
        if_condition_357736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 640, 13), result_eq_357735)
        # Assigning a type to the variable 'if_condition_357736' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 13), 'if_condition_357736', if_condition_357736)
        # SSA begins for if statement (line 640)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _real(...): (line 641)
        # Processing the call keyword arguments (line 641)
        kwargs_357739 = {}
        # Getting the type of 'self' (line 641)
        self_357737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 19), 'self', False)
        # Obtaining the member '_real' of a type (line 641)
        _real_357738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 19), self_357737, '_real')
        # Calling _real(args, kwargs) (line 641)
        _real_call_result_357740 = invoke(stypy.reporting.localization.Localization(__file__, 641, 19), _real_357738, *[], **kwargs_357739)
        
        # Assigning a type to the variable 'stypy_return_type' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'stypy_return_type', _real_call_result_357740)
        # SSA branch for the else part of an if statement (line 640)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'attr' (line 642)
        attr_357741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 13), 'attr')
        str_357742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 21), 'str', 'imag')
        # Applying the binary operator '==' (line 642)
        result_eq_357743 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 13), '==', attr_357741, str_357742)
        
        # Testing the type of an if condition (line 642)
        if_condition_357744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 642, 13), result_eq_357743)
        # Assigning a type to the variable 'if_condition_357744' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 13), 'if_condition_357744', if_condition_357744)
        # SSA begins for if statement (line 642)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _imag(...): (line 643)
        # Processing the call keyword arguments (line 643)
        kwargs_357747 = {}
        # Getting the type of 'self' (line 643)
        self_357745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 19), 'self', False)
        # Obtaining the member '_imag' of a type (line 643)
        _imag_357746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 19), self_357745, '_imag')
        # Calling _imag(args, kwargs) (line 643)
        _imag_call_result_357748 = invoke(stypy.reporting.localization.Localization(__file__, 643, 19), _imag_357746, *[], **kwargs_357747)
        
        # Assigning a type to the variable 'stypy_return_type' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'stypy_return_type', _imag_call_result_357748)
        # SSA branch for the else part of an if statement (line 642)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'attr' (line 644)
        attr_357749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 13), 'attr')
        str_357750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 21), 'str', 'size')
        # Applying the binary operator '==' (line 644)
        result_eq_357751 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 13), '==', attr_357749, str_357750)
        
        # Testing the type of an if condition (line 644)
        if_condition_357752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 13), result_eq_357751)
        # Assigning a type to the variable 'if_condition_357752' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 13), 'if_condition_357752', if_condition_357752)
        # SSA begins for if statement (line 644)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to getnnz(...): (line 645)
        # Processing the call keyword arguments (line 645)
        kwargs_357755 = {}
        # Getting the type of 'self' (line 645)
        self_357753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 19), 'self', False)
        # Obtaining the member 'getnnz' of a type (line 645)
        getnnz_357754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 19), self_357753, 'getnnz')
        # Calling getnnz(args, kwargs) (line 645)
        getnnz_call_result_357756 = invoke(stypy.reporting.localization.Localization(__file__, 645, 19), getnnz_357754, *[], **kwargs_357755)
        
        # Assigning a type to the variable 'stypy_return_type' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 12), 'stypy_return_type', getnnz_call_result_357756)
        # SSA branch for the else part of an if statement (line 644)
        module_type_store.open_ssa_branch('else')
        
        # Call to AttributeError(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'attr' (line 647)
        attr_357758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 33), 'attr', False)
        str_357759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 40), 'str', ' not found')
        # Applying the binary operator '+' (line 647)
        result_add_357760 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 33), '+', attr_357758, str_357759)
        
        # Processing the call keyword arguments (line 647)
        kwargs_357761 = {}
        # Getting the type of 'AttributeError' (line 647)
        AttributeError_357757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 647)
        AttributeError_call_result_357762 = invoke(stypy.reporting.localization.Localization(__file__, 647, 18), AttributeError_357757, *[result_add_357760], **kwargs_357761)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 647, 12), AttributeError_call_result_357762, 'raise parameter', BaseException)
        # SSA join for if statement (line 644)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 642)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 640)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 638)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 636)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 634)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 633)
        stypy_return_type_357763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_357763


    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 649)
        None_357764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 29), 'None')
        # Getting the type of 'False' (line 649)
        False_357765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 40), 'False')
        defaults = [None_357764, False_357765]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 649, 4, False)
        # Assigning a type to the variable 'self' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.transpose.__dict__.__setitem__('stypy_function_name', 'spmatrix.transpose')
        spmatrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        spmatrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transpose', localization, ['axes', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transpose(...)' code ##################

        str_357766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, (-1)), 'str', "\n        Reverses the dimensions of the sparse matrix.\n\n        Parameters\n        ----------\n        axes : None, optional\n            This argument is in the signature *solely* for NumPy\n            compatibility reasons. Do not pass in anything except\n            for the default value.\n        copy : bool, optional\n            Indicates whether or not attributes of `self` should be\n            copied whenever possible. The degree to which attributes\n            are copied varies depending on the type of sparse matrix\n            being used.\n\n        Returns\n        -------\n        p : `self` with the dimensions reversed.\n\n        See Also\n        --------\n        np.matrix.transpose : NumPy's implementation of 'transpose'\n                              for matrices\n        ")
        
        # Call to transpose(...): (line 674)
        # Processing the call keyword arguments (line 674)
        # Getting the type of 'axes' (line 674)
        axes_357772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 43), 'axes', False)
        keyword_357773 = axes_357772
        # Getting the type of 'copy' (line 674)
        copy_357774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 54), 'copy', False)
        keyword_357775 = copy_357774
        kwargs_357776 = {'copy': keyword_357775, 'axes': keyword_357773}
        
        # Call to tocsr(...): (line 674)
        # Processing the call keyword arguments (line 674)
        kwargs_357769 = {}
        # Getting the type of 'self' (line 674)
        self_357767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 674)
        tocsr_357768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 15), self_357767, 'tocsr')
        # Calling tocsr(args, kwargs) (line 674)
        tocsr_call_result_357770 = invoke(stypy.reporting.localization.Localization(__file__, 674, 15), tocsr_357768, *[], **kwargs_357769)
        
        # Obtaining the member 'transpose' of a type (line 674)
        transpose_357771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 15), tocsr_call_result_357770, 'transpose')
        # Calling transpose(args, kwargs) (line 674)
        transpose_call_result_357777 = invoke(stypy.reporting.localization.Localization(__file__, 674, 15), transpose_357771, *[], **kwargs_357776)
        
        # Assigning a type to the variable 'stypy_return_type' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'stypy_return_type', transpose_call_result_357777)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 649)
        stypy_return_type_357778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_357778


    @norecursion
    def conj(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'conj'
        module_type_store = module_type_store.open_function_context('conj', 676, 4, False)
        # Assigning a type to the variable 'self' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.conj.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.conj.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.conj.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.conj.__dict__.__setitem__('stypy_function_name', 'spmatrix.conj')
        spmatrix.conj.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.conj.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.conj.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.conj.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.conj.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.conj.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.conj.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.conj', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'conj', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'conj(...)' code ##################

        str_357779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, (-1)), 'str', 'Element-wise complex conjugation.\n\n        If the matrix is of non-complex data type, then this method does\n        nothing and the data is not copied.\n        ')
        
        # Call to conj(...): (line 682)
        # Processing the call keyword arguments (line 682)
        kwargs_357785 = {}
        
        # Call to tocsr(...): (line 682)
        # Processing the call keyword arguments (line 682)
        kwargs_357782 = {}
        # Getting the type of 'self' (line 682)
        self_357780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 682)
        tocsr_357781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 15), self_357780, 'tocsr')
        # Calling tocsr(args, kwargs) (line 682)
        tocsr_call_result_357783 = invoke(stypy.reporting.localization.Localization(__file__, 682, 15), tocsr_357781, *[], **kwargs_357782)
        
        # Obtaining the member 'conj' of a type (line 682)
        conj_357784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 15), tocsr_call_result_357783, 'conj')
        # Calling conj(args, kwargs) (line 682)
        conj_call_result_357786 = invoke(stypy.reporting.localization.Localization(__file__, 682, 15), conj_357784, *[], **kwargs_357785)
        
        # Assigning a type to the variable 'stypy_return_type' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'stypy_return_type', conj_call_result_357786)
        
        # ################# End of 'conj(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'conj' in the type store
        # Getting the type of 'stypy_return_type' (line 676)
        stypy_return_type_357787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'conj'
        return stypy_return_type_357787


    @norecursion
    def conjugate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'conjugate'
        module_type_store = module_type_store.open_function_context('conjugate', 684, 4, False)
        # Assigning a type to the variable 'self' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.conjugate.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.conjugate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.conjugate.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.conjugate.__dict__.__setitem__('stypy_function_name', 'spmatrix.conjugate')
        spmatrix.conjugate.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.conjugate.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.conjugate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.conjugate.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.conjugate.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.conjugate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.conjugate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.conjugate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'conjugate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'conjugate(...)' code ##################

        
        # Call to conj(...): (line 685)
        # Processing the call keyword arguments (line 685)
        kwargs_357790 = {}
        # Getting the type of 'self' (line 685)
        self_357788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 15), 'self', False)
        # Obtaining the member 'conj' of a type (line 685)
        conj_357789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 15), self_357788, 'conj')
        # Calling conj(args, kwargs) (line 685)
        conj_call_result_357791 = invoke(stypy.reporting.localization.Localization(__file__, 685, 15), conj_357789, *[], **kwargs_357790)
        
        # Assigning a type to the variable 'stypy_return_type' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'stypy_return_type', conj_call_result_357791)
        
        # ################# End of 'conjugate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'conjugate' in the type store
        # Getting the type of 'stypy_return_type' (line 684)
        stypy_return_type_357792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'conjugate'
        return stypy_return_type_357792

    
    # Assigning a Attribute to a Attribute (line 687):

    @norecursion
    def getH(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getH'
        module_type_store = module_type_store.open_function_context('getH', 690, 4, False)
        # Assigning a type to the variable 'self' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.getH.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.getH.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.getH.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.getH.__dict__.__setitem__('stypy_function_name', 'spmatrix.getH')
        spmatrix.getH.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.getH.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.getH.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.getH.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.getH.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.getH.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.getH.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.getH', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getH', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getH(...)' code ##################

        str_357793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, (-1)), 'str', "Return the Hermitian transpose of this matrix.\n\n        See Also\n        --------\n        np.matrix.getH : NumPy's implementation of `getH` for matrices\n        ")
        
        # Call to conj(...): (line 697)
        # Processing the call keyword arguments (line 697)
        kwargs_357799 = {}
        
        # Call to transpose(...): (line 697)
        # Processing the call keyword arguments (line 697)
        kwargs_357796 = {}
        # Getting the type of 'self' (line 697)
        self_357794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 15), 'self', False)
        # Obtaining the member 'transpose' of a type (line 697)
        transpose_357795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 15), self_357794, 'transpose')
        # Calling transpose(args, kwargs) (line 697)
        transpose_call_result_357797 = invoke(stypy.reporting.localization.Localization(__file__, 697, 15), transpose_357795, *[], **kwargs_357796)
        
        # Obtaining the member 'conj' of a type (line 697)
        conj_357798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 15), transpose_call_result_357797, 'conj')
        # Calling conj(args, kwargs) (line 697)
        conj_call_result_357800 = invoke(stypy.reporting.localization.Localization(__file__, 697, 15), conj_357798, *[], **kwargs_357799)
        
        # Assigning a type to the variable 'stypy_return_type' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'stypy_return_type', conj_call_result_357800)
        
        # ################# End of 'getH(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getH' in the type store
        # Getting the type of 'stypy_return_type' (line 690)
        stypy_return_type_357801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getH'
        return stypy_return_type_357801


    @norecursion
    def _real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_real'
        module_type_store = module_type_store.open_function_context('_real', 699, 4, False)
        # Assigning a type to the variable 'self' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._real.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._real.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._real.__dict__.__setitem__('stypy_function_name', 'spmatrix._real')
        spmatrix._real.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix._real.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._real.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._real.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_real(...)' code ##################

        
        # Call to _real(...): (line 700)
        # Processing the call keyword arguments (line 700)
        kwargs_357807 = {}
        
        # Call to tocsr(...): (line 700)
        # Processing the call keyword arguments (line 700)
        kwargs_357804 = {}
        # Getting the type of 'self' (line 700)
        self_357802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 700)
        tocsr_357803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 15), self_357802, 'tocsr')
        # Calling tocsr(args, kwargs) (line 700)
        tocsr_call_result_357805 = invoke(stypy.reporting.localization.Localization(__file__, 700, 15), tocsr_357803, *[], **kwargs_357804)
        
        # Obtaining the member '_real' of a type (line 700)
        _real_357806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 15), tocsr_call_result_357805, '_real')
        # Calling _real(args, kwargs) (line 700)
        _real_call_result_357808 = invoke(stypy.reporting.localization.Localization(__file__, 700, 15), _real_357806, *[], **kwargs_357807)
        
        # Assigning a type to the variable 'stypy_return_type' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'stypy_return_type', _real_call_result_357808)
        
        # ################# End of '_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_real' in the type store
        # Getting the type of 'stypy_return_type' (line 699)
        stypy_return_type_357809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357809)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_real'
        return stypy_return_type_357809


    @norecursion
    def _imag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_imag'
        module_type_store = module_type_store.open_function_context('_imag', 702, 4, False)
        # Assigning a type to the variable 'self' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._imag.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._imag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._imag.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._imag.__dict__.__setitem__('stypy_function_name', 'spmatrix._imag')
        spmatrix._imag.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix._imag.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._imag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._imag.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._imag.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._imag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._imag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._imag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_imag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_imag(...)' code ##################

        
        # Call to _imag(...): (line 703)
        # Processing the call keyword arguments (line 703)
        kwargs_357815 = {}
        
        # Call to tocsr(...): (line 703)
        # Processing the call keyword arguments (line 703)
        kwargs_357812 = {}
        # Getting the type of 'self' (line 703)
        self_357810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 703)
        tocsr_357811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 15), self_357810, 'tocsr')
        # Calling tocsr(args, kwargs) (line 703)
        tocsr_call_result_357813 = invoke(stypy.reporting.localization.Localization(__file__, 703, 15), tocsr_357811, *[], **kwargs_357812)
        
        # Obtaining the member '_imag' of a type (line 703)
        _imag_357814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 15), tocsr_call_result_357813, '_imag')
        # Calling _imag(args, kwargs) (line 703)
        _imag_call_result_357816 = invoke(stypy.reporting.localization.Localization(__file__, 703, 15), _imag_357814, *[], **kwargs_357815)
        
        # Assigning a type to the variable 'stypy_return_type' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'stypy_return_type', _imag_call_result_357816)
        
        # ################# End of '_imag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_imag' in the type store
        # Getting the type of 'stypy_return_type' (line 702)
        stypy_return_type_357817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_imag'
        return stypy_return_type_357817


    @norecursion
    def nonzero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'nonzero'
        module_type_store = module_type_store.open_function_context('nonzero', 705, 4, False)
        # Assigning a type to the variable 'self' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.nonzero.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.nonzero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.nonzero.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.nonzero.__dict__.__setitem__('stypy_function_name', 'spmatrix.nonzero')
        spmatrix.nonzero.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.nonzero.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.nonzero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.nonzero.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.nonzero.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.nonzero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.nonzero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.nonzero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'nonzero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'nonzero(...)' code ##################

        str_357818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, (-1)), 'str', 'nonzero indices\n\n        Returns a tuple of arrays (row,col) containing the indices\n        of the non-zero elements of the matrix.\n\n        Examples\n        --------\n        >>> from scipy.sparse import csr_matrix\n        >>> A = csr_matrix([[1,2,0],[0,0,3],[4,0,5]])\n        >>> A.nonzero()\n        (array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))\n\n        ')
        
        # Assigning a Call to a Name (line 721):
        
        # Assigning a Call to a Name (line 721):
        
        # Call to tocoo(...): (line 721)
        # Processing the call keyword arguments (line 721)
        kwargs_357821 = {}
        # Getting the type of 'self' (line 721)
        self_357819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 721)
        tocoo_357820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 12), self_357819, 'tocoo')
        # Calling tocoo(args, kwargs) (line 721)
        tocoo_call_result_357822 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), tocoo_357820, *[], **kwargs_357821)
        
        # Assigning a type to the variable 'A' (line 721)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'A', tocoo_call_result_357822)
        
        # Assigning a Compare to a Name (line 722):
        
        # Assigning a Compare to a Name (line 722):
        
        # Getting the type of 'A' (line 722)
        A_357823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 18), 'A')
        # Obtaining the member 'data' of a type (line 722)
        data_357824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 18), A_357823, 'data')
        int_357825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 28), 'int')
        # Applying the binary operator '!=' (line 722)
        result_ne_357826 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 18), '!=', data_357824, int_357825)
        
        # Assigning a type to the variable 'nz_mask' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'nz_mask', result_ne_357826)
        
        # Obtaining an instance of the builtin type 'tuple' (line 723)
        tuple_357827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 723)
        # Adding element type (line 723)
        
        # Obtaining the type of the subscript
        # Getting the type of 'nz_mask' (line 723)
        nz_mask_357828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 22), 'nz_mask')
        # Getting the type of 'A' (line 723)
        A_357829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 'A')
        # Obtaining the member 'row' of a type (line 723)
        row_357830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 16), A_357829, 'row')
        # Obtaining the member '__getitem__' of a type (line 723)
        getitem___357831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 16), row_357830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 723)
        subscript_call_result_357832 = invoke(stypy.reporting.localization.Localization(__file__, 723, 16), getitem___357831, nz_mask_357828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 723, 16), tuple_357827, subscript_call_result_357832)
        # Adding element type (line 723)
        
        # Obtaining the type of the subscript
        # Getting the type of 'nz_mask' (line 723)
        nz_mask_357833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 38), 'nz_mask')
        # Getting the type of 'A' (line 723)
        A_357834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 32), 'A')
        # Obtaining the member 'col' of a type (line 723)
        col_357835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 32), A_357834, 'col')
        # Obtaining the member '__getitem__' of a type (line 723)
        getitem___357836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 32), col_357835, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 723)
        subscript_call_result_357837 = invoke(stypy.reporting.localization.Localization(__file__, 723, 32), getitem___357836, nz_mask_357833)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 723, 16), tuple_357827, subscript_call_result_357837)
        
        # Assigning a type to the variable 'stypy_return_type' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'stypy_return_type', tuple_357827)
        
        # ################# End of 'nonzero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'nonzero' in the type store
        # Getting the type of 'stypy_return_type' (line 705)
        stypy_return_type_357838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357838)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'nonzero'
        return stypy_return_type_357838


    @norecursion
    def getcol(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getcol'
        module_type_store = module_type_store.open_function_context('getcol', 725, 4, False)
        # Assigning a type to the variable 'self' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.getcol.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.getcol.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.getcol.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.getcol.__dict__.__setitem__('stypy_function_name', 'spmatrix.getcol')
        spmatrix.getcol.__dict__.__setitem__('stypy_param_names_list', ['j'])
        spmatrix.getcol.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.getcol.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.getcol.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.getcol.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.getcol.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.getcol.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.getcol', ['j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getcol', localization, ['j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getcol(...)' code ##################

        str_357839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, (-1)), 'str', 'Returns a copy of column j of the matrix, as an (m x 1) sparse\n        matrix (column vector).\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 732, 8))
        
        # 'from scipy.sparse.csc import csc_matrix' statement (line 732)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_357840 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 732, 8), 'scipy.sparse.csc')

        if (type(import_357840) is not StypyTypeError):

            if (import_357840 != 'pyd_module'):
                __import__(import_357840)
                sys_modules_357841 = sys.modules[import_357840]
                import_from_module(stypy.reporting.localization.Localization(__file__, 732, 8), 'scipy.sparse.csc', sys_modules_357841.module_type_store, module_type_store, ['csc_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 732, 8), __file__, sys_modules_357841, sys_modules_357841.module_type_store, module_type_store)
            else:
                from scipy.sparse.csc import csc_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 732, 8), 'scipy.sparse.csc', None, module_type_store, ['csc_matrix'], [csc_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csc' (line 732)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'scipy.sparse.csc', import_357840)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Assigning a Subscript to a Name (line 733):
        
        # Assigning a Subscript to a Name (line 733):
        
        # Obtaining the type of the subscript
        int_357842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 23), 'int')
        # Getting the type of 'self' (line 733)
        self_357843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'self')
        # Obtaining the member 'shape' of a type (line 733)
        shape_357844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 12), self_357843, 'shape')
        # Obtaining the member '__getitem__' of a type (line 733)
        getitem___357845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 12), shape_357844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 733)
        subscript_call_result_357846 = invoke(stypy.reporting.localization.Localization(__file__, 733, 12), getitem___357845, int_357842)
        
        # Assigning a type to the variable 'n' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'n', subscript_call_result_357846)
        
        
        # Getting the type of 'j' (line 734)
        j_357847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 11), 'j')
        int_357848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 15), 'int')
        # Applying the binary operator '<' (line 734)
        result_lt_357849 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 11), '<', j_357847, int_357848)
        
        # Testing the type of an if condition (line 734)
        if_condition_357850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 8), result_lt_357849)
        # Assigning a type to the variable 'if_condition_357850' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'if_condition_357850', if_condition_357850)
        # SSA begins for if statement (line 734)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'j' (line 735)
        j_357851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'j')
        # Getting the type of 'n' (line 735)
        n_357852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 17), 'n')
        # Applying the binary operator '+=' (line 735)
        result_iadd_357853 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 12), '+=', j_357851, n_357852)
        # Assigning a type to the variable 'j' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'j', result_iadd_357853)
        
        # SSA join for if statement (line 734)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'j' (line 736)
        j_357854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 11), 'j')
        int_357855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 15), 'int')
        # Applying the binary operator '<' (line 736)
        result_lt_357856 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 11), '<', j_357854, int_357855)
        
        
        # Getting the type of 'j' (line 736)
        j_357857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 20), 'j')
        # Getting the type of 'n' (line 736)
        n_357858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 25), 'n')
        # Applying the binary operator '>=' (line 736)
        result_ge_357859 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 20), '>=', j_357857, n_357858)
        
        # Applying the binary operator 'or' (line 736)
        result_or_keyword_357860 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 11), 'or', result_lt_357856, result_ge_357859)
        
        # Testing the type of an if condition (line 736)
        if_condition_357861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 8), result_or_keyword_357860)
        # Assigning a type to the variable 'if_condition_357861' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'if_condition_357861', if_condition_357861)
        # SSA begins for if statement (line 736)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 737)
        # Processing the call arguments (line 737)
        str_357863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 29), 'str', 'index out of bounds')
        # Processing the call keyword arguments (line 737)
        kwargs_357864 = {}
        # Getting the type of 'IndexError' (line 737)
        IndexError_357862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 737)
        IndexError_call_result_357865 = invoke(stypy.reporting.localization.Localization(__file__, 737, 18), IndexError_357862, *[str_357863], **kwargs_357864)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 737, 12), IndexError_call_result_357865, 'raise parameter', BaseException)
        # SSA join for if statement (line 736)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 738):
        
        # Assigning a Call to a Name (line 738):
        
        # Call to csc_matrix(...): (line 738)
        # Processing the call arguments (line 738)
        
        # Obtaining an instance of the builtin type 'tuple' (line 738)
        tuple_357867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 738)
        # Adding element type (line 738)
        
        # Obtaining an instance of the builtin type 'list' (line 738)
        list_357868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 738)
        # Adding element type (line 738)
        int_357869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 35), list_357868, int_357869)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 35), tuple_357867, list_357868)
        # Adding element type (line 738)
        
        # Obtaining an instance of the builtin type 'list' (line 738)
        list_357870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 738)
        # Adding element type (line 738)
        
        # Obtaining an instance of the builtin type 'list' (line 738)
        list_357871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 738)
        # Adding element type (line 738)
        # Getting the type of 'j' (line 738)
        j_357872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 42), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 41), list_357871, j_357872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 40), list_357870, list_357871)
        # Adding element type (line 738)
        
        # Obtaining an instance of the builtin type 'list' (line 738)
        list_357873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 738)
        # Adding element type (line 738)
        int_357874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 46), list_357873, int_357874)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 40), list_357870, list_357873)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 35), tuple_357867, list_357870)
        
        # Processing the call keyword arguments (line 738)
        
        # Obtaining an instance of the builtin type 'tuple' (line 739)
        tuple_357875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 739)
        # Adding element type (line 739)
        # Getting the type of 'n' (line 739)
        n_357876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 41), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 41), tuple_357875, n_357876)
        # Adding element type (line 739)
        int_357877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 41), tuple_357875, int_357877)
        
        keyword_357878 = tuple_357875
        # Getting the type of 'self' (line 739)
        self_357879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 54), 'self', False)
        # Obtaining the member 'dtype' of a type (line 739)
        dtype_357880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 54), self_357879, 'dtype')
        keyword_357881 = dtype_357880
        kwargs_357882 = {'dtype': keyword_357881, 'shape': keyword_357878}
        # Getting the type of 'csc_matrix' (line 738)
        csc_matrix_357866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 23), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 738)
        csc_matrix_call_result_357883 = invoke(stypy.reporting.localization.Localization(__file__, 738, 23), csc_matrix_357866, *[tuple_357867], **kwargs_357882)
        
        # Assigning a type to the variable 'col_selector' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'col_selector', csc_matrix_call_result_357883)
        # Getting the type of 'self' (line 740)
        self_357884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 15), 'self')
        # Getting the type of 'col_selector' (line 740)
        col_selector_357885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 22), 'col_selector')
        # Applying the binary operator '*' (line 740)
        result_mul_357886 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 15), '*', self_357884, col_selector_357885)
        
        # Assigning a type to the variable 'stypy_return_type' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), 'stypy_return_type', result_mul_357886)
        
        # ################# End of 'getcol(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getcol' in the type store
        # Getting the type of 'stypy_return_type' (line 725)
        stypy_return_type_357887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getcol'
        return stypy_return_type_357887


    @norecursion
    def getrow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getrow'
        module_type_store = module_type_store.open_function_context('getrow', 742, 4, False)
        # Assigning a type to the variable 'self' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.getrow.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.getrow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.getrow.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.getrow.__dict__.__setitem__('stypy_function_name', 'spmatrix.getrow')
        spmatrix.getrow.__dict__.__setitem__('stypy_param_names_list', ['i'])
        spmatrix.getrow.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.getrow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.getrow.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.getrow.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.getrow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.getrow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.getrow', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getrow', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getrow(...)' code ##################

        str_357888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, (-1)), 'str', 'Returns a copy of row i of the matrix, as a (1 x n) sparse\n        matrix (row vector).\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 749, 8))
        
        # 'from scipy.sparse.csr import csr_matrix' statement (line 749)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_357889 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 749, 8), 'scipy.sparse.csr')

        if (type(import_357889) is not StypyTypeError):

            if (import_357889 != 'pyd_module'):
                __import__(import_357889)
                sys_modules_357890 = sys.modules[import_357889]
                import_from_module(stypy.reporting.localization.Localization(__file__, 749, 8), 'scipy.sparse.csr', sys_modules_357890.module_type_store, module_type_store, ['csr_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 749, 8), __file__, sys_modules_357890, sys_modules_357890.module_type_store, module_type_store)
            else:
                from scipy.sparse.csr import csr_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 749, 8), 'scipy.sparse.csr', None, module_type_store, ['csr_matrix'], [csr_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csr' (line 749)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'scipy.sparse.csr', import_357889)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Assigning a Subscript to a Name (line 750):
        
        # Assigning a Subscript to a Name (line 750):
        
        # Obtaining the type of the subscript
        int_357891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 23), 'int')
        # Getting the type of 'self' (line 750)
        self_357892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'self')
        # Obtaining the member 'shape' of a type (line 750)
        shape_357893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 12), self_357892, 'shape')
        # Obtaining the member '__getitem__' of a type (line 750)
        getitem___357894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 12), shape_357893, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 750)
        subscript_call_result_357895 = invoke(stypy.reporting.localization.Localization(__file__, 750, 12), getitem___357894, int_357891)
        
        # Assigning a type to the variable 'm' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'm', subscript_call_result_357895)
        
        
        # Getting the type of 'i' (line 751)
        i_357896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 11), 'i')
        int_357897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 15), 'int')
        # Applying the binary operator '<' (line 751)
        result_lt_357898 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 11), '<', i_357896, int_357897)
        
        # Testing the type of an if condition (line 751)
        if_condition_357899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 751, 8), result_lt_357898)
        # Assigning a type to the variable 'if_condition_357899' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'if_condition_357899', if_condition_357899)
        # SSA begins for if statement (line 751)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'i' (line 752)
        i_357900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 12), 'i')
        # Getting the type of 'm' (line 752)
        m_357901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 17), 'm')
        # Applying the binary operator '+=' (line 752)
        result_iadd_357902 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 12), '+=', i_357900, m_357901)
        # Assigning a type to the variable 'i' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 12), 'i', result_iadd_357902)
        
        # SSA join for if statement (line 751)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 753)
        i_357903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 11), 'i')
        int_357904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 15), 'int')
        # Applying the binary operator '<' (line 753)
        result_lt_357905 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 11), '<', i_357903, int_357904)
        
        
        # Getting the type of 'i' (line 753)
        i_357906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 20), 'i')
        # Getting the type of 'm' (line 753)
        m_357907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 25), 'm')
        # Applying the binary operator '>=' (line 753)
        result_ge_357908 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 20), '>=', i_357906, m_357907)
        
        # Applying the binary operator 'or' (line 753)
        result_or_keyword_357909 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 11), 'or', result_lt_357905, result_ge_357908)
        
        # Testing the type of an if condition (line 753)
        if_condition_357910 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 753, 8), result_or_keyword_357909)
        # Assigning a type to the variable 'if_condition_357910' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'if_condition_357910', if_condition_357910)
        # SSA begins for if statement (line 753)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 754)
        # Processing the call arguments (line 754)
        str_357912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 29), 'str', 'index out of bounds')
        # Processing the call keyword arguments (line 754)
        kwargs_357913 = {}
        # Getting the type of 'IndexError' (line 754)
        IndexError_357911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 754)
        IndexError_call_result_357914 = invoke(stypy.reporting.localization.Localization(__file__, 754, 18), IndexError_357911, *[str_357912], **kwargs_357913)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 754, 12), IndexError_call_result_357914, 'raise parameter', BaseException)
        # SSA join for if statement (line 753)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 755):
        
        # Assigning a Call to a Name (line 755):
        
        # Call to csr_matrix(...): (line 755)
        # Processing the call arguments (line 755)
        
        # Obtaining an instance of the builtin type 'tuple' (line 755)
        tuple_357916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 755)
        # Adding element type (line 755)
        
        # Obtaining an instance of the builtin type 'list' (line 755)
        list_357917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 755)
        # Adding element type (line 755)
        int_357918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 35), list_357917, int_357918)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 35), tuple_357916, list_357917)
        # Adding element type (line 755)
        
        # Obtaining an instance of the builtin type 'list' (line 755)
        list_357919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 755)
        # Adding element type (line 755)
        
        # Obtaining an instance of the builtin type 'list' (line 755)
        list_357920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 755)
        # Adding element type (line 755)
        int_357921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 41), list_357920, int_357921)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 40), list_357919, list_357920)
        # Adding element type (line 755)
        
        # Obtaining an instance of the builtin type 'list' (line 755)
        list_357922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 755)
        # Adding element type (line 755)
        # Getting the type of 'i' (line 755)
        i_357923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 47), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 46), list_357922, i_357923)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 40), list_357919, list_357922)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 35), tuple_357916, list_357919)
        
        # Processing the call keyword arguments (line 755)
        
        # Obtaining an instance of the builtin type 'tuple' (line 756)
        tuple_357924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 756)
        # Adding element type (line 756)
        int_357925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 41), tuple_357924, int_357925)
        # Adding element type (line 756)
        # Getting the type of 'm' (line 756)
        m_357926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 44), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 756, 41), tuple_357924, m_357926)
        
        keyword_357927 = tuple_357924
        # Getting the type of 'self' (line 756)
        self_357928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 54), 'self', False)
        # Obtaining the member 'dtype' of a type (line 756)
        dtype_357929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 54), self_357928, 'dtype')
        keyword_357930 = dtype_357929
        kwargs_357931 = {'dtype': keyword_357930, 'shape': keyword_357927}
        # Getting the type of 'csr_matrix' (line 755)
        csr_matrix_357915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 23), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 755)
        csr_matrix_call_result_357932 = invoke(stypy.reporting.localization.Localization(__file__, 755, 23), csr_matrix_357915, *[tuple_357916], **kwargs_357931)
        
        # Assigning a type to the variable 'row_selector' (line 755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'row_selector', csr_matrix_call_result_357932)
        # Getting the type of 'row_selector' (line 757)
        row_selector_357933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 15), 'row_selector')
        # Getting the type of 'self' (line 757)
        self_357934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 30), 'self')
        # Applying the binary operator '*' (line 757)
        result_mul_357935 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 15), '*', row_selector_357933, self_357934)
        
        # Assigning a type to the variable 'stypy_return_type' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'stypy_return_type', result_mul_357935)
        
        # ################# End of 'getrow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getrow' in the type store
        # Getting the type of 'stypy_return_type' (line 742)
        stypy_return_type_357936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357936)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getrow'
        return stypy_return_type_357936


    @norecursion
    def todense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 762)
        None_357937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 28), 'None')
        # Getting the type of 'None' (line 762)
        None_357938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 38), 'None')
        defaults = [None_357937, None_357938]
        # Create a new context for function 'todense'
        module_type_store = module_type_store.open_function_context('todense', 762, 4, False)
        # Assigning a type to the variable 'self' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.todense.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.todense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.todense.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.todense.__dict__.__setitem__('stypy_function_name', 'spmatrix.todense')
        spmatrix.todense.__dict__.__setitem__('stypy_param_names_list', ['order', 'out'])
        spmatrix.todense.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.todense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.todense.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.todense.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.todense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.todense.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.todense', ['order', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'todense', localization, ['order', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'todense(...)' code ##################

        str_357939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, (-1)), 'str', "\n        Return a dense matrix representation of this matrix.\n\n        Parameters\n        ----------\n        order : {'C', 'F'}, optional\n            Whether to store multi-dimensional data in C (row-major)\n            or Fortran (column-major) order in memory. The default\n            is 'None', indicating the NumPy default of C-ordered.\n            Cannot be specified in conjunction with the `out`\n            argument.\n\n        out : ndarray, 2-dimensional, optional\n            If specified, uses this array (or `numpy.matrix`) as the\n            output buffer instead of allocating a new array to\n            return. The provided array must have the same shape and\n            dtype as the sparse matrix on which you are calling the\n            method.\n\n        Returns\n        -------\n        arr : numpy.matrix, 2-dimensional\n            A NumPy matrix object with the same shape and containing\n            the same data represented by the sparse matrix, with the\n            requested memory order. If `out` was passed and was an\n            array (rather than a `numpy.matrix`), it will be filled\n            with the appropriate values and returned wrapped in a\n            `numpy.matrix` object that shares the same memory.\n        ")
        
        # Call to asmatrix(...): (line 792)
        # Processing the call arguments (line 792)
        
        # Call to toarray(...): (line 792)
        # Processing the call keyword arguments (line 792)
        # Getting the type of 'order' (line 792)
        order_357944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 46), 'order', False)
        keyword_357945 = order_357944
        # Getting the type of 'out' (line 792)
        out_357946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 57), 'out', False)
        keyword_357947 = out_357946
        kwargs_357948 = {'order': keyword_357945, 'out': keyword_357947}
        # Getting the type of 'self' (line 792)
        self_357942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 27), 'self', False)
        # Obtaining the member 'toarray' of a type (line 792)
        toarray_357943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 27), self_357942, 'toarray')
        # Calling toarray(args, kwargs) (line 792)
        toarray_call_result_357949 = invoke(stypy.reporting.localization.Localization(__file__, 792, 27), toarray_357943, *[], **kwargs_357948)
        
        # Processing the call keyword arguments (line 792)
        kwargs_357950 = {}
        # Getting the type of 'np' (line 792)
        np_357940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 15), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 792)
        asmatrix_357941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 15), np_357940, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 792)
        asmatrix_call_result_357951 = invoke(stypy.reporting.localization.Localization(__file__, 792, 15), asmatrix_357941, *[toarray_call_result_357949], **kwargs_357950)
        
        # Assigning a type to the variable 'stypy_return_type' (line 792)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'stypy_return_type', asmatrix_call_result_357951)
        
        # ################# End of 'todense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'todense' in the type store
        # Getting the type of 'stypy_return_type' (line 762)
        stypy_return_type_357952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'todense'
        return stypy_return_type_357952


    @norecursion
    def toarray(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 794)
        None_357953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 28), 'None')
        # Getting the type of 'None' (line 794)
        None_357954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 38), 'None')
        defaults = [None_357953, None_357954]
        # Create a new context for function 'toarray'
        module_type_store = module_type_store.open_function_context('toarray', 794, 4, False)
        # Assigning a type to the variable 'self' (line 795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.toarray.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.toarray.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.toarray.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.toarray.__dict__.__setitem__('stypy_function_name', 'spmatrix.toarray')
        spmatrix.toarray.__dict__.__setitem__('stypy_param_names_list', ['order', 'out'])
        spmatrix.toarray.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.toarray.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.toarray.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.toarray.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.toarray.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.toarray.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.toarray', ['order', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'toarray', localization, ['order', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'toarray(...)' code ##################

        str_357955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, (-1)), 'str', "\n        Return a dense ndarray representation of this matrix.\n\n        Parameters\n        ----------\n        order : {'C', 'F'}, optional\n            Whether to store multi-dimensional data in C (row-major)\n            or Fortran (column-major) order in memory. The default\n            is 'None', indicating the NumPy default of C-ordered.\n            Cannot be specified in conjunction with the `out`\n            argument.\n\n        out : ndarray, 2-dimensional, optional\n            If specified, uses this array as the output buffer\n            instead of allocating a new array to return. The provided\n            array must have the same shape and dtype as the sparse\n            matrix on which you are calling the method. For most\n            sparse types, `out` is required to be memory contiguous\n            (either C or Fortran ordered).\n\n        Returns\n        -------\n        arr : ndarray, 2-dimensional\n            An array with the same shape and containing the same\n            data represented by the sparse matrix, with the requested\n            memory order. If `out` was passed, the same object is\n            returned after being modified in-place to contain the\n            appropriate values.\n        ")
        
        # Call to toarray(...): (line 824)
        # Processing the call keyword arguments (line 824)
        # Getting the type of 'order' (line 824)
        order_357963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 52), 'order', False)
        keyword_357964 = order_357963
        # Getting the type of 'out' (line 824)
        out_357965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 63), 'out', False)
        keyword_357966 = out_357965
        kwargs_357967 = {'order': keyword_357964, 'out': keyword_357966}
        
        # Call to tocoo(...): (line 824)
        # Processing the call keyword arguments (line 824)
        # Getting the type of 'False' (line 824)
        False_357958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 31), 'False', False)
        keyword_357959 = False_357958
        kwargs_357960 = {'copy': keyword_357959}
        # Getting the type of 'self' (line 824)
        self_357956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 824)
        tocoo_357957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 15), self_357956, 'tocoo')
        # Calling tocoo(args, kwargs) (line 824)
        tocoo_call_result_357961 = invoke(stypy.reporting.localization.Localization(__file__, 824, 15), tocoo_357957, *[], **kwargs_357960)
        
        # Obtaining the member 'toarray' of a type (line 824)
        toarray_357962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 15), tocoo_call_result_357961, 'toarray')
        # Calling toarray(args, kwargs) (line 824)
        toarray_call_result_357968 = invoke(stypy.reporting.localization.Localization(__file__, 824, 15), toarray_357962, *[], **kwargs_357967)
        
        # Assigning a type to the variable 'stypy_return_type' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 8), 'stypy_return_type', toarray_call_result_357968)
        
        # ################# End of 'toarray(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toarray' in the type store
        # Getting the type of 'stypy_return_type' (line 794)
        stypy_return_type_357969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357969)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toarray'
        return stypy_return_type_357969


    @norecursion
    def tocsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 829)
        False_357970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 25), 'False')
        defaults = [False_357970]
        # Create a new context for function 'tocsr'
        module_type_store = module_type_store.open_function_context('tocsr', 829, 4, False)
        # Assigning a type to the variable 'self' (line 830)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.tocsr.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.tocsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.tocsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.tocsr.__dict__.__setitem__('stypy_function_name', 'spmatrix.tocsr')
        spmatrix.tocsr.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        spmatrix.tocsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.tocsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.tocsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.tocsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.tocsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.tocsr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.tocsr', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocsr', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocsr(...)' code ##################

        str_357971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, (-1)), 'str', 'Convert this matrix to Compressed Sparse Row format.\n\n        With copy=False, the data/indices may be shared between this matrix and\n        the resultant csr_matrix.\n        ')
        
        # Call to tocsr(...): (line 835)
        # Processing the call keyword arguments (line 835)
        # Getting the type of 'False' (line 835)
        False_357979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 48), 'False', False)
        keyword_357980 = False_357979
        kwargs_357981 = {'copy': keyword_357980}
        
        # Call to tocoo(...): (line 835)
        # Processing the call keyword arguments (line 835)
        # Getting the type of 'copy' (line 835)
        copy_357974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 31), 'copy', False)
        keyword_357975 = copy_357974
        kwargs_357976 = {'copy': keyword_357975}
        # Getting the type of 'self' (line 835)
        self_357972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 835)
        tocoo_357973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 15), self_357972, 'tocoo')
        # Calling tocoo(args, kwargs) (line 835)
        tocoo_call_result_357977 = invoke(stypy.reporting.localization.Localization(__file__, 835, 15), tocoo_357973, *[], **kwargs_357976)
        
        # Obtaining the member 'tocsr' of a type (line 835)
        tocsr_357978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 15), tocoo_call_result_357977, 'tocsr')
        # Calling tocsr(args, kwargs) (line 835)
        tocsr_call_result_357982 = invoke(stypy.reporting.localization.Localization(__file__, 835, 15), tocsr_357978, *[], **kwargs_357981)
        
        # Assigning a type to the variable 'stypy_return_type' (line 835)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'stypy_return_type', tocsr_call_result_357982)
        
        # ################# End of 'tocsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsr' in the type store
        # Getting the type of 'stypy_return_type' (line 829)
        stypy_return_type_357983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357983)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsr'
        return stypy_return_type_357983


    @norecursion
    def todok(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 837)
        False_357984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 25), 'False')
        defaults = [False_357984]
        # Create a new context for function 'todok'
        module_type_store = module_type_store.open_function_context('todok', 837, 4, False)
        # Assigning a type to the variable 'self' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.todok.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.todok.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.todok.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.todok.__dict__.__setitem__('stypy_function_name', 'spmatrix.todok')
        spmatrix.todok.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        spmatrix.todok.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.todok.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.todok.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.todok.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.todok.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.todok.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.todok', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'todok', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'todok(...)' code ##################

        str_357985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, (-1)), 'str', 'Convert this matrix to Dictionary Of Keys format.\n\n        With copy=False, the data/indices may be shared between this matrix and\n        the resultant dok_matrix.\n        ')
        
        # Call to todok(...): (line 843)
        # Processing the call keyword arguments (line 843)
        # Getting the type of 'False' (line 843)
        False_357993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 48), 'False', False)
        keyword_357994 = False_357993
        kwargs_357995 = {'copy': keyword_357994}
        
        # Call to tocoo(...): (line 843)
        # Processing the call keyword arguments (line 843)
        # Getting the type of 'copy' (line 843)
        copy_357988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 31), 'copy', False)
        keyword_357989 = copy_357988
        kwargs_357990 = {'copy': keyword_357989}
        # Getting the type of 'self' (line 843)
        self_357986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 843)
        tocoo_357987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 15), self_357986, 'tocoo')
        # Calling tocoo(args, kwargs) (line 843)
        tocoo_call_result_357991 = invoke(stypy.reporting.localization.Localization(__file__, 843, 15), tocoo_357987, *[], **kwargs_357990)
        
        # Obtaining the member 'todok' of a type (line 843)
        todok_357992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 15), tocoo_call_result_357991, 'todok')
        # Calling todok(args, kwargs) (line 843)
        todok_call_result_357996 = invoke(stypy.reporting.localization.Localization(__file__, 843, 15), todok_357992, *[], **kwargs_357995)
        
        # Assigning a type to the variable 'stypy_return_type' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'stypy_return_type', todok_call_result_357996)
        
        # ################# End of 'todok(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'todok' in the type store
        # Getting the type of 'stypy_return_type' (line 837)
        stypy_return_type_357997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_357997)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'todok'
        return stypy_return_type_357997


    @norecursion
    def tocoo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 845)
        False_357998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 25), 'False')
        defaults = [False_357998]
        # Create a new context for function 'tocoo'
        module_type_store = module_type_store.open_function_context('tocoo', 845, 4, False)
        # Assigning a type to the variable 'self' (line 846)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.tocoo.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.tocoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.tocoo.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.tocoo.__dict__.__setitem__('stypy_function_name', 'spmatrix.tocoo')
        spmatrix.tocoo.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        spmatrix.tocoo.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.tocoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.tocoo.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.tocoo.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.tocoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.tocoo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.tocoo', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocoo', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocoo(...)' code ##################

        str_357999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, (-1)), 'str', 'Convert this matrix to COOrdinate format.\n\n        With copy=False, the data/indices may be shared between this matrix and\n        the resultant coo_matrix.\n        ')
        
        # Call to tocoo(...): (line 851)
        # Processing the call keyword arguments (line 851)
        # Getting the type of 'copy' (line 851)
        copy_358007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 49), 'copy', False)
        keyword_358008 = copy_358007
        kwargs_358009 = {'copy': keyword_358008}
        
        # Call to tocsr(...): (line 851)
        # Processing the call keyword arguments (line 851)
        # Getting the type of 'False' (line 851)
        False_358002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 31), 'False', False)
        keyword_358003 = False_358002
        kwargs_358004 = {'copy': keyword_358003}
        # Getting the type of 'self' (line 851)
        self_358000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 851)
        tocsr_358001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 15), self_358000, 'tocsr')
        # Calling tocsr(args, kwargs) (line 851)
        tocsr_call_result_358005 = invoke(stypy.reporting.localization.Localization(__file__, 851, 15), tocsr_358001, *[], **kwargs_358004)
        
        # Obtaining the member 'tocoo' of a type (line 851)
        tocoo_358006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 15), tocsr_call_result_358005, 'tocoo')
        # Calling tocoo(args, kwargs) (line 851)
        tocoo_call_result_358010 = invoke(stypy.reporting.localization.Localization(__file__, 851, 15), tocoo_358006, *[], **kwargs_358009)
        
        # Assigning a type to the variable 'stypy_return_type' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'stypy_return_type', tocoo_call_result_358010)
        
        # ################# End of 'tocoo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocoo' in the type store
        # Getting the type of 'stypy_return_type' (line 845)
        stypy_return_type_358011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358011)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocoo'
        return stypy_return_type_358011


    @norecursion
    def tolil(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 853)
        False_358012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 25), 'False')
        defaults = [False_358012]
        # Create a new context for function 'tolil'
        module_type_store = module_type_store.open_function_context('tolil', 853, 4, False)
        # Assigning a type to the variable 'self' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.tolil.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.tolil.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.tolil.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.tolil.__dict__.__setitem__('stypy_function_name', 'spmatrix.tolil')
        spmatrix.tolil.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        spmatrix.tolil.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.tolil.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.tolil.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.tolil.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.tolil.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.tolil.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.tolil', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tolil', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tolil(...)' code ##################

        str_358013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, (-1)), 'str', 'Convert this matrix to LInked List format.\n\n        With copy=False, the data/indices may be shared between this matrix and\n        the resultant lil_matrix.\n        ')
        
        # Call to tolil(...): (line 859)
        # Processing the call keyword arguments (line 859)
        # Getting the type of 'copy' (line 859)
        copy_358021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 49), 'copy', False)
        keyword_358022 = copy_358021
        kwargs_358023 = {'copy': keyword_358022}
        
        # Call to tocsr(...): (line 859)
        # Processing the call keyword arguments (line 859)
        # Getting the type of 'False' (line 859)
        False_358016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 31), 'False', False)
        keyword_358017 = False_358016
        kwargs_358018 = {'copy': keyword_358017}
        # Getting the type of 'self' (line 859)
        self_358014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 859)
        tocsr_358015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 15), self_358014, 'tocsr')
        # Calling tocsr(args, kwargs) (line 859)
        tocsr_call_result_358019 = invoke(stypy.reporting.localization.Localization(__file__, 859, 15), tocsr_358015, *[], **kwargs_358018)
        
        # Obtaining the member 'tolil' of a type (line 859)
        tolil_358020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 15), tocsr_call_result_358019, 'tolil')
        # Calling tolil(args, kwargs) (line 859)
        tolil_call_result_358024 = invoke(stypy.reporting.localization.Localization(__file__, 859, 15), tolil_358020, *[], **kwargs_358023)
        
        # Assigning a type to the variable 'stypy_return_type' (line 859)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'stypy_return_type', tolil_call_result_358024)
        
        # ################# End of 'tolil(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tolil' in the type store
        # Getting the type of 'stypy_return_type' (line 853)
        stypy_return_type_358025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358025)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tolil'
        return stypy_return_type_358025


    @norecursion
    def todia(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 861)
        False_358026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 25), 'False')
        defaults = [False_358026]
        # Create a new context for function 'todia'
        module_type_store = module_type_store.open_function_context('todia', 861, 4, False)
        # Assigning a type to the variable 'self' (line 862)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.todia.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.todia.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.todia.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.todia.__dict__.__setitem__('stypy_function_name', 'spmatrix.todia')
        spmatrix.todia.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        spmatrix.todia.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.todia.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.todia.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.todia.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.todia.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.todia.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.todia', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'todia', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'todia(...)' code ##################

        str_358027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, (-1)), 'str', 'Convert this matrix to sparse DIAgonal format.\n\n        With copy=False, the data/indices may be shared between this matrix and\n        the resultant dia_matrix.\n        ')
        
        # Call to todia(...): (line 867)
        # Processing the call keyword arguments (line 867)
        # Getting the type of 'False' (line 867)
        False_358035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 48), 'False', False)
        keyword_358036 = False_358035
        kwargs_358037 = {'copy': keyword_358036}
        
        # Call to tocoo(...): (line 867)
        # Processing the call keyword arguments (line 867)
        # Getting the type of 'copy' (line 867)
        copy_358030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 31), 'copy', False)
        keyword_358031 = copy_358030
        kwargs_358032 = {'copy': keyword_358031}
        # Getting the type of 'self' (line 867)
        self_358028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 867)
        tocoo_358029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 15), self_358028, 'tocoo')
        # Calling tocoo(args, kwargs) (line 867)
        tocoo_call_result_358033 = invoke(stypy.reporting.localization.Localization(__file__, 867, 15), tocoo_358029, *[], **kwargs_358032)
        
        # Obtaining the member 'todia' of a type (line 867)
        todia_358034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 15), tocoo_call_result_358033, 'todia')
        # Calling todia(args, kwargs) (line 867)
        todia_call_result_358038 = invoke(stypy.reporting.localization.Localization(__file__, 867, 15), todia_358034, *[], **kwargs_358037)
        
        # Assigning a type to the variable 'stypy_return_type' (line 867)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 8), 'stypy_return_type', todia_call_result_358038)
        
        # ################# End of 'todia(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'todia' in the type store
        # Getting the type of 'stypy_return_type' (line 861)
        stypy_return_type_358039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'todia'
        return stypy_return_type_358039


    @norecursion
    def tobsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 869)
        None_358040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 30), 'None')
        # Getting the type of 'False' (line 869)
        False_358041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 41), 'False')
        defaults = [None_358040, False_358041]
        # Create a new context for function 'tobsr'
        module_type_store = module_type_store.open_function_context('tobsr', 869, 4, False)
        # Assigning a type to the variable 'self' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.tobsr.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.tobsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.tobsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.tobsr.__dict__.__setitem__('stypy_function_name', 'spmatrix.tobsr')
        spmatrix.tobsr.__dict__.__setitem__('stypy_param_names_list', ['blocksize', 'copy'])
        spmatrix.tobsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.tobsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.tobsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.tobsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.tobsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.tobsr.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.tobsr', ['blocksize', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tobsr', localization, ['blocksize', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tobsr(...)' code ##################

        str_358042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, (-1)), 'str', 'Convert this matrix to Block Sparse Row format.\n\n        With copy=False, the data/indices may be shared between this matrix and\n        the resultant bsr_matrix.\n\n        When blocksize=(R, C) is provided, it will be used for construction of\n        the bsr_matrix.\n        ')
        
        # Call to tobsr(...): (line 878)
        # Processing the call keyword arguments (line 878)
        # Getting the type of 'blocksize' (line 878)
        blocksize_358050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 54), 'blocksize', False)
        keyword_358051 = blocksize_358050
        # Getting the type of 'copy' (line 878)
        copy_358052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 70), 'copy', False)
        keyword_358053 = copy_358052
        kwargs_358054 = {'blocksize': keyword_358051, 'copy': keyword_358053}
        
        # Call to tocsr(...): (line 878)
        # Processing the call keyword arguments (line 878)
        # Getting the type of 'False' (line 878)
        False_358045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 31), 'False', False)
        keyword_358046 = False_358045
        kwargs_358047 = {'copy': keyword_358046}
        # Getting the type of 'self' (line 878)
        self_358043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 878)
        tocsr_358044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 15), self_358043, 'tocsr')
        # Calling tocsr(args, kwargs) (line 878)
        tocsr_call_result_358048 = invoke(stypy.reporting.localization.Localization(__file__, 878, 15), tocsr_358044, *[], **kwargs_358047)
        
        # Obtaining the member 'tobsr' of a type (line 878)
        tobsr_358049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 15), tocsr_call_result_358048, 'tobsr')
        # Calling tobsr(args, kwargs) (line 878)
        tobsr_call_result_358055 = invoke(stypy.reporting.localization.Localization(__file__, 878, 15), tobsr_358049, *[], **kwargs_358054)
        
        # Assigning a type to the variable 'stypy_return_type' (line 878)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 8), 'stypy_return_type', tobsr_call_result_358055)
        
        # ################# End of 'tobsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tobsr' in the type store
        # Getting the type of 'stypy_return_type' (line 869)
        stypy_return_type_358056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tobsr'
        return stypy_return_type_358056


    @norecursion
    def tocsc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 880)
        False_358057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 25), 'False')
        defaults = [False_358057]
        # Create a new context for function 'tocsc'
        module_type_store = module_type_store.open_function_context('tocsc', 880, 4, False)
        # Assigning a type to the variable 'self' (line 881)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.tocsc.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.tocsc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.tocsc.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.tocsc.__dict__.__setitem__('stypy_function_name', 'spmatrix.tocsc')
        spmatrix.tocsc.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        spmatrix.tocsc.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.tocsc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.tocsc.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.tocsc.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.tocsc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.tocsc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.tocsc', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocsc', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocsc(...)' code ##################

        str_358058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, (-1)), 'str', 'Convert this matrix to Compressed Sparse Column format.\n\n        With copy=False, the data/indices may be shared between this matrix and\n        the resultant csc_matrix.\n        ')
        
        # Call to tocsc(...): (line 886)
        # Processing the call keyword arguments (line 886)
        # Getting the type of 'False' (line 886)
        False_358066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 48), 'False', False)
        keyword_358067 = False_358066
        kwargs_358068 = {'copy': keyword_358067}
        
        # Call to tocsr(...): (line 886)
        # Processing the call keyword arguments (line 886)
        # Getting the type of 'copy' (line 886)
        copy_358061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 31), 'copy', False)
        keyword_358062 = copy_358061
        kwargs_358063 = {'copy': keyword_358062}
        # Getting the type of 'self' (line 886)
        self_358059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 886)
        tocsr_358060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 15), self_358059, 'tocsr')
        # Calling tocsr(args, kwargs) (line 886)
        tocsr_call_result_358064 = invoke(stypy.reporting.localization.Localization(__file__, 886, 15), tocsr_358060, *[], **kwargs_358063)
        
        # Obtaining the member 'tocsc' of a type (line 886)
        tocsc_358065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 15), tocsr_call_result_358064, 'tocsc')
        # Calling tocsc(args, kwargs) (line 886)
        tocsc_call_result_358069 = invoke(stypy.reporting.localization.Localization(__file__, 886, 15), tocsc_358065, *[], **kwargs_358068)
        
        # Assigning a type to the variable 'stypy_return_type' (line 886)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 8), 'stypy_return_type', tocsc_call_result_358069)
        
        # ################# End of 'tocsc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsc' in the type store
        # Getting the type of 'stypy_return_type' (line 880)
        stypy_return_type_358070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsc'
        return stypy_return_type_358070


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 888, 4, False)
        # Assigning a type to the variable 'self' (line 889)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.copy.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.copy.__dict__.__setitem__('stypy_function_name', 'spmatrix.copy')
        spmatrix.copy.__dict__.__setitem__('stypy_param_names_list', [])
        spmatrix.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        str_358071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, (-1)), 'str', 'Returns a copy of this matrix.\n\n        No data/indices will be shared between the returned value and current\n        matrix.\n        ')
        
        # Call to __class__(...): (line 894)
        # Processing the call arguments (line 894)
        # Getting the type of 'self' (line 894)
        self_358074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 30), 'self', False)
        # Processing the call keyword arguments (line 894)
        # Getting the type of 'True' (line 894)
        True_358075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 41), 'True', False)
        keyword_358076 = True_358075
        kwargs_358077 = {'copy': keyword_358076}
        # Getting the type of 'self' (line 894)
        self_358072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 894)
        class___358073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 15), self_358072, '__class__')
        # Calling __class__(args, kwargs) (line 894)
        class___call_result_358078 = invoke(stypy.reporting.localization.Localization(__file__, 894, 15), class___358073, *[self_358074], **kwargs_358077)
        
        # Assigning a type to the variable 'stypy_return_type' (line 894)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'stypy_return_type', class___call_result_358078)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 888)
        stypy_return_type_358079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358079)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_358079


    @norecursion
    def sum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 896)
        None_358080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 23), 'None')
        # Getting the type of 'None' (line 896)
        None_358081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 35), 'None')
        # Getting the type of 'None' (line 896)
        None_358082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 45), 'None')
        defaults = [None_358080, None_358081, None_358082]
        # Create a new context for function 'sum'
        module_type_store = module_type_store.open_function_context('sum', 896, 4, False)
        # Assigning a type to the variable 'self' (line 897)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.sum.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.sum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.sum.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.sum.__dict__.__setitem__('stypy_function_name', 'spmatrix.sum')
        spmatrix.sum.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out'])
        spmatrix.sum.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.sum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.sum.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.sum.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.sum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.sum.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.sum', ['axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sum', localization, ['axis', 'dtype', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sum(...)' code ##################

        str_358083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, (-1)), 'str', "\n        Sum the matrix elements over a given axis.\n\n        Parameters\n        ----------\n        axis : {-2, -1, 0, 1, None} optional\n            Axis along which the sum is computed. The default is to\n            compute the sum of all the matrix elements, returning a scalar\n            (i.e. `axis` = `None`).\n        dtype : dtype, optional\n            The type of the returned matrix and of the accumulator in which\n            the elements are summed.  The dtype of `a` is used by default\n            unless `a` has an integer dtype of less precision than the default\n            platform integer.  In that case, if `a` is signed then the platform\n            integer is used while if `a` is unsigned then an unsigned integer\n            of the same precision as the platform integer is used.\n\n            .. versionadded: 0.18.0\n\n        out : np.matrix, optional\n            Alternative output matrix in which to place the result. It must\n            have the same shape as the expected output, but the type of the\n            output values will be cast if necessary.\n\n            .. versionadded: 0.18.0\n\n        Returns\n        -------\n        sum_along_axis : np.matrix\n            A matrix with the same shape as `self`, with the specified\n            axis removed.\n\n        See Also\n        --------\n        np.matrix.sum : NumPy's implementation of 'sum' for matrices\n\n        ")
        
        # Call to validateaxis(...): (line 934)
        # Processing the call arguments (line 934)
        # Getting the type of 'axis' (line 934)
        axis_358085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 21), 'axis', False)
        # Processing the call keyword arguments (line 934)
        kwargs_358086 = {}
        # Getting the type of 'validateaxis' (line 934)
        validateaxis_358084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 8), 'validateaxis', False)
        # Calling validateaxis(args, kwargs) (line 934)
        validateaxis_call_result_358087 = invoke(stypy.reporting.localization.Localization(__file__, 934, 8), validateaxis_358084, *[axis_358085], **kwargs_358086)
        
        
        # Assigning a Attribute to a Tuple (line 939):
        
        # Assigning a Subscript to a Name (line 939):
        
        # Obtaining the type of the subscript
        int_358088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 8), 'int')
        # Getting the type of 'self' (line 939)
        self_358089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 15), 'self')
        # Obtaining the member 'shape' of a type (line 939)
        shape_358090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 15), self_358089, 'shape')
        # Obtaining the member '__getitem__' of a type (line 939)
        getitem___358091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 8), shape_358090, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 939)
        subscript_call_result_358092 = invoke(stypy.reporting.localization.Localization(__file__, 939, 8), getitem___358091, int_358088)
        
        # Assigning a type to the variable 'tuple_var_assignment_356178' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'tuple_var_assignment_356178', subscript_call_result_358092)
        
        # Assigning a Subscript to a Name (line 939):
        
        # Obtaining the type of the subscript
        int_358093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 8), 'int')
        # Getting the type of 'self' (line 939)
        self_358094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 15), 'self')
        # Obtaining the member 'shape' of a type (line 939)
        shape_358095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 15), self_358094, 'shape')
        # Obtaining the member '__getitem__' of a type (line 939)
        getitem___358096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 8), shape_358095, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 939)
        subscript_call_result_358097 = invoke(stypy.reporting.localization.Localization(__file__, 939, 8), getitem___358096, int_358093)
        
        # Assigning a type to the variable 'tuple_var_assignment_356179' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'tuple_var_assignment_356179', subscript_call_result_358097)
        
        # Assigning a Name to a Name (line 939):
        # Getting the type of 'tuple_var_assignment_356178' (line 939)
        tuple_var_assignment_356178_358098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'tuple_var_assignment_356178')
        # Assigning a type to the variable 'm' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'm', tuple_var_assignment_356178_358098)
        
        # Assigning a Name to a Name (line 939):
        # Getting the type of 'tuple_var_assignment_356179' (line 939)
        tuple_var_assignment_356179_358099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'tuple_var_assignment_356179')
        # Assigning a type to the variable 'n' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 11), 'n', tuple_var_assignment_356179_358099)
        
        # Assigning a Call to a Name (line 942):
        
        # Assigning a Call to a Name (line 942):
        
        # Call to get_sum_dtype(...): (line 942)
        # Processing the call arguments (line 942)
        # Getting the type of 'self' (line 942)
        self_358101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 34), 'self', False)
        # Obtaining the member 'dtype' of a type (line 942)
        dtype_358102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 34), self_358101, 'dtype')
        # Processing the call keyword arguments (line 942)
        kwargs_358103 = {}
        # Getting the type of 'get_sum_dtype' (line 942)
        get_sum_dtype_358100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 20), 'get_sum_dtype', False)
        # Calling get_sum_dtype(args, kwargs) (line 942)
        get_sum_dtype_call_result_358104 = invoke(stypy.reporting.localization.Localization(__file__, 942, 20), get_sum_dtype_358100, *[dtype_358102], **kwargs_358103)
        
        # Assigning a type to the variable 'res_dtype' (line 942)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 8), 'res_dtype', get_sum_dtype_call_result_358104)
        
        # Type idiom detected: calculating its left and rigth part (line 944)
        # Getting the type of 'axis' (line 944)
        axis_358105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 11), 'axis')
        # Getting the type of 'None' (line 944)
        None_358106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 19), 'None')
        
        (may_be_358107, more_types_in_union_358108) = may_be_none(axis_358105, None_358106)

        if may_be_358107:

            if more_types_in_union_358108:
                # Runtime conditional SSA (line 944)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to sum(...): (line 946)
            # Processing the call keyword arguments (line 946)
            # Getting the type of 'dtype' (line 948)
            dtype_358125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 22), 'dtype', False)
            keyword_358126 = dtype_358125
            # Getting the type of 'out' (line 948)
            out_358127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 33), 'out', False)
            keyword_358128 = out_358127
            kwargs_358129 = {'dtype': keyword_358126, 'out': keyword_358128}
            # Getting the type of 'self' (line 946)
            self_358109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 20), 'self', False)
            
            # Call to asmatrix(...): (line 946)
            # Processing the call arguments (line 946)
            
            # Call to ones(...): (line 946)
            # Processing the call arguments (line 946)
            
            # Obtaining an instance of the builtin type 'tuple' (line 947)
            tuple_358114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 17), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 947)
            # Adding element type (line 947)
            # Getting the type of 'n' (line 947)
            n_358115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 17), 'n', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 17), tuple_358114, n_358115)
            # Adding element type (line 947)
            int_358116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 20), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 17), tuple_358114, int_358116)
            
            # Processing the call keyword arguments (line 946)
            # Getting the type of 'res_dtype' (line 947)
            res_dtype_358117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 30), 'res_dtype', False)
            keyword_358118 = res_dtype_358117
            kwargs_358119 = {'dtype': keyword_358118}
            # Getting the type of 'np' (line 946)
            np_358112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 39), 'np', False)
            # Obtaining the member 'ones' of a type (line 946)
            ones_358113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 39), np_358112, 'ones')
            # Calling ones(args, kwargs) (line 946)
            ones_call_result_358120 = invoke(stypy.reporting.localization.Localization(__file__, 946, 39), ones_358113, *[tuple_358114], **kwargs_358119)
            
            # Processing the call keyword arguments (line 946)
            kwargs_358121 = {}
            # Getting the type of 'np' (line 946)
            np_358110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 27), 'np', False)
            # Obtaining the member 'asmatrix' of a type (line 946)
            asmatrix_358111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 27), np_358110, 'asmatrix')
            # Calling asmatrix(args, kwargs) (line 946)
            asmatrix_call_result_358122 = invoke(stypy.reporting.localization.Localization(__file__, 946, 27), asmatrix_358111, *[ones_call_result_358120], **kwargs_358121)
            
            # Applying the binary operator '*' (line 946)
            result_mul_358123 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 20), '*', self_358109, asmatrix_call_result_358122)
            
            # Obtaining the member 'sum' of a type (line 946)
            sum_358124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 20), result_mul_358123, 'sum')
            # Calling sum(args, kwargs) (line 946)
            sum_call_result_358130 = invoke(stypy.reporting.localization.Localization(__file__, 946, 20), sum_358124, *[], **kwargs_358129)
            
            # Assigning a type to the variable 'stypy_return_type' (line 946)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 12), 'stypy_return_type', sum_call_result_358130)

            if more_types_in_union_358108:
                # SSA join for if statement (line 944)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'axis' (line 950)
        axis_358131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 11), 'axis')
        int_358132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 18), 'int')
        # Applying the binary operator '<' (line 950)
        result_lt_358133 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 11), '<', axis_358131, int_358132)
        
        # Testing the type of an if condition (line 950)
        if_condition_358134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 950, 8), result_lt_358133)
        # Assigning a type to the variable 'if_condition_358134' (line 950)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 8), 'if_condition_358134', if_condition_358134)
        # SSA begins for if statement (line 950)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'axis' (line 951)
        axis_358135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 12), 'axis')
        int_358136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 20), 'int')
        # Applying the binary operator '+=' (line 951)
        result_iadd_358137 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 12), '+=', axis_358135, int_358136)
        # Assigning a type to the variable 'axis' (line 951)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 12), 'axis', result_iadd_358137)
        
        # SSA join for if statement (line 950)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'axis' (line 954)
        axis_358138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 11), 'axis')
        int_358139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 19), 'int')
        # Applying the binary operator '==' (line 954)
        result_eq_358140 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 11), '==', axis_358138, int_358139)
        
        # Testing the type of an if condition (line 954)
        if_condition_358141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 954, 8), result_eq_358140)
        # Assigning a type to the variable 'if_condition_358141' (line 954)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 8), 'if_condition_358141', if_condition_358141)
        # SSA begins for if statement (line 954)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 956):
        
        # Assigning a BinOp to a Name (line 956):
        
        # Call to asmatrix(...): (line 956)
        # Processing the call arguments (line 956)
        
        # Call to ones(...): (line 956)
        # Processing the call arguments (line 956)
        
        # Obtaining an instance of the builtin type 'tuple' (line 957)
        tuple_358146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 957)
        # Adding element type (line 957)
        int_358147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 957, 17), tuple_358146, int_358147)
        # Adding element type (line 957)
        # Getting the type of 'm' (line 957)
        m_358148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 20), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 957, 17), tuple_358146, m_358148)
        
        # Processing the call keyword arguments (line 956)
        # Getting the type of 'res_dtype' (line 957)
        res_dtype_358149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 30), 'res_dtype', False)
        keyword_358150 = res_dtype_358149
        kwargs_358151 = {'dtype': keyword_358150}
        # Getting the type of 'np' (line 956)
        np_358144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 30), 'np', False)
        # Obtaining the member 'ones' of a type (line 956)
        ones_358145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 30), np_358144, 'ones')
        # Calling ones(args, kwargs) (line 956)
        ones_call_result_358152 = invoke(stypy.reporting.localization.Localization(__file__, 956, 30), ones_358145, *[tuple_358146], **kwargs_358151)
        
        # Processing the call keyword arguments (line 956)
        kwargs_358153 = {}
        # Getting the type of 'np' (line 956)
        np_358142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 18), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 956)
        asmatrix_358143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 18), np_358142, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 956)
        asmatrix_call_result_358154 = invoke(stypy.reporting.localization.Localization(__file__, 956, 18), asmatrix_358143, *[ones_call_result_358152], **kwargs_358153)
        
        # Getting the type of 'self' (line 957)
        self_358155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 44), 'self')
        # Applying the binary operator '*' (line 956)
        result_mul_358156 = python_operator(stypy.reporting.localization.Localization(__file__, 956, 18), '*', asmatrix_call_result_358154, self_358155)
        
        # Assigning a type to the variable 'ret' (line 956)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 12), 'ret', result_mul_358156)
        # SSA branch for the else part of an if statement (line 954)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 960):
        
        # Assigning a BinOp to a Name (line 960):
        # Getting the type of 'self' (line 960)
        self_358157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 18), 'self')
        
        # Call to asmatrix(...): (line 960)
        # Processing the call arguments (line 960)
        
        # Call to ones(...): (line 961)
        # Processing the call arguments (line 961)
        
        # Obtaining an instance of the builtin type 'tuple' (line 961)
        tuple_358162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 961)
        # Adding element type (line 961)
        # Getting the type of 'n' (line 961)
        n_358163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 25), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 25), tuple_358162, n_358163)
        # Adding element type (line 961)
        int_358164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 25), tuple_358162, int_358164)
        
        # Processing the call keyword arguments (line 961)
        # Getting the type of 'res_dtype' (line 961)
        res_dtype_358165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 38), 'res_dtype', False)
        keyword_358166 = res_dtype_358165
        kwargs_358167 = {'dtype': keyword_358166}
        # Getting the type of 'np' (line 961)
        np_358160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 16), 'np', False)
        # Obtaining the member 'ones' of a type (line 961)
        ones_358161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 16), np_358160, 'ones')
        # Calling ones(args, kwargs) (line 961)
        ones_call_result_358168 = invoke(stypy.reporting.localization.Localization(__file__, 961, 16), ones_358161, *[tuple_358162], **kwargs_358167)
        
        # Processing the call keyword arguments (line 960)
        kwargs_358169 = {}
        # Getting the type of 'np' (line 960)
        np_358158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 25), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 960)
        asmatrix_358159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 25), np_358158, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 960)
        asmatrix_call_result_358170 = invoke(stypy.reporting.localization.Localization(__file__, 960, 25), asmatrix_358159, *[ones_call_result_358168], **kwargs_358169)
        
        # Applying the binary operator '*' (line 960)
        result_mul_358171 = python_operator(stypy.reporting.localization.Localization(__file__, 960, 18), '*', self_358157, asmatrix_call_result_358170)
        
        # Assigning a type to the variable 'ret' (line 960)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 12), 'ret', result_mul_358171)
        # SSA join for if statement (line 954)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'out' (line 963)
        out_358172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 11), 'out')
        # Getting the type of 'None' (line 963)
        None_358173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 22), 'None')
        # Applying the binary operator 'isnot' (line 963)
        result_is_not_358174 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 11), 'isnot', out_358172, None_358173)
        
        
        # Getting the type of 'out' (line 963)
        out_358175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 31), 'out')
        # Obtaining the member 'shape' of a type (line 963)
        shape_358176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 31), out_358175, 'shape')
        # Getting the type of 'ret' (line 963)
        ret_358177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 44), 'ret')
        # Obtaining the member 'shape' of a type (line 963)
        shape_358178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 44), ret_358177, 'shape')
        # Applying the binary operator '!=' (line 963)
        result_ne_358179 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 31), '!=', shape_358176, shape_358178)
        
        # Applying the binary operator 'and' (line 963)
        result_and_keyword_358180 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 11), 'and', result_is_not_358174, result_ne_358179)
        
        # Testing the type of an if condition (line 963)
        if_condition_358181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 963, 8), result_and_keyword_358180)
        # Assigning a type to the variable 'if_condition_358181' (line 963)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 8), 'if_condition_358181', if_condition_358181)
        # SSA begins for if statement (line 963)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 964)
        # Processing the call arguments (line 964)
        str_358183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 29), 'str', 'dimensions do not match')
        # Processing the call keyword arguments (line 964)
        kwargs_358184 = {}
        # Getting the type of 'ValueError' (line 964)
        ValueError_358182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 964)
        ValueError_call_result_358185 = invoke(stypy.reporting.localization.Localization(__file__, 964, 18), ValueError_358182, *[str_358183], **kwargs_358184)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 964, 12), ValueError_call_result_358185, 'raise parameter', BaseException)
        # SSA join for if statement (line 963)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to sum(...): (line 966)
        # Processing the call keyword arguments (line 966)
        
        # Obtaining an instance of the builtin type 'tuple' (line 966)
        tuple_358188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 966)
        
        keyword_358189 = tuple_358188
        # Getting the type of 'dtype' (line 966)
        dtype_358190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 38), 'dtype', False)
        keyword_358191 = dtype_358190
        # Getting the type of 'out' (line 966)
        out_358192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 49), 'out', False)
        keyword_358193 = out_358192
        kwargs_358194 = {'dtype': keyword_358191, 'out': keyword_358193, 'axis': keyword_358189}
        # Getting the type of 'ret' (line 966)
        ret_358186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 15), 'ret', False)
        # Obtaining the member 'sum' of a type (line 966)
        sum_358187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 15), ret_358186, 'sum')
        # Calling sum(args, kwargs) (line 966)
        sum_call_result_358195 = invoke(stypy.reporting.localization.Localization(__file__, 966, 15), sum_358187, *[], **kwargs_358194)
        
        # Assigning a type to the variable 'stypy_return_type' (line 966)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'stypy_return_type', sum_call_result_358195)
        
        # ################# End of 'sum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sum' in the type store
        # Getting the type of 'stypy_return_type' (line 896)
        stypy_return_type_358196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sum'
        return stypy_return_type_358196


    @norecursion
    def mean(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 968)
        None_358197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 24), 'None')
        # Getting the type of 'None' (line 968)
        None_358198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 36), 'None')
        # Getting the type of 'None' (line 968)
        None_358199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 46), 'None')
        defaults = [None_358197, None_358198, None_358199]
        # Create a new context for function 'mean'
        module_type_store = module_type_store.open_function_context('mean', 968, 4, False)
        # Assigning a type to the variable 'self' (line 969)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.mean.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.mean.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.mean.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.mean.__dict__.__setitem__('stypy_function_name', 'spmatrix.mean')
        spmatrix.mean.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out'])
        spmatrix.mean.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.mean.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.mean.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.mean.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.mean.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.mean.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.mean', ['axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mean', localization, ['axis', 'dtype', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mean(...)' code ##################

        str_358200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1004, (-1)), 'str', "\n        Compute the arithmetic mean along the specified axis.\n\n        Returns the average of the matrix elements. The average is taken\n        over all elements in the matrix by default, otherwise over the\n        specified axis. `float64` intermediate and return values are used\n        for integer inputs.\n\n        Parameters\n        ----------\n        axis : {-2, -1, 0, 1, None} optional\n            Axis along which the mean is computed. The default is to compute\n            the mean of all elements in the matrix (i.e. `axis` = `None`).\n        dtype : data-type, optional\n            Type to use in computing the mean. For integer inputs, the default\n            is `float64`; for floating point inputs, it is the same as the\n            input dtype.\n\n            .. versionadded: 0.18.0\n\n        out : np.matrix, optional\n            Alternative output matrix in which to place the result. It must\n            have the same shape as the expected output, but the type of the\n            output values will be cast if necessary.\n\n            .. versionadded: 0.18.0\n\n        Returns\n        -------\n        m : np.matrix\n\n        See Also\n        --------\n        np.matrix.mean : NumPy's implementation of 'mean' for matrices\n\n        ")

        @norecursion
        def _is_integral(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_is_integral'
            module_type_store = module_type_store.open_function_context('_is_integral', 1005, 8, False)
            
            # Passed parameters checking function
            _is_integral.stypy_localization = localization
            _is_integral.stypy_type_of_self = None
            _is_integral.stypy_type_store = module_type_store
            _is_integral.stypy_function_name = '_is_integral'
            _is_integral.stypy_param_names_list = ['dtype']
            _is_integral.stypy_varargs_param_name = None
            _is_integral.stypy_kwargs_param_name = None
            _is_integral.stypy_call_defaults = defaults
            _is_integral.stypy_call_varargs = varargs
            _is_integral.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_is_integral', ['dtype'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_is_integral', localization, ['dtype'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_is_integral(...)' code ##################

            
            # Evaluating a boolean operation
            
            # Call to issubdtype(...): (line 1006)
            # Processing the call arguments (line 1006)
            # Getting the type of 'dtype' (line 1006)
            dtype_358203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 34), 'dtype', False)
            # Getting the type of 'np' (line 1006)
            np_358204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 41), 'np', False)
            # Obtaining the member 'integer' of a type (line 1006)
            integer_358205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 41), np_358204, 'integer')
            # Processing the call keyword arguments (line 1006)
            kwargs_358206 = {}
            # Getting the type of 'np' (line 1006)
            np_358201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 20), 'np', False)
            # Obtaining the member 'issubdtype' of a type (line 1006)
            issubdtype_358202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 20), np_358201, 'issubdtype')
            # Calling issubdtype(args, kwargs) (line 1006)
            issubdtype_call_result_358207 = invoke(stypy.reporting.localization.Localization(__file__, 1006, 20), issubdtype_358202, *[dtype_358203, integer_358205], **kwargs_358206)
            
            
            # Call to issubdtype(...): (line 1007)
            # Processing the call arguments (line 1007)
            # Getting the type of 'dtype' (line 1007)
            dtype_358210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 34), 'dtype', False)
            # Getting the type of 'np' (line 1007)
            np_358211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 41), 'np', False)
            # Obtaining the member 'bool_' of a type (line 1007)
            bool__358212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 41), np_358211, 'bool_')
            # Processing the call keyword arguments (line 1007)
            kwargs_358213 = {}
            # Getting the type of 'np' (line 1007)
            np_358208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 20), 'np', False)
            # Obtaining the member 'issubdtype' of a type (line 1007)
            issubdtype_358209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 20), np_358208, 'issubdtype')
            # Calling issubdtype(args, kwargs) (line 1007)
            issubdtype_call_result_358214 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 20), issubdtype_358209, *[dtype_358210, bool__358212], **kwargs_358213)
            
            # Applying the binary operator 'or' (line 1006)
            result_or_keyword_358215 = python_operator(stypy.reporting.localization.Localization(__file__, 1006, 20), 'or', issubdtype_call_result_358207, issubdtype_call_result_358214)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1006)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 12), 'stypy_return_type', result_or_keyword_358215)
            
            # ################# End of '_is_integral(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_is_integral' in the type store
            # Getting the type of 'stypy_return_type' (line 1005)
            stypy_return_type_358216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_358216)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_is_integral'
            return stypy_return_type_358216

        # Assigning a type to the variable '_is_integral' (line 1005)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 8), '_is_integral', _is_integral)
        
        # Call to validateaxis(...): (line 1009)
        # Processing the call arguments (line 1009)
        # Getting the type of 'axis' (line 1009)
        axis_358218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 21), 'axis', False)
        # Processing the call keyword arguments (line 1009)
        kwargs_358219 = {}
        # Getting the type of 'validateaxis' (line 1009)
        validateaxis_358217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'validateaxis', False)
        # Calling validateaxis(args, kwargs) (line 1009)
        validateaxis_call_result_358220 = invoke(stypy.reporting.localization.Localization(__file__, 1009, 8), validateaxis_358217, *[axis_358218], **kwargs_358219)
        
        
        # Assigning a Attribute to a Name (line 1011):
        
        # Assigning a Attribute to a Name (line 1011):
        # Getting the type of 'self' (line 1011)
        self_358221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 20), 'self')
        # Obtaining the member 'dtype' of a type (line 1011)
        dtype_358222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 20), self_358221, 'dtype')
        # Obtaining the member 'type' of a type (line 1011)
        type_358223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 20), dtype_358222, 'type')
        # Assigning a type to the variable 'res_dtype' (line 1011)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 8), 'res_dtype', type_358223)
        
        # Assigning a Call to a Name (line 1012):
        
        # Assigning a Call to a Name (line 1012):
        
        # Call to _is_integral(...): (line 1012)
        # Processing the call arguments (line 1012)
        # Getting the type of 'self' (line 1012)
        self_358225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 1012)
        dtype_358226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 32), self_358225, 'dtype')
        # Processing the call keyword arguments (line 1012)
        kwargs_358227 = {}
        # Getting the type of '_is_integral' (line 1012)
        _is_integral_358224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 19), '_is_integral', False)
        # Calling _is_integral(args, kwargs) (line 1012)
        _is_integral_call_result_358228 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 19), _is_integral_358224, *[dtype_358226], **kwargs_358227)
        
        # Assigning a type to the variable 'integral' (line 1012)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 8), 'integral', _is_integral_call_result_358228)
        
        # Type idiom detected: calculating its left and rigth part (line 1015)
        # Getting the type of 'dtype' (line 1015)
        dtype_358229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 11), 'dtype')
        # Getting the type of 'None' (line 1015)
        None_358230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 20), 'None')
        
        (may_be_358231, more_types_in_union_358232) = may_be_none(dtype_358229, None_358230)

        if may_be_358231:

            if more_types_in_union_358232:
                # Runtime conditional SSA (line 1015)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'integral' (line 1016)
            integral_358233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 15), 'integral')
            # Testing the type of an if condition (line 1016)
            if_condition_358234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1016, 12), integral_358233)
            # Assigning a type to the variable 'if_condition_358234' (line 1016)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1016, 12), 'if_condition_358234', if_condition_358234)
            # SSA begins for if statement (line 1016)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 1017):
            
            # Assigning a Attribute to a Name (line 1017):
            # Getting the type of 'np' (line 1017)
            np_358235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 28), 'np')
            # Obtaining the member 'float64' of a type (line 1017)
            float64_358236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 28), np_358235, 'float64')
            # Assigning a type to the variable 'res_dtype' (line 1017)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 16), 'res_dtype', float64_358236)
            # SSA join for if statement (line 1016)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_358232:
                # Runtime conditional SSA for else branch (line 1015)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_358231) or more_types_in_union_358232):
            
            # Assigning a Attribute to a Name (line 1019):
            
            # Assigning a Attribute to a Name (line 1019):
            
            # Call to dtype(...): (line 1019)
            # Processing the call arguments (line 1019)
            # Getting the type of 'dtype' (line 1019)
            dtype_358239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 33), 'dtype', False)
            # Processing the call keyword arguments (line 1019)
            kwargs_358240 = {}
            # Getting the type of 'np' (line 1019)
            np_358237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 24), 'np', False)
            # Obtaining the member 'dtype' of a type (line 1019)
            dtype_358238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 24), np_358237, 'dtype')
            # Calling dtype(args, kwargs) (line 1019)
            dtype_call_result_358241 = invoke(stypy.reporting.localization.Localization(__file__, 1019, 24), dtype_358238, *[dtype_358239], **kwargs_358240)
            
            # Obtaining the member 'type' of a type (line 1019)
            type_358242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 24), dtype_call_result_358241, 'type')
            # Assigning a type to the variable 'res_dtype' (line 1019)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1019, 12), 'res_dtype', type_358242)

            if (may_be_358231 and more_types_in_union_358232):
                # SSA join for if statement (line 1015)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a IfExp to a Name (line 1022):
        
        # Assigning a IfExp to a Name (line 1022):
        
        # Getting the type of 'integral' (line 1022)
        integral_358243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 36), 'integral')
        # Testing the type of an if expression (line 1022)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1022, 22), integral_358243)
        # SSA begins for if expression (line 1022)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'np' (line 1022)
        np_358244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 22), 'np')
        # Obtaining the member 'float64' of a type (line 1022)
        float64_358245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1022, 22), np_358244, 'float64')
        # SSA branch for the else part of an if expression (line 1022)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'res_dtype' (line 1022)
        res_dtype_358246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 50), 'res_dtype')
        # SSA join for if expression (line 1022)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_358247 = union_type.UnionType.add(float64_358245, res_dtype_358246)
        
        # Assigning a type to the variable 'inter_dtype' (line 1022)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1022, 8), 'inter_dtype', if_exp_358247)
        
        # Assigning a Call to a Name (line 1023):
        
        # Assigning a Call to a Name (line 1023):
        
        # Call to astype(...): (line 1023)
        # Processing the call arguments (line 1023)
        # Getting the type of 'inter_dtype' (line 1023)
        inter_dtype_358250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 33), 'inter_dtype', False)
        # Processing the call keyword arguments (line 1023)
        kwargs_358251 = {}
        # Getting the type of 'self' (line 1023)
        self_358248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 21), 'self', False)
        # Obtaining the member 'astype' of a type (line 1023)
        astype_358249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1023, 21), self_358248, 'astype')
        # Calling astype(args, kwargs) (line 1023)
        astype_call_result_358252 = invoke(stypy.reporting.localization.Localization(__file__, 1023, 21), astype_358249, *[inter_dtype_358250], **kwargs_358251)
        
        # Assigning a type to the variable 'inter_self' (line 1023)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1023, 8), 'inter_self', astype_call_result_358252)
        
        # Type idiom detected: calculating its left and rigth part (line 1025)
        # Getting the type of 'axis' (line 1025)
        axis_358253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 11), 'axis')
        # Getting the type of 'None' (line 1025)
        None_358254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 19), 'None')
        
        (may_be_358255, more_types_in_union_358256) = may_be_none(axis_358253, None_358254)

        if may_be_358255:

            if more_types_in_union_358256:
                # Runtime conditional SSA (line 1025)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to sum(...): (line 1026)
            # Processing the call keyword arguments (line 1026)
            # Getting the type of 'res_dtype' (line 1028)
            res_dtype_358275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 27), 'res_dtype', False)
            keyword_358276 = res_dtype_358275
            # Getting the type of 'out' (line 1028)
            out_358277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 42), 'out', False)
            keyword_358278 = out_358277
            kwargs_358279 = {'dtype': keyword_358276, 'out': keyword_358278}
            # Getting the type of 'inter_self' (line 1026)
            inter_self_358257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 20), 'inter_self', False)
            
            # Call to array(...): (line 1026)
            # Processing the call arguments (line 1026)
            
            # Obtaining the type of the subscript
            int_358260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 27), 'int')
            # Getting the type of 'self' (line 1027)
            self_358261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 16), 'self', False)
            # Obtaining the member 'shape' of a type (line 1027)
            shape_358262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1027, 16), self_358261, 'shape')
            # Obtaining the member '__getitem__' of a type (line 1027)
            getitem___358263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1027, 16), shape_358262, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1027)
            subscript_call_result_358264 = invoke(stypy.reporting.localization.Localization(__file__, 1027, 16), getitem___358263, int_358260)
            
            
            # Obtaining the type of the subscript
            int_358265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 43), 'int')
            # Getting the type of 'self' (line 1027)
            self_358266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 32), 'self', False)
            # Obtaining the member 'shape' of a type (line 1027)
            shape_358267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1027, 32), self_358266, 'shape')
            # Obtaining the member '__getitem__' of a type (line 1027)
            getitem___358268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1027, 32), shape_358267, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1027)
            subscript_call_result_358269 = invoke(stypy.reporting.localization.Localization(__file__, 1027, 32), getitem___358268, int_358265)
            
            # Applying the binary operator '*' (line 1027)
            result_mul_358270 = python_operator(stypy.reporting.localization.Localization(__file__, 1027, 16), '*', subscript_call_result_358264, subscript_call_result_358269)
            
            # Processing the call keyword arguments (line 1026)
            kwargs_358271 = {}
            # Getting the type of 'np' (line 1026)
            np_358258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 33), 'np', False)
            # Obtaining the member 'array' of a type (line 1026)
            array_358259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 33), np_358258, 'array')
            # Calling array(args, kwargs) (line 1026)
            array_call_result_358272 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 33), array_358259, *[result_mul_358270], **kwargs_358271)
            
            # Applying the binary operator 'div' (line 1026)
            result_div_358273 = python_operator(stypy.reporting.localization.Localization(__file__, 1026, 20), 'div', inter_self_358257, array_call_result_358272)
            
            # Obtaining the member 'sum' of a type (line 1026)
            sum_358274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 20), result_div_358273, 'sum')
            # Calling sum(args, kwargs) (line 1026)
            sum_call_result_358280 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 20), sum_358274, *[], **kwargs_358279)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1026)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 12), 'stypy_return_type', sum_call_result_358280)

            if more_types_in_union_358256:
                # SSA join for if statement (line 1025)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'axis' (line 1030)
        axis_358281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 11), 'axis')
        int_358282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 18), 'int')
        # Applying the binary operator '<' (line 1030)
        result_lt_358283 = python_operator(stypy.reporting.localization.Localization(__file__, 1030, 11), '<', axis_358281, int_358282)
        
        # Testing the type of an if condition (line 1030)
        if_condition_358284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1030, 8), result_lt_358283)
        # Assigning a type to the variable 'if_condition_358284' (line 1030)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'if_condition_358284', if_condition_358284)
        # SSA begins for if statement (line 1030)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'axis' (line 1031)
        axis_358285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 12), 'axis')
        int_358286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 20), 'int')
        # Applying the binary operator '+=' (line 1031)
        result_iadd_358287 = python_operator(stypy.reporting.localization.Localization(__file__, 1031, 12), '+=', axis_358285, int_358286)
        # Assigning a type to the variable 'axis' (line 1031)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 12), 'axis', result_iadd_358287)
        
        # SSA join for if statement (line 1030)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'axis' (line 1034)
        axis_358288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 11), 'axis')
        int_358289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 19), 'int')
        # Applying the binary operator '==' (line 1034)
        result_eq_358290 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 11), '==', axis_358288, int_358289)
        
        # Testing the type of an if condition (line 1034)
        if_condition_358291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1034, 8), result_eq_358290)
        # Assigning a type to the variable 'if_condition_358291' (line 1034)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'if_condition_358291', if_condition_358291)
        # SSA begins for if statement (line 1034)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to sum(...): (line 1035)
        # Processing the call keyword arguments (line 1035)
        int_358302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 21), 'int')
        keyword_358303 = int_358302
        # Getting the type of 'res_dtype' (line 1036)
        res_dtype_358304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 30), 'res_dtype', False)
        keyword_358305 = res_dtype_358304
        # Getting the type of 'out' (line 1036)
        out_358306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 45), 'out', False)
        keyword_358307 = out_358306
        kwargs_358308 = {'dtype': keyword_358305, 'out': keyword_358307, 'axis': keyword_358303}
        # Getting the type of 'inter_self' (line 1035)
        inter_self_358292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 20), 'inter_self', False)
        float_358293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 34), 'float')
        
        # Obtaining the type of the subscript
        int_358294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 51), 'int')
        # Getting the type of 'self' (line 1035)
        self_358295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 40), 'self', False)
        # Obtaining the member 'shape' of a type (line 1035)
        shape_358296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 40), self_358295, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1035)
        getitem___358297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 40), shape_358296, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1035)
        subscript_call_result_358298 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 40), getitem___358297, int_358294)
        
        # Applying the binary operator 'div' (line 1035)
        result_div_358299 = python_operator(stypy.reporting.localization.Localization(__file__, 1035, 34), 'div', float_358293, subscript_call_result_358298)
        
        # Applying the binary operator '*' (line 1035)
        result_mul_358300 = python_operator(stypy.reporting.localization.Localization(__file__, 1035, 20), '*', inter_self_358292, result_div_358299)
        
        # Obtaining the member 'sum' of a type (line 1035)
        sum_358301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 20), result_mul_358300, 'sum')
        # Calling sum(args, kwargs) (line 1035)
        sum_call_result_358309 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 20), sum_358301, *[], **kwargs_358308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1035)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 12), 'stypy_return_type', sum_call_result_358309)
        # SSA branch for the else part of an if statement (line 1034)
        module_type_store.open_ssa_branch('else')
        
        # Call to sum(...): (line 1038)
        # Processing the call keyword arguments (line 1038)
        int_358320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 21), 'int')
        keyword_358321 = int_358320
        # Getting the type of 'res_dtype' (line 1039)
        res_dtype_358322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 30), 'res_dtype', False)
        keyword_358323 = res_dtype_358322
        # Getting the type of 'out' (line 1039)
        out_358324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 45), 'out', False)
        keyword_358325 = out_358324
        kwargs_358326 = {'dtype': keyword_358323, 'out': keyword_358325, 'axis': keyword_358321}
        # Getting the type of 'inter_self' (line 1038)
        inter_self_358310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 20), 'inter_self', False)
        float_358311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 34), 'float')
        
        # Obtaining the type of the subscript
        int_358312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 51), 'int')
        # Getting the type of 'self' (line 1038)
        self_358313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 40), 'self', False)
        # Obtaining the member 'shape' of a type (line 1038)
        shape_358314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 40), self_358313, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1038)
        getitem___358315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 40), shape_358314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1038)
        subscript_call_result_358316 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 40), getitem___358315, int_358312)
        
        # Applying the binary operator 'div' (line 1038)
        result_div_358317 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 34), 'div', float_358311, subscript_call_result_358316)
        
        # Applying the binary operator '*' (line 1038)
        result_mul_358318 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 20), '*', inter_self_358310, result_div_358317)
        
        # Obtaining the member 'sum' of a type (line 1038)
        sum_358319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 20), result_mul_358318, 'sum')
        # Calling sum(args, kwargs) (line 1038)
        sum_call_result_358327 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 20), sum_358319, *[], **kwargs_358326)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1038)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 12), 'stypy_return_type', sum_call_result_358327)
        # SSA join for if statement (line 1034)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'mean(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mean' in the type store
        # Getting the type of 'stypy_return_type' (line 968)
        stypy_return_type_358328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mean'
        return stypy_return_type_358328


    @norecursion
    def diagonal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_358329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 25), 'int')
        defaults = [int_358329]
        # Create a new context for function 'diagonal'
        module_type_store = module_type_store.open_function_context('diagonal', 1041, 4, False)
        # Assigning a type to the variable 'self' (line 1042)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.diagonal.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.diagonal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.diagonal.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.diagonal.__dict__.__setitem__('stypy_function_name', 'spmatrix.diagonal')
        spmatrix.diagonal.__dict__.__setitem__('stypy_param_names_list', ['k'])
        spmatrix.diagonal.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.diagonal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.diagonal.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.diagonal.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.diagonal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.diagonal.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.diagonal', ['k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'diagonal', localization, ['k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'diagonal(...)' code ##################

        str_358330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, (-1)), 'str', 'Returns the k-th diagonal of the matrix.\n\n        Parameters\n        ----------\n        k : int, optional\n            Which diagonal to set, corresponding to elements a[i, i+k].\n            Default: 0 (the main diagonal).\n\n            .. versionadded: 1.0\n\n        See also\n        --------\n        numpy.diagonal : Equivalent numpy function.\n\n        Examples\n        --------\n        >>> from scipy.sparse import csr_matrix\n        >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])\n        >>> A.diagonal()\n        array([1, 0, 5])\n        >>> A.diagonal(k=1)\n        array([2, 3])\n        ')
        
        # Call to diagonal(...): (line 1065)
        # Processing the call keyword arguments (line 1065)
        # Getting the type of 'k' (line 1065)
        k_358336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 39), 'k', False)
        keyword_358337 = k_358336
        kwargs_358338 = {'k': keyword_358337}
        
        # Call to tocsr(...): (line 1065)
        # Processing the call keyword arguments (line 1065)
        kwargs_358333 = {}
        # Getting the type of 'self' (line 1065)
        self_358331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 1065)
        tocsr_358332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 15), self_358331, 'tocsr')
        # Calling tocsr(args, kwargs) (line 1065)
        tocsr_call_result_358334 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 15), tocsr_358332, *[], **kwargs_358333)
        
        # Obtaining the member 'diagonal' of a type (line 1065)
        diagonal_358335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 15), tocsr_call_result_358334, 'diagonal')
        # Calling diagonal(args, kwargs) (line 1065)
        diagonal_call_result_358339 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 15), diagonal_358335, *[], **kwargs_358338)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 8), 'stypy_return_type', diagonal_call_result_358339)
        
        # ################# End of 'diagonal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'diagonal' in the type store
        # Getting the type of 'stypy_return_type' (line 1041)
        stypy_return_type_358340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358340)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'diagonal'
        return stypy_return_type_358340


    @norecursion
    def setdiag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_358341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 32), 'int')
        defaults = [int_358341]
        # Create a new context for function 'setdiag'
        module_type_store = module_type_store.open_function_context('setdiag', 1067, 4, False)
        # Assigning a type to the variable 'self' (line 1068)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1068, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.setdiag.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.setdiag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.setdiag.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.setdiag.__dict__.__setitem__('stypy_function_name', 'spmatrix.setdiag')
        spmatrix.setdiag.__dict__.__setitem__('stypy_param_names_list', ['values', 'k'])
        spmatrix.setdiag.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.setdiag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix.setdiag.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.setdiag.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.setdiag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.setdiag.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.setdiag', ['values', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setdiag', localization, ['values', 'k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setdiag(...)' code ##################

        str_358342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, (-1)), 'str', '\n        Set diagonal or off-diagonal elements of the array.\n\n        Parameters\n        ----------\n        values : array_like\n            New values of the diagonal elements.\n\n            Values may have any length.  If the diagonal is longer than values,\n            then the remaining diagonal entries will not be set.  If values if\n            longer than the diagonal, then the remaining values are ignored.\n\n            If a scalar value is given, all of the diagonal is set to it.\n\n        k : int, optional\n            Which off-diagonal to set, corresponding to elements a[i,i+k].\n            Default: 0 (the main diagonal).\n\n        ')
        
        # Assigning a Attribute to a Tuple (line 1087):
        
        # Assigning a Subscript to a Name (line 1087):
        
        # Obtaining the type of the subscript
        int_358343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 8), 'int')
        # Getting the type of 'self' (line 1087)
        self_358344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 15), 'self')
        # Obtaining the member 'shape' of a type (line 1087)
        shape_358345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 15), self_358344, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1087)
        getitem___358346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 8), shape_358345, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1087)
        subscript_call_result_358347 = invoke(stypy.reporting.localization.Localization(__file__, 1087, 8), getitem___358346, int_358343)
        
        # Assigning a type to the variable 'tuple_var_assignment_356180' (line 1087)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1087, 8), 'tuple_var_assignment_356180', subscript_call_result_358347)
        
        # Assigning a Subscript to a Name (line 1087):
        
        # Obtaining the type of the subscript
        int_358348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 8), 'int')
        # Getting the type of 'self' (line 1087)
        self_358349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 15), 'self')
        # Obtaining the member 'shape' of a type (line 1087)
        shape_358350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 15), self_358349, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1087)
        getitem___358351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 8), shape_358350, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1087)
        subscript_call_result_358352 = invoke(stypy.reporting.localization.Localization(__file__, 1087, 8), getitem___358351, int_358348)
        
        # Assigning a type to the variable 'tuple_var_assignment_356181' (line 1087)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1087, 8), 'tuple_var_assignment_356181', subscript_call_result_358352)
        
        # Assigning a Name to a Name (line 1087):
        # Getting the type of 'tuple_var_assignment_356180' (line 1087)
        tuple_var_assignment_356180_358353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 8), 'tuple_var_assignment_356180')
        # Assigning a type to the variable 'M' (line 1087)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1087, 8), 'M', tuple_var_assignment_356180_358353)
        
        # Assigning a Name to a Name (line 1087):
        # Getting the type of 'tuple_var_assignment_356181' (line 1087)
        tuple_var_assignment_356181_358354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 8), 'tuple_var_assignment_356181')
        # Assigning a type to the variable 'N' (line 1087)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1087, 11), 'N', tuple_var_assignment_356181_358354)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'k' (line 1088)
        k_358355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 12), 'k')
        int_358356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 16), 'int')
        # Applying the binary operator '>' (line 1088)
        result_gt_358357 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 12), '>', k_358355, int_358356)
        
        
        # Getting the type of 'k' (line 1088)
        k_358358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 22), 'k')
        # Getting the type of 'N' (line 1088)
        N_358359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 27), 'N')
        # Applying the binary operator '>=' (line 1088)
        result_ge_358360 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 22), '>=', k_358358, N_358359)
        
        # Applying the binary operator 'and' (line 1088)
        result_and_keyword_358361 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 12), 'and', result_gt_358357, result_ge_358360)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'k' (line 1088)
        k_358362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 34), 'k')
        int_358363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 38), 'int')
        # Applying the binary operator '<' (line 1088)
        result_lt_358364 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 34), '<', k_358362, int_358363)
        
        
        
        # Getting the type of 'k' (line 1088)
        k_358365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 45), 'k')
        # Applying the 'usub' unary operator (line 1088)
        result___neg___358366 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 44), 'usub', k_358365)
        
        # Getting the type of 'M' (line 1088)
        M_358367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 50), 'M')
        # Applying the binary operator '>=' (line 1088)
        result_ge_358368 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 44), '>=', result___neg___358366, M_358367)
        
        # Applying the binary operator 'and' (line 1088)
        result_and_keyword_358369 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 34), 'and', result_lt_358364, result_ge_358368)
        
        # Applying the binary operator 'or' (line 1088)
        result_or_keyword_358370 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 11), 'or', result_and_keyword_358361, result_and_keyword_358369)
        
        # Testing the type of an if condition (line 1088)
        if_condition_358371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1088, 8), result_or_keyword_358370)
        # Assigning a type to the variable 'if_condition_358371' (line 1088)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'if_condition_358371', if_condition_358371)
        # SSA begins for if statement (line 1088)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1089)
        # Processing the call arguments (line 1089)
        str_358373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 29), 'str', 'k exceeds matrix dimensions')
        # Processing the call keyword arguments (line 1089)
        kwargs_358374 = {}
        # Getting the type of 'ValueError' (line 1089)
        ValueError_358372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1089)
        ValueError_call_result_358375 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 18), ValueError_358372, *[str_358373], **kwargs_358374)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1089, 12), ValueError_call_result_358375, 'raise parameter', BaseException)
        # SSA join for if statement (line 1088)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _setdiag(...): (line 1090)
        # Processing the call arguments (line 1090)
        
        # Call to asarray(...): (line 1090)
        # Processing the call arguments (line 1090)
        # Getting the type of 'values' (line 1090)
        values_358380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 33), 'values', False)
        # Processing the call keyword arguments (line 1090)
        kwargs_358381 = {}
        # Getting the type of 'np' (line 1090)
        np_358378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 22), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1090)
        asarray_358379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 22), np_358378, 'asarray')
        # Calling asarray(args, kwargs) (line 1090)
        asarray_call_result_358382 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 22), asarray_358379, *[values_358380], **kwargs_358381)
        
        # Getting the type of 'k' (line 1090)
        k_358383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 42), 'k', False)
        # Processing the call keyword arguments (line 1090)
        kwargs_358384 = {}
        # Getting the type of 'self' (line 1090)
        self_358376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 8), 'self', False)
        # Obtaining the member '_setdiag' of a type (line 1090)
        _setdiag_358377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 8), self_358376, '_setdiag')
        # Calling _setdiag(args, kwargs) (line 1090)
        _setdiag_call_result_358385 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 8), _setdiag_358377, *[asarray_call_result_358382, k_358383], **kwargs_358384)
        
        
        # ################# End of 'setdiag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setdiag' in the type store
        # Getting the type of 'stypy_return_type' (line 1067)
        stypy_return_type_358386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setdiag'
        return stypy_return_type_358386


    @norecursion
    def _setdiag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_setdiag'
        module_type_store = module_type_store.open_function_context('_setdiag', 1092, 4, False)
        # Assigning a type to the variable 'self' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._setdiag.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._setdiag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._setdiag.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._setdiag.__dict__.__setitem__('stypy_function_name', 'spmatrix._setdiag')
        spmatrix._setdiag.__dict__.__setitem__('stypy_param_names_list', ['values', 'k'])
        spmatrix._setdiag.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._setdiag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._setdiag.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._setdiag.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._setdiag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._setdiag.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._setdiag', ['values', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_setdiag', localization, ['values', 'k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_setdiag(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 1093):
        
        # Assigning a Subscript to a Name (line 1093):
        
        # Obtaining the type of the subscript
        int_358387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 8), 'int')
        # Getting the type of 'self' (line 1093)
        self_358388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 15), 'self')
        # Obtaining the member 'shape' of a type (line 1093)
        shape_358389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 15), self_358388, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1093)
        getitem___358390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 8), shape_358389, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1093)
        subscript_call_result_358391 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 8), getitem___358390, int_358387)
        
        # Assigning a type to the variable 'tuple_var_assignment_356182' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'tuple_var_assignment_356182', subscript_call_result_358391)
        
        # Assigning a Subscript to a Name (line 1093):
        
        # Obtaining the type of the subscript
        int_358392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 8), 'int')
        # Getting the type of 'self' (line 1093)
        self_358393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 15), 'self')
        # Obtaining the member 'shape' of a type (line 1093)
        shape_358394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 15), self_358393, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1093)
        getitem___358395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 8), shape_358394, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1093)
        subscript_call_result_358396 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 8), getitem___358395, int_358392)
        
        # Assigning a type to the variable 'tuple_var_assignment_356183' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'tuple_var_assignment_356183', subscript_call_result_358396)
        
        # Assigning a Name to a Name (line 1093):
        # Getting the type of 'tuple_var_assignment_356182' (line 1093)
        tuple_var_assignment_356182_358397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'tuple_var_assignment_356182')
        # Assigning a type to the variable 'M' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'M', tuple_var_assignment_356182_358397)
        
        # Assigning a Name to a Name (line 1093):
        # Getting the type of 'tuple_var_assignment_356183' (line 1093)
        tuple_var_assignment_356183_358398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'tuple_var_assignment_356183')
        # Assigning a type to the variable 'N' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 11), 'N', tuple_var_assignment_356183_358398)
        
        
        # Getting the type of 'k' (line 1094)
        k_358399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 11), 'k')
        int_358400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 15), 'int')
        # Applying the binary operator '<' (line 1094)
        result_lt_358401 = python_operator(stypy.reporting.localization.Localization(__file__, 1094, 11), '<', k_358399, int_358400)
        
        # Testing the type of an if condition (line 1094)
        if_condition_358402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1094, 8), result_lt_358401)
        # Assigning a type to the variable 'if_condition_358402' (line 1094)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'if_condition_358402', if_condition_358402)
        # SSA begins for if statement (line 1094)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'values' (line 1095)
        values_358403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 15), 'values')
        # Obtaining the member 'ndim' of a type (line 1095)
        ndim_358404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1095, 15), values_358403, 'ndim')
        int_358405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1095, 30), 'int')
        # Applying the binary operator '==' (line 1095)
        result_eq_358406 = python_operator(stypy.reporting.localization.Localization(__file__, 1095, 15), '==', ndim_358404, int_358405)
        
        # Testing the type of an if condition (line 1095)
        if_condition_358407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1095, 12), result_eq_358406)
        # Assigning a type to the variable 'if_condition_358407' (line 1095)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 12), 'if_condition_358407', if_condition_358407)
        # SSA begins for if statement (line 1095)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1097):
        
        # Assigning a Call to a Name (line 1097):
        
        # Call to min(...): (line 1097)
        # Processing the call arguments (line 1097)
        # Getting the type of 'M' (line 1097)
        M_358409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 32), 'M', False)
        # Getting the type of 'k' (line 1097)
        k_358410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 34), 'k', False)
        # Applying the binary operator '+' (line 1097)
        result_add_358411 = python_operator(stypy.reporting.localization.Localization(__file__, 1097, 32), '+', M_358409, k_358410)
        
        # Getting the type of 'N' (line 1097)
        N_358412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 37), 'N', False)
        # Processing the call keyword arguments (line 1097)
        kwargs_358413 = {}
        # Getting the type of 'min' (line 1097)
        min_358408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 28), 'min', False)
        # Calling min(args, kwargs) (line 1097)
        min_call_result_358414 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 28), min_358408, *[result_add_358411, N_358412], **kwargs_358413)
        
        # Assigning a type to the variable 'max_index' (line 1097)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 16), 'max_index', min_call_result_358414)
        
        
        # Call to xrange(...): (line 1098)
        # Processing the call arguments (line 1098)
        # Getting the type of 'max_index' (line 1098)
        max_index_358416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 32), 'max_index', False)
        # Processing the call keyword arguments (line 1098)
        kwargs_358417 = {}
        # Getting the type of 'xrange' (line 1098)
        xrange_358415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 25), 'xrange', False)
        # Calling xrange(args, kwargs) (line 1098)
        xrange_call_result_358418 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 25), xrange_358415, *[max_index_358416], **kwargs_358417)
        
        # Testing the type of a for loop iterable (line 1098)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1098, 16), xrange_call_result_358418)
        # Getting the type of the for loop variable (line 1098)
        for_loop_var_358419 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1098, 16), xrange_call_result_358418)
        # Assigning a type to the variable 'i' (line 1098)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1098, 16), 'i', for_loop_var_358419)
        # SSA begins for a for statement (line 1098)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 1099):
        
        # Assigning a Name to a Subscript (line 1099):
        # Getting the type of 'values' (line 1099)
        values_358420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 37), 'values')
        # Getting the type of 'self' (line 1099)
        self_358421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 20), 'self')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1099)
        tuple_358422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1099)
        # Adding element type (line 1099)
        # Getting the type of 'i' (line 1099)
        i_358423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 25), 'i')
        # Getting the type of 'k' (line 1099)
        k_358424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 29), 'k')
        # Applying the binary operator '-' (line 1099)
        result_sub_358425 = python_operator(stypy.reporting.localization.Localization(__file__, 1099, 25), '-', i_358423, k_358424)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1099, 25), tuple_358422, result_sub_358425)
        # Adding element type (line 1099)
        # Getting the type of 'i' (line 1099)
        i_358426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 32), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1099, 25), tuple_358422, i_358426)
        
        # Storing an element on a container (line 1099)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1099, 20), self_358421, (tuple_358422, values_358420))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1095)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1101):
        
        # Assigning a Call to a Name (line 1101):
        
        # Call to min(...): (line 1101)
        # Processing the call arguments (line 1101)
        # Getting the type of 'M' (line 1101)
        M_358428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 32), 'M', False)
        # Getting the type of 'k' (line 1101)
        k_358429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 34), 'k', False)
        # Applying the binary operator '+' (line 1101)
        result_add_358430 = python_operator(stypy.reporting.localization.Localization(__file__, 1101, 32), '+', M_358428, k_358429)
        
        # Getting the type of 'N' (line 1101)
        N_358431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 37), 'N', False)
        
        # Call to len(...): (line 1101)
        # Processing the call arguments (line 1101)
        # Getting the type of 'values' (line 1101)
        values_358433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 44), 'values', False)
        # Processing the call keyword arguments (line 1101)
        kwargs_358434 = {}
        # Getting the type of 'len' (line 1101)
        len_358432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 40), 'len', False)
        # Calling len(args, kwargs) (line 1101)
        len_call_result_358435 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 40), len_358432, *[values_358433], **kwargs_358434)
        
        # Processing the call keyword arguments (line 1101)
        kwargs_358436 = {}
        # Getting the type of 'min' (line 1101)
        min_358427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 28), 'min', False)
        # Calling min(args, kwargs) (line 1101)
        min_call_result_358437 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 28), min_358427, *[result_add_358430, N_358431, len_call_result_358435], **kwargs_358436)
        
        # Assigning a type to the variable 'max_index' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 16), 'max_index', min_call_result_358437)
        
        
        # Getting the type of 'max_index' (line 1102)
        max_index_358438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 19), 'max_index')
        int_358439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 32), 'int')
        # Applying the binary operator '<=' (line 1102)
        result_le_358440 = python_operator(stypy.reporting.localization.Localization(__file__, 1102, 19), '<=', max_index_358438, int_358439)
        
        # Testing the type of an if condition (line 1102)
        if_condition_358441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1102, 16), result_le_358440)
        # Assigning a type to the variable 'if_condition_358441' (line 1102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1102, 16), 'if_condition_358441', if_condition_358441)
        # SSA begins for if statement (line 1102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 1103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 20), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 1102)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 1104)
        # Processing the call arguments (line 1104)
        
        # Obtaining the type of the subscript
        # Getting the type of 'max_index' (line 1104)
        max_index_358443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 46), 'max_index', False)
        slice_358444 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1104, 38), None, max_index_358443, None)
        # Getting the type of 'values' (line 1104)
        values_358445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 38), 'values', False)
        # Obtaining the member '__getitem__' of a type (line 1104)
        getitem___358446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1104, 38), values_358445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1104)
        subscript_call_result_358447 = invoke(stypy.reporting.localization.Localization(__file__, 1104, 38), getitem___358446, slice_358444)
        
        # Processing the call keyword arguments (line 1104)
        kwargs_358448 = {}
        # Getting the type of 'enumerate' (line 1104)
        enumerate_358442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 28), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 1104)
        enumerate_call_result_358449 = invoke(stypy.reporting.localization.Localization(__file__, 1104, 28), enumerate_358442, *[subscript_call_result_358447], **kwargs_358448)
        
        # Testing the type of a for loop iterable (line 1104)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1104, 16), enumerate_call_result_358449)
        # Getting the type of the for loop variable (line 1104)
        for_loop_var_358450 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1104, 16), enumerate_call_result_358449)
        # Assigning a type to the variable 'i' (line 1104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 16), for_loop_var_358450))
        # Assigning a type to the variable 'v' (line 1104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 16), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 16), for_loop_var_358450))
        # SSA begins for a for statement (line 1104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 1105):
        
        # Assigning a Name to a Subscript (line 1105):
        # Getting the type of 'v' (line 1105)
        v_358451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 37), 'v')
        # Getting the type of 'self' (line 1105)
        self_358452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 20), 'self')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1105)
        tuple_358453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1105)
        # Adding element type (line 1105)
        # Getting the type of 'i' (line 1105)
        i_358454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 25), 'i')
        # Getting the type of 'k' (line 1105)
        k_358455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 29), 'k')
        # Applying the binary operator '-' (line 1105)
        result_sub_358456 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 25), '-', i_358454, k_358455)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 25), tuple_358453, result_sub_358456)
        # Adding element type (line 1105)
        # Getting the type of 'i' (line 1105)
        i_358457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 32), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 25), tuple_358453, i_358457)
        
        # Storing an element on a container (line 1105)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 20), self_358452, (tuple_358453, v_358451))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1095)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1094)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'values' (line 1107)
        values_358458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 15), 'values')
        # Obtaining the member 'ndim' of a type (line 1107)
        ndim_358459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1107, 15), values_358458, 'ndim')
        int_358460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 30), 'int')
        # Applying the binary operator '==' (line 1107)
        result_eq_358461 = python_operator(stypy.reporting.localization.Localization(__file__, 1107, 15), '==', ndim_358459, int_358460)
        
        # Testing the type of an if condition (line 1107)
        if_condition_358462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1107, 12), result_eq_358461)
        # Assigning a type to the variable 'if_condition_358462' (line 1107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1107, 12), 'if_condition_358462', if_condition_358462)
        # SSA begins for if statement (line 1107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1109):
        
        # Assigning a Call to a Name (line 1109):
        
        # Call to min(...): (line 1109)
        # Processing the call arguments (line 1109)
        # Getting the type of 'M' (line 1109)
        M_358464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 32), 'M', False)
        # Getting the type of 'N' (line 1109)
        N_358465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 35), 'N', False)
        # Getting the type of 'k' (line 1109)
        k_358466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 37), 'k', False)
        # Applying the binary operator '-' (line 1109)
        result_sub_358467 = python_operator(stypy.reporting.localization.Localization(__file__, 1109, 35), '-', N_358465, k_358466)
        
        # Processing the call keyword arguments (line 1109)
        kwargs_358468 = {}
        # Getting the type of 'min' (line 1109)
        min_358463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 28), 'min', False)
        # Calling min(args, kwargs) (line 1109)
        min_call_result_358469 = invoke(stypy.reporting.localization.Localization(__file__, 1109, 28), min_358463, *[M_358464, result_sub_358467], **kwargs_358468)
        
        # Assigning a type to the variable 'max_index' (line 1109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 16), 'max_index', min_call_result_358469)
        
        
        # Call to xrange(...): (line 1110)
        # Processing the call arguments (line 1110)
        # Getting the type of 'max_index' (line 1110)
        max_index_358471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 32), 'max_index', False)
        # Processing the call keyword arguments (line 1110)
        kwargs_358472 = {}
        # Getting the type of 'xrange' (line 1110)
        xrange_358470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 25), 'xrange', False)
        # Calling xrange(args, kwargs) (line 1110)
        xrange_call_result_358473 = invoke(stypy.reporting.localization.Localization(__file__, 1110, 25), xrange_358470, *[max_index_358471], **kwargs_358472)
        
        # Testing the type of a for loop iterable (line 1110)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1110, 16), xrange_call_result_358473)
        # Getting the type of the for loop variable (line 1110)
        for_loop_var_358474 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1110, 16), xrange_call_result_358473)
        # Assigning a type to the variable 'i' (line 1110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 16), 'i', for_loop_var_358474)
        # SSA begins for a for statement (line 1110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 1111):
        
        # Assigning a Name to a Subscript (line 1111):
        # Getting the type of 'values' (line 1111)
        values_358475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 37), 'values')
        # Getting the type of 'self' (line 1111)
        self_358476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 20), 'self')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1111)
        tuple_358477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1111, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1111)
        # Adding element type (line 1111)
        # Getting the type of 'i' (line 1111)
        i_358478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 25), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1111, 25), tuple_358477, i_358478)
        # Adding element type (line 1111)
        # Getting the type of 'i' (line 1111)
        i_358479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 28), 'i')
        # Getting the type of 'k' (line 1111)
        k_358480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 32), 'k')
        # Applying the binary operator '+' (line 1111)
        result_add_358481 = python_operator(stypy.reporting.localization.Localization(__file__, 1111, 28), '+', i_358479, k_358480)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1111, 25), tuple_358477, result_add_358481)
        
        # Storing an element on a container (line 1111)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1111, 20), self_358476, (tuple_358477, values_358475))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1107)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1113):
        
        # Assigning a Call to a Name (line 1113):
        
        # Call to min(...): (line 1113)
        # Processing the call arguments (line 1113)
        # Getting the type of 'M' (line 1113)
        M_358483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 32), 'M', False)
        # Getting the type of 'N' (line 1113)
        N_358484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 35), 'N', False)
        # Getting the type of 'k' (line 1113)
        k_358485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 37), 'k', False)
        # Applying the binary operator '-' (line 1113)
        result_sub_358486 = python_operator(stypy.reporting.localization.Localization(__file__, 1113, 35), '-', N_358484, k_358485)
        
        
        # Call to len(...): (line 1113)
        # Processing the call arguments (line 1113)
        # Getting the type of 'values' (line 1113)
        values_358488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 44), 'values', False)
        # Processing the call keyword arguments (line 1113)
        kwargs_358489 = {}
        # Getting the type of 'len' (line 1113)
        len_358487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 40), 'len', False)
        # Calling len(args, kwargs) (line 1113)
        len_call_result_358490 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 40), len_358487, *[values_358488], **kwargs_358489)
        
        # Processing the call keyword arguments (line 1113)
        kwargs_358491 = {}
        # Getting the type of 'min' (line 1113)
        min_358482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 28), 'min', False)
        # Calling min(args, kwargs) (line 1113)
        min_call_result_358492 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 28), min_358482, *[M_358483, result_sub_358486, len_call_result_358490], **kwargs_358491)
        
        # Assigning a type to the variable 'max_index' (line 1113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 16), 'max_index', min_call_result_358492)
        
        
        # Getting the type of 'max_index' (line 1114)
        max_index_358493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 19), 'max_index')
        int_358494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 32), 'int')
        # Applying the binary operator '<=' (line 1114)
        result_le_358495 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 19), '<=', max_index_358493, int_358494)
        
        # Testing the type of an if condition (line 1114)
        if_condition_358496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1114, 16), result_le_358495)
        # Assigning a type to the variable 'if_condition_358496' (line 1114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 16), 'if_condition_358496', if_condition_358496)
        # SSA begins for if statement (line 1114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 1115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 20), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 1114)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 1116)
        # Processing the call arguments (line 1116)
        
        # Obtaining the type of the subscript
        # Getting the type of 'max_index' (line 1116)
        max_index_358498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 46), 'max_index', False)
        slice_358499 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1116, 38), None, max_index_358498, None)
        # Getting the type of 'values' (line 1116)
        values_358500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 38), 'values', False)
        # Obtaining the member '__getitem__' of a type (line 1116)
        getitem___358501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1116, 38), values_358500, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1116)
        subscript_call_result_358502 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 38), getitem___358501, slice_358499)
        
        # Processing the call keyword arguments (line 1116)
        kwargs_358503 = {}
        # Getting the type of 'enumerate' (line 1116)
        enumerate_358497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 28), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 1116)
        enumerate_call_result_358504 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 28), enumerate_358497, *[subscript_call_result_358502], **kwargs_358503)
        
        # Testing the type of a for loop iterable (line 1116)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1116, 16), enumerate_call_result_358504)
        # Getting the type of the for loop variable (line 1116)
        for_loop_var_358505 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1116, 16), enumerate_call_result_358504)
        # Assigning a type to the variable 'i' (line 1116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 16), for_loop_var_358505))
        # Assigning a type to the variable 'v' (line 1116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 16), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 16), for_loop_var_358505))
        # SSA begins for a for statement (line 1116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 1117):
        
        # Assigning a Name to a Subscript (line 1117):
        # Getting the type of 'v' (line 1117)
        v_358506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 37), 'v')
        # Getting the type of 'self' (line 1117)
        self_358507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 20), 'self')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1117)
        tuple_358508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1117)
        # Adding element type (line 1117)
        # Getting the type of 'i' (line 1117)
        i_358509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 25), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 25), tuple_358508, i_358509)
        # Adding element type (line 1117)
        # Getting the type of 'i' (line 1117)
        i_358510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 28), 'i')
        # Getting the type of 'k' (line 1117)
        k_358511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 32), 'k')
        # Applying the binary operator '+' (line 1117)
        result_add_358512 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 28), '+', i_358510, k_358511)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 25), tuple_358508, result_add_358512)
        
        # Storing an element on a container (line 1117)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 20), self_358507, (tuple_358508, v_358506))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1107)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1094)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_setdiag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setdiag' in the type store
        # Getting the type of 'stypy_return_type' (line 1092)
        stypy_return_type_358513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setdiag'
        return stypy_return_type_358513


    @norecursion
    def _process_toarray_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_process_toarray_args'
        module_type_store = module_type_store.open_function_context('_process_toarray_args', 1119, 4, False)
        # Assigning a type to the variable 'self' (line 1120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_localization', localization)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_function_name', 'spmatrix._process_toarray_args')
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_param_names_list', ['order', 'out'])
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix._process_toarray_args.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix._process_toarray_args', ['order', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_process_toarray_args', localization, ['order', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_process_toarray_args(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 1120)
        # Getting the type of 'out' (line 1120)
        out_358514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 8), 'out')
        # Getting the type of 'None' (line 1120)
        None_358515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 22), 'None')
        
        (may_be_358516, more_types_in_union_358517) = may_not_be_none(out_358514, None_358515)

        if may_be_358516:

            if more_types_in_union_358517:
                # Runtime conditional SSA (line 1120)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 1121)
            # Getting the type of 'order' (line 1121)
            order_358518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 12), 'order')
            # Getting the type of 'None' (line 1121)
            None_358519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 28), 'None')
            
            (may_be_358520, more_types_in_union_358521) = may_not_be_none(order_358518, None_358519)

            if may_be_358520:

                if more_types_in_union_358521:
                    # Runtime conditional SSA (line 1121)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 1122)
                # Processing the call arguments (line 1122)
                str_358523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 33), 'str', 'order cannot be specified if out is not None')
                # Processing the call keyword arguments (line 1122)
                kwargs_358524 = {}
                # Getting the type of 'ValueError' (line 1122)
                ValueError_358522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 1122)
                ValueError_call_result_358525 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 22), ValueError_358522, *[str_358523], **kwargs_358524)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1122, 16), ValueError_call_result_358525, 'raise parameter', BaseException)

                if more_types_in_union_358521:
                    # SSA join for if statement (line 1121)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'out' (line 1124)
            out_358526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 15), 'out')
            # Obtaining the member 'shape' of a type (line 1124)
            shape_358527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 15), out_358526, 'shape')
            # Getting the type of 'self' (line 1124)
            self_358528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 28), 'self')
            # Obtaining the member 'shape' of a type (line 1124)
            shape_358529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 28), self_358528, 'shape')
            # Applying the binary operator '!=' (line 1124)
            result_ne_358530 = python_operator(stypy.reporting.localization.Localization(__file__, 1124, 15), '!=', shape_358527, shape_358529)
            
            
            # Getting the type of 'out' (line 1124)
            out_358531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 42), 'out')
            # Obtaining the member 'dtype' of a type (line 1124)
            dtype_358532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 42), out_358531, 'dtype')
            # Getting the type of 'self' (line 1124)
            self_358533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 55), 'self')
            # Obtaining the member 'dtype' of a type (line 1124)
            dtype_358534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 55), self_358533, 'dtype')
            # Applying the binary operator '!=' (line 1124)
            result_ne_358535 = python_operator(stypy.reporting.localization.Localization(__file__, 1124, 42), '!=', dtype_358532, dtype_358534)
            
            # Applying the binary operator 'or' (line 1124)
            result_or_keyword_358536 = python_operator(stypy.reporting.localization.Localization(__file__, 1124, 15), 'or', result_ne_358530, result_ne_358535)
            
            # Testing the type of an if condition (line 1124)
            if_condition_358537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1124, 12), result_or_keyword_358536)
            # Assigning a type to the variable 'if_condition_358537' (line 1124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 12), 'if_condition_358537', if_condition_358537)
            # SSA begins for if statement (line 1124)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 1125)
            # Processing the call arguments (line 1125)
            str_358539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 33), 'str', 'out array must be same dtype and shape as sparse matrix')
            # Processing the call keyword arguments (line 1125)
            kwargs_358540 = {}
            # Getting the type of 'ValueError' (line 1125)
            ValueError_358538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 1125)
            ValueError_call_result_358541 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 22), ValueError_358538, *[str_358539], **kwargs_358540)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1125, 16), ValueError_call_result_358541, 'raise parameter', BaseException)
            # SSA join for if statement (line 1124)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Num to a Subscript (line 1127):
            
            # Assigning a Num to a Subscript (line 1127):
            float_358542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 23), 'float')
            # Getting the type of 'out' (line 1127)
            out_358543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 12), 'out')
            Ellipsis_358544 = Ellipsis
            # Storing an element on a container (line 1127)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1127, 12), out_358543, (Ellipsis_358544, float_358542))
            # Getting the type of 'out' (line 1128)
            out_358545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 19), 'out')
            # Assigning a type to the variable 'stypy_return_type' (line 1128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1128, 12), 'stypy_return_type', out_358545)

            if more_types_in_union_358517:
                # Runtime conditional SSA for else branch (line 1120)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_358516) or more_types_in_union_358517):
            
            # Call to zeros(...): (line 1130)
            # Processing the call arguments (line 1130)
            # Getting the type of 'self' (line 1130)
            self_358548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 28), 'self', False)
            # Obtaining the member 'shape' of a type (line 1130)
            shape_358549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 28), self_358548, 'shape')
            # Processing the call keyword arguments (line 1130)
            # Getting the type of 'self' (line 1130)
            self_358550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 46), 'self', False)
            # Obtaining the member 'dtype' of a type (line 1130)
            dtype_358551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 46), self_358550, 'dtype')
            keyword_358552 = dtype_358551
            # Getting the type of 'order' (line 1130)
            order_358553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 64), 'order', False)
            keyword_358554 = order_358553
            kwargs_358555 = {'dtype': keyword_358552, 'order': keyword_358554}
            # Getting the type of 'np' (line 1130)
            np_358546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 19), 'np', False)
            # Obtaining the member 'zeros' of a type (line 1130)
            zeros_358547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 19), np_358546, 'zeros')
            # Calling zeros(args, kwargs) (line 1130)
            zeros_call_result_358556 = invoke(stypy.reporting.localization.Localization(__file__, 1130, 19), zeros_358547, *[shape_358549], **kwargs_358555)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1130, 12), 'stypy_return_type', zeros_call_result_358556)

            if (may_be_358516 and more_types_in_union_358517):
                # SSA join for if statement (line 1120)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_process_toarray_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_process_toarray_args' in the type store
        # Getting the type of 'stypy_return_type' (line 1119)
        stypy_return_type_358557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_process_toarray_args'
        return stypy_return_type_358557


    @norecursion
    def __numpy_ufunc__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__numpy_ufunc__'
        module_type_store = module_type_store.open_function_context('__numpy_ufunc__', 1132, 4, False)
        # Assigning a type to the variable 'self' (line 1133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_localization', localization)
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_function_name', 'spmatrix.__numpy_ufunc__')
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_param_names_list', ['func', 'method', 'pos', 'inputs'])
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spmatrix.__numpy_ufunc__.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spmatrix.__numpy_ufunc__', ['func', 'method', 'pos', 'inputs'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__numpy_ufunc__', localization, ['func', 'method', 'pos', 'inputs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__numpy_ufunc__(...)' code ##################

        str_358558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, (-1)), 'str', "Method for compatibility with NumPy's ufuncs and dot\n        functions.\n        ")
        
        
        # Call to any(...): (line 1137)
        # Processing the call arguments (line 1137)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 1137, 15, True)
        # Calculating comprehension expression
        # Getting the type of 'inputs' (line 1138)
        inputs_358575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 24), 'inputs', False)
        comprehension_358576 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 15), inputs_358575)
        # Assigning a type to the variable 'x' (line 1137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 15), 'x', comprehension_358576)
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 1137)
        # Processing the call arguments (line 1137)
        # Getting the type of 'x' (line 1137)
        x_358561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 30), 'x', False)
        # Getting the type of 'spmatrix' (line 1137)
        spmatrix_358562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 33), 'spmatrix', False)
        # Processing the call keyword arguments (line 1137)
        kwargs_358563 = {}
        # Getting the type of 'isinstance' (line 1137)
        isinstance_358560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1137)
        isinstance_call_result_358564 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 19), isinstance_358560, *[x_358561, spmatrix_358562], **kwargs_358563)
        
        # Applying the 'not' unary operator (line 1137)
        result_not__358565 = python_operator(stypy.reporting.localization.Localization(__file__, 1137, 15), 'not', isinstance_call_result_358564)
        
        
        
        # Call to asarray(...): (line 1137)
        # Processing the call arguments (line 1137)
        # Getting the type of 'x' (line 1137)
        x_358568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 58), 'x', False)
        # Processing the call keyword arguments (line 1137)
        kwargs_358569 = {}
        # Getting the type of 'np' (line 1137)
        np_358566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 47), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1137)
        asarray_358567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 47), np_358566, 'asarray')
        # Calling asarray(args, kwargs) (line 1137)
        asarray_call_result_358570 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 47), asarray_358567, *[x_358568], **kwargs_358569)
        
        # Obtaining the member 'dtype' of a type (line 1137)
        dtype_358571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 47), asarray_call_result_358570, 'dtype')
        # Getting the type of 'object' (line 1137)
        object_358572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 70), 'object', False)
        # Applying the binary operator '==' (line 1137)
        result_eq_358573 = python_operator(stypy.reporting.localization.Localization(__file__, 1137, 47), '==', dtype_358571, object_358572)
        
        # Applying the binary operator 'and' (line 1137)
        result_and_keyword_358574 = python_operator(stypy.reporting.localization.Localization(__file__, 1137, 15), 'and', result_not__358565, result_eq_358573)
        
        list_358577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 15), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 15), list_358577, result_and_keyword_358574)
        # Processing the call keyword arguments (line 1137)
        kwargs_358578 = {}
        # Getting the type of 'any' (line 1137)
        any_358559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 11), 'any', False)
        # Calling any(args, kwargs) (line 1137)
        any_call_result_358579 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 11), any_358559, *[list_358577], **kwargs_358578)
        
        # Testing the type of an if condition (line 1137)
        if_condition_358580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1137, 8), any_call_result_358579)
        # Assigning a type to the variable 'if_condition_358580' (line 1137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 8), 'if_condition_358580', if_condition_358580)
        # SSA begins for if statement (line 1137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1140):
        
        # Assigning a Call to a Name (line 1140):
        
        # Call to list(...): (line 1140)
        # Processing the call arguments (line 1140)
        # Getting the type of 'inputs' (line 1140)
        inputs_358582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 29), 'inputs', False)
        # Processing the call keyword arguments (line 1140)
        kwargs_358583 = {}
        # Getting the type of 'list' (line 1140)
        list_358581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 24), 'list', False)
        # Calling list(args, kwargs) (line 1140)
        list_call_result_358584 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 24), list_358581, *[inputs_358582], **kwargs_358583)
        
        # Assigning a type to the variable 'with_self' (line 1140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 12), 'with_self', list_call_result_358584)
        
        # Assigning a Call to a Subscript (line 1141):
        
        # Assigning a Call to a Subscript (line 1141):
        
        # Call to asarray(...): (line 1141)
        # Processing the call arguments (line 1141)
        # Getting the type of 'self' (line 1141)
        self_358587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 40), 'self', False)
        # Processing the call keyword arguments (line 1141)
        # Getting the type of 'object' (line 1141)
        object_358588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 52), 'object', False)
        keyword_358589 = object_358588
        kwargs_358590 = {'dtype': keyword_358589}
        # Getting the type of 'np' (line 1141)
        np_358585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 29), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1141)
        asarray_358586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 29), np_358585, 'asarray')
        # Calling asarray(args, kwargs) (line 1141)
        asarray_call_result_358591 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 29), asarray_358586, *[self_358587], **kwargs_358590)
        
        # Getting the type of 'with_self' (line 1141)
        with_self_358592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 12), 'with_self')
        # Getting the type of 'pos' (line 1141)
        pos_358593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 22), 'pos')
        # Storing an element on a container (line 1141)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1141, 12), with_self_358592, (pos_358593, asarray_call_result_358591))
        
        # Call to (...): (line 1142)
        # Getting the type of 'with_self' (line 1142)
        with_self_358599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 42), 'with_self', False)
        # Processing the call keyword arguments (line 1142)
        # Getting the type of 'kwargs' (line 1142)
        kwargs_358600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 55), 'kwargs', False)
        kwargs_358601 = {'kwargs_358600': kwargs_358600}
        
        # Call to getattr(...): (line 1142)
        # Processing the call arguments (line 1142)
        # Getting the type of 'func' (line 1142)
        func_358595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 27), 'func', False)
        # Getting the type of 'method' (line 1142)
        method_358596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 33), 'method', False)
        # Processing the call keyword arguments (line 1142)
        kwargs_358597 = {}
        # Getting the type of 'getattr' (line 1142)
        getattr_358594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1142)
        getattr_call_result_358598 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 19), getattr_358594, *[func_358595, method_358596], **kwargs_358597)
        
        # Calling (args, kwargs) (line 1142)
        _call_result_358602 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 19), getattr_call_result_358598, *[with_self_358599], **kwargs_358601)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 12), 'stypy_return_type', _call_result_358602)
        # SSA join for if statement (line 1137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1144):
        
        # Assigning a Call to a Name (line 1144):
        
        # Call to pop(...): (line 1144)
        # Processing the call arguments (line 1144)
        str_358605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 25), 'str', 'out')
        # Getting the type of 'None' (line 1144)
        None_358606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 32), 'None', False)
        # Processing the call keyword arguments (line 1144)
        kwargs_358607 = {}
        # Getting the type of 'kwargs' (line 1144)
        kwargs_358603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 14), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 1144)
        pop_358604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 14), kwargs_358603, 'pop')
        # Calling pop(args, kwargs) (line 1144)
        pop_call_result_358608 = invoke(stypy.reporting.localization.Localization(__file__, 1144, 14), pop_358604, *[str_358605, None_358606], **kwargs_358607)
        
        # Assigning a type to the variable 'out' (line 1144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 8), 'out', pop_call_result_358608)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'method' (line 1145)
        method_358609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 11), 'method')
        str_358610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 21), 'str', '__call__')
        # Applying the binary operator '!=' (line 1145)
        result_ne_358611 = python_operator(stypy.reporting.localization.Localization(__file__, 1145, 11), '!=', method_358609, str_358610)
        
        # Getting the type of 'kwargs' (line 1145)
        kwargs_358612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 35), 'kwargs')
        # Applying the binary operator 'or' (line 1145)
        result_or_keyword_358613 = python_operator(stypy.reporting.localization.Localization(__file__, 1145, 11), 'or', result_ne_358611, kwargs_358612)
        
        # Testing the type of an if condition (line 1145)
        if_condition_358614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1145, 8), result_or_keyword_358613)
        # Assigning a type to the variable 'if_condition_358614' (line 1145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1145, 8), 'if_condition_358614', if_condition_358614)
        # SSA begins for if statement (line 1145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 1146)
        NotImplemented_358615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 1146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 12), 'stypy_return_type', NotImplemented_358615)
        # SSA join for if statement (line 1145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1148):
        
        # Assigning a Call to a Name (line 1148):
        
        # Call to list(...): (line 1148)
        # Processing the call arguments (line 1148)
        # Getting the type of 'inputs' (line 1148)
        inputs_358617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 28), 'inputs', False)
        # Processing the call keyword arguments (line 1148)
        kwargs_358618 = {}
        # Getting the type of 'list' (line 1148)
        list_358616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 23), 'list', False)
        # Calling list(args, kwargs) (line 1148)
        list_call_result_358619 = invoke(stypy.reporting.localization.Localization(__file__, 1148, 23), list_358616, *[inputs_358617], **kwargs_358618)
        
        # Assigning a type to the variable 'without_self' (line 1148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 8), 'without_self', list_call_result_358619)
        # Deleting a member
        # Getting the type of 'without_self' (line 1149)
        without_self_358620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 12), 'without_self')
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 1149)
        pos_358621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 25), 'pos')
        # Getting the type of 'without_self' (line 1149)
        without_self_358622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 12), 'without_self')
        # Obtaining the member '__getitem__' of a type (line 1149)
        getitem___358623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 12), without_self_358622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1149)
        subscript_call_result_358624 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 12), getitem___358623, pos_358621)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1149, 8), without_self_358620, subscript_call_result_358624)
        
        # Assigning a Call to a Name (line 1150):
        
        # Assigning a Call to a Name (line 1150):
        
        # Call to tuple(...): (line 1150)
        # Processing the call arguments (line 1150)
        # Getting the type of 'without_self' (line 1150)
        without_self_358626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 29), 'without_self', False)
        # Processing the call keyword arguments (line 1150)
        kwargs_358627 = {}
        # Getting the type of 'tuple' (line 1150)
        tuple_358625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 23), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1150)
        tuple_call_result_358628 = invoke(stypy.reporting.localization.Localization(__file__, 1150, 23), tuple_358625, *[without_self_358626], **kwargs_358627)
        
        # Assigning a type to the variable 'without_self' (line 1150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 8), 'without_self', tuple_call_result_358628)
        
        
        # Getting the type of 'func' (line 1152)
        func_358629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 11), 'func')
        # Getting the type of 'np' (line 1152)
        np_358630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 19), 'np')
        # Obtaining the member 'multiply' of a type (line 1152)
        multiply_358631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1152, 19), np_358630, 'multiply')
        # Applying the binary operator 'is' (line 1152)
        result_is__358632 = python_operator(stypy.reporting.localization.Localization(__file__, 1152, 11), 'is', func_358629, multiply_358631)
        
        # Testing the type of an if condition (line 1152)
        if_condition_358633 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1152, 8), result_is__358632)
        # Assigning a type to the variable 'if_condition_358633' (line 1152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 8), 'if_condition_358633', if_condition_358633)
        # SSA begins for if statement (line 1152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1153):
        
        # Assigning a Call to a Name (line 1153):
        
        # Call to multiply(...): (line 1153)
        # Getting the type of 'without_self' (line 1153)
        without_self_358636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 36), 'without_self', False)
        # Processing the call keyword arguments (line 1153)
        kwargs_358637 = {}
        # Getting the type of 'self' (line 1153)
        self_358634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 21), 'self', False)
        # Obtaining the member 'multiply' of a type (line 1153)
        multiply_358635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 21), self_358634, 'multiply')
        # Calling multiply(args, kwargs) (line 1153)
        multiply_call_result_358638 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 21), multiply_358635, *[without_self_358636], **kwargs_358637)
        
        # Assigning a type to the variable 'result' (line 1153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'result', multiply_call_result_358638)
        # SSA branch for the else part of an if statement (line 1152)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1154)
        func_358639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 13), 'func')
        # Getting the type of 'np' (line 1154)
        np_358640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 21), 'np')
        # Obtaining the member 'add' of a type (line 1154)
        add_358641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1154, 21), np_358640, 'add')
        # Applying the binary operator 'is' (line 1154)
        result_is__358642 = python_operator(stypy.reporting.localization.Localization(__file__, 1154, 13), 'is', func_358639, add_358641)
        
        # Testing the type of an if condition (line 1154)
        if_condition_358643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1154, 13), result_is__358642)
        # Assigning a type to the variable 'if_condition_358643' (line 1154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 13), 'if_condition_358643', if_condition_358643)
        # SSA begins for if statement (line 1154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1155):
        
        # Assigning a Call to a Name (line 1155):
        
        # Call to __add__(...): (line 1155)
        # Getting the type of 'without_self' (line 1155)
        without_self_358646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 35), 'without_self', False)
        # Processing the call keyword arguments (line 1155)
        kwargs_358647 = {}
        # Getting the type of 'self' (line 1155)
        self_358644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 21), 'self', False)
        # Obtaining the member '__add__' of a type (line 1155)
        add___358645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1155, 21), self_358644, '__add__')
        # Calling __add__(args, kwargs) (line 1155)
        add___call_result_358648 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 21), add___358645, *[without_self_358646], **kwargs_358647)
        
        # Assigning a type to the variable 'result' (line 1155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 12), 'result', add___call_result_358648)
        # SSA branch for the else part of an if statement (line 1154)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1156)
        func_358649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 13), 'func')
        # Getting the type of 'np' (line 1156)
        np_358650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 21), 'np')
        # Obtaining the member 'dot' of a type (line 1156)
        dot_358651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1156, 21), np_358650, 'dot')
        # Applying the binary operator 'is' (line 1156)
        result_is__358652 = python_operator(stypy.reporting.localization.Localization(__file__, 1156, 13), 'is', func_358649, dot_358651)
        
        # Testing the type of an if condition (line 1156)
        if_condition_358653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1156, 13), result_is__358652)
        # Assigning a type to the variable 'if_condition_358653' (line 1156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 13), 'if_condition_358653', if_condition_358653)
        # SSA begins for if statement (line 1156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'pos' (line 1157)
        pos_358654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 15), 'pos')
        int_358655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1157, 22), 'int')
        # Applying the binary operator '==' (line 1157)
        result_eq_358656 = python_operator(stypy.reporting.localization.Localization(__file__, 1157, 15), '==', pos_358654, int_358655)
        
        # Testing the type of an if condition (line 1157)
        if_condition_358657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1157, 12), result_eq_358656)
        # Assigning a type to the variable 'if_condition_358657' (line 1157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 12), 'if_condition_358657', if_condition_358657)
        # SSA begins for if statement (line 1157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1158):
        
        # Assigning a Call to a Name (line 1158):
        
        # Call to __mul__(...): (line 1158)
        # Processing the call arguments (line 1158)
        
        # Obtaining the type of the subscript
        int_358660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 45), 'int')
        # Getting the type of 'inputs' (line 1158)
        inputs_358661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 38), 'inputs', False)
        # Obtaining the member '__getitem__' of a type (line 1158)
        getitem___358662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 38), inputs_358661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1158)
        subscript_call_result_358663 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 38), getitem___358662, int_358660)
        
        # Processing the call keyword arguments (line 1158)
        kwargs_358664 = {}
        # Getting the type of 'self' (line 1158)
        self_358658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 25), 'self', False)
        # Obtaining the member '__mul__' of a type (line 1158)
        mul___358659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 25), self_358658, '__mul__')
        # Calling __mul__(args, kwargs) (line 1158)
        mul___call_result_358665 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 25), mul___358659, *[subscript_call_result_358663], **kwargs_358664)
        
        # Assigning a type to the variable 'result' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 16), 'result', mul___call_result_358665)
        # SSA branch for the else part of an if statement (line 1157)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1160):
        
        # Assigning a Call to a Name (line 1160):
        
        # Call to __rmul__(...): (line 1160)
        # Processing the call arguments (line 1160)
        
        # Obtaining the type of the subscript
        int_358668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 46), 'int')
        # Getting the type of 'inputs' (line 1160)
        inputs_358669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 39), 'inputs', False)
        # Obtaining the member '__getitem__' of a type (line 1160)
        getitem___358670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1160, 39), inputs_358669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1160)
        subscript_call_result_358671 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 39), getitem___358670, int_358668)
        
        # Processing the call keyword arguments (line 1160)
        kwargs_358672 = {}
        # Getting the type of 'self' (line 1160)
        self_358666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 25), 'self', False)
        # Obtaining the member '__rmul__' of a type (line 1160)
        rmul___358667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1160, 25), self_358666, '__rmul__')
        # Calling __rmul__(args, kwargs) (line 1160)
        rmul___call_result_358673 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 25), rmul___358667, *[subscript_call_result_358671], **kwargs_358672)
        
        # Assigning a type to the variable 'result' (line 1160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1160, 16), 'result', rmul___call_result_358673)
        # SSA join for if statement (line 1157)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1156)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1161)
        func_358674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 13), 'func')
        # Getting the type of 'np' (line 1161)
        np_358675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 21), 'np')
        # Obtaining the member 'subtract' of a type (line 1161)
        subtract_358676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1161, 21), np_358675, 'subtract')
        # Applying the binary operator 'is' (line 1161)
        result_is__358677 = python_operator(stypy.reporting.localization.Localization(__file__, 1161, 13), 'is', func_358674, subtract_358676)
        
        # Testing the type of an if condition (line 1161)
        if_condition_358678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1161, 13), result_is__358677)
        # Assigning a type to the variable 'if_condition_358678' (line 1161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 13), 'if_condition_358678', if_condition_358678)
        # SSA begins for if statement (line 1161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'pos' (line 1162)
        pos_358679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 15), 'pos')
        int_358680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1162, 22), 'int')
        # Applying the binary operator '==' (line 1162)
        result_eq_358681 = python_operator(stypy.reporting.localization.Localization(__file__, 1162, 15), '==', pos_358679, int_358680)
        
        # Testing the type of an if condition (line 1162)
        if_condition_358682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1162, 12), result_eq_358681)
        # Assigning a type to the variable 'if_condition_358682' (line 1162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 12), 'if_condition_358682', if_condition_358682)
        # SSA begins for if statement (line 1162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1163):
        
        # Assigning a Call to a Name (line 1163):
        
        # Call to __sub__(...): (line 1163)
        # Processing the call arguments (line 1163)
        
        # Obtaining the type of the subscript
        int_358685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 45), 'int')
        # Getting the type of 'inputs' (line 1163)
        inputs_358686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 38), 'inputs', False)
        # Obtaining the member '__getitem__' of a type (line 1163)
        getitem___358687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1163, 38), inputs_358686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1163)
        subscript_call_result_358688 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 38), getitem___358687, int_358685)
        
        # Processing the call keyword arguments (line 1163)
        kwargs_358689 = {}
        # Getting the type of 'self' (line 1163)
        self_358683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 25), 'self', False)
        # Obtaining the member '__sub__' of a type (line 1163)
        sub___358684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1163, 25), self_358683, '__sub__')
        # Calling __sub__(args, kwargs) (line 1163)
        sub___call_result_358690 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 25), sub___358684, *[subscript_call_result_358688], **kwargs_358689)
        
        # Assigning a type to the variable 'result' (line 1163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1163, 16), 'result', sub___call_result_358690)
        # SSA branch for the else part of an if statement (line 1162)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1165):
        
        # Assigning a Call to a Name (line 1165):
        
        # Call to __rsub__(...): (line 1165)
        # Processing the call arguments (line 1165)
        
        # Obtaining the type of the subscript
        int_358693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1165, 46), 'int')
        # Getting the type of 'inputs' (line 1165)
        inputs_358694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 39), 'inputs', False)
        # Obtaining the member '__getitem__' of a type (line 1165)
        getitem___358695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 39), inputs_358694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1165)
        subscript_call_result_358696 = invoke(stypy.reporting.localization.Localization(__file__, 1165, 39), getitem___358695, int_358693)
        
        # Processing the call keyword arguments (line 1165)
        kwargs_358697 = {}
        # Getting the type of 'self' (line 1165)
        self_358691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 25), 'self', False)
        # Obtaining the member '__rsub__' of a type (line 1165)
        rsub___358692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 25), self_358691, '__rsub__')
        # Calling __rsub__(args, kwargs) (line 1165)
        rsub___call_result_358698 = invoke(stypy.reporting.localization.Localization(__file__, 1165, 25), rsub___358692, *[subscript_call_result_358696], **kwargs_358697)
        
        # Assigning a type to the variable 'result' (line 1165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 16), 'result', rsub___call_result_358698)
        # SSA join for if statement (line 1162)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1161)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1166)
        func_358699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 13), 'func')
        # Getting the type of 'np' (line 1166)
        np_358700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 21), 'np')
        # Obtaining the member 'divide' of a type (line 1166)
        divide_358701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 21), np_358700, 'divide')
        # Applying the binary operator 'is' (line 1166)
        result_is__358702 = python_operator(stypy.reporting.localization.Localization(__file__, 1166, 13), 'is', func_358699, divide_358701)
        
        # Testing the type of an if condition (line 1166)
        if_condition_358703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1166, 13), result_is__358702)
        # Assigning a type to the variable 'if_condition_358703' (line 1166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1166, 13), 'if_condition_358703', if_condition_358703)
        # SSA begins for if statement (line 1166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Compare to a Name (line 1167):
        
        # Assigning a Compare to a Name (line 1167):
        
        
        # Obtaining the type of the subscript
        int_358704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 44), 'int')
        # Getting the type of 'sys' (line 1167)
        sys_358705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 27), 'sys')
        # Obtaining the member 'version_info' of a type (line 1167)
        version_info_358706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 27), sys_358705, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 1167)
        getitem___358707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 27), version_info_358706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1167)
        subscript_call_result_358708 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 27), getitem___358707, int_358704)
        
        int_358709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 50), 'int')
        # Applying the binary operator '>=' (line 1167)
        result_ge_358710 = python_operator(stypy.reporting.localization.Localization(__file__, 1167, 27), '>=', subscript_call_result_358708, int_358709)
        
        # Assigning a type to the variable 'true_divide' (line 1167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 12), 'true_divide', result_ge_358710)
        
        # Assigning a Compare to a Name (line 1168):
        
        # Assigning a Compare to a Name (line 1168):
        
        # Getting the type of 'pos' (line 1168)
        pos_358711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 23), 'pos')
        int_358712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, 30), 'int')
        # Applying the binary operator '==' (line 1168)
        result_eq_358713 = python_operator(stypy.reporting.localization.Localization(__file__, 1168, 23), '==', pos_358711, int_358712)
        
        # Assigning a type to the variable 'rdivide' (line 1168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1168, 12), 'rdivide', result_eq_358713)
        
        # Assigning a Call to a Name (line 1169):
        
        # Assigning a Call to a Name (line 1169):
        
        # Call to _divide(...): (line 1169)
        # Getting the type of 'without_self' (line 1169)
        without_self_358716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 35), 'without_self', False)
        # Processing the call keyword arguments (line 1169)
        # Getting the type of 'true_divide' (line 1170)
        true_divide_358717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 46), 'true_divide', False)
        keyword_358718 = true_divide_358717
        # Getting the type of 'rdivide' (line 1171)
        rdivide_358719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 42), 'rdivide', False)
        keyword_358720 = rdivide_358719
        kwargs_358721 = {'true_divide': keyword_358718, 'rdivide': keyword_358720}
        # Getting the type of 'self' (line 1169)
        self_358714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 21), 'self', False)
        # Obtaining the member '_divide' of a type (line 1169)
        _divide_358715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1169, 21), self_358714, '_divide')
        # Calling _divide(args, kwargs) (line 1169)
        _divide_call_result_358722 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 21), _divide_358715, *[without_self_358716], **kwargs_358721)
        
        # Assigning a type to the variable 'result' (line 1169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 12), 'result', _divide_call_result_358722)
        # SSA branch for the else part of an if statement (line 1166)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1172)
        func_358723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 13), 'func')
        # Getting the type of 'np' (line 1172)
        np_358724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 21), 'np')
        # Obtaining the member 'true_divide' of a type (line 1172)
        true_divide_358725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1172, 21), np_358724, 'true_divide')
        # Applying the binary operator 'is' (line 1172)
        result_is__358726 = python_operator(stypy.reporting.localization.Localization(__file__, 1172, 13), 'is', func_358723, true_divide_358725)
        
        # Testing the type of an if condition (line 1172)
        if_condition_358727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1172, 13), result_is__358726)
        # Assigning a type to the variable 'if_condition_358727' (line 1172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 13), 'if_condition_358727', if_condition_358727)
        # SSA begins for if statement (line 1172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Compare to a Name (line 1173):
        
        # Assigning a Compare to a Name (line 1173):
        
        # Getting the type of 'pos' (line 1173)
        pos_358728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 23), 'pos')
        int_358729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1173, 30), 'int')
        # Applying the binary operator '==' (line 1173)
        result_eq_358730 = python_operator(stypy.reporting.localization.Localization(__file__, 1173, 23), '==', pos_358728, int_358729)
        
        # Assigning a type to the variable 'rdivide' (line 1173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 12), 'rdivide', result_eq_358730)
        
        # Assigning a Call to a Name (line 1174):
        
        # Assigning a Call to a Name (line 1174):
        
        # Call to _divide(...): (line 1174)
        # Getting the type of 'without_self' (line 1174)
        without_self_358733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 35), 'without_self', False)
        # Processing the call keyword arguments (line 1174)
        # Getting the type of 'True' (line 1175)
        True_358734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 46), 'True', False)
        keyword_358735 = True_358734
        # Getting the type of 'rdivide' (line 1176)
        rdivide_358736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 42), 'rdivide', False)
        keyword_358737 = rdivide_358736
        kwargs_358738 = {'true_divide': keyword_358735, 'rdivide': keyword_358737}
        # Getting the type of 'self' (line 1174)
        self_358731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 21), 'self', False)
        # Obtaining the member '_divide' of a type (line 1174)
        _divide_358732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1174, 21), self_358731, '_divide')
        # Calling _divide(args, kwargs) (line 1174)
        _divide_call_result_358739 = invoke(stypy.reporting.localization.Localization(__file__, 1174, 21), _divide_358732, *[without_self_358733], **kwargs_358738)
        
        # Assigning a type to the variable 'result' (line 1174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 12), 'result', _divide_call_result_358739)
        # SSA branch for the else part of an if statement (line 1172)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1177)
        func_358740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 13), 'func')
        # Getting the type of 'np' (line 1177)
        np_358741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 21), 'np')
        # Obtaining the member 'maximum' of a type (line 1177)
        maximum_358742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1177, 21), np_358741, 'maximum')
        # Applying the binary operator 'is' (line 1177)
        result_is__358743 = python_operator(stypy.reporting.localization.Localization(__file__, 1177, 13), 'is', func_358740, maximum_358742)
        
        # Testing the type of an if condition (line 1177)
        if_condition_358744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1177, 13), result_is__358743)
        # Assigning a type to the variable 'if_condition_358744' (line 1177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 13), 'if_condition_358744', if_condition_358744)
        # SSA begins for if statement (line 1177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1178):
        
        # Assigning a Call to a Name (line 1178):
        
        # Call to maximum(...): (line 1178)
        # Getting the type of 'without_self' (line 1178)
        without_self_358747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 35), 'without_self', False)
        # Processing the call keyword arguments (line 1178)
        kwargs_358748 = {}
        # Getting the type of 'self' (line 1178)
        self_358745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 21), 'self', False)
        # Obtaining the member 'maximum' of a type (line 1178)
        maximum_358746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1178, 21), self_358745, 'maximum')
        # Calling maximum(args, kwargs) (line 1178)
        maximum_call_result_358749 = invoke(stypy.reporting.localization.Localization(__file__, 1178, 21), maximum_358746, *[without_self_358747], **kwargs_358748)
        
        # Assigning a type to the variable 'result' (line 1178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1178, 12), 'result', maximum_call_result_358749)
        # SSA branch for the else part of an if statement (line 1177)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1179)
        func_358750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1179, 13), 'func')
        # Getting the type of 'np' (line 1179)
        np_358751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1179, 21), 'np')
        # Obtaining the member 'minimum' of a type (line 1179)
        minimum_358752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1179, 21), np_358751, 'minimum')
        # Applying the binary operator 'is' (line 1179)
        result_is__358753 = python_operator(stypy.reporting.localization.Localization(__file__, 1179, 13), 'is', func_358750, minimum_358752)
        
        # Testing the type of an if condition (line 1179)
        if_condition_358754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1179, 13), result_is__358753)
        # Assigning a type to the variable 'if_condition_358754' (line 1179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1179, 13), 'if_condition_358754', if_condition_358754)
        # SSA begins for if statement (line 1179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1180):
        
        # Assigning a Call to a Name (line 1180):
        
        # Call to minimum(...): (line 1180)
        # Getting the type of 'without_self' (line 1180)
        without_self_358757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 35), 'without_self', False)
        # Processing the call keyword arguments (line 1180)
        kwargs_358758 = {}
        # Getting the type of 'self' (line 1180)
        self_358755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 21), 'self', False)
        # Obtaining the member 'minimum' of a type (line 1180)
        minimum_358756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1180, 21), self_358755, 'minimum')
        # Calling minimum(args, kwargs) (line 1180)
        minimum_call_result_358759 = invoke(stypy.reporting.localization.Localization(__file__, 1180, 21), minimum_358756, *[without_self_358757], **kwargs_358758)
        
        # Assigning a type to the variable 'result' (line 1180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 12), 'result', minimum_call_result_358759)
        # SSA branch for the else part of an if statement (line 1179)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1181)
        func_358760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 13), 'func')
        # Getting the type of 'np' (line 1181)
        np_358761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 21), 'np')
        # Obtaining the member 'absolute' of a type (line 1181)
        absolute_358762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1181, 21), np_358761, 'absolute')
        # Applying the binary operator 'is' (line 1181)
        result_is__358763 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 13), 'is', func_358760, absolute_358762)
        
        # Testing the type of an if condition (line 1181)
        if_condition_358764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1181, 13), result_is__358763)
        # Assigning a type to the variable 'if_condition_358764' (line 1181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 13), 'if_condition_358764', if_condition_358764)
        # SSA begins for if statement (line 1181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1182):
        
        # Assigning a Call to a Name (line 1182):
        
        # Call to abs(...): (line 1182)
        # Processing the call arguments (line 1182)
        # Getting the type of 'self' (line 1182)
        self_358766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 25), 'self', False)
        # Processing the call keyword arguments (line 1182)
        kwargs_358767 = {}
        # Getting the type of 'abs' (line 1182)
        abs_358765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 21), 'abs', False)
        # Calling abs(args, kwargs) (line 1182)
        abs_call_result_358768 = invoke(stypy.reporting.localization.Localization(__file__, 1182, 21), abs_358765, *[self_358766], **kwargs_358767)
        
        # Assigning a type to the variable 'result' (line 1182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 12), 'result', abs_call_result_358768)
        # SSA branch for the else part of an if statement (line 1181)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'func' (line 1183)
        func_358769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 13), 'func')
        # Getting the type of '_ufuncs_with_fixed_point_at_zero' (line 1183)
        _ufuncs_with_fixed_point_at_zero_358770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 21), '_ufuncs_with_fixed_point_at_zero')
        # Applying the binary operator 'in' (line 1183)
        result_contains_358771 = python_operator(stypy.reporting.localization.Localization(__file__, 1183, 13), 'in', func_358769, _ufuncs_with_fixed_point_at_zero_358770)
        
        # Testing the type of an if condition (line 1183)
        if_condition_358772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1183, 13), result_contains_358771)
        # Assigning a type to the variable 'if_condition_358772' (line 1183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1183, 13), 'if_condition_358772', if_condition_358772)
        # SSA begins for if statement (line 1183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 1184):
        
        # Assigning a Attribute to a Name (line 1184):
        # Getting the type of 'func' (line 1184)
        func_358773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 24), 'func')
        # Obtaining the member '__name__' of a type (line 1184)
        name___358774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1184, 24), func_358773, '__name__')
        # Assigning a type to the variable 'func_name' (line 1184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 12), 'func_name', name___358774)
        
        
        # Call to hasattr(...): (line 1185)
        # Processing the call arguments (line 1185)
        # Getting the type of 'self' (line 1185)
        self_358776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 23), 'self', False)
        # Getting the type of 'func_name' (line 1185)
        func_name_358777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 29), 'func_name', False)
        # Processing the call keyword arguments (line 1185)
        kwargs_358778 = {}
        # Getting the type of 'hasattr' (line 1185)
        hasattr_358775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 1185)
        hasattr_call_result_358779 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 15), hasattr_358775, *[self_358776, func_name_358777], **kwargs_358778)
        
        # Testing the type of an if condition (line 1185)
        if_condition_358780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1185, 12), hasattr_call_result_358779)
        # Assigning a type to the variable 'if_condition_358780' (line 1185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 12), 'if_condition_358780', if_condition_358780)
        # SSA begins for if statement (line 1185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1186):
        
        # Assigning a Call to a Name (line 1186):
        
        # Call to (...): (line 1186)
        # Processing the call keyword arguments (line 1186)
        kwargs_358786 = {}
        
        # Call to getattr(...): (line 1186)
        # Processing the call arguments (line 1186)
        # Getting the type of 'self' (line 1186)
        self_358782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 33), 'self', False)
        # Getting the type of 'func_name' (line 1186)
        func_name_358783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 39), 'func_name', False)
        # Processing the call keyword arguments (line 1186)
        kwargs_358784 = {}
        # Getting the type of 'getattr' (line 1186)
        getattr_358781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 25), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1186)
        getattr_call_result_358785 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 25), getattr_358781, *[self_358782, func_name_358783], **kwargs_358784)
        
        # Calling (args, kwargs) (line 1186)
        _call_result_358787 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 25), getattr_call_result_358785, *[], **kwargs_358786)
        
        # Assigning a type to the variable 'result' (line 1186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 16), 'result', _call_result_358787)
        # SSA branch for the else part of an if statement (line 1185)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1188):
        
        # Assigning a Call to a Name (line 1188):
        
        # Call to (...): (line 1188)
        # Processing the call keyword arguments (line 1188)
        kwargs_358796 = {}
        
        # Call to getattr(...): (line 1188)
        # Processing the call arguments (line 1188)
        
        # Call to tocsr(...): (line 1188)
        # Processing the call keyword arguments (line 1188)
        kwargs_358791 = {}
        # Getting the type of 'self' (line 1188)
        self_358789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 33), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 1188)
        tocsr_358790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 33), self_358789, 'tocsr')
        # Calling tocsr(args, kwargs) (line 1188)
        tocsr_call_result_358792 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 33), tocsr_358790, *[], **kwargs_358791)
        
        # Getting the type of 'func_name' (line 1188)
        func_name_358793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 47), 'func_name', False)
        # Processing the call keyword arguments (line 1188)
        kwargs_358794 = {}
        # Getting the type of 'getattr' (line 1188)
        getattr_358788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 25), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1188)
        getattr_call_result_358795 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 25), getattr_358788, *[tocsr_call_result_358792, func_name_358793], **kwargs_358794)
        
        # Calling (args, kwargs) (line 1188)
        _call_result_358797 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 25), getattr_call_result_358795, *[], **kwargs_358796)
        
        # Assigning a type to the variable 'result' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 16), 'result', _call_result_358797)
        # SSA join for if statement (line 1185)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1183)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 1190)
        NotImplemented_358798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 1190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 12), 'stypy_return_type', NotImplemented_358798)
        # SSA join for if statement (line 1183)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1181)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1179)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1177)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1172)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1166)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1161)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1156)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1154)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1152)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 1192)
        # Getting the type of 'out' (line 1192)
        out_358799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 8), 'out')
        # Getting the type of 'None' (line 1192)
        None_358800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 22), 'None')
        
        (may_be_358801, more_types_in_union_358802) = may_not_be_none(out_358799, None_358800)

        if may_be_358801:

            if more_types_in_union_358802:
                # Runtime conditional SSA (line 1192)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Evaluating a boolean operation
            
            
            # Call to isinstance(...): (line 1193)
            # Processing the call arguments (line 1193)
            # Getting the type of 'out' (line 1193)
            out_358804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 30), 'out', False)
            # Getting the type of 'spmatrix' (line 1193)
            spmatrix_358805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 35), 'spmatrix', False)
            # Processing the call keyword arguments (line 1193)
            kwargs_358806 = {}
            # Getting the type of 'isinstance' (line 1193)
            isinstance_358803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 1193)
            isinstance_call_result_358807 = invoke(stypy.reporting.localization.Localization(__file__, 1193, 19), isinstance_358803, *[out_358804, spmatrix_358805], **kwargs_358806)
            
            # Applying the 'not' unary operator (line 1193)
            result_not__358808 = python_operator(stypy.reporting.localization.Localization(__file__, 1193, 15), 'not', isinstance_call_result_358807)
            
            
            # Call to isinstance(...): (line 1193)
            # Processing the call arguments (line 1193)
            # Getting the type of 'result' (line 1193)
            result_358810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 60), 'result', False)
            # Getting the type of 'spmatrix' (line 1193)
            spmatrix_358811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 68), 'spmatrix', False)
            # Processing the call keyword arguments (line 1193)
            kwargs_358812 = {}
            # Getting the type of 'isinstance' (line 1193)
            isinstance_358809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 49), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 1193)
            isinstance_call_result_358813 = invoke(stypy.reporting.localization.Localization(__file__, 1193, 49), isinstance_358809, *[result_358810, spmatrix_358811], **kwargs_358812)
            
            # Applying the binary operator 'and' (line 1193)
            result_and_keyword_358814 = python_operator(stypy.reporting.localization.Localization(__file__, 1193, 15), 'and', result_not__358808, isinstance_call_result_358813)
            
            # Testing the type of an if condition (line 1193)
            if_condition_358815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1193, 12), result_and_keyword_358814)
            # Assigning a type to the variable 'if_condition_358815' (line 1193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1193, 12), 'if_condition_358815', if_condition_358815)
            # SSA begins for if statement (line 1193)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 1194):
            
            # Assigning a Call to a Subscript (line 1194):
            
            # Call to todense(...): (line 1194)
            # Processing the call keyword arguments (line 1194)
            kwargs_358818 = {}
            # Getting the type of 'result' (line 1194)
            result_358816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 27), 'result', False)
            # Obtaining the member 'todense' of a type (line 1194)
            todense_358817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1194, 27), result_358816, 'todense')
            # Calling todense(args, kwargs) (line 1194)
            todense_call_result_358819 = invoke(stypy.reporting.localization.Localization(__file__, 1194, 27), todense_358817, *[], **kwargs_358818)
            
            # Getting the type of 'out' (line 1194)
            out_358820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 16), 'out')
            Ellipsis_358821 = Ellipsis
            # Storing an element on a container (line 1194)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1194, 16), out_358820, (Ellipsis_358821, todense_call_result_358819))
            # SSA branch for the else part of an if statement (line 1193)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Subscript (line 1196):
            
            # Assigning a Name to a Subscript (line 1196):
            # Getting the type of 'result' (line 1196)
            result_358822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 27), 'result')
            # Getting the type of 'out' (line 1196)
            out_358823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 16), 'out')
            Ellipsis_358824 = Ellipsis
            # Storing an element on a container (line 1196)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1196, 16), out_358823, (Ellipsis_358824, result_358822))
            # SSA join for if statement (line 1193)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 1197):
            
            # Assigning a Name to a Name (line 1197):
            # Getting the type of 'out' (line 1197)
            out_358825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 21), 'out')
            # Assigning a type to the variable 'result' (line 1197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 12), 'result', out_358825)

            if more_types_in_union_358802:
                # SSA join for if statement (line 1192)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'result' (line 1199)
        result_358826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 1199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1199, 8), 'stypy_return_type', result_358826)
        
        # ################# End of '__numpy_ufunc__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__numpy_ufunc__' in the type store
        # Getting the type of 'stypy_return_type' (line 1132)
        stypy_return_type_358827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_358827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__numpy_ufunc__'
        return stypy_return_type_358827


# Assigning a type to the variable 'spmatrix' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'spmatrix', spmatrix)

# Assigning a Num to a Name (line 68):
float_358828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'float')
# Getting the type of 'spmatrix'
spmatrix_358829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'spmatrix')
# Setting the type of the member '__array_priority__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), spmatrix_358829, '__array_priority__', float_358828)

# Assigning a Num to a Name (line 69):
int_358830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'int')
# Getting the type of 'spmatrix'
spmatrix_358831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'spmatrix')
# Setting the type of the member 'ndim' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), spmatrix_358831, 'ndim', int_358830)

# Assigning a Call to a Name (line 105):

# Call to property(...): (line 105)
# Processing the call keyword arguments (line 105)
# Getting the type of 'get_shape' (line 105)
get_shape_358833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'get_shape', False)
keyword_358834 = get_shape_358833
# Getting the type of 'set_shape' (line 105)
set_shape_358835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'set_shape', False)
keyword_358836 = set_shape_358835
kwargs_358837 = {'fset': keyword_358836, 'fget': keyword_358834}
# Getting the type of 'property' (line 105)
property_358832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'property', False)
# Calling property(args, kwargs) (line 105)
property_call_result_358838 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), property_358832, *[], **kwargs_358837)

# Getting the type of 'spmatrix'
spmatrix_358839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'spmatrix')
# Setting the type of the member 'shape' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), spmatrix_358839, 'shape', property_call_result_358838)

# Assigning a Name to a Name (line 260):
# Getting the type of 'spmatrix'
spmatrix_358840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'spmatrix')
# Obtaining the member '__bool__' of a type
bool___358841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), spmatrix_358840, '__bool__')
# Getting the type of 'spmatrix'
spmatrix_358842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'spmatrix')
# Setting the type of the member '__nonzero__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), spmatrix_358842, '__nonzero__', bool___358841)

# Assigning a Attribute to a Attribute (line 687):
# Getting the type of 'spmatrix'
spmatrix_358843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'spmatrix')
# Obtaining the member 'conj' of a type
conj_358844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), spmatrix_358843, 'conj')
# Obtaining the member '__doc__' of a type (line 687)
doc___358845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 24), conj_358844, '__doc__')
# Getting the type of 'spmatrix'
spmatrix_358846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'spmatrix')
# Obtaining the member 'conjugate' of a type
conjugate_358847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), spmatrix_358846, 'conjugate')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), conjugate_358847, '__doc__', doc___358845)

@norecursion
def isspmatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix'
    module_type_store = module_type_store.open_function_context('isspmatrix', 1202, 0, False)
    
    # Passed parameters checking function
    isspmatrix.stypy_localization = localization
    isspmatrix.stypy_type_of_self = None
    isspmatrix.stypy_type_store = module_type_store
    isspmatrix.stypy_function_name = 'isspmatrix'
    isspmatrix.stypy_param_names_list = ['x']
    isspmatrix.stypy_varargs_param_name = None
    isspmatrix.stypy_kwargs_param_name = None
    isspmatrix.stypy_call_defaults = defaults
    isspmatrix.stypy_call_varargs = varargs
    isspmatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix(...)' code ##################

    str_358848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1228, (-1)), 'str', 'Is x of a sparse matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a sparse matrix\n\n    Returns\n    -------\n    bool\n        True if x is a sparse matrix, False otherwise\n\n    Notes\n    -----\n    issparse and isspmatrix are aliases for the same function.\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, isspmatrix\n    >>> isspmatrix(csr_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import isspmatrix\n    >>> isspmatrix(5)\n    False\n    ')
    
    # Call to isinstance(...): (line 1229)
    # Processing the call arguments (line 1229)
    # Getting the type of 'x' (line 1229)
    x_358850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 22), 'x', False)
    # Getting the type of 'spmatrix' (line 1229)
    spmatrix_358851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 25), 'spmatrix', False)
    # Processing the call keyword arguments (line 1229)
    kwargs_358852 = {}
    # Getting the type of 'isinstance' (line 1229)
    isinstance_358849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1229)
    isinstance_call_result_358853 = invoke(stypy.reporting.localization.Localization(__file__, 1229, 11), isinstance_358849, *[x_358850, spmatrix_358851], **kwargs_358852)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 4), 'stypy_return_type', isinstance_call_result_358853)
    
    # ################# End of 'isspmatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 1202)
    stypy_return_type_358854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_358854)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix'
    return stypy_return_type_358854

# Assigning a type to the variable 'isspmatrix' (line 1202)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1202, 0), 'isspmatrix', isspmatrix)

# Assigning a Name to a Name (line 1231):

# Assigning a Name to a Name (line 1231):
# Getting the type of 'isspmatrix' (line 1231)
isspmatrix_358855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 11), 'isspmatrix')
# Assigning a type to the variable 'issparse' (line 1231)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 0), 'issparse', isspmatrix_358855)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
