
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2:   Matrix Market I/O in Python.
3:   See http://math.nist.gov/MatrixMarket/formats.html
4:   for information about the Matrix Market format.
5: '''
6: #
7: # Author: Pearu Peterson <pearu@cens.ioc.ee>
8: # Created: October, 2004
9: #
10: # References:
11: #  http://math.nist.gov/MatrixMarket/
12: #
13: from __future__ import division, print_function, absolute_import
14: 
15: import os
16: import sys
17: 
18: from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
19:                    ones, ascontiguousarray, vstack, savetxt, fromfile,
20:                    fromstring, can_cast)
21: from numpy.compat import asbytes, asstr
22: 
23: from scipy._lib.six import string_types
24: from scipy.sparse import coo_matrix, isspmatrix
25: 
26: __all__ = ['mminfo', 'mmread', 'mmwrite', 'MMFile']
27: 
28: 
29: # -----------------------------------------------------------------------------
30: def mminfo(source):
31:     '''
32:     Return size and storage parameters from Matrix Market file-like 'source'.
33: 
34:     Parameters
35:     ----------
36:     source : str or file-like
37:         Matrix Market filename (extension .mtx) or open file-like object
38: 
39:     Returns
40:     -------
41:     rows : int
42:         Number of matrix rows.
43:     cols : int
44:         Number of matrix columns.
45:     entries : int
46:         Number of non-zero entries of a sparse matrix
47:         or rows*cols for a dense matrix.
48:     format : str
49:         Either 'coordinate' or 'array'.
50:     field : str
51:         Either 'real', 'complex', 'pattern', or 'integer'.
52:     symmetry : str
53:         Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
54:     '''
55:     return MMFile.info(source)
56: 
57: # -----------------------------------------------------------------------------
58: 
59: 
60: def mmread(source):
61:     '''
62:     Reads the contents of a Matrix Market file-like 'source' into a matrix.
63: 
64:     Parameters
65:     ----------
66:     source : str or file-like
67:         Matrix Market filename (extensions .mtx, .mtz.gz)
68:         or open file-like object.
69: 
70:     Returns
71:     -------
72:     a : ndarray or coo_matrix
73:         Dense or sparse matrix depending on the matrix format in the
74:         Matrix Market file.
75:     '''
76:     return MMFile().read(source)
77: 
78: # -----------------------------------------------------------------------------
79: 
80: 
81: def mmwrite(target, a, comment='', field=None, precision=None, symmetry=None):
82:     '''
83:     Writes the sparse or dense array `a` to Matrix Market file-like `target`.
84: 
85:     Parameters
86:     ----------
87:     target : str or file-like
88:         Matrix Market filename (extension .mtx) or open file-like object.
89:     a : array like
90:         Sparse or dense 2D array.
91:     comment : str, optional
92:         Comments to be prepended to the Matrix Market file.
93:     field : None or str, optional
94:         Either 'real', 'complex', 'pattern', or 'integer'.
95:     precision : None or int, optional
96:         Number of digits to display for real or complex values.
97:     symmetry : None or str, optional
98:         Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
99:         If symmetry is None the symmetry type of 'a' is determined by its
100:         values.
101:     '''
102:     MMFile().write(target, a, comment, field, precision, symmetry)
103: 
104: 
105: ###############################################################################
106: class MMFile (object):
107:     __slots__ = ('_rows',
108:                  '_cols',
109:                  '_entries',
110:                  '_format',
111:                  '_field',
112:                  '_symmetry')
113: 
114:     @property
115:     def rows(self):
116:         return self._rows
117: 
118:     @property
119:     def cols(self):
120:         return self._cols
121: 
122:     @property
123:     def entries(self):
124:         return self._entries
125: 
126:     @property
127:     def format(self):
128:         return self._format
129: 
130:     @property
131:     def field(self):
132:         return self._field
133: 
134:     @property
135:     def symmetry(self):
136:         return self._symmetry
137: 
138:     @property
139:     def has_symmetry(self):
140:         return self._symmetry in (self.SYMMETRY_SYMMETRIC,
141:                                   self.SYMMETRY_SKEW_SYMMETRIC,
142:                                   self.SYMMETRY_HERMITIAN)
143: 
144:     # format values
145:     FORMAT_COORDINATE = 'coordinate'
146:     FORMAT_ARRAY = 'array'
147:     FORMAT_VALUES = (FORMAT_COORDINATE, FORMAT_ARRAY)
148: 
149:     @classmethod
150:     def _validate_format(self, format):
151:         if format not in self.FORMAT_VALUES:
152:             raise ValueError('unknown format type %s, must be one of %s' %
153:                              (format, self.FORMAT_VALUES))
154: 
155:     # field values
156:     FIELD_INTEGER = 'integer'
157:     FIELD_REAL = 'real'
158:     FIELD_COMPLEX = 'complex'
159:     FIELD_PATTERN = 'pattern'
160:     FIELD_VALUES = (FIELD_INTEGER, FIELD_REAL, FIELD_COMPLEX, FIELD_PATTERN)
161: 
162:     @classmethod
163:     def _validate_field(self, field):
164:         if field not in self.FIELD_VALUES:
165:             raise ValueError('unknown field type %s, must be one of %s' %
166:                              (field, self.FIELD_VALUES))
167: 
168:     # symmetry values
169:     SYMMETRY_GENERAL = 'general'
170:     SYMMETRY_SYMMETRIC = 'symmetric'
171:     SYMMETRY_SKEW_SYMMETRIC = 'skew-symmetric'
172:     SYMMETRY_HERMITIAN = 'hermitian'
173:     SYMMETRY_VALUES = (SYMMETRY_GENERAL, SYMMETRY_SYMMETRIC,
174:                        SYMMETRY_SKEW_SYMMETRIC, SYMMETRY_HERMITIAN)
175: 
176:     @classmethod
177:     def _validate_symmetry(self, symmetry):
178:         if symmetry not in self.SYMMETRY_VALUES:
179:             raise ValueError('unknown symmetry type %s, must be one of %s' %
180:                              (symmetry, self.SYMMETRY_VALUES))
181: 
182:     DTYPES_BY_FIELD = {FIELD_INTEGER: 'intp',
183:                        FIELD_REAL: 'd',
184:                        FIELD_COMPLEX: 'D',
185:                        FIELD_PATTERN: 'd'}
186: 
187:     # -------------------------------------------------------------------------
188:     @staticmethod
189:     def reader():
190:         pass
191: 
192:     # -------------------------------------------------------------------------
193:     @staticmethod
194:     def writer():
195:         pass
196: 
197:     # -------------------------------------------------------------------------
198:     @classmethod
199:     def info(self, source):
200:         '''
201:         Return size, storage parameters from Matrix Market file-like 'source'.
202: 
203:         Parameters
204:         ----------
205:         source : str or file-like
206:             Matrix Market filename (extension .mtx) or open file-like object
207: 
208:         Returns
209:         -------
210:         rows : int
211:             Number of matrix rows.
212:         cols : int
213:             Number of matrix columns.
214:         entries : int
215:             Number of non-zero entries of a sparse matrix
216:             or rows*cols for a dense matrix.
217:         format : str
218:             Either 'coordinate' or 'array'.
219:         field : str
220:             Either 'real', 'complex', 'pattern', or 'integer'.
221:         symmetry : str
222:             Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
223:         '''
224: 
225:         stream, close_it = self._open(source)
226: 
227:         try:
228: 
229:             # read and validate header line
230:             line = stream.readline()
231:             mmid, matrix, format, field, symmetry = \
232:                 [asstr(part.strip()) for part in line.split()]
233:             if not mmid.startswith('%%MatrixMarket'):
234:                 raise ValueError('source is not in Matrix Market format')
235:             if not matrix.lower() == 'matrix':
236:                 raise ValueError("Problem reading file header: " + line)
237: 
238:             # http://math.nist.gov/MatrixMarket/formats.html
239:             if format.lower() == 'array':
240:                 format = self.FORMAT_ARRAY
241:             elif format.lower() == 'coordinate':
242:                 format = self.FORMAT_COORDINATE
243: 
244:             # skip comments
245:             while line.startswith(b'%'):
246:                 line = stream.readline()
247: 
248:             line = line.split()
249:             if format == self.FORMAT_ARRAY:
250:                 if not len(line) == 2:
251:                     raise ValueError("Header line not of length 2: " + line)
252:                 rows, cols = map(int, line)
253:                 entries = rows * cols
254:             else:
255:                 if not len(line) == 3:
256:                     raise ValueError("Header line not of length 3: " + line)
257:                 rows, cols, entries = map(int, line)
258: 
259:             return (rows, cols, entries, format, field.lower(),
260:                     symmetry.lower())
261: 
262:         finally:
263:             if close_it:
264:                 stream.close()
265: 
266:     # -------------------------------------------------------------------------
267:     @staticmethod
268:     def _open(filespec, mode='rb'):
269:         ''' Return an open file stream for reading based on source.
270: 
271:         If source is a file name, open it (after trying to find it with mtx and
272:         gzipped mtx extensions).  Otherwise, just return source.
273: 
274:         Parameters
275:         ----------
276:         filespec : str or file-like
277:             String giving file name or file-like object
278:         mode : str, optional
279:             Mode with which to open file, if `filespec` is a file name.
280: 
281:         Returns
282:         -------
283:         fobj : file-like
284:             Open file-like object.
285:         close_it : bool
286:             True if the calling function should close this file when done,
287:             false otherwise.
288:         '''
289:         close_it = False
290:         if isinstance(filespec, string_types):
291:             close_it = True
292: 
293:             # open for reading
294:             if mode[0] == 'r':
295: 
296:                 # determine filename plus extension
297:                 if not os.path.isfile(filespec):
298:                     if os.path.isfile(filespec+'.mtx'):
299:                         filespec = filespec + '.mtx'
300:                     elif os.path.isfile(filespec+'.mtx.gz'):
301:                         filespec = filespec + '.mtx.gz'
302:                     elif os.path.isfile(filespec+'.mtx.bz2'):
303:                         filespec = filespec + '.mtx.bz2'
304:                 # open filename
305:                 if filespec.endswith('.gz'):
306:                     import gzip
307:                     stream = gzip.open(filespec, mode)
308:                 elif filespec.endswith('.bz2'):
309:                     import bz2
310:                     stream = bz2.BZ2File(filespec, 'rb')
311:                 else:
312:                     stream = open(filespec, mode)
313: 
314:             # open for writing
315:             else:
316:                 if filespec[-4:] != '.mtx':
317:                     filespec = filespec + '.mtx'
318:                 stream = open(filespec, mode)
319:         else:
320:             stream = filespec
321: 
322:         return stream, close_it
323: 
324:     # -------------------------------------------------------------------------
325:     @staticmethod
326:     def _get_symmetry(a):
327:         m, n = a.shape
328:         if m != n:
329:             return MMFile.SYMMETRY_GENERAL
330:         issymm = True
331:         isskew = True
332:         isherm = a.dtype.char in 'FD'
333: 
334:         # sparse input
335:         if isspmatrix(a):
336:             # check if number of nonzero entries of lower and upper triangle
337:             # matrix are equal
338:             a = a.tocoo()
339:             (row, col) = a.nonzero()
340:             if (row < col).sum() != (row > col).sum():
341:                 return MMFile.SYMMETRY_GENERAL
342: 
343:             # define iterator over symmetric pair entries
344:             a = a.todok()
345: 
346:             def symm_iterator():
347:                 for ((i, j), aij) in a.items():
348:                     if i > j:
349:                         aji = a[j, i]
350:                         yield (aij, aji)
351: 
352:         # non-sparse input
353:         else:
354:             # define iterator over symmetric pair entries
355:             def symm_iterator():
356:                 for j in range(n):
357:                     for i in range(j+1, n):
358:                         aij, aji = a[i][j], a[j][i]
359:                         yield (aij, aji)
360: 
361:         # check for symmetry
362:         for (aij, aji) in symm_iterator():
363:             if issymm and aij != aji:
364:                 issymm = False
365:             if isskew and aij != -aji:
366:                 isskew = False
367:             if isherm and aij != conj(aji):
368:                 isherm = False
369:             if not (issymm or isskew or isherm):
370:                 break
371: 
372:         # return symmetry value
373:         if issymm:
374:             return MMFile.SYMMETRY_SYMMETRIC
375:         if isskew:
376:             return MMFile.SYMMETRY_SKEW_SYMMETRIC
377:         if isherm:
378:             return MMFile.SYMMETRY_HERMITIAN
379:         return MMFile.SYMMETRY_GENERAL
380: 
381:     # -------------------------------------------------------------------------
382:     @staticmethod
383:     def _field_template(field, precision):
384:         return {MMFile.FIELD_REAL: '%%.%ie\n' % precision,
385:                 MMFile.FIELD_INTEGER: '%i\n',
386:                 MMFile.FIELD_COMPLEX: '%%.%ie %%.%ie\n' %
387:                     (precision, precision)
388:                 }.get(field, None)
389: 
390:     # -------------------------------------------------------------------------
391:     def __init__(self, **kwargs):
392:         self._init_attrs(**kwargs)
393: 
394:     # -------------------------------------------------------------------------
395:     def read(self, source):
396:         '''
397:         Reads the contents of a Matrix Market file-like 'source' into a matrix.
398: 
399:         Parameters
400:         ----------
401:         source : str or file-like
402:             Matrix Market filename (extensions .mtx, .mtz.gz)
403:             or open file object.
404: 
405:         Returns
406:         -------
407:         a : ndarray or coo_matrix
408:             Dense or sparse matrix depending on the matrix format in the
409:             Matrix Market file.
410:         '''
411:         stream, close_it = self._open(source)
412: 
413:         try:
414:             self._parse_header(stream)
415:             return self._parse_body(stream)
416: 
417:         finally:
418:             if close_it:
419:                 stream.close()
420: 
421:     # -------------------------------------------------------------------------
422:     def write(self, target, a, comment='', field=None, precision=None,
423:               symmetry=None):
424:         '''
425:         Writes sparse or dense array `a` to Matrix Market file-like `target`.
426: 
427:         Parameters
428:         ----------
429:         target : str or file-like
430:             Matrix Market filename (extension .mtx) or open file-like object.
431:         a : array like
432:             Sparse or dense 2D array.
433:         comment : str, optional
434:             Comments to be prepended to the Matrix Market file.
435:         field : None or str, optional
436:             Either 'real', 'complex', 'pattern', or 'integer'.
437:         precision : None or int, optional
438:             Number of digits to display for real or complex values.
439:         symmetry : None or str, optional
440:             Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
441:             If symmetry is None the symmetry type of 'a' is determined by its
442:             values.
443:         '''
444: 
445:         stream, close_it = self._open(target, 'wb')
446: 
447:         try:
448:             self._write(stream, a, comment, field, precision, symmetry)
449: 
450:         finally:
451:             if close_it:
452:                 stream.close()
453:             else:
454:                 stream.flush()
455: 
456:     # -------------------------------------------------------------------------
457:     def _init_attrs(self, **kwargs):
458:         '''
459:         Initialize each attributes with the corresponding keyword arg value
460:         or a default of None
461:         '''
462: 
463:         attrs = self.__class__.__slots__
464:         public_attrs = [attr[1:] for attr in attrs]
465:         invalid_keys = set(kwargs.keys()) - set(public_attrs)
466: 
467:         if invalid_keys:
468:             raise ValueError('''found %s invalid keyword arguments, please only
469:                                 use %s''' % (tuple(invalid_keys),
470:                                              public_attrs))
471: 
472:         for attr in attrs:
473:             setattr(self, attr, kwargs.get(attr[1:], None))
474: 
475:     # -------------------------------------------------------------------------
476:     def _parse_header(self, stream):
477:         rows, cols, entries, format, field, symmetry = \
478:             self.__class__.info(stream)
479:         self._init_attrs(rows=rows, cols=cols, entries=entries, format=format,
480:                          field=field, symmetry=symmetry)
481: 
482:     # -------------------------------------------------------------------------
483:     def _parse_body(self, stream):
484:         rows, cols, entries, format, field, symm = (self.rows, self.cols,
485:                                                     self.entries, self.format,
486:                                                     self.field, self.symmetry)
487: 
488:         try:
489:             from scipy.sparse import coo_matrix
490:         except ImportError:
491:             coo_matrix = None
492: 
493:         dtype = self.DTYPES_BY_FIELD.get(field, None)
494: 
495:         has_symmetry = self.has_symmetry
496:         is_integer = field == self.FIELD_INTEGER
497:         is_complex = field == self.FIELD_COMPLEX
498:         is_skew = symm == self.SYMMETRY_SKEW_SYMMETRIC
499:         is_herm = symm == self.SYMMETRY_HERMITIAN
500:         is_pattern = field == self.FIELD_PATTERN
501: 
502:         if format == self.FORMAT_ARRAY:
503:             a = zeros((rows, cols), dtype=dtype)
504:             line = 1
505:             i, j = 0, 0
506:             while line:
507:                 line = stream.readline()
508:                 if not line or line.startswith(b'%'):
509:                     continue
510:                 if is_integer:
511:                     aij = int(line)
512:                 elif is_complex:
513:                     aij = complex(*map(float, line.split()))
514:                 else:
515:                     aij = float(line)
516:                 a[i, j] = aij
517:                 if has_symmetry and i != j:
518:                     if is_skew:
519:                         a[j, i] = -aij
520:                     elif is_herm:
521:                         a[j, i] = conj(aij)
522:                     else:
523:                         a[j, i] = aij
524:                 if i < rows-1:
525:                     i = i + 1
526:                 else:
527:                     j = j + 1
528:                     if not has_symmetry:
529:                         i = 0
530:                     else:
531:                         i = j
532:             if not (i in [0, j] and j == cols):
533:                 raise ValueError("Parse error, did not read all lines.")
534: 
535:         elif format == self.FORMAT_COORDINATE and coo_matrix is None:
536:             # Read sparse matrix to dense when coo_matrix is not available.
537:             a = zeros((rows, cols), dtype=dtype)
538:             line = 1
539:             k = 0
540:             while line:
541:                 line = stream.readline()
542:                 if not line or line.startswith(b'%'):
543:                     continue
544:                 l = line.split()
545:                 i, j = map(int, l[:2])
546:                 i, j = i-1, j-1
547:                 if is_integer:
548:                     aij = int(l[2])
549:                 elif is_complex:
550:                     aij = complex(*map(float, l[2:]))
551:                 else:
552:                     aij = float(l[2])
553:                 a[i, j] = aij
554:                 if has_symmetry and i != j:
555:                     if is_skew:
556:                         a[j, i] = -aij
557:                     elif is_herm:
558:                         a[j, i] = conj(aij)
559:                     else:
560:                         a[j, i] = aij
561:                 k = k + 1
562:             if not k == entries:
563:                 ValueError("Did not read all entries")
564: 
565:         elif format == self.FORMAT_COORDINATE:
566:             # Read sparse COOrdinate format
567: 
568:             if entries == 0:
569:                 # empty matrix
570:                 return coo_matrix((rows, cols), dtype=dtype)
571: 
572:             I = zeros(entries, dtype='intc')
573:             J = zeros(entries, dtype='intc')
574:             if is_pattern:
575:                 V = ones(entries, dtype='int8')
576:             elif is_integer:
577:                 V = zeros(entries, dtype='intp')
578:             elif is_complex:
579:                 V = zeros(entries, dtype='complex')
580:             else:
581:                 V = zeros(entries, dtype='float')
582: 
583:             entry_number = 0
584:             for line in stream:
585:                 if not line or line.startswith(b'%'):
586:                     continue
587: 
588:                 if entry_number+1 > entries:
589:                     raise ValueError("'entries' in header is smaller than "
590:                                      "number of entries")
591:                 l = line.split()
592:                 I[entry_number], J[entry_number] = map(int, l[:2])
593: 
594:                 if not is_pattern:
595:                     if is_integer:
596:                         V[entry_number] = int(l[2])
597:                     elif is_complex:
598:                         V[entry_number] = complex(*map(float, l[2:]))
599:                     else:
600:                         V[entry_number] = float(l[2])
601:                 entry_number += 1
602:             if entry_number < entries:
603:                 raise ValueError("'entries' in header is larger than "
604:                                  "number of entries")
605: 
606:             I -= 1  # adjust indices (base 1 -> base 0)
607:             J -= 1
608: 
609:             if has_symmetry:
610:                 mask = (I != J)       # off diagonal mask
611:                 od_I = I[mask]
612:                 od_J = J[mask]
613:                 od_V = V[mask]
614: 
615:                 I = concatenate((I, od_J))
616:                 J = concatenate((J, od_I))
617: 
618:                 if is_skew:
619:                     od_V *= -1
620:                 elif is_herm:
621:                     od_V = od_V.conjugate()
622: 
623:                 V = concatenate((V, od_V))
624: 
625:             a = coo_matrix((V, (I, J)), shape=(rows, cols), dtype=dtype)
626:         else:
627:             raise NotImplementedError(format)
628: 
629:         return a
630: 
631:     #  ------------------------------------------------------------------------
632:     def _write(self, stream, a, comment='', field=None, precision=None,
633:                symmetry=None):
634: 
635:         if isinstance(a, list) or isinstance(a, ndarray) or \
636:            isinstance(a, tuple) or hasattr(a, '__array__'):
637:             rep = self.FORMAT_ARRAY
638:             a = asarray(a)
639:             if len(a.shape) != 2:
640:                 raise ValueError('Expected 2 dimensional array')
641:             rows, cols = a.shape
642: 
643:             if field is not None:
644: 
645:                 if field == self.FIELD_INTEGER:
646:                     if not can_cast(a.dtype, 'intp'):
647:                         raise OverflowError("mmwrite does not support integer "
648:                                             "dtypes larger than native 'intp'.")
649:                     a = a.astype('intp')
650:                 elif field == self.FIELD_REAL:
651:                     if a.dtype.char not in 'fd':
652:                         a = a.astype('d')
653:                 elif field == self.FIELD_COMPLEX:
654:                     if a.dtype.char not in 'FD':
655:                         a = a.astype('D')
656: 
657:         else:
658:             if not isspmatrix(a):
659:                 raise ValueError('unknown matrix type: %s' % type(a))
660:             rep = 'coordinate'
661:             rows, cols = a.shape
662: 
663:         typecode = a.dtype.char
664: 
665:         if precision is None:
666:             if typecode in 'fF':
667:                 precision = 8
668:             else:
669:                 precision = 16
670: 
671:         if field is None:
672:             kind = a.dtype.kind
673:             if kind == 'i':
674:                 if not can_cast(a.dtype, 'intp'):
675:                     raise OverflowError("mmwrite does not support integer "
676:                                         "dtypes larger than native 'intp'.")
677:                 field = 'integer'
678:             elif kind == 'f':
679:                 field = 'real'
680:             elif kind == 'c':
681:                 field = 'complex'
682:             else:
683:                 raise TypeError('unexpected dtype kind ' + kind)
684: 
685:         if symmetry is None:
686:             symmetry = self._get_symmetry(a)
687: 
688:         # validate rep, field, and symmetry
689:         self.__class__._validate_format(rep)
690:         self.__class__._validate_field(field)
691:         self.__class__._validate_symmetry(symmetry)
692: 
693:         # write initial header line
694:         stream.write(asbytes('%%MatrixMarket matrix {0} {1} {2}\n'.format(rep,
695:             field, symmetry)))
696: 
697:         # write comments
698:         for line in comment.split('\n'):
699:             stream.write(asbytes('%%%s\n' % (line)))
700: 
701:         template = self._field_template(field, precision)
702: 
703:         # write dense format
704:         if rep == self.FORMAT_ARRAY:
705: 
706:             # write shape spec
707:             stream.write(asbytes('%i %i\n' % (rows, cols)))
708: 
709:             if field in (self.FIELD_INTEGER, self.FIELD_REAL):
710: 
711:                 if symmetry == self.SYMMETRY_GENERAL:
712:                     for j in range(cols):
713:                         for i in range(rows):
714:                             stream.write(asbytes(template % a[i, j]))
715:                 else:
716:                     for j in range(cols):
717:                         for i in range(j, rows):
718:                             stream.write(asbytes(template % a[i, j]))
719: 
720:             elif field == self.FIELD_COMPLEX:
721: 
722:                 if symmetry == self.SYMMETRY_GENERAL:
723:                     for j in range(cols):
724:                         for i in range(rows):
725:                             aij = a[i, j]
726:                             stream.write(asbytes(template % (real(aij),
727:                                                              imag(aij))))
728:                 else:
729:                     for j in range(cols):
730:                         for i in range(j, rows):
731:                             aij = a[i, j]
732:                             stream.write(asbytes(template % (real(aij),
733:                                                              imag(aij))))
734: 
735:             elif field == self.FIELD_PATTERN:
736:                 raise ValueError('pattern type inconsisted with dense format')
737: 
738:             else:
739:                 raise TypeError('Unknown field type %s' % field)
740: 
741:         # write sparse format
742:         else:
743: 
744:             coo = a.tocoo()  # convert to COOrdinate format
745: 
746:             # if symmetry format used, remove values above main diagonal
747:             if symmetry != self.SYMMETRY_GENERAL:
748:                 lower_triangle_mask = coo.row >= coo.col
749:                 coo = coo_matrix((coo.data[lower_triangle_mask],
750:                                  (coo.row[lower_triangle_mask],
751:                                   coo.col[lower_triangle_mask])),
752:                                  shape=coo.shape)
753: 
754:             # write shape spec
755:             stream.write(asbytes('%i %i %i\n' % (rows, cols, coo.nnz)))
756: 
757:             template = self._field_template(field, precision-1)
758: 
759:             if field == self.FIELD_PATTERN:
760:                 for r, c in zip(coo.row+1, coo.col+1):
761:                     stream.write(asbytes("%i %i\n" % (r, c)))
762:             elif field in (self.FIELD_INTEGER, self.FIELD_REAL):
763:                 for r, c, d in zip(coo.row+1, coo.col+1, coo.data):
764:                     stream.write(asbytes(("%i %i " % (r, c)) +
765:                                          (template % d)))
766:             elif field == self.FIELD_COMPLEX:
767:                 for r, c, d in zip(coo.row+1, coo.col+1, coo.data):
768:                     stream.write(asbytes(("%i %i " % (r, c)) +
769:                                          (template % (d.real, d.imag))))
770:             else:
771:                 raise TypeError('Unknown field type %s' % field)
772: 
773: 
774: def _is_fromfile_compatible(stream):
775:     '''
776:     Check whether `stream` is compatible with numpy.fromfile.
777: 
778:     Passing a gzipped file object to ``fromfile/fromstring`` doesn't work with
779:     Python3.
780:     '''
781:     if sys.version_info[0] < 3:
782:         return True
783: 
784:     bad_cls = []
785:     try:
786:         import gzip
787:         bad_cls.append(gzip.GzipFile)
788:     except ImportError:
789:         pass
790:     try:
791:         import bz2
792:         bad_cls.append(bz2.BZ2File)
793:     except ImportError:
794:         pass
795: 
796:     bad_cls = tuple(bad_cls)
797:     return not isinstance(stream, bad_cls)
798: 
799: 
800: # -----------------------------------------------------------------------------
801: if __name__ == '__main__':
802:     import time
803:     for filename in sys.argv[1:]:
804:         print('Reading', filename, '...', end=' ')
805:         sys.stdout.flush()
806:         t = time.time()
807:         mmread(filename)
808:         print('took %s seconds' % (time.time() - t))
809: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_122325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\n  Matrix Market I/O in Python.\n  See http://math.nist.gov/MatrixMarket/formats.html\n  for information about the Matrix Market format.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import os' statement (line 15)
import os

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import sys' statement (line 16)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy import asarray, real, imag, conj, zeros, ndarray, concatenate, ones, ascontiguousarray, vstack, savetxt, fromfile, fromstring, can_cast' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_122326 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy')

if (type(import_122326) is not StypyTypeError):

    if (import_122326 != 'pyd_module'):
        __import__(import_122326)
        sys_modules_122327 = sys.modules[import_122326]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy', sys_modules_122327.module_type_store, module_type_store, ['asarray', 'real', 'imag', 'conj', 'zeros', 'ndarray', 'concatenate', 'ones', 'ascontiguousarray', 'vstack', 'savetxt', 'fromfile', 'fromstring', 'can_cast'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_122327, sys_modules_122327.module_type_store, module_type_store)
    else:
        from numpy import asarray, real, imag, conj, zeros, ndarray, concatenate, ones, ascontiguousarray, vstack, savetxt, fromfile, fromstring, can_cast

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy', None, module_type_store, ['asarray', 'real', 'imag', 'conj', 'zeros', 'ndarray', 'concatenate', 'ones', 'ascontiguousarray', 'vstack', 'savetxt', 'fromfile', 'fromstring', 'can_cast'], [asarray, real, imag, conj, zeros, ndarray, concatenate, ones, ascontiguousarray, vstack, savetxt, fromfile, fromstring, can_cast])

else:
    # Assigning a type to the variable 'numpy' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy', import_122326)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.compat import asbytes, asstr' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_122328 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.compat')

if (type(import_122328) is not StypyTypeError):

    if (import_122328 != 'pyd_module'):
        __import__(import_122328)
        sys_modules_122329 = sys.modules[import_122328]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.compat', sys_modules_122329.module_type_store, module_type_store, ['asbytes', 'asstr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_122329, sys_modules_122329.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, asstr

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'asstr'], [asbytes, asstr])

else:
    # Assigning a type to the variable 'numpy.compat' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.compat', import_122328)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from scipy._lib.six import string_types' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_122330 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib.six')

if (type(import_122330) is not StypyTypeError):

    if (import_122330 != 'pyd_module'):
        __import__(import_122330)
        sys_modules_122331 = sys.modules[import_122330]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib.six', sys_modules_122331.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_122331, sys_modules_122331.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'scipy._lib.six', import_122330)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy.sparse import coo_matrix, isspmatrix' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_122332 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse')

if (type(import_122332) is not StypyTypeError):

    if (import_122332 != 'pyd_module'):
        __import__(import_122332)
        sys_modules_122333 = sys.modules[import_122332]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse', sys_modules_122333.module_type_store, module_type_store, ['coo_matrix', 'isspmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_122333, sys_modules_122333.module_type_store, module_type_store)
    else:
        from scipy.sparse import coo_matrix, isspmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse', None, module_type_store, ['coo_matrix', 'isspmatrix'], [coo_matrix, isspmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.sparse', import_122332)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')


# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):
__all__ = ['mminfo', 'mmread', 'mmwrite', 'MMFile']
module_type_store.set_exportable_members(['mminfo', 'mmread', 'mmwrite', 'MMFile'])

# Obtaining an instance of the builtin type 'list' (line 26)
list_122334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_122335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', 'mminfo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122334, str_122335)
# Adding element type (line 26)
str_122336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'str', 'mmread')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122334, str_122336)
# Adding element type (line 26)
str_122337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'str', 'mmwrite')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122334, str_122337)
# Adding element type (line 26)
str_122338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'str', 'MMFile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_122334, str_122338)

# Assigning a type to the variable '__all__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__all__', list_122334)

@norecursion
def mminfo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mminfo'
    module_type_store = module_type_store.open_function_context('mminfo', 30, 0, False)
    
    # Passed parameters checking function
    mminfo.stypy_localization = localization
    mminfo.stypy_type_of_self = None
    mminfo.stypy_type_store = module_type_store
    mminfo.stypy_function_name = 'mminfo'
    mminfo.stypy_param_names_list = ['source']
    mminfo.stypy_varargs_param_name = None
    mminfo.stypy_kwargs_param_name = None
    mminfo.stypy_call_defaults = defaults
    mminfo.stypy_call_varargs = varargs
    mminfo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mminfo', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mminfo', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mminfo(...)' code ##################

    str_122339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', "\n    Return size and storage parameters from Matrix Market file-like 'source'.\n\n    Parameters\n    ----------\n    source : str or file-like\n        Matrix Market filename (extension .mtx) or open file-like object\n\n    Returns\n    -------\n    rows : int\n        Number of matrix rows.\n    cols : int\n        Number of matrix columns.\n    entries : int\n        Number of non-zero entries of a sparse matrix\n        or rows*cols for a dense matrix.\n    format : str\n        Either 'coordinate' or 'array'.\n    field : str\n        Either 'real', 'complex', 'pattern', or 'integer'.\n    symmetry : str\n        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n    ")
    
    # Call to info(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'source' (line 55)
    source_122342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'source', False)
    # Processing the call keyword arguments (line 55)
    kwargs_122343 = {}
    # Getting the type of 'MMFile' (line 55)
    MMFile_122340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'MMFile', False)
    # Obtaining the member 'info' of a type (line 55)
    info_122341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), MMFile_122340, 'info')
    # Calling info(args, kwargs) (line 55)
    info_call_result_122344 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), info_122341, *[source_122342], **kwargs_122343)
    
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type', info_call_result_122344)
    
    # ################# End of 'mminfo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mminfo' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_122345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122345)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mminfo'
    return stypy_return_type_122345

# Assigning a type to the variable 'mminfo' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'mminfo', mminfo)

@norecursion
def mmread(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mmread'
    module_type_store = module_type_store.open_function_context('mmread', 60, 0, False)
    
    # Passed parameters checking function
    mmread.stypy_localization = localization
    mmread.stypy_type_of_self = None
    mmread.stypy_type_store = module_type_store
    mmread.stypy_function_name = 'mmread'
    mmread.stypy_param_names_list = ['source']
    mmread.stypy_varargs_param_name = None
    mmread.stypy_kwargs_param_name = None
    mmread.stypy_call_defaults = defaults
    mmread.stypy_call_varargs = varargs
    mmread.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mmread', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mmread', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mmread(...)' code ##################

    str_122346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', "\n    Reads the contents of a Matrix Market file-like 'source' into a matrix.\n\n    Parameters\n    ----------\n    source : str or file-like\n        Matrix Market filename (extensions .mtx, .mtz.gz)\n        or open file-like object.\n\n    Returns\n    -------\n    a : ndarray or coo_matrix\n        Dense or sparse matrix depending on the matrix format in the\n        Matrix Market file.\n    ")
    
    # Call to read(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'source' (line 76)
    source_122351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'source', False)
    # Processing the call keyword arguments (line 76)
    kwargs_122352 = {}
    
    # Call to MMFile(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_122348 = {}
    # Getting the type of 'MMFile' (line 76)
    MMFile_122347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'MMFile', False)
    # Calling MMFile(args, kwargs) (line 76)
    MMFile_call_result_122349 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), MMFile_122347, *[], **kwargs_122348)
    
    # Obtaining the member 'read' of a type (line 76)
    read_122350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), MMFile_call_result_122349, 'read')
    # Calling read(args, kwargs) (line 76)
    read_call_result_122353 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), read_122350, *[source_122351], **kwargs_122352)
    
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type', read_call_result_122353)
    
    # ################# End of 'mmread(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mmread' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_122354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122354)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mmread'
    return stypy_return_type_122354

# Assigning a type to the variable 'mmread' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'mmread', mmread)

@norecursion
def mmwrite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_122355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'str', '')
    # Getting the type of 'None' (line 81)
    None_122356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 41), 'None')
    # Getting the type of 'None' (line 81)
    None_122357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 57), 'None')
    # Getting the type of 'None' (line 81)
    None_122358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 72), 'None')
    defaults = [str_122355, None_122356, None_122357, None_122358]
    # Create a new context for function 'mmwrite'
    module_type_store = module_type_store.open_function_context('mmwrite', 81, 0, False)
    
    # Passed parameters checking function
    mmwrite.stypy_localization = localization
    mmwrite.stypy_type_of_self = None
    mmwrite.stypy_type_store = module_type_store
    mmwrite.stypy_function_name = 'mmwrite'
    mmwrite.stypy_param_names_list = ['target', 'a', 'comment', 'field', 'precision', 'symmetry']
    mmwrite.stypy_varargs_param_name = None
    mmwrite.stypy_kwargs_param_name = None
    mmwrite.stypy_call_defaults = defaults
    mmwrite.stypy_call_varargs = varargs
    mmwrite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mmwrite', ['target', 'a', 'comment', 'field', 'precision', 'symmetry'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mmwrite', localization, ['target', 'a', 'comment', 'field', 'precision', 'symmetry'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mmwrite(...)' code ##################

    str_122359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, (-1)), 'str', "\n    Writes the sparse or dense array `a` to Matrix Market file-like `target`.\n\n    Parameters\n    ----------\n    target : str or file-like\n        Matrix Market filename (extension .mtx) or open file-like object.\n    a : array like\n        Sparse or dense 2D array.\n    comment : str, optional\n        Comments to be prepended to the Matrix Market file.\n    field : None or str, optional\n        Either 'real', 'complex', 'pattern', or 'integer'.\n    precision : None or int, optional\n        Number of digits to display for real or complex values.\n    symmetry : None or str, optional\n        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n        If symmetry is None the symmetry type of 'a' is determined by its\n        values.\n    ")
    
    # Call to write(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'target' (line 102)
    target_122364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'target', False)
    # Getting the type of 'a' (line 102)
    a_122365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'a', False)
    # Getting the type of 'comment' (line 102)
    comment_122366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'comment', False)
    # Getting the type of 'field' (line 102)
    field_122367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'field', False)
    # Getting the type of 'precision' (line 102)
    precision_122368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 46), 'precision', False)
    # Getting the type of 'symmetry' (line 102)
    symmetry_122369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 57), 'symmetry', False)
    # Processing the call keyword arguments (line 102)
    kwargs_122370 = {}
    
    # Call to MMFile(...): (line 102)
    # Processing the call keyword arguments (line 102)
    kwargs_122361 = {}
    # Getting the type of 'MMFile' (line 102)
    MMFile_122360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'MMFile', False)
    # Calling MMFile(args, kwargs) (line 102)
    MMFile_call_result_122362 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), MMFile_122360, *[], **kwargs_122361)
    
    # Obtaining the member 'write' of a type (line 102)
    write_122363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 4), MMFile_call_result_122362, 'write')
    # Calling write(args, kwargs) (line 102)
    write_call_result_122371 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), write_122363, *[target_122364, a_122365, comment_122366, field_122367, precision_122368, symmetry_122369], **kwargs_122370)
    
    
    # ################# End of 'mmwrite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mmwrite' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_122372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mmwrite'
    return stypy_return_type_122372

# Assigning a type to the variable 'mmwrite' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'mmwrite', mmwrite)
# Declaration of the 'MMFile' class

class MMFile(object, ):
    
    # Assigning a Tuple to a Name (line 107):

    @norecursion
    def rows(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'rows'
        module_type_store = module_type_store.open_function_context('rows', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.rows.__dict__.__setitem__('stypy_localization', localization)
        MMFile.rows.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.rows.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.rows.__dict__.__setitem__('stypy_function_name', 'MMFile.rows')
        MMFile.rows.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.rows.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.rows.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.rows.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.rows.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.rows.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.rows.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.rows', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'rows', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'rows(...)' code ##################

        # Getting the type of 'self' (line 116)
        self_122373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'self')
        # Obtaining the member '_rows' of a type (line 116)
        _rows_122374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), self_122373, '_rows')
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', _rows_122374)
        
        # ################# End of 'rows(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'rows' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_122375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'rows'
        return stypy_return_type_122375


    @norecursion
    def cols(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cols'
        module_type_store = module_type_store.open_function_context('cols', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.cols.__dict__.__setitem__('stypy_localization', localization)
        MMFile.cols.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.cols.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.cols.__dict__.__setitem__('stypy_function_name', 'MMFile.cols')
        MMFile.cols.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.cols.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.cols.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.cols.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.cols.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.cols.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.cols.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.cols', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cols', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cols(...)' code ##################

        # Getting the type of 'self' (line 120)
        self_122376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'self')
        # Obtaining the member '_cols' of a type (line 120)
        _cols_122377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), self_122376, '_cols')
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', _cols_122377)
        
        # ################# End of 'cols(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cols' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_122378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122378)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cols'
        return stypy_return_type_122378


    @norecursion
    def entries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'entries'
        module_type_store = module_type_store.open_function_context('entries', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.entries.__dict__.__setitem__('stypy_localization', localization)
        MMFile.entries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.entries.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.entries.__dict__.__setitem__('stypy_function_name', 'MMFile.entries')
        MMFile.entries.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.entries.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.entries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.entries.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.entries.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.entries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.entries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.entries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'entries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'entries(...)' code ##################

        # Getting the type of 'self' (line 124)
        self_122379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'self')
        # Obtaining the member '_entries' of a type (line 124)
        _entries_122380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), self_122379, '_entries')
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', _entries_122380)
        
        # ################# End of 'entries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'entries' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_122381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'entries'
        return stypy_return_type_122381


    @norecursion
    def format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format'
        module_type_store = module_type_store.open_function_context('format', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.format.__dict__.__setitem__('stypy_localization', localization)
        MMFile.format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.format.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.format.__dict__.__setitem__('stypy_function_name', 'MMFile.format')
        MMFile.format.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.format.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.format.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.format.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.format.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.format', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format(...)' code ##################

        # Getting the type of 'self' (line 128)
        self_122382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'self')
        # Obtaining the member '_format' of a type (line 128)
        _format_122383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), self_122382, '_format')
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'stypy_return_type', _format_122383)
        
        # ################# End of 'format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_122384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122384)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format'
        return stypy_return_type_122384


    @norecursion
    def field(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'field'
        module_type_store = module_type_store.open_function_context('field', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.field.__dict__.__setitem__('stypy_localization', localization)
        MMFile.field.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.field.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.field.__dict__.__setitem__('stypy_function_name', 'MMFile.field')
        MMFile.field.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.field.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.field.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.field.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.field.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.field.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.field.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.field', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'field', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'field(...)' code ##################

        # Getting the type of 'self' (line 132)
        self_122385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'self')
        # Obtaining the member '_field' of a type (line 132)
        _field_122386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), self_122385, '_field')
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', _field_122386)
        
        # ################# End of 'field(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'field' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_122387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'field'
        return stypy_return_type_122387


    @norecursion
    def symmetry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'symmetry'
        module_type_store = module_type_store.open_function_context('symmetry', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.symmetry.__dict__.__setitem__('stypy_localization', localization)
        MMFile.symmetry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.symmetry.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.symmetry.__dict__.__setitem__('stypy_function_name', 'MMFile.symmetry')
        MMFile.symmetry.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.symmetry.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.symmetry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.symmetry.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.symmetry.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.symmetry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.symmetry.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.symmetry', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'symmetry', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'symmetry(...)' code ##################

        # Getting the type of 'self' (line 136)
        self_122388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'self')
        # Obtaining the member '_symmetry' of a type (line 136)
        _symmetry_122389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 15), self_122388, '_symmetry')
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type', _symmetry_122389)
        
        # ################# End of 'symmetry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'symmetry' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_122390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122390)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'symmetry'
        return stypy_return_type_122390


    @norecursion
    def has_symmetry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_symmetry'
        module_type_store = module_type_store.open_function_context('has_symmetry', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.has_symmetry.__dict__.__setitem__('stypy_localization', localization)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_function_name', 'MMFile.has_symmetry')
        MMFile.has_symmetry.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.has_symmetry.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.has_symmetry.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.has_symmetry', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_symmetry', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_symmetry(...)' code ##################

        
        # Getting the type of 'self' (line 140)
        self_122391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'self')
        # Obtaining the member '_symmetry' of a type (line 140)
        _symmetry_122392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 15), self_122391, '_symmetry')
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_122393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        # Getting the type of 'self' (line 140)
        self_122394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 34), 'self')
        # Obtaining the member 'SYMMETRY_SYMMETRIC' of a type (line 140)
        SYMMETRY_SYMMETRIC_122395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 34), self_122394, 'SYMMETRY_SYMMETRIC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 34), tuple_122393, SYMMETRY_SYMMETRIC_122395)
        # Adding element type (line 140)
        # Getting the type of 'self' (line 141)
        self_122396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'self')
        # Obtaining the member 'SYMMETRY_SKEW_SYMMETRIC' of a type (line 141)
        SYMMETRY_SKEW_SYMMETRIC_122397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 34), self_122396, 'SYMMETRY_SKEW_SYMMETRIC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 34), tuple_122393, SYMMETRY_SKEW_SYMMETRIC_122397)
        # Adding element type (line 140)
        # Getting the type of 'self' (line 142)
        self_122398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'self')
        # Obtaining the member 'SYMMETRY_HERMITIAN' of a type (line 142)
        SYMMETRY_HERMITIAN_122399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 34), self_122398, 'SYMMETRY_HERMITIAN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 34), tuple_122393, SYMMETRY_HERMITIAN_122399)
        
        # Applying the binary operator 'in' (line 140)
        result_contains_122400 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 15), 'in', _symmetry_122392, tuple_122393)
        
        # Assigning a type to the variable 'stypy_return_type' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', result_contains_122400)
        
        # ################# End of 'has_symmetry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_symmetry' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_122401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122401)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_symmetry'
        return stypy_return_type_122401

    
    # Assigning a Str to a Name (line 145):
    
    # Assigning a Str to a Name (line 146):
    
    # Assigning a Tuple to a Name (line 147):

    @norecursion
    def _validate_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_validate_format'
        module_type_store = module_type_store.open_function_context('_validate_format', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile._validate_format.__dict__.__setitem__('stypy_localization', localization)
        MMFile._validate_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile._validate_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._validate_format.__dict__.__setitem__('stypy_function_name', 'MMFile._validate_format')
        MMFile._validate_format.__dict__.__setitem__('stypy_param_names_list', ['format'])
        MMFile._validate_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._validate_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._validate_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._validate_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._validate_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._validate_format.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile._validate_format', ['format'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_validate_format', localization, ['format'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_validate_format(...)' code ##################

        
        
        # Getting the type of 'format' (line 151)
        format_122402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'format')
        # Getting the type of 'self' (line 151)
        self_122403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'self')
        # Obtaining the member 'FORMAT_VALUES' of a type (line 151)
        FORMAT_VALUES_122404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), self_122403, 'FORMAT_VALUES')
        # Applying the binary operator 'notin' (line 151)
        result_contains_122405 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), 'notin', format_122402, FORMAT_VALUES_122404)
        
        # Testing the type of an if condition (line 151)
        if_condition_122406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_contains_122405)
        # Assigning a type to the variable 'if_condition_122406' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_122406', if_condition_122406)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 152)
        # Processing the call arguments (line 152)
        str_122408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 29), 'str', 'unknown format type %s, must be one of %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_122409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        # Getting the type of 'format' (line 153)
        format_122410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'format', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 30), tuple_122409, format_122410)
        # Adding element type (line 153)
        # Getting the type of 'self' (line 153)
        self_122411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'self', False)
        # Obtaining the member 'FORMAT_VALUES' of a type (line 153)
        FORMAT_VALUES_122412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 38), self_122411, 'FORMAT_VALUES')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 30), tuple_122409, FORMAT_VALUES_122412)
        
        # Applying the binary operator '%' (line 152)
        result_mod_122413 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 29), '%', str_122408, tuple_122409)
        
        # Processing the call keyword arguments (line 152)
        kwargs_122414 = {}
        # Getting the type of 'ValueError' (line 152)
        ValueError_122407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 152)
        ValueError_call_result_122415 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), ValueError_122407, *[result_mod_122413], **kwargs_122414)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 152, 12), ValueError_call_result_122415, 'raise parameter', BaseException)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_validate_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_validate_format' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_122416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122416)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_validate_format'
        return stypy_return_type_122416

    
    # Assigning a Str to a Name (line 156):
    
    # Assigning a Str to a Name (line 157):
    
    # Assigning a Str to a Name (line 158):
    
    # Assigning a Str to a Name (line 159):
    
    # Assigning a Tuple to a Name (line 160):

    @norecursion
    def _validate_field(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_validate_field'
        module_type_store = module_type_store.open_function_context('_validate_field', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile._validate_field.__dict__.__setitem__('stypy_localization', localization)
        MMFile._validate_field.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile._validate_field.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._validate_field.__dict__.__setitem__('stypy_function_name', 'MMFile._validate_field')
        MMFile._validate_field.__dict__.__setitem__('stypy_param_names_list', ['field'])
        MMFile._validate_field.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._validate_field.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._validate_field.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._validate_field.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._validate_field.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._validate_field.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile._validate_field', ['field'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_validate_field', localization, ['field'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_validate_field(...)' code ##################

        
        
        # Getting the type of 'field' (line 164)
        field_122417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'field')
        # Getting the type of 'self' (line 164)
        self_122418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'self')
        # Obtaining the member 'FIELD_VALUES' of a type (line 164)
        FIELD_VALUES_122419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 24), self_122418, 'FIELD_VALUES')
        # Applying the binary operator 'notin' (line 164)
        result_contains_122420 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 11), 'notin', field_122417, FIELD_VALUES_122419)
        
        # Testing the type of an if condition (line 164)
        if_condition_122421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), result_contains_122420)
        # Assigning a type to the variable 'if_condition_122421' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_122421', if_condition_122421)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 165)
        # Processing the call arguments (line 165)
        str_122423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 29), 'str', 'unknown field type %s, must be one of %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 166)
        tuple_122424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 166)
        # Adding element type (line 166)
        # Getting the type of 'field' (line 166)
        field_122425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'field', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 30), tuple_122424, field_122425)
        # Adding element type (line 166)
        # Getting the type of 'self' (line 166)
        self_122426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'self', False)
        # Obtaining the member 'FIELD_VALUES' of a type (line 166)
        FIELD_VALUES_122427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 37), self_122426, 'FIELD_VALUES')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 30), tuple_122424, FIELD_VALUES_122427)
        
        # Applying the binary operator '%' (line 165)
        result_mod_122428 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 29), '%', str_122423, tuple_122424)
        
        # Processing the call keyword arguments (line 165)
        kwargs_122429 = {}
        # Getting the type of 'ValueError' (line 165)
        ValueError_122422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 165)
        ValueError_call_result_122430 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), ValueError_122422, *[result_mod_122428], **kwargs_122429)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 165, 12), ValueError_call_result_122430, 'raise parameter', BaseException)
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_validate_field(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_validate_field' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_122431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_validate_field'
        return stypy_return_type_122431

    
    # Assigning a Str to a Name (line 169):
    
    # Assigning a Str to a Name (line 170):
    
    # Assigning a Str to a Name (line 171):
    
    # Assigning a Str to a Name (line 172):
    
    # Assigning a Tuple to a Name (line 173):

    @norecursion
    def _validate_symmetry(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_validate_symmetry'
        module_type_store = module_type_store.open_function_context('_validate_symmetry', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_localization', localization)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_function_name', 'MMFile._validate_symmetry')
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_param_names_list', ['symmetry'])
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._validate_symmetry.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile._validate_symmetry', ['symmetry'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_validate_symmetry', localization, ['symmetry'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_validate_symmetry(...)' code ##################

        
        
        # Getting the type of 'symmetry' (line 178)
        symmetry_122432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'symmetry')
        # Getting the type of 'self' (line 178)
        self_122433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'self')
        # Obtaining the member 'SYMMETRY_VALUES' of a type (line 178)
        SYMMETRY_VALUES_122434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 27), self_122433, 'SYMMETRY_VALUES')
        # Applying the binary operator 'notin' (line 178)
        result_contains_122435 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 11), 'notin', symmetry_122432, SYMMETRY_VALUES_122434)
        
        # Testing the type of an if condition (line 178)
        if_condition_122436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 8), result_contains_122435)
        # Assigning a type to the variable 'if_condition_122436' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'if_condition_122436', if_condition_122436)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 179)
        # Processing the call arguments (line 179)
        str_122438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'str', 'unknown symmetry type %s, must be one of %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_122439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        # Getting the type of 'symmetry' (line 180)
        symmetry_122440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'symmetry', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 30), tuple_122439, symmetry_122440)
        # Adding element type (line 180)
        # Getting the type of 'self' (line 180)
        self_122441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 40), 'self', False)
        # Obtaining the member 'SYMMETRY_VALUES' of a type (line 180)
        SYMMETRY_VALUES_122442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 40), self_122441, 'SYMMETRY_VALUES')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 30), tuple_122439, SYMMETRY_VALUES_122442)
        
        # Applying the binary operator '%' (line 179)
        result_mod_122443 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 29), '%', str_122438, tuple_122439)
        
        # Processing the call keyword arguments (line 179)
        kwargs_122444 = {}
        # Getting the type of 'ValueError' (line 179)
        ValueError_122437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 179)
        ValueError_call_result_122445 = invoke(stypy.reporting.localization.Localization(__file__, 179, 18), ValueError_122437, *[result_mod_122443], **kwargs_122444)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 179, 12), ValueError_call_result_122445, 'raise parameter', BaseException)
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_validate_symmetry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_validate_symmetry' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_122446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122446)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_validate_symmetry'
        return stypy_return_type_122446

    
    # Assigning a Dict to a Name (line 182):

    @staticmethod
    @norecursion
    def reader(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reader'
        module_type_store = module_type_store.open_function_context('reader', 188, 4, False)
        
        # Passed parameters checking function
        MMFile.reader.__dict__.__setitem__('stypy_localization', localization)
        MMFile.reader.__dict__.__setitem__('stypy_type_of_self', None)
        MMFile.reader.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.reader.__dict__.__setitem__('stypy_function_name', 'reader')
        MMFile.reader.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.reader.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.reader.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.reader.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.reader.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.reader.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.reader.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'reader', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reader', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reader(...)' code ##################

        pass
        
        # ################# End of 'reader(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reader' in the type store
        # Getting the type of 'stypy_return_type' (line 188)
        stypy_return_type_122447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122447)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reader'
        return stypy_return_type_122447


    @staticmethod
    @norecursion
    def writer(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'writer'
        module_type_store = module_type_store.open_function_context('writer', 193, 4, False)
        
        # Passed parameters checking function
        MMFile.writer.__dict__.__setitem__('stypy_localization', localization)
        MMFile.writer.__dict__.__setitem__('stypy_type_of_self', None)
        MMFile.writer.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.writer.__dict__.__setitem__('stypy_function_name', 'writer')
        MMFile.writer.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile.writer.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.writer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.writer.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.writer.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.writer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.writer.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'writer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'writer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'writer(...)' code ##################

        pass
        
        # ################# End of 'writer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'writer' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_122448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122448)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'writer'
        return stypy_return_type_122448


    @norecursion
    def info(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'info'
        module_type_store = module_type_store.open_function_context('info', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.info.__dict__.__setitem__('stypy_localization', localization)
        MMFile.info.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.info.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.info.__dict__.__setitem__('stypy_function_name', 'MMFile.info')
        MMFile.info.__dict__.__setitem__('stypy_param_names_list', ['source'])
        MMFile.info.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.info.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.info.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.info.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.info.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.info.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.info', ['source'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'info', localization, ['source'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'info(...)' code ##################

        str_122449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'str', "\n        Return size, storage parameters from Matrix Market file-like 'source'.\n\n        Parameters\n        ----------\n        source : str or file-like\n            Matrix Market filename (extension .mtx) or open file-like object\n\n        Returns\n        -------\n        rows : int\n            Number of matrix rows.\n        cols : int\n            Number of matrix columns.\n        entries : int\n            Number of non-zero entries of a sparse matrix\n            or rows*cols for a dense matrix.\n        format : str\n            Either 'coordinate' or 'array'.\n        field : str\n            Either 'real', 'complex', 'pattern', or 'integer'.\n        symmetry : str\n            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n        ")
        
        # Assigning a Call to a Tuple (line 225):
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_122450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        
        # Call to _open(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'source' (line 225)
        source_122453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 38), 'source', False)
        # Processing the call keyword arguments (line 225)
        kwargs_122454 = {}
        # Getting the type of 'self' (line 225)
        self_122451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'self', False)
        # Obtaining the member '_open' of a type (line 225)
        _open_122452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 27), self_122451, '_open')
        # Calling _open(args, kwargs) (line 225)
        _open_call_result_122455 = invoke(stypy.reporting.localization.Localization(__file__, 225, 27), _open_122452, *[source_122453], **kwargs_122454)
        
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___122456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), _open_call_result_122455, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_122457 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___122456, int_122450)
        
        # Assigning a type to the variable 'tuple_var_assignment_122279' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_122279', subscript_call_result_122457)
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_122458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        
        # Call to _open(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'source' (line 225)
        source_122461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 38), 'source', False)
        # Processing the call keyword arguments (line 225)
        kwargs_122462 = {}
        # Getting the type of 'self' (line 225)
        self_122459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'self', False)
        # Obtaining the member '_open' of a type (line 225)
        _open_122460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 27), self_122459, '_open')
        # Calling _open(args, kwargs) (line 225)
        _open_call_result_122463 = invoke(stypy.reporting.localization.Localization(__file__, 225, 27), _open_122460, *[source_122461], **kwargs_122462)
        
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___122464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), _open_call_result_122463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_122465 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___122464, int_122458)
        
        # Assigning a type to the variable 'tuple_var_assignment_122280' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_122280', subscript_call_result_122465)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_122279' (line 225)
        tuple_var_assignment_122279_122466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_122279')
        # Assigning a type to the variable 'stream' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'stream', tuple_var_assignment_122279_122466)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_122280' (line 225)
        tuple_var_assignment_122280_122467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_122280')
        # Assigning a type to the variable 'close_it' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'close_it', tuple_var_assignment_122280_122467)
        
        # Try-finally block (line 227)
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to readline(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_122470 = {}
        # Getting the type of 'stream' (line 230)
        stream_122468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'stream', False)
        # Obtaining the member 'readline' of a type (line 230)
        readline_122469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 19), stream_122468, 'readline')
        # Calling readline(args, kwargs) (line 230)
        readline_call_result_122471 = invoke(stypy.reporting.localization.Localization(__file__, 230, 19), readline_122469, *[], **kwargs_122470)
        
        # Assigning a type to the variable 'line' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'line', readline_call_result_122471)
        
        # Assigning a ListComp to a Tuple (line 231):
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_122472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122482 = {}
        # Getting the type of 'line' (line 232)
        line_122480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'line', False)
        # Obtaining the member 'split' of a type (line 232)
        split_122481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 49), line_122480, 'split')
        # Calling split(args, kwargs) (line 232)
        split_call_result_122483 = invoke(stypy.reporting.localization.Localization(__file__, 232, 49), split_122481, *[], **kwargs_122482)
        
        comprehension_122484 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), split_call_result_122483)
        # Assigning a type to the variable 'part' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'part', comprehension_122484)
        
        # Call to asstr(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to strip(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122476 = {}
        # Getting the type of 'part' (line 232)
        part_122474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'part', False)
        # Obtaining the member 'strip' of a type (line 232)
        strip_122475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), part_122474, 'strip')
        # Calling strip(args, kwargs) (line 232)
        strip_call_result_122477 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), strip_122475, *[], **kwargs_122476)
        
        # Processing the call keyword arguments (line 232)
        kwargs_122478 = {}
        # Getting the type of 'asstr' (line 232)
        asstr_122473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'asstr', False)
        # Calling asstr(args, kwargs) (line 232)
        asstr_call_result_122479 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), asstr_122473, *[strip_call_result_122477], **kwargs_122478)
        
        list_122485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), list_122485, asstr_call_result_122479)
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___122486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), list_122485, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_122487 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___122486, int_122472)
        
        # Assigning a type to the variable 'tuple_var_assignment_122281' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122281', subscript_call_result_122487)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_122488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122498 = {}
        # Getting the type of 'line' (line 232)
        line_122496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'line', False)
        # Obtaining the member 'split' of a type (line 232)
        split_122497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 49), line_122496, 'split')
        # Calling split(args, kwargs) (line 232)
        split_call_result_122499 = invoke(stypy.reporting.localization.Localization(__file__, 232, 49), split_122497, *[], **kwargs_122498)
        
        comprehension_122500 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), split_call_result_122499)
        # Assigning a type to the variable 'part' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'part', comprehension_122500)
        
        # Call to asstr(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to strip(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122492 = {}
        # Getting the type of 'part' (line 232)
        part_122490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'part', False)
        # Obtaining the member 'strip' of a type (line 232)
        strip_122491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), part_122490, 'strip')
        # Calling strip(args, kwargs) (line 232)
        strip_call_result_122493 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), strip_122491, *[], **kwargs_122492)
        
        # Processing the call keyword arguments (line 232)
        kwargs_122494 = {}
        # Getting the type of 'asstr' (line 232)
        asstr_122489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'asstr', False)
        # Calling asstr(args, kwargs) (line 232)
        asstr_call_result_122495 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), asstr_122489, *[strip_call_result_122493], **kwargs_122494)
        
        list_122501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), list_122501, asstr_call_result_122495)
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___122502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), list_122501, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_122503 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___122502, int_122488)
        
        # Assigning a type to the variable 'tuple_var_assignment_122282' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122282', subscript_call_result_122503)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_122504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122514 = {}
        # Getting the type of 'line' (line 232)
        line_122512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'line', False)
        # Obtaining the member 'split' of a type (line 232)
        split_122513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 49), line_122512, 'split')
        # Calling split(args, kwargs) (line 232)
        split_call_result_122515 = invoke(stypy.reporting.localization.Localization(__file__, 232, 49), split_122513, *[], **kwargs_122514)
        
        comprehension_122516 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), split_call_result_122515)
        # Assigning a type to the variable 'part' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'part', comprehension_122516)
        
        # Call to asstr(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to strip(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122508 = {}
        # Getting the type of 'part' (line 232)
        part_122506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'part', False)
        # Obtaining the member 'strip' of a type (line 232)
        strip_122507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), part_122506, 'strip')
        # Calling strip(args, kwargs) (line 232)
        strip_call_result_122509 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), strip_122507, *[], **kwargs_122508)
        
        # Processing the call keyword arguments (line 232)
        kwargs_122510 = {}
        # Getting the type of 'asstr' (line 232)
        asstr_122505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'asstr', False)
        # Calling asstr(args, kwargs) (line 232)
        asstr_call_result_122511 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), asstr_122505, *[strip_call_result_122509], **kwargs_122510)
        
        list_122517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), list_122517, asstr_call_result_122511)
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___122518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), list_122517, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_122519 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___122518, int_122504)
        
        # Assigning a type to the variable 'tuple_var_assignment_122283' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122283', subscript_call_result_122519)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_122520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122530 = {}
        # Getting the type of 'line' (line 232)
        line_122528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'line', False)
        # Obtaining the member 'split' of a type (line 232)
        split_122529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 49), line_122528, 'split')
        # Calling split(args, kwargs) (line 232)
        split_call_result_122531 = invoke(stypy.reporting.localization.Localization(__file__, 232, 49), split_122529, *[], **kwargs_122530)
        
        comprehension_122532 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), split_call_result_122531)
        # Assigning a type to the variable 'part' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'part', comprehension_122532)
        
        # Call to asstr(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to strip(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122524 = {}
        # Getting the type of 'part' (line 232)
        part_122522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'part', False)
        # Obtaining the member 'strip' of a type (line 232)
        strip_122523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), part_122522, 'strip')
        # Calling strip(args, kwargs) (line 232)
        strip_call_result_122525 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), strip_122523, *[], **kwargs_122524)
        
        # Processing the call keyword arguments (line 232)
        kwargs_122526 = {}
        # Getting the type of 'asstr' (line 232)
        asstr_122521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'asstr', False)
        # Calling asstr(args, kwargs) (line 232)
        asstr_call_result_122527 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), asstr_122521, *[strip_call_result_122525], **kwargs_122526)
        
        list_122533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), list_122533, asstr_call_result_122527)
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___122534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), list_122533, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_122535 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___122534, int_122520)
        
        # Assigning a type to the variable 'tuple_var_assignment_122284' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122284', subscript_call_result_122535)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_122536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122546 = {}
        # Getting the type of 'line' (line 232)
        line_122544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'line', False)
        # Obtaining the member 'split' of a type (line 232)
        split_122545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 49), line_122544, 'split')
        # Calling split(args, kwargs) (line 232)
        split_call_result_122547 = invoke(stypy.reporting.localization.Localization(__file__, 232, 49), split_122545, *[], **kwargs_122546)
        
        comprehension_122548 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), split_call_result_122547)
        # Assigning a type to the variable 'part' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'part', comprehension_122548)
        
        # Call to asstr(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to strip(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_122540 = {}
        # Getting the type of 'part' (line 232)
        part_122538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'part', False)
        # Obtaining the member 'strip' of a type (line 232)
        strip_122539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), part_122538, 'strip')
        # Calling strip(args, kwargs) (line 232)
        strip_call_result_122541 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), strip_122539, *[], **kwargs_122540)
        
        # Processing the call keyword arguments (line 232)
        kwargs_122542 = {}
        # Getting the type of 'asstr' (line 232)
        asstr_122537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'asstr', False)
        # Calling asstr(args, kwargs) (line 232)
        asstr_call_result_122543 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), asstr_122537, *[strip_call_result_122541], **kwargs_122542)
        
        list_122549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 17), list_122549, asstr_call_result_122543)
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___122550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), list_122549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_122551 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___122550, int_122536)
        
        # Assigning a type to the variable 'tuple_var_assignment_122285' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122285', subscript_call_result_122551)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_122281' (line 231)
        tuple_var_assignment_122281_122552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122281')
        # Assigning a type to the variable 'mmid' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'mmid', tuple_var_assignment_122281_122552)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_122282' (line 231)
        tuple_var_assignment_122282_122553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122282')
        # Assigning a type to the variable 'matrix' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'matrix', tuple_var_assignment_122282_122553)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_122283' (line 231)
        tuple_var_assignment_122283_122554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122283')
        # Assigning a type to the variable 'format' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'format', tuple_var_assignment_122283_122554)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_122284' (line 231)
        tuple_var_assignment_122284_122555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122284')
        # Assigning a type to the variable 'field' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'field', tuple_var_assignment_122284_122555)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_122285' (line 231)
        tuple_var_assignment_122285_122556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple_var_assignment_122285')
        # Assigning a type to the variable 'symmetry' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 41), 'symmetry', tuple_var_assignment_122285_122556)
        
        
        
        # Call to startswith(...): (line 233)
        # Processing the call arguments (line 233)
        str_122559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 35), 'str', '%%MatrixMarket')
        # Processing the call keyword arguments (line 233)
        kwargs_122560 = {}
        # Getting the type of 'mmid' (line 233)
        mmid_122557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'mmid', False)
        # Obtaining the member 'startswith' of a type (line 233)
        startswith_122558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 19), mmid_122557, 'startswith')
        # Calling startswith(args, kwargs) (line 233)
        startswith_call_result_122561 = invoke(stypy.reporting.localization.Localization(__file__, 233, 19), startswith_122558, *[str_122559], **kwargs_122560)
        
        # Applying the 'not' unary operator (line 233)
        result_not__122562 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), 'not', startswith_call_result_122561)
        
        # Testing the type of an if condition (line 233)
        if_condition_122563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 12), result_not__122562)
        # Assigning a type to the variable 'if_condition_122563' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'if_condition_122563', if_condition_122563)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 234)
        # Processing the call arguments (line 234)
        str_122565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 33), 'str', 'source is not in Matrix Market format')
        # Processing the call keyword arguments (line 234)
        kwargs_122566 = {}
        # Getting the type of 'ValueError' (line 234)
        ValueError_122564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 234)
        ValueError_call_result_122567 = invoke(stypy.reporting.localization.Localization(__file__, 234, 22), ValueError_122564, *[str_122565], **kwargs_122566)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 234, 16), ValueError_call_result_122567, 'raise parameter', BaseException)
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        
        # Call to lower(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_122570 = {}
        # Getting the type of 'matrix' (line 235)
        matrix_122568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'matrix', False)
        # Obtaining the member 'lower' of a type (line 235)
        lower_122569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 19), matrix_122568, 'lower')
        # Calling lower(args, kwargs) (line 235)
        lower_call_result_122571 = invoke(stypy.reporting.localization.Localization(__file__, 235, 19), lower_122569, *[], **kwargs_122570)
        
        str_122572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 37), 'str', 'matrix')
        # Applying the binary operator '==' (line 235)
        result_eq_122573 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 19), '==', lower_call_result_122571, str_122572)
        
        # Applying the 'not' unary operator (line 235)
        result_not__122574 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 15), 'not', result_eq_122573)
        
        # Testing the type of an if condition (line 235)
        if_condition_122575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 12), result_not__122574)
        # Assigning a type to the variable 'if_condition_122575' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'if_condition_122575', if_condition_122575)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 236)
        # Processing the call arguments (line 236)
        str_122577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 33), 'str', 'Problem reading file header: ')
        # Getting the type of 'line' (line 236)
        line_122578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 67), 'line', False)
        # Applying the binary operator '+' (line 236)
        result_add_122579 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 33), '+', str_122577, line_122578)
        
        # Processing the call keyword arguments (line 236)
        kwargs_122580 = {}
        # Getting the type of 'ValueError' (line 236)
        ValueError_122576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 236)
        ValueError_call_result_122581 = invoke(stypy.reporting.localization.Localization(__file__, 236, 22), ValueError_122576, *[result_add_122579], **kwargs_122580)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 236, 16), ValueError_call_result_122581, 'raise parameter', BaseException)
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to lower(...): (line 239)
        # Processing the call keyword arguments (line 239)
        kwargs_122584 = {}
        # Getting the type of 'format' (line 239)
        format_122582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'format', False)
        # Obtaining the member 'lower' of a type (line 239)
        lower_122583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 15), format_122582, 'lower')
        # Calling lower(args, kwargs) (line 239)
        lower_call_result_122585 = invoke(stypy.reporting.localization.Localization(__file__, 239, 15), lower_122583, *[], **kwargs_122584)
        
        str_122586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 33), 'str', 'array')
        # Applying the binary operator '==' (line 239)
        result_eq_122587 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 15), '==', lower_call_result_122585, str_122586)
        
        # Testing the type of an if condition (line 239)
        if_condition_122588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 12), result_eq_122587)
        # Assigning a type to the variable 'if_condition_122588' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'if_condition_122588', if_condition_122588)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 240):
        
        # Assigning a Attribute to a Name (line 240):
        # Getting the type of 'self' (line 240)
        self_122589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'self')
        # Obtaining the member 'FORMAT_ARRAY' of a type (line 240)
        FORMAT_ARRAY_122590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 25), self_122589, 'FORMAT_ARRAY')
        # Assigning a type to the variable 'format' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'format', FORMAT_ARRAY_122590)
        # SSA branch for the else part of an if statement (line 239)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to lower(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_122593 = {}
        # Getting the type of 'format' (line 241)
        format_122591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'format', False)
        # Obtaining the member 'lower' of a type (line 241)
        lower_122592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 17), format_122591, 'lower')
        # Calling lower(args, kwargs) (line 241)
        lower_call_result_122594 = invoke(stypy.reporting.localization.Localization(__file__, 241, 17), lower_122592, *[], **kwargs_122593)
        
        str_122595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 35), 'str', 'coordinate')
        # Applying the binary operator '==' (line 241)
        result_eq_122596 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 17), '==', lower_call_result_122594, str_122595)
        
        # Testing the type of an if condition (line 241)
        if_condition_122597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 17), result_eq_122596)
        # Assigning a type to the variable 'if_condition_122597' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'if_condition_122597', if_condition_122597)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 242):
        
        # Assigning a Attribute to a Name (line 242):
        # Getting the type of 'self' (line 242)
        self_122598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 25), 'self')
        # Obtaining the member 'FORMAT_COORDINATE' of a type (line 242)
        FORMAT_COORDINATE_122599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 25), self_122598, 'FORMAT_COORDINATE')
        # Assigning a type to the variable 'format' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'format', FORMAT_COORDINATE_122599)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to startswith(...): (line 245)
        # Processing the call arguments (line 245)
        str_122602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 34), 'str', '%')
        # Processing the call keyword arguments (line 245)
        kwargs_122603 = {}
        # Getting the type of 'line' (line 245)
        line_122600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 18), 'line', False)
        # Obtaining the member 'startswith' of a type (line 245)
        startswith_122601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 18), line_122600, 'startswith')
        # Calling startswith(args, kwargs) (line 245)
        startswith_call_result_122604 = invoke(stypy.reporting.localization.Localization(__file__, 245, 18), startswith_122601, *[str_122602], **kwargs_122603)
        
        # Testing the type of an if condition (line 245)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 12), startswith_call_result_122604)
        # SSA begins for while statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to readline(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_122607 = {}
        # Getting the type of 'stream' (line 246)
        stream_122605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'stream', False)
        # Obtaining the member 'readline' of a type (line 246)
        readline_122606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 23), stream_122605, 'readline')
        # Calling readline(args, kwargs) (line 246)
        readline_call_result_122608 = invoke(stypy.reporting.localization.Localization(__file__, 246, 23), readline_122606, *[], **kwargs_122607)
        
        # Assigning a type to the variable 'line' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'line', readline_call_result_122608)
        # SSA join for while statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to split(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_122611 = {}
        # Getting the type of 'line' (line 248)
        line_122609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'line', False)
        # Obtaining the member 'split' of a type (line 248)
        split_122610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 19), line_122609, 'split')
        # Calling split(args, kwargs) (line 248)
        split_call_result_122612 = invoke(stypy.reporting.localization.Localization(__file__, 248, 19), split_122610, *[], **kwargs_122611)
        
        # Assigning a type to the variable 'line' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'line', split_call_result_122612)
        
        
        # Getting the type of 'format' (line 249)
        format_122613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 15), 'format')
        # Getting the type of 'self' (line 249)
        self_122614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'self')
        # Obtaining the member 'FORMAT_ARRAY' of a type (line 249)
        FORMAT_ARRAY_122615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), self_122614, 'FORMAT_ARRAY')
        # Applying the binary operator '==' (line 249)
        result_eq_122616 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 15), '==', format_122613, FORMAT_ARRAY_122615)
        
        # Testing the type of an if condition (line 249)
        if_condition_122617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 12), result_eq_122616)
        # Assigning a type to the variable 'if_condition_122617' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'if_condition_122617', if_condition_122617)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        
        # Call to len(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'line' (line 250)
        line_122619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'line', False)
        # Processing the call keyword arguments (line 250)
        kwargs_122620 = {}
        # Getting the type of 'len' (line 250)
        len_122618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 23), 'len', False)
        # Calling len(args, kwargs) (line 250)
        len_call_result_122621 = invoke(stypy.reporting.localization.Localization(__file__, 250, 23), len_122618, *[line_122619], **kwargs_122620)
        
        int_122622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 36), 'int')
        # Applying the binary operator '==' (line 250)
        result_eq_122623 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 23), '==', len_call_result_122621, int_122622)
        
        # Applying the 'not' unary operator (line 250)
        result_not__122624 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 19), 'not', result_eq_122623)
        
        # Testing the type of an if condition (line 250)
        if_condition_122625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 16), result_not__122624)
        # Assigning a type to the variable 'if_condition_122625' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'if_condition_122625', if_condition_122625)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 251)
        # Processing the call arguments (line 251)
        str_122627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 37), 'str', 'Header line not of length 2: ')
        # Getting the type of 'line' (line 251)
        line_122628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 71), 'line', False)
        # Applying the binary operator '+' (line 251)
        result_add_122629 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 37), '+', str_122627, line_122628)
        
        # Processing the call keyword arguments (line 251)
        kwargs_122630 = {}
        # Getting the type of 'ValueError' (line 251)
        ValueError_122626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 251)
        ValueError_call_result_122631 = invoke(stypy.reporting.localization.Localization(__file__, 251, 26), ValueError_122626, *[result_add_122629], **kwargs_122630)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 251, 20), ValueError_call_result_122631, 'raise parameter', BaseException)
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 252):
        
        # Assigning a Subscript to a Name (line 252):
        
        # Obtaining the type of the subscript
        int_122632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 16), 'int')
        
        # Call to map(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'int' (line 252)
        int_122634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 33), 'int', False)
        # Getting the type of 'line' (line 252)
        line_122635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 38), 'line', False)
        # Processing the call keyword arguments (line 252)
        kwargs_122636 = {}
        # Getting the type of 'map' (line 252)
        map_122633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'map', False)
        # Calling map(args, kwargs) (line 252)
        map_call_result_122637 = invoke(stypy.reporting.localization.Localization(__file__, 252, 29), map_122633, *[int_122634, line_122635], **kwargs_122636)
        
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___122638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), map_call_result_122637, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_122639 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), getitem___122638, int_122632)
        
        # Assigning a type to the variable 'tuple_var_assignment_122286' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'tuple_var_assignment_122286', subscript_call_result_122639)
        
        # Assigning a Subscript to a Name (line 252):
        
        # Obtaining the type of the subscript
        int_122640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 16), 'int')
        
        # Call to map(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'int' (line 252)
        int_122642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 33), 'int', False)
        # Getting the type of 'line' (line 252)
        line_122643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 38), 'line', False)
        # Processing the call keyword arguments (line 252)
        kwargs_122644 = {}
        # Getting the type of 'map' (line 252)
        map_122641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'map', False)
        # Calling map(args, kwargs) (line 252)
        map_call_result_122645 = invoke(stypy.reporting.localization.Localization(__file__, 252, 29), map_122641, *[int_122642, line_122643], **kwargs_122644)
        
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___122646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), map_call_result_122645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_122647 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), getitem___122646, int_122640)
        
        # Assigning a type to the variable 'tuple_var_assignment_122287' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'tuple_var_assignment_122287', subscript_call_result_122647)
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'tuple_var_assignment_122286' (line 252)
        tuple_var_assignment_122286_122648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'tuple_var_assignment_122286')
        # Assigning a type to the variable 'rows' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'rows', tuple_var_assignment_122286_122648)
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'tuple_var_assignment_122287' (line 252)
        tuple_var_assignment_122287_122649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'tuple_var_assignment_122287')
        # Assigning a type to the variable 'cols' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 22), 'cols', tuple_var_assignment_122287_122649)
        
        # Assigning a BinOp to a Name (line 253):
        
        # Assigning a BinOp to a Name (line 253):
        # Getting the type of 'rows' (line 253)
        rows_122650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'rows')
        # Getting the type of 'cols' (line 253)
        cols_122651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'cols')
        # Applying the binary operator '*' (line 253)
        result_mul_122652 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 26), '*', rows_122650, cols_122651)
        
        # Assigning a type to the variable 'entries' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'entries', result_mul_122652)
        # SSA branch for the else part of an if statement (line 249)
        module_type_store.open_ssa_branch('else')
        
        
        
        
        # Call to len(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'line' (line 255)
        line_122654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'line', False)
        # Processing the call keyword arguments (line 255)
        kwargs_122655 = {}
        # Getting the type of 'len' (line 255)
        len_122653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'len', False)
        # Calling len(args, kwargs) (line 255)
        len_call_result_122656 = invoke(stypy.reporting.localization.Localization(__file__, 255, 23), len_122653, *[line_122654], **kwargs_122655)
        
        int_122657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 36), 'int')
        # Applying the binary operator '==' (line 255)
        result_eq_122658 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 23), '==', len_call_result_122656, int_122657)
        
        # Applying the 'not' unary operator (line 255)
        result_not__122659 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 19), 'not', result_eq_122658)
        
        # Testing the type of an if condition (line 255)
        if_condition_122660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 16), result_not__122659)
        # Assigning a type to the variable 'if_condition_122660' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'if_condition_122660', if_condition_122660)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 256)
        # Processing the call arguments (line 256)
        str_122662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 37), 'str', 'Header line not of length 3: ')
        # Getting the type of 'line' (line 256)
        line_122663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 71), 'line', False)
        # Applying the binary operator '+' (line 256)
        result_add_122664 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 37), '+', str_122662, line_122663)
        
        # Processing the call keyword arguments (line 256)
        kwargs_122665 = {}
        # Getting the type of 'ValueError' (line 256)
        ValueError_122661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 256)
        ValueError_call_result_122666 = invoke(stypy.reporting.localization.Localization(__file__, 256, 26), ValueError_122661, *[result_add_122664], **kwargs_122665)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 20), ValueError_call_result_122666, 'raise parameter', BaseException)
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 257):
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_122667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'int')
        
        # Call to map(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'int' (line 257)
        int_122669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 42), 'int', False)
        # Getting the type of 'line' (line 257)
        line_122670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 47), 'line', False)
        # Processing the call keyword arguments (line 257)
        kwargs_122671 = {}
        # Getting the type of 'map' (line 257)
        map_122668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'map', False)
        # Calling map(args, kwargs) (line 257)
        map_call_result_122672 = invoke(stypy.reporting.localization.Localization(__file__, 257, 38), map_122668, *[int_122669, line_122670], **kwargs_122671)
        
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___122673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), map_call_result_122672, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_122674 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), getitem___122673, int_122667)
        
        # Assigning a type to the variable 'tuple_var_assignment_122288' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'tuple_var_assignment_122288', subscript_call_result_122674)
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_122675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'int')
        
        # Call to map(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'int' (line 257)
        int_122677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 42), 'int', False)
        # Getting the type of 'line' (line 257)
        line_122678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 47), 'line', False)
        # Processing the call keyword arguments (line 257)
        kwargs_122679 = {}
        # Getting the type of 'map' (line 257)
        map_122676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'map', False)
        # Calling map(args, kwargs) (line 257)
        map_call_result_122680 = invoke(stypy.reporting.localization.Localization(__file__, 257, 38), map_122676, *[int_122677, line_122678], **kwargs_122679)
        
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___122681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), map_call_result_122680, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_122682 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), getitem___122681, int_122675)
        
        # Assigning a type to the variable 'tuple_var_assignment_122289' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'tuple_var_assignment_122289', subscript_call_result_122682)
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_122683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'int')
        
        # Call to map(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'int' (line 257)
        int_122685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 42), 'int', False)
        # Getting the type of 'line' (line 257)
        line_122686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 47), 'line', False)
        # Processing the call keyword arguments (line 257)
        kwargs_122687 = {}
        # Getting the type of 'map' (line 257)
        map_122684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'map', False)
        # Calling map(args, kwargs) (line 257)
        map_call_result_122688 = invoke(stypy.reporting.localization.Localization(__file__, 257, 38), map_122684, *[int_122685, line_122686], **kwargs_122687)
        
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___122689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), map_call_result_122688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_122690 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), getitem___122689, int_122683)
        
        # Assigning a type to the variable 'tuple_var_assignment_122290' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'tuple_var_assignment_122290', subscript_call_result_122690)
        
        # Assigning a Name to a Name (line 257):
        # Getting the type of 'tuple_var_assignment_122288' (line 257)
        tuple_var_assignment_122288_122691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'tuple_var_assignment_122288')
        # Assigning a type to the variable 'rows' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'rows', tuple_var_assignment_122288_122691)
        
        # Assigning a Name to a Name (line 257):
        # Getting the type of 'tuple_var_assignment_122289' (line 257)
        tuple_var_assignment_122289_122692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'tuple_var_assignment_122289')
        # Assigning a type to the variable 'cols' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'cols', tuple_var_assignment_122289_122692)
        
        # Assigning a Name to a Name (line 257):
        # Getting the type of 'tuple_var_assignment_122290' (line 257)
        tuple_var_assignment_122290_122693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'tuple_var_assignment_122290')
        # Assigning a type to the variable 'entries' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'entries', tuple_var_assignment_122290_122693)
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_122694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        # Getting the type of 'rows' (line 259)
        rows_122695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'rows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), tuple_122694, rows_122695)
        # Adding element type (line 259)
        # Getting the type of 'cols' (line 259)
        cols_122696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'cols')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), tuple_122694, cols_122696)
        # Adding element type (line 259)
        # Getting the type of 'entries' (line 259)
        entries_122697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 32), 'entries')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), tuple_122694, entries_122697)
        # Adding element type (line 259)
        # Getting the type of 'format' (line 259)
        format_122698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 41), 'format')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), tuple_122694, format_122698)
        # Adding element type (line 259)
        
        # Call to lower(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_122701 = {}
        # Getting the type of 'field' (line 259)
        field_122699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 49), 'field', False)
        # Obtaining the member 'lower' of a type (line 259)
        lower_122700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 49), field_122699, 'lower')
        # Calling lower(args, kwargs) (line 259)
        lower_call_result_122702 = invoke(stypy.reporting.localization.Localization(__file__, 259, 49), lower_122700, *[], **kwargs_122701)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), tuple_122694, lower_call_result_122702)
        # Adding element type (line 259)
        
        # Call to lower(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_122705 = {}
        # Getting the type of 'symmetry' (line 260)
        symmetry_122703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'symmetry', False)
        # Obtaining the member 'lower' of a type (line 260)
        lower_122704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 20), symmetry_122703, 'lower')
        # Calling lower(args, kwargs) (line 260)
        lower_call_result_122706 = invoke(stypy.reporting.localization.Localization(__file__, 260, 20), lower_122704, *[], **kwargs_122705)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 20), tuple_122694, lower_call_result_122706)
        
        # Assigning a type to the variable 'stypy_return_type' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'stypy_return_type', tuple_122694)
        
        # finally branch of the try-finally block (line 227)
        
        # Getting the type of 'close_it' (line 263)
        close_it_122707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'close_it')
        # Testing the type of an if condition (line 263)
        if_condition_122708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 12), close_it_122707)
        # Assigning a type to the variable 'if_condition_122708' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'if_condition_122708', if_condition_122708)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 264)
        # Processing the call keyword arguments (line 264)
        kwargs_122711 = {}
        # Getting the type of 'stream' (line 264)
        stream_122709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'stream', False)
        # Obtaining the member 'close' of a type (line 264)
        close_122710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), stream_122709, 'close')
        # Calling close(args, kwargs) (line 264)
        close_call_result_122712 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), close_122710, *[], **kwargs_122711)
        
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # ################# End of 'info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'info' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_122713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'info'
        return stypy_return_type_122713


    @staticmethod
    @norecursion
    def _open(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_122714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 29), 'str', 'rb')
        defaults = [str_122714]
        # Create a new context for function '_open'
        module_type_store = module_type_store.open_function_context('_open', 267, 4, False)
        
        # Passed parameters checking function
        MMFile._open.__dict__.__setitem__('stypy_localization', localization)
        MMFile._open.__dict__.__setitem__('stypy_type_of_self', None)
        MMFile._open.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._open.__dict__.__setitem__('stypy_function_name', '_open')
        MMFile._open.__dict__.__setitem__('stypy_param_names_list', ['filespec', 'mode'])
        MMFile._open.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._open.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._open.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._open.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._open.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._open.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '_open', ['filespec', 'mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_open', localization, ['mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_open(...)' code ##################

        str_122715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'str', ' Return an open file stream for reading based on source.\n\n        If source is a file name, open it (after trying to find it with mtx and\n        gzipped mtx extensions).  Otherwise, just return source.\n\n        Parameters\n        ----------\n        filespec : str or file-like\n            String giving file name or file-like object\n        mode : str, optional\n            Mode with which to open file, if `filespec` is a file name.\n\n        Returns\n        -------\n        fobj : file-like\n            Open file-like object.\n        close_it : bool\n            True if the calling function should close this file when done,\n            false otherwise.\n        ')
        
        # Assigning a Name to a Name (line 289):
        
        # Assigning a Name to a Name (line 289):
        # Getting the type of 'False' (line 289)
        False_122716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 19), 'False')
        # Assigning a type to the variable 'close_it' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'close_it', False_122716)
        
        
        # Call to isinstance(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'filespec' (line 290)
        filespec_122718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'filespec', False)
        # Getting the type of 'string_types' (line 290)
        string_types_122719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'string_types', False)
        # Processing the call keyword arguments (line 290)
        kwargs_122720 = {}
        # Getting the type of 'isinstance' (line 290)
        isinstance_122717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 290)
        isinstance_call_result_122721 = invoke(stypy.reporting.localization.Localization(__file__, 290, 11), isinstance_122717, *[filespec_122718, string_types_122719], **kwargs_122720)
        
        # Testing the type of an if condition (line 290)
        if_condition_122722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 8), isinstance_call_result_122721)
        # Assigning a type to the variable 'if_condition_122722' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'if_condition_122722', if_condition_122722)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 291):
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'True' (line 291)
        True_122723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'True')
        # Assigning a type to the variable 'close_it' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'close_it', True_122723)
        
        
        
        # Obtaining the type of the subscript
        int_122724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'int')
        # Getting the type of 'mode' (line 294)
        mode_122725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'mode')
        # Obtaining the member '__getitem__' of a type (line 294)
        getitem___122726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), mode_122725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 294)
        subscript_call_result_122727 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), getitem___122726, int_122724)
        
        str_122728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'str', 'r')
        # Applying the binary operator '==' (line 294)
        result_eq_122729 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), '==', subscript_call_result_122727, str_122728)
        
        # Testing the type of an if condition (line 294)
        if_condition_122730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_eq_122729)
        # Assigning a type to the variable 'if_condition_122730' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_122730', if_condition_122730)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to isfile(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'filespec' (line 297)
        filespec_122734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 38), 'filespec', False)
        # Processing the call keyword arguments (line 297)
        kwargs_122735 = {}
        # Getting the type of 'os' (line 297)
        os_122731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 297)
        path_122732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 23), os_122731, 'path')
        # Obtaining the member 'isfile' of a type (line 297)
        isfile_122733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 23), path_122732, 'isfile')
        # Calling isfile(args, kwargs) (line 297)
        isfile_call_result_122736 = invoke(stypy.reporting.localization.Localization(__file__, 297, 23), isfile_122733, *[filespec_122734], **kwargs_122735)
        
        # Applying the 'not' unary operator (line 297)
        result_not__122737 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 19), 'not', isfile_call_result_122736)
        
        # Testing the type of an if condition (line 297)
        if_condition_122738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 16), result_not__122737)
        # Assigning a type to the variable 'if_condition_122738' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'if_condition_122738', if_condition_122738)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to isfile(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'filespec' (line 298)
        filespec_122742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 38), 'filespec', False)
        str_122743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 47), 'str', '.mtx')
        # Applying the binary operator '+' (line 298)
        result_add_122744 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 38), '+', filespec_122742, str_122743)
        
        # Processing the call keyword arguments (line 298)
        kwargs_122745 = {}
        # Getting the type of 'os' (line 298)
        os_122739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 298)
        path_122740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 23), os_122739, 'path')
        # Obtaining the member 'isfile' of a type (line 298)
        isfile_122741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 23), path_122740, 'isfile')
        # Calling isfile(args, kwargs) (line 298)
        isfile_call_result_122746 = invoke(stypy.reporting.localization.Localization(__file__, 298, 23), isfile_122741, *[result_add_122744], **kwargs_122745)
        
        # Testing the type of an if condition (line 298)
        if_condition_122747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 20), isfile_call_result_122746)
        # Assigning a type to the variable 'if_condition_122747' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'if_condition_122747', if_condition_122747)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 299):
        
        # Assigning a BinOp to a Name (line 299):
        # Getting the type of 'filespec' (line 299)
        filespec_122748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 35), 'filespec')
        str_122749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 46), 'str', '.mtx')
        # Applying the binary operator '+' (line 299)
        result_add_122750 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 35), '+', filespec_122748, str_122749)
        
        # Assigning a type to the variable 'filespec' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'filespec', result_add_122750)
        # SSA branch for the else part of an if statement (line 298)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isfile(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'filespec' (line 300)
        filespec_122754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 40), 'filespec', False)
        str_122755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 49), 'str', '.mtx.gz')
        # Applying the binary operator '+' (line 300)
        result_add_122756 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 40), '+', filespec_122754, str_122755)
        
        # Processing the call keyword arguments (line 300)
        kwargs_122757 = {}
        # Getting the type of 'os' (line 300)
        os_122751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 300)
        path_122752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 25), os_122751, 'path')
        # Obtaining the member 'isfile' of a type (line 300)
        isfile_122753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 25), path_122752, 'isfile')
        # Calling isfile(args, kwargs) (line 300)
        isfile_call_result_122758 = invoke(stypy.reporting.localization.Localization(__file__, 300, 25), isfile_122753, *[result_add_122756], **kwargs_122757)
        
        # Testing the type of an if condition (line 300)
        if_condition_122759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 25), isfile_call_result_122758)
        # Assigning a type to the variable 'if_condition_122759' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 25), 'if_condition_122759', if_condition_122759)
        # SSA begins for if statement (line 300)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 301):
        
        # Assigning a BinOp to a Name (line 301):
        # Getting the type of 'filespec' (line 301)
        filespec_122760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 35), 'filespec')
        str_122761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 46), 'str', '.mtx.gz')
        # Applying the binary operator '+' (line 301)
        result_add_122762 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 35), '+', filespec_122760, str_122761)
        
        # Assigning a type to the variable 'filespec' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 24), 'filespec', result_add_122762)
        # SSA branch for the else part of an if statement (line 300)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isfile(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'filespec' (line 302)
        filespec_122766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 40), 'filespec', False)
        str_122767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 49), 'str', '.mtx.bz2')
        # Applying the binary operator '+' (line 302)
        result_add_122768 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 40), '+', filespec_122766, str_122767)
        
        # Processing the call keyword arguments (line 302)
        kwargs_122769 = {}
        # Getting the type of 'os' (line 302)
        os_122763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 302)
        path_122764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 25), os_122763, 'path')
        # Obtaining the member 'isfile' of a type (line 302)
        isfile_122765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 25), path_122764, 'isfile')
        # Calling isfile(args, kwargs) (line 302)
        isfile_call_result_122770 = invoke(stypy.reporting.localization.Localization(__file__, 302, 25), isfile_122765, *[result_add_122768], **kwargs_122769)
        
        # Testing the type of an if condition (line 302)
        if_condition_122771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 25), isfile_call_result_122770)
        # Assigning a type to the variable 'if_condition_122771' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 25), 'if_condition_122771', if_condition_122771)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 303):
        
        # Assigning a BinOp to a Name (line 303):
        # Getting the type of 'filespec' (line 303)
        filespec_122772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 35), 'filespec')
        str_122773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 46), 'str', '.mtx.bz2')
        # Applying the binary operator '+' (line 303)
        result_add_122774 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 35), '+', filespec_122772, str_122773)
        
        # Assigning a type to the variable 'filespec' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'filespec', result_add_122774)
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 300)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to endswith(...): (line 305)
        # Processing the call arguments (line 305)
        str_122777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 37), 'str', '.gz')
        # Processing the call keyword arguments (line 305)
        kwargs_122778 = {}
        # Getting the type of 'filespec' (line 305)
        filespec_122775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'filespec', False)
        # Obtaining the member 'endswith' of a type (line 305)
        endswith_122776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 19), filespec_122775, 'endswith')
        # Calling endswith(args, kwargs) (line 305)
        endswith_call_result_122779 = invoke(stypy.reporting.localization.Localization(__file__, 305, 19), endswith_122776, *[str_122777], **kwargs_122778)
        
        # Testing the type of an if condition (line 305)
        if_condition_122780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 16), endswith_call_result_122779)
        # Assigning a type to the variable 'if_condition_122780' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'if_condition_122780', if_condition_122780)
        # SSA begins for if statement (line 305)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 306, 20))
        
        # 'import gzip' statement (line 306)
        import gzip

        import_module(stypy.reporting.localization.Localization(__file__, 306, 20), 'gzip', gzip, module_type_store)
        
        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to open(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'filespec' (line 307)
        filespec_122783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 39), 'filespec', False)
        # Getting the type of 'mode' (line 307)
        mode_122784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 49), 'mode', False)
        # Processing the call keyword arguments (line 307)
        kwargs_122785 = {}
        # Getting the type of 'gzip' (line 307)
        gzip_122781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 29), 'gzip', False)
        # Obtaining the member 'open' of a type (line 307)
        open_122782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 29), gzip_122781, 'open')
        # Calling open(args, kwargs) (line 307)
        open_call_result_122786 = invoke(stypy.reporting.localization.Localization(__file__, 307, 29), open_122782, *[filespec_122783, mode_122784], **kwargs_122785)
        
        # Assigning a type to the variable 'stream' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'stream', open_call_result_122786)
        # SSA branch for the else part of an if statement (line 305)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to endswith(...): (line 308)
        # Processing the call arguments (line 308)
        str_122789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 39), 'str', '.bz2')
        # Processing the call keyword arguments (line 308)
        kwargs_122790 = {}
        # Getting the type of 'filespec' (line 308)
        filespec_122787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 21), 'filespec', False)
        # Obtaining the member 'endswith' of a type (line 308)
        endswith_122788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 21), filespec_122787, 'endswith')
        # Calling endswith(args, kwargs) (line 308)
        endswith_call_result_122791 = invoke(stypy.reporting.localization.Localization(__file__, 308, 21), endswith_122788, *[str_122789], **kwargs_122790)
        
        # Testing the type of an if condition (line 308)
        if_condition_122792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 21), endswith_call_result_122791)
        # Assigning a type to the variable 'if_condition_122792' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 21), 'if_condition_122792', if_condition_122792)
        # SSA begins for if statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 309, 20))
        
        # 'import bz2' statement (line 309)
        import bz2

        import_module(stypy.reporting.localization.Localization(__file__, 309, 20), 'bz2', bz2, module_type_store)
        
        
        # Assigning a Call to a Name (line 310):
        
        # Assigning a Call to a Name (line 310):
        
        # Call to BZ2File(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'filespec' (line 310)
        filespec_122795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 41), 'filespec', False)
        str_122796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 51), 'str', 'rb')
        # Processing the call keyword arguments (line 310)
        kwargs_122797 = {}
        # Getting the type of 'bz2' (line 310)
        bz2_122793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 29), 'bz2', False)
        # Obtaining the member 'BZ2File' of a type (line 310)
        BZ2File_122794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 29), bz2_122793, 'BZ2File')
        # Calling BZ2File(args, kwargs) (line 310)
        BZ2File_call_result_122798 = invoke(stypy.reporting.localization.Localization(__file__, 310, 29), BZ2File_122794, *[filespec_122795, str_122796], **kwargs_122797)
        
        # Assigning a type to the variable 'stream' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'stream', BZ2File_call_result_122798)
        # SSA branch for the else part of an if statement (line 308)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to open(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'filespec' (line 312)
        filespec_122800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'filespec', False)
        # Getting the type of 'mode' (line 312)
        mode_122801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 44), 'mode', False)
        # Processing the call keyword arguments (line 312)
        kwargs_122802 = {}
        # Getting the type of 'open' (line 312)
        open_122799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 29), 'open', False)
        # Calling open(args, kwargs) (line 312)
        open_call_result_122803 = invoke(stypy.reporting.localization.Localization(__file__, 312, 29), open_122799, *[filespec_122800, mode_122801], **kwargs_122802)
        
        # Assigning a type to the variable 'stream' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'stream', open_call_result_122803)
        # SSA join for if statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 305)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 294)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_122804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 28), 'int')
        slice_122805 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 316, 19), int_122804, None, None)
        # Getting the type of 'filespec' (line 316)
        filespec_122806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 19), 'filespec')
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___122807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 19), filespec_122806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_122808 = invoke(stypy.reporting.localization.Localization(__file__, 316, 19), getitem___122807, slice_122805)
        
        str_122809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 36), 'str', '.mtx')
        # Applying the binary operator '!=' (line 316)
        result_ne_122810 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 19), '!=', subscript_call_result_122808, str_122809)
        
        # Testing the type of an if condition (line 316)
        if_condition_122811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 16), result_ne_122810)
        # Assigning a type to the variable 'if_condition_122811' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'if_condition_122811', if_condition_122811)
        # SSA begins for if statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 317):
        
        # Assigning a BinOp to a Name (line 317):
        # Getting the type of 'filespec' (line 317)
        filespec_122812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'filespec')
        str_122813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 42), 'str', '.mtx')
        # Applying the binary operator '+' (line 317)
        result_add_122814 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 31), '+', filespec_122812, str_122813)
        
        # Assigning a type to the variable 'filespec' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'filespec', result_add_122814)
        # SSA join for if statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to open(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'filespec' (line 318)
        filespec_122816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'filespec', False)
        # Getting the type of 'mode' (line 318)
        mode_122817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 40), 'mode', False)
        # Processing the call keyword arguments (line 318)
        kwargs_122818 = {}
        # Getting the type of 'open' (line 318)
        open_122815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'open', False)
        # Calling open(args, kwargs) (line 318)
        open_call_result_122819 = invoke(stypy.reporting.localization.Localization(__file__, 318, 25), open_122815, *[filespec_122816, mode_122817], **kwargs_122818)
        
        # Assigning a type to the variable 'stream' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'stream', open_call_result_122819)
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 290)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 320):
        
        # Assigning a Name to a Name (line 320):
        # Getting the type of 'filespec' (line 320)
        filespec_122820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 21), 'filespec')
        # Assigning a type to the variable 'stream' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'stream', filespec_122820)
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 322)
        tuple_122821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 322)
        # Adding element type (line 322)
        # Getting the type of 'stream' (line 322)
        stream_122822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'stream')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 15), tuple_122821, stream_122822)
        # Adding element type (line 322)
        # Getting the type of 'close_it' (line 322)
        close_it_122823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 23), 'close_it')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 15), tuple_122821, close_it_122823)
        
        # Assigning a type to the variable 'stypy_return_type' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'stypy_return_type', tuple_122821)
        
        # ################# End of '_open(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_open' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_122824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122824)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_open'
        return stypy_return_type_122824


    @staticmethod
    @norecursion
    def _get_symmetry(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_symmetry'
        module_type_store = module_type_store.open_function_context('_get_symmetry', 325, 4, False)
        
        # Passed parameters checking function
        MMFile._get_symmetry.__dict__.__setitem__('stypy_localization', localization)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_type_of_self', None)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_function_name', '_get_symmetry')
        MMFile._get_symmetry.__dict__.__setitem__('stypy_param_names_list', ['a'])
        MMFile._get_symmetry.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._get_symmetry.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '_get_symmetry', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_symmetry', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_symmetry(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 327):
        
        # Assigning a Subscript to a Name (line 327):
        
        # Obtaining the type of the subscript
        int_122825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'int')
        # Getting the type of 'a' (line 327)
        a_122826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'a')
        # Obtaining the member 'shape' of a type (line 327)
        shape_122827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 15), a_122826, 'shape')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___122828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), shape_122827, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_122829 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), getitem___122828, int_122825)
        
        # Assigning a type to the variable 'tuple_var_assignment_122291' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_122291', subscript_call_result_122829)
        
        # Assigning a Subscript to a Name (line 327):
        
        # Obtaining the type of the subscript
        int_122830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'int')
        # Getting the type of 'a' (line 327)
        a_122831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'a')
        # Obtaining the member 'shape' of a type (line 327)
        shape_122832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 15), a_122831, 'shape')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___122833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), shape_122832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_122834 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), getitem___122833, int_122830)
        
        # Assigning a type to the variable 'tuple_var_assignment_122292' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_122292', subscript_call_result_122834)
        
        # Assigning a Name to a Name (line 327):
        # Getting the type of 'tuple_var_assignment_122291' (line 327)
        tuple_var_assignment_122291_122835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_122291')
        # Assigning a type to the variable 'm' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'm', tuple_var_assignment_122291_122835)
        
        # Assigning a Name to a Name (line 327):
        # Getting the type of 'tuple_var_assignment_122292' (line 327)
        tuple_var_assignment_122292_122836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_122292')
        # Assigning a type to the variable 'n' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'n', tuple_var_assignment_122292_122836)
        
        
        # Getting the type of 'm' (line 328)
        m_122837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'm')
        # Getting the type of 'n' (line 328)
        n_122838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'n')
        # Applying the binary operator '!=' (line 328)
        result_ne_122839 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 11), '!=', m_122837, n_122838)
        
        # Testing the type of an if condition (line 328)
        if_condition_122840 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 8), result_ne_122839)
        # Assigning a type to the variable 'if_condition_122840' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'if_condition_122840', if_condition_122840)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'MMFile' (line 329)
        MMFile_122841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 19), 'MMFile')
        # Obtaining the member 'SYMMETRY_GENERAL' of a type (line 329)
        SYMMETRY_GENERAL_122842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 19), MMFile_122841, 'SYMMETRY_GENERAL')
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type', SYMMETRY_GENERAL_122842)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 330):
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'True' (line 330)
        True_122843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'True')
        # Assigning a type to the variable 'issymm' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'issymm', True_122843)
        
        # Assigning a Name to a Name (line 331):
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'True' (line 331)
        True_122844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 17), 'True')
        # Assigning a type to the variable 'isskew' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'isskew', True_122844)
        
        # Assigning a Compare to a Name (line 332):
        
        # Assigning a Compare to a Name (line 332):
        
        # Getting the type of 'a' (line 332)
        a_122845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 17), 'a')
        # Obtaining the member 'dtype' of a type (line 332)
        dtype_122846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 17), a_122845, 'dtype')
        # Obtaining the member 'char' of a type (line 332)
        char_122847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 17), dtype_122846, 'char')
        str_122848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'str', 'FD')
        # Applying the binary operator 'in' (line 332)
        result_contains_122849 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 17), 'in', char_122847, str_122848)
        
        # Assigning a type to the variable 'isherm' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'isherm', result_contains_122849)
        
        
        # Call to isspmatrix(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'a' (line 335)
        a_122851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'a', False)
        # Processing the call keyword arguments (line 335)
        kwargs_122852 = {}
        # Getting the type of 'isspmatrix' (line 335)
        isspmatrix_122850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 335)
        isspmatrix_call_result_122853 = invoke(stypy.reporting.localization.Localization(__file__, 335, 11), isspmatrix_122850, *[a_122851], **kwargs_122852)
        
        # Testing the type of an if condition (line 335)
        if_condition_122854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 8), isspmatrix_call_result_122853)
        # Assigning a type to the variable 'if_condition_122854' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'if_condition_122854', if_condition_122854)
        # SSA begins for if statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to tocoo(...): (line 338)
        # Processing the call keyword arguments (line 338)
        kwargs_122857 = {}
        # Getting the type of 'a' (line 338)
        a_122855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'a', False)
        # Obtaining the member 'tocoo' of a type (line 338)
        tocoo_122856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 16), a_122855, 'tocoo')
        # Calling tocoo(args, kwargs) (line 338)
        tocoo_call_result_122858 = invoke(stypy.reporting.localization.Localization(__file__, 338, 16), tocoo_122856, *[], **kwargs_122857)
        
        # Assigning a type to the variable 'a' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'a', tocoo_call_result_122858)
        
        # Assigning a Call to a Tuple (line 339):
        
        # Assigning a Subscript to a Name (line 339):
        
        # Obtaining the type of the subscript
        int_122859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 12), 'int')
        
        # Call to nonzero(...): (line 339)
        # Processing the call keyword arguments (line 339)
        kwargs_122862 = {}
        # Getting the type of 'a' (line 339)
        a_122860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 25), 'a', False)
        # Obtaining the member 'nonzero' of a type (line 339)
        nonzero_122861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 25), a_122860, 'nonzero')
        # Calling nonzero(args, kwargs) (line 339)
        nonzero_call_result_122863 = invoke(stypy.reporting.localization.Localization(__file__, 339, 25), nonzero_122861, *[], **kwargs_122862)
        
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___122864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), nonzero_call_result_122863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_122865 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), getitem___122864, int_122859)
        
        # Assigning a type to the variable 'tuple_var_assignment_122293' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'tuple_var_assignment_122293', subscript_call_result_122865)
        
        # Assigning a Subscript to a Name (line 339):
        
        # Obtaining the type of the subscript
        int_122866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 12), 'int')
        
        # Call to nonzero(...): (line 339)
        # Processing the call keyword arguments (line 339)
        kwargs_122869 = {}
        # Getting the type of 'a' (line 339)
        a_122867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 25), 'a', False)
        # Obtaining the member 'nonzero' of a type (line 339)
        nonzero_122868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 25), a_122867, 'nonzero')
        # Calling nonzero(args, kwargs) (line 339)
        nonzero_call_result_122870 = invoke(stypy.reporting.localization.Localization(__file__, 339, 25), nonzero_122868, *[], **kwargs_122869)
        
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___122871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), nonzero_call_result_122870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_122872 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), getitem___122871, int_122866)
        
        # Assigning a type to the variable 'tuple_var_assignment_122294' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'tuple_var_assignment_122294', subscript_call_result_122872)
        
        # Assigning a Name to a Name (line 339):
        # Getting the type of 'tuple_var_assignment_122293' (line 339)
        tuple_var_assignment_122293_122873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'tuple_var_assignment_122293')
        # Assigning a type to the variable 'row' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 13), 'row', tuple_var_assignment_122293_122873)
        
        # Assigning a Name to a Name (line 339):
        # Getting the type of 'tuple_var_assignment_122294' (line 339)
        tuple_var_assignment_122294_122874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'tuple_var_assignment_122294')
        # Assigning a type to the variable 'col' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 18), 'col', tuple_var_assignment_122294_122874)
        
        
        
        # Call to sum(...): (line 340)
        # Processing the call keyword arguments (line 340)
        kwargs_122879 = {}
        
        # Getting the type of 'row' (line 340)
        row_122875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'row', False)
        # Getting the type of 'col' (line 340)
        col_122876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 22), 'col', False)
        # Applying the binary operator '<' (line 340)
        result_lt_122877 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 16), '<', row_122875, col_122876)
        
        # Obtaining the member 'sum' of a type (line 340)
        sum_122878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 16), result_lt_122877, 'sum')
        # Calling sum(args, kwargs) (line 340)
        sum_call_result_122880 = invoke(stypy.reporting.localization.Localization(__file__, 340, 16), sum_122878, *[], **kwargs_122879)
        
        
        # Call to sum(...): (line 340)
        # Processing the call keyword arguments (line 340)
        kwargs_122885 = {}
        
        # Getting the type of 'row' (line 340)
        row_122881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'row', False)
        # Getting the type of 'col' (line 340)
        col_122882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 43), 'col', False)
        # Applying the binary operator '>' (line 340)
        result_gt_122883 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 37), '>', row_122881, col_122882)
        
        # Obtaining the member 'sum' of a type (line 340)
        sum_122884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 37), result_gt_122883, 'sum')
        # Calling sum(args, kwargs) (line 340)
        sum_call_result_122886 = invoke(stypy.reporting.localization.Localization(__file__, 340, 37), sum_122884, *[], **kwargs_122885)
        
        # Applying the binary operator '!=' (line 340)
        result_ne_122887 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 15), '!=', sum_call_result_122880, sum_call_result_122886)
        
        # Testing the type of an if condition (line 340)
        if_condition_122888 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 12), result_ne_122887)
        # Assigning a type to the variable 'if_condition_122888' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'if_condition_122888', if_condition_122888)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'MMFile' (line 341)
        MMFile_122889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 23), 'MMFile')
        # Obtaining the member 'SYMMETRY_GENERAL' of a type (line 341)
        SYMMETRY_GENERAL_122890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 23), MMFile_122889, 'SYMMETRY_GENERAL')
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'stypy_return_type', SYMMETRY_GENERAL_122890)
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to todok(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_122893 = {}
        # Getting the type of 'a' (line 344)
        a_122891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'a', False)
        # Obtaining the member 'todok' of a type (line 344)
        todok_122892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), a_122891, 'todok')
        # Calling todok(args, kwargs) (line 344)
        todok_call_result_122894 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), todok_122892, *[], **kwargs_122893)
        
        # Assigning a type to the variable 'a' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'a', todok_call_result_122894)

        @norecursion
        def symm_iterator(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'symm_iterator'
            module_type_store = module_type_store.open_function_context('symm_iterator', 346, 12, False)
            
            # Passed parameters checking function
            symm_iterator.stypy_localization = localization
            symm_iterator.stypy_type_of_self = None
            symm_iterator.stypy_type_store = module_type_store
            symm_iterator.stypy_function_name = 'symm_iterator'
            symm_iterator.stypy_param_names_list = []
            symm_iterator.stypy_varargs_param_name = None
            symm_iterator.stypy_kwargs_param_name = None
            symm_iterator.stypy_call_defaults = defaults
            symm_iterator.stypy_call_varargs = varargs
            symm_iterator.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'symm_iterator', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'symm_iterator', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'symm_iterator(...)' code ##################

            
            
            # Call to items(...): (line 347)
            # Processing the call keyword arguments (line 347)
            kwargs_122897 = {}
            # Getting the type of 'a' (line 347)
            a_122895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 37), 'a', False)
            # Obtaining the member 'items' of a type (line 347)
            items_122896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 37), a_122895, 'items')
            # Calling items(args, kwargs) (line 347)
            items_call_result_122898 = invoke(stypy.reporting.localization.Localization(__file__, 347, 37), items_122896, *[], **kwargs_122897)
            
            # Testing the type of a for loop iterable (line 347)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 347, 16), items_call_result_122898)
            # Getting the type of the for loop variable (line 347)
            for_loop_var_122899 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 347, 16), items_call_result_122898)
            # Assigning a type to the variable 'i' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 16), for_loop_var_122899))
            # Assigning a type to the variable 'j' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 16), for_loop_var_122899))
            # Assigning a type to the variable 'aij' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'aij', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 16), for_loop_var_122899))
            # SSA begins for a for statement (line 347)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'i' (line 348)
            i_122900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 23), 'i')
            # Getting the type of 'j' (line 348)
            j_122901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'j')
            # Applying the binary operator '>' (line 348)
            result_gt_122902 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 23), '>', i_122900, j_122901)
            
            # Testing the type of an if condition (line 348)
            if_condition_122903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 20), result_gt_122902)
            # Assigning a type to the variable 'if_condition_122903' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'if_condition_122903', if_condition_122903)
            # SSA begins for if statement (line 348)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 349):
            
            # Assigning a Subscript to a Name (line 349):
            
            # Obtaining the type of the subscript
            
            # Obtaining an instance of the builtin type 'tuple' (line 349)
            tuple_122904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 32), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 349)
            # Adding element type (line 349)
            # Getting the type of 'j' (line 349)
            j_122905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 32), 'j')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 32), tuple_122904, j_122905)
            # Adding element type (line 349)
            # Getting the type of 'i' (line 349)
            i_122906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 35), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 32), tuple_122904, i_122906)
            
            # Getting the type of 'a' (line 349)
            a_122907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 30), 'a')
            # Obtaining the member '__getitem__' of a type (line 349)
            getitem___122908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 30), a_122907, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 349)
            subscript_call_result_122909 = invoke(stypy.reporting.localization.Localization(__file__, 349, 30), getitem___122908, tuple_122904)
            
            # Assigning a type to the variable 'aji' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'aji', subscript_call_result_122909)
            # Creating a generator
            
            # Obtaining an instance of the builtin type 'tuple' (line 350)
            tuple_122910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 350)
            # Adding element type (line 350)
            # Getting the type of 'aij' (line 350)
            aij_122911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), 'aij')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 31), tuple_122910, aij_122911)
            # Adding element type (line 350)
            # Getting the type of 'aji' (line 350)
            aji_122912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'aji')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 31), tuple_122910, aji_122912)
            
            GeneratorType_122913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 24), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 24), GeneratorType_122913, tuple_122910)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'stypy_return_type', GeneratorType_122913)
            # SSA join for if statement (line 348)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'symm_iterator(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'symm_iterator' in the type store
            # Getting the type of 'stypy_return_type' (line 346)
            stypy_return_type_122914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_122914)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'symm_iterator'
            return stypy_return_type_122914

        # Assigning a type to the variable 'symm_iterator' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'symm_iterator', symm_iterator)
        # SSA branch for the else part of an if statement (line 335)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def symm_iterator(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'symm_iterator'
            module_type_store = module_type_store.open_function_context('symm_iterator', 355, 12, False)
            
            # Passed parameters checking function
            symm_iterator.stypy_localization = localization
            symm_iterator.stypy_type_of_self = None
            symm_iterator.stypy_type_store = module_type_store
            symm_iterator.stypy_function_name = 'symm_iterator'
            symm_iterator.stypy_param_names_list = []
            symm_iterator.stypy_varargs_param_name = None
            symm_iterator.stypy_kwargs_param_name = None
            symm_iterator.stypy_call_defaults = defaults
            symm_iterator.stypy_call_varargs = varargs
            symm_iterator.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'symm_iterator', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'symm_iterator', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'symm_iterator(...)' code ##################

            
            
            # Call to range(...): (line 356)
            # Processing the call arguments (line 356)
            # Getting the type of 'n' (line 356)
            n_122916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 31), 'n', False)
            # Processing the call keyword arguments (line 356)
            kwargs_122917 = {}
            # Getting the type of 'range' (line 356)
            range_122915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 25), 'range', False)
            # Calling range(args, kwargs) (line 356)
            range_call_result_122918 = invoke(stypy.reporting.localization.Localization(__file__, 356, 25), range_122915, *[n_122916], **kwargs_122917)
            
            # Testing the type of a for loop iterable (line 356)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 356, 16), range_call_result_122918)
            # Getting the type of the for loop variable (line 356)
            for_loop_var_122919 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 356, 16), range_call_result_122918)
            # Assigning a type to the variable 'j' (line 356)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'j', for_loop_var_122919)
            # SSA begins for a for statement (line 356)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 357)
            # Processing the call arguments (line 357)
            # Getting the type of 'j' (line 357)
            j_122921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 35), 'j', False)
            int_122922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 37), 'int')
            # Applying the binary operator '+' (line 357)
            result_add_122923 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 35), '+', j_122921, int_122922)
            
            # Getting the type of 'n' (line 357)
            n_122924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 40), 'n', False)
            # Processing the call keyword arguments (line 357)
            kwargs_122925 = {}
            # Getting the type of 'range' (line 357)
            range_122920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 29), 'range', False)
            # Calling range(args, kwargs) (line 357)
            range_call_result_122926 = invoke(stypy.reporting.localization.Localization(__file__, 357, 29), range_122920, *[result_add_122923, n_122924], **kwargs_122925)
            
            # Testing the type of a for loop iterable (line 357)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 357, 20), range_call_result_122926)
            # Getting the type of the for loop variable (line 357)
            for_loop_var_122927 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 357, 20), range_call_result_122926)
            # Assigning a type to the variable 'i' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'i', for_loop_var_122927)
            # SSA begins for a for statement (line 357)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Tuple to a Tuple (line 358):
            
            # Assigning a Subscript to a Name (line 358):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 358)
            j_122928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 40), 'j')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 358)
            i_122929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 37), 'i')
            # Getting the type of 'a' (line 358)
            a_122930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), 'a')
            # Obtaining the member '__getitem__' of a type (line 358)
            getitem___122931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 35), a_122930, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 358)
            subscript_call_result_122932 = invoke(stypy.reporting.localization.Localization(__file__, 358, 35), getitem___122931, i_122929)
            
            # Obtaining the member '__getitem__' of a type (line 358)
            getitem___122933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 35), subscript_call_result_122932, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 358)
            subscript_call_result_122934 = invoke(stypy.reporting.localization.Localization(__file__, 358, 35), getitem___122933, j_122928)
            
            # Assigning a type to the variable 'tuple_assignment_122295' (line 358)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'tuple_assignment_122295', subscript_call_result_122934)
            
            # Assigning a Subscript to a Name (line 358):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 358)
            i_122935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'i')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 358)
            j_122936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 46), 'j')
            # Getting the type of 'a' (line 358)
            a_122937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 44), 'a')
            # Obtaining the member '__getitem__' of a type (line 358)
            getitem___122938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 44), a_122937, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 358)
            subscript_call_result_122939 = invoke(stypy.reporting.localization.Localization(__file__, 358, 44), getitem___122938, j_122936)
            
            # Obtaining the member '__getitem__' of a type (line 358)
            getitem___122940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 44), subscript_call_result_122939, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 358)
            subscript_call_result_122941 = invoke(stypy.reporting.localization.Localization(__file__, 358, 44), getitem___122940, i_122935)
            
            # Assigning a type to the variable 'tuple_assignment_122296' (line 358)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'tuple_assignment_122296', subscript_call_result_122941)
            
            # Assigning a Name to a Name (line 358):
            # Getting the type of 'tuple_assignment_122295' (line 358)
            tuple_assignment_122295_122942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'tuple_assignment_122295')
            # Assigning a type to the variable 'aij' (line 358)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'aij', tuple_assignment_122295_122942)
            
            # Assigning a Name to a Name (line 358):
            # Getting the type of 'tuple_assignment_122296' (line 358)
            tuple_assignment_122296_122943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'tuple_assignment_122296')
            # Assigning a type to the variable 'aji' (line 358)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'aji', tuple_assignment_122296_122943)
            # Creating a generator
            
            # Obtaining an instance of the builtin type 'tuple' (line 359)
            tuple_122944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 359)
            # Adding element type (line 359)
            # Getting the type of 'aij' (line 359)
            aij_122945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 31), 'aij')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 31), tuple_122944, aij_122945)
            # Adding element type (line 359)
            # Getting the type of 'aji' (line 359)
            aji_122946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 36), 'aji')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 31), tuple_122944, aji_122946)
            
            GeneratorType_122947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 24), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 24), GeneratorType_122947, tuple_122944)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'stypy_return_type', GeneratorType_122947)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'symm_iterator(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'symm_iterator' in the type store
            # Getting the type of 'stypy_return_type' (line 355)
            stypy_return_type_122948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_122948)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'symm_iterator'
            return stypy_return_type_122948

        # Assigning a type to the variable 'symm_iterator' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'symm_iterator', symm_iterator)
        # SSA join for if statement (line 335)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to symm_iterator(...): (line 362)
        # Processing the call keyword arguments (line 362)
        kwargs_122950 = {}
        # Getting the type of 'symm_iterator' (line 362)
        symm_iterator_122949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 26), 'symm_iterator', False)
        # Calling symm_iterator(args, kwargs) (line 362)
        symm_iterator_call_result_122951 = invoke(stypy.reporting.localization.Localization(__file__, 362, 26), symm_iterator_122949, *[], **kwargs_122950)
        
        # Testing the type of a for loop iterable (line 362)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 362, 8), symm_iterator_call_result_122951)
        # Getting the type of the for loop variable (line 362)
        for_loop_var_122952 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 362, 8), symm_iterator_call_result_122951)
        # Assigning a type to the variable 'aij' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'aij', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 8), for_loop_var_122952))
        # Assigning a type to the variable 'aji' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'aji', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 8), for_loop_var_122952))
        # SSA begins for a for statement (line 362)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'issymm' (line 363)
        issymm_122953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'issymm')
        
        # Getting the type of 'aij' (line 363)
        aij_122954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'aij')
        # Getting the type of 'aji' (line 363)
        aji_122955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 33), 'aji')
        # Applying the binary operator '!=' (line 363)
        result_ne_122956 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 26), '!=', aij_122954, aji_122955)
        
        # Applying the binary operator 'and' (line 363)
        result_and_keyword_122957 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 15), 'and', issymm_122953, result_ne_122956)
        
        # Testing the type of an if condition (line 363)
        if_condition_122958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 12), result_and_keyword_122957)
        # Assigning a type to the variable 'if_condition_122958' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'if_condition_122958', if_condition_122958)
        # SSA begins for if statement (line 363)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 364):
        
        # Assigning a Name to a Name (line 364):
        # Getting the type of 'False' (line 364)
        False_122959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 25), 'False')
        # Assigning a type to the variable 'issymm' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'issymm', False_122959)
        # SSA join for if statement (line 363)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'isskew' (line 365)
        isskew_122960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'isskew')
        
        # Getting the type of 'aij' (line 365)
        aij_122961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 26), 'aij')
        
        # Getting the type of 'aji' (line 365)
        aji_122962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'aji')
        # Applying the 'usub' unary operator (line 365)
        result___neg___122963 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 33), 'usub', aji_122962)
        
        # Applying the binary operator '!=' (line 365)
        result_ne_122964 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 26), '!=', aij_122961, result___neg___122963)
        
        # Applying the binary operator 'and' (line 365)
        result_and_keyword_122965 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 15), 'and', isskew_122960, result_ne_122964)
        
        # Testing the type of an if condition (line 365)
        if_condition_122966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 12), result_and_keyword_122965)
        # Assigning a type to the variable 'if_condition_122966' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'if_condition_122966', if_condition_122966)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 366):
        
        # Assigning a Name to a Name (line 366):
        # Getting the type of 'False' (line 366)
        False_122967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 25), 'False')
        # Assigning a type to the variable 'isskew' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'isskew', False_122967)
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'isherm' (line 367)
        isherm_122968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'isherm')
        
        # Getting the type of 'aij' (line 367)
        aij_122969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 26), 'aij')
        
        # Call to conj(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'aji' (line 367)
        aji_122971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 38), 'aji', False)
        # Processing the call keyword arguments (line 367)
        kwargs_122972 = {}
        # Getting the type of 'conj' (line 367)
        conj_122970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 33), 'conj', False)
        # Calling conj(args, kwargs) (line 367)
        conj_call_result_122973 = invoke(stypy.reporting.localization.Localization(__file__, 367, 33), conj_122970, *[aji_122971], **kwargs_122972)
        
        # Applying the binary operator '!=' (line 367)
        result_ne_122974 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 26), '!=', aij_122969, conj_call_result_122973)
        
        # Applying the binary operator 'and' (line 367)
        result_and_keyword_122975 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 15), 'and', isherm_122968, result_ne_122974)
        
        # Testing the type of an if condition (line 367)
        if_condition_122976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 12), result_and_keyword_122975)
        # Assigning a type to the variable 'if_condition_122976' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'if_condition_122976', if_condition_122976)
        # SSA begins for if statement (line 367)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 368):
        
        # Assigning a Name to a Name (line 368):
        # Getting the type of 'False' (line 368)
        False_122977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), 'False')
        # Assigning a type to the variable 'isherm' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'isherm', False_122977)
        # SSA join for if statement (line 367)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'issymm' (line 369)
        issymm_122978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'issymm')
        # Getting the type of 'isskew' (line 369)
        isskew_122979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 30), 'isskew')
        # Applying the binary operator 'or' (line 369)
        result_or_keyword_122980 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 20), 'or', issymm_122978, isskew_122979)
        # Getting the type of 'isherm' (line 369)
        isherm_122981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 40), 'isherm')
        # Applying the binary operator 'or' (line 369)
        result_or_keyword_122982 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 20), 'or', result_or_keyword_122980, isherm_122981)
        
        # Applying the 'not' unary operator (line 369)
        result_not__122983 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 15), 'not', result_or_keyword_122982)
        
        # Testing the type of an if condition (line 369)
        if_condition_122984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 12), result_not__122983)
        # Assigning a type to the variable 'if_condition_122984' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'if_condition_122984', if_condition_122984)
        # SSA begins for if statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 369)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'issymm' (line 373)
        issymm_122985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), 'issymm')
        # Testing the type of an if condition (line 373)
        if_condition_122986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 8), issymm_122985)
        # Assigning a type to the variable 'if_condition_122986' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'if_condition_122986', if_condition_122986)
        # SSA begins for if statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'MMFile' (line 374)
        MMFile_122987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 19), 'MMFile')
        # Obtaining the member 'SYMMETRY_SYMMETRIC' of a type (line 374)
        SYMMETRY_SYMMETRIC_122988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 19), MMFile_122987, 'SYMMETRY_SYMMETRIC')
        # Assigning a type to the variable 'stypy_return_type' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'stypy_return_type', SYMMETRY_SYMMETRIC_122988)
        # SSA join for if statement (line 373)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'isskew' (line 375)
        isskew_122989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'isskew')
        # Testing the type of an if condition (line 375)
        if_condition_122990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), isskew_122989)
        # Assigning a type to the variable 'if_condition_122990' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_122990', if_condition_122990)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'MMFile' (line 376)
        MMFile_122991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'MMFile')
        # Obtaining the member 'SYMMETRY_SKEW_SYMMETRIC' of a type (line 376)
        SYMMETRY_SKEW_SYMMETRIC_122992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 19), MMFile_122991, 'SYMMETRY_SKEW_SYMMETRIC')
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', SYMMETRY_SKEW_SYMMETRIC_122992)
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'isherm' (line 377)
        isherm_122993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), 'isherm')
        # Testing the type of an if condition (line 377)
        if_condition_122994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 8), isherm_122993)
        # Assigning a type to the variable 'if_condition_122994' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'if_condition_122994', if_condition_122994)
        # SSA begins for if statement (line 377)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'MMFile' (line 378)
        MMFile_122995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 'MMFile')
        # Obtaining the member 'SYMMETRY_HERMITIAN' of a type (line 378)
        SYMMETRY_HERMITIAN_122996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 19), MMFile_122995, 'SYMMETRY_HERMITIAN')
        # Assigning a type to the variable 'stypy_return_type' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'stypy_return_type', SYMMETRY_HERMITIAN_122996)
        # SSA join for if statement (line 377)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'MMFile' (line 379)
        MMFile_122997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'MMFile')
        # Obtaining the member 'SYMMETRY_GENERAL' of a type (line 379)
        SYMMETRY_GENERAL_122998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), MMFile_122997, 'SYMMETRY_GENERAL')
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', SYMMETRY_GENERAL_122998)
        
        # ################# End of '_get_symmetry(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_symmetry' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_122999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_symmetry'
        return stypy_return_type_122999


    @staticmethod
    @norecursion
    def _field_template(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_field_template'
        module_type_store = module_type_store.open_function_context('_field_template', 382, 4, False)
        
        # Passed parameters checking function
        MMFile._field_template.__dict__.__setitem__('stypy_localization', localization)
        MMFile._field_template.__dict__.__setitem__('stypy_type_of_self', None)
        MMFile._field_template.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._field_template.__dict__.__setitem__('stypy_function_name', '_field_template')
        MMFile._field_template.__dict__.__setitem__('stypy_param_names_list', ['field', 'precision'])
        MMFile._field_template.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._field_template.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._field_template.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._field_template.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._field_template.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._field_template.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '_field_template', ['field', 'precision'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_field_template', localization, ['precision'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_field_template(...)' code ##################

        
        # Call to get(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'field' (line 388)
        field_123017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 22), 'field', False)
        # Getting the type of 'None' (line 388)
        None_123018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'None', False)
        # Processing the call keyword arguments (line 384)
        kwargs_123019 = {}
        
        # Obtaining an instance of the builtin type 'dict' (line 384)
        dict_123000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 384)
        # Adding element type (key, value) (line 384)
        # Getting the type of 'MMFile' (line 384)
        MMFile_123001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'MMFile', False)
        # Obtaining the member 'FIELD_REAL' of a type (line 384)
        FIELD_REAL_123002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 16), MMFile_123001, 'FIELD_REAL')
        str_123003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 35), 'str', '%%.%ie\n')
        # Getting the type of 'precision' (line 384)
        precision_123004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 48), 'precision', False)
        # Applying the binary operator '%' (line 384)
        result_mod_123005 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 35), '%', str_123003, precision_123004)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 15), dict_123000, (FIELD_REAL_123002, result_mod_123005))
        # Adding element type (key, value) (line 384)
        # Getting the type of 'MMFile' (line 385)
        MMFile_123006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'MMFile', False)
        # Obtaining the member 'FIELD_INTEGER' of a type (line 385)
        FIELD_INTEGER_123007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 16), MMFile_123006, 'FIELD_INTEGER')
        str_123008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 38), 'str', '%i\n')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 15), dict_123000, (FIELD_INTEGER_123007, str_123008))
        # Adding element type (key, value) (line 384)
        # Getting the type of 'MMFile' (line 386)
        MMFile_123009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'MMFile', False)
        # Obtaining the member 'FIELD_COMPLEX' of a type (line 386)
        FIELD_COMPLEX_123010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 16), MMFile_123009, 'FIELD_COMPLEX')
        str_123011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 38), 'str', '%%.%ie %%.%ie\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 387)
        tuple_123012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 387)
        # Adding element type (line 387)
        # Getting the type of 'precision' (line 387)
        precision_123013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 21), 'precision', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 21), tuple_123012, precision_123013)
        # Adding element type (line 387)
        # Getting the type of 'precision' (line 387)
        precision_123014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 'precision', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 21), tuple_123012, precision_123014)
        
        # Applying the binary operator '%' (line 386)
        result_mod_123015 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 38), '%', str_123011, tuple_123012)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 15), dict_123000, (FIELD_COMPLEX_123010, result_mod_123015))
        
        # Obtaining the member 'get' of a type (line 384)
        get_123016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 15), dict_123000, 'get')
        # Calling get(args, kwargs) (line 384)
        get_call_result_123020 = invoke(stypy.reporting.localization.Localization(__file__, 384, 15), get_123016, *[field_123017, None_123018], **kwargs_123019)
        
        # Assigning a type to the variable 'stypy_return_type' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'stypy_return_type', get_call_result_123020)
        
        # ################# End of '_field_template(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_field_template' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_123021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_field_template'
        return stypy_return_type_123021


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.__init__', [], None, 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to _init_attrs(...): (line 392)
        # Processing the call keyword arguments (line 392)
        # Getting the type of 'kwargs' (line 392)
        kwargs_123024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 27), 'kwargs', False)
        kwargs_123025 = {'kwargs_123024': kwargs_123024}
        # Getting the type of 'self' (line 392)
        self_123022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self', False)
        # Obtaining the member '_init_attrs' of a type (line 392)
        _init_attrs_123023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_123022, '_init_attrs')
        # Calling _init_attrs(args, kwargs) (line 392)
        _init_attrs_call_result_123026 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), _init_attrs_123023, *[], **kwargs_123025)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read'
        module_type_store = module_type_store.open_function_context('read', 395, 4, False)
        # Assigning a type to the variable 'self' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.read.__dict__.__setitem__('stypy_localization', localization)
        MMFile.read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.read.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.read.__dict__.__setitem__('stypy_function_name', 'MMFile.read')
        MMFile.read.__dict__.__setitem__('stypy_param_names_list', ['source'])
        MMFile.read.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.read.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.read.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.read.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.read', ['source'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read', localization, ['source'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read(...)' code ##################

        str_123027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, (-1)), 'str', "\n        Reads the contents of a Matrix Market file-like 'source' into a matrix.\n\n        Parameters\n        ----------\n        source : str or file-like\n            Matrix Market filename (extensions .mtx, .mtz.gz)\n            or open file object.\n\n        Returns\n        -------\n        a : ndarray or coo_matrix\n            Dense or sparse matrix depending on the matrix format in the\n            Matrix Market file.\n        ")
        
        # Assigning a Call to a Tuple (line 411):
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        int_123028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
        
        # Call to _open(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'source' (line 411)
        source_123031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 38), 'source', False)
        # Processing the call keyword arguments (line 411)
        kwargs_123032 = {}
        # Getting the type of 'self' (line 411)
        self_123029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 27), 'self', False)
        # Obtaining the member '_open' of a type (line 411)
        _open_123030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 27), self_123029, '_open')
        # Calling _open(args, kwargs) (line 411)
        _open_call_result_123033 = invoke(stypy.reporting.localization.Localization(__file__, 411, 27), _open_123030, *[source_123031], **kwargs_123032)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___123034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), _open_call_result_123033, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_123035 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), getitem___123034, int_123028)
        
        # Assigning a type to the variable 'tuple_var_assignment_122297' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_122297', subscript_call_result_123035)
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        int_123036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
        
        # Call to _open(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'source' (line 411)
        source_123039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 38), 'source', False)
        # Processing the call keyword arguments (line 411)
        kwargs_123040 = {}
        # Getting the type of 'self' (line 411)
        self_123037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 27), 'self', False)
        # Obtaining the member '_open' of a type (line 411)
        _open_123038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 27), self_123037, '_open')
        # Calling _open(args, kwargs) (line 411)
        _open_call_result_123041 = invoke(stypy.reporting.localization.Localization(__file__, 411, 27), _open_123038, *[source_123039], **kwargs_123040)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___123042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), _open_call_result_123041, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_123043 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), getitem___123042, int_123036)
        
        # Assigning a type to the variable 'tuple_var_assignment_122298' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_122298', subscript_call_result_123043)
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'tuple_var_assignment_122297' (line 411)
        tuple_var_assignment_122297_123044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_122297')
        # Assigning a type to the variable 'stream' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'stream', tuple_var_assignment_122297_123044)
        
        # Assigning a Name to a Name (line 411):
        # Getting the type of 'tuple_var_assignment_122298' (line 411)
        tuple_var_assignment_122298_123045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'tuple_var_assignment_122298')
        # Assigning a type to the variable 'close_it' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'close_it', tuple_var_assignment_122298_123045)
        
        # Try-finally block (line 413)
        
        # Call to _parse_header(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'stream' (line 414)
        stream_123048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 31), 'stream', False)
        # Processing the call keyword arguments (line 414)
        kwargs_123049 = {}
        # Getting the type of 'self' (line 414)
        self_123046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'self', False)
        # Obtaining the member '_parse_header' of a type (line 414)
        _parse_header_123047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), self_123046, '_parse_header')
        # Calling _parse_header(args, kwargs) (line 414)
        _parse_header_call_result_123050 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), _parse_header_123047, *[stream_123048], **kwargs_123049)
        
        
        # Call to _parse_body(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'stream' (line 415)
        stream_123053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 36), 'stream', False)
        # Processing the call keyword arguments (line 415)
        kwargs_123054 = {}
        # Getting the type of 'self' (line 415)
        self_123051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 19), 'self', False)
        # Obtaining the member '_parse_body' of a type (line 415)
        _parse_body_123052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 19), self_123051, '_parse_body')
        # Calling _parse_body(args, kwargs) (line 415)
        _parse_body_call_result_123055 = invoke(stypy.reporting.localization.Localization(__file__, 415, 19), _parse_body_123052, *[stream_123053], **kwargs_123054)
        
        # Assigning a type to the variable 'stypy_return_type' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'stypy_return_type', _parse_body_call_result_123055)
        
        # finally branch of the try-finally block (line 413)
        
        # Getting the type of 'close_it' (line 418)
        close_it_123056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'close_it')
        # Testing the type of an if condition (line 418)
        if_condition_123057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 12), close_it_123056)
        # Assigning a type to the variable 'if_condition_123057' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'if_condition_123057', if_condition_123057)
        # SSA begins for if statement (line 418)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 419)
        # Processing the call keyword arguments (line 419)
        kwargs_123060 = {}
        # Getting the type of 'stream' (line 419)
        stream_123058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'stream', False)
        # Obtaining the member 'close' of a type (line 419)
        close_123059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 16), stream_123058, 'close')
        # Calling close(args, kwargs) (line 419)
        close_call_result_123061 = invoke(stypy.reporting.localization.Localization(__file__, 419, 16), close_123059, *[], **kwargs_123060)
        
        # SSA join for if statement (line 418)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # ################# End of 'read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read' in the type store
        # Getting the type of 'stypy_return_type' (line 395)
        stypy_return_type_123062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read'
        return stypy_return_type_123062


    @norecursion
    def write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_123063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 39), 'str', '')
        # Getting the type of 'None' (line 422)
        None_123064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 49), 'None')
        # Getting the type of 'None' (line 422)
        None_123065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 65), 'None')
        # Getting the type of 'None' (line 423)
        None_123066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 23), 'None')
        defaults = [str_123063, None_123064, None_123065, None_123066]
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile.write.__dict__.__setitem__('stypy_localization', localization)
        MMFile.write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile.write.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile.write.__dict__.__setitem__('stypy_function_name', 'MMFile.write')
        MMFile.write.__dict__.__setitem__('stypy_param_names_list', ['target', 'a', 'comment', 'field', 'precision', 'symmetry'])
        MMFile.write.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile.write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile.write.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile.write.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile.write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile.write.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile.write', ['target', 'a', 'comment', 'field', 'precision', 'symmetry'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write', localization, ['target', 'a', 'comment', 'field', 'precision', 'symmetry'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write(...)' code ##################

        str_123067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, (-1)), 'str', "\n        Writes sparse or dense array `a` to Matrix Market file-like `target`.\n\n        Parameters\n        ----------\n        target : str or file-like\n            Matrix Market filename (extension .mtx) or open file-like object.\n        a : array like\n            Sparse or dense 2D array.\n        comment : str, optional\n            Comments to be prepended to the Matrix Market file.\n        field : None or str, optional\n            Either 'real', 'complex', 'pattern', or 'integer'.\n        precision : None or int, optional\n            Number of digits to display for real or complex values.\n        symmetry : None or str, optional\n            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n            If symmetry is None the symmetry type of 'a' is determined by its\n            values.\n        ")
        
        # Assigning a Call to a Tuple (line 445):
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_123068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        
        # Call to _open(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'target' (line 445)
        target_123071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 38), 'target', False)
        str_123072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 46), 'str', 'wb')
        # Processing the call keyword arguments (line 445)
        kwargs_123073 = {}
        # Getting the type of 'self' (line 445)
        self_123069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 27), 'self', False)
        # Obtaining the member '_open' of a type (line 445)
        _open_123070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 27), self_123069, '_open')
        # Calling _open(args, kwargs) (line 445)
        _open_call_result_123074 = invoke(stypy.reporting.localization.Localization(__file__, 445, 27), _open_123070, *[target_123071, str_123072], **kwargs_123073)
        
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___123075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), _open_call_result_123074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_123076 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___123075, int_123068)
        
        # Assigning a type to the variable 'tuple_var_assignment_122299' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_122299', subscript_call_result_123076)
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_123077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        
        # Call to _open(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'target' (line 445)
        target_123080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 38), 'target', False)
        str_123081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 46), 'str', 'wb')
        # Processing the call keyword arguments (line 445)
        kwargs_123082 = {}
        # Getting the type of 'self' (line 445)
        self_123078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 27), 'self', False)
        # Obtaining the member '_open' of a type (line 445)
        _open_123079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 27), self_123078, '_open')
        # Calling _open(args, kwargs) (line 445)
        _open_call_result_123083 = invoke(stypy.reporting.localization.Localization(__file__, 445, 27), _open_123079, *[target_123080, str_123081], **kwargs_123082)
        
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___123084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), _open_call_result_123083, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_123085 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___123084, int_123077)
        
        # Assigning a type to the variable 'tuple_var_assignment_122300' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_122300', subscript_call_result_123085)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_122299' (line 445)
        tuple_var_assignment_122299_123086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_122299')
        # Assigning a type to the variable 'stream' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'stream', tuple_var_assignment_122299_123086)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_122300' (line 445)
        tuple_var_assignment_122300_123087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_122300')
        # Assigning a type to the variable 'close_it' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'close_it', tuple_var_assignment_122300_123087)
        
        # Try-finally block (line 447)
        
        # Call to _write(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'stream' (line 448)
        stream_123090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'stream', False)
        # Getting the type of 'a' (line 448)
        a_123091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 32), 'a', False)
        # Getting the type of 'comment' (line 448)
        comment_123092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 35), 'comment', False)
        # Getting the type of 'field' (line 448)
        field_123093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 44), 'field', False)
        # Getting the type of 'precision' (line 448)
        precision_123094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 51), 'precision', False)
        # Getting the type of 'symmetry' (line 448)
        symmetry_123095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 62), 'symmetry', False)
        # Processing the call keyword arguments (line 448)
        kwargs_123096 = {}
        # Getting the type of 'self' (line 448)
        self_123088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'self', False)
        # Obtaining the member '_write' of a type (line 448)
        _write_123089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), self_123088, '_write')
        # Calling _write(args, kwargs) (line 448)
        _write_call_result_123097 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), _write_123089, *[stream_123090, a_123091, comment_123092, field_123093, precision_123094, symmetry_123095], **kwargs_123096)
        
        
        # finally branch of the try-finally block (line 447)
        
        # Getting the type of 'close_it' (line 451)
        close_it_123098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 15), 'close_it')
        # Testing the type of an if condition (line 451)
        if_condition_123099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 12), close_it_123098)
        # Assigning a type to the variable 'if_condition_123099' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'if_condition_123099', if_condition_123099)
        # SSA begins for if statement (line 451)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 452)
        # Processing the call keyword arguments (line 452)
        kwargs_123102 = {}
        # Getting the type of 'stream' (line 452)
        stream_123100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'stream', False)
        # Obtaining the member 'close' of a type (line 452)
        close_123101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 16), stream_123100, 'close')
        # Calling close(args, kwargs) (line 452)
        close_call_result_123103 = invoke(stypy.reporting.localization.Localization(__file__, 452, 16), close_123101, *[], **kwargs_123102)
        
        # SSA branch for the else part of an if statement (line 451)
        module_type_store.open_ssa_branch('else')
        
        # Call to flush(...): (line 454)
        # Processing the call keyword arguments (line 454)
        kwargs_123106 = {}
        # Getting the type of 'stream' (line 454)
        stream_123104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'stream', False)
        # Obtaining the member 'flush' of a type (line 454)
        flush_123105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 16), stream_123104, 'flush')
        # Calling flush(args, kwargs) (line 454)
        flush_call_result_123107 = invoke(stypy.reporting.localization.Localization(__file__, 454, 16), flush_123105, *[], **kwargs_123106)
        
        # SSA join for if statement (line 451)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_123108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123108)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_123108


    @norecursion
    def _init_attrs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_attrs'
        module_type_store = module_type_store.open_function_context('_init_attrs', 457, 4, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile._init_attrs.__dict__.__setitem__('stypy_localization', localization)
        MMFile._init_attrs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile._init_attrs.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._init_attrs.__dict__.__setitem__('stypy_function_name', 'MMFile._init_attrs')
        MMFile._init_attrs.__dict__.__setitem__('stypy_param_names_list', [])
        MMFile._init_attrs.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._init_attrs.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        MMFile._init_attrs.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._init_attrs.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._init_attrs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._init_attrs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile._init_attrs', [], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_attrs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_attrs(...)' code ##################

        str_123109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, (-1)), 'str', '\n        Initialize each attributes with the corresponding keyword arg value\n        or a default of None\n        ')
        
        # Assigning a Attribute to a Name (line 463):
        
        # Assigning a Attribute to a Name (line 463):
        # Getting the type of 'self' (line 463)
        self_123110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'self')
        # Obtaining the member '__class__' of a type (line 463)
        class___123111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 16), self_123110, '__class__')
        # Obtaining the member '__slots__' of a type (line 463)
        slots___123112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 16), class___123111, '__slots__')
        # Assigning a type to the variable 'attrs' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'attrs', slots___123112)
        
        # Assigning a ListComp to a Name (line 464):
        
        # Assigning a ListComp to a Name (line 464):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'attrs' (line 464)
        attrs_123118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 45), 'attrs')
        comprehension_123119 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 24), attrs_123118)
        # Assigning a type to the variable 'attr' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'attr', comprehension_123119)
        
        # Obtaining the type of the subscript
        int_123113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 29), 'int')
        slice_123114 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 464, 24), int_123113, None, None)
        # Getting the type of 'attr' (line 464)
        attr_123115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'attr')
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___123116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 24), attr_123115, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 464)
        subscript_call_result_123117 = invoke(stypy.reporting.localization.Localization(__file__, 464, 24), getitem___123116, slice_123114)
        
        list_123120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 24), list_123120, subscript_call_result_123117)
        # Assigning a type to the variable 'public_attrs' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'public_attrs', list_123120)
        
        # Assigning a BinOp to a Name (line 465):
        
        # Assigning a BinOp to a Name (line 465):
        
        # Call to set(...): (line 465)
        # Processing the call arguments (line 465)
        
        # Call to keys(...): (line 465)
        # Processing the call keyword arguments (line 465)
        kwargs_123124 = {}
        # Getting the type of 'kwargs' (line 465)
        kwargs_123122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 27), 'kwargs', False)
        # Obtaining the member 'keys' of a type (line 465)
        keys_123123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 27), kwargs_123122, 'keys')
        # Calling keys(args, kwargs) (line 465)
        keys_call_result_123125 = invoke(stypy.reporting.localization.Localization(__file__, 465, 27), keys_123123, *[], **kwargs_123124)
        
        # Processing the call keyword arguments (line 465)
        kwargs_123126 = {}
        # Getting the type of 'set' (line 465)
        set_123121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 23), 'set', False)
        # Calling set(args, kwargs) (line 465)
        set_call_result_123127 = invoke(stypy.reporting.localization.Localization(__file__, 465, 23), set_123121, *[keys_call_result_123125], **kwargs_123126)
        
        
        # Call to set(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'public_attrs' (line 465)
        public_attrs_123129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 48), 'public_attrs', False)
        # Processing the call keyword arguments (line 465)
        kwargs_123130 = {}
        # Getting the type of 'set' (line 465)
        set_123128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 44), 'set', False)
        # Calling set(args, kwargs) (line 465)
        set_call_result_123131 = invoke(stypy.reporting.localization.Localization(__file__, 465, 44), set_123128, *[public_attrs_123129], **kwargs_123130)
        
        # Applying the binary operator '-' (line 465)
        result_sub_123132 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 23), '-', set_call_result_123127, set_call_result_123131)
        
        # Assigning a type to the variable 'invalid_keys' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'invalid_keys', result_sub_123132)
        
        # Getting the type of 'invalid_keys' (line 467)
        invalid_keys_123133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'invalid_keys')
        # Testing the type of an if condition (line 467)
        if_condition_123134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 8), invalid_keys_123133)
        # Assigning a type to the variable 'if_condition_123134' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'if_condition_123134', if_condition_123134)
        # SSA begins for if statement (line 467)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 468)
        # Processing the call arguments (line 468)
        str_123136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, (-1)), 'str', 'found %s invalid keyword arguments, please only\n                                use %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 469)
        tuple_123137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 469)
        # Adding element type (line 469)
        
        # Call to tuple(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'invalid_keys' (line 469)
        invalid_keys_123139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 51), 'invalid_keys', False)
        # Processing the call keyword arguments (line 469)
        kwargs_123140 = {}
        # Getting the type of 'tuple' (line 469)
        tuple_123138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 45), 'tuple', False)
        # Calling tuple(args, kwargs) (line 469)
        tuple_call_result_123141 = invoke(stypy.reporting.localization.Localization(__file__, 469, 45), tuple_123138, *[invalid_keys_123139], **kwargs_123140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 45), tuple_123137, tuple_call_result_123141)
        # Adding element type (line 469)
        # Getting the type of 'public_attrs' (line 470)
        public_attrs_123142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 45), 'public_attrs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 45), tuple_123137, public_attrs_123142)
        
        # Applying the binary operator '%' (line 469)
        result_mod_123143 = python_operator(stypy.reporting.localization.Localization(__file__, 469, (-1)), '%', str_123136, tuple_123137)
        
        # Processing the call keyword arguments (line 468)
        kwargs_123144 = {}
        # Getting the type of 'ValueError' (line 468)
        ValueError_123135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 468)
        ValueError_call_result_123145 = invoke(stypy.reporting.localization.Localization(__file__, 468, 18), ValueError_123135, *[result_mod_123143], **kwargs_123144)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 468, 12), ValueError_call_result_123145, 'raise parameter', BaseException)
        # SSA join for if statement (line 467)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'attrs' (line 472)
        attrs_123146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 20), 'attrs')
        # Testing the type of a for loop iterable (line 472)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 472, 8), attrs_123146)
        # Getting the type of the for loop variable (line 472)
        for_loop_var_123147 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 472, 8), attrs_123146)
        # Assigning a type to the variable 'attr' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'attr', for_loop_var_123147)
        # SSA begins for a for statement (line 472)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'self' (line 473)
        self_123149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 20), 'self', False)
        # Getting the type of 'attr' (line 473)
        attr_123150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 26), 'attr', False)
        
        # Call to get(...): (line 473)
        # Processing the call arguments (line 473)
        
        # Obtaining the type of the subscript
        int_123153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 48), 'int')
        slice_123154 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 473, 43), int_123153, None, None)
        # Getting the type of 'attr' (line 473)
        attr_123155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 43), 'attr', False)
        # Obtaining the member '__getitem__' of a type (line 473)
        getitem___123156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 43), attr_123155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 473)
        subscript_call_result_123157 = invoke(stypy.reporting.localization.Localization(__file__, 473, 43), getitem___123156, slice_123154)
        
        # Getting the type of 'None' (line 473)
        None_123158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 53), 'None', False)
        # Processing the call keyword arguments (line 473)
        kwargs_123159 = {}
        # Getting the type of 'kwargs' (line 473)
        kwargs_123151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 32), 'kwargs', False)
        # Obtaining the member 'get' of a type (line 473)
        get_123152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 32), kwargs_123151, 'get')
        # Calling get(args, kwargs) (line 473)
        get_call_result_123160 = invoke(stypy.reporting.localization.Localization(__file__, 473, 32), get_123152, *[subscript_call_result_123157, None_123158], **kwargs_123159)
        
        # Processing the call keyword arguments (line 473)
        kwargs_123161 = {}
        # Getting the type of 'setattr' (line 473)
        setattr_123148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 473)
        setattr_call_result_123162 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), setattr_123148, *[self_123149, attr_123150, get_call_result_123160], **kwargs_123161)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_init_attrs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_attrs' in the type store
        # Getting the type of 'stypy_return_type' (line 457)
        stypy_return_type_123163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123163)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_attrs'
        return stypy_return_type_123163


    @norecursion
    def _parse_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse_header'
        module_type_store = module_type_store.open_function_context('_parse_header', 476, 4, False)
        # Assigning a type to the variable 'self' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile._parse_header.__dict__.__setitem__('stypy_localization', localization)
        MMFile._parse_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile._parse_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._parse_header.__dict__.__setitem__('stypy_function_name', 'MMFile._parse_header')
        MMFile._parse_header.__dict__.__setitem__('stypy_param_names_list', ['stream'])
        MMFile._parse_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._parse_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._parse_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._parse_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._parse_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._parse_header.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile._parse_header', ['stream'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_header', localization, ['stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_header(...)' code ##################

        
        # Assigning a Call to a Tuple (line 477):
        
        # Assigning a Subscript to a Name (line 477):
        
        # Obtaining the type of the subscript
        int_123164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 8), 'int')
        
        # Call to info(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'stream' (line 478)
        stream_123168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'stream', False)
        # Processing the call keyword arguments (line 478)
        kwargs_123169 = {}
        # Getting the type of 'self' (line 478)
        self_123165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'self', False)
        # Obtaining the member '__class__' of a type (line 478)
        class___123166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), self_123165, '__class__')
        # Obtaining the member 'info' of a type (line 478)
        info_123167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), class___123166, 'info')
        # Calling info(args, kwargs) (line 478)
        info_call_result_123170 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), info_123167, *[stream_123168], **kwargs_123169)
        
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___123171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), info_call_result_123170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_123172 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___123171, int_123164)
        
        # Assigning a type to the variable 'tuple_var_assignment_122301' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122301', subscript_call_result_123172)
        
        # Assigning a Subscript to a Name (line 477):
        
        # Obtaining the type of the subscript
        int_123173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 8), 'int')
        
        # Call to info(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'stream' (line 478)
        stream_123177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'stream', False)
        # Processing the call keyword arguments (line 478)
        kwargs_123178 = {}
        # Getting the type of 'self' (line 478)
        self_123174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'self', False)
        # Obtaining the member '__class__' of a type (line 478)
        class___123175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), self_123174, '__class__')
        # Obtaining the member 'info' of a type (line 478)
        info_123176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), class___123175, 'info')
        # Calling info(args, kwargs) (line 478)
        info_call_result_123179 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), info_123176, *[stream_123177], **kwargs_123178)
        
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___123180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), info_call_result_123179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_123181 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___123180, int_123173)
        
        # Assigning a type to the variable 'tuple_var_assignment_122302' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122302', subscript_call_result_123181)
        
        # Assigning a Subscript to a Name (line 477):
        
        # Obtaining the type of the subscript
        int_123182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 8), 'int')
        
        # Call to info(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'stream' (line 478)
        stream_123186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'stream', False)
        # Processing the call keyword arguments (line 478)
        kwargs_123187 = {}
        # Getting the type of 'self' (line 478)
        self_123183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'self', False)
        # Obtaining the member '__class__' of a type (line 478)
        class___123184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), self_123183, '__class__')
        # Obtaining the member 'info' of a type (line 478)
        info_123185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), class___123184, 'info')
        # Calling info(args, kwargs) (line 478)
        info_call_result_123188 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), info_123185, *[stream_123186], **kwargs_123187)
        
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___123189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), info_call_result_123188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_123190 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___123189, int_123182)
        
        # Assigning a type to the variable 'tuple_var_assignment_122303' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122303', subscript_call_result_123190)
        
        # Assigning a Subscript to a Name (line 477):
        
        # Obtaining the type of the subscript
        int_123191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 8), 'int')
        
        # Call to info(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'stream' (line 478)
        stream_123195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'stream', False)
        # Processing the call keyword arguments (line 478)
        kwargs_123196 = {}
        # Getting the type of 'self' (line 478)
        self_123192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'self', False)
        # Obtaining the member '__class__' of a type (line 478)
        class___123193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), self_123192, '__class__')
        # Obtaining the member 'info' of a type (line 478)
        info_123194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), class___123193, 'info')
        # Calling info(args, kwargs) (line 478)
        info_call_result_123197 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), info_123194, *[stream_123195], **kwargs_123196)
        
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___123198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), info_call_result_123197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_123199 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___123198, int_123191)
        
        # Assigning a type to the variable 'tuple_var_assignment_122304' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122304', subscript_call_result_123199)
        
        # Assigning a Subscript to a Name (line 477):
        
        # Obtaining the type of the subscript
        int_123200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 8), 'int')
        
        # Call to info(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'stream' (line 478)
        stream_123204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'stream', False)
        # Processing the call keyword arguments (line 478)
        kwargs_123205 = {}
        # Getting the type of 'self' (line 478)
        self_123201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'self', False)
        # Obtaining the member '__class__' of a type (line 478)
        class___123202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), self_123201, '__class__')
        # Obtaining the member 'info' of a type (line 478)
        info_123203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), class___123202, 'info')
        # Calling info(args, kwargs) (line 478)
        info_call_result_123206 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), info_123203, *[stream_123204], **kwargs_123205)
        
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___123207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), info_call_result_123206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_123208 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___123207, int_123200)
        
        # Assigning a type to the variable 'tuple_var_assignment_122305' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122305', subscript_call_result_123208)
        
        # Assigning a Subscript to a Name (line 477):
        
        # Obtaining the type of the subscript
        int_123209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 8), 'int')
        
        # Call to info(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'stream' (line 478)
        stream_123213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'stream', False)
        # Processing the call keyword arguments (line 478)
        kwargs_123214 = {}
        # Getting the type of 'self' (line 478)
        self_123210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'self', False)
        # Obtaining the member '__class__' of a type (line 478)
        class___123211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), self_123210, '__class__')
        # Obtaining the member 'info' of a type (line 478)
        info_123212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), class___123211, 'info')
        # Calling info(args, kwargs) (line 478)
        info_call_result_123215 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), info_123212, *[stream_123213], **kwargs_123214)
        
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___123216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), info_call_result_123215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_123217 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), getitem___123216, int_123209)
        
        # Assigning a type to the variable 'tuple_var_assignment_122306' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122306', subscript_call_result_123217)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_var_assignment_122301' (line 477)
        tuple_var_assignment_122301_123218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122301')
        # Assigning a type to the variable 'rows' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'rows', tuple_var_assignment_122301_123218)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_var_assignment_122302' (line 477)
        tuple_var_assignment_122302_123219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122302')
        # Assigning a type to the variable 'cols' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 14), 'cols', tuple_var_assignment_122302_123219)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_var_assignment_122303' (line 477)
        tuple_var_assignment_122303_123220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122303')
        # Assigning a type to the variable 'entries' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 20), 'entries', tuple_var_assignment_122303_123220)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_var_assignment_122304' (line 477)
        tuple_var_assignment_122304_123221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122304')
        # Assigning a type to the variable 'format' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 29), 'format', tuple_var_assignment_122304_123221)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_var_assignment_122305' (line 477)
        tuple_var_assignment_122305_123222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122305')
        # Assigning a type to the variable 'field' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 37), 'field', tuple_var_assignment_122305_123222)
        
        # Assigning a Name to a Name (line 477):
        # Getting the type of 'tuple_var_assignment_122306' (line 477)
        tuple_var_assignment_122306_123223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'tuple_var_assignment_122306')
        # Assigning a type to the variable 'symmetry' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 44), 'symmetry', tuple_var_assignment_122306_123223)
        
        # Call to _init_attrs(...): (line 479)
        # Processing the call keyword arguments (line 479)
        # Getting the type of 'rows' (line 479)
        rows_123226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 30), 'rows', False)
        keyword_123227 = rows_123226
        # Getting the type of 'cols' (line 479)
        cols_123228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 41), 'cols', False)
        keyword_123229 = cols_123228
        # Getting the type of 'entries' (line 479)
        entries_123230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 55), 'entries', False)
        keyword_123231 = entries_123230
        # Getting the type of 'format' (line 479)
        format_123232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 71), 'format', False)
        keyword_123233 = format_123232
        # Getting the type of 'field' (line 480)
        field_123234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 31), 'field', False)
        keyword_123235 = field_123234
        # Getting the type of 'symmetry' (line 480)
        symmetry_123236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 47), 'symmetry', False)
        keyword_123237 = symmetry_123236
        kwargs_123238 = {'rows': keyword_123227, 'symmetry': keyword_123237, 'format': keyword_123233, 'cols': keyword_123229, 'field': keyword_123235, 'entries': keyword_123231}
        # Getting the type of 'self' (line 479)
        self_123224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'self', False)
        # Obtaining the member '_init_attrs' of a type (line 479)
        _init_attrs_123225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), self_123224, '_init_attrs')
        # Calling _init_attrs(args, kwargs) (line 479)
        _init_attrs_call_result_123239 = invoke(stypy.reporting.localization.Localization(__file__, 479, 8), _init_attrs_123225, *[], **kwargs_123238)
        
        
        # ################# End of '_parse_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_header' in the type store
        # Getting the type of 'stypy_return_type' (line 476)
        stypy_return_type_123240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123240)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_header'
        return stypy_return_type_123240


    @norecursion
    def _parse_body(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse_body'
        module_type_store = module_type_store.open_function_context('_parse_body', 483, 4, False)
        # Assigning a type to the variable 'self' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile._parse_body.__dict__.__setitem__('stypy_localization', localization)
        MMFile._parse_body.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile._parse_body.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._parse_body.__dict__.__setitem__('stypy_function_name', 'MMFile._parse_body')
        MMFile._parse_body.__dict__.__setitem__('stypy_param_names_list', ['stream'])
        MMFile._parse_body.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._parse_body.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._parse_body.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._parse_body.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._parse_body.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._parse_body.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile._parse_body', ['stream'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_body', localization, ['stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_body(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 484):
        
        # Assigning a Attribute to a Name (line 484):
        # Getting the type of 'self' (line 484)
        self_123241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 52), 'self')
        # Obtaining the member 'rows' of a type (line 484)
        rows_123242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 52), self_123241, 'rows')
        # Assigning a type to the variable 'tuple_assignment_122307' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122307', rows_123242)
        
        # Assigning a Attribute to a Name (line 484):
        # Getting the type of 'self' (line 484)
        self_123243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 63), 'self')
        # Obtaining the member 'cols' of a type (line 484)
        cols_123244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 63), self_123243, 'cols')
        # Assigning a type to the variable 'tuple_assignment_122308' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122308', cols_123244)
        
        # Assigning a Attribute to a Name (line 484):
        # Getting the type of 'self' (line 485)
        self_123245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 52), 'self')
        # Obtaining the member 'entries' of a type (line 485)
        entries_123246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 52), self_123245, 'entries')
        # Assigning a type to the variable 'tuple_assignment_122309' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122309', entries_123246)
        
        # Assigning a Attribute to a Name (line 484):
        # Getting the type of 'self' (line 485)
        self_123247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 66), 'self')
        # Obtaining the member 'format' of a type (line 485)
        format_123248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 66), self_123247, 'format')
        # Assigning a type to the variable 'tuple_assignment_122310' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122310', format_123248)
        
        # Assigning a Attribute to a Name (line 484):
        # Getting the type of 'self' (line 486)
        self_123249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 52), 'self')
        # Obtaining the member 'field' of a type (line 486)
        field_123250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 52), self_123249, 'field')
        # Assigning a type to the variable 'tuple_assignment_122311' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122311', field_123250)
        
        # Assigning a Attribute to a Name (line 484):
        # Getting the type of 'self' (line 486)
        self_123251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 64), 'self')
        # Obtaining the member 'symmetry' of a type (line 486)
        symmetry_123252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 64), self_123251, 'symmetry')
        # Assigning a type to the variable 'tuple_assignment_122312' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122312', symmetry_123252)
        
        # Assigning a Name to a Name (line 484):
        # Getting the type of 'tuple_assignment_122307' (line 484)
        tuple_assignment_122307_123253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122307')
        # Assigning a type to the variable 'rows' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'rows', tuple_assignment_122307_123253)
        
        # Assigning a Name to a Name (line 484):
        # Getting the type of 'tuple_assignment_122308' (line 484)
        tuple_assignment_122308_123254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122308')
        # Assigning a type to the variable 'cols' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 14), 'cols', tuple_assignment_122308_123254)
        
        # Assigning a Name to a Name (line 484):
        # Getting the type of 'tuple_assignment_122309' (line 484)
        tuple_assignment_122309_123255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122309')
        # Assigning a type to the variable 'entries' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 20), 'entries', tuple_assignment_122309_123255)
        
        # Assigning a Name to a Name (line 484):
        # Getting the type of 'tuple_assignment_122310' (line 484)
        tuple_assignment_122310_123256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122310')
        # Assigning a type to the variable 'format' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 29), 'format', tuple_assignment_122310_123256)
        
        # Assigning a Name to a Name (line 484):
        # Getting the type of 'tuple_assignment_122311' (line 484)
        tuple_assignment_122311_123257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122311')
        # Assigning a type to the variable 'field' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 37), 'field', tuple_assignment_122311_123257)
        
        # Assigning a Name to a Name (line 484):
        # Getting the type of 'tuple_assignment_122312' (line 484)
        tuple_assignment_122312_123258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'tuple_assignment_122312')
        # Assigning a type to the variable 'symm' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 44), 'symm', tuple_assignment_122312_123258)
        
        
        # SSA begins for try-except statement (line 488)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 489, 12))
        
        # 'from scipy.sparse import coo_matrix' statement (line 489)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
        import_123259 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 489, 12), 'scipy.sparse')

        if (type(import_123259) is not StypyTypeError):

            if (import_123259 != 'pyd_module'):
                __import__(import_123259)
                sys_modules_123260 = sys.modules[import_123259]
                import_from_module(stypy.reporting.localization.Localization(__file__, 489, 12), 'scipy.sparse', sys_modules_123260.module_type_store, module_type_store, ['coo_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 489, 12), __file__, sys_modules_123260, sys_modules_123260.module_type_store, module_type_store)
            else:
                from scipy.sparse import coo_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 489, 12), 'scipy.sparse', None, module_type_store, ['coo_matrix'], [coo_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse' (line 489)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'scipy.sparse', import_123259)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')
        
        # SSA branch for the except part of a try statement (line 488)
        # SSA branch for the except 'ImportError' branch of a try statement (line 488)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 491):
        
        # Assigning a Name to a Name (line 491):
        # Getting the type of 'None' (line 491)
        None_123261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 25), 'None')
        # Assigning a type to the variable 'coo_matrix' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'coo_matrix', None_123261)
        # SSA join for try-except statement (line 488)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to get(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'field' (line 493)
        field_123265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 41), 'field', False)
        # Getting the type of 'None' (line 493)
        None_123266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 48), 'None', False)
        # Processing the call keyword arguments (line 493)
        kwargs_123267 = {}
        # Getting the type of 'self' (line 493)
        self_123262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 16), 'self', False)
        # Obtaining the member 'DTYPES_BY_FIELD' of a type (line 493)
        DTYPES_BY_FIELD_123263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 16), self_123262, 'DTYPES_BY_FIELD')
        # Obtaining the member 'get' of a type (line 493)
        get_123264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 16), DTYPES_BY_FIELD_123263, 'get')
        # Calling get(args, kwargs) (line 493)
        get_call_result_123268 = invoke(stypy.reporting.localization.Localization(__file__, 493, 16), get_123264, *[field_123265, None_123266], **kwargs_123267)
        
        # Assigning a type to the variable 'dtype' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'dtype', get_call_result_123268)
        
        # Assigning a Attribute to a Name (line 495):
        
        # Assigning a Attribute to a Name (line 495):
        # Getting the type of 'self' (line 495)
        self_123269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 23), 'self')
        # Obtaining the member 'has_symmetry' of a type (line 495)
        has_symmetry_123270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 23), self_123269, 'has_symmetry')
        # Assigning a type to the variable 'has_symmetry' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'has_symmetry', has_symmetry_123270)
        
        # Assigning a Compare to a Name (line 496):
        
        # Assigning a Compare to a Name (line 496):
        
        # Getting the type of 'field' (line 496)
        field_123271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 21), 'field')
        # Getting the type of 'self' (line 496)
        self_123272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 30), 'self')
        # Obtaining the member 'FIELD_INTEGER' of a type (line 496)
        FIELD_INTEGER_123273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 30), self_123272, 'FIELD_INTEGER')
        # Applying the binary operator '==' (line 496)
        result_eq_123274 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 21), '==', field_123271, FIELD_INTEGER_123273)
        
        # Assigning a type to the variable 'is_integer' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'is_integer', result_eq_123274)
        
        # Assigning a Compare to a Name (line 497):
        
        # Assigning a Compare to a Name (line 497):
        
        # Getting the type of 'field' (line 497)
        field_123275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 21), 'field')
        # Getting the type of 'self' (line 497)
        self_123276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 30), 'self')
        # Obtaining the member 'FIELD_COMPLEX' of a type (line 497)
        FIELD_COMPLEX_123277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 30), self_123276, 'FIELD_COMPLEX')
        # Applying the binary operator '==' (line 497)
        result_eq_123278 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 21), '==', field_123275, FIELD_COMPLEX_123277)
        
        # Assigning a type to the variable 'is_complex' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'is_complex', result_eq_123278)
        
        # Assigning a Compare to a Name (line 498):
        
        # Assigning a Compare to a Name (line 498):
        
        # Getting the type of 'symm' (line 498)
        symm_123279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 18), 'symm')
        # Getting the type of 'self' (line 498)
        self_123280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 26), 'self')
        # Obtaining the member 'SYMMETRY_SKEW_SYMMETRIC' of a type (line 498)
        SYMMETRY_SKEW_SYMMETRIC_123281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 26), self_123280, 'SYMMETRY_SKEW_SYMMETRIC')
        # Applying the binary operator '==' (line 498)
        result_eq_123282 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 18), '==', symm_123279, SYMMETRY_SKEW_SYMMETRIC_123281)
        
        # Assigning a type to the variable 'is_skew' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'is_skew', result_eq_123282)
        
        # Assigning a Compare to a Name (line 499):
        
        # Assigning a Compare to a Name (line 499):
        
        # Getting the type of 'symm' (line 499)
        symm_123283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 18), 'symm')
        # Getting the type of 'self' (line 499)
        self_123284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 26), 'self')
        # Obtaining the member 'SYMMETRY_HERMITIAN' of a type (line 499)
        SYMMETRY_HERMITIAN_123285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 26), self_123284, 'SYMMETRY_HERMITIAN')
        # Applying the binary operator '==' (line 499)
        result_eq_123286 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 18), '==', symm_123283, SYMMETRY_HERMITIAN_123285)
        
        # Assigning a type to the variable 'is_herm' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'is_herm', result_eq_123286)
        
        # Assigning a Compare to a Name (line 500):
        
        # Assigning a Compare to a Name (line 500):
        
        # Getting the type of 'field' (line 500)
        field_123287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 21), 'field')
        # Getting the type of 'self' (line 500)
        self_123288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 30), 'self')
        # Obtaining the member 'FIELD_PATTERN' of a type (line 500)
        FIELD_PATTERN_123289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 30), self_123288, 'FIELD_PATTERN')
        # Applying the binary operator '==' (line 500)
        result_eq_123290 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 21), '==', field_123287, FIELD_PATTERN_123289)
        
        # Assigning a type to the variable 'is_pattern' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'is_pattern', result_eq_123290)
        
        
        # Getting the type of 'format' (line 502)
        format_123291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'format')
        # Getting the type of 'self' (line 502)
        self_123292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 21), 'self')
        # Obtaining the member 'FORMAT_ARRAY' of a type (line 502)
        FORMAT_ARRAY_123293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 21), self_123292, 'FORMAT_ARRAY')
        # Applying the binary operator '==' (line 502)
        result_eq_123294 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 11), '==', format_123291, FORMAT_ARRAY_123293)
        
        # Testing the type of an if condition (line 502)
        if_condition_123295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 8), result_eq_123294)
        # Assigning a type to the variable 'if_condition_123295' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'if_condition_123295', if_condition_123295)
        # SSA begins for if statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 503):
        
        # Assigning a Call to a Name (line 503):
        
        # Call to zeros(...): (line 503)
        # Processing the call arguments (line 503)
        
        # Obtaining an instance of the builtin type 'tuple' (line 503)
        tuple_123297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 503)
        # Adding element type (line 503)
        # Getting the type of 'rows' (line 503)
        rows_123298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 23), 'rows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 23), tuple_123297, rows_123298)
        # Adding element type (line 503)
        # Getting the type of 'cols' (line 503)
        cols_123299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 29), 'cols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 23), tuple_123297, cols_123299)
        
        # Processing the call keyword arguments (line 503)
        # Getting the type of 'dtype' (line 503)
        dtype_123300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 42), 'dtype', False)
        keyword_123301 = dtype_123300
        kwargs_123302 = {'dtype': keyword_123301}
        # Getting the type of 'zeros' (line 503)
        zeros_123296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 503)
        zeros_call_result_123303 = invoke(stypy.reporting.localization.Localization(__file__, 503, 16), zeros_123296, *[tuple_123297], **kwargs_123302)
        
        # Assigning a type to the variable 'a' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'a', zeros_call_result_123303)
        
        # Assigning a Num to a Name (line 504):
        
        # Assigning a Num to a Name (line 504):
        int_123304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 19), 'int')
        # Assigning a type to the variable 'line' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'line', int_123304)
        
        # Assigning a Tuple to a Tuple (line 505):
        
        # Assigning a Num to a Name (line 505):
        int_123305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 19), 'int')
        # Assigning a type to the variable 'tuple_assignment_122313' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_assignment_122313', int_123305)
        
        # Assigning a Num to a Name (line 505):
        int_123306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 22), 'int')
        # Assigning a type to the variable 'tuple_assignment_122314' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_assignment_122314', int_123306)
        
        # Assigning a Name to a Name (line 505):
        # Getting the type of 'tuple_assignment_122313' (line 505)
        tuple_assignment_122313_123307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_assignment_122313')
        # Assigning a type to the variable 'i' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'i', tuple_assignment_122313_123307)
        
        # Assigning a Name to a Name (line 505):
        # Getting the type of 'tuple_assignment_122314' (line 505)
        tuple_assignment_122314_123308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'tuple_assignment_122314')
        # Assigning a type to the variable 'j' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 15), 'j', tuple_assignment_122314_123308)
        
        # Getting the type of 'line' (line 506)
        line_123309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 18), 'line')
        # Testing the type of an if condition (line 506)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 506, 12), line_123309)
        # SSA begins for while statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 507):
        
        # Assigning a Call to a Name (line 507):
        
        # Call to readline(...): (line 507)
        # Processing the call keyword arguments (line 507)
        kwargs_123312 = {}
        # Getting the type of 'stream' (line 507)
        stream_123310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 23), 'stream', False)
        # Obtaining the member 'readline' of a type (line 507)
        readline_123311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 23), stream_123310, 'readline')
        # Calling readline(args, kwargs) (line 507)
        readline_call_result_123313 = invoke(stypy.reporting.localization.Localization(__file__, 507, 23), readline_123311, *[], **kwargs_123312)
        
        # Assigning a type to the variable 'line' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'line', readline_call_result_123313)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'line' (line 508)
        line_123314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 23), 'line')
        # Applying the 'not' unary operator (line 508)
        result_not__123315 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 19), 'not', line_123314)
        
        
        # Call to startswith(...): (line 508)
        # Processing the call arguments (line 508)
        str_123318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 47), 'str', '%')
        # Processing the call keyword arguments (line 508)
        kwargs_123319 = {}
        # Getting the type of 'line' (line 508)
        line_123316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 31), 'line', False)
        # Obtaining the member 'startswith' of a type (line 508)
        startswith_123317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 31), line_123316, 'startswith')
        # Calling startswith(args, kwargs) (line 508)
        startswith_call_result_123320 = invoke(stypy.reporting.localization.Localization(__file__, 508, 31), startswith_123317, *[str_123318], **kwargs_123319)
        
        # Applying the binary operator 'or' (line 508)
        result_or_keyword_123321 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 19), 'or', result_not__123315, startswith_call_result_123320)
        
        # Testing the type of an if condition (line 508)
        if_condition_123322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 508, 16), result_or_keyword_123321)
        # Assigning a type to the variable 'if_condition_123322' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'if_condition_123322', if_condition_123322)
        # SSA begins for if statement (line 508)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 508)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'is_integer' (line 510)
        is_integer_123323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), 'is_integer')
        # Testing the type of an if condition (line 510)
        if_condition_123324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 16), is_integer_123323)
        # Assigning a type to the variable 'if_condition_123324' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'if_condition_123324', if_condition_123324)
        # SSA begins for if statement (line 510)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 511):
        
        # Assigning a Call to a Name (line 511):
        
        # Call to int(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'line' (line 511)
        line_123326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 30), 'line', False)
        # Processing the call keyword arguments (line 511)
        kwargs_123327 = {}
        # Getting the type of 'int' (line 511)
        int_123325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 26), 'int', False)
        # Calling int(args, kwargs) (line 511)
        int_call_result_123328 = invoke(stypy.reporting.localization.Localization(__file__, 511, 26), int_123325, *[line_123326], **kwargs_123327)
        
        # Assigning a type to the variable 'aij' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 20), 'aij', int_call_result_123328)
        # SSA branch for the else part of an if statement (line 510)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_complex' (line 512)
        is_complex_123329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 21), 'is_complex')
        # Testing the type of an if condition (line 512)
        if_condition_123330 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 21), is_complex_123329)
        # Assigning a type to the variable 'if_condition_123330' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 21), 'if_condition_123330', if_condition_123330)
        # SSA begins for if statement (line 512)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 513):
        
        # Assigning a Call to a Name (line 513):
        
        # Call to complex(...): (line 513)
        
        # Call to map(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'float' (line 513)
        float_123333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 39), 'float', False)
        
        # Call to split(...): (line 513)
        # Processing the call keyword arguments (line 513)
        kwargs_123336 = {}
        # Getting the type of 'line' (line 513)
        line_123334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 46), 'line', False)
        # Obtaining the member 'split' of a type (line 513)
        split_123335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 46), line_123334, 'split')
        # Calling split(args, kwargs) (line 513)
        split_call_result_123337 = invoke(stypy.reporting.localization.Localization(__file__, 513, 46), split_123335, *[], **kwargs_123336)
        
        # Processing the call keyword arguments (line 513)
        kwargs_123338 = {}
        # Getting the type of 'map' (line 513)
        map_123332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 35), 'map', False)
        # Calling map(args, kwargs) (line 513)
        map_call_result_123339 = invoke(stypy.reporting.localization.Localization(__file__, 513, 35), map_123332, *[float_123333, split_call_result_123337], **kwargs_123338)
        
        # Processing the call keyword arguments (line 513)
        kwargs_123340 = {}
        # Getting the type of 'complex' (line 513)
        complex_123331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 26), 'complex', False)
        # Calling complex(args, kwargs) (line 513)
        complex_call_result_123341 = invoke(stypy.reporting.localization.Localization(__file__, 513, 26), complex_123331, *[map_call_result_123339], **kwargs_123340)
        
        # Assigning a type to the variable 'aij' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'aij', complex_call_result_123341)
        # SSA branch for the else part of an if statement (line 512)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 515):
        
        # Assigning a Call to a Name (line 515):
        
        # Call to float(...): (line 515)
        # Processing the call arguments (line 515)
        # Getting the type of 'line' (line 515)
        line_123343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 32), 'line', False)
        # Processing the call keyword arguments (line 515)
        kwargs_123344 = {}
        # Getting the type of 'float' (line 515)
        float_123342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 26), 'float', False)
        # Calling float(args, kwargs) (line 515)
        float_call_result_123345 = invoke(stypy.reporting.localization.Localization(__file__, 515, 26), float_123342, *[line_123343], **kwargs_123344)
        
        # Assigning a type to the variable 'aij' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 20), 'aij', float_call_result_123345)
        # SSA join for if statement (line 512)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 510)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 516):
        
        # Assigning a Name to a Subscript (line 516):
        # Getting the type of 'aij' (line 516)
        aij_123346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 26), 'aij')
        # Getting the type of 'a' (line 516)
        a_123347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_123348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)
        # Getting the type of 'i' (line 516)
        i_123349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 18), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 18), tuple_123348, i_123349)
        # Adding element type (line 516)
        # Getting the type of 'j' (line 516)
        j_123350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 21), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 18), tuple_123348, j_123350)
        
        # Storing an element on a container (line 516)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 16), a_123347, (tuple_123348, aij_123346))
        
        
        # Evaluating a boolean operation
        # Getting the type of 'has_symmetry' (line 517)
        has_symmetry_123351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 19), 'has_symmetry')
        
        # Getting the type of 'i' (line 517)
        i_123352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 36), 'i')
        # Getting the type of 'j' (line 517)
        j_123353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 41), 'j')
        # Applying the binary operator '!=' (line 517)
        result_ne_123354 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 36), '!=', i_123352, j_123353)
        
        # Applying the binary operator 'and' (line 517)
        result_and_keyword_123355 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 19), 'and', has_symmetry_123351, result_ne_123354)
        
        # Testing the type of an if condition (line 517)
        if_condition_123356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 16), result_and_keyword_123355)
        # Assigning a type to the variable 'if_condition_123356' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 16), 'if_condition_123356', if_condition_123356)
        # SSA begins for if statement (line 517)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'is_skew' (line 518)
        is_skew_123357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 23), 'is_skew')
        # Testing the type of an if condition (line 518)
        if_condition_123358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 518, 20), is_skew_123357)
        # Assigning a type to the variable 'if_condition_123358' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 20), 'if_condition_123358', if_condition_123358)
        # SSA begins for if statement (line 518)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Subscript (line 519):
        
        # Assigning a UnaryOp to a Subscript (line 519):
        
        # Getting the type of 'aij' (line 519)
        aij_123359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 35), 'aij')
        # Applying the 'usub' unary operator (line 519)
        result___neg___123360 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 34), 'usub', aij_123359)
        
        # Getting the type of 'a' (line 519)
        a_123361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 24), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 519)
        tuple_123362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 519)
        # Adding element type (line 519)
        # Getting the type of 'j' (line 519)
        j_123363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 26), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 26), tuple_123362, j_123363)
        # Adding element type (line 519)
        # Getting the type of 'i' (line 519)
        i_123364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 26), tuple_123362, i_123364)
        
        # Storing an element on a container (line 519)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 24), a_123361, (tuple_123362, result___neg___123360))
        # SSA branch for the else part of an if statement (line 518)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_herm' (line 520)
        is_herm_123365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 25), 'is_herm')
        # Testing the type of an if condition (line 520)
        if_condition_123366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 25), is_herm_123365)
        # Assigning a type to the variable 'if_condition_123366' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 25), 'if_condition_123366', if_condition_123366)
        # SSA begins for if statement (line 520)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 521):
        
        # Assigning a Call to a Subscript (line 521):
        
        # Call to conj(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'aij' (line 521)
        aij_123368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 39), 'aij', False)
        # Processing the call keyword arguments (line 521)
        kwargs_123369 = {}
        # Getting the type of 'conj' (line 521)
        conj_123367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 34), 'conj', False)
        # Calling conj(args, kwargs) (line 521)
        conj_call_result_123370 = invoke(stypy.reporting.localization.Localization(__file__, 521, 34), conj_123367, *[aij_123368], **kwargs_123369)
        
        # Getting the type of 'a' (line 521)
        a_123371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_123372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        # Getting the type of 'j' (line 521)
        j_123373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 26), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 26), tuple_123372, j_123373)
        # Adding element type (line 521)
        # Getting the type of 'i' (line 521)
        i_123374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 26), tuple_123372, i_123374)
        
        # Storing an element on a container (line 521)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 24), a_123371, (tuple_123372, conj_call_result_123370))
        # SSA branch for the else part of an if statement (line 520)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Subscript (line 523):
        
        # Assigning a Name to a Subscript (line 523):
        # Getting the type of 'aij' (line 523)
        aij_123375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 34), 'aij')
        # Getting the type of 'a' (line 523)
        a_123376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 523)
        tuple_123377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 523)
        # Adding element type (line 523)
        # Getting the type of 'j' (line 523)
        j_123378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 26), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 26), tuple_123377, j_123378)
        # Adding element type (line 523)
        # Getting the type of 'i' (line 523)
        i_123379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 26), tuple_123377, i_123379)
        
        # Storing an element on a container (line 523)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 24), a_123376, (tuple_123377, aij_123375))
        # SSA join for if statement (line 520)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 518)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 517)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'i' (line 524)
        i_123380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'i')
        # Getting the type of 'rows' (line 524)
        rows_123381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 23), 'rows')
        int_123382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 28), 'int')
        # Applying the binary operator '-' (line 524)
        result_sub_123383 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 23), '-', rows_123381, int_123382)
        
        # Applying the binary operator '<' (line 524)
        result_lt_123384 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 19), '<', i_123380, result_sub_123383)
        
        # Testing the type of an if condition (line 524)
        if_condition_123385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 524, 16), result_lt_123384)
        # Assigning a type to the variable 'if_condition_123385' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'if_condition_123385', if_condition_123385)
        # SSA begins for if statement (line 524)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 525):
        
        # Assigning a BinOp to a Name (line 525):
        # Getting the type of 'i' (line 525)
        i_123386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 24), 'i')
        int_123387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 28), 'int')
        # Applying the binary operator '+' (line 525)
        result_add_123388 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 24), '+', i_123386, int_123387)
        
        # Assigning a type to the variable 'i' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 20), 'i', result_add_123388)
        # SSA branch for the else part of an if statement (line 524)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 527):
        
        # Assigning a BinOp to a Name (line 527):
        # Getting the type of 'j' (line 527)
        j_123389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 24), 'j')
        int_123390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 28), 'int')
        # Applying the binary operator '+' (line 527)
        result_add_123391 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 24), '+', j_123389, int_123390)
        
        # Assigning a type to the variable 'j' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 20), 'j', result_add_123391)
        
        
        # Getting the type of 'has_symmetry' (line 528)
        has_symmetry_123392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 27), 'has_symmetry')
        # Applying the 'not' unary operator (line 528)
        result_not__123393 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 23), 'not', has_symmetry_123392)
        
        # Testing the type of an if condition (line 528)
        if_condition_123394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 20), result_not__123393)
        # Assigning a type to the variable 'if_condition_123394' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'if_condition_123394', if_condition_123394)
        # SSA begins for if statement (line 528)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 529):
        
        # Assigning a Num to a Name (line 529):
        int_123395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 28), 'int')
        # Assigning a type to the variable 'i' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 24), 'i', int_123395)
        # SSA branch for the else part of an if statement (line 528)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 531):
        
        # Assigning a Name to a Name (line 531):
        # Getting the type of 'j' (line 531)
        j_123396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 28), 'j')
        # Assigning a type to the variable 'i' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 24), 'i', j_123396)
        # SSA join for if statement (line 528)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 524)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 506)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 532)
        i_123397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 20), 'i')
        
        # Obtaining an instance of the builtin type 'list' (line 532)
        list_123398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 532)
        # Adding element type (line 532)
        int_123399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 25), list_123398, int_123399)
        # Adding element type (line 532)
        # Getting the type of 'j' (line 532)
        j_123400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 29), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 25), list_123398, j_123400)
        
        # Applying the binary operator 'in' (line 532)
        result_contains_123401 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 20), 'in', i_123397, list_123398)
        
        
        # Getting the type of 'j' (line 532)
        j_123402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 36), 'j')
        # Getting the type of 'cols' (line 532)
        cols_123403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 41), 'cols')
        # Applying the binary operator '==' (line 532)
        result_eq_123404 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 36), '==', j_123402, cols_123403)
        
        # Applying the binary operator 'and' (line 532)
        result_and_keyword_123405 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 20), 'and', result_contains_123401, result_eq_123404)
        
        # Applying the 'not' unary operator (line 532)
        result_not__123406 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 15), 'not', result_and_keyword_123405)
        
        # Testing the type of an if condition (line 532)
        if_condition_123407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 12), result_not__123406)
        # Assigning a type to the variable 'if_condition_123407' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'if_condition_123407', if_condition_123407)
        # SSA begins for if statement (line 532)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 533)
        # Processing the call arguments (line 533)
        str_123409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 33), 'str', 'Parse error, did not read all lines.')
        # Processing the call keyword arguments (line 533)
        kwargs_123410 = {}
        # Getting the type of 'ValueError' (line 533)
        ValueError_123408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 533)
        ValueError_call_result_123411 = invoke(stypy.reporting.localization.Localization(__file__, 533, 22), ValueError_123408, *[str_123409], **kwargs_123410)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 533, 16), ValueError_call_result_123411, 'raise parameter', BaseException)
        # SSA join for if statement (line 532)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 502)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'format' (line 535)
        format_123412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 13), 'format')
        # Getting the type of 'self' (line 535)
        self_123413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 'self')
        # Obtaining the member 'FORMAT_COORDINATE' of a type (line 535)
        FORMAT_COORDINATE_123414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 23), self_123413, 'FORMAT_COORDINATE')
        # Applying the binary operator '==' (line 535)
        result_eq_123415 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 13), '==', format_123412, FORMAT_COORDINATE_123414)
        
        
        # Getting the type of 'coo_matrix' (line 535)
        coo_matrix_123416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 50), 'coo_matrix')
        # Getting the type of 'None' (line 535)
        None_123417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 64), 'None')
        # Applying the binary operator 'is' (line 535)
        result_is__123418 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 50), 'is', coo_matrix_123416, None_123417)
        
        # Applying the binary operator 'and' (line 535)
        result_and_keyword_123419 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 13), 'and', result_eq_123415, result_is__123418)
        
        # Testing the type of an if condition (line 535)
        if_condition_123420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 13), result_and_keyword_123419)
        # Assigning a type to the variable 'if_condition_123420' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 13), 'if_condition_123420', if_condition_123420)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 537):
        
        # Assigning a Call to a Name (line 537):
        
        # Call to zeros(...): (line 537)
        # Processing the call arguments (line 537)
        
        # Obtaining an instance of the builtin type 'tuple' (line 537)
        tuple_123422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 537)
        # Adding element type (line 537)
        # Getting the type of 'rows' (line 537)
        rows_123423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 23), 'rows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 23), tuple_123422, rows_123423)
        # Adding element type (line 537)
        # Getting the type of 'cols' (line 537)
        cols_123424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 29), 'cols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 23), tuple_123422, cols_123424)
        
        # Processing the call keyword arguments (line 537)
        # Getting the type of 'dtype' (line 537)
        dtype_123425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 42), 'dtype', False)
        keyword_123426 = dtype_123425
        kwargs_123427 = {'dtype': keyword_123426}
        # Getting the type of 'zeros' (line 537)
        zeros_123421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 537)
        zeros_call_result_123428 = invoke(stypy.reporting.localization.Localization(__file__, 537, 16), zeros_123421, *[tuple_123422], **kwargs_123427)
        
        # Assigning a type to the variable 'a' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'a', zeros_call_result_123428)
        
        # Assigning a Num to a Name (line 538):
        
        # Assigning a Num to a Name (line 538):
        int_123429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 19), 'int')
        # Assigning a type to the variable 'line' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'line', int_123429)
        
        # Assigning a Num to a Name (line 539):
        
        # Assigning a Num to a Name (line 539):
        int_123430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 16), 'int')
        # Assigning a type to the variable 'k' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'k', int_123430)
        
        # Getting the type of 'line' (line 540)
        line_123431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 18), 'line')
        # Testing the type of an if condition (line 540)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 12), line_123431)
        # SSA begins for while statement (line 540)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 541):
        
        # Assigning a Call to a Name (line 541):
        
        # Call to readline(...): (line 541)
        # Processing the call keyword arguments (line 541)
        kwargs_123434 = {}
        # Getting the type of 'stream' (line 541)
        stream_123432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 23), 'stream', False)
        # Obtaining the member 'readline' of a type (line 541)
        readline_123433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 23), stream_123432, 'readline')
        # Calling readline(args, kwargs) (line 541)
        readline_call_result_123435 = invoke(stypy.reporting.localization.Localization(__file__, 541, 23), readline_123433, *[], **kwargs_123434)
        
        # Assigning a type to the variable 'line' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'line', readline_call_result_123435)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'line' (line 542)
        line_123436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 23), 'line')
        # Applying the 'not' unary operator (line 542)
        result_not__123437 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 19), 'not', line_123436)
        
        
        # Call to startswith(...): (line 542)
        # Processing the call arguments (line 542)
        str_123440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 47), 'str', '%')
        # Processing the call keyword arguments (line 542)
        kwargs_123441 = {}
        # Getting the type of 'line' (line 542)
        line_123438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 31), 'line', False)
        # Obtaining the member 'startswith' of a type (line 542)
        startswith_123439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 31), line_123438, 'startswith')
        # Calling startswith(args, kwargs) (line 542)
        startswith_call_result_123442 = invoke(stypy.reporting.localization.Localization(__file__, 542, 31), startswith_123439, *[str_123440], **kwargs_123441)
        
        # Applying the binary operator 'or' (line 542)
        result_or_keyword_123443 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 19), 'or', result_not__123437, startswith_call_result_123442)
        
        # Testing the type of an if condition (line 542)
        if_condition_123444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 542, 16), result_or_keyword_123443)
        # Assigning a type to the variable 'if_condition_123444' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'if_condition_123444', if_condition_123444)
        # SSA begins for if statement (line 542)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 542)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 544):
        
        # Assigning a Call to a Name (line 544):
        
        # Call to split(...): (line 544)
        # Processing the call keyword arguments (line 544)
        kwargs_123447 = {}
        # Getting the type of 'line' (line 544)
        line_123445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 20), 'line', False)
        # Obtaining the member 'split' of a type (line 544)
        split_123446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 20), line_123445, 'split')
        # Calling split(args, kwargs) (line 544)
        split_call_result_123448 = invoke(stypy.reporting.localization.Localization(__file__, 544, 20), split_123446, *[], **kwargs_123447)
        
        # Assigning a type to the variable 'l' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'l', split_call_result_123448)
        
        # Assigning a Call to a Tuple (line 545):
        
        # Assigning a Subscript to a Name (line 545):
        
        # Obtaining the type of the subscript
        int_123449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 16), 'int')
        
        # Call to map(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'int' (line 545)
        int_123451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 27), 'int', False)
        
        # Obtaining the type of the subscript
        int_123452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 35), 'int')
        slice_123453 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 545, 32), None, int_123452, None)
        # Getting the type of 'l' (line 545)
        l_123454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 32), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 545)
        getitem___123455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 32), l_123454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 545)
        subscript_call_result_123456 = invoke(stypy.reporting.localization.Localization(__file__, 545, 32), getitem___123455, slice_123453)
        
        # Processing the call keyword arguments (line 545)
        kwargs_123457 = {}
        # Getting the type of 'map' (line 545)
        map_123450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 23), 'map', False)
        # Calling map(args, kwargs) (line 545)
        map_call_result_123458 = invoke(stypy.reporting.localization.Localization(__file__, 545, 23), map_123450, *[int_123451, subscript_call_result_123456], **kwargs_123457)
        
        # Obtaining the member '__getitem__' of a type (line 545)
        getitem___123459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 16), map_call_result_123458, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 545)
        subscript_call_result_123460 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), getitem___123459, int_123449)
        
        # Assigning a type to the variable 'tuple_var_assignment_122315' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'tuple_var_assignment_122315', subscript_call_result_123460)
        
        # Assigning a Subscript to a Name (line 545):
        
        # Obtaining the type of the subscript
        int_123461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 16), 'int')
        
        # Call to map(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'int' (line 545)
        int_123463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 27), 'int', False)
        
        # Obtaining the type of the subscript
        int_123464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 35), 'int')
        slice_123465 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 545, 32), None, int_123464, None)
        # Getting the type of 'l' (line 545)
        l_123466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 32), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 545)
        getitem___123467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 32), l_123466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 545)
        subscript_call_result_123468 = invoke(stypy.reporting.localization.Localization(__file__, 545, 32), getitem___123467, slice_123465)
        
        # Processing the call keyword arguments (line 545)
        kwargs_123469 = {}
        # Getting the type of 'map' (line 545)
        map_123462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 23), 'map', False)
        # Calling map(args, kwargs) (line 545)
        map_call_result_123470 = invoke(stypy.reporting.localization.Localization(__file__, 545, 23), map_123462, *[int_123463, subscript_call_result_123468], **kwargs_123469)
        
        # Obtaining the member '__getitem__' of a type (line 545)
        getitem___123471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 16), map_call_result_123470, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 545)
        subscript_call_result_123472 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), getitem___123471, int_123461)
        
        # Assigning a type to the variable 'tuple_var_assignment_122316' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'tuple_var_assignment_122316', subscript_call_result_123472)
        
        # Assigning a Name to a Name (line 545):
        # Getting the type of 'tuple_var_assignment_122315' (line 545)
        tuple_var_assignment_122315_123473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'tuple_var_assignment_122315')
        # Assigning a type to the variable 'i' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'i', tuple_var_assignment_122315_123473)
        
        # Assigning a Name to a Name (line 545):
        # Getting the type of 'tuple_var_assignment_122316' (line 545)
        tuple_var_assignment_122316_123474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'tuple_var_assignment_122316')
        # Assigning a type to the variable 'j' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'j', tuple_var_assignment_122316_123474)
        
        # Assigning a Tuple to a Tuple (line 546):
        
        # Assigning a BinOp to a Name (line 546):
        # Getting the type of 'i' (line 546)
        i_123475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'i')
        int_123476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 25), 'int')
        # Applying the binary operator '-' (line 546)
        result_sub_123477 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 23), '-', i_123475, int_123476)
        
        # Assigning a type to the variable 'tuple_assignment_122317' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'tuple_assignment_122317', result_sub_123477)
        
        # Assigning a BinOp to a Name (line 546):
        # Getting the type of 'j' (line 546)
        j_123478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 28), 'j')
        int_123479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 30), 'int')
        # Applying the binary operator '-' (line 546)
        result_sub_123480 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 28), '-', j_123478, int_123479)
        
        # Assigning a type to the variable 'tuple_assignment_122318' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'tuple_assignment_122318', result_sub_123480)
        
        # Assigning a Name to a Name (line 546):
        # Getting the type of 'tuple_assignment_122317' (line 546)
        tuple_assignment_122317_123481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'tuple_assignment_122317')
        # Assigning a type to the variable 'i' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'i', tuple_assignment_122317_123481)
        
        # Assigning a Name to a Name (line 546):
        # Getting the type of 'tuple_assignment_122318' (line 546)
        tuple_assignment_122318_123482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'tuple_assignment_122318')
        # Assigning a type to the variable 'j' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 19), 'j', tuple_assignment_122318_123482)
        
        # Getting the type of 'is_integer' (line 547)
        is_integer_123483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 19), 'is_integer')
        # Testing the type of an if condition (line 547)
        if_condition_123484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 16), is_integer_123483)
        # Assigning a type to the variable 'if_condition_123484' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'if_condition_123484', if_condition_123484)
        # SSA begins for if statement (line 547)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 548):
        
        # Assigning a Call to a Name (line 548):
        
        # Call to int(...): (line 548)
        # Processing the call arguments (line 548)
        
        # Obtaining the type of the subscript
        int_123486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 32), 'int')
        # Getting the type of 'l' (line 548)
        l_123487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 30), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___123488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 30), l_123487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_123489 = invoke(stypy.reporting.localization.Localization(__file__, 548, 30), getitem___123488, int_123486)
        
        # Processing the call keyword arguments (line 548)
        kwargs_123490 = {}
        # Getting the type of 'int' (line 548)
        int_123485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 26), 'int', False)
        # Calling int(args, kwargs) (line 548)
        int_call_result_123491 = invoke(stypy.reporting.localization.Localization(__file__, 548, 26), int_123485, *[subscript_call_result_123489], **kwargs_123490)
        
        # Assigning a type to the variable 'aij' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'aij', int_call_result_123491)
        # SSA branch for the else part of an if statement (line 547)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_complex' (line 549)
        is_complex_123492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 21), 'is_complex')
        # Testing the type of an if condition (line 549)
        if_condition_123493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 549, 21), is_complex_123492)
        # Assigning a type to the variable 'if_condition_123493' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 21), 'if_condition_123493', if_condition_123493)
        # SSA begins for if statement (line 549)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 550):
        
        # Assigning a Call to a Name (line 550):
        
        # Call to complex(...): (line 550)
        
        # Call to map(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'float' (line 550)
        float_123496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 39), 'float', False)
        
        # Obtaining the type of the subscript
        int_123497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 48), 'int')
        slice_123498 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 550, 46), int_123497, None, None)
        # Getting the type of 'l' (line 550)
        l_123499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 46), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 550)
        getitem___123500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 46), l_123499, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 550)
        subscript_call_result_123501 = invoke(stypy.reporting.localization.Localization(__file__, 550, 46), getitem___123500, slice_123498)
        
        # Processing the call keyword arguments (line 550)
        kwargs_123502 = {}
        # Getting the type of 'map' (line 550)
        map_123495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'map', False)
        # Calling map(args, kwargs) (line 550)
        map_call_result_123503 = invoke(stypy.reporting.localization.Localization(__file__, 550, 35), map_123495, *[float_123496, subscript_call_result_123501], **kwargs_123502)
        
        # Processing the call keyword arguments (line 550)
        kwargs_123504 = {}
        # Getting the type of 'complex' (line 550)
        complex_123494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 26), 'complex', False)
        # Calling complex(args, kwargs) (line 550)
        complex_call_result_123505 = invoke(stypy.reporting.localization.Localization(__file__, 550, 26), complex_123494, *[map_call_result_123503], **kwargs_123504)
        
        # Assigning a type to the variable 'aij' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 20), 'aij', complex_call_result_123505)
        # SSA branch for the else part of an if statement (line 549)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 552):
        
        # Assigning a Call to a Name (line 552):
        
        # Call to float(...): (line 552)
        # Processing the call arguments (line 552)
        
        # Obtaining the type of the subscript
        int_123507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 34), 'int')
        # Getting the type of 'l' (line 552)
        l_123508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 32), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 552)
        getitem___123509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 32), l_123508, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 552)
        subscript_call_result_123510 = invoke(stypy.reporting.localization.Localization(__file__, 552, 32), getitem___123509, int_123507)
        
        # Processing the call keyword arguments (line 552)
        kwargs_123511 = {}
        # Getting the type of 'float' (line 552)
        float_123506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 26), 'float', False)
        # Calling float(args, kwargs) (line 552)
        float_call_result_123512 = invoke(stypy.reporting.localization.Localization(__file__, 552, 26), float_123506, *[subscript_call_result_123510], **kwargs_123511)
        
        # Assigning a type to the variable 'aij' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 20), 'aij', float_call_result_123512)
        # SSA join for if statement (line 549)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 547)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 553):
        
        # Assigning a Name to a Subscript (line 553):
        # Getting the type of 'aij' (line 553)
        aij_123513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 26), 'aij')
        # Getting the type of 'a' (line 553)
        a_123514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 553)
        tuple_123515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 553)
        # Adding element type (line 553)
        # Getting the type of 'i' (line 553)
        i_123516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 18), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 18), tuple_123515, i_123516)
        # Adding element type (line 553)
        # Getting the type of 'j' (line 553)
        j_123517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 21), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 18), tuple_123515, j_123517)
        
        # Storing an element on a container (line 553)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 16), a_123514, (tuple_123515, aij_123513))
        
        
        # Evaluating a boolean operation
        # Getting the type of 'has_symmetry' (line 554)
        has_symmetry_123518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'has_symmetry')
        
        # Getting the type of 'i' (line 554)
        i_123519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 36), 'i')
        # Getting the type of 'j' (line 554)
        j_123520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 41), 'j')
        # Applying the binary operator '!=' (line 554)
        result_ne_123521 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 36), '!=', i_123519, j_123520)
        
        # Applying the binary operator 'and' (line 554)
        result_and_keyword_123522 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 19), 'and', has_symmetry_123518, result_ne_123521)
        
        # Testing the type of an if condition (line 554)
        if_condition_123523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 16), result_and_keyword_123522)
        # Assigning a type to the variable 'if_condition_123523' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'if_condition_123523', if_condition_123523)
        # SSA begins for if statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'is_skew' (line 555)
        is_skew_123524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 23), 'is_skew')
        # Testing the type of an if condition (line 555)
        if_condition_123525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 20), is_skew_123524)
        # Assigning a type to the variable 'if_condition_123525' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 20), 'if_condition_123525', if_condition_123525)
        # SSA begins for if statement (line 555)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Subscript (line 556):
        
        # Assigning a UnaryOp to a Subscript (line 556):
        
        # Getting the type of 'aij' (line 556)
        aij_123526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 35), 'aij')
        # Applying the 'usub' unary operator (line 556)
        result___neg___123527 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 34), 'usub', aij_123526)
        
        # Getting the type of 'a' (line 556)
        a_123528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 24), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 556)
        tuple_123529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 556)
        # Adding element type (line 556)
        # Getting the type of 'j' (line 556)
        j_123530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 26), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 26), tuple_123529, j_123530)
        # Adding element type (line 556)
        # Getting the type of 'i' (line 556)
        i_123531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 26), tuple_123529, i_123531)
        
        # Storing an element on a container (line 556)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 24), a_123528, (tuple_123529, result___neg___123527))
        # SSA branch for the else part of an if statement (line 555)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_herm' (line 557)
        is_herm_123532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 25), 'is_herm')
        # Testing the type of an if condition (line 557)
        if_condition_123533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 25), is_herm_123532)
        # Assigning a type to the variable 'if_condition_123533' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 25), 'if_condition_123533', if_condition_123533)
        # SSA begins for if statement (line 557)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 558):
        
        # Assigning a Call to a Subscript (line 558):
        
        # Call to conj(...): (line 558)
        # Processing the call arguments (line 558)
        # Getting the type of 'aij' (line 558)
        aij_123535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 39), 'aij', False)
        # Processing the call keyword arguments (line 558)
        kwargs_123536 = {}
        # Getting the type of 'conj' (line 558)
        conj_123534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 34), 'conj', False)
        # Calling conj(args, kwargs) (line 558)
        conj_call_result_123537 = invoke(stypy.reporting.localization.Localization(__file__, 558, 34), conj_123534, *[aij_123535], **kwargs_123536)
        
        # Getting the type of 'a' (line 558)
        a_123538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 24), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 558)
        tuple_123539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 558)
        # Adding element type (line 558)
        # Getting the type of 'j' (line 558)
        j_123540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 26), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 26), tuple_123539, j_123540)
        # Adding element type (line 558)
        # Getting the type of 'i' (line 558)
        i_123541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 26), tuple_123539, i_123541)
        
        # Storing an element on a container (line 558)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 24), a_123538, (tuple_123539, conj_call_result_123537))
        # SSA branch for the else part of an if statement (line 557)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Subscript (line 560):
        
        # Assigning a Name to a Subscript (line 560):
        # Getting the type of 'aij' (line 560)
        aij_123542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 34), 'aij')
        # Getting the type of 'a' (line 560)
        a_123543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 24), 'a')
        
        # Obtaining an instance of the builtin type 'tuple' (line 560)
        tuple_123544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 560)
        # Adding element type (line 560)
        # Getting the type of 'j' (line 560)
        j_123545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 26), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 26), tuple_123544, j_123545)
        # Adding element type (line 560)
        # Getting the type of 'i' (line 560)
        i_123546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 26), tuple_123544, i_123546)
        
        # Storing an element on a container (line 560)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 24), a_123543, (tuple_123544, aij_123542))
        # SSA join for if statement (line 557)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 555)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 554)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 561):
        
        # Assigning a BinOp to a Name (line 561):
        # Getting the type of 'k' (line 561)
        k_123547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 20), 'k')
        int_123548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 24), 'int')
        # Applying the binary operator '+' (line 561)
        result_add_123549 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 20), '+', k_123547, int_123548)
        
        # Assigning a type to the variable 'k' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'k', result_add_123549)
        # SSA join for while statement (line 540)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Getting the type of 'k' (line 562)
        k_123550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 19), 'k')
        # Getting the type of 'entries' (line 562)
        entries_123551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 24), 'entries')
        # Applying the binary operator '==' (line 562)
        result_eq_123552 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 19), '==', k_123550, entries_123551)
        
        # Applying the 'not' unary operator (line 562)
        result_not__123553 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 15), 'not', result_eq_123552)
        
        # Testing the type of an if condition (line 562)
        if_condition_123554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 12), result_not__123553)
        # Assigning a type to the variable 'if_condition_123554' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'if_condition_123554', if_condition_123554)
        # SSA begins for if statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 563)
        # Processing the call arguments (line 563)
        str_123556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 27), 'str', 'Did not read all entries')
        # Processing the call keyword arguments (line 563)
        kwargs_123557 = {}
        # Getting the type of 'ValueError' (line 563)
        ValueError_123555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 563)
        ValueError_call_result_123558 = invoke(stypy.reporting.localization.Localization(__file__, 563, 16), ValueError_123555, *[str_123556], **kwargs_123557)
        
        # SSA join for if statement (line 562)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 535)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'format' (line 565)
        format_123559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 13), 'format')
        # Getting the type of 'self' (line 565)
        self_123560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 23), 'self')
        # Obtaining the member 'FORMAT_COORDINATE' of a type (line 565)
        FORMAT_COORDINATE_123561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 23), self_123560, 'FORMAT_COORDINATE')
        # Applying the binary operator '==' (line 565)
        result_eq_123562 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 13), '==', format_123559, FORMAT_COORDINATE_123561)
        
        # Testing the type of an if condition (line 565)
        if_condition_123563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 13), result_eq_123562)
        # Assigning a type to the variable 'if_condition_123563' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 13), 'if_condition_123563', if_condition_123563)
        # SSA begins for if statement (line 565)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'entries' (line 568)
        entries_123564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'entries')
        int_123565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 26), 'int')
        # Applying the binary operator '==' (line 568)
        result_eq_123566 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 15), '==', entries_123564, int_123565)
        
        # Testing the type of an if condition (line 568)
        if_condition_123567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 12), result_eq_123566)
        # Assigning a type to the variable 'if_condition_123567' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'if_condition_123567', if_condition_123567)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to coo_matrix(...): (line 570)
        # Processing the call arguments (line 570)
        
        # Obtaining an instance of the builtin type 'tuple' (line 570)
        tuple_123569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 570)
        # Adding element type (line 570)
        # Getting the type of 'rows' (line 570)
        rows_123570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 35), 'rows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 35), tuple_123569, rows_123570)
        # Adding element type (line 570)
        # Getting the type of 'cols' (line 570)
        cols_123571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 41), 'cols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 35), tuple_123569, cols_123571)
        
        # Processing the call keyword arguments (line 570)
        # Getting the type of 'dtype' (line 570)
        dtype_123572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 54), 'dtype', False)
        keyword_123573 = dtype_123572
        kwargs_123574 = {'dtype': keyword_123573}
        # Getting the type of 'coo_matrix' (line 570)
        coo_matrix_123568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 570)
        coo_matrix_call_result_123575 = invoke(stypy.reporting.localization.Localization(__file__, 570, 23), coo_matrix_123568, *[tuple_123569], **kwargs_123574)
        
        # Assigning a type to the variable 'stypy_return_type' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'stypy_return_type', coo_matrix_call_result_123575)
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 572):
        
        # Assigning a Call to a Name (line 572):
        
        # Call to zeros(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'entries' (line 572)
        entries_123577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 22), 'entries', False)
        # Processing the call keyword arguments (line 572)
        str_123578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 37), 'str', 'intc')
        keyword_123579 = str_123578
        kwargs_123580 = {'dtype': keyword_123579}
        # Getting the type of 'zeros' (line 572)
        zeros_123576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 572)
        zeros_call_result_123581 = invoke(stypy.reporting.localization.Localization(__file__, 572, 16), zeros_123576, *[entries_123577], **kwargs_123580)
        
        # Assigning a type to the variable 'I' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'I', zeros_call_result_123581)
        
        # Assigning a Call to a Name (line 573):
        
        # Assigning a Call to a Name (line 573):
        
        # Call to zeros(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'entries' (line 573)
        entries_123583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 22), 'entries', False)
        # Processing the call keyword arguments (line 573)
        str_123584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 37), 'str', 'intc')
        keyword_123585 = str_123584
        kwargs_123586 = {'dtype': keyword_123585}
        # Getting the type of 'zeros' (line 573)
        zeros_123582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'zeros', False)
        # Calling zeros(args, kwargs) (line 573)
        zeros_call_result_123587 = invoke(stypy.reporting.localization.Localization(__file__, 573, 16), zeros_123582, *[entries_123583], **kwargs_123586)
        
        # Assigning a type to the variable 'J' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'J', zeros_call_result_123587)
        
        # Getting the type of 'is_pattern' (line 574)
        is_pattern_123588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 15), 'is_pattern')
        # Testing the type of an if condition (line 574)
        if_condition_123589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 12), is_pattern_123588)
        # Assigning a type to the variable 'if_condition_123589' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'if_condition_123589', if_condition_123589)
        # SSA begins for if statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 575):
        
        # Assigning a Call to a Name (line 575):
        
        # Call to ones(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'entries' (line 575)
        entries_123591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 25), 'entries', False)
        # Processing the call keyword arguments (line 575)
        str_123592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 40), 'str', 'int8')
        keyword_123593 = str_123592
        kwargs_123594 = {'dtype': keyword_123593}
        # Getting the type of 'ones' (line 575)
        ones_123590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 20), 'ones', False)
        # Calling ones(args, kwargs) (line 575)
        ones_call_result_123595 = invoke(stypy.reporting.localization.Localization(__file__, 575, 20), ones_123590, *[entries_123591], **kwargs_123594)
        
        # Assigning a type to the variable 'V' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'V', ones_call_result_123595)
        # SSA branch for the else part of an if statement (line 574)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_integer' (line 576)
        is_integer_123596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 17), 'is_integer')
        # Testing the type of an if condition (line 576)
        if_condition_123597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 17), is_integer_123596)
        # Assigning a type to the variable 'if_condition_123597' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 17), 'if_condition_123597', if_condition_123597)
        # SSA begins for if statement (line 576)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 577):
        
        # Assigning a Call to a Name (line 577):
        
        # Call to zeros(...): (line 577)
        # Processing the call arguments (line 577)
        # Getting the type of 'entries' (line 577)
        entries_123599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 26), 'entries', False)
        # Processing the call keyword arguments (line 577)
        str_123600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 41), 'str', 'intp')
        keyword_123601 = str_123600
        kwargs_123602 = {'dtype': keyword_123601}
        # Getting the type of 'zeros' (line 577)
        zeros_123598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'zeros', False)
        # Calling zeros(args, kwargs) (line 577)
        zeros_call_result_123603 = invoke(stypy.reporting.localization.Localization(__file__, 577, 20), zeros_123598, *[entries_123599], **kwargs_123602)
        
        # Assigning a type to the variable 'V' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'V', zeros_call_result_123603)
        # SSA branch for the else part of an if statement (line 576)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_complex' (line 578)
        is_complex_123604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 17), 'is_complex')
        # Testing the type of an if condition (line 578)
        if_condition_123605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 17), is_complex_123604)
        # Assigning a type to the variable 'if_condition_123605' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 17), 'if_condition_123605', if_condition_123605)
        # SSA begins for if statement (line 578)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 579):
        
        # Assigning a Call to a Name (line 579):
        
        # Call to zeros(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'entries' (line 579)
        entries_123607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), 'entries', False)
        # Processing the call keyword arguments (line 579)
        str_123608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 41), 'str', 'complex')
        keyword_123609 = str_123608
        kwargs_123610 = {'dtype': keyword_123609}
        # Getting the type of 'zeros' (line 579)
        zeros_123606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'zeros', False)
        # Calling zeros(args, kwargs) (line 579)
        zeros_call_result_123611 = invoke(stypy.reporting.localization.Localization(__file__, 579, 20), zeros_123606, *[entries_123607], **kwargs_123610)
        
        # Assigning a type to the variable 'V' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'V', zeros_call_result_123611)
        # SSA branch for the else part of an if statement (line 578)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 581):
        
        # Assigning a Call to a Name (line 581):
        
        # Call to zeros(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'entries' (line 581)
        entries_123613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 26), 'entries', False)
        # Processing the call keyword arguments (line 581)
        str_123614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 41), 'str', 'float')
        keyword_123615 = str_123614
        kwargs_123616 = {'dtype': keyword_123615}
        # Getting the type of 'zeros' (line 581)
        zeros_123612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 20), 'zeros', False)
        # Calling zeros(args, kwargs) (line 581)
        zeros_call_result_123617 = invoke(stypy.reporting.localization.Localization(__file__, 581, 20), zeros_123612, *[entries_123613], **kwargs_123616)
        
        # Assigning a type to the variable 'V' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'V', zeros_call_result_123617)
        # SSA join for if statement (line 578)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 576)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 574)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 583):
        
        # Assigning a Num to a Name (line 583):
        int_123618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 27), 'int')
        # Assigning a type to the variable 'entry_number' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'entry_number', int_123618)
        
        # Getting the type of 'stream' (line 584)
        stream_123619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 24), 'stream')
        # Testing the type of a for loop iterable (line 584)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 584, 12), stream_123619)
        # Getting the type of the for loop variable (line 584)
        for_loop_var_123620 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 584, 12), stream_123619)
        # Assigning a type to the variable 'line' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'line', for_loop_var_123620)
        # SSA begins for a for statement (line 584)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'line' (line 585)
        line_123621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 23), 'line')
        # Applying the 'not' unary operator (line 585)
        result_not__123622 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 19), 'not', line_123621)
        
        
        # Call to startswith(...): (line 585)
        # Processing the call arguments (line 585)
        str_123625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 47), 'str', '%')
        # Processing the call keyword arguments (line 585)
        kwargs_123626 = {}
        # Getting the type of 'line' (line 585)
        line_123623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 31), 'line', False)
        # Obtaining the member 'startswith' of a type (line 585)
        startswith_123624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 31), line_123623, 'startswith')
        # Calling startswith(args, kwargs) (line 585)
        startswith_call_result_123627 = invoke(stypy.reporting.localization.Localization(__file__, 585, 31), startswith_123624, *[str_123625], **kwargs_123626)
        
        # Applying the binary operator 'or' (line 585)
        result_or_keyword_123628 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 19), 'or', result_not__123622, startswith_call_result_123627)
        
        # Testing the type of an if condition (line 585)
        if_condition_123629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 16), result_or_keyword_123628)
        # Assigning a type to the variable 'if_condition_123629' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 16), 'if_condition_123629', if_condition_123629)
        # SSA begins for if statement (line 585)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 585)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'entry_number' (line 588)
        entry_number_123630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 19), 'entry_number')
        int_123631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 32), 'int')
        # Applying the binary operator '+' (line 588)
        result_add_123632 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 19), '+', entry_number_123630, int_123631)
        
        # Getting the type of 'entries' (line 588)
        entries_123633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 36), 'entries')
        # Applying the binary operator '>' (line 588)
        result_gt_123634 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 19), '>', result_add_123632, entries_123633)
        
        # Testing the type of an if condition (line 588)
        if_condition_123635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 588, 16), result_gt_123634)
        # Assigning a type to the variable 'if_condition_123635' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 16), 'if_condition_123635', if_condition_123635)
        # SSA begins for if statement (line 588)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 589)
        # Processing the call arguments (line 589)
        str_123637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 37), 'str', "'entries' in header is smaller than number of entries")
        # Processing the call keyword arguments (line 589)
        kwargs_123638 = {}
        # Getting the type of 'ValueError' (line 589)
        ValueError_123636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 589)
        ValueError_call_result_123639 = invoke(stypy.reporting.localization.Localization(__file__, 589, 26), ValueError_123636, *[str_123637], **kwargs_123638)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 589, 20), ValueError_call_result_123639, 'raise parameter', BaseException)
        # SSA join for if statement (line 588)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 591):
        
        # Assigning a Call to a Name (line 591):
        
        # Call to split(...): (line 591)
        # Processing the call keyword arguments (line 591)
        kwargs_123642 = {}
        # Getting the type of 'line' (line 591)
        line_123640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 20), 'line', False)
        # Obtaining the member 'split' of a type (line 591)
        split_123641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 20), line_123640, 'split')
        # Calling split(args, kwargs) (line 591)
        split_call_result_123643 = invoke(stypy.reporting.localization.Localization(__file__, 591, 20), split_123641, *[], **kwargs_123642)
        
        # Assigning a type to the variable 'l' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'l', split_call_result_123643)
        
        # Assigning a Call to a Tuple (line 592):
        
        # Assigning a Subscript to a Name (line 592):
        
        # Obtaining the type of the subscript
        int_123644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 16), 'int')
        
        # Call to map(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'int' (line 592)
        int_123646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 55), 'int', False)
        
        # Obtaining the type of the subscript
        int_123647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 63), 'int')
        slice_123648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 592, 60), None, int_123647, None)
        # Getting the type of 'l' (line 592)
        l_123649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 60), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___123650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 60), l_123649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_123651 = invoke(stypy.reporting.localization.Localization(__file__, 592, 60), getitem___123650, slice_123648)
        
        # Processing the call keyword arguments (line 592)
        kwargs_123652 = {}
        # Getting the type of 'map' (line 592)
        map_123645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 51), 'map', False)
        # Calling map(args, kwargs) (line 592)
        map_call_result_123653 = invoke(stypy.reporting.localization.Localization(__file__, 592, 51), map_123645, *[int_123646, subscript_call_result_123651], **kwargs_123652)
        
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___123654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 16), map_call_result_123653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_123655 = invoke(stypy.reporting.localization.Localization(__file__, 592, 16), getitem___123654, int_123644)
        
        # Assigning a type to the variable 'tuple_var_assignment_122319' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'tuple_var_assignment_122319', subscript_call_result_123655)
        
        # Assigning a Subscript to a Name (line 592):
        
        # Obtaining the type of the subscript
        int_123656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 16), 'int')
        
        # Call to map(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'int' (line 592)
        int_123658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 55), 'int', False)
        
        # Obtaining the type of the subscript
        int_123659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 63), 'int')
        slice_123660 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 592, 60), None, int_123659, None)
        # Getting the type of 'l' (line 592)
        l_123661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 60), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___123662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 60), l_123661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_123663 = invoke(stypy.reporting.localization.Localization(__file__, 592, 60), getitem___123662, slice_123660)
        
        # Processing the call keyword arguments (line 592)
        kwargs_123664 = {}
        # Getting the type of 'map' (line 592)
        map_123657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 51), 'map', False)
        # Calling map(args, kwargs) (line 592)
        map_call_result_123665 = invoke(stypy.reporting.localization.Localization(__file__, 592, 51), map_123657, *[int_123658, subscript_call_result_123663], **kwargs_123664)
        
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___123666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 16), map_call_result_123665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_123667 = invoke(stypy.reporting.localization.Localization(__file__, 592, 16), getitem___123666, int_123656)
        
        # Assigning a type to the variable 'tuple_var_assignment_122320' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'tuple_var_assignment_122320', subscript_call_result_123667)
        
        # Assigning a Name to a Subscript (line 592):
        # Getting the type of 'tuple_var_assignment_122319' (line 592)
        tuple_var_assignment_122319_123668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'tuple_var_assignment_122319')
        # Getting the type of 'I' (line 592)
        I_123669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'I')
        # Getting the type of 'entry_number' (line 592)
        entry_number_123670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 18), 'entry_number')
        # Storing an element on a container (line 592)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 16), I_123669, (entry_number_123670, tuple_var_assignment_122319_123668))
        
        # Assigning a Name to a Subscript (line 592):
        # Getting the type of 'tuple_var_assignment_122320' (line 592)
        tuple_var_assignment_122320_123671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'tuple_var_assignment_122320')
        # Getting the type of 'J' (line 592)
        J_123672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 33), 'J')
        # Getting the type of 'entry_number' (line 592)
        entry_number_123673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 35), 'entry_number')
        # Storing an element on a container (line 592)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 33), J_123672, (entry_number_123673, tuple_var_assignment_122320_123671))
        
        
        # Getting the type of 'is_pattern' (line 594)
        is_pattern_123674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 23), 'is_pattern')
        # Applying the 'not' unary operator (line 594)
        result_not__123675 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 19), 'not', is_pattern_123674)
        
        # Testing the type of an if condition (line 594)
        if_condition_123676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 594, 16), result_not__123675)
        # Assigning a type to the variable 'if_condition_123676' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'if_condition_123676', if_condition_123676)
        # SSA begins for if statement (line 594)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'is_integer' (line 595)
        is_integer_123677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 23), 'is_integer')
        # Testing the type of an if condition (line 595)
        if_condition_123678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 20), is_integer_123677)
        # Assigning a type to the variable 'if_condition_123678' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 20), 'if_condition_123678', if_condition_123678)
        # SSA begins for if statement (line 595)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 596):
        
        # Assigning a Call to a Subscript (line 596):
        
        # Call to int(...): (line 596)
        # Processing the call arguments (line 596)
        
        # Obtaining the type of the subscript
        int_123680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 48), 'int')
        # Getting the type of 'l' (line 596)
        l_123681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 46), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 596)
        getitem___123682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 46), l_123681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 596)
        subscript_call_result_123683 = invoke(stypy.reporting.localization.Localization(__file__, 596, 46), getitem___123682, int_123680)
        
        # Processing the call keyword arguments (line 596)
        kwargs_123684 = {}
        # Getting the type of 'int' (line 596)
        int_123679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 42), 'int', False)
        # Calling int(args, kwargs) (line 596)
        int_call_result_123685 = invoke(stypy.reporting.localization.Localization(__file__, 596, 42), int_123679, *[subscript_call_result_123683], **kwargs_123684)
        
        # Getting the type of 'V' (line 596)
        V_123686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 24), 'V')
        # Getting the type of 'entry_number' (line 596)
        entry_number_123687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 26), 'entry_number')
        # Storing an element on a container (line 596)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 24), V_123686, (entry_number_123687, int_call_result_123685))
        # SSA branch for the else part of an if statement (line 595)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_complex' (line 597)
        is_complex_123688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 25), 'is_complex')
        # Testing the type of an if condition (line 597)
        if_condition_123689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 25), is_complex_123688)
        # Assigning a type to the variable 'if_condition_123689' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 25), 'if_condition_123689', if_condition_123689)
        # SSA begins for if statement (line 597)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 598):
        
        # Assigning a Call to a Subscript (line 598):
        
        # Call to complex(...): (line 598)
        
        # Call to map(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 'float' (line 598)
        float_123692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 55), 'float', False)
        
        # Obtaining the type of the subscript
        int_123693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 64), 'int')
        slice_123694 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 598, 62), int_123693, None, None)
        # Getting the type of 'l' (line 598)
        l_123695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 62), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 598)
        getitem___123696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 62), l_123695, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 598)
        subscript_call_result_123697 = invoke(stypy.reporting.localization.Localization(__file__, 598, 62), getitem___123696, slice_123694)
        
        # Processing the call keyword arguments (line 598)
        kwargs_123698 = {}
        # Getting the type of 'map' (line 598)
        map_123691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 51), 'map', False)
        # Calling map(args, kwargs) (line 598)
        map_call_result_123699 = invoke(stypy.reporting.localization.Localization(__file__, 598, 51), map_123691, *[float_123692, subscript_call_result_123697], **kwargs_123698)
        
        # Processing the call keyword arguments (line 598)
        kwargs_123700 = {}
        # Getting the type of 'complex' (line 598)
        complex_123690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 42), 'complex', False)
        # Calling complex(args, kwargs) (line 598)
        complex_call_result_123701 = invoke(stypy.reporting.localization.Localization(__file__, 598, 42), complex_123690, *[map_call_result_123699], **kwargs_123700)
        
        # Getting the type of 'V' (line 598)
        V_123702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 24), 'V')
        # Getting the type of 'entry_number' (line 598)
        entry_number_123703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 26), 'entry_number')
        # Storing an element on a container (line 598)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 24), V_123702, (entry_number_123703, complex_call_result_123701))
        # SSA branch for the else part of an if statement (line 597)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Subscript (line 600):
        
        # Assigning a Call to a Subscript (line 600):
        
        # Call to float(...): (line 600)
        # Processing the call arguments (line 600)
        
        # Obtaining the type of the subscript
        int_123705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 50), 'int')
        # Getting the type of 'l' (line 600)
        l_123706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 48), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 600)
        getitem___123707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 48), l_123706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 600)
        subscript_call_result_123708 = invoke(stypy.reporting.localization.Localization(__file__, 600, 48), getitem___123707, int_123705)
        
        # Processing the call keyword arguments (line 600)
        kwargs_123709 = {}
        # Getting the type of 'float' (line 600)
        float_123704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 42), 'float', False)
        # Calling float(args, kwargs) (line 600)
        float_call_result_123710 = invoke(stypy.reporting.localization.Localization(__file__, 600, 42), float_123704, *[subscript_call_result_123708], **kwargs_123709)
        
        # Getting the type of 'V' (line 600)
        V_123711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 24), 'V')
        # Getting the type of 'entry_number' (line 600)
        entry_number_123712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 26), 'entry_number')
        # Storing an element on a container (line 600)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 24), V_123711, (entry_number_123712, float_call_result_123710))
        # SSA join for if statement (line 597)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 595)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 594)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'entry_number' (line 601)
        entry_number_123713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 16), 'entry_number')
        int_123714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 32), 'int')
        # Applying the binary operator '+=' (line 601)
        result_iadd_123715 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 16), '+=', entry_number_123713, int_123714)
        # Assigning a type to the variable 'entry_number' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 16), 'entry_number', result_iadd_123715)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'entry_number' (line 602)
        entry_number_123716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 15), 'entry_number')
        # Getting the type of 'entries' (line 602)
        entries_123717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 30), 'entries')
        # Applying the binary operator '<' (line 602)
        result_lt_123718 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 15), '<', entry_number_123716, entries_123717)
        
        # Testing the type of an if condition (line 602)
        if_condition_123719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 602, 12), result_lt_123718)
        # Assigning a type to the variable 'if_condition_123719' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'if_condition_123719', if_condition_123719)
        # SSA begins for if statement (line 602)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 603)
        # Processing the call arguments (line 603)
        str_123721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 33), 'str', "'entries' in header is larger than number of entries")
        # Processing the call keyword arguments (line 603)
        kwargs_123722 = {}
        # Getting the type of 'ValueError' (line 603)
        ValueError_123720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 603)
        ValueError_call_result_123723 = invoke(stypy.reporting.localization.Localization(__file__, 603, 22), ValueError_123720, *[str_123721], **kwargs_123722)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 603, 16), ValueError_call_result_123723, 'raise parameter', BaseException)
        # SSA join for if statement (line 602)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'I' (line 606)
        I_123724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'I')
        int_123725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 17), 'int')
        # Applying the binary operator '-=' (line 606)
        result_isub_123726 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 12), '-=', I_123724, int_123725)
        # Assigning a type to the variable 'I' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'I', result_isub_123726)
        
        
        # Getting the type of 'J' (line 607)
        J_123727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'J')
        int_123728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 17), 'int')
        # Applying the binary operator '-=' (line 607)
        result_isub_123729 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 12), '-=', J_123727, int_123728)
        # Assigning a type to the variable 'J' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'J', result_isub_123729)
        
        
        # Getting the type of 'has_symmetry' (line 609)
        has_symmetry_123730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 15), 'has_symmetry')
        # Testing the type of an if condition (line 609)
        if_condition_123731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 12), has_symmetry_123730)
        # Assigning a type to the variable 'if_condition_123731' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'if_condition_123731', if_condition_123731)
        # SSA begins for if statement (line 609)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Compare to a Name (line 610):
        
        # Assigning a Compare to a Name (line 610):
        
        # Getting the type of 'I' (line 610)
        I_123732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 24), 'I')
        # Getting the type of 'J' (line 610)
        J_123733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 29), 'J')
        # Applying the binary operator '!=' (line 610)
        result_ne_123734 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 24), '!=', I_123732, J_123733)
        
        # Assigning a type to the variable 'mask' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 16), 'mask', result_ne_123734)
        
        # Assigning a Subscript to a Name (line 611):
        
        # Assigning a Subscript to a Name (line 611):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 611)
        mask_123735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 25), 'mask')
        # Getting the type of 'I' (line 611)
        I_123736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 23), 'I')
        # Obtaining the member '__getitem__' of a type (line 611)
        getitem___123737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 23), I_123736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 611)
        subscript_call_result_123738 = invoke(stypy.reporting.localization.Localization(__file__, 611, 23), getitem___123737, mask_123735)
        
        # Assigning a type to the variable 'od_I' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'od_I', subscript_call_result_123738)
        
        # Assigning a Subscript to a Name (line 612):
        
        # Assigning a Subscript to a Name (line 612):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 612)
        mask_123739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 25), 'mask')
        # Getting the type of 'J' (line 612)
        J_123740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 23), 'J')
        # Obtaining the member '__getitem__' of a type (line 612)
        getitem___123741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 23), J_123740, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 612)
        subscript_call_result_123742 = invoke(stypy.reporting.localization.Localization(__file__, 612, 23), getitem___123741, mask_123739)
        
        # Assigning a type to the variable 'od_J' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 16), 'od_J', subscript_call_result_123742)
        
        # Assigning a Subscript to a Name (line 613):
        
        # Assigning a Subscript to a Name (line 613):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 613)
        mask_123743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 25), 'mask')
        # Getting the type of 'V' (line 613)
        V_123744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 23), 'V')
        # Obtaining the member '__getitem__' of a type (line 613)
        getitem___123745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 23), V_123744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 613)
        subscript_call_result_123746 = invoke(stypy.reporting.localization.Localization(__file__, 613, 23), getitem___123745, mask_123743)
        
        # Assigning a type to the variable 'od_V' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 16), 'od_V', subscript_call_result_123746)
        
        # Assigning a Call to a Name (line 615):
        
        # Assigning a Call to a Name (line 615):
        
        # Call to concatenate(...): (line 615)
        # Processing the call arguments (line 615)
        
        # Obtaining an instance of the builtin type 'tuple' (line 615)
        tuple_123748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 615)
        # Adding element type (line 615)
        # Getting the type of 'I' (line 615)
        I_123749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 33), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 33), tuple_123748, I_123749)
        # Adding element type (line 615)
        # Getting the type of 'od_J' (line 615)
        od_J_123750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 36), 'od_J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 33), tuple_123748, od_J_123750)
        
        # Processing the call keyword arguments (line 615)
        kwargs_123751 = {}
        # Getting the type of 'concatenate' (line 615)
        concatenate_123747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'concatenate', False)
        # Calling concatenate(args, kwargs) (line 615)
        concatenate_call_result_123752 = invoke(stypy.reporting.localization.Localization(__file__, 615, 20), concatenate_123747, *[tuple_123748], **kwargs_123751)
        
        # Assigning a type to the variable 'I' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 16), 'I', concatenate_call_result_123752)
        
        # Assigning a Call to a Name (line 616):
        
        # Assigning a Call to a Name (line 616):
        
        # Call to concatenate(...): (line 616)
        # Processing the call arguments (line 616)
        
        # Obtaining an instance of the builtin type 'tuple' (line 616)
        tuple_123754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 616)
        # Adding element type (line 616)
        # Getting the type of 'J' (line 616)
        J_123755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 33), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 33), tuple_123754, J_123755)
        # Adding element type (line 616)
        # Getting the type of 'od_I' (line 616)
        od_I_123756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 36), 'od_I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 33), tuple_123754, od_I_123756)
        
        # Processing the call keyword arguments (line 616)
        kwargs_123757 = {}
        # Getting the type of 'concatenate' (line 616)
        concatenate_123753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'concatenate', False)
        # Calling concatenate(args, kwargs) (line 616)
        concatenate_call_result_123758 = invoke(stypy.reporting.localization.Localization(__file__, 616, 20), concatenate_123753, *[tuple_123754], **kwargs_123757)
        
        # Assigning a type to the variable 'J' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'J', concatenate_call_result_123758)
        
        # Getting the type of 'is_skew' (line 618)
        is_skew_123759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 19), 'is_skew')
        # Testing the type of an if condition (line 618)
        if_condition_123760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 16), is_skew_123759)
        # Assigning a type to the variable 'if_condition_123760' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'if_condition_123760', if_condition_123760)
        # SSA begins for if statement (line 618)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'od_V' (line 619)
        od_V_123761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'od_V')
        int_123762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 28), 'int')
        # Applying the binary operator '*=' (line 619)
        result_imul_123763 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 20), '*=', od_V_123761, int_123762)
        # Assigning a type to the variable 'od_V' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'od_V', result_imul_123763)
        
        # SSA branch for the else part of an if statement (line 618)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'is_herm' (line 620)
        is_herm_123764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'is_herm')
        # Testing the type of an if condition (line 620)
        if_condition_123765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 21), is_herm_123764)
        # Assigning a type to the variable 'if_condition_123765' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'if_condition_123765', if_condition_123765)
        # SSA begins for if statement (line 620)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 621):
        
        # Assigning a Call to a Name (line 621):
        
        # Call to conjugate(...): (line 621)
        # Processing the call keyword arguments (line 621)
        kwargs_123768 = {}
        # Getting the type of 'od_V' (line 621)
        od_V_123766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 27), 'od_V', False)
        # Obtaining the member 'conjugate' of a type (line 621)
        conjugate_123767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 27), od_V_123766, 'conjugate')
        # Calling conjugate(args, kwargs) (line 621)
        conjugate_call_result_123769 = invoke(stypy.reporting.localization.Localization(__file__, 621, 27), conjugate_123767, *[], **kwargs_123768)
        
        # Assigning a type to the variable 'od_V' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'od_V', conjugate_call_result_123769)
        # SSA join for if statement (line 620)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 618)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 623):
        
        # Assigning a Call to a Name (line 623):
        
        # Call to concatenate(...): (line 623)
        # Processing the call arguments (line 623)
        
        # Obtaining an instance of the builtin type 'tuple' (line 623)
        tuple_123771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 623)
        # Adding element type (line 623)
        # Getting the type of 'V' (line 623)
        V_123772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 33), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 33), tuple_123771, V_123772)
        # Adding element type (line 623)
        # Getting the type of 'od_V' (line 623)
        od_V_123773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 36), 'od_V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 33), tuple_123771, od_V_123773)
        
        # Processing the call keyword arguments (line 623)
        kwargs_123774 = {}
        # Getting the type of 'concatenate' (line 623)
        concatenate_123770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 20), 'concatenate', False)
        # Calling concatenate(args, kwargs) (line 623)
        concatenate_call_result_123775 = invoke(stypy.reporting.localization.Localization(__file__, 623, 20), concatenate_123770, *[tuple_123771], **kwargs_123774)
        
        # Assigning a type to the variable 'V' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'V', concatenate_call_result_123775)
        # SSA join for if statement (line 609)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 625):
        
        # Assigning a Call to a Name (line 625):
        
        # Call to coo_matrix(...): (line 625)
        # Processing the call arguments (line 625)
        
        # Obtaining an instance of the builtin type 'tuple' (line 625)
        tuple_123777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 625)
        # Adding element type (line 625)
        # Getting the type of 'V' (line 625)
        V_123778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 28), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 28), tuple_123777, V_123778)
        # Adding element type (line 625)
        
        # Obtaining an instance of the builtin type 'tuple' (line 625)
        tuple_123779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 625)
        # Adding element type (line 625)
        # Getting the type of 'I' (line 625)
        I_123780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 32), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 32), tuple_123779, I_123780)
        # Adding element type (line 625)
        # Getting the type of 'J' (line 625)
        J_123781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 35), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 32), tuple_123779, J_123781)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 28), tuple_123777, tuple_123779)
        
        # Processing the call keyword arguments (line 625)
        
        # Obtaining an instance of the builtin type 'tuple' (line 625)
        tuple_123782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 625)
        # Adding element type (line 625)
        # Getting the type of 'rows' (line 625)
        rows_123783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 47), 'rows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 47), tuple_123782, rows_123783)
        # Adding element type (line 625)
        # Getting the type of 'cols' (line 625)
        cols_123784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 53), 'cols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 47), tuple_123782, cols_123784)
        
        keyword_123785 = tuple_123782
        # Getting the type of 'dtype' (line 625)
        dtype_123786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 66), 'dtype', False)
        keyword_123787 = dtype_123786
        kwargs_123788 = {'dtype': keyword_123787, 'shape': keyword_123785}
        # Getting the type of 'coo_matrix' (line 625)
        coo_matrix_123776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 16), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 625)
        coo_matrix_call_result_123789 = invoke(stypy.reporting.localization.Localization(__file__, 625, 16), coo_matrix_123776, *[tuple_123777], **kwargs_123788)
        
        # Assigning a type to the variable 'a' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'a', coo_matrix_call_result_123789)
        # SSA branch for the else part of an if statement (line 565)
        module_type_store.open_ssa_branch('else')
        
        # Call to NotImplementedError(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 'format' (line 627)
        format_123791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 38), 'format', False)
        # Processing the call keyword arguments (line 627)
        kwargs_123792 = {}
        # Getting the type of 'NotImplementedError' (line 627)
        NotImplementedError_123790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 627)
        NotImplementedError_call_result_123793 = invoke(stypy.reporting.localization.Localization(__file__, 627, 18), NotImplementedError_123790, *[format_123791], **kwargs_123792)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 627, 12), NotImplementedError_call_result_123793, 'raise parameter', BaseException)
        # SSA join for if statement (line 565)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 502)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'a' (line 629)
        a_123794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 15), 'a')
        # Assigning a type to the variable 'stypy_return_type' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'stypy_return_type', a_123794)
        
        # ################# End of '_parse_body(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_body' in the type store
        # Getting the type of 'stypy_return_type' (line 483)
        stypy_return_type_123795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_body'
        return stypy_return_type_123795


    @norecursion
    def _write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_123796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 40), 'str', '')
        # Getting the type of 'None' (line 632)
        None_123797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 50), 'None')
        # Getting the type of 'None' (line 632)
        None_123798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 66), 'None')
        # Getting the type of 'None' (line 633)
        None_123799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 24), 'None')
        defaults = [str_123796, None_123797, None_123798, None_123799]
        # Create a new context for function '_write'
        module_type_store = module_type_store.open_function_context('_write', 632, 4, False)
        # Assigning a type to the variable 'self' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MMFile._write.__dict__.__setitem__('stypy_localization', localization)
        MMFile._write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MMFile._write.__dict__.__setitem__('stypy_type_store', module_type_store)
        MMFile._write.__dict__.__setitem__('stypy_function_name', 'MMFile._write')
        MMFile._write.__dict__.__setitem__('stypy_param_names_list', ['stream', 'a', 'comment', 'field', 'precision', 'symmetry'])
        MMFile._write.__dict__.__setitem__('stypy_varargs_param_name', None)
        MMFile._write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MMFile._write.__dict__.__setitem__('stypy_call_defaults', defaults)
        MMFile._write.__dict__.__setitem__('stypy_call_varargs', varargs)
        MMFile._write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MMFile._write.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MMFile._write', ['stream', 'a', 'comment', 'field', 'precision', 'symmetry'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write', localization, ['stream', 'a', 'comment', 'field', 'precision', 'symmetry'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'a' (line 635)
        a_123801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 22), 'a', False)
        # Getting the type of 'list' (line 635)
        list_123802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 25), 'list', False)
        # Processing the call keyword arguments (line 635)
        kwargs_123803 = {}
        # Getting the type of 'isinstance' (line 635)
        isinstance_123800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 635)
        isinstance_call_result_123804 = invoke(stypy.reporting.localization.Localization(__file__, 635, 11), isinstance_123800, *[a_123801, list_123802], **kwargs_123803)
        
        
        # Call to isinstance(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'a' (line 635)
        a_123806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 45), 'a', False)
        # Getting the type of 'ndarray' (line 635)
        ndarray_123807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 48), 'ndarray', False)
        # Processing the call keyword arguments (line 635)
        kwargs_123808 = {}
        # Getting the type of 'isinstance' (line 635)
        isinstance_123805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 34), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 635)
        isinstance_call_result_123809 = invoke(stypy.reporting.localization.Localization(__file__, 635, 34), isinstance_123805, *[a_123806, ndarray_123807], **kwargs_123808)
        
        # Applying the binary operator 'or' (line 635)
        result_or_keyword_123810 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 11), 'or', isinstance_call_result_123804, isinstance_call_result_123809)
        
        # Call to isinstance(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'a' (line 636)
        a_123812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 22), 'a', False)
        # Getting the type of 'tuple' (line 636)
        tuple_123813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 25), 'tuple', False)
        # Processing the call keyword arguments (line 636)
        kwargs_123814 = {}
        # Getting the type of 'isinstance' (line 636)
        isinstance_123811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 636)
        isinstance_call_result_123815 = invoke(stypy.reporting.localization.Localization(__file__, 636, 11), isinstance_123811, *[a_123812, tuple_123813], **kwargs_123814)
        
        # Applying the binary operator 'or' (line 635)
        result_or_keyword_123816 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 11), 'or', result_or_keyword_123810, isinstance_call_result_123815)
        
        # Call to hasattr(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'a' (line 636)
        a_123818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 43), 'a', False)
        str_123819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 46), 'str', '__array__')
        # Processing the call keyword arguments (line 636)
        kwargs_123820 = {}
        # Getting the type of 'hasattr' (line 636)
        hasattr_123817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 35), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 636)
        hasattr_call_result_123821 = invoke(stypy.reporting.localization.Localization(__file__, 636, 35), hasattr_123817, *[a_123818, str_123819], **kwargs_123820)
        
        # Applying the binary operator 'or' (line 635)
        result_or_keyword_123822 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 11), 'or', result_or_keyword_123816, hasattr_call_result_123821)
        
        # Testing the type of an if condition (line 635)
        if_condition_123823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 635, 8), result_or_keyword_123822)
        # Assigning a type to the variable 'if_condition_123823' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'if_condition_123823', if_condition_123823)
        # SSA begins for if statement (line 635)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 637):
        
        # Assigning a Attribute to a Name (line 637):
        # Getting the type of 'self' (line 637)
        self_123824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 18), 'self')
        # Obtaining the member 'FORMAT_ARRAY' of a type (line 637)
        FORMAT_ARRAY_123825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 18), self_123824, 'FORMAT_ARRAY')
        # Assigning a type to the variable 'rep' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'rep', FORMAT_ARRAY_123825)
        
        # Assigning a Call to a Name (line 638):
        
        # Assigning a Call to a Name (line 638):
        
        # Call to asarray(...): (line 638)
        # Processing the call arguments (line 638)
        # Getting the type of 'a' (line 638)
        a_123827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 24), 'a', False)
        # Processing the call keyword arguments (line 638)
        kwargs_123828 = {}
        # Getting the type of 'asarray' (line 638)
        asarray_123826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 16), 'asarray', False)
        # Calling asarray(args, kwargs) (line 638)
        asarray_call_result_123829 = invoke(stypy.reporting.localization.Localization(__file__, 638, 16), asarray_123826, *[a_123827], **kwargs_123828)
        
        # Assigning a type to the variable 'a' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'a', asarray_call_result_123829)
        
        
        
        # Call to len(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'a' (line 639)
        a_123831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 19), 'a', False)
        # Obtaining the member 'shape' of a type (line 639)
        shape_123832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 19), a_123831, 'shape')
        # Processing the call keyword arguments (line 639)
        kwargs_123833 = {}
        # Getting the type of 'len' (line 639)
        len_123830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 15), 'len', False)
        # Calling len(args, kwargs) (line 639)
        len_call_result_123834 = invoke(stypy.reporting.localization.Localization(__file__, 639, 15), len_123830, *[shape_123832], **kwargs_123833)
        
        int_123835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 31), 'int')
        # Applying the binary operator '!=' (line 639)
        result_ne_123836 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 15), '!=', len_call_result_123834, int_123835)
        
        # Testing the type of an if condition (line 639)
        if_condition_123837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 12), result_ne_123836)
        # Assigning a type to the variable 'if_condition_123837' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'if_condition_123837', if_condition_123837)
        # SSA begins for if statement (line 639)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 640)
        # Processing the call arguments (line 640)
        str_123839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 33), 'str', 'Expected 2 dimensional array')
        # Processing the call keyword arguments (line 640)
        kwargs_123840 = {}
        # Getting the type of 'ValueError' (line 640)
        ValueError_123838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 640)
        ValueError_call_result_123841 = invoke(stypy.reporting.localization.Localization(__file__, 640, 22), ValueError_123838, *[str_123839], **kwargs_123840)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 640, 16), ValueError_call_result_123841, 'raise parameter', BaseException)
        # SSA join for if statement (line 639)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 641):
        
        # Assigning a Subscript to a Name (line 641):
        
        # Obtaining the type of the subscript
        int_123842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 12), 'int')
        # Getting the type of 'a' (line 641)
        a_123843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 25), 'a')
        # Obtaining the member 'shape' of a type (line 641)
        shape_123844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 25), a_123843, 'shape')
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___123845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 12), shape_123844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 641)
        subscript_call_result_123846 = invoke(stypy.reporting.localization.Localization(__file__, 641, 12), getitem___123845, int_123842)
        
        # Assigning a type to the variable 'tuple_var_assignment_122321' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'tuple_var_assignment_122321', subscript_call_result_123846)
        
        # Assigning a Subscript to a Name (line 641):
        
        # Obtaining the type of the subscript
        int_123847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 12), 'int')
        # Getting the type of 'a' (line 641)
        a_123848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 25), 'a')
        # Obtaining the member 'shape' of a type (line 641)
        shape_123849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 25), a_123848, 'shape')
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___123850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 12), shape_123849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 641)
        subscript_call_result_123851 = invoke(stypy.reporting.localization.Localization(__file__, 641, 12), getitem___123850, int_123847)
        
        # Assigning a type to the variable 'tuple_var_assignment_122322' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'tuple_var_assignment_122322', subscript_call_result_123851)
        
        # Assigning a Name to a Name (line 641):
        # Getting the type of 'tuple_var_assignment_122321' (line 641)
        tuple_var_assignment_122321_123852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'tuple_var_assignment_122321')
        # Assigning a type to the variable 'rows' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'rows', tuple_var_assignment_122321_123852)
        
        # Assigning a Name to a Name (line 641):
        # Getting the type of 'tuple_var_assignment_122322' (line 641)
        tuple_var_assignment_122322_123853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 12), 'tuple_var_assignment_122322')
        # Assigning a type to the variable 'cols' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 18), 'cols', tuple_var_assignment_122322_123853)
        
        # Type idiom detected: calculating its left and rigth part (line 643)
        # Getting the type of 'field' (line 643)
        field_123854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'field')
        # Getting the type of 'None' (line 643)
        None_123855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 28), 'None')
        
        (may_be_123856, more_types_in_union_123857) = may_not_be_none(field_123854, None_123855)

        if may_be_123856:

            if more_types_in_union_123857:
                # Runtime conditional SSA (line 643)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'field' (line 645)
            field_123858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 19), 'field')
            # Getting the type of 'self' (line 645)
            self_123859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 28), 'self')
            # Obtaining the member 'FIELD_INTEGER' of a type (line 645)
            FIELD_INTEGER_123860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 28), self_123859, 'FIELD_INTEGER')
            # Applying the binary operator '==' (line 645)
            result_eq_123861 = python_operator(stypy.reporting.localization.Localization(__file__, 645, 19), '==', field_123858, FIELD_INTEGER_123860)
            
            # Testing the type of an if condition (line 645)
            if_condition_123862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 645, 16), result_eq_123861)
            # Assigning a type to the variable 'if_condition_123862' (line 645)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 16), 'if_condition_123862', if_condition_123862)
            # SSA begins for if statement (line 645)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to can_cast(...): (line 646)
            # Processing the call arguments (line 646)
            # Getting the type of 'a' (line 646)
            a_123864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 36), 'a', False)
            # Obtaining the member 'dtype' of a type (line 646)
            dtype_123865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 36), a_123864, 'dtype')
            str_123866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 45), 'str', 'intp')
            # Processing the call keyword arguments (line 646)
            kwargs_123867 = {}
            # Getting the type of 'can_cast' (line 646)
            can_cast_123863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 27), 'can_cast', False)
            # Calling can_cast(args, kwargs) (line 646)
            can_cast_call_result_123868 = invoke(stypy.reporting.localization.Localization(__file__, 646, 27), can_cast_123863, *[dtype_123865, str_123866], **kwargs_123867)
            
            # Applying the 'not' unary operator (line 646)
            result_not__123869 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 23), 'not', can_cast_call_result_123868)
            
            # Testing the type of an if condition (line 646)
            if_condition_123870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 20), result_not__123869)
            # Assigning a type to the variable 'if_condition_123870' (line 646)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 20), 'if_condition_123870', if_condition_123870)
            # SSA begins for if statement (line 646)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to OverflowError(...): (line 647)
            # Processing the call arguments (line 647)
            str_123872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 44), 'str', "mmwrite does not support integer dtypes larger than native 'intp'.")
            # Processing the call keyword arguments (line 647)
            kwargs_123873 = {}
            # Getting the type of 'OverflowError' (line 647)
            OverflowError_123871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 30), 'OverflowError', False)
            # Calling OverflowError(args, kwargs) (line 647)
            OverflowError_call_result_123874 = invoke(stypy.reporting.localization.Localization(__file__, 647, 30), OverflowError_123871, *[str_123872], **kwargs_123873)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 647, 24), OverflowError_call_result_123874, 'raise parameter', BaseException)
            # SSA join for if statement (line 646)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 649):
            
            # Assigning a Call to a Name (line 649):
            
            # Call to astype(...): (line 649)
            # Processing the call arguments (line 649)
            str_123877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 33), 'str', 'intp')
            # Processing the call keyword arguments (line 649)
            kwargs_123878 = {}
            # Getting the type of 'a' (line 649)
            a_123875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 24), 'a', False)
            # Obtaining the member 'astype' of a type (line 649)
            astype_123876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 24), a_123875, 'astype')
            # Calling astype(args, kwargs) (line 649)
            astype_call_result_123879 = invoke(stypy.reporting.localization.Localization(__file__, 649, 24), astype_123876, *[str_123877], **kwargs_123878)
            
            # Assigning a type to the variable 'a' (line 649)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 20), 'a', astype_call_result_123879)
            # SSA branch for the else part of an if statement (line 645)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'field' (line 650)
            field_123880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 21), 'field')
            # Getting the type of 'self' (line 650)
            self_123881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 30), 'self')
            # Obtaining the member 'FIELD_REAL' of a type (line 650)
            FIELD_REAL_123882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 30), self_123881, 'FIELD_REAL')
            # Applying the binary operator '==' (line 650)
            result_eq_123883 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 21), '==', field_123880, FIELD_REAL_123882)
            
            # Testing the type of an if condition (line 650)
            if_condition_123884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 21), result_eq_123883)
            # Assigning a type to the variable 'if_condition_123884' (line 650)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 21), 'if_condition_123884', if_condition_123884)
            # SSA begins for if statement (line 650)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Getting the type of 'a' (line 651)
            a_123885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 23), 'a')
            # Obtaining the member 'dtype' of a type (line 651)
            dtype_123886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 23), a_123885, 'dtype')
            # Obtaining the member 'char' of a type (line 651)
            char_123887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 23), dtype_123886, 'char')
            str_123888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 43), 'str', 'fd')
            # Applying the binary operator 'notin' (line 651)
            result_contains_123889 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 23), 'notin', char_123887, str_123888)
            
            # Testing the type of an if condition (line 651)
            if_condition_123890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 20), result_contains_123889)
            # Assigning a type to the variable 'if_condition_123890' (line 651)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 20), 'if_condition_123890', if_condition_123890)
            # SSA begins for if statement (line 651)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 652):
            
            # Assigning a Call to a Name (line 652):
            
            # Call to astype(...): (line 652)
            # Processing the call arguments (line 652)
            str_123893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 37), 'str', 'd')
            # Processing the call keyword arguments (line 652)
            kwargs_123894 = {}
            # Getting the type of 'a' (line 652)
            a_123891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 28), 'a', False)
            # Obtaining the member 'astype' of a type (line 652)
            astype_123892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 28), a_123891, 'astype')
            # Calling astype(args, kwargs) (line 652)
            astype_call_result_123895 = invoke(stypy.reporting.localization.Localization(__file__, 652, 28), astype_123892, *[str_123893], **kwargs_123894)
            
            # Assigning a type to the variable 'a' (line 652)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 24), 'a', astype_call_result_123895)
            # SSA join for if statement (line 651)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 650)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'field' (line 653)
            field_123896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 21), 'field')
            # Getting the type of 'self' (line 653)
            self_123897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 30), 'self')
            # Obtaining the member 'FIELD_COMPLEX' of a type (line 653)
            FIELD_COMPLEX_123898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 30), self_123897, 'FIELD_COMPLEX')
            # Applying the binary operator '==' (line 653)
            result_eq_123899 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 21), '==', field_123896, FIELD_COMPLEX_123898)
            
            # Testing the type of an if condition (line 653)
            if_condition_123900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 653, 21), result_eq_123899)
            # Assigning a type to the variable 'if_condition_123900' (line 653)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 21), 'if_condition_123900', if_condition_123900)
            # SSA begins for if statement (line 653)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Getting the type of 'a' (line 654)
            a_123901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 23), 'a')
            # Obtaining the member 'dtype' of a type (line 654)
            dtype_123902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 23), a_123901, 'dtype')
            # Obtaining the member 'char' of a type (line 654)
            char_123903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 23), dtype_123902, 'char')
            str_123904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 43), 'str', 'FD')
            # Applying the binary operator 'notin' (line 654)
            result_contains_123905 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 23), 'notin', char_123903, str_123904)
            
            # Testing the type of an if condition (line 654)
            if_condition_123906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 20), result_contains_123905)
            # Assigning a type to the variable 'if_condition_123906' (line 654)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 20), 'if_condition_123906', if_condition_123906)
            # SSA begins for if statement (line 654)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 655):
            
            # Assigning a Call to a Name (line 655):
            
            # Call to astype(...): (line 655)
            # Processing the call arguments (line 655)
            str_123909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 37), 'str', 'D')
            # Processing the call keyword arguments (line 655)
            kwargs_123910 = {}
            # Getting the type of 'a' (line 655)
            a_123907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 28), 'a', False)
            # Obtaining the member 'astype' of a type (line 655)
            astype_123908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 28), a_123907, 'astype')
            # Calling astype(args, kwargs) (line 655)
            astype_call_result_123911 = invoke(stypy.reporting.localization.Localization(__file__, 655, 28), astype_123908, *[str_123909], **kwargs_123910)
            
            # Assigning a type to the variable 'a' (line 655)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 24), 'a', astype_call_result_123911)
            # SSA join for if statement (line 654)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 653)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 650)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 645)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_123857:
                # SSA join for if statement (line 643)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 635)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to isspmatrix(...): (line 658)
        # Processing the call arguments (line 658)
        # Getting the type of 'a' (line 658)
        a_123913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 30), 'a', False)
        # Processing the call keyword arguments (line 658)
        kwargs_123914 = {}
        # Getting the type of 'isspmatrix' (line 658)
        isspmatrix_123912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 19), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 658)
        isspmatrix_call_result_123915 = invoke(stypy.reporting.localization.Localization(__file__, 658, 19), isspmatrix_123912, *[a_123913], **kwargs_123914)
        
        # Applying the 'not' unary operator (line 658)
        result_not__123916 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 15), 'not', isspmatrix_call_result_123915)
        
        # Testing the type of an if condition (line 658)
        if_condition_123917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 12), result_not__123916)
        # Assigning a type to the variable 'if_condition_123917' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'if_condition_123917', if_condition_123917)
        # SSA begins for if statement (line 658)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 659)
        # Processing the call arguments (line 659)
        str_123919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 33), 'str', 'unknown matrix type: %s')
        
        # Call to type(...): (line 659)
        # Processing the call arguments (line 659)
        # Getting the type of 'a' (line 659)
        a_123921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 66), 'a', False)
        # Processing the call keyword arguments (line 659)
        kwargs_123922 = {}
        # Getting the type of 'type' (line 659)
        type_123920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 61), 'type', False)
        # Calling type(args, kwargs) (line 659)
        type_call_result_123923 = invoke(stypy.reporting.localization.Localization(__file__, 659, 61), type_123920, *[a_123921], **kwargs_123922)
        
        # Applying the binary operator '%' (line 659)
        result_mod_123924 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 33), '%', str_123919, type_call_result_123923)
        
        # Processing the call keyword arguments (line 659)
        kwargs_123925 = {}
        # Getting the type of 'ValueError' (line 659)
        ValueError_123918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 659)
        ValueError_call_result_123926 = invoke(stypy.reporting.localization.Localization(__file__, 659, 22), ValueError_123918, *[result_mod_123924], **kwargs_123925)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 659, 16), ValueError_call_result_123926, 'raise parameter', BaseException)
        # SSA join for if statement (line 658)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 660):
        
        # Assigning a Str to a Name (line 660):
        str_123927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 18), 'str', 'coordinate')
        # Assigning a type to the variable 'rep' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'rep', str_123927)
        
        # Assigning a Attribute to a Tuple (line 661):
        
        # Assigning a Subscript to a Name (line 661):
        
        # Obtaining the type of the subscript
        int_123928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 12), 'int')
        # Getting the type of 'a' (line 661)
        a_123929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 25), 'a')
        # Obtaining the member 'shape' of a type (line 661)
        shape_123930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 25), a_123929, 'shape')
        # Obtaining the member '__getitem__' of a type (line 661)
        getitem___123931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 12), shape_123930, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 661)
        subscript_call_result_123932 = invoke(stypy.reporting.localization.Localization(__file__, 661, 12), getitem___123931, int_123928)
        
        # Assigning a type to the variable 'tuple_var_assignment_122323' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'tuple_var_assignment_122323', subscript_call_result_123932)
        
        # Assigning a Subscript to a Name (line 661):
        
        # Obtaining the type of the subscript
        int_123933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 12), 'int')
        # Getting the type of 'a' (line 661)
        a_123934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 25), 'a')
        # Obtaining the member 'shape' of a type (line 661)
        shape_123935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 25), a_123934, 'shape')
        # Obtaining the member '__getitem__' of a type (line 661)
        getitem___123936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 12), shape_123935, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 661)
        subscript_call_result_123937 = invoke(stypy.reporting.localization.Localization(__file__, 661, 12), getitem___123936, int_123933)
        
        # Assigning a type to the variable 'tuple_var_assignment_122324' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'tuple_var_assignment_122324', subscript_call_result_123937)
        
        # Assigning a Name to a Name (line 661):
        # Getting the type of 'tuple_var_assignment_122323' (line 661)
        tuple_var_assignment_122323_123938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'tuple_var_assignment_122323')
        # Assigning a type to the variable 'rows' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'rows', tuple_var_assignment_122323_123938)
        
        # Assigning a Name to a Name (line 661):
        # Getting the type of 'tuple_var_assignment_122324' (line 661)
        tuple_var_assignment_122324_123939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'tuple_var_assignment_122324')
        # Assigning a type to the variable 'cols' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 18), 'cols', tuple_var_assignment_122324_123939)
        # SSA join for if statement (line 635)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 663):
        
        # Assigning a Attribute to a Name (line 663):
        # Getting the type of 'a' (line 663)
        a_123940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 19), 'a')
        # Obtaining the member 'dtype' of a type (line 663)
        dtype_123941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 19), a_123940, 'dtype')
        # Obtaining the member 'char' of a type (line 663)
        char_123942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 19), dtype_123941, 'char')
        # Assigning a type to the variable 'typecode' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'typecode', char_123942)
        
        # Type idiom detected: calculating its left and rigth part (line 665)
        # Getting the type of 'precision' (line 665)
        precision_123943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 11), 'precision')
        # Getting the type of 'None' (line 665)
        None_123944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 24), 'None')
        
        (may_be_123945, more_types_in_union_123946) = may_be_none(precision_123943, None_123944)

        if may_be_123945:

            if more_types_in_union_123946:
                # Runtime conditional SSA (line 665)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'typecode' (line 666)
            typecode_123947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'typecode')
            str_123948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 27), 'str', 'fF')
            # Applying the binary operator 'in' (line 666)
            result_contains_123949 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 15), 'in', typecode_123947, str_123948)
            
            # Testing the type of an if condition (line 666)
            if_condition_123950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 12), result_contains_123949)
            # Assigning a type to the variable 'if_condition_123950' (line 666)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'if_condition_123950', if_condition_123950)
            # SSA begins for if statement (line 666)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 667):
            
            # Assigning a Num to a Name (line 667):
            int_123951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 28), 'int')
            # Assigning a type to the variable 'precision' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'precision', int_123951)
            # SSA branch for the else part of an if statement (line 666)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 669):
            
            # Assigning a Num to a Name (line 669):
            int_123952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 28), 'int')
            # Assigning a type to the variable 'precision' (line 669)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 16), 'precision', int_123952)
            # SSA join for if statement (line 666)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_123946:
                # SSA join for if statement (line 665)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 671)
        # Getting the type of 'field' (line 671)
        field_123953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 11), 'field')
        # Getting the type of 'None' (line 671)
        None_123954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 20), 'None')
        
        (may_be_123955, more_types_in_union_123956) = may_be_none(field_123953, None_123954)

        if may_be_123955:

            if more_types_in_union_123956:
                # Runtime conditional SSA (line 671)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 672):
            
            # Assigning a Attribute to a Name (line 672):
            # Getting the type of 'a' (line 672)
            a_123957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 19), 'a')
            # Obtaining the member 'dtype' of a type (line 672)
            dtype_123958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 19), a_123957, 'dtype')
            # Obtaining the member 'kind' of a type (line 672)
            kind_123959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 19), dtype_123958, 'kind')
            # Assigning a type to the variable 'kind' (line 672)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'kind', kind_123959)
            
            
            # Getting the type of 'kind' (line 673)
            kind_123960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 15), 'kind')
            str_123961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 23), 'str', 'i')
            # Applying the binary operator '==' (line 673)
            result_eq_123962 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 15), '==', kind_123960, str_123961)
            
            # Testing the type of an if condition (line 673)
            if_condition_123963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 673, 12), result_eq_123962)
            # Assigning a type to the variable 'if_condition_123963' (line 673)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'if_condition_123963', if_condition_123963)
            # SSA begins for if statement (line 673)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to can_cast(...): (line 674)
            # Processing the call arguments (line 674)
            # Getting the type of 'a' (line 674)
            a_123965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 32), 'a', False)
            # Obtaining the member 'dtype' of a type (line 674)
            dtype_123966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 32), a_123965, 'dtype')
            str_123967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 41), 'str', 'intp')
            # Processing the call keyword arguments (line 674)
            kwargs_123968 = {}
            # Getting the type of 'can_cast' (line 674)
            can_cast_123964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 23), 'can_cast', False)
            # Calling can_cast(args, kwargs) (line 674)
            can_cast_call_result_123969 = invoke(stypy.reporting.localization.Localization(__file__, 674, 23), can_cast_123964, *[dtype_123966, str_123967], **kwargs_123968)
            
            # Applying the 'not' unary operator (line 674)
            result_not__123970 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 19), 'not', can_cast_call_result_123969)
            
            # Testing the type of an if condition (line 674)
            if_condition_123971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 16), result_not__123970)
            # Assigning a type to the variable 'if_condition_123971' (line 674)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'if_condition_123971', if_condition_123971)
            # SSA begins for if statement (line 674)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to OverflowError(...): (line 675)
            # Processing the call arguments (line 675)
            str_123973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 40), 'str', "mmwrite does not support integer dtypes larger than native 'intp'.")
            # Processing the call keyword arguments (line 675)
            kwargs_123974 = {}
            # Getting the type of 'OverflowError' (line 675)
            OverflowError_123972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 26), 'OverflowError', False)
            # Calling OverflowError(args, kwargs) (line 675)
            OverflowError_call_result_123975 = invoke(stypy.reporting.localization.Localization(__file__, 675, 26), OverflowError_123972, *[str_123973], **kwargs_123974)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 675, 20), OverflowError_call_result_123975, 'raise parameter', BaseException)
            # SSA join for if statement (line 674)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Str to a Name (line 677):
            
            # Assigning a Str to a Name (line 677):
            str_123976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 24), 'str', 'integer')
            # Assigning a type to the variable 'field' (line 677)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'field', str_123976)
            # SSA branch for the else part of an if statement (line 673)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'kind' (line 678)
            kind_123977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 17), 'kind')
            str_123978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 25), 'str', 'f')
            # Applying the binary operator '==' (line 678)
            result_eq_123979 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 17), '==', kind_123977, str_123978)
            
            # Testing the type of an if condition (line 678)
            if_condition_123980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 678, 17), result_eq_123979)
            # Assigning a type to the variable 'if_condition_123980' (line 678)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 17), 'if_condition_123980', if_condition_123980)
            # SSA begins for if statement (line 678)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 679):
            
            # Assigning a Str to a Name (line 679):
            str_123981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 24), 'str', 'real')
            # Assigning a type to the variable 'field' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'field', str_123981)
            # SSA branch for the else part of an if statement (line 678)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'kind' (line 680)
            kind_123982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 17), 'kind')
            str_123983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 25), 'str', 'c')
            # Applying the binary operator '==' (line 680)
            result_eq_123984 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 17), '==', kind_123982, str_123983)
            
            # Testing the type of an if condition (line 680)
            if_condition_123985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 680, 17), result_eq_123984)
            # Assigning a type to the variable 'if_condition_123985' (line 680)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 17), 'if_condition_123985', if_condition_123985)
            # SSA begins for if statement (line 680)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 681):
            
            # Assigning a Str to a Name (line 681):
            str_123986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 24), 'str', 'complex')
            # Assigning a type to the variable 'field' (line 681)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'field', str_123986)
            # SSA branch for the else part of an if statement (line 680)
            module_type_store.open_ssa_branch('else')
            
            # Call to TypeError(...): (line 683)
            # Processing the call arguments (line 683)
            str_123988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 32), 'str', 'unexpected dtype kind ')
            # Getting the type of 'kind' (line 683)
            kind_123989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 59), 'kind', False)
            # Applying the binary operator '+' (line 683)
            result_add_123990 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 32), '+', str_123988, kind_123989)
            
            # Processing the call keyword arguments (line 683)
            kwargs_123991 = {}
            # Getting the type of 'TypeError' (line 683)
            TypeError_123987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 683)
            TypeError_call_result_123992 = invoke(stypy.reporting.localization.Localization(__file__, 683, 22), TypeError_123987, *[result_add_123990], **kwargs_123991)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 683, 16), TypeError_call_result_123992, 'raise parameter', BaseException)
            # SSA join for if statement (line 680)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 678)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 673)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_123956:
                # SSA join for if statement (line 671)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 685)
        # Getting the type of 'symmetry' (line 685)
        symmetry_123993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 11), 'symmetry')
        # Getting the type of 'None' (line 685)
        None_123994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 23), 'None')
        
        (may_be_123995, more_types_in_union_123996) = may_be_none(symmetry_123993, None_123994)

        if may_be_123995:

            if more_types_in_union_123996:
                # Runtime conditional SSA (line 685)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 686):
            
            # Assigning a Call to a Name (line 686):
            
            # Call to _get_symmetry(...): (line 686)
            # Processing the call arguments (line 686)
            # Getting the type of 'a' (line 686)
            a_123999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 42), 'a', False)
            # Processing the call keyword arguments (line 686)
            kwargs_124000 = {}
            # Getting the type of 'self' (line 686)
            self_123997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 23), 'self', False)
            # Obtaining the member '_get_symmetry' of a type (line 686)
            _get_symmetry_123998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 23), self_123997, '_get_symmetry')
            # Calling _get_symmetry(args, kwargs) (line 686)
            _get_symmetry_call_result_124001 = invoke(stypy.reporting.localization.Localization(__file__, 686, 23), _get_symmetry_123998, *[a_123999], **kwargs_124000)
            
            # Assigning a type to the variable 'symmetry' (line 686)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'symmetry', _get_symmetry_call_result_124001)

            if more_types_in_union_123996:
                # SSA join for if statement (line 685)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _validate_format(...): (line 689)
        # Processing the call arguments (line 689)
        # Getting the type of 'rep' (line 689)
        rep_124005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 40), 'rep', False)
        # Processing the call keyword arguments (line 689)
        kwargs_124006 = {}
        # Getting the type of 'self' (line 689)
        self_124002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'self', False)
        # Obtaining the member '__class__' of a type (line 689)
        class___124003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), self_124002, '__class__')
        # Obtaining the member '_validate_format' of a type (line 689)
        _validate_format_124004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), class___124003, '_validate_format')
        # Calling _validate_format(args, kwargs) (line 689)
        _validate_format_call_result_124007 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), _validate_format_124004, *[rep_124005], **kwargs_124006)
        
        
        # Call to _validate_field(...): (line 690)
        # Processing the call arguments (line 690)
        # Getting the type of 'field' (line 690)
        field_124011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 39), 'field', False)
        # Processing the call keyword arguments (line 690)
        kwargs_124012 = {}
        # Getting the type of 'self' (line 690)
        self_124008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'self', False)
        # Obtaining the member '__class__' of a type (line 690)
        class___124009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 8), self_124008, '__class__')
        # Obtaining the member '_validate_field' of a type (line 690)
        _validate_field_124010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 8), class___124009, '_validate_field')
        # Calling _validate_field(args, kwargs) (line 690)
        _validate_field_call_result_124013 = invoke(stypy.reporting.localization.Localization(__file__, 690, 8), _validate_field_124010, *[field_124011], **kwargs_124012)
        
        
        # Call to _validate_symmetry(...): (line 691)
        # Processing the call arguments (line 691)
        # Getting the type of 'symmetry' (line 691)
        symmetry_124017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 42), 'symmetry', False)
        # Processing the call keyword arguments (line 691)
        kwargs_124018 = {}
        # Getting the type of 'self' (line 691)
        self_124014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'self', False)
        # Obtaining the member '__class__' of a type (line 691)
        class___124015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 8), self_124014, '__class__')
        # Obtaining the member '_validate_symmetry' of a type (line 691)
        _validate_symmetry_124016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 8), class___124015, '_validate_symmetry')
        # Calling _validate_symmetry(args, kwargs) (line 691)
        _validate_symmetry_call_result_124019 = invoke(stypy.reporting.localization.Localization(__file__, 691, 8), _validate_symmetry_124016, *[symmetry_124017], **kwargs_124018)
        
        
        # Call to write(...): (line 694)
        # Processing the call arguments (line 694)
        
        # Call to asbytes(...): (line 694)
        # Processing the call arguments (line 694)
        
        # Call to format(...): (line 694)
        # Processing the call arguments (line 694)
        # Getting the type of 'rep' (line 694)
        rep_124025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 74), 'rep', False)
        # Getting the type of 'field' (line 695)
        field_124026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'field', False)
        # Getting the type of 'symmetry' (line 695)
        symmetry_124027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 19), 'symmetry', False)
        # Processing the call keyword arguments (line 694)
        kwargs_124028 = {}
        str_124023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 29), 'str', '%%MatrixMarket matrix {0} {1} {2}\n')
        # Obtaining the member 'format' of a type (line 694)
        format_124024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 29), str_124023, 'format')
        # Calling format(args, kwargs) (line 694)
        format_call_result_124029 = invoke(stypy.reporting.localization.Localization(__file__, 694, 29), format_124024, *[rep_124025, field_124026, symmetry_124027], **kwargs_124028)
        
        # Processing the call keyword arguments (line 694)
        kwargs_124030 = {}
        # Getting the type of 'asbytes' (line 694)
        asbytes_124022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 21), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 694)
        asbytes_call_result_124031 = invoke(stypy.reporting.localization.Localization(__file__, 694, 21), asbytes_124022, *[format_call_result_124029], **kwargs_124030)
        
        # Processing the call keyword arguments (line 694)
        kwargs_124032 = {}
        # Getting the type of 'stream' (line 694)
        stream_124020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'stream', False)
        # Obtaining the member 'write' of a type (line 694)
        write_124021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 8), stream_124020, 'write')
        # Calling write(args, kwargs) (line 694)
        write_call_result_124033 = invoke(stypy.reporting.localization.Localization(__file__, 694, 8), write_124021, *[asbytes_call_result_124031], **kwargs_124032)
        
        
        
        # Call to split(...): (line 698)
        # Processing the call arguments (line 698)
        str_124036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 34), 'str', '\n')
        # Processing the call keyword arguments (line 698)
        kwargs_124037 = {}
        # Getting the type of 'comment' (line 698)
        comment_124034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 20), 'comment', False)
        # Obtaining the member 'split' of a type (line 698)
        split_124035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 20), comment_124034, 'split')
        # Calling split(args, kwargs) (line 698)
        split_call_result_124038 = invoke(stypy.reporting.localization.Localization(__file__, 698, 20), split_124035, *[str_124036], **kwargs_124037)
        
        # Testing the type of a for loop iterable (line 698)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 698, 8), split_call_result_124038)
        # Getting the type of the for loop variable (line 698)
        for_loop_var_124039 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 698, 8), split_call_result_124038)
        # Assigning a type to the variable 'line' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'line', for_loop_var_124039)
        # SSA begins for a for statement (line 698)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 699)
        # Processing the call arguments (line 699)
        
        # Call to asbytes(...): (line 699)
        # Processing the call arguments (line 699)
        str_124043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 33), 'str', '%%%s\n')
        # Getting the type of 'line' (line 699)
        line_124044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 45), 'line', False)
        # Applying the binary operator '%' (line 699)
        result_mod_124045 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 33), '%', str_124043, line_124044)
        
        # Processing the call keyword arguments (line 699)
        kwargs_124046 = {}
        # Getting the type of 'asbytes' (line 699)
        asbytes_124042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 25), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 699)
        asbytes_call_result_124047 = invoke(stypy.reporting.localization.Localization(__file__, 699, 25), asbytes_124042, *[result_mod_124045], **kwargs_124046)
        
        # Processing the call keyword arguments (line 699)
        kwargs_124048 = {}
        # Getting the type of 'stream' (line 699)
        stream_124040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'stream', False)
        # Obtaining the member 'write' of a type (line 699)
        write_124041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 12), stream_124040, 'write')
        # Calling write(args, kwargs) (line 699)
        write_call_result_124049 = invoke(stypy.reporting.localization.Localization(__file__, 699, 12), write_124041, *[asbytes_call_result_124047], **kwargs_124048)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 701):
        
        # Assigning a Call to a Name (line 701):
        
        # Call to _field_template(...): (line 701)
        # Processing the call arguments (line 701)
        # Getting the type of 'field' (line 701)
        field_124052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 40), 'field', False)
        # Getting the type of 'precision' (line 701)
        precision_124053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 47), 'precision', False)
        # Processing the call keyword arguments (line 701)
        kwargs_124054 = {}
        # Getting the type of 'self' (line 701)
        self_124050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 19), 'self', False)
        # Obtaining the member '_field_template' of a type (line 701)
        _field_template_124051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 19), self_124050, '_field_template')
        # Calling _field_template(args, kwargs) (line 701)
        _field_template_call_result_124055 = invoke(stypy.reporting.localization.Localization(__file__, 701, 19), _field_template_124051, *[field_124052, precision_124053], **kwargs_124054)
        
        # Assigning a type to the variable 'template' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'template', _field_template_call_result_124055)
        
        
        # Getting the type of 'rep' (line 704)
        rep_124056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 11), 'rep')
        # Getting the type of 'self' (line 704)
        self_124057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 18), 'self')
        # Obtaining the member 'FORMAT_ARRAY' of a type (line 704)
        FORMAT_ARRAY_124058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 18), self_124057, 'FORMAT_ARRAY')
        # Applying the binary operator '==' (line 704)
        result_eq_124059 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 11), '==', rep_124056, FORMAT_ARRAY_124058)
        
        # Testing the type of an if condition (line 704)
        if_condition_124060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 704, 8), result_eq_124059)
        # Assigning a type to the variable 'if_condition_124060' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'if_condition_124060', if_condition_124060)
        # SSA begins for if statement (line 704)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 707)
        # Processing the call arguments (line 707)
        
        # Call to asbytes(...): (line 707)
        # Processing the call arguments (line 707)
        str_124064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 33), 'str', '%i %i\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 707)
        tuple_124065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 707)
        # Adding element type (line 707)
        # Getting the type of 'rows' (line 707)
        rows_124066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 46), 'rows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 46), tuple_124065, rows_124066)
        # Adding element type (line 707)
        # Getting the type of 'cols' (line 707)
        cols_124067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 52), 'cols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 46), tuple_124065, cols_124067)
        
        # Applying the binary operator '%' (line 707)
        result_mod_124068 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 33), '%', str_124064, tuple_124065)
        
        # Processing the call keyword arguments (line 707)
        kwargs_124069 = {}
        # Getting the type of 'asbytes' (line 707)
        asbytes_124063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 25), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 707)
        asbytes_call_result_124070 = invoke(stypy.reporting.localization.Localization(__file__, 707, 25), asbytes_124063, *[result_mod_124068], **kwargs_124069)
        
        # Processing the call keyword arguments (line 707)
        kwargs_124071 = {}
        # Getting the type of 'stream' (line 707)
        stream_124061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 12), 'stream', False)
        # Obtaining the member 'write' of a type (line 707)
        write_124062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 12), stream_124061, 'write')
        # Calling write(args, kwargs) (line 707)
        write_call_result_124072 = invoke(stypy.reporting.localization.Localization(__file__, 707, 12), write_124062, *[asbytes_call_result_124070], **kwargs_124071)
        
        
        
        # Getting the type of 'field' (line 709)
        field_124073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 15), 'field')
        
        # Obtaining an instance of the builtin type 'tuple' (line 709)
        tuple_124074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 709)
        # Adding element type (line 709)
        # Getting the type of 'self' (line 709)
        self_124075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 25), 'self')
        # Obtaining the member 'FIELD_INTEGER' of a type (line 709)
        FIELD_INTEGER_124076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 25), self_124075, 'FIELD_INTEGER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 25), tuple_124074, FIELD_INTEGER_124076)
        # Adding element type (line 709)
        # Getting the type of 'self' (line 709)
        self_124077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 45), 'self')
        # Obtaining the member 'FIELD_REAL' of a type (line 709)
        FIELD_REAL_124078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 45), self_124077, 'FIELD_REAL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 25), tuple_124074, FIELD_REAL_124078)
        
        # Applying the binary operator 'in' (line 709)
        result_contains_124079 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 15), 'in', field_124073, tuple_124074)
        
        # Testing the type of an if condition (line 709)
        if_condition_124080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 12), result_contains_124079)
        # Assigning a type to the variable 'if_condition_124080' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 12), 'if_condition_124080', if_condition_124080)
        # SSA begins for if statement (line 709)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'symmetry' (line 711)
        symmetry_124081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 19), 'symmetry')
        # Getting the type of 'self' (line 711)
        self_124082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 31), 'self')
        # Obtaining the member 'SYMMETRY_GENERAL' of a type (line 711)
        SYMMETRY_GENERAL_124083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 31), self_124082, 'SYMMETRY_GENERAL')
        # Applying the binary operator '==' (line 711)
        result_eq_124084 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 19), '==', symmetry_124081, SYMMETRY_GENERAL_124083)
        
        # Testing the type of an if condition (line 711)
        if_condition_124085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 711, 16), result_eq_124084)
        # Assigning a type to the variable 'if_condition_124085' (line 711)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 16), 'if_condition_124085', if_condition_124085)
        # SSA begins for if statement (line 711)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to range(...): (line 712)
        # Processing the call arguments (line 712)
        # Getting the type of 'cols' (line 712)
        cols_124087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 35), 'cols', False)
        # Processing the call keyword arguments (line 712)
        kwargs_124088 = {}
        # Getting the type of 'range' (line 712)
        range_124086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 29), 'range', False)
        # Calling range(args, kwargs) (line 712)
        range_call_result_124089 = invoke(stypy.reporting.localization.Localization(__file__, 712, 29), range_124086, *[cols_124087], **kwargs_124088)
        
        # Testing the type of a for loop iterable (line 712)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 712, 20), range_call_result_124089)
        # Getting the type of the for loop variable (line 712)
        for_loop_var_124090 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 712, 20), range_call_result_124089)
        # Assigning a type to the variable 'j' (line 712)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 20), 'j', for_loop_var_124090)
        # SSA begins for a for statement (line 712)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'rows' (line 713)
        rows_124092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 39), 'rows', False)
        # Processing the call keyword arguments (line 713)
        kwargs_124093 = {}
        # Getting the type of 'range' (line 713)
        range_124091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 33), 'range', False)
        # Calling range(args, kwargs) (line 713)
        range_call_result_124094 = invoke(stypy.reporting.localization.Localization(__file__, 713, 33), range_124091, *[rows_124092], **kwargs_124093)
        
        # Testing the type of a for loop iterable (line 713)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 713, 24), range_call_result_124094)
        # Getting the type of the for loop variable (line 713)
        for_loop_var_124095 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 713, 24), range_call_result_124094)
        # Assigning a type to the variable 'i' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 24), 'i', for_loop_var_124095)
        # SSA begins for a for statement (line 713)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 714)
        # Processing the call arguments (line 714)
        
        # Call to asbytes(...): (line 714)
        # Processing the call arguments (line 714)
        # Getting the type of 'template' (line 714)
        template_124099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 49), 'template', False)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 714)
        tuple_124100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 714)
        # Adding element type (line 714)
        # Getting the type of 'i' (line 714)
        i_124101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 62), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 62), tuple_124100, i_124101)
        # Adding element type (line 714)
        # Getting the type of 'j' (line 714)
        j_124102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 65), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 62), tuple_124100, j_124102)
        
        # Getting the type of 'a' (line 714)
        a_124103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 60), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 714)
        getitem___124104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 60), a_124103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 714)
        subscript_call_result_124105 = invoke(stypy.reporting.localization.Localization(__file__, 714, 60), getitem___124104, tuple_124100)
        
        # Applying the binary operator '%' (line 714)
        result_mod_124106 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 49), '%', template_124099, subscript_call_result_124105)
        
        # Processing the call keyword arguments (line 714)
        kwargs_124107 = {}
        # Getting the type of 'asbytes' (line 714)
        asbytes_124098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 41), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 714)
        asbytes_call_result_124108 = invoke(stypy.reporting.localization.Localization(__file__, 714, 41), asbytes_124098, *[result_mod_124106], **kwargs_124107)
        
        # Processing the call keyword arguments (line 714)
        kwargs_124109 = {}
        # Getting the type of 'stream' (line 714)
        stream_124096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 28), 'stream', False)
        # Obtaining the member 'write' of a type (line 714)
        write_124097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 28), stream_124096, 'write')
        # Calling write(args, kwargs) (line 714)
        write_call_result_124110 = invoke(stypy.reporting.localization.Localization(__file__, 714, 28), write_124097, *[asbytes_call_result_124108], **kwargs_124109)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 711)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to range(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'cols' (line 716)
        cols_124112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 35), 'cols', False)
        # Processing the call keyword arguments (line 716)
        kwargs_124113 = {}
        # Getting the type of 'range' (line 716)
        range_124111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 29), 'range', False)
        # Calling range(args, kwargs) (line 716)
        range_call_result_124114 = invoke(stypy.reporting.localization.Localization(__file__, 716, 29), range_124111, *[cols_124112], **kwargs_124113)
        
        # Testing the type of a for loop iterable (line 716)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 716, 20), range_call_result_124114)
        # Getting the type of the for loop variable (line 716)
        for_loop_var_124115 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 716, 20), range_call_result_124114)
        # Assigning a type to the variable 'j' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 20), 'j', for_loop_var_124115)
        # SSA begins for a for statement (line 716)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 717)
        # Processing the call arguments (line 717)
        # Getting the type of 'j' (line 717)
        j_124117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 39), 'j', False)
        # Getting the type of 'rows' (line 717)
        rows_124118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 42), 'rows', False)
        # Processing the call keyword arguments (line 717)
        kwargs_124119 = {}
        # Getting the type of 'range' (line 717)
        range_124116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 33), 'range', False)
        # Calling range(args, kwargs) (line 717)
        range_call_result_124120 = invoke(stypy.reporting.localization.Localization(__file__, 717, 33), range_124116, *[j_124117, rows_124118], **kwargs_124119)
        
        # Testing the type of a for loop iterable (line 717)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 717, 24), range_call_result_124120)
        # Getting the type of the for loop variable (line 717)
        for_loop_var_124121 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 717, 24), range_call_result_124120)
        # Assigning a type to the variable 'i' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 24), 'i', for_loop_var_124121)
        # SSA begins for a for statement (line 717)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 718)
        # Processing the call arguments (line 718)
        
        # Call to asbytes(...): (line 718)
        # Processing the call arguments (line 718)
        # Getting the type of 'template' (line 718)
        template_124125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 49), 'template', False)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 718)
        tuple_124126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 718)
        # Adding element type (line 718)
        # Getting the type of 'i' (line 718)
        i_124127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 62), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 62), tuple_124126, i_124127)
        # Adding element type (line 718)
        # Getting the type of 'j' (line 718)
        j_124128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 65), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 62), tuple_124126, j_124128)
        
        # Getting the type of 'a' (line 718)
        a_124129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 60), 'a', False)
        # Obtaining the member '__getitem__' of a type (line 718)
        getitem___124130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 60), a_124129, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 718)
        subscript_call_result_124131 = invoke(stypy.reporting.localization.Localization(__file__, 718, 60), getitem___124130, tuple_124126)
        
        # Applying the binary operator '%' (line 718)
        result_mod_124132 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 49), '%', template_124125, subscript_call_result_124131)
        
        # Processing the call keyword arguments (line 718)
        kwargs_124133 = {}
        # Getting the type of 'asbytes' (line 718)
        asbytes_124124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 41), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 718)
        asbytes_call_result_124134 = invoke(stypy.reporting.localization.Localization(__file__, 718, 41), asbytes_124124, *[result_mod_124132], **kwargs_124133)
        
        # Processing the call keyword arguments (line 718)
        kwargs_124135 = {}
        # Getting the type of 'stream' (line 718)
        stream_124122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 28), 'stream', False)
        # Obtaining the member 'write' of a type (line 718)
        write_124123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 28), stream_124122, 'write')
        # Calling write(args, kwargs) (line 718)
        write_call_result_124136 = invoke(stypy.reporting.localization.Localization(__file__, 718, 28), write_124123, *[asbytes_call_result_124134], **kwargs_124135)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 711)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 709)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'field' (line 720)
        field_124137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 17), 'field')
        # Getting the type of 'self' (line 720)
        self_124138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 26), 'self')
        # Obtaining the member 'FIELD_COMPLEX' of a type (line 720)
        FIELD_COMPLEX_124139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 26), self_124138, 'FIELD_COMPLEX')
        # Applying the binary operator '==' (line 720)
        result_eq_124140 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 17), '==', field_124137, FIELD_COMPLEX_124139)
        
        # Testing the type of an if condition (line 720)
        if_condition_124141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 720, 17), result_eq_124140)
        # Assigning a type to the variable 'if_condition_124141' (line 720)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 17), 'if_condition_124141', if_condition_124141)
        # SSA begins for if statement (line 720)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'symmetry' (line 722)
        symmetry_124142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 19), 'symmetry')
        # Getting the type of 'self' (line 722)
        self_124143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 31), 'self')
        # Obtaining the member 'SYMMETRY_GENERAL' of a type (line 722)
        SYMMETRY_GENERAL_124144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 31), self_124143, 'SYMMETRY_GENERAL')
        # Applying the binary operator '==' (line 722)
        result_eq_124145 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 19), '==', symmetry_124142, SYMMETRY_GENERAL_124144)
        
        # Testing the type of an if condition (line 722)
        if_condition_124146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 722, 16), result_eq_124145)
        # Assigning a type to the variable 'if_condition_124146' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 16), 'if_condition_124146', if_condition_124146)
        # SSA begins for if statement (line 722)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to range(...): (line 723)
        # Processing the call arguments (line 723)
        # Getting the type of 'cols' (line 723)
        cols_124148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 35), 'cols', False)
        # Processing the call keyword arguments (line 723)
        kwargs_124149 = {}
        # Getting the type of 'range' (line 723)
        range_124147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 29), 'range', False)
        # Calling range(args, kwargs) (line 723)
        range_call_result_124150 = invoke(stypy.reporting.localization.Localization(__file__, 723, 29), range_124147, *[cols_124148], **kwargs_124149)
        
        # Testing the type of a for loop iterable (line 723)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 723, 20), range_call_result_124150)
        # Getting the type of the for loop variable (line 723)
        for_loop_var_124151 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 723, 20), range_call_result_124150)
        # Assigning a type to the variable 'j' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 20), 'j', for_loop_var_124151)
        # SSA begins for a for statement (line 723)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 724)
        # Processing the call arguments (line 724)
        # Getting the type of 'rows' (line 724)
        rows_124153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 39), 'rows', False)
        # Processing the call keyword arguments (line 724)
        kwargs_124154 = {}
        # Getting the type of 'range' (line 724)
        range_124152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 33), 'range', False)
        # Calling range(args, kwargs) (line 724)
        range_call_result_124155 = invoke(stypy.reporting.localization.Localization(__file__, 724, 33), range_124152, *[rows_124153], **kwargs_124154)
        
        # Testing the type of a for loop iterable (line 724)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 724, 24), range_call_result_124155)
        # Getting the type of the for loop variable (line 724)
        for_loop_var_124156 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 724, 24), range_call_result_124155)
        # Assigning a type to the variable 'i' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 24), 'i', for_loop_var_124156)
        # SSA begins for a for statement (line 724)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 725):
        
        # Assigning a Subscript to a Name (line 725):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 725)
        tuple_124157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 725)
        # Adding element type (line 725)
        # Getting the type of 'i' (line 725)
        i_124158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 36), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 36), tuple_124157, i_124158)
        # Adding element type (line 725)
        # Getting the type of 'j' (line 725)
        j_124159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 39), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 36), tuple_124157, j_124159)
        
        # Getting the type of 'a' (line 725)
        a_124160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 34), 'a')
        # Obtaining the member '__getitem__' of a type (line 725)
        getitem___124161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 34), a_124160, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 725)
        subscript_call_result_124162 = invoke(stypy.reporting.localization.Localization(__file__, 725, 34), getitem___124161, tuple_124157)
        
        # Assigning a type to the variable 'aij' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 28), 'aij', subscript_call_result_124162)
        
        # Call to write(...): (line 726)
        # Processing the call arguments (line 726)
        
        # Call to asbytes(...): (line 726)
        # Processing the call arguments (line 726)
        # Getting the type of 'template' (line 726)
        template_124166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 49), 'template', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 726)
        tuple_124167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 726)
        # Adding element type (line 726)
        
        # Call to real(...): (line 726)
        # Processing the call arguments (line 726)
        # Getting the type of 'aij' (line 726)
        aij_124169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 66), 'aij', False)
        # Processing the call keyword arguments (line 726)
        kwargs_124170 = {}
        # Getting the type of 'real' (line 726)
        real_124168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 61), 'real', False)
        # Calling real(args, kwargs) (line 726)
        real_call_result_124171 = invoke(stypy.reporting.localization.Localization(__file__, 726, 61), real_124168, *[aij_124169], **kwargs_124170)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 61), tuple_124167, real_call_result_124171)
        # Adding element type (line 726)
        
        # Call to imag(...): (line 727)
        # Processing the call arguments (line 727)
        # Getting the type of 'aij' (line 727)
        aij_124173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 66), 'aij', False)
        # Processing the call keyword arguments (line 727)
        kwargs_124174 = {}
        # Getting the type of 'imag' (line 727)
        imag_124172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 61), 'imag', False)
        # Calling imag(args, kwargs) (line 727)
        imag_call_result_124175 = invoke(stypy.reporting.localization.Localization(__file__, 727, 61), imag_124172, *[aij_124173], **kwargs_124174)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 61), tuple_124167, imag_call_result_124175)
        
        # Applying the binary operator '%' (line 726)
        result_mod_124176 = python_operator(stypy.reporting.localization.Localization(__file__, 726, 49), '%', template_124166, tuple_124167)
        
        # Processing the call keyword arguments (line 726)
        kwargs_124177 = {}
        # Getting the type of 'asbytes' (line 726)
        asbytes_124165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 41), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 726)
        asbytes_call_result_124178 = invoke(stypy.reporting.localization.Localization(__file__, 726, 41), asbytes_124165, *[result_mod_124176], **kwargs_124177)
        
        # Processing the call keyword arguments (line 726)
        kwargs_124179 = {}
        # Getting the type of 'stream' (line 726)
        stream_124163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 28), 'stream', False)
        # Obtaining the member 'write' of a type (line 726)
        write_124164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 28), stream_124163, 'write')
        # Calling write(args, kwargs) (line 726)
        write_call_result_124180 = invoke(stypy.reporting.localization.Localization(__file__, 726, 28), write_124164, *[asbytes_call_result_124178], **kwargs_124179)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 722)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to range(...): (line 729)
        # Processing the call arguments (line 729)
        # Getting the type of 'cols' (line 729)
        cols_124182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 35), 'cols', False)
        # Processing the call keyword arguments (line 729)
        kwargs_124183 = {}
        # Getting the type of 'range' (line 729)
        range_124181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 29), 'range', False)
        # Calling range(args, kwargs) (line 729)
        range_call_result_124184 = invoke(stypy.reporting.localization.Localization(__file__, 729, 29), range_124181, *[cols_124182], **kwargs_124183)
        
        # Testing the type of a for loop iterable (line 729)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 729, 20), range_call_result_124184)
        # Getting the type of the for loop variable (line 729)
        for_loop_var_124185 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 729, 20), range_call_result_124184)
        # Assigning a type to the variable 'j' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 20), 'j', for_loop_var_124185)
        # SSA begins for a for statement (line 729)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 730)
        # Processing the call arguments (line 730)
        # Getting the type of 'j' (line 730)
        j_124187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 39), 'j', False)
        # Getting the type of 'rows' (line 730)
        rows_124188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 42), 'rows', False)
        # Processing the call keyword arguments (line 730)
        kwargs_124189 = {}
        # Getting the type of 'range' (line 730)
        range_124186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 33), 'range', False)
        # Calling range(args, kwargs) (line 730)
        range_call_result_124190 = invoke(stypy.reporting.localization.Localization(__file__, 730, 33), range_124186, *[j_124187, rows_124188], **kwargs_124189)
        
        # Testing the type of a for loop iterable (line 730)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 730, 24), range_call_result_124190)
        # Getting the type of the for loop variable (line 730)
        for_loop_var_124191 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 730, 24), range_call_result_124190)
        # Assigning a type to the variable 'i' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 24), 'i', for_loop_var_124191)
        # SSA begins for a for statement (line 730)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 731):
        
        # Assigning a Subscript to a Name (line 731):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 731)
        tuple_124192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 731)
        # Adding element type (line 731)
        # Getting the type of 'i' (line 731)
        i_124193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 36), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 36), tuple_124192, i_124193)
        # Adding element type (line 731)
        # Getting the type of 'j' (line 731)
        j_124194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 39), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 36), tuple_124192, j_124194)
        
        # Getting the type of 'a' (line 731)
        a_124195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 34), 'a')
        # Obtaining the member '__getitem__' of a type (line 731)
        getitem___124196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 34), a_124195, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 731)
        subscript_call_result_124197 = invoke(stypy.reporting.localization.Localization(__file__, 731, 34), getitem___124196, tuple_124192)
        
        # Assigning a type to the variable 'aij' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 28), 'aij', subscript_call_result_124197)
        
        # Call to write(...): (line 732)
        # Processing the call arguments (line 732)
        
        # Call to asbytes(...): (line 732)
        # Processing the call arguments (line 732)
        # Getting the type of 'template' (line 732)
        template_124201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 49), 'template', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 732)
        tuple_124202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 732)
        # Adding element type (line 732)
        
        # Call to real(...): (line 732)
        # Processing the call arguments (line 732)
        # Getting the type of 'aij' (line 732)
        aij_124204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 66), 'aij', False)
        # Processing the call keyword arguments (line 732)
        kwargs_124205 = {}
        # Getting the type of 'real' (line 732)
        real_124203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 61), 'real', False)
        # Calling real(args, kwargs) (line 732)
        real_call_result_124206 = invoke(stypy.reporting.localization.Localization(__file__, 732, 61), real_124203, *[aij_124204], **kwargs_124205)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 61), tuple_124202, real_call_result_124206)
        # Adding element type (line 732)
        
        # Call to imag(...): (line 733)
        # Processing the call arguments (line 733)
        # Getting the type of 'aij' (line 733)
        aij_124208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 66), 'aij', False)
        # Processing the call keyword arguments (line 733)
        kwargs_124209 = {}
        # Getting the type of 'imag' (line 733)
        imag_124207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 61), 'imag', False)
        # Calling imag(args, kwargs) (line 733)
        imag_call_result_124210 = invoke(stypy.reporting.localization.Localization(__file__, 733, 61), imag_124207, *[aij_124208], **kwargs_124209)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 61), tuple_124202, imag_call_result_124210)
        
        # Applying the binary operator '%' (line 732)
        result_mod_124211 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 49), '%', template_124201, tuple_124202)
        
        # Processing the call keyword arguments (line 732)
        kwargs_124212 = {}
        # Getting the type of 'asbytes' (line 732)
        asbytes_124200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 41), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 732)
        asbytes_call_result_124213 = invoke(stypy.reporting.localization.Localization(__file__, 732, 41), asbytes_124200, *[result_mod_124211], **kwargs_124212)
        
        # Processing the call keyword arguments (line 732)
        kwargs_124214 = {}
        # Getting the type of 'stream' (line 732)
        stream_124198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 28), 'stream', False)
        # Obtaining the member 'write' of a type (line 732)
        write_124199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 28), stream_124198, 'write')
        # Calling write(args, kwargs) (line 732)
        write_call_result_124215 = invoke(stypy.reporting.localization.Localization(__file__, 732, 28), write_124199, *[asbytes_call_result_124213], **kwargs_124214)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 722)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 720)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'field' (line 735)
        field_124216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 17), 'field')
        # Getting the type of 'self' (line 735)
        self_124217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 26), 'self')
        # Obtaining the member 'FIELD_PATTERN' of a type (line 735)
        FIELD_PATTERN_124218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 26), self_124217, 'FIELD_PATTERN')
        # Applying the binary operator '==' (line 735)
        result_eq_124219 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 17), '==', field_124216, FIELD_PATTERN_124218)
        
        # Testing the type of an if condition (line 735)
        if_condition_124220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 735, 17), result_eq_124219)
        # Assigning a type to the variable 'if_condition_124220' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 17), 'if_condition_124220', if_condition_124220)
        # SSA begins for if statement (line 735)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 736)
        # Processing the call arguments (line 736)
        str_124222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 33), 'str', 'pattern type inconsisted with dense format')
        # Processing the call keyword arguments (line 736)
        kwargs_124223 = {}
        # Getting the type of 'ValueError' (line 736)
        ValueError_124221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 736)
        ValueError_call_result_124224 = invoke(stypy.reporting.localization.Localization(__file__, 736, 22), ValueError_124221, *[str_124222], **kwargs_124223)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 736, 16), ValueError_call_result_124224, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 735)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 739)
        # Processing the call arguments (line 739)
        str_124226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 32), 'str', 'Unknown field type %s')
        # Getting the type of 'field' (line 739)
        field_124227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 58), 'field', False)
        # Applying the binary operator '%' (line 739)
        result_mod_124228 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 32), '%', str_124226, field_124227)
        
        # Processing the call keyword arguments (line 739)
        kwargs_124229 = {}
        # Getting the type of 'TypeError' (line 739)
        TypeError_124225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 739)
        TypeError_call_result_124230 = invoke(stypy.reporting.localization.Localization(__file__, 739, 22), TypeError_124225, *[result_mod_124228], **kwargs_124229)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 739, 16), TypeError_call_result_124230, 'raise parameter', BaseException)
        # SSA join for if statement (line 735)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 720)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 709)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 704)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 744):
        
        # Assigning a Call to a Name (line 744):
        
        # Call to tocoo(...): (line 744)
        # Processing the call keyword arguments (line 744)
        kwargs_124233 = {}
        # Getting the type of 'a' (line 744)
        a_124231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 18), 'a', False)
        # Obtaining the member 'tocoo' of a type (line 744)
        tocoo_124232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 18), a_124231, 'tocoo')
        # Calling tocoo(args, kwargs) (line 744)
        tocoo_call_result_124234 = invoke(stypy.reporting.localization.Localization(__file__, 744, 18), tocoo_124232, *[], **kwargs_124233)
        
        # Assigning a type to the variable 'coo' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'coo', tocoo_call_result_124234)
        
        
        # Getting the type of 'symmetry' (line 747)
        symmetry_124235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 15), 'symmetry')
        # Getting the type of 'self' (line 747)
        self_124236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 27), 'self')
        # Obtaining the member 'SYMMETRY_GENERAL' of a type (line 747)
        SYMMETRY_GENERAL_124237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 27), self_124236, 'SYMMETRY_GENERAL')
        # Applying the binary operator '!=' (line 747)
        result_ne_124238 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 15), '!=', symmetry_124235, SYMMETRY_GENERAL_124237)
        
        # Testing the type of an if condition (line 747)
        if_condition_124239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 747, 12), result_ne_124238)
        # Assigning a type to the variable 'if_condition_124239' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'if_condition_124239', if_condition_124239)
        # SSA begins for if statement (line 747)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Compare to a Name (line 748):
        
        # Assigning a Compare to a Name (line 748):
        
        # Getting the type of 'coo' (line 748)
        coo_124240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 38), 'coo')
        # Obtaining the member 'row' of a type (line 748)
        row_124241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 38), coo_124240, 'row')
        # Getting the type of 'coo' (line 748)
        coo_124242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 49), 'coo')
        # Obtaining the member 'col' of a type (line 748)
        col_124243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 49), coo_124242, 'col')
        # Applying the binary operator '>=' (line 748)
        result_ge_124244 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 38), '>=', row_124241, col_124243)
        
        # Assigning a type to the variable 'lower_triangle_mask' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 16), 'lower_triangle_mask', result_ge_124244)
        
        # Assigning a Call to a Name (line 749):
        
        # Assigning a Call to a Name (line 749):
        
        # Call to coo_matrix(...): (line 749)
        # Processing the call arguments (line 749)
        
        # Obtaining an instance of the builtin type 'tuple' (line 749)
        tuple_124246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 749)
        # Adding element type (line 749)
        
        # Obtaining the type of the subscript
        # Getting the type of 'lower_triangle_mask' (line 749)
        lower_triangle_mask_124247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 43), 'lower_triangle_mask', False)
        # Getting the type of 'coo' (line 749)
        coo_124248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 34), 'coo', False)
        # Obtaining the member 'data' of a type (line 749)
        data_124249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 34), coo_124248, 'data')
        # Obtaining the member '__getitem__' of a type (line 749)
        getitem___124250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 34), data_124249, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 749)
        subscript_call_result_124251 = invoke(stypy.reporting.localization.Localization(__file__, 749, 34), getitem___124250, lower_triangle_mask_124247)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 34), tuple_124246, subscript_call_result_124251)
        # Adding element type (line 749)
        
        # Obtaining an instance of the builtin type 'tuple' (line 750)
        tuple_124252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 750)
        # Adding element type (line 750)
        
        # Obtaining the type of the subscript
        # Getting the type of 'lower_triangle_mask' (line 750)
        lower_triangle_mask_124253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 42), 'lower_triangle_mask', False)
        # Getting the type of 'coo' (line 750)
        coo_124254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 34), 'coo', False)
        # Obtaining the member 'row' of a type (line 750)
        row_124255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 34), coo_124254, 'row')
        # Obtaining the member '__getitem__' of a type (line 750)
        getitem___124256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 34), row_124255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 750)
        subscript_call_result_124257 = invoke(stypy.reporting.localization.Localization(__file__, 750, 34), getitem___124256, lower_triangle_mask_124253)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 34), tuple_124252, subscript_call_result_124257)
        # Adding element type (line 750)
        
        # Obtaining the type of the subscript
        # Getting the type of 'lower_triangle_mask' (line 751)
        lower_triangle_mask_124258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 42), 'lower_triangle_mask', False)
        # Getting the type of 'coo' (line 751)
        coo_124259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 34), 'coo', False)
        # Obtaining the member 'col' of a type (line 751)
        col_124260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 34), coo_124259, 'col')
        # Obtaining the member '__getitem__' of a type (line 751)
        getitem___124261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 34), col_124260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 751)
        subscript_call_result_124262 = invoke(stypy.reporting.localization.Localization(__file__, 751, 34), getitem___124261, lower_triangle_mask_124258)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 34), tuple_124252, subscript_call_result_124262)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 34), tuple_124246, tuple_124252)
        
        # Processing the call keyword arguments (line 749)
        # Getting the type of 'coo' (line 752)
        coo_124263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 39), 'coo', False)
        # Obtaining the member 'shape' of a type (line 752)
        shape_124264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 39), coo_124263, 'shape')
        keyword_124265 = shape_124264
        kwargs_124266 = {'shape': keyword_124265}
        # Getting the type of 'coo_matrix' (line 749)
        coo_matrix_124245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 22), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 749)
        coo_matrix_call_result_124267 = invoke(stypy.reporting.localization.Localization(__file__, 749, 22), coo_matrix_124245, *[tuple_124246], **kwargs_124266)
        
        # Assigning a type to the variable 'coo' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 16), 'coo', coo_matrix_call_result_124267)
        # SSA join for if statement (line 747)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 755)
        # Processing the call arguments (line 755)
        
        # Call to asbytes(...): (line 755)
        # Processing the call arguments (line 755)
        str_124271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 33), 'str', '%i %i %i\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 755)
        tuple_124272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 755)
        # Adding element type (line 755)
        # Getting the type of 'rows' (line 755)
        rows_124273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 49), 'rows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 49), tuple_124272, rows_124273)
        # Adding element type (line 755)
        # Getting the type of 'cols' (line 755)
        cols_124274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 55), 'cols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 49), tuple_124272, cols_124274)
        # Adding element type (line 755)
        # Getting the type of 'coo' (line 755)
        coo_124275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 61), 'coo', False)
        # Obtaining the member 'nnz' of a type (line 755)
        nnz_124276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 61), coo_124275, 'nnz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 755, 49), tuple_124272, nnz_124276)
        
        # Applying the binary operator '%' (line 755)
        result_mod_124277 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 33), '%', str_124271, tuple_124272)
        
        # Processing the call keyword arguments (line 755)
        kwargs_124278 = {}
        # Getting the type of 'asbytes' (line 755)
        asbytes_124270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 25), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 755)
        asbytes_call_result_124279 = invoke(stypy.reporting.localization.Localization(__file__, 755, 25), asbytes_124270, *[result_mod_124277], **kwargs_124278)
        
        # Processing the call keyword arguments (line 755)
        kwargs_124280 = {}
        # Getting the type of 'stream' (line 755)
        stream_124268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'stream', False)
        # Obtaining the member 'write' of a type (line 755)
        write_124269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 12), stream_124268, 'write')
        # Calling write(args, kwargs) (line 755)
        write_call_result_124281 = invoke(stypy.reporting.localization.Localization(__file__, 755, 12), write_124269, *[asbytes_call_result_124279], **kwargs_124280)
        
        
        # Assigning a Call to a Name (line 757):
        
        # Assigning a Call to a Name (line 757):
        
        # Call to _field_template(...): (line 757)
        # Processing the call arguments (line 757)
        # Getting the type of 'field' (line 757)
        field_124284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 44), 'field', False)
        # Getting the type of 'precision' (line 757)
        precision_124285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 51), 'precision', False)
        int_124286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 61), 'int')
        # Applying the binary operator '-' (line 757)
        result_sub_124287 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 51), '-', precision_124285, int_124286)
        
        # Processing the call keyword arguments (line 757)
        kwargs_124288 = {}
        # Getting the type of 'self' (line 757)
        self_124282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 23), 'self', False)
        # Obtaining the member '_field_template' of a type (line 757)
        _field_template_124283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 23), self_124282, '_field_template')
        # Calling _field_template(args, kwargs) (line 757)
        _field_template_call_result_124289 = invoke(stypy.reporting.localization.Localization(__file__, 757, 23), _field_template_124283, *[field_124284, result_sub_124287], **kwargs_124288)
        
        # Assigning a type to the variable 'template' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 12), 'template', _field_template_call_result_124289)
        
        
        # Getting the type of 'field' (line 759)
        field_124290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 15), 'field')
        # Getting the type of 'self' (line 759)
        self_124291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 24), 'self')
        # Obtaining the member 'FIELD_PATTERN' of a type (line 759)
        FIELD_PATTERN_124292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 24), self_124291, 'FIELD_PATTERN')
        # Applying the binary operator '==' (line 759)
        result_eq_124293 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 15), '==', field_124290, FIELD_PATTERN_124292)
        
        # Testing the type of an if condition (line 759)
        if_condition_124294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 759, 12), result_eq_124293)
        # Assigning a type to the variable 'if_condition_124294' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'if_condition_124294', if_condition_124294)
        # SSA begins for if statement (line 759)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to zip(...): (line 760)
        # Processing the call arguments (line 760)
        # Getting the type of 'coo' (line 760)
        coo_124296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 32), 'coo', False)
        # Obtaining the member 'row' of a type (line 760)
        row_124297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 32), coo_124296, 'row')
        int_124298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 40), 'int')
        # Applying the binary operator '+' (line 760)
        result_add_124299 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 32), '+', row_124297, int_124298)
        
        # Getting the type of 'coo' (line 760)
        coo_124300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 43), 'coo', False)
        # Obtaining the member 'col' of a type (line 760)
        col_124301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 43), coo_124300, 'col')
        int_124302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 51), 'int')
        # Applying the binary operator '+' (line 760)
        result_add_124303 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 43), '+', col_124301, int_124302)
        
        # Processing the call keyword arguments (line 760)
        kwargs_124304 = {}
        # Getting the type of 'zip' (line 760)
        zip_124295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 28), 'zip', False)
        # Calling zip(args, kwargs) (line 760)
        zip_call_result_124305 = invoke(stypy.reporting.localization.Localization(__file__, 760, 28), zip_124295, *[result_add_124299, result_add_124303], **kwargs_124304)
        
        # Testing the type of a for loop iterable (line 760)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 760, 16), zip_call_result_124305)
        # Getting the type of the for loop variable (line 760)
        for_loop_var_124306 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 760, 16), zip_call_result_124305)
        # Assigning a type to the variable 'r' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 16), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 16), for_loop_var_124306))
        # Assigning a type to the variable 'c' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 16), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 16), for_loop_var_124306))
        # SSA begins for a for statement (line 760)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 761)
        # Processing the call arguments (line 761)
        
        # Call to asbytes(...): (line 761)
        # Processing the call arguments (line 761)
        str_124310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 41), 'str', '%i %i\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 761)
        tuple_124311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 761)
        # Adding element type (line 761)
        # Getting the type of 'r' (line 761)
        r_124312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 54), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 54), tuple_124311, r_124312)
        # Adding element type (line 761)
        # Getting the type of 'c' (line 761)
        c_124313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 57), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 54), tuple_124311, c_124313)
        
        # Applying the binary operator '%' (line 761)
        result_mod_124314 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 41), '%', str_124310, tuple_124311)
        
        # Processing the call keyword arguments (line 761)
        kwargs_124315 = {}
        # Getting the type of 'asbytes' (line 761)
        asbytes_124309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 33), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 761)
        asbytes_call_result_124316 = invoke(stypy.reporting.localization.Localization(__file__, 761, 33), asbytes_124309, *[result_mod_124314], **kwargs_124315)
        
        # Processing the call keyword arguments (line 761)
        kwargs_124317 = {}
        # Getting the type of 'stream' (line 761)
        stream_124307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 20), 'stream', False)
        # Obtaining the member 'write' of a type (line 761)
        write_124308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 20), stream_124307, 'write')
        # Calling write(args, kwargs) (line 761)
        write_call_result_124318 = invoke(stypy.reporting.localization.Localization(__file__, 761, 20), write_124308, *[asbytes_call_result_124316], **kwargs_124317)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 759)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'field' (line 762)
        field_124319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 17), 'field')
        
        # Obtaining an instance of the builtin type 'tuple' (line 762)
        tuple_124320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 762)
        # Adding element type (line 762)
        # Getting the type of 'self' (line 762)
        self_124321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 27), 'self')
        # Obtaining the member 'FIELD_INTEGER' of a type (line 762)
        FIELD_INTEGER_124322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 27), self_124321, 'FIELD_INTEGER')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 27), tuple_124320, FIELD_INTEGER_124322)
        # Adding element type (line 762)
        # Getting the type of 'self' (line 762)
        self_124323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 47), 'self')
        # Obtaining the member 'FIELD_REAL' of a type (line 762)
        FIELD_REAL_124324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 47), self_124323, 'FIELD_REAL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 27), tuple_124320, FIELD_REAL_124324)
        
        # Applying the binary operator 'in' (line 762)
        result_contains_124325 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 17), 'in', field_124319, tuple_124320)
        
        # Testing the type of an if condition (line 762)
        if_condition_124326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 17), result_contains_124325)
        # Assigning a type to the variable 'if_condition_124326' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 17), 'if_condition_124326', if_condition_124326)
        # SSA begins for if statement (line 762)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to zip(...): (line 763)
        # Processing the call arguments (line 763)
        # Getting the type of 'coo' (line 763)
        coo_124328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 35), 'coo', False)
        # Obtaining the member 'row' of a type (line 763)
        row_124329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 35), coo_124328, 'row')
        int_124330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 43), 'int')
        # Applying the binary operator '+' (line 763)
        result_add_124331 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 35), '+', row_124329, int_124330)
        
        # Getting the type of 'coo' (line 763)
        coo_124332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 46), 'coo', False)
        # Obtaining the member 'col' of a type (line 763)
        col_124333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 46), coo_124332, 'col')
        int_124334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 54), 'int')
        # Applying the binary operator '+' (line 763)
        result_add_124335 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 46), '+', col_124333, int_124334)
        
        # Getting the type of 'coo' (line 763)
        coo_124336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 57), 'coo', False)
        # Obtaining the member 'data' of a type (line 763)
        data_124337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 57), coo_124336, 'data')
        # Processing the call keyword arguments (line 763)
        kwargs_124338 = {}
        # Getting the type of 'zip' (line 763)
        zip_124327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 31), 'zip', False)
        # Calling zip(args, kwargs) (line 763)
        zip_call_result_124339 = invoke(stypy.reporting.localization.Localization(__file__, 763, 31), zip_124327, *[result_add_124331, result_add_124335, data_124337], **kwargs_124338)
        
        # Testing the type of a for loop iterable (line 763)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 763, 16), zip_call_result_124339)
        # Getting the type of the for loop variable (line 763)
        for_loop_var_124340 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 763, 16), zip_call_result_124339)
        # Assigning a type to the variable 'r' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 16), for_loop_var_124340))
        # Assigning a type to the variable 'c' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 16), for_loop_var_124340))
        # Assigning a type to the variable 'd' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 16), for_loop_var_124340))
        # SSA begins for a for statement (line 763)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 764)
        # Processing the call arguments (line 764)
        
        # Call to asbytes(...): (line 764)
        # Processing the call arguments (line 764)
        str_124344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 42), 'str', '%i %i ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 764)
        tuple_124345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 764)
        # Adding element type (line 764)
        # Getting the type of 'r' (line 764)
        r_124346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 54), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 54), tuple_124345, r_124346)
        # Adding element type (line 764)
        # Getting the type of 'c' (line 764)
        c_124347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 57), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 54), tuple_124345, c_124347)
        
        # Applying the binary operator '%' (line 764)
        result_mod_124348 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 42), '%', str_124344, tuple_124345)
        
        # Getting the type of 'template' (line 765)
        template_124349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 42), 'template', False)
        # Getting the type of 'd' (line 765)
        d_124350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 53), 'd', False)
        # Applying the binary operator '%' (line 765)
        result_mod_124351 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 42), '%', template_124349, d_124350)
        
        # Applying the binary operator '+' (line 764)
        result_add_124352 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 41), '+', result_mod_124348, result_mod_124351)
        
        # Processing the call keyword arguments (line 764)
        kwargs_124353 = {}
        # Getting the type of 'asbytes' (line 764)
        asbytes_124343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 33), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 764)
        asbytes_call_result_124354 = invoke(stypy.reporting.localization.Localization(__file__, 764, 33), asbytes_124343, *[result_add_124352], **kwargs_124353)
        
        # Processing the call keyword arguments (line 764)
        kwargs_124355 = {}
        # Getting the type of 'stream' (line 764)
        stream_124341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 20), 'stream', False)
        # Obtaining the member 'write' of a type (line 764)
        write_124342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 20), stream_124341, 'write')
        # Calling write(args, kwargs) (line 764)
        write_call_result_124356 = invoke(stypy.reporting.localization.Localization(__file__, 764, 20), write_124342, *[asbytes_call_result_124354], **kwargs_124355)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 762)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'field' (line 766)
        field_124357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 17), 'field')
        # Getting the type of 'self' (line 766)
        self_124358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 26), 'self')
        # Obtaining the member 'FIELD_COMPLEX' of a type (line 766)
        FIELD_COMPLEX_124359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 26), self_124358, 'FIELD_COMPLEX')
        # Applying the binary operator '==' (line 766)
        result_eq_124360 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 17), '==', field_124357, FIELD_COMPLEX_124359)
        
        # Testing the type of an if condition (line 766)
        if_condition_124361 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 17), result_eq_124360)
        # Assigning a type to the variable 'if_condition_124361' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 17), 'if_condition_124361', if_condition_124361)
        # SSA begins for if statement (line 766)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to zip(...): (line 767)
        # Processing the call arguments (line 767)
        # Getting the type of 'coo' (line 767)
        coo_124363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 35), 'coo', False)
        # Obtaining the member 'row' of a type (line 767)
        row_124364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 35), coo_124363, 'row')
        int_124365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 43), 'int')
        # Applying the binary operator '+' (line 767)
        result_add_124366 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 35), '+', row_124364, int_124365)
        
        # Getting the type of 'coo' (line 767)
        coo_124367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 46), 'coo', False)
        # Obtaining the member 'col' of a type (line 767)
        col_124368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 46), coo_124367, 'col')
        int_124369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 54), 'int')
        # Applying the binary operator '+' (line 767)
        result_add_124370 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 46), '+', col_124368, int_124369)
        
        # Getting the type of 'coo' (line 767)
        coo_124371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 57), 'coo', False)
        # Obtaining the member 'data' of a type (line 767)
        data_124372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 57), coo_124371, 'data')
        # Processing the call keyword arguments (line 767)
        kwargs_124373 = {}
        # Getting the type of 'zip' (line 767)
        zip_124362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 31), 'zip', False)
        # Calling zip(args, kwargs) (line 767)
        zip_call_result_124374 = invoke(stypy.reporting.localization.Localization(__file__, 767, 31), zip_124362, *[result_add_124366, result_add_124370, data_124372], **kwargs_124373)
        
        # Testing the type of a for loop iterable (line 767)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 767, 16), zip_call_result_124374)
        # Getting the type of the for loop variable (line 767)
        for_loop_var_124375 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 767, 16), zip_call_result_124374)
        # Assigning a type to the variable 'r' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 16), for_loop_var_124375))
        # Assigning a type to the variable 'c' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 16), for_loop_var_124375))
        # Assigning a type to the variable 'd' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 16), for_loop_var_124375))
        # SSA begins for a for statement (line 767)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 768)
        # Processing the call arguments (line 768)
        
        # Call to asbytes(...): (line 768)
        # Processing the call arguments (line 768)
        str_124379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 42), 'str', '%i %i ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 768)
        tuple_124380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 768)
        # Adding element type (line 768)
        # Getting the type of 'r' (line 768)
        r_124381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 54), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 54), tuple_124380, r_124381)
        # Adding element type (line 768)
        # Getting the type of 'c' (line 768)
        c_124382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 57), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 54), tuple_124380, c_124382)
        
        # Applying the binary operator '%' (line 768)
        result_mod_124383 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 42), '%', str_124379, tuple_124380)
        
        # Getting the type of 'template' (line 769)
        template_124384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 42), 'template', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 769)
        tuple_124385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 769)
        # Adding element type (line 769)
        # Getting the type of 'd' (line 769)
        d_124386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 54), 'd', False)
        # Obtaining the member 'real' of a type (line 769)
        real_124387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 54), d_124386, 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 54), tuple_124385, real_124387)
        # Adding element type (line 769)
        # Getting the type of 'd' (line 769)
        d_124388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 62), 'd', False)
        # Obtaining the member 'imag' of a type (line 769)
        imag_124389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 62), d_124388, 'imag')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 54), tuple_124385, imag_124389)
        
        # Applying the binary operator '%' (line 769)
        result_mod_124390 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 42), '%', template_124384, tuple_124385)
        
        # Applying the binary operator '+' (line 768)
        result_add_124391 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 41), '+', result_mod_124383, result_mod_124390)
        
        # Processing the call keyword arguments (line 768)
        kwargs_124392 = {}
        # Getting the type of 'asbytes' (line 768)
        asbytes_124378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 33), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 768)
        asbytes_call_result_124393 = invoke(stypy.reporting.localization.Localization(__file__, 768, 33), asbytes_124378, *[result_add_124391], **kwargs_124392)
        
        # Processing the call keyword arguments (line 768)
        kwargs_124394 = {}
        # Getting the type of 'stream' (line 768)
        stream_124376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 20), 'stream', False)
        # Obtaining the member 'write' of a type (line 768)
        write_124377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 20), stream_124376, 'write')
        # Calling write(args, kwargs) (line 768)
        write_call_result_124395 = invoke(stypy.reporting.localization.Localization(__file__, 768, 20), write_124377, *[asbytes_call_result_124393], **kwargs_124394)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 766)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 771)
        # Processing the call arguments (line 771)
        str_124397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 32), 'str', 'Unknown field type %s')
        # Getting the type of 'field' (line 771)
        field_124398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 58), 'field', False)
        # Applying the binary operator '%' (line 771)
        result_mod_124399 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 32), '%', str_124397, field_124398)
        
        # Processing the call keyword arguments (line 771)
        kwargs_124400 = {}
        # Getting the type of 'TypeError' (line 771)
        TypeError_124396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 771)
        TypeError_call_result_124401 = invoke(stypy.reporting.localization.Localization(__file__, 771, 22), TypeError_124396, *[result_mod_124399], **kwargs_124400)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 771, 16), TypeError_call_result_124401, 'raise parameter', BaseException)
        # SSA join for if statement (line 766)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 762)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 759)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 704)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write' in the type store
        # Getting the type of 'stypy_return_type' (line 632)
        stypy_return_type_124402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write'
        return stypy_return_type_124402


# Assigning a type to the variable 'MMFile' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'MMFile', MMFile)

# Assigning a Tuple to a Name (line 107):

# Obtaining an instance of the builtin type 'tuple' (line 107)
tuple_124403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 107)
# Adding element type (line 107)
str_124404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'str', '_rows')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), tuple_124403, str_124404)
# Adding element type (line 107)
str_124405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 17), 'str', '_cols')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), tuple_124403, str_124405)
# Adding element type (line 107)
str_124406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'str', '_entries')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), tuple_124403, str_124406)
# Adding element type (line 107)
str_124407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 17), 'str', '_format')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), tuple_124403, str_124407)
# Adding element type (line 107)
str_124408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 17), 'str', '_field')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), tuple_124403, str_124408)
# Adding element type (line 107)
str_124409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 17), 'str', '_symmetry')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), tuple_124403, str_124409)

# Getting the type of 'MMFile'
MMFile_124410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124410, '__slots__', tuple_124403)

# Assigning a Str to a Name (line 145):
str_124411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'str', 'coordinate')
# Getting the type of 'MMFile'
MMFile_124412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FORMAT_COORDINATE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124412, 'FORMAT_COORDINATE', str_124411)

# Assigning a Str to a Name (line 146):
str_124413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'str', 'array')
# Getting the type of 'MMFile'
MMFile_124414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FORMAT_ARRAY' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124414, 'FORMAT_ARRAY', str_124413)

# Assigning a Tuple to a Name (line 147):

# Obtaining an instance of the builtin type 'tuple' (line 147)
tuple_124415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 147)
# Adding element type (line 147)
# Getting the type of 'MMFile'
MMFile_124416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FORMAT_COORDINATE' of a type
FORMAT_COORDINATE_124417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124416, 'FORMAT_COORDINATE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), tuple_124415, FORMAT_COORDINATE_124417)
# Adding element type (line 147)
# Getting the type of 'MMFile'
MMFile_124418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FORMAT_ARRAY' of a type
FORMAT_ARRAY_124419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124418, 'FORMAT_ARRAY')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 21), tuple_124415, FORMAT_ARRAY_124419)

# Getting the type of 'MMFile'
MMFile_124420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FORMAT_VALUES' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124420, 'FORMAT_VALUES', tuple_124415)

# Assigning a Str to a Name (line 156):
str_124421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'str', 'integer')
# Getting the type of 'MMFile'
MMFile_124422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FIELD_INTEGER' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124422, 'FIELD_INTEGER', str_124421)

# Assigning a Str to a Name (line 157):
str_124423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 17), 'str', 'real')
# Getting the type of 'MMFile'
MMFile_124424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FIELD_REAL' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124424, 'FIELD_REAL', str_124423)

# Assigning a Str to a Name (line 158):
str_124425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'str', 'complex')
# Getting the type of 'MMFile'
MMFile_124426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FIELD_COMPLEX' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124426, 'FIELD_COMPLEX', str_124425)

# Assigning a Str to a Name (line 159):
str_124427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'str', 'pattern')
# Getting the type of 'MMFile'
MMFile_124428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FIELD_PATTERN' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124428, 'FIELD_PATTERN', str_124427)

# Assigning a Tuple to a Name (line 160):

# Obtaining an instance of the builtin type 'tuple' (line 160)
tuple_124429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 160)
# Adding element type (line 160)
# Getting the type of 'MMFile'
MMFile_124430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_INTEGER' of a type
FIELD_INTEGER_124431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124430, 'FIELD_INTEGER')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), tuple_124429, FIELD_INTEGER_124431)
# Adding element type (line 160)
# Getting the type of 'MMFile'
MMFile_124432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_REAL' of a type
FIELD_REAL_124433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124432, 'FIELD_REAL')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), tuple_124429, FIELD_REAL_124433)
# Adding element type (line 160)
# Getting the type of 'MMFile'
MMFile_124434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_COMPLEX' of a type
FIELD_COMPLEX_124435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124434, 'FIELD_COMPLEX')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), tuple_124429, FIELD_COMPLEX_124435)
# Adding element type (line 160)
# Getting the type of 'MMFile'
MMFile_124436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_PATTERN' of a type
FIELD_PATTERN_124437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124436, 'FIELD_PATTERN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), tuple_124429, FIELD_PATTERN_124437)

# Getting the type of 'MMFile'
MMFile_124438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'FIELD_VALUES' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124438, 'FIELD_VALUES', tuple_124429)

# Assigning a Str to a Name (line 169):
str_124439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 23), 'str', 'general')
# Getting the type of 'MMFile'
MMFile_124440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'SYMMETRY_GENERAL' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124440, 'SYMMETRY_GENERAL', str_124439)

# Assigning a Str to a Name (line 170):
str_124441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'str', 'symmetric')
# Getting the type of 'MMFile'
MMFile_124442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'SYMMETRY_SYMMETRIC' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124442, 'SYMMETRY_SYMMETRIC', str_124441)

# Assigning a Str to a Name (line 171):
str_124443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'str', 'skew-symmetric')
# Getting the type of 'MMFile'
MMFile_124444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'SYMMETRY_SKEW_SYMMETRIC' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124444, 'SYMMETRY_SKEW_SYMMETRIC', str_124443)

# Assigning a Str to a Name (line 172):
str_124445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'str', 'hermitian')
# Getting the type of 'MMFile'
MMFile_124446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'SYMMETRY_HERMITIAN' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124446, 'SYMMETRY_HERMITIAN', str_124445)

# Assigning a Tuple to a Name (line 173):

# Obtaining an instance of the builtin type 'tuple' (line 173)
tuple_124447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 173)
# Adding element type (line 173)
# Getting the type of 'MMFile'
MMFile_124448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'SYMMETRY_GENERAL' of a type
SYMMETRY_GENERAL_124449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124448, 'SYMMETRY_GENERAL')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 23), tuple_124447, SYMMETRY_GENERAL_124449)
# Adding element type (line 173)
# Getting the type of 'MMFile'
MMFile_124450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'SYMMETRY_SYMMETRIC' of a type
SYMMETRY_SYMMETRIC_124451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124450, 'SYMMETRY_SYMMETRIC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 23), tuple_124447, SYMMETRY_SYMMETRIC_124451)
# Adding element type (line 173)
# Getting the type of 'MMFile'
MMFile_124452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'SYMMETRY_SKEW_SYMMETRIC' of a type
SYMMETRY_SKEW_SYMMETRIC_124453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124452, 'SYMMETRY_SKEW_SYMMETRIC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 23), tuple_124447, SYMMETRY_SKEW_SYMMETRIC_124453)
# Adding element type (line 173)
# Getting the type of 'MMFile'
MMFile_124454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'SYMMETRY_HERMITIAN' of a type
SYMMETRY_HERMITIAN_124455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124454, 'SYMMETRY_HERMITIAN')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 23), tuple_124447, SYMMETRY_HERMITIAN_124455)

# Getting the type of 'MMFile'
MMFile_124456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'SYMMETRY_VALUES' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124456, 'SYMMETRY_VALUES', tuple_124447)

# Assigning a Dict to a Name (line 182):

# Obtaining an instance of the builtin type 'dict' (line 182)
dict_124457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 182)
# Adding element type (key, value) (line 182)
# Getting the type of 'MMFile'
MMFile_124458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_INTEGER' of a type
FIELD_INTEGER_124459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124458, 'FIELD_INTEGER')
str_124460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 38), 'str', 'intp')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 22), dict_124457, (FIELD_INTEGER_124459, str_124460))
# Adding element type (key, value) (line 182)
# Getting the type of 'MMFile'
MMFile_124461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_REAL' of a type
FIELD_REAL_124462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124461, 'FIELD_REAL')
str_124463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 35), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 22), dict_124457, (FIELD_REAL_124462, str_124463))
# Adding element type (key, value) (line 182)
# Getting the type of 'MMFile'
MMFile_124464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_COMPLEX' of a type
FIELD_COMPLEX_124465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124464, 'FIELD_COMPLEX')
str_124466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 38), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 22), dict_124457, (FIELD_COMPLEX_124465, str_124466))
# Adding element type (key, value) (line 182)
# Getting the type of 'MMFile'
MMFile_124467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Obtaining the member 'FIELD_PATTERN' of a type
FIELD_PATTERN_124468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124467, 'FIELD_PATTERN')
str_124469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 38), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 22), dict_124457, (FIELD_PATTERN_124468, str_124469))

# Getting the type of 'MMFile'
MMFile_124470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MMFile')
# Setting the type of the member 'DTYPES_BY_FIELD' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MMFile_124470, 'DTYPES_BY_FIELD', dict_124457)

@norecursion
def _is_fromfile_compatible(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_fromfile_compatible'
    module_type_store = module_type_store.open_function_context('_is_fromfile_compatible', 774, 0, False)
    
    # Passed parameters checking function
    _is_fromfile_compatible.stypy_localization = localization
    _is_fromfile_compatible.stypy_type_of_self = None
    _is_fromfile_compatible.stypy_type_store = module_type_store
    _is_fromfile_compatible.stypy_function_name = '_is_fromfile_compatible'
    _is_fromfile_compatible.stypy_param_names_list = ['stream']
    _is_fromfile_compatible.stypy_varargs_param_name = None
    _is_fromfile_compatible.stypy_kwargs_param_name = None
    _is_fromfile_compatible.stypy_call_defaults = defaults
    _is_fromfile_compatible.stypy_call_varargs = varargs
    _is_fromfile_compatible.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_fromfile_compatible', ['stream'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_fromfile_compatible', localization, ['stream'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_fromfile_compatible(...)' code ##################

    str_124471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, (-1)), 'str', "\n    Check whether `stream` is compatible with numpy.fromfile.\n\n    Passing a gzipped file object to ``fromfile/fromstring`` doesn't work with\n    Python3.\n    ")
    
    
    
    # Obtaining the type of the subscript
    int_124472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 24), 'int')
    # Getting the type of 'sys' (line 781)
    sys_124473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 781)
    version_info_124474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 7), sys_124473, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 781)
    getitem___124475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 7), version_info_124474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 781)
    subscript_call_result_124476 = invoke(stypy.reporting.localization.Localization(__file__, 781, 7), getitem___124475, int_124472)
    
    int_124477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 29), 'int')
    # Applying the binary operator '<' (line 781)
    result_lt_124478 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 7), '<', subscript_call_result_124476, int_124477)
    
    # Testing the type of an if condition (line 781)
    if_condition_124479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 781, 4), result_lt_124478)
    # Assigning a type to the variable 'if_condition_124479' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'if_condition_124479', if_condition_124479)
    # SSA begins for if statement (line 781)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 782)
    True_124480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'stypy_return_type', True_124480)
    # SSA join for if statement (line 781)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 784):
    
    # Assigning a List to a Name (line 784):
    
    # Obtaining an instance of the builtin type 'list' (line 784)
    list_124481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 784)
    
    # Assigning a type to the variable 'bad_cls' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'bad_cls', list_124481)
    
    
    # SSA begins for try-except statement (line 785)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 786, 8))
    
    # 'import gzip' statement (line 786)
    import gzip

    import_module(stypy.reporting.localization.Localization(__file__, 786, 8), 'gzip', gzip, module_type_store)
    
    
    # Call to append(...): (line 787)
    # Processing the call arguments (line 787)
    # Getting the type of 'gzip' (line 787)
    gzip_124484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 23), 'gzip', False)
    # Obtaining the member 'GzipFile' of a type (line 787)
    GzipFile_124485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 23), gzip_124484, 'GzipFile')
    # Processing the call keyword arguments (line 787)
    kwargs_124486 = {}
    # Getting the type of 'bad_cls' (line 787)
    bad_cls_124482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'bad_cls', False)
    # Obtaining the member 'append' of a type (line 787)
    append_124483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 8), bad_cls_124482, 'append')
    # Calling append(args, kwargs) (line 787)
    append_call_result_124487 = invoke(stypy.reporting.localization.Localization(__file__, 787, 8), append_124483, *[GzipFile_124485], **kwargs_124486)
    
    # SSA branch for the except part of a try statement (line 785)
    # SSA branch for the except 'ImportError' branch of a try statement (line 785)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 785)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 790)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 791, 8))
    
    # 'import bz2' statement (line 791)
    import bz2

    import_module(stypy.reporting.localization.Localization(__file__, 791, 8), 'bz2', bz2, module_type_store)
    
    
    # Call to append(...): (line 792)
    # Processing the call arguments (line 792)
    # Getting the type of 'bz2' (line 792)
    bz2_124490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 23), 'bz2', False)
    # Obtaining the member 'BZ2File' of a type (line 792)
    BZ2File_124491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 23), bz2_124490, 'BZ2File')
    # Processing the call keyword arguments (line 792)
    kwargs_124492 = {}
    # Getting the type of 'bad_cls' (line 792)
    bad_cls_124488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'bad_cls', False)
    # Obtaining the member 'append' of a type (line 792)
    append_124489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), bad_cls_124488, 'append')
    # Calling append(args, kwargs) (line 792)
    append_call_result_124493 = invoke(stypy.reporting.localization.Localization(__file__, 792, 8), append_124489, *[BZ2File_124491], **kwargs_124492)
    
    # SSA branch for the except part of a try statement (line 790)
    # SSA branch for the except 'ImportError' branch of a try statement (line 790)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 790)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 796):
    
    # Assigning a Call to a Name (line 796):
    
    # Call to tuple(...): (line 796)
    # Processing the call arguments (line 796)
    # Getting the type of 'bad_cls' (line 796)
    bad_cls_124495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 20), 'bad_cls', False)
    # Processing the call keyword arguments (line 796)
    kwargs_124496 = {}
    # Getting the type of 'tuple' (line 796)
    tuple_124494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 14), 'tuple', False)
    # Calling tuple(args, kwargs) (line 796)
    tuple_call_result_124497 = invoke(stypy.reporting.localization.Localization(__file__, 796, 14), tuple_124494, *[bad_cls_124495], **kwargs_124496)
    
    # Assigning a type to the variable 'bad_cls' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'bad_cls', tuple_call_result_124497)
    
    
    # Call to isinstance(...): (line 797)
    # Processing the call arguments (line 797)
    # Getting the type of 'stream' (line 797)
    stream_124499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 26), 'stream', False)
    # Getting the type of 'bad_cls' (line 797)
    bad_cls_124500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 34), 'bad_cls', False)
    # Processing the call keyword arguments (line 797)
    kwargs_124501 = {}
    # Getting the type of 'isinstance' (line 797)
    isinstance_124498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 797)
    isinstance_call_result_124502 = invoke(stypy.reporting.localization.Localization(__file__, 797, 15), isinstance_124498, *[stream_124499, bad_cls_124500], **kwargs_124501)
    
    # Applying the 'not' unary operator (line 797)
    result_not__124503 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 11), 'not', isinstance_call_result_124502)
    
    # Assigning a type to the variable 'stypy_return_type' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'stypy_return_type', result_not__124503)
    
    # ################# End of '_is_fromfile_compatible(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_fromfile_compatible' in the type store
    # Getting the type of 'stypy_return_type' (line 774)
    stypy_return_type_124504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124504)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_fromfile_compatible'
    return stypy_return_type_124504

# Assigning a type to the variable '_is_fromfile_compatible' (line 774)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 0), '_is_fromfile_compatible', _is_fromfile_compatible)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 802, 4))
    
    # 'import time' statement (line 802)
    import time

    import_module(stypy.reporting.localization.Localization(__file__, 802, 4), 'time', time, module_type_store)
    
    
    
    # Obtaining the type of the subscript
    int_124505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 29), 'int')
    slice_124506 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 803, 20), int_124505, None, None)
    # Getting the type of 'sys' (line 803)
    sys_124507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 20), 'sys')
    # Obtaining the member 'argv' of a type (line 803)
    argv_124508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 20), sys_124507, 'argv')
    # Obtaining the member '__getitem__' of a type (line 803)
    getitem___124509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 20), argv_124508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 803)
    subscript_call_result_124510 = invoke(stypy.reporting.localization.Localization(__file__, 803, 20), getitem___124509, slice_124506)
    
    # Testing the type of a for loop iterable (line 803)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 803, 4), subscript_call_result_124510)
    # Getting the type of the for loop variable (line 803)
    for_loop_var_124511 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 803, 4), subscript_call_result_124510)
    # Assigning a type to the variable 'filename' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 4), 'filename', for_loop_var_124511)
    # SSA begins for a for statement (line 803)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 804)
    # Processing the call arguments (line 804)
    str_124513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 14), 'str', 'Reading')
    # Getting the type of 'filename' (line 804)
    filename_124514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 25), 'filename', False)
    str_124515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 35), 'str', '...')
    # Processing the call keyword arguments (line 804)
    str_124516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 46), 'str', ' ')
    keyword_124517 = str_124516
    kwargs_124518 = {'end': keyword_124517}
    # Getting the type of 'print' (line 804)
    print_124512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 8), 'print', False)
    # Calling print(args, kwargs) (line 804)
    print_call_result_124519 = invoke(stypy.reporting.localization.Localization(__file__, 804, 8), print_124512, *[str_124513, filename_124514, str_124515], **kwargs_124518)
    
    
    # Call to flush(...): (line 805)
    # Processing the call keyword arguments (line 805)
    kwargs_124523 = {}
    # Getting the type of 'sys' (line 805)
    sys_124520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 805)
    stdout_124521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 8), sys_124520, 'stdout')
    # Obtaining the member 'flush' of a type (line 805)
    flush_124522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 8), stdout_124521, 'flush')
    # Calling flush(args, kwargs) (line 805)
    flush_call_result_124524 = invoke(stypy.reporting.localization.Localization(__file__, 805, 8), flush_124522, *[], **kwargs_124523)
    
    
    # Assigning a Call to a Name (line 806):
    
    # Assigning a Call to a Name (line 806):
    
    # Call to time(...): (line 806)
    # Processing the call keyword arguments (line 806)
    kwargs_124527 = {}
    # Getting the type of 'time' (line 806)
    time_124525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'time', False)
    # Obtaining the member 'time' of a type (line 806)
    time_124526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 12), time_124525, 'time')
    # Calling time(args, kwargs) (line 806)
    time_call_result_124528 = invoke(stypy.reporting.localization.Localization(__file__, 806, 12), time_124526, *[], **kwargs_124527)
    
    # Assigning a type to the variable 't' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 't', time_call_result_124528)
    
    # Call to mmread(...): (line 807)
    # Processing the call arguments (line 807)
    # Getting the type of 'filename' (line 807)
    filename_124530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 15), 'filename', False)
    # Processing the call keyword arguments (line 807)
    kwargs_124531 = {}
    # Getting the type of 'mmread' (line 807)
    mmread_124529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'mmread', False)
    # Calling mmread(args, kwargs) (line 807)
    mmread_call_result_124532 = invoke(stypy.reporting.localization.Localization(__file__, 807, 8), mmread_124529, *[filename_124530], **kwargs_124531)
    
    
    # Call to print(...): (line 808)
    # Processing the call arguments (line 808)
    str_124534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 14), 'str', 'took %s seconds')
    
    # Call to time(...): (line 808)
    # Processing the call keyword arguments (line 808)
    kwargs_124537 = {}
    # Getting the type of 'time' (line 808)
    time_124535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 35), 'time', False)
    # Obtaining the member 'time' of a type (line 808)
    time_124536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 35), time_124535, 'time')
    # Calling time(args, kwargs) (line 808)
    time_call_result_124538 = invoke(stypy.reporting.localization.Localization(__file__, 808, 35), time_124536, *[], **kwargs_124537)
    
    # Getting the type of 't' (line 808)
    t_124539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 49), 't', False)
    # Applying the binary operator '-' (line 808)
    result_sub_124540 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 35), '-', time_call_result_124538, t_124539)
    
    # Applying the binary operator '%' (line 808)
    result_mod_124541 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 14), '%', str_124534, result_sub_124540)
    
    # Processing the call keyword arguments (line 808)
    kwargs_124542 = {}
    # Getting the type of 'print' (line 808)
    print_124533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'print', False)
    # Calling print(args, kwargs) (line 808)
    print_call_result_124543 = invoke(stypy.reporting.localization.Localization(__file__, 808, 8), print_124533, *[result_mod_124541], **kwargs_124542)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
