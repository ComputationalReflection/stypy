
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Define a simple format for saving numpy arrays to disk with the full
3: information about them.
4: 
5: The ``.npy`` format is the standard binary file format in NumPy for
6: persisting a *single* arbitrary NumPy array on disk. The format stores all
7: of the shape and dtype information necessary to reconstruct the array
8: correctly even on another machine with a different architecture.
9: The format is designed to be as simple as possible while achieving
10: its limited goals.
11: 
12: The ``.npz`` format is the standard format for persisting *multiple* NumPy
13: arrays on disk. A ``.npz`` file is a zip file containing multiple ``.npy``
14: files, one for each array.
15: 
16: Capabilities
17: ------------
18: 
19: - Can represent all NumPy arrays including nested record arrays and
20:   object arrays.
21: 
22: - Represents the data in its native binary form.
23: 
24: - Supports Fortran-contiguous arrays directly.
25: 
26: - Stores all of the necessary information to reconstruct the array
27:   including shape and dtype on a machine of a different
28:   architecture.  Both little-endian and big-endian arrays are
29:   supported, and a file with little-endian numbers will yield
30:   a little-endian array on any machine reading the file. The
31:   types are described in terms of their actual sizes. For example,
32:   if a machine with a 64-bit C "long int" writes out an array with
33:   "long ints", a reading machine with 32-bit C "long ints" will yield
34:   an array with 64-bit integers.
35: 
36: - Is straightforward to reverse engineer. Datasets often live longer than
37:   the programs that created them. A competent developer should be
38:   able to create a solution in their preferred programming language to
39:   read most ``.npy`` files that he has been given without much
40:   documentation.
41: 
42: - Allows memory-mapping of the data. See `open_memmep`.
43: 
44: - Can be read from a filelike stream object instead of an actual file.
45: 
46: - Stores object arrays, i.e. arrays containing elements that are arbitrary
47:   Python objects. Files with object arrays are not to be mmapable, but
48:   can be read and written to disk.
49: 
50: Limitations
51: -----------
52: 
53: - Arbitrary subclasses of numpy.ndarray are not completely preserved.
54:   Subclasses will be accepted for writing, but only the array data will
55:   be written out. A regular numpy.ndarray object will be created
56:   upon reading the file.
57: 
58: .. warning::
59: 
60:   Due to limitations in the interpretation of structured dtypes, dtypes
61:   with fields with empty names will have the names replaced by 'f0', 'f1',
62:   etc. Such arrays will not round-trip through the format entirely
63:   accurately. The data is intact; only the field names will differ. We are
64:   working on a fix for this. This fix will not require a change in the
65:   file format. The arrays with such structures can still be saved and
66:   restored, and the correct dtype may be restored by using the
67:   ``loadedarray.view(correct_dtype)`` method.
68: 
69: File extensions
70: ---------------
71: 
72: We recommend using the ``.npy`` and ``.npz`` extensions for files saved
73: in this format. This is by no means a requirement; applications may wish
74: to use these file formats but use an extension specific to the
75: application. In the absence of an obvious alternative, however,
76: we suggest using ``.npy`` and ``.npz``.
77: 
78: Version numbering
79: -----------------
80: 
81: The version numbering of these formats is independent of NumPy version
82: numbering. If the format is upgraded, the code in `numpy.io` will still
83: be able to read and write Version 1.0 files.
84: 
85: Format Version 1.0
86: ------------------
87: 
88: The first 6 bytes are a magic string: exactly ``\\x93NUMPY``.
89: 
90: The next 1 byte is an unsigned byte: the major version number of the file
91: format, e.g. ``\\x01``.
92: 
93: The next 1 byte is an unsigned byte: the minor version number of the file
94: format, e.g. ``\\x00``. Note: the version of the file format is not tied
95: to the version of the numpy package.
96: 
97: The next 2 bytes form a little-endian unsigned short int: the length of
98: the header data HEADER_LEN.
99: 
100: The next HEADER_LEN bytes form the header data describing the array's
101: format. It is an ASCII string which contains a Python literal expression
102: of a dictionary. It is terminated by a newline (``\\n``) and padded with
103: spaces (``\\x20``) to make the total length of
104: ``magic string + 4 + HEADER_LEN`` be evenly divisible by 16 for alignment
105: purposes.
106: 
107: The dictionary contains three keys:
108: 
109:     "descr" : dtype.descr
110:       An object that can be passed as an argument to the `numpy.dtype`
111:       constructor to create the array's dtype.
112:     "fortran_order" : bool
113:       Whether the array data is Fortran-contiguous or not. Since
114:       Fortran-contiguous arrays are a common form of non-C-contiguity,
115:       we allow them to be written directly to disk for efficiency.
116:     "shape" : tuple of int
117:       The shape of the array.
118: 
119: For repeatability and readability, the dictionary keys are sorted in
120: alphabetic order. This is for convenience only. A writer SHOULD implement
121: this if possible. A reader MUST NOT depend on this.
122: 
123: Following the header comes the array data. If the dtype contains Python
124: objects (i.e. ``dtype.hasobject is True``), then the data is a Python
125: pickle of the array. Otherwise the data is the contiguous (either C-
126: or Fortran-, depending on ``fortran_order``) bytes of the array.
127: Consumers can figure out the number of bytes by multiplying the number
128: of elements given by the shape (noting that ``shape=()`` means there is
129: 1 element) by ``dtype.itemsize``.
130: 
131: Format Version 2.0
132: ------------------
133: 
134: The version 1.0 format only allowed the array header to have a total size of
135: 65535 bytes.  This can be exceeded by structured arrays with a large number of
136: columns.  The version 2.0 format extends the header size to 4 GiB.
137: `numpy.save` will automatically save in 2.0 format if the data requires it,
138: else it will always use the more compatible 1.0 format.
139: 
140: The description of the fourth element of the header therefore has become:
141: "The next 4 bytes form a little-endian unsigned int: the length of the header
142: data HEADER_LEN."
143: 
144: Notes
145: -----
146: The ``.npy`` format, including reasons for creating it and a comparison of
147: alternatives, is described fully in the "npy-format" NEP.
148: 
149: '''
150: from __future__ import division, absolute_import, print_function
151: 
152: import numpy
153: import sys
154: import io
155: import warnings
156: from numpy.lib.utils import safe_eval
157: from numpy.compat import asbytes, asstr, isfileobj, long, basestring
158: 
159: if sys.version_info[0] >= 3:
160:     import pickle
161: else:
162:     import cPickle as pickle
163: 
164: MAGIC_PREFIX = asbytes('\x93NUMPY')
165: MAGIC_LEN = len(MAGIC_PREFIX) + 2
166: BUFFER_SIZE = 2**18  # size of buffer for reading npz files in bytes
167: 
168: # difference between version 1.0 and 2.0 is a 4 byte (I) header length
169: # instead of 2 bytes (H) allowing storage of large structured arrays
170: 
171: def _check_version(version):
172:     if version not in [(1, 0), (2, 0), None]:
173:         msg = "we only support format version (1,0) and (2, 0), not %s"
174:         raise ValueError(msg % (version,))
175: 
176: def magic(major, minor):
177:     ''' Return the magic string for the given file format version.
178: 
179:     Parameters
180:     ----------
181:     major : int in [0, 255]
182:     minor : int in [0, 255]
183: 
184:     Returns
185:     -------
186:     magic : str
187: 
188:     Raises
189:     ------
190:     ValueError if the version cannot be formatted.
191:     '''
192:     if major < 0 or major > 255:
193:         raise ValueError("major version must be 0 <= major < 256")
194:     if minor < 0 or minor > 255:
195:         raise ValueError("minor version must be 0 <= minor < 256")
196:     if sys.version_info[0] < 3:
197:         return MAGIC_PREFIX + chr(major) + chr(minor)
198:     else:
199:         return MAGIC_PREFIX + bytes([major, minor])
200: 
201: def read_magic(fp):
202:     ''' Read the magic string to get the version of the file format.
203: 
204:     Parameters
205:     ----------
206:     fp : filelike object
207: 
208:     Returns
209:     -------
210:     major : int
211:     minor : int
212:     '''
213:     magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
214:     if magic_str[:-2] != MAGIC_PREFIX:
215:         msg = "the magic string is not correct; expected %r, got %r"
216:         raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
217:     if sys.version_info[0] < 3:
218:         major, minor = map(ord, magic_str[-2:])
219:     else:
220:         major, minor = magic_str[-2:]
221:     return major, minor
222: 
223: def dtype_to_descr(dtype):
224:     '''
225:     Get a serializable descriptor from the dtype.
226: 
227:     The .descr attribute of a dtype object cannot be round-tripped through
228:     the dtype() constructor. Simple types, like dtype('float32'), have
229:     a descr which looks like a record array with one field with '' as
230:     a name. The dtype() constructor interprets this as a request to give
231:     a default name.  Instead, we construct descriptor that can be passed to
232:     dtype().
233: 
234:     Parameters
235:     ----------
236:     dtype : dtype
237:         The dtype of the array that will be written to disk.
238: 
239:     Returns
240:     -------
241:     descr : object
242:         An object that can be passed to `numpy.dtype()` in order to
243:         replicate the input dtype.
244: 
245:     '''
246:     if dtype.names is not None:
247:         # This is a record array. The .descr is fine.  XXX: parts of the
248:         # record array with an empty name, like padding bytes, still get
249:         # fiddled with. This needs to be fixed in the C implementation of
250:         # dtype().
251:         return dtype.descr
252:     else:
253:         return dtype.str
254: 
255: def header_data_from_array_1_0(array):
256:     ''' Get the dictionary of header metadata from a numpy.ndarray.
257: 
258:     Parameters
259:     ----------
260:     array : numpy.ndarray
261: 
262:     Returns
263:     -------
264:     d : dict
265:         This has the appropriate entries for writing its string representation
266:         to the header of the file.
267:     '''
268:     d = {'shape': array.shape}
269:     if array.flags.c_contiguous:
270:         d['fortran_order'] = False
271:     elif array.flags.f_contiguous:
272:         d['fortran_order'] = True
273:     else:
274:         # Totally non-contiguous data. We will have to make it C-contiguous
275:         # before writing. Note that we need to test for C_CONTIGUOUS first
276:         # because a 1-D array is both C_CONTIGUOUS and F_CONTIGUOUS.
277:         d['fortran_order'] = False
278: 
279:     d['descr'] = dtype_to_descr(array.dtype)
280:     return d
281: 
282: def _write_array_header(fp, d, version=None):
283:     ''' Write the header for an array and returns the version used
284: 
285:     Parameters
286:     ----------
287:     fp : filelike object
288:     d : dict
289:         This has the appropriate entries for writing its string representation
290:         to the header of the file.
291:     version: tuple or None
292:         None means use oldest that works
293:         explicit version will raise a ValueError if the format does not
294:         allow saving this data.  Default: None
295:     Returns
296:     -------
297:     version : tuple of int
298:         the file version which needs to be used to store the data
299:     '''
300:     import struct
301:     header = ["{"]
302:     for key, value in sorted(d.items()):
303:         # Need to use repr here, since we eval these when reading
304:         header.append("'%s': %s, " % (key, repr(value)))
305:     header.append("}")
306:     header = "".join(header)
307:     # Pad the header with spaces and a final newline such that the magic
308:     # string, the header-length short and the header are aligned on a
309:     # 16-byte boundary.  Hopefully, some system, possibly memory-mapping,
310:     # can take advantage of our premature optimization.
311:     current_header_len = MAGIC_LEN + 2 + len(header) + 1  # 1 for the newline
312:     topad = 16 - (current_header_len % 16)
313:     header = header + ' '*topad + '\n'
314:     header = asbytes(_filter_header(header))
315: 
316:     hlen = len(header)
317:     if hlen < 256*256 and version in (None, (1, 0)):
318:         version = (1, 0)
319:         header_prefix = magic(1, 0) + struct.pack('<H', hlen)
320:     elif hlen < 2**32 and version in (None, (2, 0)):
321:         version = (2, 0)
322:         header_prefix = magic(2, 0) + struct.pack('<I', hlen)
323:     else:
324:         msg = "Header length %s too big for version=%s"
325:         msg %= (hlen, version)
326:         raise ValueError(msg)
327: 
328:     fp.write(header_prefix)
329:     fp.write(header)
330:     return version
331: 
332: def write_array_header_1_0(fp, d):
333:     ''' Write the header for an array using the 1.0 format.
334: 
335:     Parameters
336:     ----------
337:     fp : filelike object
338:     d : dict
339:         This has the appropriate entries for writing its string
340:         representation to the header of the file.
341:     '''
342:     _write_array_header(fp, d, (1, 0))
343: 
344: 
345: def write_array_header_2_0(fp, d):
346:     ''' Write the header for an array using the 2.0 format.
347:         The 2.0 format allows storing very large structured arrays.
348: 
349:     .. versionadded:: 1.9.0
350: 
351:     Parameters
352:     ----------
353:     fp : filelike object
354:     d : dict
355:         This has the appropriate entries for writing its string
356:         representation to the header of the file.
357:     '''
358:     _write_array_header(fp, d, (2, 0))
359: 
360: def read_array_header_1_0(fp):
361:     '''
362:     Read an array header from a filelike object using the 1.0 file format
363:     version.
364: 
365:     This will leave the file object located just after the header.
366: 
367:     Parameters
368:     ----------
369:     fp : filelike object
370:         A file object or something with a `.read()` method like a file.
371: 
372:     Returns
373:     -------
374:     shape : tuple of int
375:         The shape of the array.
376:     fortran_order : bool
377:         The array data will be written out directly if it is either
378:         C-contiguous or Fortran-contiguous. Otherwise, it will be made
379:         contiguous before writing it out.
380:     dtype : dtype
381:         The dtype of the file's data.
382: 
383:     Raises
384:     ------
385:     ValueError
386:         If the data is invalid.
387: 
388:     '''
389:     return _read_array_header(fp, version=(1, 0))
390: 
391: def read_array_header_2_0(fp):
392:     '''
393:     Read an array header from a filelike object using the 2.0 file format
394:     version.
395: 
396:     This will leave the file object located just after the header.
397: 
398:     .. versionadded:: 1.9.0
399: 
400:     Parameters
401:     ----------
402:     fp : filelike object
403:         A file object or something with a `.read()` method like a file.
404: 
405:     Returns
406:     -------
407:     shape : tuple of int
408:         The shape of the array.
409:     fortran_order : bool
410:         The array data will be written out directly if it is either
411:         C-contiguous or Fortran-contiguous. Otherwise, it will be made
412:         contiguous before writing it out.
413:     dtype : dtype
414:         The dtype of the file's data.
415: 
416:     Raises
417:     ------
418:     ValueError
419:         If the data is invalid.
420: 
421:     '''
422:     return _read_array_header(fp, version=(2, 0))
423: 
424: 
425: def _filter_header(s):
426:     '''Clean up 'L' in npz header ints.
427: 
428:     Cleans up the 'L' in strings representing integers. Needed to allow npz
429:     headers produced in Python2 to be read in Python3.
430: 
431:     Parameters
432:     ----------
433:     s : byte string
434:         Npy file header.
435: 
436:     Returns
437:     -------
438:     header : str
439:         Cleaned up header.
440: 
441:     '''
442:     import tokenize
443:     if sys.version_info[0] >= 3:
444:         from io import StringIO
445:     else:
446:         from StringIO import StringIO
447: 
448:     tokens = []
449:     last_token_was_number = False
450:     for token in tokenize.generate_tokens(StringIO(asstr(s)).read):
451:         token_type = token[0]
452:         token_string = token[1]
453:         if (last_token_was_number and
454:                 token_type == tokenize.NAME and
455:                 token_string == "L"):
456:             continue
457:         else:
458:             tokens.append(token)
459:         last_token_was_number = (token_type == tokenize.NUMBER)
460:     return tokenize.untokenize(tokens)
461: 
462: 
463: def _read_array_header(fp, version):
464:     '''
465:     see read_array_header_1_0
466:     '''
467:     # Read an unsigned, little-endian short int which has the length of the
468:     # header.
469:     import struct
470:     if version == (1, 0):
471:         hlength_str = _read_bytes(fp, 2, "array header length")
472:         header_length = struct.unpack('<H', hlength_str)[0]
473:         header = _read_bytes(fp, header_length, "array header")
474:     elif version == (2, 0):
475:         hlength_str = _read_bytes(fp, 4, "array header length")
476:         header_length = struct.unpack('<I', hlength_str)[0]
477:         header = _read_bytes(fp, header_length, "array header")
478:     else:
479:         raise ValueError("Invalid version %r" % version)
480: 
481:     # The header is a pretty-printed string representation of a literal
482:     # Python dictionary with trailing newlines padded to a 16-byte
483:     # boundary. The keys are strings.
484:     #   "shape" : tuple of int
485:     #   "fortran_order" : bool
486:     #   "descr" : dtype.descr
487:     header = _filter_header(header)
488:     try:
489:         d = safe_eval(header)
490:     except SyntaxError as e:
491:         msg = "Cannot parse header: %r\nException: %r"
492:         raise ValueError(msg % (header, e))
493:     if not isinstance(d, dict):
494:         msg = "Header is not a dictionary: %r"
495:         raise ValueError(msg % d)
496:     keys = sorted(d.keys())
497:     if keys != ['descr', 'fortran_order', 'shape']:
498:         msg = "Header does not contain the correct keys: %r"
499:         raise ValueError(msg % (keys,))
500: 
501:     # Sanity-check the values.
502:     if (not isinstance(d['shape'], tuple) or
503:             not numpy.all([isinstance(x, (int, long)) for x in d['shape']])):
504:         msg = "shape is not valid: %r"
505:         raise ValueError(msg % (d['shape'],))
506:     if not isinstance(d['fortran_order'], bool):
507:         msg = "fortran_order is not a valid bool: %r"
508:         raise ValueError(msg % (d['fortran_order'],))
509:     try:
510:         dtype = numpy.dtype(d['descr'])
511:     except TypeError as e:
512:         msg = "descr is not a valid dtype descriptor: %r"
513:         raise ValueError(msg % (d['descr'],))
514: 
515:     return d['shape'], d['fortran_order'], dtype
516: 
517: def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
518:     '''
519:     Write an array to an NPY file, including a header.
520: 
521:     If the array is neither C-contiguous nor Fortran-contiguous AND the
522:     file_like object is not a real file object, this function will have to
523:     copy data in memory.
524: 
525:     Parameters
526:     ----------
527:     fp : file_like object
528:         An open, writable file object, or similar object with a
529:         ``.write()`` method.
530:     array : ndarray
531:         The array to write to disk.
532:     version : (int, int) or None, optional
533:         The version number of the format. None means use the oldest
534:         supported version that is able to store the data.  Default: None
535:     allow_pickle : bool, optional
536:         Whether to allow writing pickled data. Default: True
537:     pickle_kwargs : dict, optional
538:         Additional keyword arguments to pass to pickle.dump, excluding
539:         'protocol'. These are only useful when pickling objects in object
540:         arrays on Python 3 to Python 2 compatible format.
541: 
542:     Raises
543:     ------
544:     ValueError
545:         If the array cannot be persisted. This includes the case of
546:         allow_pickle=False and array being an object array.
547:     Various other errors
548:         If the array contains Python objects as part of its dtype, the
549:         process of pickling them may raise various errors if the objects
550:         are not picklable.
551: 
552:     '''
553:     _check_version(version)
554:     used_ver = _write_array_header(fp, header_data_from_array_1_0(array),
555:                                    version)
556:     # this warning can be removed when 1.9 has aged enough
557:     if version != (2, 0) and used_ver == (2, 0):
558:         warnings.warn("Stored array in format 2.0. It can only be"
559:                       "read by NumPy >= 1.9", UserWarning)
560: 
561:     # Set buffer size to 16 MiB to hide the Python loop overhead.
562:     buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)
563: 
564:     if array.dtype.hasobject:
565:         # We contain Python objects so we cannot write out the data
566:         # directly.  Instead, we will pickle it out with version 2 of the
567:         # pickle protocol.
568:         if not allow_pickle:
569:             raise ValueError("Object arrays cannot be saved when "
570:                              "allow_pickle=False")
571:         if pickle_kwargs is None:
572:             pickle_kwargs = {}
573:         pickle.dump(array, fp, protocol=2, **pickle_kwargs)
574:     elif array.flags.f_contiguous and not array.flags.c_contiguous:
575:         if isfileobj(fp):
576:             array.T.tofile(fp)
577:         else:
578:             for chunk in numpy.nditer(
579:                     array, flags=['external_loop', 'buffered', 'zerosize_ok'],
580:                     buffersize=buffersize, order='F'):
581:                 fp.write(chunk.tobytes('C'))
582:     else:
583:         if isfileobj(fp):
584:             array.tofile(fp)
585:         else:
586:             for chunk in numpy.nditer(
587:                     array, flags=['external_loop', 'buffered', 'zerosize_ok'],
588:                     buffersize=buffersize, order='C'):
589:                 fp.write(chunk.tobytes('C'))
590: 
591: 
592: def read_array(fp, allow_pickle=True, pickle_kwargs=None):
593:     '''
594:     Read an array from an NPY file.
595: 
596:     Parameters
597:     ----------
598:     fp : file_like object
599:         If this is not a real file object, then this may take extra memory
600:         and time.
601:     allow_pickle : bool, optional
602:         Whether to allow reading pickled data. Default: True
603:     pickle_kwargs : dict
604:         Additional keyword arguments to pass to pickle.load. These are only
605:         useful when loading object arrays saved on Python 2 when using
606:         Python 3.
607: 
608:     Returns
609:     -------
610:     array : ndarray
611:         The array from the data on disk.
612: 
613:     Raises
614:     ------
615:     ValueError
616:         If the data is invalid, or allow_pickle=False and the file contains
617:         an object array.
618: 
619:     '''
620:     version = read_magic(fp)
621:     _check_version(version)
622:     shape, fortran_order, dtype = _read_array_header(fp, version)
623:     if len(shape) == 0:
624:         count = 1
625:     else:
626:         count = numpy.multiply.reduce(shape)
627: 
628:     # Now read the actual data.
629:     if dtype.hasobject:
630:         # The array contained Python objects. We need to unpickle the data.
631:         if not allow_pickle:
632:             raise ValueError("Object arrays cannot be loaded when "
633:                              "allow_pickle=False")
634:         if pickle_kwargs is None:
635:             pickle_kwargs = {}
636:         try:
637:             array = pickle.load(fp, **pickle_kwargs)
638:         except UnicodeError as err:
639:             if sys.version_info[0] >= 3:
640:                 # Friendlier error message
641:                 raise UnicodeError("Unpickling a python object failed: %r\n"
642:                                    "You may need to pass the encoding= option "
643:                                    "to numpy.load" % (err,))
644:             raise
645:     else:
646:         if isfileobj(fp):
647:             # We can use the fast fromfile() function.
648:             array = numpy.fromfile(fp, dtype=dtype, count=count)
649:         else:
650:             # This is not a real file. We have to read it the
651:             # memory-intensive way.
652:             # crc32 module fails on reads greater than 2 ** 32 bytes,
653:             # breaking large reads from gzip streams. Chunk reads to
654:             # BUFFER_SIZE bytes to avoid issue and reduce memory overhead
655:             # of the read. In non-chunked case count < max_read_count, so
656:             # only one read is performed.
657: 
658:             max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)
659: 
660:             array = numpy.empty(count, dtype=dtype)
661:             for i in range(0, count, max_read_count):
662:                 read_count = min(max_read_count, count - i)
663:                 read_size = int(read_count * dtype.itemsize)
664:                 data = _read_bytes(fp, read_size, "array data")
665:                 array[i:i+read_count] = numpy.frombuffer(data, dtype=dtype,
666:                                                          count=read_count)
667: 
668:         if fortran_order:
669:             array.shape = shape[::-1]
670:             array = array.transpose()
671:         else:
672:             array.shape = shape
673: 
674:     return array
675: 
676: 
677: def open_memmap(filename, mode='r+', dtype=None, shape=None,
678:                 fortran_order=False, version=None):
679:     '''
680:     Open a .npy file as a memory-mapped array.
681: 
682:     This may be used to read an existing file or create a new one.
683: 
684:     Parameters
685:     ----------
686:     filename : str
687:         The name of the file on disk.  This may *not* be a file-like
688:         object.
689:     mode : str, optional
690:         The mode in which to open the file; the default is 'r+'.  In
691:         addition to the standard file modes, 'c' is also accepted to mean
692:         "copy on write."  See `memmap` for the available mode strings.
693:     dtype : data-type, optional
694:         The data type of the array if we are creating a new file in "write"
695:         mode, if not, `dtype` is ignored.  The default value is None, which
696:         results in a data-type of `float64`.
697:     shape : tuple of int
698:         The shape of the array if we are creating a new file in "write"
699:         mode, in which case this parameter is required.  Otherwise, this
700:         parameter is ignored and is thus optional.
701:     fortran_order : bool, optional
702:         Whether the array should be Fortran-contiguous (True) or
703:         C-contiguous (False, the default) if we are creating a new file in
704:         "write" mode.
705:     version : tuple of int (major, minor) or None
706:         If the mode is a "write" mode, then this is the version of the file
707:         format used to create the file.  None means use the oldest
708:         supported version that is able to store the data.  Default: None
709: 
710:     Returns
711:     -------
712:     marray : memmap
713:         The memory-mapped array.
714: 
715:     Raises
716:     ------
717:     ValueError
718:         If the data or the mode is invalid.
719:     IOError
720:         If the file is not found or cannot be opened correctly.
721: 
722:     See Also
723:     --------
724:     memmap
725: 
726:     '''
727:     if not isinstance(filename, basestring):
728:         raise ValueError("Filename must be a string.  Memmap cannot use"
729:                          " existing file handles.")
730: 
731:     if 'w' in mode:
732:         # We are creating the file, not reading it.
733:         # Check if we ought to create the file.
734:         _check_version(version)
735:         # Ensure that the given dtype is an authentic dtype object rather
736:         # than just something that can be interpreted as a dtype object.
737:         dtype = numpy.dtype(dtype)
738:         if dtype.hasobject:
739:             msg = "Array can't be memory-mapped: Python objects in dtype."
740:             raise ValueError(msg)
741:         d = dict(
742:             descr=dtype_to_descr(dtype),
743:             fortran_order=fortran_order,
744:             shape=shape,
745:         )
746:         # If we got here, then it should be safe to create the file.
747:         fp = open(filename, mode+'b')
748:         try:
749:             used_ver = _write_array_header(fp, d, version)
750:             # this warning can be removed when 1.9 has aged enough
751:             if version != (2, 0) and used_ver == (2, 0):
752:                 warnings.warn("Stored array in format 2.0. It can only be"
753:                               "read by NumPy >= 1.9", UserWarning)
754:             offset = fp.tell()
755:         finally:
756:             fp.close()
757:     else:
758:         # Read the header of the file first.
759:         fp = open(filename, 'rb')
760:         try:
761:             version = read_magic(fp)
762:             _check_version(version)
763: 
764:             shape, fortran_order, dtype = _read_array_header(fp, version)
765:             if dtype.hasobject:
766:                 msg = "Array can't be memory-mapped: Python objects in dtype."
767:                 raise ValueError(msg)
768:             offset = fp.tell()
769:         finally:
770:             fp.close()
771: 
772:     if fortran_order:
773:         order = 'F'
774:     else:
775:         order = 'C'
776: 
777:     # We need to change a write-only mode to a read-write mode since we've
778:     # already written data to the file.
779:     if mode == 'w+':
780:         mode = 'r+'
781: 
782:     marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order,
783:         mode=mode, offset=offset)
784: 
785:     return marray
786: 
787: 
788: def _read_bytes(fp, size, error_template="ran out of data"):
789:     '''
790:     Read from file-like object until size bytes are read.
791:     Raises ValueError if not EOF is encountered before size bytes are read.
792:     Non-blocking objects only supported if they derive from io objects.
793: 
794:     Required as e.g. ZipExtFile in python 2.6 can return less data than
795:     requested.
796:     '''
797:     data = bytes()
798:     while True:
799:         # io files (default in python3) return None or raise on
800:         # would-block, python2 file will truncate, probably nothing can be
801:         # done about that.  note that regular files can't be non-blocking
802:         try:
803:             r = fp.read(size - len(data))
804:             data += r
805:             if len(r) == 0 or len(data) == size:
806:                 break
807:         except io.BlockingIOError:
808:             pass
809:     if len(data) != size:
810:         msg = "EOF: reading %s, expected %d bytes got %d"
811:         raise ValueError(msg % (error_template, size, len(data)))
812:     else:
813:         return data
814: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_106198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', '\nDefine a simple format for saving numpy arrays to disk with the full\ninformation about them.\n\nThe ``.npy`` format is the standard binary file format in NumPy for\npersisting a *single* arbitrary NumPy array on disk. The format stores all\nof the shape and dtype information necessary to reconstruct the array\ncorrectly even on another machine with a different architecture.\nThe format is designed to be as simple as possible while achieving\nits limited goals.\n\nThe ``.npz`` format is the standard format for persisting *multiple* NumPy\narrays on disk. A ``.npz`` file is a zip file containing multiple ``.npy``\nfiles, one for each array.\n\nCapabilities\n------------\n\n- Can represent all NumPy arrays including nested record arrays and\n  object arrays.\n\n- Represents the data in its native binary form.\n\n- Supports Fortran-contiguous arrays directly.\n\n- Stores all of the necessary information to reconstruct the array\n  including shape and dtype on a machine of a different\n  architecture.  Both little-endian and big-endian arrays are\n  supported, and a file with little-endian numbers will yield\n  a little-endian array on any machine reading the file. The\n  types are described in terms of their actual sizes. For example,\n  if a machine with a 64-bit C "long int" writes out an array with\n  "long ints", a reading machine with 32-bit C "long ints" will yield\n  an array with 64-bit integers.\n\n- Is straightforward to reverse engineer. Datasets often live longer than\n  the programs that created them. A competent developer should be\n  able to create a solution in their preferred programming language to\n  read most ``.npy`` files that he has been given without much\n  documentation.\n\n- Allows memory-mapping of the data. See `open_memmep`.\n\n- Can be read from a filelike stream object instead of an actual file.\n\n- Stores object arrays, i.e. arrays containing elements that are arbitrary\n  Python objects. Files with object arrays are not to be mmapable, but\n  can be read and written to disk.\n\nLimitations\n-----------\n\n- Arbitrary subclasses of numpy.ndarray are not completely preserved.\n  Subclasses will be accepted for writing, but only the array data will\n  be written out. A regular numpy.ndarray object will be created\n  upon reading the file.\n\n.. warning::\n\n  Due to limitations in the interpretation of structured dtypes, dtypes\n  with fields with empty names will have the names replaced by \'f0\', \'f1\',\n  etc. Such arrays will not round-trip through the format entirely\n  accurately. The data is intact; only the field names will differ. We are\n  working on a fix for this. This fix will not require a change in the\n  file format. The arrays with such structures can still be saved and\n  restored, and the correct dtype may be restored by using the\n  ``loadedarray.view(correct_dtype)`` method.\n\nFile extensions\n---------------\n\nWe recommend using the ``.npy`` and ``.npz`` extensions for files saved\nin this format. This is by no means a requirement; applications may wish\nto use these file formats but use an extension specific to the\napplication. In the absence of an obvious alternative, however,\nwe suggest using ``.npy`` and ``.npz``.\n\nVersion numbering\n-----------------\n\nThe version numbering of these formats is independent of NumPy version\nnumbering. If the format is upgraded, the code in `numpy.io` will still\nbe able to read and write Version 1.0 files.\n\nFormat Version 1.0\n------------------\n\nThe first 6 bytes are a magic string: exactly ``\\x93NUMPY``.\n\nThe next 1 byte is an unsigned byte: the major version number of the file\nformat, e.g. ``\\x01``.\n\nThe next 1 byte is an unsigned byte: the minor version number of the file\nformat, e.g. ``\\x00``. Note: the version of the file format is not tied\nto the version of the numpy package.\n\nThe next 2 bytes form a little-endian unsigned short int: the length of\nthe header data HEADER_LEN.\n\nThe next HEADER_LEN bytes form the header data describing the array\'s\nformat. It is an ASCII string which contains a Python literal expression\nof a dictionary. It is terminated by a newline (``\\n``) and padded with\nspaces (``\\x20``) to make the total length of\n``magic string + 4 + HEADER_LEN`` be evenly divisible by 16 for alignment\npurposes.\n\nThe dictionary contains three keys:\n\n    "descr" : dtype.descr\n      An object that can be passed as an argument to the `numpy.dtype`\n      constructor to create the array\'s dtype.\n    "fortran_order" : bool\n      Whether the array data is Fortran-contiguous or not. Since\n      Fortran-contiguous arrays are a common form of non-C-contiguity,\n      we allow them to be written directly to disk for efficiency.\n    "shape" : tuple of int\n      The shape of the array.\n\nFor repeatability and readability, the dictionary keys are sorted in\nalphabetic order. This is for convenience only. A writer SHOULD implement\nthis if possible. A reader MUST NOT depend on this.\n\nFollowing the header comes the array data. If the dtype contains Python\nobjects (i.e. ``dtype.hasobject is True``), then the data is a Python\npickle of the array. Otherwise the data is the contiguous (either C-\nor Fortran-, depending on ``fortran_order``) bytes of the array.\nConsumers can figure out the number of bytes by multiplying the number\nof elements given by the shape (noting that ``shape=()`` means there is\n1 element) by ``dtype.itemsize``.\n\nFormat Version 2.0\n------------------\n\nThe version 1.0 format only allowed the array header to have a total size of\n65535 bytes.  This can be exceeded by structured arrays with a large number of\ncolumns.  The version 2.0 format extends the header size to 4 GiB.\n`numpy.save` will automatically save in 2.0 format if the data requires it,\nelse it will always use the more compatible 1.0 format.\n\nThe description of the fourth element of the header therefore has become:\n"The next 4 bytes form a little-endian unsigned int: the length of the header\ndata HEADER_LEN."\n\nNotes\n-----\nThe ``.npy`` format, including reasons for creating it and a comparison of\nalternatives, is described fully in the "npy-format" NEP.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 152, 0))

# 'import numpy' statement (line 152)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_106199 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 152, 0), 'numpy')

if (type(import_106199) is not StypyTypeError):

    if (import_106199 != 'pyd_module'):
        __import__(import_106199)
        sys_modules_106200 = sys.modules[import_106199]
        import_module(stypy.reporting.localization.Localization(__file__, 152, 0), 'numpy', sys_modules_106200.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 152, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'numpy', import_106199)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 153, 0))

# 'import sys' statement (line 153)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 153, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 154, 0))

# 'import io' statement (line 154)
import io

import_module(stypy.reporting.localization.Localization(__file__, 154, 0), 'io', io, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 155, 0))

# 'import warnings' statement (line 155)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 155, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 156, 0))

# 'from numpy.lib.utils import safe_eval' statement (line 156)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_106201 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 156, 0), 'numpy.lib.utils')

if (type(import_106201) is not StypyTypeError):

    if (import_106201 != 'pyd_module'):
        __import__(import_106201)
        sys_modules_106202 = sys.modules[import_106201]
        import_from_module(stypy.reporting.localization.Localization(__file__, 156, 0), 'numpy.lib.utils', sys_modules_106202.module_type_store, module_type_store, ['safe_eval'])
        nest_module(stypy.reporting.localization.Localization(__file__, 156, 0), __file__, sys_modules_106202, sys_modules_106202.module_type_store, module_type_store)
    else:
        from numpy.lib.utils import safe_eval

        import_from_module(stypy.reporting.localization.Localization(__file__, 156, 0), 'numpy.lib.utils', None, module_type_store, ['safe_eval'], [safe_eval])

else:
    # Assigning a type to the variable 'numpy.lib.utils' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'numpy.lib.utils', import_106201)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 157, 0))

# 'from numpy.compat import asbytes, asstr, isfileobj, long, basestring' statement (line 157)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_106203 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 157, 0), 'numpy.compat')

if (type(import_106203) is not StypyTypeError):

    if (import_106203 != 'pyd_module'):
        __import__(import_106203)
        sys_modules_106204 = sys.modules[import_106203]
        import_from_module(stypy.reporting.localization.Localization(__file__, 157, 0), 'numpy.compat', sys_modules_106204.module_type_store, module_type_store, ['asbytes', 'asstr', 'isfileobj', 'long', 'basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 157, 0), __file__, sys_modules_106204, sys_modules_106204.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, asstr, isfileobj, long, basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 157, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'asstr', 'isfileobj', 'long', 'basestring'], [asbytes, asstr, isfileobj, long, basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'numpy.compat', import_106203)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')




# Obtaining the type of the subscript
int_106205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'int')
# Getting the type of 'sys' (line 159)
sys_106206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 159)
version_info_106207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 3), sys_106206, 'version_info')
# Obtaining the member '__getitem__' of a type (line 159)
getitem___106208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 3), version_info_106207, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 159)
subscript_call_result_106209 = invoke(stypy.reporting.localization.Localization(__file__, 159, 3), getitem___106208, int_106205)

int_106210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 26), 'int')
# Applying the binary operator '>=' (line 159)
result_ge_106211 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 3), '>=', subscript_call_result_106209, int_106210)

# Testing the type of an if condition (line 159)
if_condition_106212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 0), result_ge_106211)
# Assigning a type to the variable 'if_condition_106212' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'if_condition_106212', if_condition_106212)
# SSA begins for if statement (line 159)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 160, 4))

# 'import pickle' statement (line 160)
import pickle

import_module(stypy.reporting.localization.Localization(__file__, 160, 4), 'pickle', pickle, module_type_store)

# SSA branch for the else part of an if statement (line 159)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 162, 4))

# 'import cPickle' statement (line 162)
import cPickle as pickle

import_module(stypy.reporting.localization.Localization(__file__, 162, 4), 'pickle', pickle, module_type_store)

# SSA join for if statement (line 159)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 164):

# Assigning a Call to a Name (line 164):

# Call to asbytes(...): (line 164)
# Processing the call arguments (line 164)
str_106214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'str', '\x93NUMPY')
# Processing the call keyword arguments (line 164)
kwargs_106215 = {}
# Getting the type of 'asbytes' (line 164)
asbytes_106213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 164)
asbytes_call_result_106216 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), asbytes_106213, *[str_106214], **kwargs_106215)

# Assigning a type to the variable 'MAGIC_PREFIX' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'MAGIC_PREFIX', asbytes_call_result_106216)

# Assigning a BinOp to a Name (line 165):

# Assigning a BinOp to a Name (line 165):

# Call to len(...): (line 165)
# Processing the call arguments (line 165)
# Getting the type of 'MAGIC_PREFIX' (line 165)
MAGIC_PREFIX_106218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'MAGIC_PREFIX', False)
# Processing the call keyword arguments (line 165)
kwargs_106219 = {}
# Getting the type of 'len' (line 165)
len_106217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'len', False)
# Calling len(args, kwargs) (line 165)
len_call_result_106220 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), len_106217, *[MAGIC_PREFIX_106218], **kwargs_106219)

int_106221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
# Applying the binary operator '+' (line 165)
result_add_106222 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 12), '+', len_call_result_106220, int_106221)

# Assigning a type to the variable 'MAGIC_LEN' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'MAGIC_LEN', result_add_106222)

# Assigning a BinOp to a Name (line 166):

# Assigning a BinOp to a Name (line 166):
int_106223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 14), 'int')
int_106224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 17), 'int')
# Applying the binary operator '**' (line 166)
result_pow_106225 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 14), '**', int_106223, int_106224)

# Assigning a type to the variable 'BUFFER_SIZE' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'BUFFER_SIZE', result_pow_106225)

@norecursion
def _check_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_version'
    module_type_store = module_type_store.open_function_context('_check_version', 171, 0, False)
    
    # Passed parameters checking function
    _check_version.stypy_localization = localization
    _check_version.stypy_type_of_self = None
    _check_version.stypy_type_store = module_type_store
    _check_version.stypy_function_name = '_check_version'
    _check_version.stypy_param_names_list = ['version']
    _check_version.stypy_varargs_param_name = None
    _check_version.stypy_kwargs_param_name = None
    _check_version.stypy_call_defaults = defaults
    _check_version.stypy_call_varargs = varargs
    _check_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_version', ['version'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_version', localization, ['version'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_version(...)' code ##################

    
    
    # Getting the type of 'version' (line 172)
    version_106226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 7), 'version')
    
    # Obtaining an instance of the builtin type 'list' (line 172)
    list_106227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 172)
    # Adding element type (line 172)
    
    # Obtaining an instance of the builtin type 'tuple' (line 172)
    tuple_106228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 172)
    # Adding element type (line 172)
    int_106229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 24), tuple_106228, int_106229)
    # Adding element type (line 172)
    int_106230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 24), tuple_106228, int_106230)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 22), list_106227, tuple_106228)
    # Adding element type (line 172)
    
    # Obtaining an instance of the builtin type 'tuple' (line 172)
    tuple_106231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 172)
    # Adding element type (line 172)
    int_106232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 32), tuple_106231, int_106232)
    # Adding element type (line 172)
    int_106233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 32), tuple_106231, int_106233)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 22), list_106227, tuple_106231)
    # Adding element type (line 172)
    # Getting the type of 'None' (line 172)
    None_106234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 39), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 22), list_106227, None_106234)
    
    # Applying the binary operator 'notin' (line 172)
    result_contains_106235 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), 'notin', version_106226, list_106227)
    
    # Testing the type of an if condition (line 172)
    if_condition_106236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), result_contains_106235)
    # Assigning a type to the variable 'if_condition_106236' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_106236', if_condition_106236)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 173):
    
    # Assigning a Str to a Name (line 173):
    str_106237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 14), 'str', 'we only support format version (1,0) and (2, 0), not %s')
    # Assigning a type to the variable 'msg' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'msg', str_106237)
    
    # Call to ValueError(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'msg' (line 174)
    msg_106239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 174)
    tuple_106240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 174)
    # Adding element type (line 174)
    # Getting the type of 'version' (line 174)
    version_106241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'version', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), tuple_106240, version_106241)
    
    # Applying the binary operator '%' (line 174)
    result_mod_106242 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 25), '%', msg_106239, tuple_106240)
    
    # Processing the call keyword arguments (line 174)
    kwargs_106243 = {}
    # Getting the type of 'ValueError' (line 174)
    ValueError_106238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 174)
    ValueError_call_result_106244 = invoke(stypy.reporting.localization.Localization(__file__, 174, 14), ValueError_106238, *[result_mod_106242], **kwargs_106243)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 174, 8), ValueError_call_result_106244, 'raise parameter', BaseException)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_version' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_106245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106245)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_version'
    return stypy_return_type_106245

# Assigning a type to the variable '_check_version' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), '_check_version', _check_version)

@norecursion
def magic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'magic'
    module_type_store = module_type_store.open_function_context('magic', 176, 0, False)
    
    # Passed parameters checking function
    magic.stypy_localization = localization
    magic.stypy_type_of_self = None
    magic.stypy_type_store = module_type_store
    magic.stypy_function_name = 'magic'
    magic.stypy_param_names_list = ['major', 'minor']
    magic.stypy_varargs_param_name = None
    magic.stypy_kwargs_param_name = None
    magic.stypy_call_defaults = defaults
    magic.stypy_call_varargs = varargs
    magic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'magic', ['major', 'minor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'magic', localization, ['major', 'minor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'magic(...)' code ##################

    str_106246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, (-1)), 'str', ' Return the magic string for the given file format version.\n\n    Parameters\n    ----------\n    major : int in [0, 255]\n    minor : int in [0, 255]\n\n    Returns\n    -------\n    magic : str\n\n    Raises\n    ------\n    ValueError if the version cannot be formatted.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'major' (line 192)
    major_106247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 7), 'major')
    int_106248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 15), 'int')
    # Applying the binary operator '<' (line 192)
    result_lt_106249 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 7), '<', major_106247, int_106248)
    
    
    # Getting the type of 'major' (line 192)
    major_106250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'major')
    int_106251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 28), 'int')
    # Applying the binary operator '>' (line 192)
    result_gt_106252 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 20), '>', major_106250, int_106251)
    
    # Applying the binary operator 'or' (line 192)
    result_or_keyword_106253 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 7), 'or', result_lt_106249, result_gt_106252)
    
    # Testing the type of an if condition (line 192)
    if_condition_106254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), result_or_keyword_106253)
    # Assigning a type to the variable 'if_condition_106254' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'if_condition_106254', if_condition_106254)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 193)
    # Processing the call arguments (line 193)
    str_106256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 25), 'str', 'major version must be 0 <= major < 256')
    # Processing the call keyword arguments (line 193)
    kwargs_106257 = {}
    # Getting the type of 'ValueError' (line 193)
    ValueError_106255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 193)
    ValueError_call_result_106258 = invoke(stypy.reporting.localization.Localization(__file__, 193, 14), ValueError_106255, *[str_106256], **kwargs_106257)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 193, 8), ValueError_call_result_106258, 'raise parameter', BaseException)
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'minor' (line 194)
    minor_106259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 7), 'minor')
    int_106260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 15), 'int')
    # Applying the binary operator '<' (line 194)
    result_lt_106261 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 7), '<', minor_106259, int_106260)
    
    
    # Getting the type of 'minor' (line 194)
    minor_106262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'minor')
    int_106263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 28), 'int')
    # Applying the binary operator '>' (line 194)
    result_gt_106264 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 20), '>', minor_106262, int_106263)
    
    # Applying the binary operator 'or' (line 194)
    result_or_keyword_106265 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 7), 'or', result_lt_106261, result_gt_106264)
    
    # Testing the type of an if condition (line 194)
    if_condition_106266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), result_or_keyword_106265)
    # Assigning a type to the variable 'if_condition_106266' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_106266', if_condition_106266)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 195)
    # Processing the call arguments (line 195)
    str_106268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'str', 'minor version must be 0 <= minor < 256')
    # Processing the call keyword arguments (line 195)
    kwargs_106269 = {}
    # Getting the type of 'ValueError' (line 195)
    ValueError_106267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 195)
    ValueError_call_result_106270 = invoke(stypy.reporting.localization.Localization(__file__, 195, 14), ValueError_106267, *[str_106268], **kwargs_106269)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 195, 8), ValueError_call_result_106270, 'raise parameter', BaseException)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_106271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 24), 'int')
    # Getting the type of 'sys' (line 196)
    sys_106272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 196)
    version_info_106273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 7), sys_106272, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___106274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 7), version_info_106273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_106275 = invoke(stypy.reporting.localization.Localization(__file__, 196, 7), getitem___106274, int_106271)
    
    int_106276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 29), 'int')
    # Applying the binary operator '<' (line 196)
    result_lt_106277 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 7), '<', subscript_call_result_106275, int_106276)
    
    # Testing the type of an if condition (line 196)
    if_condition_106278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 4), result_lt_106277)
    # Assigning a type to the variable 'if_condition_106278' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'if_condition_106278', if_condition_106278)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'MAGIC_PREFIX' (line 197)
    MAGIC_PREFIX_106279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'MAGIC_PREFIX')
    
    # Call to chr(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'major' (line 197)
    major_106281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 34), 'major', False)
    # Processing the call keyword arguments (line 197)
    kwargs_106282 = {}
    # Getting the type of 'chr' (line 197)
    chr_106280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 30), 'chr', False)
    # Calling chr(args, kwargs) (line 197)
    chr_call_result_106283 = invoke(stypy.reporting.localization.Localization(__file__, 197, 30), chr_106280, *[major_106281], **kwargs_106282)
    
    # Applying the binary operator '+' (line 197)
    result_add_106284 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 15), '+', MAGIC_PREFIX_106279, chr_call_result_106283)
    
    
    # Call to chr(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'minor' (line 197)
    minor_106286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 47), 'minor', False)
    # Processing the call keyword arguments (line 197)
    kwargs_106287 = {}
    # Getting the type of 'chr' (line 197)
    chr_106285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 43), 'chr', False)
    # Calling chr(args, kwargs) (line 197)
    chr_call_result_106288 = invoke(stypy.reporting.localization.Localization(__file__, 197, 43), chr_106285, *[minor_106286], **kwargs_106287)
    
    # Applying the binary operator '+' (line 197)
    result_add_106289 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 41), '+', result_add_106284, chr_call_result_106288)
    
    # Assigning a type to the variable 'stypy_return_type' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'stypy_return_type', result_add_106289)
    # SSA branch for the else part of an if statement (line 196)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'MAGIC_PREFIX' (line 199)
    MAGIC_PREFIX_106290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'MAGIC_PREFIX')
    
    # Call to bytes(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Obtaining an instance of the builtin type 'list' (line 199)
    list_106292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 199)
    # Adding element type (line 199)
    # Getting the type of 'major' (line 199)
    major_106293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 'major', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 36), list_106292, major_106293)
    # Adding element type (line 199)
    # Getting the type of 'minor' (line 199)
    minor_106294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 44), 'minor', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 36), list_106292, minor_106294)
    
    # Processing the call keyword arguments (line 199)
    kwargs_106295 = {}
    # Getting the type of 'bytes' (line 199)
    bytes_106291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'bytes', False)
    # Calling bytes(args, kwargs) (line 199)
    bytes_call_result_106296 = invoke(stypy.reporting.localization.Localization(__file__, 199, 30), bytes_106291, *[list_106292], **kwargs_106295)
    
    # Applying the binary operator '+' (line 199)
    result_add_106297 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 15), '+', MAGIC_PREFIX_106290, bytes_call_result_106296)
    
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', result_add_106297)
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'magic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'magic' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_106298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'magic'
    return stypy_return_type_106298

# Assigning a type to the variable 'magic' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'magic', magic)

@norecursion
def read_magic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_magic'
    module_type_store = module_type_store.open_function_context('read_magic', 201, 0, False)
    
    # Passed parameters checking function
    read_magic.stypy_localization = localization
    read_magic.stypy_type_of_self = None
    read_magic.stypy_type_store = module_type_store
    read_magic.stypy_function_name = 'read_magic'
    read_magic.stypy_param_names_list = ['fp']
    read_magic.stypy_varargs_param_name = None
    read_magic.stypy_kwargs_param_name = None
    read_magic.stypy_call_defaults = defaults
    read_magic.stypy_call_varargs = varargs
    read_magic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_magic', ['fp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_magic', localization, ['fp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_magic(...)' code ##################

    str_106299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, (-1)), 'str', ' Read the magic string to get the version of the file format.\n\n    Parameters\n    ----------\n    fp : filelike object\n\n    Returns\n    -------\n    major : int\n    minor : int\n    ')
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to _read_bytes(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'fp' (line 213)
    fp_106301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'fp', False)
    # Getting the type of 'MAGIC_LEN' (line 213)
    MAGIC_LEN_106302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'MAGIC_LEN', False)
    str_106303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 43), 'str', 'magic string')
    # Processing the call keyword arguments (line 213)
    kwargs_106304 = {}
    # Getting the type of '_read_bytes' (line 213)
    _read_bytes_106300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 213)
    _read_bytes_call_result_106305 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), _read_bytes_106300, *[fp_106301, MAGIC_LEN_106302, str_106303], **kwargs_106304)
    
    # Assigning a type to the variable 'magic_str' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'magic_str', _read_bytes_call_result_106305)
    
    
    
    # Obtaining the type of the subscript
    int_106306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 18), 'int')
    slice_106307 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 214, 7), None, int_106306, None)
    # Getting the type of 'magic_str' (line 214)
    magic_str_106308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 7), 'magic_str')
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___106309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 7), magic_str_106308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_106310 = invoke(stypy.reporting.localization.Localization(__file__, 214, 7), getitem___106309, slice_106307)
    
    # Getting the type of 'MAGIC_PREFIX' (line 214)
    MAGIC_PREFIX_106311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 25), 'MAGIC_PREFIX')
    # Applying the binary operator '!=' (line 214)
    result_ne_106312 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 7), '!=', subscript_call_result_106310, MAGIC_PREFIX_106311)
    
    # Testing the type of an if condition (line 214)
    if_condition_106313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 4), result_ne_106312)
    # Assigning a type to the variable 'if_condition_106313' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'if_condition_106313', if_condition_106313)
    # SSA begins for if statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 215):
    
    # Assigning a Str to a Name (line 215):
    str_106314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 14), 'str', 'the magic string is not correct; expected %r, got %r')
    # Assigning a type to the variable 'msg' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'msg', str_106314)
    
    # Call to ValueError(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'msg' (line 216)
    msg_106316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 216)
    tuple_106317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 216)
    # Adding element type (line 216)
    # Getting the type of 'MAGIC_PREFIX' (line 216)
    MAGIC_PREFIX_106318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 32), 'MAGIC_PREFIX', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 32), tuple_106317, MAGIC_PREFIX_106318)
    # Adding element type (line 216)
    
    # Obtaining the type of the subscript
    int_106319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 57), 'int')
    slice_106320 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 216, 46), None, int_106319, None)
    # Getting the type of 'magic_str' (line 216)
    magic_str_106321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 46), 'magic_str', False)
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___106322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 46), magic_str_106321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_106323 = invoke(stypy.reporting.localization.Localization(__file__, 216, 46), getitem___106322, slice_106320)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 32), tuple_106317, subscript_call_result_106323)
    
    # Applying the binary operator '%' (line 216)
    result_mod_106324 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 25), '%', msg_106316, tuple_106317)
    
    # Processing the call keyword arguments (line 216)
    kwargs_106325 = {}
    # Getting the type of 'ValueError' (line 216)
    ValueError_106315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 216)
    ValueError_call_result_106326 = invoke(stypy.reporting.localization.Localization(__file__, 216, 14), ValueError_106315, *[result_mod_106324], **kwargs_106325)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 216, 8), ValueError_call_result_106326, 'raise parameter', BaseException)
    # SSA join for if statement (line 214)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_106327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 24), 'int')
    # Getting the type of 'sys' (line 217)
    sys_106328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 217)
    version_info_106329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 7), sys_106328, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___106330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 7), version_info_106329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_106331 = invoke(stypy.reporting.localization.Localization(__file__, 217, 7), getitem___106330, int_106327)
    
    int_106332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'int')
    # Applying the binary operator '<' (line 217)
    result_lt_106333 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 7), '<', subscript_call_result_106331, int_106332)
    
    # Testing the type of an if condition (line 217)
    if_condition_106334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 4), result_lt_106333)
    # Assigning a type to the variable 'if_condition_106334' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'if_condition_106334', if_condition_106334)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 218):
    
    # Assigning a Call to a Name:
    
    # Call to map(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'ord' (line 218)
    ord_106336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'ord', False)
    
    # Obtaining the type of the subscript
    int_106337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 42), 'int')
    slice_106338 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 218, 32), int_106337, None, None)
    # Getting the type of 'magic_str' (line 218)
    magic_str_106339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 32), 'magic_str', False)
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___106340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 32), magic_str_106339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_106341 = invoke(stypy.reporting.localization.Localization(__file__, 218, 32), getitem___106340, slice_106338)
    
    # Processing the call keyword arguments (line 218)
    kwargs_106342 = {}
    # Getting the type of 'map' (line 218)
    map_106335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'map', False)
    # Calling map(args, kwargs) (line 218)
    map_call_result_106343 = invoke(stypy.reporting.localization.Localization(__file__, 218, 23), map_106335, *[ord_106336, subscript_call_result_106341], **kwargs_106342)
    
    # Assigning a type to the variable 'call_assignment_106185' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'call_assignment_106185', map_call_result_106343)
    
    # Assigning a Call to a Name (line 218):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_106346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 8), 'int')
    # Processing the call keyword arguments
    kwargs_106347 = {}
    # Getting the type of 'call_assignment_106185' (line 218)
    call_assignment_106185_106344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'call_assignment_106185', False)
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___106345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), call_assignment_106185_106344, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_106348 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___106345, *[int_106346], **kwargs_106347)
    
    # Assigning a type to the variable 'call_assignment_106186' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'call_assignment_106186', getitem___call_result_106348)
    
    # Assigning a Name to a Name (line 218):
    # Getting the type of 'call_assignment_106186' (line 218)
    call_assignment_106186_106349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'call_assignment_106186')
    # Assigning a type to the variable 'major' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'major', call_assignment_106186_106349)
    
    # Assigning a Call to a Name (line 218):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_106352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 8), 'int')
    # Processing the call keyword arguments
    kwargs_106353 = {}
    # Getting the type of 'call_assignment_106185' (line 218)
    call_assignment_106185_106350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'call_assignment_106185', False)
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___106351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), call_assignment_106185_106350, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_106354 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___106351, *[int_106352], **kwargs_106353)
    
    # Assigning a type to the variable 'call_assignment_106187' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'call_assignment_106187', getitem___call_result_106354)
    
    # Assigning a Name to a Name (line 218):
    # Getting the type of 'call_assignment_106187' (line 218)
    call_assignment_106187_106355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'call_assignment_106187')
    # Assigning a type to the variable 'minor' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'minor', call_assignment_106187_106355)
    # SSA branch for the else part of an if statement (line 217)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Tuple (line 220):
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    int_106356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'int')
    
    # Obtaining the type of the subscript
    int_106357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 33), 'int')
    slice_106358 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 220, 23), int_106357, None, None)
    # Getting the type of 'magic_str' (line 220)
    magic_str_106359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'magic_str')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___106360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 23), magic_str_106359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_106361 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), getitem___106360, slice_106358)
    
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___106362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), subscript_call_result_106361, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_106363 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), getitem___106362, int_106356)
    
    # Assigning a type to the variable 'tuple_var_assignment_106188' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_106188', subscript_call_result_106363)
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    int_106364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'int')
    
    # Obtaining the type of the subscript
    int_106365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 33), 'int')
    slice_106366 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 220, 23), int_106365, None, None)
    # Getting the type of 'magic_str' (line 220)
    magic_str_106367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'magic_str')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___106368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 23), magic_str_106367, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_106369 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), getitem___106368, slice_106366)
    
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___106370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), subscript_call_result_106369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_106371 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), getitem___106370, int_106364)
    
    # Assigning a type to the variable 'tuple_var_assignment_106189' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_106189', subscript_call_result_106371)
    
    # Assigning a Name to a Name (line 220):
    # Getting the type of 'tuple_var_assignment_106188' (line 220)
    tuple_var_assignment_106188_106372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_106188')
    # Assigning a type to the variable 'major' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'major', tuple_var_assignment_106188_106372)
    
    # Assigning a Name to a Name (line 220):
    # Getting the type of 'tuple_var_assignment_106189' (line 220)
    tuple_var_assignment_106189_106373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'tuple_var_assignment_106189')
    # Assigning a type to the variable 'minor' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'minor', tuple_var_assignment_106189_106373)
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 221)
    tuple_106374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 221)
    # Adding element type (line 221)
    # Getting the type of 'major' (line 221)
    major_106375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'major')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 11), tuple_106374, major_106375)
    # Adding element type (line 221)
    # Getting the type of 'minor' (line 221)
    minor_106376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 18), 'minor')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 11), tuple_106374, minor_106376)
    
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', tuple_106374)
    
    # ################# End of 'read_magic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_magic' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_106377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106377)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_magic'
    return stypy_return_type_106377

# Assigning a type to the variable 'read_magic' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'read_magic', read_magic)

@norecursion
def dtype_to_descr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dtype_to_descr'
    module_type_store = module_type_store.open_function_context('dtype_to_descr', 223, 0, False)
    
    # Passed parameters checking function
    dtype_to_descr.stypy_localization = localization
    dtype_to_descr.stypy_type_of_self = None
    dtype_to_descr.stypy_type_store = module_type_store
    dtype_to_descr.stypy_function_name = 'dtype_to_descr'
    dtype_to_descr.stypy_param_names_list = ['dtype']
    dtype_to_descr.stypy_varargs_param_name = None
    dtype_to_descr.stypy_kwargs_param_name = None
    dtype_to_descr.stypy_call_defaults = defaults
    dtype_to_descr.stypy_call_varargs = varargs
    dtype_to_descr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dtype_to_descr', ['dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dtype_to_descr', localization, ['dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dtype_to_descr(...)' code ##################

    str_106378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, (-1)), 'str', "\n    Get a serializable descriptor from the dtype.\n\n    The .descr attribute of a dtype object cannot be round-tripped through\n    the dtype() constructor. Simple types, like dtype('float32'), have\n    a descr which looks like a record array with one field with '' as\n    a name. The dtype() constructor interprets this as a request to give\n    a default name.  Instead, we construct descriptor that can be passed to\n    dtype().\n\n    Parameters\n    ----------\n    dtype : dtype\n        The dtype of the array that will be written to disk.\n\n    Returns\n    -------\n    descr : object\n        An object that can be passed to `numpy.dtype()` in order to\n        replicate the input dtype.\n\n    ")
    
    
    # Getting the type of 'dtype' (line 246)
    dtype_106379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'dtype')
    # Obtaining the member 'names' of a type (line 246)
    names_106380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 7), dtype_106379, 'names')
    # Getting the type of 'None' (line 246)
    None_106381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'None')
    # Applying the binary operator 'isnot' (line 246)
    result_is_not_106382 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 7), 'isnot', names_106380, None_106381)
    
    # Testing the type of an if condition (line 246)
    if_condition_106383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 4), result_is_not_106382)
    # Assigning a type to the variable 'if_condition_106383' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'if_condition_106383', if_condition_106383)
    # SSA begins for if statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'dtype' (line 251)
    dtype_106384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'dtype')
    # Obtaining the member 'descr' of a type (line 251)
    descr_106385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), dtype_106384, 'descr')
    # Assigning a type to the variable 'stypy_return_type' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', descr_106385)
    # SSA branch for the else part of an if statement (line 246)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'dtype' (line 253)
    dtype_106386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'dtype')
    # Obtaining the member 'str' of a type (line 253)
    str_106387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 15), dtype_106386, 'str')
    # Assigning a type to the variable 'stypy_return_type' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type', str_106387)
    # SSA join for if statement (line 246)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'dtype_to_descr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dtype_to_descr' in the type store
    # Getting the type of 'stypy_return_type' (line 223)
    stypy_return_type_106388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106388)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dtype_to_descr'
    return stypy_return_type_106388

# Assigning a type to the variable 'dtype_to_descr' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'dtype_to_descr', dtype_to_descr)

@norecursion
def header_data_from_array_1_0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'header_data_from_array_1_0'
    module_type_store = module_type_store.open_function_context('header_data_from_array_1_0', 255, 0, False)
    
    # Passed parameters checking function
    header_data_from_array_1_0.stypy_localization = localization
    header_data_from_array_1_0.stypy_type_of_self = None
    header_data_from_array_1_0.stypy_type_store = module_type_store
    header_data_from_array_1_0.stypy_function_name = 'header_data_from_array_1_0'
    header_data_from_array_1_0.stypy_param_names_list = ['array']
    header_data_from_array_1_0.stypy_varargs_param_name = None
    header_data_from_array_1_0.stypy_kwargs_param_name = None
    header_data_from_array_1_0.stypy_call_defaults = defaults
    header_data_from_array_1_0.stypy_call_varargs = varargs
    header_data_from_array_1_0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'header_data_from_array_1_0', ['array'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'header_data_from_array_1_0', localization, ['array'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'header_data_from_array_1_0(...)' code ##################

    str_106389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', ' Get the dictionary of header metadata from a numpy.ndarray.\n\n    Parameters\n    ----------\n    array : numpy.ndarray\n\n    Returns\n    -------\n    d : dict\n        This has the appropriate entries for writing its string representation\n        to the header of the file.\n    ')
    
    # Assigning a Dict to a Name (line 268):
    
    # Assigning a Dict to a Name (line 268):
    
    # Obtaining an instance of the builtin type 'dict' (line 268)
    dict_106390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 268)
    # Adding element type (key, value) (line 268)
    str_106391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 9), 'str', 'shape')
    # Getting the type of 'array' (line 268)
    array_106392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 18), 'array')
    # Obtaining the member 'shape' of a type (line 268)
    shape_106393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 18), array_106392, 'shape')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 8), dict_106390, (str_106391, shape_106393))
    
    # Assigning a type to the variable 'd' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'd', dict_106390)
    
    # Getting the type of 'array' (line 269)
    array_106394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 7), 'array')
    # Obtaining the member 'flags' of a type (line 269)
    flags_106395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 7), array_106394, 'flags')
    # Obtaining the member 'c_contiguous' of a type (line 269)
    c_contiguous_106396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 7), flags_106395, 'c_contiguous')
    # Testing the type of an if condition (line 269)
    if_condition_106397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 4), c_contiguous_106396)
    # Assigning a type to the variable 'if_condition_106397' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'if_condition_106397', if_condition_106397)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 270):
    
    # Assigning a Name to a Subscript (line 270):
    # Getting the type of 'False' (line 270)
    False_106398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'False')
    # Getting the type of 'd' (line 270)
    d_106399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'd')
    str_106400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 10), 'str', 'fortran_order')
    # Storing an element on a container (line 270)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), d_106399, (str_106400, False_106398))
    # SSA branch for the else part of an if statement (line 269)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'array' (line 271)
    array_106401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 9), 'array')
    # Obtaining the member 'flags' of a type (line 271)
    flags_106402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 9), array_106401, 'flags')
    # Obtaining the member 'f_contiguous' of a type (line 271)
    f_contiguous_106403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 9), flags_106402, 'f_contiguous')
    # Testing the type of an if condition (line 271)
    if_condition_106404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 9), f_contiguous_106403)
    # Assigning a type to the variable 'if_condition_106404' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 9), 'if_condition_106404', if_condition_106404)
    # SSA begins for if statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 272):
    
    # Assigning a Name to a Subscript (line 272):
    # Getting the type of 'True' (line 272)
    True_106405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'True')
    # Getting the type of 'd' (line 272)
    d_106406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'd')
    str_106407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 10), 'str', 'fortran_order')
    # Storing an element on a container (line 272)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 8), d_106406, (str_106407, True_106405))
    # SSA branch for the else part of an if statement (line 271)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 277):
    
    # Assigning a Name to a Subscript (line 277):
    # Getting the type of 'False' (line 277)
    False_106408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 29), 'False')
    # Getting the type of 'd' (line 277)
    d_106409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'd')
    str_106410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 10), 'str', 'fortran_order')
    # Storing an element on a container (line 277)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 8), d_106409, (str_106410, False_106408))
    # SSA join for if statement (line 271)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 279):
    
    # Assigning a Call to a Subscript (line 279):
    
    # Call to dtype_to_descr(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'array' (line 279)
    array_106412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 32), 'array', False)
    # Obtaining the member 'dtype' of a type (line 279)
    dtype_106413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 32), array_106412, 'dtype')
    # Processing the call keyword arguments (line 279)
    kwargs_106414 = {}
    # Getting the type of 'dtype_to_descr' (line 279)
    dtype_to_descr_106411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'dtype_to_descr', False)
    # Calling dtype_to_descr(args, kwargs) (line 279)
    dtype_to_descr_call_result_106415 = invoke(stypy.reporting.localization.Localization(__file__, 279, 17), dtype_to_descr_106411, *[dtype_106413], **kwargs_106414)
    
    # Getting the type of 'd' (line 279)
    d_106416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'd')
    str_106417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 6), 'str', 'descr')
    # Storing an element on a container (line 279)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 4), d_106416, (str_106417, dtype_to_descr_call_result_106415))
    # Getting the type of 'd' (line 280)
    d_106418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type', d_106418)
    
    # ################# End of 'header_data_from_array_1_0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'header_data_from_array_1_0' in the type store
    # Getting the type of 'stypy_return_type' (line 255)
    stypy_return_type_106419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106419)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'header_data_from_array_1_0'
    return stypy_return_type_106419

# Assigning a type to the variable 'header_data_from_array_1_0' (line 255)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'header_data_from_array_1_0', header_data_from_array_1_0)

@norecursion
def _write_array_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 282)
    None_106420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 39), 'None')
    defaults = [None_106420]
    # Create a new context for function '_write_array_header'
    module_type_store = module_type_store.open_function_context('_write_array_header', 282, 0, False)
    
    # Passed parameters checking function
    _write_array_header.stypy_localization = localization
    _write_array_header.stypy_type_of_self = None
    _write_array_header.stypy_type_store = module_type_store
    _write_array_header.stypy_function_name = '_write_array_header'
    _write_array_header.stypy_param_names_list = ['fp', 'd', 'version']
    _write_array_header.stypy_varargs_param_name = None
    _write_array_header.stypy_kwargs_param_name = None
    _write_array_header.stypy_call_defaults = defaults
    _write_array_header.stypy_call_varargs = varargs
    _write_array_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_write_array_header', ['fp', 'd', 'version'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_write_array_header', localization, ['fp', 'd', 'version'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_write_array_header(...)' code ##################

    str_106421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, (-1)), 'str', ' Write the header for an array and returns the version used\n\n    Parameters\n    ----------\n    fp : filelike object\n    d : dict\n        This has the appropriate entries for writing its string representation\n        to the header of the file.\n    version: tuple or None\n        None means use oldest that works\n        explicit version will raise a ValueError if the format does not\n        allow saving this data.  Default: None\n    Returns\n    -------\n    version : tuple of int\n        the file version which needs to be used to store the data\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 300, 4))
    
    # 'import struct' statement (line 300)
    import struct

    import_module(stypy.reporting.localization.Localization(__file__, 300, 4), 'struct', struct, module_type_store)
    
    
    # Assigning a List to a Name (line 301):
    
    # Assigning a List to a Name (line 301):
    
    # Obtaining an instance of the builtin type 'list' (line 301)
    list_106422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 301)
    # Adding element type (line 301)
    str_106423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 14), 'str', '{')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 13), list_106422, str_106423)
    
    # Assigning a type to the variable 'header' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'header', list_106422)
    
    
    # Call to sorted(...): (line 302)
    # Processing the call arguments (line 302)
    
    # Call to items(...): (line 302)
    # Processing the call keyword arguments (line 302)
    kwargs_106427 = {}
    # Getting the type of 'd' (line 302)
    d_106425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 29), 'd', False)
    # Obtaining the member 'items' of a type (line 302)
    items_106426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 29), d_106425, 'items')
    # Calling items(args, kwargs) (line 302)
    items_call_result_106428 = invoke(stypy.reporting.localization.Localization(__file__, 302, 29), items_106426, *[], **kwargs_106427)
    
    # Processing the call keyword arguments (line 302)
    kwargs_106429 = {}
    # Getting the type of 'sorted' (line 302)
    sorted_106424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 22), 'sorted', False)
    # Calling sorted(args, kwargs) (line 302)
    sorted_call_result_106430 = invoke(stypy.reporting.localization.Localization(__file__, 302, 22), sorted_106424, *[items_call_result_106428], **kwargs_106429)
    
    # Testing the type of a for loop iterable (line 302)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 302, 4), sorted_call_result_106430)
    # Getting the type of the for loop variable (line 302)
    for_loop_var_106431 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 302, 4), sorted_call_result_106430)
    # Assigning a type to the variable 'key' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 4), for_loop_var_106431))
    # Assigning a type to the variable 'value' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 4), for_loop_var_106431))
    # SSA begins for a for statement (line 302)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 304)
    # Processing the call arguments (line 304)
    str_106434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 22), 'str', "'%s': %s, ")
    
    # Obtaining an instance of the builtin type 'tuple' (line 304)
    tuple_106435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 304)
    # Adding element type (line 304)
    # Getting the type of 'key' (line 304)
    key_106436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 38), 'key', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 38), tuple_106435, key_106436)
    # Adding element type (line 304)
    
    # Call to repr(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'value' (line 304)
    value_106438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 48), 'value', False)
    # Processing the call keyword arguments (line 304)
    kwargs_106439 = {}
    # Getting the type of 'repr' (line 304)
    repr_106437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'repr', False)
    # Calling repr(args, kwargs) (line 304)
    repr_call_result_106440 = invoke(stypy.reporting.localization.Localization(__file__, 304, 43), repr_106437, *[value_106438], **kwargs_106439)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 38), tuple_106435, repr_call_result_106440)
    
    # Applying the binary operator '%' (line 304)
    result_mod_106441 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 22), '%', str_106434, tuple_106435)
    
    # Processing the call keyword arguments (line 304)
    kwargs_106442 = {}
    # Getting the type of 'header' (line 304)
    header_106432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'header', False)
    # Obtaining the member 'append' of a type (line 304)
    append_106433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), header_106432, 'append')
    # Calling append(args, kwargs) (line 304)
    append_call_result_106443 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), append_106433, *[result_mod_106441], **kwargs_106442)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 305)
    # Processing the call arguments (line 305)
    str_106446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 18), 'str', '}')
    # Processing the call keyword arguments (line 305)
    kwargs_106447 = {}
    # Getting the type of 'header' (line 305)
    header_106444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'header', False)
    # Obtaining the member 'append' of a type (line 305)
    append_106445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 4), header_106444, 'append')
    # Calling append(args, kwargs) (line 305)
    append_call_result_106448 = invoke(stypy.reporting.localization.Localization(__file__, 305, 4), append_106445, *[str_106446], **kwargs_106447)
    
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to join(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'header' (line 306)
    header_106451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'header', False)
    # Processing the call keyword arguments (line 306)
    kwargs_106452 = {}
    str_106449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 13), 'str', '')
    # Obtaining the member 'join' of a type (line 306)
    join_106450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 13), str_106449, 'join')
    # Calling join(args, kwargs) (line 306)
    join_call_result_106453 = invoke(stypy.reporting.localization.Localization(__file__, 306, 13), join_106450, *[header_106451], **kwargs_106452)
    
    # Assigning a type to the variable 'header' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'header', join_call_result_106453)
    
    # Assigning a BinOp to a Name (line 311):
    
    # Assigning a BinOp to a Name (line 311):
    # Getting the type of 'MAGIC_LEN' (line 311)
    MAGIC_LEN_106454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 25), 'MAGIC_LEN')
    int_106455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 37), 'int')
    # Applying the binary operator '+' (line 311)
    result_add_106456 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 25), '+', MAGIC_LEN_106454, int_106455)
    
    
    # Call to len(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'header' (line 311)
    header_106458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 45), 'header', False)
    # Processing the call keyword arguments (line 311)
    kwargs_106459 = {}
    # Getting the type of 'len' (line 311)
    len_106457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 41), 'len', False)
    # Calling len(args, kwargs) (line 311)
    len_call_result_106460 = invoke(stypy.reporting.localization.Localization(__file__, 311, 41), len_106457, *[header_106458], **kwargs_106459)
    
    # Applying the binary operator '+' (line 311)
    result_add_106461 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 39), '+', result_add_106456, len_call_result_106460)
    
    int_106462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 55), 'int')
    # Applying the binary operator '+' (line 311)
    result_add_106463 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 53), '+', result_add_106461, int_106462)
    
    # Assigning a type to the variable 'current_header_len' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'current_header_len', result_add_106463)
    
    # Assigning a BinOp to a Name (line 312):
    
    # Assigning a BinOp to a Name (line 312):
    int_106464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'int')
    # Getting the type of 'current_header_len' (line 312)
    current_header_len_106465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'current_header_len')
    int_106466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 39), 'int')
    # Applying the binary operator '%' (line 312)
    result_mod_106467 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 18), '%', current_header_len_106465, int_106466)
    
    # Applying the binary operator '-' (line 312)
    result_sub_106468 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 12), '-', int_106464, result_mod_106467)
    
    # Assigning a type to the variable 'topad' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'topad', result_sub_106468)
    
    # Assigning a BinOp to a Name (line 313):
    
    # Assigning a BinOp to a Name (line 313):
    # Getting the type of 'header' (line 313)
    header_106469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'header')
    str_106470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 22), 'str', ' ')
    # Getting the type of 'topad' (line 313)
    topad_106471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'topad')
    # Applying the binary operator '*' (line 313)
    result_mul_106472 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 22), '*', str_106470, topad_106471)
    
    # Applying the binary operator '+' (line 313)
    result_add_106473 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 13), '+', header_106469, result_mul_106472)
    
    str_106474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 34), 'str', '\n')
    # Applying the binary operator '+' (line 313)
    result_add_106475 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 32), '+', result_add_106473, str_106474)
    
    # Assigning a type to the variable 'header' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'header', result_add_106475)
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to asbytes(...): (line 314)
    # Processing the call arguments (line 314)
    
    # Call to _filter_header(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'header' (line 314)
    header_106478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'header', False)
    # Processing the call keyword arguments (line 314)
    kwargs_106479 = {}
    # Getting the type of '_filter_header' (line 314)
    _filter_header_106477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 21), '_filter_header', False)
    # Calling _filter_header(args, kwargs) (line 314)
    _filter_header_call_result_106480 = invoke(stypy.reporting.localization.Localization(__file__, 314, 21), _filter_header_106477, *[header_106478], **kwargs_106479)
    
    # Processing the call keyword arguments (line 314)
    kwargs_106481 = {}
    # Getting the type of 'asbytes' (line 314)
    asbytes_106476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 13), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 314)
    asbytes_call_result_106482 = invoke(stypy.reporting.localization.Localization(__file__, 314, 13), asbytes_106476, *[_filter_header_call_result_106480], **kwargs_106481)
    
    # Assigning a type to the variable 'header' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'header', asbytes_call_result_106482)
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to len(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'header' (line 316)
    header_106484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 15), 'header', False)
    # Processing the call keyword arguments (line 316)
    kwargs_106485 = {}
    # Getting the type of 'len' (line 316)
    len_106483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 'len', False)
    # Calling len(args, kwargs) (line 316)
    len_call_result_106486 = invoke(stypy.reporting.localization.Localization(__file__, 316, 11), len_106483, *[header_106484], **kwargs_106485)
    
    # Assigning a type to the variable 'hlen' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'hlen', len_call_result_106486)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'hlen' (line 317)
    hlen_106487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 7), 'hlen')
    int_106488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 14), 'int')
    int_106489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 18), 'int')
    # Applying the binary operator '*' (line 317)
    result_mul_106490 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 14), '*', int_106488, int_106489)
    
    # Applying the binary operator '<' (line 317)
    result_lt_106491 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 7), '<', hlen_106487, result_mul_106490)
    
    
    # Getting the type of 'version' (line 317)
    version_106492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'version')
    
    # Obtaining an instance of the builtin type 'tuple' (line 317)
    tuple_106493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 317)
    # Adding element type (line 317)
    # Getting the type of 'None' (line 317)
    None_106494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 38), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 38), tuple_106493, None_106494)
    # Adding element type (line 317)
    
    # Obtaining an instance of the builtin type 'tuple' (line 317)
    tuple_106495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 317)
    # Adding element type (line 317)
    int_106496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 45), tuple_106495, int_106496)
    # Adding element type (line 317)
    int_106497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 45), tuple_106495, int_106497)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 38), tuple_106493, tuple_106495)
    
    # Applying the binary operator 'in' (line 317)
    result_contains_106498 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 26), 'in', version_106492, tuple_106493)
    
    # Applying the binary operator 'and' (line 317)
    result_and_keyword_106499 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 7), 'and', result_lt_106491, result_contains_106498)
    
    # Testing the type of an if condition (line 317)
    if_condition_106500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 4), result_and_keyword_106499)
    # Assigning a type to the variable 'if_condition_106500' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'if_condition_106500', if_condition_106500)
    # SSA begins for if statement (line 317)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 318):
    
    # Assigning a Tuple to a Name (line 318):
    
    # Obtaining an instance of the builtin type 'tuple' (line 318)
    tuple_106501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 318)
    # Adding element type (line 318)
    int_106502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 19), tuple_106501, int_106502)
    # Adding element type (line 318)
    int_106503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 19), tuple_106501, int_106503)
    
    # Assigning a type to the variable 'version' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'version', tuple_106501)
    
    # Assigning a BinOp to a Name (line 319):
    
    # Assigning a BinOp to a Name (line 319):
    
    # Call to magic(...): (line 319)
    # Processing the call arguments (line 319)
    int_106505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'int')
    int_106506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 33), 'int')
    # Processing the call keyword arguments (line 319)
    kwargs_106507 = {}
    # Getting the type of 'magic' (line 319)
    magic_106504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'magic', False)
    # Calling magic(args, kwargs) (line 319)
    magic_call_result_106508 = invoke(stypy.reporting.localization.Localization(__file__, 319, 24), magic_106504, *[int_106505, int_106506], **kwargs_106507)
    
    
    # Call to pack(...): (line 319)
    # Processing the call arguments (line 319)
    str_106511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 50), 'str', '<H')
    # Getting the type of 'hlen' (line 319)
    hlen_106512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 56), 'hlen', False)
    # Processing the call keyword arguments (line 319)
    kwargs_106513 = {}
    # Getting the type of 'struct' (line 319)
    struct_106509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 38), 'struct', False)
    # Obtaining the member 'pack' of a type (line 319)
    pack_106510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 38), struct_106509, 'pack')
    # Calling pack(args, kwargs) (line 319)
    pack_call_result_106514 = invoke(stypy.reporting.localization.Localization(__file__, 319, 38), pack_106510, *[str_106511, hlen_106512], **kwargs_106513)
    
    # Applying the binary operator '+' (line 319)
    result_add_106515 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 24), '+', magic_call_result_106508, pack_call_result_106514)
    
    # Assigning a type to the variable 'header_prefix' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'header_prefix', result_add_106515)
    # SSA branch for the else part of an if statement (line 317)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'hlen' (line 320)
    hlen_106516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 9), 'hlen')
    int_106517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 16), 'int')
    int_106518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 19), 'int')
    # Applying the binary operator '**' (line 320)
    result_pow_106519 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 16), '**', int_106517, int_106518)
    
    # Applying the binary operator '<' (line 320)
    result_lt_106520 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 9), '<', hlen_106516, result_pow_106519)
    
    
    # Getting the type of 'version' (line 320)
    version_106521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'version')
    
    # Obtaining an instance of the builtin type 'tuple' (line 320)
    tuple_106522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 320)
    # Adding element type (line 320)
    # Getting the type of 'None' (line 320)
    None_106523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 38), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 38), tuple_106522, None_106523)
    # Adding element type (line 320)
    
    # Obtaining an instance of the builtin type 'tuple' (line 320)
    tuple_106524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 320)
    # Adding element type (line 320)
    int_106525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 45), tuple_106524, int_106525)
    # Adding element type (line 320)
    int_106526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 45), tuple_106524, int_106526)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 38), tuple_106522, tuple_106524)
    
    # Applying the binary operator 'in' (line 320)
    result_contains_106527 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 26), 'in', version_106521, tuple_106522)
    
    # Applying the binary operator 'and' (line 320)
    result_and_keyword_106528 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 9), 'and', result_lt_106520, result_contains_106527)
    
    # Testing the type of an if condition (line 320)
    if_condition_106529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 9), result_and_keyword_106528)
    # Assigning a type to the variable 'if_condition_106529' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 9), 'if_condition_106529', if_condition_106529)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 321):
    
    # Assigning a Tuple to a Name (line 321):
    
    # Obtaining an instance of the builtin type 'tuple' (line 321)
    tuple_106530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 321)
    # Adding element type (line 321)
    int_106531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 19), tuple_106530, int_106531)
    # Adding element type (line 321)
    int_106532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 19), tuple_106530, int_106532)
    
    # Assigning a type to the variable 'version' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'version', tuple_106530)
    
    # Assigning a BinOp to a Name (line 322):
    
    # Assigning a BinOp to a Name (line 322):
    
    # Call to magic(...): (line 322)
    # Processing the call arguments (line 322)
    int_106534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 30), 'int')
    int_106535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 33), 'int')
    # Processing the call keyword arguments (line 322)
    kwargs_106536 = {}
    # Getting the type of 'magic' (line 322)
    magic_106533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 24), 'magic', False)
    # Calling magic(args, kwargs) (line 322)
    magic_call_result_106537 = invoke(stypy.reporting.localization.Localization(__file__, 322, 24), magic_106533, *[int_106534, int_106535], **kwargs_106536)
    
    
    # Call to pack(...): (line 322)
    # Processing the call arguments (line 322)
    str_106540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 50), 'str', '<I')
    # Getting the type of 'hlen' (line 322)
    hlen_106541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'hlen', False)
    # Processing the call keyword arguments (line 322)
    kwargs_106542 = {}
    # Getting the type of 'struct' (line 322)
    struct_106538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 38), 'struct', False)
    # Obtaining the member 'pack' of a type (line 322)
    pack_106539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 38), struct_106538, 'pack')
    # Calling pack(args, kwargs) (line 322)
    pack_call_result_106543 = invoke(stypy.reporting.localization.Localization(__file__, 322, 38), pack_106539, *[str_106540, hlen_106541], **kwargs_106542)
    
    # Applying the binary operator '+' (line 322)
    result_add_106544 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 24), '+', magic_call_result_106537, pack_call_result_106543)
    
    # Assigning a type to the variable 'header_prefix' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'header_prefix', result_add_106544)
    # SSA branch for the else part of an if statement (line 320)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 324):
    
    # Assigning a Str to a Name (line 324):
    str_106545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 14), 'str', 'Header length %s too big for version=%s')
    # Assigning a type to the variable 'msg' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'msg', str_106545)
    
    # Getting the type of 'msg' (line 325)
    msg_106546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'msg')
    
    # Obtaining an instance of the builtin type 'tuple' (line 325)
    tuple_106547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 325)
    # Adding element type (line 325)
    # Getting the type of 'hlen' (line 325)
    hlen_106548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'hlen')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 16), tuple_106547, hlen_106548)
    # Adding element type (line 325)
    # Getting the type of 'version' (line 325)
    version_106549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 22), 'version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 16), tuple_106547, version_106549)
    
    # Applying the binary operator '%=' (line 325)
    result_imod_106550 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 8), '%=', msg_106546, tuple_106547)
    # Assigning a type to the variable 'msg' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'msg', result_imod_106550)
    
    
    # Call to ValueError(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'msg' (line 326)
    msg_106552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 25), 'msg', False)
    # Processing the call keyword arguments (line 326)
    kwargs_106553 = {}
    # Getting the type of 'ValueError' (line 326)
    ValueError_106551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 326)
    ValueError_call_result_106554 = invoke(stypy.reporting.localization.Localization(__file__, 326, 14), ValueError_106551, *[msg_106552], **kwargs_106553)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 326, 8), ValueError_call_result_106554, 'raise parameter', BaseException)
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 317)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'header_prefix' (line 328)
    header_prefix_106557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 13), 'header_prefix', False)
    # Processing the call keyword arguments (line 328)
    kwargs_106558 = {}
    # Getting the type of 'fp' (line 328)
    fp_106555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'fp', False)
    # Obtaining the member 'write' of a type (line 328)
    write_106556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 4), fp_106555, 'write')
    # Calling write(args, kwargs) (line 328)
    write_call_result_106559 = invoke(stypy.reporting.localization.Localization(__file__, 328, 4), write_106556, *[header_prefix_106557], **kwargs_106558)
    
    
    # Call to write(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'header' (line 329)
    header_106562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 13), 'header', False)
    # Processing the call keyword arguments (line 329)
    kwargs_106563 = {}
    # Getting the type of 'fp' (line 329)
    fp_106560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'fp', False)
    # Obtaining the member 'write' of a type (line 329)
    write_106561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 4), fp_106560, 'write')
    # Calling write(args, kwargs) (line 329)
    write_call_result_106564 = invoke(stypy.reporting.localization.Localization(__file__, 329, 4), write_106561, *[header_106562], **kwargs_106563)
    
    # Getting the type of 'version' (line 330)
    version_106565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'version')
    # Assigning a type to the variable 'stypy_return_type' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type', version_106565)
    
    # ################# End of '_write_array_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_write_array_header' in the type store
    # Getting the type of 'stypy_return_type' (line 282)
    stypy_return_type_106566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106566)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_write_array_header'
    return stypy_return_type_106566

# Assigning a type to the variable '_write_array_header' (line 282)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), '_write_array_header', _write_array_header)

@norecursion
def write_array_header_1_0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'write_array_header_1_0'
    module_type_store = module_type_store.open_function_context('write_array_header_1_0', 332, 0, False)
    
    # Passed parameters checking function
    write_array_header_1_0.stypy_localization = localization
    write_array_header_1_0.stypy_type_of_self = None
    write_array_header_1_0.stypy_type_store = module_type_store
    write_array_header_1_0.stypy_function_name = 'write_array_header_1_0'
    write_array_header_1_0.stypy_param_names_list = ['fp', 'd']
    write_array_header_1_0.stypy_varargs_param_name = None
    write_array_header_1_0.stypy_kwargs_param_name = None
    write_array_header_1_0.stypy_call_defaults = defaults
    write_array_header_1_0.stypy_call_varargs = varargs
    write_array_header_1_0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'write_array_header_1_0', ['fp', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'write_array_header_1_0', localization, ['fp', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'write_array_header_1_0(...)' code ##################

    str_106567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', ' Write the header for an array using the 1.0 format.\n\n    Parameters\n    ----------\n    fp : filelike object\n    d : dict\n        This has the appropriate entries for writing its string\n        representation to the header of the file.\n    ')
    
    # Call to _write_array_header(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'fp' (line 342)
    fp_106569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 24), 'fp', False)
    # Getting the type of 'd' (line 342)
    d_106570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 28), 'd', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 342)
    tuple_106571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 342)
    # Adding element type (line 342)
    int_106572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 32), tuple_106571, int_106572)
    # Adding element type (line 342)
    int_106573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 32), tuple_106571, int_106573)
    
    # Processing the call keyword arguments (line 342)
    kwargs_106574 = {}
    # Getting the type of '_write_array_header' (line 342)
    _write_array_header_106568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), '_write_array_header', False)
    # Calling _write_array_header(args, kwargs) (line 342)
    _write_array_header_call_result_106575 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), _write_array_header_106568, *[fp_106569, d_106570, tuple_106571], **kwargs_106574)
    
    
    # ################# End of 'write_array_header_1_0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'write_array_header_1_0' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_106576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106576)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'write_array_header_1_0'
    return stypy_return_type_106576

# Assigning a type to the variable 'write_array_header_1_0' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'write_array_header_1_0', write_array_header_1_0)

@norecursion
def write_array_header_2_0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'write_array_header_2_0'
    module_type_store = module_type_store.open_function_context('write_array_header_2_0', 345, 0, False)
    
    # Passed parameters checking function
    write_array_header_2_0.stypy_localization = localization
    write_array_header_2_0.stypy_type_of_self = None
    write_array_header_2_0.stypy_type_store = module_type_store
    write_array_header_2_0.stypy_function_name = 'write_array_header_2_0'
    write_array_header_2_0.stypy_param_names_list = ['fp', 'd']
    write_array_header_2_0.stypy_varargs_param_name = None
    write_array_header_2_0.stypy_kwargs_param_name = None
    write_array_header_2_0.stypy_call_defaults = defaults
    write_array_header_2_0.stypy_call_varargs = varargs
    write_array_header_2_0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'write_array_header_2_0', ['fp', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'write_array_header_2_0', localization, ['fp', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'write_array_header_2_0(...)' code ##################

    str_106577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, (-1)), 'str', ' Write the header for an array using the 2.0 format.\n        The 2.0 format allows storing very large structured arrays.\n\n    .. versionadded:: 1.9.0\n\n    Parameters\n    ----------\n    fp : filelike object\n    d : dict\n        This has the appropriate entries for writing its string\n        representation to the header of the file.\n    ')
    
    # Call to _write_array_header(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'fp' (line 358)
    fp_106579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'fp', False)
    # Getting the type of 'd' (line 358)
    d_106580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 28), 'd', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 358)
    tuple_106581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 358)
    # Adding element type (line 358)
    int_106582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 32), tuple_106581, int_106582)
    # Adding element type (line 358)
    int_106583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 32), tuple_106581, int_106583)
    
    # Processing the call keyword arguments (line 358)
    kwargs_106584 = {}
    # Getting the type of '_write_array_header' (line 358)
    _write_array_header_106578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), '_write_array_header', False)
    # Calling _write_array_header(args, kwargs) (line 358)
    _write_array_header_call_result_106585 = invoke(stypy.reporting.localization.Localization(__file__, 358, 4), _write_array_header_106578, *[fp_106579, d_106580, tuple_106581], **kwargs_106584)
    
    
    # ################# End of 'write_array_header_2_0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'write_array_header_2_0' in the type store
    # Getting the type of 'stypy_return_type' (line 345)
    stypy_return_type_106586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106586)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'write_array_header_2_0'
    return stypy_return_type_106586

# Assigning a type to the variable 'write_array_header_2_0' (line 345)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 0), 'write_array_header_2_0', write_array_header_2_0)

@norecursion
def read_array_header_1_0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_array_header_1_0'
    module_type_store = module_type_store.open_function_context('read_array_header_1_0', 360, 0, False)
    
    # Passed parameters checking function
    read_array_header_1_0.stypy_localization = localization
    read_array_header_1_0.stypy_type_of_self = None
    read_array_header_1_0.stypy_type_store = module_type_store
    read_array_header_1_0.stypy_function_name = 'read_array_header_1_0'
    read_array_header_1_0.stypy_param_names_list = ['fp']
    read_array_header_1_0.stypy_varargs_param_name = None
    read_array_header_1_0.stypy_kwargs_param_name = None
    read_array_header_1_0.stypy_call_defaults = defaults
    read_array_header_1_0.stypy_call_varargs = varargs
    read_array_header_1_0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_array_header_1_0', ['fp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_array_header_1_0', localization, ['fp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_array_header_1_0(...)' code ##################

    str_106587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, (-1)), 'str', "\n    Read an array header from a filelike object using the 1.0 file format\n    version.\n\n    This will leave the file object located just after the header.\n\n    Parameters\n    ----------\n    fp : filelike object\n        A file object or something with a `.read()` method like a file.\n\n    Returns\n    -------\n    shape : tuple of int\n        The shape of the array.\n    fortran_order : bool\n        The array data will be written out directly if it is either\n        C-contiguous or Fortran-contiguous. Otherwise, it will be made\n        contiguous before writing it out.\n    dtype : dtype\n        The dtype of the file's data.\n\n    Raises\n    ------\n    ValueError\n        If the data is invalid.\n\n    ")
    
    # Call to _read_array_header(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'fp' (line 389)
    fp_106589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 30), 'fp', False)
    # Processing the call keyword arguments (line 389)
    
    # Obtaining an instance of the builtin type 'tuple' (line 389)
    tuple_106590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 389)
    # Adding element type (line 389)
    int_106591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 43), tuple_106590, int_106591)
    # Adding element type (line 389)
    int_106592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 43), tuple_106590, int_106592)
    
    keyword_106593 = tuple_106590
    kwargs_106594 = {'version': keyword_106593}
    # Getting the type of '_read_array_header' (line 389)
    _read_array_header_106588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), '_read_array_header', False)
    # Calling _read_array_header(args, kwargs) (line 389)
    _read_array_header_call_result_106595 = invoke(stypy.reporting.localization.Localization(__file__, 389, 11), _read_array_header_106588, *[fp_106589], **kwargs_106594)
    
    # Assigning a type to the variable 'stypy_return_type' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'stypy_return_type', _read_array_header_call_result_106595)
    
    # ################# End of 'read_array_header_1_0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_array_header_1_0' in the type store
    # Getting the type of 'stypy_return_type' (line 360)
    stypy_return_type_106596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106596)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_array_header_1_0'
    return stypy_return_type_106596

# Assigning a type to the variable 'read_array_header_1_0' (line 360)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'read_array_header_1_0', read_array_header_1_0)

@norecursion
def read_array_header_2_0(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_array_header_2_0'
    module_type_store = module_type_store.open_function_context('read_array_header_2_0', 391, 0, False)
    
    # Passed parameters checking function
    read_array_header_2_0.stypy_localization = localization
    read_array_header_2_0.stypy_type_of_self = None
    read_array_header_2_0.stypy_type_store = module_type_store
    read_array_header_2_0.stypy_function_name = 'read_array_header_2_0'
    read_array_header_2_0.stypy_param_names_list = ['fp']
    read_array_header_2_0.stypy_varargs_param_name = None
    read_array_header_2_0.stypy_kwargs_param_name = None
    read_array_header_2_0.stypy_call_defaults = defaults
    read_array_header_2_0.stypy_call_varargs = varargs
    read_array_header_2_0.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_array_header_2_0', ['fp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_array_header_2_0', localization, ['fp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_array_header_2_0(...)' code ##################

    str_106597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', "\n    Read an array header from a filelike object using the 2.0 file format\n    version.\n\n    This will leave the file object located just after the header.\n\n    .. versionadded:: 1.9.0\n\n    Parameters\n    ----------\n    fp : filelike object\n        A file object or something with a `.read()` method like a file.\n\n    Returns\n    -------\n    shape : tuple of int\n        The shape of the array.\n    fortran_order : bool\n        The array data will be written out directly if it is either\n        C-contiguous or Fortran-contiguous. Otherwise, it will be made\n        contiguous before writing it out.\n    dtype : dtype\n        The dtype of the file's data.\n\n    Raises\n    ------\n    ValueError\n        If the data is invalid.\n\n    ")
    
    # Call to _read_array_header(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'fp' (line 422)
    fp_106599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 30), 'fp', False)
    # Processing the call keyword arguments (line 422)
    
    # Obtaining an instance of the builtin type 'tuple' (line 422)
    tuple_106600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 422)
    # Adding element type (line 422)
    int_106601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 43), tuple_106600, int_106601)
    # Adding element type (line 422)
    int_106602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 43), tuple_106600, int_106602)
    
    keyword_106603 = tuple_106600
    kwargs_106604 = {'version': keyword_106603}
    # Getting the type of '_read_array_header' (line 422)
    _read_array_header_106598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), '_read_array_header', False)
    # Calling _read_array_header(args, kwargs) (line 422)
    _read_array_header_call_result_106605 = invoke(stypy.reporting.localization.Localization(__file__, 422, 11), _read_array_header_106598, *[fp_106599], **kwargs_106604)
    
    # Assigning a type to the variable 'stypy_return_type' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type', _read_array_header_call_result_106605)
    
    # ################# End of 'read_array_header_2_0(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_array_header_2_0' in the type store
    # Getting the type of 'stypy_return_type' (line 391)
    stypy_return_type_106606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106606)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_array_header_2_0'
    return stypy_return_type_106606

# Assigning a type to the variable 'read_array_header_2_0' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'read_array_header_2_0', read_array_header_2_0)

@norecursion
def _filter_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_filter_header'
    module_type_store = module_type_store.open_function_context('_filter_header', 425, 0, False)
    
    # Passed parameters checking function
    _filter_header.stypy_localization = localization
    _filter_header.stypy_type_of_self = None
    _filter_header.stypy_type_store = module_type_store
    _filter_header.stypy_function_name = '_filter_header'
    _filter_header.stypy_param_names_list = ['s']
    _filter_header.stypy_varargs_param_name = None
    _filter_header.stypy_kwargs_param_name = None
    _filter_header.stypy_call_defaults = defaults
    _filter_header.stypy_call_varargs = varargs
    _filter_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_filter_header', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_filter_header', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_filter_header(...)' code ##################

    str_106607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, (-1)), 'str', "Clean up 'L' in npz header ints.\n\n    Cleans up the 'L' in strings representing integers. Needed to allow npz\n    headers produced in Python2 to be read in Python3.\n\n    Parameters\n    ----------\n    s : byte string\n        Npy file header.\n\n    Returns\n    -------\n    header : str\n        Cleaned up header.\n\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 442, 4))
    
    # 'import tokenize' statement (line 442)
    import tokenize

    import_module(stypy.reporting.localization.Localization(__file__, 442, 4), 'tokenize', tokenize, module_type_store)
    
    
    
    
    # Obtaining the type of the subscript
    int_106608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 24), 'int')
    # Getting the type of 'sys' (line 443)
    sys_106609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 443)
    version_info_106610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 7), sys_106609, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 443)
    getitem___106611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 7), version_info_106610, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 443)
    subscript_call_result_106612 = invoke(stypy.reporting.localization.Localization(__file__, 443, 7), getitem___106611, int_106608)
    
    int_106613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 30), 'int')
    # Applying the binary operator '>=' (line 443)
    result_ge_106614 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 7), '>=', subscript_call_result_106612, int_106613)
    
    # Testing the type of an if condition (line 443)
    if_condition_106615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 4), result_ge_106614)
    # Assigning a type to the variable 'if_condition_106615' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'if_condition_106615', if_condition_106615)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 444, 8))
    
    # 'from io import StringIO' statement (line 444)
    from io import StringIO

    import_from_module(stypy.reporting.localization.Localization(__file__, 444, 8), 'io', None, module_type_store, ['StringIO'], [StringIO])
    
    # SSA branch for the else part of an if statement (line 443)
    module_type_store.open_ssa_branch('else')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 446, 8))
    
    # 'from StringIO import StringIO' statement (line 446)
    from StringIO import StringIO

    import_from_module(stypy.reporting.localization.Localization(__file__, 446, 8), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])
    
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 448):
    
    # Assigning a List to a Name (line 448):
    
    # Obtaining an instance of the builtin type 'list' (line 448)
    list_106616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 448)
    
    # Assigning a type to the variable 'tokens' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'tokens', list_106616)
    
    # Assigning a Name to a Name (line 449):
    
    # Assigning a Name to a Name (line 449):
    # Getting the type of 'False' (line 449)
    False_106617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 28), 'False')
    # Assigning a type to the variable 'last_token_was_number' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'last_token_was_number', False_106617)
    
    
    # Call to generate_tokens(...): (line 450)
    # Processing the call arguments (line 450)
    
    # Call to StringIO(...): (line 450)
    # Processing the call arguments (line 450)
    
    # Call to asstr(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 's' (line 450)
    s_106622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 57), 's', False)
    # Processing the call keyword arguments (line 450)
    kwargs_106623 = {}
    # Getting the type of 'asstr' (line 450)
    asstr_106621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 51), 'asstr', False)
    # Calling asstr(args, kwargs) (line 450)
    asstr_call_result_106624 = invoke(stypy.reporting.localization.Localization(__file__, 450, 51), asstr_106621, *[s_106622], **kwargs_106623)
    
    # Processing the call keyword arguments (line 450)
    kwargs_106625 = {}
    # Getting the type of 'StringIO' (line 450)
    StringIO_106620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 42), 'StringIO', False)
    # Calling StringIO(args, kwargs) (line 450)
    StringIO_call_result_106626 = invoke(stypy.reporting.localization.Localization(__file__, 450, 42), StringIO_106620, *[asstr_call_result_106624], **kwargs_106625)
    
    # Obtaining the member 'read' of a type (line 450)
    read_106627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 42), StringIO_call_result_106626, 'read')
    # Processing the call keyword arguments (line 450)
    kwargs_106628 = {}
    # Getting the type of 'tokenize' (line 450)
    tokenize_106618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'tokenize', False)
    # Obtaining the member 'generate_tokens' of a type (line 450)
    generate_tokens_106619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 17), tokenize_106618, 'generate_tokens')
    # Calling generate_tokens(args, kwargs) (line 450)
    generate_tokens_call_result_106629 = invoke(stypy.reporting.localization.Localization(__file__, 450, 17), generate_tokens_106619, *[read_106627], **kwargs_106628)
    
    # Testing the type of a for loop iterable (line 450)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 450, 4), generate_tokens_call_result_106629)
    # Getting the type of the for loop variable (line 450)
    for_loop_var_106630 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 450, 4), generate_tokens_call_result_106629)
    # Assigning a type to the variable 'token' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'token', for_loop_var_106630)
    # SSA begins for a for statement (line 450)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 451):
    
    # Assigning a Subscript to a Name (line 451):
    
    # Obtaining the type of the subscript
    int_106631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 27), 'int')
    # Getting the type of 'token' (line 451)
    token_106632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 21), 'token')
    # Obtaining the member '__getitem__' of a type (line 451)
    getitem___106633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 21), token_106632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 451)
    subscript_call_result_106634 = invoke(stypy.reporting.localization.Localization(__file__, 451, 21), getitem___106633, int_106631)
    
    # Assigning a type to the variable 'token_type' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'token_type', subscript_call_result_106634)
    
    # Assigning a Subscript to a Name (line 452):
    
    # Assigning a Subscript to a Name (line 452):
    
    # Obtaining the type of the subscript
    int_106635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 29), 'int')
    # Getting the type of 'token' (line 452)
    token_106636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 23), 'token')
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___106637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 23), token_106636, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_106638 = invoke(stypy.reporting.localization.Localization(__file__, 452, 23), getitem___106637, int_106635)
    
    # Assigning a type to the variable 'token_string' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'token_string', subscript_call_result_106638)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'last_token_was_number' (line 453)
    last_token_was_number_106639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'last_token_was_number')
    
    # Getting the type of 'token_type' (line 454)
    token_type_106640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), 'token_type')
    # Getting the type of 'tokenize' (line 454)
    tokenize_106641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 30), 'tokenize')
    # Obtaining the member 'NAME' of a type (line 454)
    NAME_106642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 30), tokenize_106641, 'NAME')
    # Applying the binary operator '==' (line 454)
    result_eq_106643 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 16), '==', token_type_106640, NAME_106642)
    
    # Applying the binary operator 'and' (line 453)
    result_and_keyword_106644 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 12), 'and', last_token_was_number_106639, result_eq_106643)
    
    # Getting the type of 'token_string' (line 455)
    token_string_106645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'token_string')
    str_106646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 32), 'str', 'L')
    # Applying the binary operator '==' (line 455)
    result_eq_106647 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 16), '==', token_string_106645, str_106646)
    
    # Applying the binary operator 'and' (line 453)
    result_and_keyword_106648 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 12), 'and', result_and_keyword_106644, result_eq_106647)
    
    # Testing the type of an if condition (line 453)
    if_condition_106649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), result_and_keyword_106648)
    # Assigning a type to the variable 'if_condition_106649' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_106649', if_condition_106649)
    # SSA begins for if statement (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 453)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'token' (line 458)
    token_106652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 26), 'token', False)
    # Processing the call keyword arguments (line 458)
    kwargs_106653 = {}
    # Getting the type of 'tokens' (line 458)
    tokens_106650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'tokens', False)
    # Obtaining the member 'append' of a type (line 458)
    append_106651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), tokens_106650, 'append')
    # Calling append(args, kwargs) (line 458)
    append_call_result_106654 = invoke(stypy.reporting.localization.Localization(__file__, 458, 12), append_106651, *[token_106652], **kwargs_106653)
    
    # SSA join for if statement (line 453)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Name (line 459):
    
    # Assigning a Compare to a Name (line 459):
    
    # Getting the type of 'token_type' (line 459)
    token_type_106655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 33), 'token_type')
    # Getting the type of 'tokenize' (line 459)
    tokenize_106656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 47), 'tokenize')
    # Obtaining the member 'NUMBER' of a type (line 459)
    NUMBER_106657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 47), tokenize_106656, 'NUMBER')
    # Applying the binary operator '==' (line 459)
    result_eq_106658 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 33), '==', token_type_106655, NUMBER_106657)
    
    # Assigning a type to the variable 'last_token_was_number' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'last_token_was_number', result_eq_106658)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to untokenize(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'tokens' (line 460)
    tokens_106661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 31), 'tokens', False)
    # Processing the call keyword arguments (line 460)
    kwargs_106662 = {}
    # Getting the type of 'tokenize' (line 460)
    tokenize_106659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'tokenize', False)
    # Obtaining the member 'untokenize' of a type (line 460)
    untokenize_106660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 11), tokenize_106659, 'untokenize')
    # Calling untokenize(args, kwargs) (line 460)
    untokenize_call_result_106663 = invoke(stypy.reporting.localization.Localization(__file__, 460, 11), untokenize_106660, *[tokens_106661], **kwargs_106662)
    
    # Assigning a type to the variable 'stypy_return_type' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'stypy_return_type', untokenize_call_result_106663)
    
    # ################# End of '_filter_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_filter_header' in the type store
    # Getting the type of 'stypy_return_type' (line 425)
    stypy_return_type_106664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106664)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_filter_header'
    return stypy_return_type_106664

# Assigning a type to the variable '_filter_header' (line 425)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 0), '_filter_header', _filter_header)

@norecursion
def _read_array_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_array_header'
    module_type_store = module_type_store.open_function_context('_read_array_header', 463, 0, False)
    
    # Passed parameters checking function
    _read_array_header.stypy_localization = localization
    _read_array_header.stypy_type_of_self = None
    _read_array_header.stypy_type_store = module_type_store
    _read_array_header.stypy_function_name = '_read_array_header'
    _read_array_header.stypy_param_names_list = ['fp', 'version']
    _read_array_header.stypy_varargs_param_name = None
    _read_array_header.stypy_kwargs_param_name = None
    _read_array_header.stypy_call_defaults = defaults
    _read_array_header.stypy_call_varargs = varargs
    _read_array_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_array_header', ['fp', 'version'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_array_header', localization, ['fp', 'version'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_array_header(...)' code ##################

    str_106665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, (-1)), 'str', '\n    see read_array_header_1_0\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 469, 4))
    
    # 'import struct' statement (line 469)
    import struct

    import_module(stypy.reporting.localization.Localization(__file__, 469, 4), 'struct', struct, module_type_store)
    
    
    
    # Getting the type of 'version' (line 470)
    version_106666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 7), 'version')
    
    # Obtaining an instance of the builtin type 'tuple' (line 470)
    tuple_106667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 470)
    # Adding element type (line 470)
    int_106668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 19), tuple_106667, int_106668)
    # Adding element type (line 470)
    int_106669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 19), tuple_106667, int_106669)
    
    # Applying the binary operator '==' (line 470)
    result_eq_106670 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 7), '==', version_106666, tuple_106667)
    
    # Testing the type of an if condition (line 470)
    if_condition_106671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 4), result_eq_106670)
    # Assigning a type to the variable 'if_condition_106671' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'if_condition_106671', if_condition_106671)
    # SSA begins for if statement (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 471):
    
    # Assigning a Call to a Name (line 471):
    
    # Call to _read_bytes(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'fp' (line 471)
    fp_106673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 34), 'fp', False)
    int_106674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 38), 'int')
    str_106675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 41), 'str', 'array header length')
    # Processing the call keyword arguments (line 471)
    kwargs_106676 = {}
    # Getting the type of '_read_bytes' (line 471)
    _read_bytes_106672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 22), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 471)
    _read_bytes_call_result_106677 = invoke(stypy.reporting.localization.Localization(__file__, 471, 22), _read_bytes_106672, *[fp_106673, int_106674, str_106675], **kwargs_106676)
    
    # Assigning a type to the variable 'hlength_str' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'hlength_str', _read_bytes_call_result_106677)
    
    # Assigning a Subscript to a Name (line 472):
    
    # Assigning a Subscript to a Name (line 472):
    
    # Obtaining the type of the subscript
    int_106678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 57), 'int')
    
    # Call to unpack(...): (line 472)
    # Processing the call arguments (line 472)
    str_106681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 38), 'str', '<H')
    # Getting the type of 'hlength_str' (line 472)
    hlength_str_106682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 44), 'hlength_str', False)
    # Processing the call keyword arguments (line 472)
    kwargs_106683 = {}
    # Getting the type of 'struct' (line 472)
    struct_106679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 24), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 472)
    unpack_106680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 24), struct_106679, 'unpack')
    # Calling unpack(args, kwargs) (line 472)
    unpack_call_result_106684 = invoke(stypy.reporting.localization.Localization(__file__, 472, 24), unpack_106680, *[str_106681, hlength_str_106682], **kwargs_106683)
    
    # Obtaining the member '__getitem__' of a type (line 472)
    getitem___106685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 24), unpack_call_result_106684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 472)
    subscript_call_result_106686 = invoke(stypy.reporting.localization.Localization(__file__, 472, 24), getitem___106685, int_106678)
    
    # Assigning a type to the variable 'header_length' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'header_length', subscript_call_result_106686)
    
    # Assigning a Call to a Name (line 473):
    
    # Assigning a Call to a Name (line 473):
    
    # Call to _read_bytes(...): (line 473)
    # Processing the call arguments (line 473)
    # Getting the type of 'fp' (line 473)
    fp_106688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 29), 'fp', False)
    # Getting the type of 'header_length' (line 473)
    header_length_106689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 33), 'header_length', False)
    str_106690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 48), 'str', 'array header')
    # Processing the call keyword arguments (line 473)
    kwargs_106691 = {}
    # Getting the type of '_read_bytes' (line 473)
    _read_bytes_106687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 17), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 473)
    _read_bytes_call_result_106692 = invoke(stypy.reporting.localization.Localization(__file__, 473, 17), _read_bytes_106687, *[fp_106688, header_length_106689, str_106690], **kwargs_106691)
    
    # Assigning a type to the variable 'header' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'header', _read_bytes_call_result_106692)
    # SSA branch for the else part of an if statement (line 470)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'version' (line 474)
    version_106693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 9), 'version')
    
    # Obtaining an instance of the builtin type 'tuple' (line 474)
    tuple_106694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 474)
    # Adding element type (line 474)
    int_106695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 21), tuple_106694, int_106695)
    # Adding element type (line 474)
    int_106696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 21), tuple_106694, int_106696)
    
    # Applying the binary operator '==' (line 474)
    result_eq_106697 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 9), '==', version_106693, tuple_106694)
    
    # Testing the type of an if condition (line 474)
    if_condition_106698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 9), result_eq_106697)
    # Assigning a type to the variable 'if_condition_106698' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 9), 'if_condition_106698', if_condition_106698)
    # SSA begins for if statement (line 474)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to _read_bytes(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'fp' (line 475)
    fp_106700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 34), 'fp', False)
    int_106701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 38), 'int')
    str_106702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 41), 'str', 'array header length')
    # Processing the call keyword arguments (line 475)
    kwargs_106703 = {}
    # Getting the type of '_read_bytes' (line 475)
    _read_bytes_106699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 22), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 475)
    _read_bytes_call_result_106704 = invoke(stypy.reporting.localization.Localization(__file__, 475, 22), _read_bytes_106699, *[fp_106700, int_106701, str_106702], **kwargs_106703)
    
    # Assigning a type to the variable 'hlength_str' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'hlength_str', _read_bytes_call_result_106704)
    
    # Assigning a Subscript to a Name (line 476):
    
    # Assigning a Subscript to a Name (line 476):
    
    # Obtaining the type of the subscript
    int_106705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 57), 'int')
    
    # Call to unpack(...): (line 476)
    # Processing the call arguments (line 476)
    str_106708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 38), 'str', '<I')
    # Getting the type of 'hlength_str' (line 476)
    hlength_str_106709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 44), 'hlength_str', False)
    # Processing the call keyword arguments (line 476)
    kwargs_106710 = {}
    # Getting the type of 'struct' (line 476)
    struct_106706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 24), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 476)
    unpack_106707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 24), struct_106706, 'unpack')
    # Calling unpack(args, kwargs) (line 476)
    unpack_call_result_106711 = invoke(stypy.reporting.localization.Localization(__file__, 476, 24), unpack_106707, *[str_106708, hlength_str_106709], **kwargs_106710)
    
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___106712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 24), unpack_call_result_106711, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_106713 = invoke(stypy.reporting.localization.Localization(__file__, 476, 24), getitem___106712, int_106705)
    
    # Assigning a type to the variable 'header_length' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'header_length', subscript_call_result_106713)
    
    # Assigning a Call to a Name (line 477):
    
    # Assigning a Call to a Name (line 477):
    
    # Call to _read_bytes(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'fp' (line 477)
    fp_106715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 29), 'fp', False)
    # Getting the type of 'header_length' (line 477)
    header_length_106716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 33), 'header_length', False)
    str_106717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 48), 'str', 'array header')
    # Processing the call keyword arguments (line 477)
    kwargs_106718 = {}
    # Getting the type of '_read_bytes' (line 477)
    _read_bytes_106714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 477)
    _read_bytes_call_result_106719 = invoke(stypy.reporting.localization.Localization(__file__, 477, 17), _read_bytes_106714, *[fp_106715, header_length_106716, str_106717], **kwargs_106718)
    
    # Assigning a type to the variable 'header' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'header', _read_bytes_call_result_106719)
    # SSA branch for the else part of an if statement (line 474)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 479)
    # Processing the call arguments (line 479)
    str_106721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 25), 'str', 'Invalid version %r')
    # Getting the type of 'version' (line 479)
    version_106722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 48), 'version', False)
    # Applying the binary operator '%' (line 479)
    result_mod_106723 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 25), '%', str_106721, version_106722)
    
    # Processing the call keyword arguments (line 479)
    kwargs_106724 = {}
    # Getting the type of 'ValueError' (line 479)
    ValueError_106720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 479)
    ValueError_call_result_106725 = invoke(stypy.reporting.localization.Localization(__file__, 479, 14), ValueError_106720, *[result_mod_106723], **kwargs_106724)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 479, 8), ValueError_call_result_106725, 'raise parameter', BaseException)
    # SSA join for if statement (line 474)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 470)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 487):
    
    # Assigning a Call to a Name (line 487):
    
    # Call to _filter_header(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 'header' (line 487)
    header_106727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 28), 'header', False)
    # Processing the call keyword arguments (line 487)
    kwargs_106728 = {}
    # Getting the type of '_filter_header' (line 487)
    _filter_header_106726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 13), '_filter_header', False)
    # Calling _filter_header(args, kwargs) (line 487)
    _filter_header_call_result_106729 = invoke(stypy.reporting.localization.Localization(__file__, 487, 13), _filter_header_106726, *[header_106727], **kwargs_106728)
    
    # Assigning a type to the variable 'header' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'header', _filter_header_call_result_106729)
    
    
    # SSA begins for try-except statement (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 489):
    
    # Assigning a Call to a Name (line 489):
    
    # Call to safe_eval(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'header' (line 489)
    header_106731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 22), 'header', False)
    # Processing the call keyword arguments (line 489)
    kwargs_106732 = {}
    # Getting the type of 'safe_eval' (line 489)
    safe_eval_106730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'safe_eval', False)
    # Calling safe_eval(args, kwargs) (line 489)
    safe_eval_call_result_106733 = invoke(stypy.reporting.localization.Localization(__file__, 489, 12), safe_eval_106730, *[header_106731], **kwargs_106732)
    
    # Assigning a type to the variable 'd' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'd', safe_eval_call_result_106733)
    # SSA branch for the except part of a try statement (line 488)
    # SSA branch for the except 'SyntaxError' branch of a try statement (line 488)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'SyntaxError' (line 490)
    SyntaxError_106734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'SyntaxError')
    # Assigning a type to the variable 'e' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'e', SyntaxError_106734)
    
    # Assigning a Str to a Name (line 491):
    
    # Assigning a Str to a Name (line 491):
    str_106735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 14), 'str', 'Cannot parse header: %r\nException: %r')
    # Assigning a type to the variable 'msg' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'msg', str_106735)
    
    # Call to ValueError(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'msg' (line 492)
    msg_106737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 25), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 492)
    tuple_106738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 492)
    # Adding element type (line 492)
    # Getting the type of 'header' (line 492)
    header_106739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 32), 'header', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 32), tuple_106738, header_106739)
    # Adding element type (line 492)
    # Getting the type of 'e' (line 492)
    e_106740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 40), 'e', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 32), tuple_106738, e_106740)
    
    # Applying the binary operator '%' (line 492)
    result_mod_106741 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 25), '%', msg_106737, tuple_106738)
    
    # Processing the call keyword arguments (line 492)
    kwargs_106742 = {}
    # Getting the type of 'ValueError' (line 492)
    ValueError_106736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 492)
    ValueError_call_result_106743 = invoke(stypy.reporting.localization.Localization(__file__, 492, 14), ValueError_106736, *[result_mod_106741], **kwargs_106742)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 492, 8), ValueError_call_result_106743, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 488)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 493)
    # Getting the type of 'dict' (line 493)
    dict_106744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 25), 'dict')
    # Getting the type of 'd' (line 493)
    d_106745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 22), 'd')
    
    (may_be_106746, more_types_in_union_106747) = may_not_be_subtype(dict_106744, d_106745)

    if may_be_106746:

        if more_types_in_union_106747:
            # Runtime conditional SSA (line 493)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'd' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'd', remove_subtype_from_union(d_106745, dict))
        
        # Assigning a Str to a Name (line 494):
        
        # Assigning a Str to a Name (line 494):
        str_106748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 14), 'str', 'Header is not a dictionary: %r')
        # Assigning a type to the variable 'msg' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'msg', str_106748)
        
        # Call to ValueError(...): (line 495)
        # Processing the call arguments (line 495)
        # Getting the type of 'msg' (line 495)
        msg_106750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 25), 'msg', False)
        # Getting the type of 'd' (line 495)
        d_106751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 31), 'd', False)
        # Applying the binary operator '%' (line 495)
        result_mod_106752 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 25), '%', msg_106750, d_106751)
        
        # Processing the call keyword arguments (line 495)
        kwargs_106753 = {}
        # Getting the type of 'ValueError' (line 495)
        ValueError_106749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 495)
        ValueError_call_result_106754 = invoke(stypy.reporting.localization.Localization(__file__, 495, 14), ValueError_106749, *[result_mod_106752], **kwargs_106753)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 495, 8), ValueError_call_result_106754, 'raise parameter', BaseException)

        if more_types_in_union_106747:
            # SSA join for if statement (line 493)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 496):
    
    # Assigning a Call to a Name (line 496):
    
    # Call to sorted(...): (line 496)
    # Processing the call arguments (line 496)
    
    # Call to keys(...): (line 496)
    # Processing the call keyword arguments (line 496)
    kwargs_106758 = {}
    # Getting the type of 'd' (line 496)
    d_106756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 18), 'd', False)
    # Obtaining the member 'keys' of a type (line 496)
    keys_106757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 18), d_106756, 'keys')
    # Calling keys(args, kwargs) (line 496)
    keys_call_result_106759 = invoke(stypy.reporting.localization.Localization(__file__, 496, 18), keys_106757, *[], **kwargs_106758)
    
    # Processing the call keyword arguments (line 496)
    kwargs_106760 = {}
    # Getting the type of 'sorted' (line 496)
    sorted_106755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 11), 'sorted', False)
    # Calling sorted(args, kwargs) (line 496)
    sorted_call_result_106761 = invoke(stypy.reporting.localization.Localization(__file__, 496, 11), sorted_106755, *[keys_call_result_106759], **kwargs_106760)
    
    # Assigning a type to the variable 'keys' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'keys', sorted_call_result_106761)
    
    
    # Getting the type of 'keys' (line 497)
    keys_106762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 7), 'keys')
    
    # Obtaining an instance of the builtin type 'list' (line 497)
    list_106763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 497)
    # Adding element type (line 497)
    str_106764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 16), 'str', 'descr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 15), list_106763, str_106764)
    # Adding element type (line 497)
    str_106765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 25), 'str', 'fortran_order')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 15), list_106763, str_106765)
    # Adding element type (line 497)
    str_106766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 42), 'str', 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 15), list_106763, str_106766)
    
    # Applying the binary operator '!=' (line 497)
    result_ne_106767 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 7), '!=', keys_106762, list_106763)
    
    # Testing the type of an if condition (line 497)
    if_condition_106768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 4), result_ne_106767)
    # Assigning a type to the variable 'if_condition_106768' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'if_condition_106768', if_condition_106768)
    # SSA begins for if statement (line 497)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 498):
    
    # Assigning a Str to a Name (line 498):
    str_106769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 14), 'str', 'Header does not contain the correct keys: %r')
    # Assigning a type to the variable 'msg' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'msg', str_106769)
    
    # Call to ValueError(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'msg' (line 499)
    msg_106771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 25), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 499)
    tuple_106772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 499)
    # Adding element type (line 499)
    # Getting the type of 'keys' (line 499)
    keys_106773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 32), 'keys', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 32), tuple_106772, keys_106773)
    
    # Applying the binary operator '%' (line 499)
    result_mod_106774 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 25), '%', msg_106771, tuple_106772)
    
    # Processing the call keyword arguments (line 499)
    kwargs_106775 = {}
    # Getting the type of 'ValueError' (line 499)
    ValueError_106770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 499)
    ValueError_call_result_106776 = invoke(stypy.reporting.localization.Localization(__file__, 499, 14), ValueError_106770, *[result_mod_106774], **kwargs_106775)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 499, 8), ValueError_call_result_106776, 'raise parameter', BaseException)
    # SSA join for if statement (line 497)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 502)
    # Processing the call arguments (line 502)
    
    # Obtaining the type of the subscript
    str_106778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 25), 'str', 'shape')
    # Getting the type of 'd' (line 502)
    d_106779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 23), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___106780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 23), d_106779, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_106781 = invoke(stypy.reporting.localization.Localization(__file__, 502, 23), getitem___106780, str_106778)
    
    # Getting the type of 'tuple' (line 502)
    tuple_106782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 35), 'tuple', False)
    # Processing the call keyword arguments (line 502)
    kwargs_106783 = {}
    # Getting the type of 'isinstance' (line 502)
    isinstance_106777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 502)
    isinstance_call_result_106784 = invoke(stypy.reporting.localization.Localization(__file__, 502, 12), isinstance_106777, *[subscript_call_result_106781, tuple_106782], **kwargs_106783)
    
    # Applying the 'not' unary operator (line 502)
    result_not__106785 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 8), 'not', isinstance_call_result_106784)
    
    
    
    # Call to all(...): (line 503)
    # Processing the call arguments (line 503)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    str_106795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 65), 'str', 'shape')
    # Getting the type of 'd' (line 503)
    d_106796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 63), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___106797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 63), d_106796, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_106798 = invoke(stypy.reporting.localization.Localization(__file__, 503, 63), getitem___106797, str_106795)
    
    comprehension_106799 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 27), subscript_call_result_106798)
    # Assigning a type to the variable 'x' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 27), 'x', comprehension_106799)
    
    # Call to isinstance(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'x' (line 503)
    x_106789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 38), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 503)
    tuple_106790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 503)
    # Adding element type (line 503)
    # Getting the type of 'int' (line 503)
    int_106791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 42), 'int', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 42), tuple_106790, int_106791)
    # Adding element type (line 503)
    # Getting the type of 'long' (line 503)
    long_106792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 47), 'long', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 42), tuple_106790, long_106792)
    
    # Processing the call keyword arguments (line 503)
    kwargs_106793 = {}
    # Getting the type of 'isinstance' (line 503)
    isinstance_106788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 27), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 503)
    isinstance_call_result_106794 = invoke(stypy.reporting.localization.Localization(__file__, 503, 27), isinstance_106788, *[x_106789, tuple_106790], **kwargs_106793)
    
    list_106800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 27), list_106800, isinstance_call_result_106794)
    # Processing the call keyword arguments (line 503)
    kwargs_106801 = {}
    # Getting the type of 'numpy' (line 503)
    numpy_106786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'numpy', False)
    # Obtaining the member 'all' of a type (line 503)
    all_106787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 16), numpy_106786, 'all')
    # Calling all(args, kwargs) (line 503)
    all_call_result_106802 = invoke(stypy.reporting.localization.Localization(__file__, 503, 16), all_106787, *[list_106800], **kwargs_106801)
    
    # Applying the 'not' unary operator (line 503)
    result_not__106803 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 12), 'not', all_call_result_106802)
    
    # Applying the binary operator 'or' (line 502)
    result_or_keyword_106804 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 8), 'or', result_not__106785, result_not__106803)
    
    # Testing the type of an if condition (line 502)
    if_condition_106805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 4), result_or_keyword_106804)
    # Assigning a type to the variable 'if_condition_106805' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'if_condition_106805', if_condition_106805)
    # SSA begins for if statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 504):
    
    # Assigning a Str to a Name (line 504):
    str_106806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 14), 'str', 'shape is not valid: %r')
    # Assigning a type to the variable 'msg' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'msg', str_106806)
    
    # Call to ValueError(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'msg' (line 505)
    msg_106808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 25), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 505)
    tuple_106809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 505)
    # Adding element type (line 505)
    
    # Obtaining the type of the subscript
    str_106810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 34), 'str', 'shape')
    # Getting the type of 'd' (line 505)
    d_106811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___106812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 32), d_106811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 505)
    subscript_call_result_106813 = invoke(stypy.reporting.localization.Localization(__file__, 505, 32), getitem___106812, str_106810)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 32), tuple_106809, subscript_call_result_106813)
    
    # Applying the binary operator '%' (line 505)
    result_mod_106814 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 25), '%', msg_106808, tuple_106809)
    
    # Processing the call keyword arguments (line 505)
    kwargs_106815 = {}
    # Getting the type of 'ValueError' (line 505)
    ValueError_106807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 505)
    ValueError_call_result_106816 = invoke(stypy.reporting.localization.Localization(__file__, 505, 14), ValueError_106807, *[result_mod_106814], **kwargs_106815)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 505, 8), ValueError_call_result_106816, 'raise parameter', BaseException)
    # SSA join for if statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 506)
    # Getting the type of 'bool' (line 506)
    bool_106817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 42), 'bool')
    
    # Obtaining the type of the subscript
    str_106818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 24), 'str', 'fortran_order')
    # Getting the type of 'd' (line 506)
    d_106819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 22), 'd')
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___106820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 22), d_106819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_106821 = invoke(stypy.reporting.localization.Localization(__file__, 506, 22), getitem___106820, str_106818)
    
    
    (may_be_106822, more_types_in_union_106823) = may_not_be_subtype(bool_106817, subscript_call_result_106821)

    if may_be_106822:

        if more_types_in_union_106823:
            # Runtime conditional SSA (line 506)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 507):
        
        # Assigning a Str to a Name (line 507):
        str_106824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 14), 'str', 'fortran_order is not a valid bool: %r')
        # Assigning a type to the variable 'msg' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'msg', str_106824)
        
        # Call to ValueError(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'msg' (line 508)
        msg_106826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 25), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 508)
        tuple_106827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 508)
        # Adding element type (line 508)
        
        # Obtaining the type of the subscript
        str_106828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 34), 'str', 'fortran_order')
        # Getting the type of 'd' (line 508)
        d_106829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 32), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 508)
        getitem___106830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 32), d_106829, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 508)
        subscript_call_result_106831 = invoke(stypy.reporting.localization.Localization(__file__, 508, 32), getitem___106830, str_106828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 32), tuple_106827, subscript_call_result_106831)
        
        # Applying the binary operator '%' (line 508)
        result_mod_106832 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 25), '%', msg_106826, tuple_106827)
        
        # Processing the call keyword arguments (line 508)
        kwargs_106833 = {}
        # Getting the type of 'ValueError' (line 508)
        ValueError_106825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 508)
        ValueError_call_result_106834 = invoke(stypy.reporting.localization.Localization(__file__, 508, 14), ValueError_106825, *[result_mod_106832], **kwargs_106833)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 508, 8), ValueError_call_result_106834, 'raise parameter', BaseException)

        if more_types_in_union_106823:
            # SSA join for if statement (line 506)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 510):
    
    # Assigning a Call to a Name (line 510):
    
    # Call to dtype(...): (line 510)
    # Processing the call arguments (line 510)
    
    # Obtaining the type of the subscript
    str_106837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 30), 'str', 'descr')
    # Getting the type of 'd' (line 510)
    d_106838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 28), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 510)
    getitem___106839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 28), d_106838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 510)
    subscript_call_result_106840 = invoke(stypy.reporting.localization.Localization(__file__, 510, 28), getitem___106839, str_106837)
    
    # Processing the call keyword arguments (line 510)
    kwargs_106841 = {}
    # Getting the type of 'numpy' (line 510)
    numpy_106835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'numpy', False)
    # Obtaining the member 'dtype' of a type (line 510)
    dtype_106836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), numpy_106835, 'dtype')
    # Calling dtype(args, kwargs) (line 510)
    dtype_call_result_106842 = invoke(stypy.reporting.localization.Localization(__file__, 510, 16), dtype_106836, *[subscript_call_result_106840], **kwargs_106841)
    
    # Assigning a type to the variable 'dtype' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'dtype', dtype_call_result_106842)
    # SSA branch for the except part of a try statement (line 509)
    # SSA branch for the except 'TypeError' branch of a try statement (line 509)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'TypeError' (line 511)
    TypeError_106843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 11), 'TypeError')
    # Assigning a type to the variable 'e' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'e', TypeError_106843)
    
    # Assigning a Str to a Name (line 512):
    
    # Assigning a Str to a Name (line 512):
    str_106844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 14), 'str', 'descr is not a valid dtype descriptor: %r')
    # Assigning a type to the variable 'msg' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'msg', str_106844)
    
    # Call to ValueError(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'msg' (line 513)
    msg_106846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 25), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 513)
    tuple_106847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 513)
    # Adding element type (line 513)
    
    # Obtaining the type of the subscript
    str_106848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 34), 'str', 'descr')
    # Getting the type of 'd' (line 513)
    d_106849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 32), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 513)
    getitem___106850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 32), d_106849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 513)
    subscript_call_result_106851 = invoke(stypy.reporting.localization.Localization(__file__, 513, 32), getitem___106850, str_106848)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 32), tuple_106847, subscript_call_result_106851)
    
    # Applying the binary operator '%' (line 513)
    result_mod_106852 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 25), '%', msg_106846, tuple_106847)
    
    # Processing the call keyword arguments (line 513)
    kwargs_106853 = {}
    # Getting the type of 'ValueError' (line 513)
    ValueError_106845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 513)
    ValueError_call_result_106854 = invoke(stypy.reporting.localization.Localization(__file__, 513, 14), ValueError_106845, *[result_mod_106852], **kwargs_106853)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 513, 8), ValueError_call_result_106854, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 509)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_106855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    
    # Obtaining the type of the subscript
    str_106856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 13), 'str', 'shape')
    # Getting the type of 'd' (line 515)
    d_106857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 11), 'd')
    # Obtaining the member '__getitem__' of a type (line 515)
    getitem___106858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 11), d_106857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 515)
    subscript_call_result_106859 = invoke(stypy.reporting.localization.Localization(__file__, 515, 11), getitem___106858, str_106856)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 11), tuple_106855, subscript_call_result_106859)
    # Adding element type (line 515)
    
    # Obtaining the type of the subscript
    str_106860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 25), 'str', 'fortran_order')
    # Getting the type of 'd' (line 515)
    d_106861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 23), 'd')
    # Obtaining the member '__getitem__' of a type (line 515)
    getitem___106862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 23), d_106861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 515)
    subscript_call_result_106863 = invoke(stypy.reporting.localization.Localization(__file__, 515, 23), getitem___106862, str_106860)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 11), tuple_106855, subscript_call_result_106863)
    # Adding element type (line 515)
    # Getting the type of 'dtype' (line 515)
    dtype_106864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 43), 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 11), tuple_106855, dtype_106864)
    
    # Assigning a type to the variable 'stypy_return_type' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type', tuple_106855)
    
    # ################# End of '_read_array_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_array_header' in the type store
    # Getting the type of 'stypy_return_type' (line 463)
    stypy_return_type_106865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_106865)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_array_header'
    return stypy_return_type_106865

# Assigning a type to the variable '_read_array_header' (line 463)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 0), '_read_array_header', _read_array_header)

@norecursion
def write_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 517)
    None_106866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 35), 'None')
    # Getting the type of 'True' (line 517)
    True_106867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 54), 'True')
    # Getting the type of 'None' (line 517)
    None_106868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 74), 'None')
    defaults = [None_106866, True_106867, None_106868]
    # Create a new context for function 'write_array'
    module_type_store = module_type_store.open_function_context('write_array', 517, 0, False)
    
    # Passed parameters checking function
    write_array.stypy_localization = localization
    write_array.stypy_type_of_self = None
    write_array.stypy_type_store = module_type_store
    write_array.stypy_function_name = 'write_array'
    write_array.stypy_param_names_list = ['fp', 'array', 'version', 'allow_pickle', 'pickle_kwargs']
    write_array.stypy_varargs_param_name = None
    write_array.stypy_kwargs_param_name = None
    write_array.stypy_call_defaults = defaults
    write_array.stypy_call_varargs = varargs
    write_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'write_array', ['fp', 'array', 'version', 'allow_pickle', 'pickle_kwargs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'write_array', localization, ['fp', 'array', 'version', 'allow_pickle', 'pickle_kwargs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'write_array(...)' code ##################

    str_106869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, (-1)), 'str', "\n    Write an array to an NPY file, including a header.\n\n    If the array is neither C-contiguous nor Fortran-contiguous AND the\n    file_like object is not a real file object, this function will have to\n    copy data in memory.\n\n    Parameters\n    ----------\n    fp : file_like object\n        An open, writable file object, or similar object with a\n        ``.write()`` method.\n    array : ndarray\n        The array to write to disk.\n    version : (int, int) or None, optional\n        The version number of the format. None means use the oldest\n        supported version that is able to store the data.  Default: None\n    allow_pickle : bool, optional\n        Whether to allow writing pickled data. Default: True\n    pickle_kwargs : dict, optional\n        Additional keyword arguments to pass to pickle.dump, excluding\n        'protocol'. These are only useful when pickling objects in object\n        arrays on Python 3 to Python 2 compatible format.\n\n    Raises\n    ------\n    ValueError\n        If the array cannot be persisted. This includes the case of\n        allow_pickle=False and array being an object array.\n    Various other errors\n        If the array contains Python objects as part of its dtype, the\n        process of pickling them may raise various errors if the objects\n        are not picklable.\n\n    ")
    
    # Call to _check_version(...): (line 553)
    # Processing the call arguments (line 553)
    # Getting the type of 'version' (line 553)
    version_106871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 19), 'version', False)
    # Processing the call keyword arguments (line 553)
    kwargs_106872 = {}
    # Getting the type of '_check_version' (line 553)
    _check_version_106870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), '_check_version', False)
    # Calling _check_version(args, kwargs) (line 553)
    _check_version_call_result_106873 = invoke(stypy.reporting.localization.Localization(__file__, 553, 4), _check_version_106870, *[version_106871], **kwargs_106872)
    
    
    # Assigning a Call to a Name (line 554):
    
    # Assigning a Call to a Name (line 554):
    
    # Call to _write_array_header(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'fp' (line 554)
    fp_106875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 35), 'fp', False)
    
    # Call to header_data_from_array_1_0(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'array' (line 554)
    array_106877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 66), 'array', False)
    # Processing the call keyword arguments (line 554)
    kwargs_106878 = {}
    # Getting the type of 'header_data_from_array_1_0' (line 554)
    header_data_from_array_1_0_106876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 39), 'header_data_from_array_1_0', False)
    # Calling header_data_from_array_1_0(args, kwargs) (line 554)
    header_data_from_array_1_0_call_result_106879 = invoke(stypy.reporting.localization.Localization(__file__, 554, 39), header_data_from_array_1_0_106876, *[array_106877], **kwargs_106878)
    
    # Getting the type of 'version' (line 555)
    version_106880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 35), 'version', False)
    # Processing the call keyword arguments (line 554)
    kwargs_106881 = {}
    # Getting the type of '_write_array_header' (line 554)
    _write_array_header_106874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), '_write_array_header', False)
    # Calling _write_array_header(args, kwargs) (line 554)
    _write_array_header_call_result_106882 = invoke(stypy.reporting.localization.Localization(__file__, 554, 15), _write_array_header_106874, *[fp_106875, header_data_from_array_1_0_call_result_106879, version_106880], **kwargs_106881)
    
    # Assigning a type to the variable 'used_ver' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'used_ver', _write_array_header_call_result_106882)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'version' (line 557)
    version_106883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 7), 'version')
    
    # Obtaining an instance of the builtin type 'tuple' (line 557)
    tuple_106884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 557)
    # Adding element type (line 557)
    int_106885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 19), tuple_106884, int_106885)
    # Adding element type (line 557)
    int_106886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 19), tuple_106884, int_106886)
    
    # Applying the binary operator '!=' (line 557)
    result_ne_106887 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 7), '!=', version_106883, tuple_106884)
    
    
    # Getting the type of 'used_ver' (line 557)
    used_ver_106888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 29), 'used_ver')
    
    # Obtaining an instance of the builtin type 'tuple' (line 557)
    tuple_106889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 557)
    # Adding element type (line 557)
    int_106890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 42), tuple_106889, int_106890)
    # Adding element type (line 557)
    int_106891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 42), tuple_106889, int_106891)
    
    # Applying the binary operator '==' (line 557)
    result_eq_106892 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 29), '==', used_ver_106888, tuple_106889)
    
    # Applying the binary operator 'and' (line 557)
    result_and_keyword_106893 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 7), 'and', result_ne_106887, result_eq_106892)
    
    # Testing the type of an if condition (line 557)
    if_condition_106894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 4), result_and_keyword_106893)
    # Assigning a type to the variable 'if_condition_106894' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'if_condition_106894', if_condition_106894)
    # SSA begins for if statement (line 557)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 558)
    # Processing the call arguments (line 558)
    str_106897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 22), 'str', 'Stored array in format 2.0. It can only beread by NumPy >= 1.9')
    # Getting the type of 'UserWarning' (line 559)
    UserWarning_106898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 46), 'UserWarning', False)
    # Processing the call keyword arguments (line 558)
    kwargs_106899 = {}
    # Getting the type of 'warnings' (line 558)
    warnings_106895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 558)
    warn_106896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), warnings_106895, 'warn')
    # Calling warn(args, kwargs) (line 558)
    warn_call_result_106900 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), warn_106896, *[str_106897, UserWarning_106898], **kwargs_106899)
    
    # SSA join for if statement (line 557)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to max(...): (line 562)
    # Processing the call arguments (line 562)
    int_106902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 21), 'int')
    int_106903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 26), 'int')
    int_106904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 34), 'int')
    # Applying the binary operator '**' (line 562)
    result_pow_106905 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 26), '**', int_106903, int_106904)
    
    # Applying the binary operator '*' (line 562)
    result_mul_106906 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 21), '*', int_106902, result_pow_106905)
    
    # Getting the type of 'array' (line 562)
    array_106907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 39), 'array', False)
    # Obtaining the member 'itemsize' of a type (line 562)
    itemsize_106908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 39), array_106907, 'itemsize')
    # Applying the binary operator '//' (line 562)
    result_floordiv_106909 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 36), '//', result_mul_106906, itemsize_106908)
    
    int_106910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 55), 'int')
    # Processing the call keyword arguments (line 562)
    kwargs_106911 = {}
    # Getting the type of 'max' (line 562)
    max_106901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 17), 'max', False)
    # Calling max(args, kwargs) (line 562)
    max_call_result_106912 = invoke(stypy.reporting.localization.Localization(__file__, 562, 17), max_106901, *[result_floordiv_106909, int_106910], **kwargs_106911)
    
    # Assigning a type to the variable 'buffersize' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'buffersize', max_call_result_106912)
    
    # Getting the type of 'array' (line 564)
    array_106913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 7), 'array')
    # Obtaining the member 'dtype' of a type (line 564)
    dtype_106914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 7), array_106913, 'dtype')
    # Obtaining the member 'hasobject' of a type (line 564)
    hasobject_106915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 7), dtype_106914, 'hasobject')
    # Testing the type of an if condition (line 564)
    if_condition_106916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 564, 4), hasobject_106915)
    # Assigning a type to the variable 'if_condition_106916' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'if_condition_106916', if_condition_106916)
    # SSA begins for if statement (line 564)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'allow_pickle' (line 568)
    allow_pickle_106917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'allow_pickle')
    # Applying the 'not' unary operator (line 568)
    result_not__106918 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 11), 'not', allow_pickle_106917)
    
    # Testing the type of an if condition (line 568)
    if_condition_106919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), result_not__106918)
    # Assigning a type to the variable 'if_condition_106919' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_106919', if_condition_106919)
    # SSA begins for if statement (line 568)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 569)
    # Processing the call arguments (line 569)
    str_106921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 29), 'str', 'Object arrays cannot be saved when allow_pickle=False')
    # Processing the call keyword arguments (line 569)
    kwargs_106922 = {}
    # Getting the type of 'ValueError' (line 569)
    ValueError_106920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 569)
    ValueError_call_result_106923 = invoke(stypy.reporting.localization.Localization(__file__, 569, 18), ValueError_106920, *[str_106921], **kwargs_106922)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 569, 12), ValueError_call_result_106923, 'raise parameter', BaseException)
    # SSA join for if statement (line 568)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 571)
    # Getting the type of 'pickle_kwargs' (line 571)
    pickle_kwargs_106924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 11), 'pickle_kwargs')
    # Getting the type of 'None' (line 571)
    None_106925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 28), 'None')
    
    (may_be_106926, more_types_in_union_106927) = may_be_none(pickle_kwargs_106924, None_106925)

    if may_be_106926:

        if more_types_in_union_106927:
            # Runtime conditional SSA (line 571)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 572):
        
        # Assigning a Dict to a Name (line 572):
        
        # Obtaining an instance of the builtin type 'dict' (line 572)
        dict_106928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 572)
        
        # Assigning a type to the variable 'pickle_kwargs' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'pickle_kwargs', dict_106928)

        if more_types_in_union_106927:
            # SSA join for if statement (line 571)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to dump(...): (line 573)
    # Processing the call arguments (line 573)
    # Getting the type of 'array' (line 573)
    array_106931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'array', False)
    # Getting the type of 'fp' (line 573)
    fp_106932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'fp', False)
    # Processing the call keyword arguments (line 573)
    int_106933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 40), 'int')
    keyword_106934 = int_106933
    # Getting the type of 'pickle_kwargs' (line 573)
    pickle_kwargs_106935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 45), 'pickle_kwargs', False)
    kwargs_106936 = {'protocol': keyword_106934, 'pickle_kwargs_106935': pickle_kwargs_106935}
    # Getting the type of 'pickle' (line 573)
    pickle_106929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'pickle', False)
    # Obtaining the member 'dump' of a type (line 573)
    dump_106930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), pickle_106929, 'dump')
    # Calling dump(args, kwargs) (line 573)
    dump_call_result_106937 = invoke(stypy.reporting.localization.Localization(__file__, 573, 8), dump_106930, *[array_106931, fp_106932], **kwargs_106936)
    
    # SSA branch for the else part of an if statement (line 564)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'array' (line 574)
    array_106938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 9), 'array')
    # Obtaining the member 'flags' of a type (line 574)
    flags_106939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 9), array_106938, 'flags')
    # Obtaining the member 'f_contiguous' of a type (line 574)
    f_contiguous_106940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 9), flags_106939, 'f_contiguous')
    
    # Getting the type of 'array' (line 574)
    array_106941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'array')
    # Obtaining the member 'flags' of a type (line 574)
    flags_106942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 42), array_106941, 'flags')
    # Obtaining the member 'c_contiguous' of a type (line 574)
    c_contiguous_106943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 42), flags_106942, 'c_contiguous')
    # Applying the 'not' unary operator (line 574)
    result_not__106944 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 38), 'not', c_contiguous_106943)
    
    # Applying the binary operator 'and' (line 574)
    result_and_keyword_106945 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 9), 'and', f_contiguous_106940, result_not__106944)
    
    # Testing the type of an if condition (line 574)
    if_condition_106946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 9), result_and_keyword_106945)
    # Assigning a type to the variable 'if_condition_106946' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 9), 'if_condition_106946', if_condition_106946)
    # SSA begins for if statement (line 574)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isfileobj(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'fp' (line 575)
    fp_106948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'fp', False)
    # Processing the call keyword arguments (line 575)
    kwargs_106949 = {}
    # Getting the type of 'isfileobj' (line 575)
    isfileobj_106947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'isfileobj', False)
    # Calling isfileobj(args, kwargs) (line 575)
    isfileobj_call_result_106950 = invoke(stypy.reporting.localization.Localization(__file__, 575, 11), isfileobj_106947, *[fp_106948], **kwargs_106949)
    
    # Testing the type of an if condition (line 575)
    if_condition_106951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 8), isfileobj_call_result_106950)
    # Assigning a type to the variable 'if_condition_106951' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'if_condition_106951', if_condition_106951)
    # SSA begins for if statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tofile(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'fp' (line 576)
    fp_106955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 27), 'fp', False)
    # Processing the call keyword arguments (line 576)
    kwargs_106956 = {}
    # Getting the type of 'array' (line 576)
    array_106952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'array', False)
    # Obtaining the member 'T' of a type (line 576)
    T_106953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 12), array_106952, 'T')
    # Obtaining the member 'tofile' of a type (line 576)
    tofile_106954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 12), T_106953, 'tofile')
    # Calling tofile(args, kwargs) (line 576)
    tofile_call_result_106957 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), tofile_106954, *[fp_106955], **kwargs_106956)
    
    # SSA branch for the else part of an if statement (line 575)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to nditer(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'array' (line 579)
    array_106960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'array', False)
    # Processing the call keyword arguments (line 578)
    
    # Obtaining an instance of the builtin type 'list' (line 579)
    list_106961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 579)
    # Adding element type (line 579)
    str_106962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 34), 'str', 'external_loop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 33), list_106961, str_106962)
    # Adding element type (line 579)
    str_106963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 51), 'str', 'buffered')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 33), list_106961, str_106963)
    # Adding element type (line 579)
    str_106964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 63), 'str', 'zerosize_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 33), list_106961, str_106964)
    
    keyword_106965 = list_106961
    # Getting the type of 'buffersize' (line 580)
    buffersize_106966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 31), 'buffersize', False)
    keyword_106967 = buffersize_106966
    str_106968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 49), 'str', 'F')
    keyword_106969 = str_106968
    kwargs_106970 = {'buffersize': keyword_106967, 'flags': keyword_106965, 'order': keyword_106969}
    # Getting the type of 'numpy' (line 578)
    numpy_106958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 25), 'numpy', False)
    # Obtaining the member 'nditer' of a type (line 578)
    nditer_106959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 25), numpy_106958, 'nditer')
    # Calling nditer(args, kwargs) (line 578)
    nditer_call_result_106971 = invoke(stypy.reporting.localization.Localization(__file__, 578, 25), nditer_106959, *[array_106960], **kwargs_106970)
    
    # Testing the type of a for loop iterable (line 578)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 578, 12), nditer_call_result_106971)
    # Getting the type of the for loop variable (line 578)
    for_loop_var_106972 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 578, 12), nditer_call_result_106971)
    # Assigning a type to the variable 'chunk' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'chunk', for_loop_var_106972)
    # SSA begins for a for statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to write(...): (line 581)
    # Processing the call arguments (line 581)
    
    # Call to tobytes(...): (line 581)
    # Processing the call arguments (line 581)
    str_106977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 39), 'str', 'C')
    # Processing the call keyword arguments (line 581)
    kwargs_106978 = {}
    # Getting the type of 'chunk' (line 581)
    chunk_106975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'chunk', False)
    # Obtaining the member 'tobytes' of a type (line 581)
    tobytes_106976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 25), chunk_106975, 'tobytes')
    # Calling tobytes(args, kwargs) (line 581)
    tobytes_call_result_106979 = invoke(stypy.reporting.localization.Localization(__file__, 581, 25), tobytes_106976, *[str_106977], **kwargs_106978)
    
    # Processing the call keyword arguments (line 581)
    kwargs_106980 = {}
    # Getting the type of 'fp' (line 581)
    fp_106973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'fp', False)
    # Obtaining the member 'write' of a type (line 581)
    write_106974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 16), fp_106973, 'write')
    # Calling write(args, kwargs) (line 581)
    write_call_result_106981 = invoke(stypy.reporting.localization.Localization(__file__, 581, 16), write_106974, *[tobytes_call_result_106979], **kwargs_106980)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 575)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 574)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isfileobj(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'fp' (line 583)
    fp_106983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 21), 'fp', False)
    # Processing the call keyword arguments (line 583)
    kwargs_106984 = {}
    # Getting the type of 'isfileobj' (line 583)
    isfileobj_106982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 11), 'isfileobj', False)
    # Calling isfileobj(args, kwargs) (line 583)
    isfileobj_call_result_106985 = invoke(stypy.reporting.localization.Localization(__file__, 583, 11), isfileobj_106982, *[fp_106983], **kwargs_106984)
    
    # Testing the type of an if condition (line 583)
    if_condition_106986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 8), isfileobj_call_result_106985)
    # Assigning a type to the variable 'if_condition_106986' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'if_condition_106986', if_condition_106986)
    # SSA begins for if statement (line 583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tofile(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'fp' (line 584)
    fp_106989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 25), 'fp', False)
    # Processing the call keyword arguments (line 584)
    kwargs_106990 = {}
    # Getting the type of 'array' (line 584)
    array_106987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'array', False)
    # Obtaining the member 'tofile' of a type (line 584)
    tofile_106988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 12), array_106987, 'tofile')
    # Calling tofile(args, kwargs) (line 584)
    tofile_call_result_106991 = invoke(stypy.reporting.localization.Localization(__file__, 584, 12), tofile_106988, *[fp_106989], **kwargs_106990)
    
    # SSA branch for the else part of an if statement (line 583)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to nditer(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'array' (line 587)
    array_106994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'array', False)
    # Processing the call keyword arguments (line 586)
    
    # Obtaining an instance of the builtin type 'list' (line 587)
    list_106995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 587)
    # Adding element type (line 587)
    str_106996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 34), 'str', 'external_loop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 33), list_106995, str_106996)
    # Adding element type (line 587)
    str_106997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 51), 'str', 'buffered')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 33), list_106995, str_106997)
    # Adding element type (line 587)
    str_106998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 63), 'str', 'zerosize_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 33), list_106995, str_106998)
    
    keyword_106999 = list_106995
    # Getting the type of 'buffersize' (line 588)
    buffersize_107000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 31), 'buffersize', False)
    keyword_107001 = buffersize_107000
    str_107002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 49), 'str', 'C')
    keyword_107003 = str_107002
    kwargs_107004 = {'buffersize': keyword_107001, 'flags': keyword_106999, 'order': keyword_107003}
    # Getting the type of 'numpy' (line 586)
    numpy_106992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 25), 'numpy', False)
    # Obtaining the member 'nditer' of a type (line 586)
    nditer_106993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 25), numpy_106992, 'nditer')
    # Calling nditer(args, kwargs) (line 586)
    nditer_call_result_107005 = invoke(stypy.reporting.localization.Localization(__file__, 586, 25), nditer_106993, *[array_106994], **kwargs_107004)
    
    # Testing the type of a for loop iterable (line 586)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 586, 12), nditer_call_result_107005)
    # Getting the type of the for loop variable (line 586)
    for_loop_var_107006 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 586, 12), nditer_call_result_107005)
    # Assigning a type to the variable 'chunk' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'chunk', for_loop_var_107006)
    # SSA begins for a for statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to write(...): (line 589)
    # Processing the call arguments (line 589)
    
    # Call to tobytes(...): (line 589)
    # Processing the call arguments (line 589)
    str_107011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 39), 'str', 'C')
    # Processing the call keyword arguments (line 589)
    kwargs_107012 = {}
    # Getting the type of 'chunk' (line 589)
    chunk_107009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 25), 'chunk', False)
    # Obtaining the member 'tobytes' of a type (line 589)
    tobytes_107010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 25), chunk_107009, 'tobytes')
    # Calling tobytes(args, kwargs) (line 589)
    tobytes_call_result_107013 = invoke(stypy.reporting.localization.Localization(__file__, 589, 25), tobytes_107010, *[str_107011], **kwargs_107012)
    
    # Processing the call keyword arguments (line 589)
    kwargs_107014 = {}
    # Getting the type of 'fp' (line 589)
    fp_107007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'fp', False)
    # Obtaining the member 'write' of a type (line 589)
    write_107008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), fp_107007, 'write')
    # Calling write(args, kwargs) (line 589)
    write_call_result_107015 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), write_107008, *[tobytes_call_result_107013], **kwargs_107014)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 583)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 574)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 564)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'write_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'write_array' in the type store
    # Getting the type of 'stypy_return_type' (line 517)
    stypy_return_type_107016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_107016)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'write_array'
    return stypy_return_type_107016

# Assigning a type to the variable 'write_array' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'write_array', write_array)

@norecursion
def read_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 592)
    True_107017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 32), 'True')
    # Getting the type of 'None' (line 592)
    None_107018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 52), 'None')
    defaults = [True_107017, None_107018]
    # Create a new context for function 'read_array'
    module_type_store = module_type_store.open_function_context('read_array', 592, 0, False)
    
    # Passed parameters checking function
    read_array.stypy_localization = localization
    read_array.stypy_type_of_self = None
    read_array.stypy_type_store = module_type_store
    read_array.stypy_function_name = 'read_array'
    read_array.stypy_param_names_list = ['fp', 'allow_pickle', 'pickle_kwargs']
    read_array.stypy_varargs_param_name = None
    read_array.stypy_kwargs_param_name = None
    read_array.stypy_call_defaults = defaults
    read_array.stypy_call_varargs = varargs
    read_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_array', ['fp', 'allow_pickle', 'pickle_kwargs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_array', localization, ['fp', 'allow_pickle', 'pickle_kwargs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_array(...)' code ##################

    str_107019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, (-1)), 'str', '\n    Read an array from an NPY file.\n\n    Parameters\n    ----------\n    fp : file_like object\n        If this is not a real file object, then this may take extra memory\n        and time.\n    allow_pickle : bool, optional\n        Whether to allow reading pickled data. Default: True\n    pickle_kwargs : dict\n        Additional keyword arguments to pass to pickle.load. These are only\n        useful when loading object arrays saved on Python 2 when using\n        Python 3.\n\n    Returns\n    -------\n    array : ndarray\n        The array from the data on disk.\n\n    Raises\n    ------\n    ValueError\n        If the data is invalid, or allow_pickle=False and the file contains\n        an object array.\n\n    ')
    
    # Assigning a Call to a Name (line 620):
    
    # Assigning a Call to a Name (line 620):
    
    # Call to read_magic(...): (line 620)
    # Processing the call arguments (line 620)
    # Getting the type of 'fp' (line 620)
    fp_107021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 25), 'fp', False)
    # Processing the call keyword arguments (line 620)
    kwargs_107022 = {}
    # Getting the type of 'read_magic' (line 620)
    read_magic_107020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 14), 'read_magic', False)
    # Calling read_magic(args, kwargs) (line 620)
    read_magic_call_result_107023 = invoke(stypy.reporting.localization.Localization(__file__, 620, 14), read_magic_107020, *[fp_107021], **kwargs_107022)
    
    # Assigning a type to the variable 'version' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'version', read_magic_call_result_107023)
    
    # Call to _check_version(...): (line 621)
    # Processing the call arguments (line 621)
    # Getting the type of 'version' (line 621)
    version_107025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'version', False)
    # Processing the call keyword arguments (line 621)
    kwargs_107026 = {}
    # Getting the type of '_check_version' (line 621)
    _check_version_107024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), '_check_version', False)
    # Calling _check_version(args, kwargs) (line 621)
    _check_version_call_result_107027 = invoke(stypy.reporting.localization.Localization(__file__, 621, 4), _check_version_107024, *[version_107025], **kwargs_107026)
    
    
    # Assigning a Call to a Tuple (line 622):
    
    # Assigning a Call to a Name:
    
    # Call to _read_array_header(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'fp' (line 622)
    fp_107029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 53), 'fp', False)
    # Getting the type of 'version' (line 622)
    version_107030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 57), 'version', False)
    # Processing the call keyword arguments (line 622)
    kwargs_107031 = {}
    # Getting the type of '_read_array_header' (line 622)
    _read_array_header_107028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 34), '_read_array_header', False)
    # Calling _read_array_header(args, kwargs) (line 622)
    _read_array_header_call_result_107032 = invoke(stypy.reporting.localization.Localization(__file__, 622, 34), _read_array_header_107028, *[fp_107029, version_107030], **kwargs_107031)
    
    # Assigning a type to the variable 'call_assignment_106190' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106190', _read_array_header_call_result_107032)
    
    # Assigning a Call to a Name (line 622):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_107035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 4), 'int')
    # Processing the call keyword arguments
    kwargs_107036 = {}
    # Getting the type of 'call_assignment_106190' (line 622)
    call_assignment_106190_107033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106190', False)
    # Obtaining the member '__getitem__' of a type (line 622)
    getitem___107034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 4), call_assignment_106190_107033, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_107037 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___107034, *[int_107035], **kwargs_107036)
    
    # Assigning a type to the variable 'call_assignment_106191' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106191', getitem___call_result_107037)
    
    # Assigning a Name to a Name (line 622):
    # Getting the type of 'call_assignment_106191' (line 622)
    call_assignment_106191_107038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106191')
    # Assigning a type to the variable 'shape' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'shape', call_assignment_106191_107038)
    
    # Assigning a Call to a Name (line 622):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_107041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 4), 'int')
    # Processing the call keyword arguments
    kwargs_107042 = {}
    # Getting the type of 'call_assignment_106190' (line 622)
    call_assignment_106190_107039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106190', False)
    # Obtaining the member '__getitem__' of a type (line 622)
    getitem___107040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 4), call_assignment_106190_107039, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_107043 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___107040, *[int_107041], **kwargs_107042)
    
    # Assigning a type to the variable 'call_assignment_106192' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106192', getitem___call_result_107043)
    
    # Assigning a Name to a Name (line 622):
    # Getting the type of 'call_assignment_106192' (line 622)
    call_assignment_106192_107044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106192')
    # Assigning a type to the variable 'fortran_order' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 11), 'fortran_order', call_assignment_106192_107044)
    
    # Assigning a Call to a Name (line 622):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_107047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 4), 'int')
    # Processing the call keyword arguments
    kwargs_107048 = {}
    # Getting the type of 'call_assignment_106190' (line 622)
    call_assignment_106190_107045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106190', False)
    # Obtaining the member '__getitem__' of a type (line 622)
    getitem___107046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 4), call_assignment_106190_107045, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_107049 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___107046, *[int_107047], **kwargs_107048)
    
    # Assigning a type to the variable 'call_assignment_106193' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106193', getitem___call_result_107049)
    
    # Assigning a Name to a Name (line 622):
    # Getting the type of 'call_assignment_106193' (line 622)
    call_assignment_106193_107050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'call_assignment_106193')
    # Assigning a type to the variable 'dtype' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 26), 'dtype', call_assignment_106193_107050)
    
    
    
    # Call to len(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'shape' (line 623)
    shape_107052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 11), 'shape', False)
    # Processing the call keyword arguments (line 623)
    kwargs_107053 = {}
    # Getting the type of 'len' (line 623)
    len_107051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 7), 'len', False)
    # Calling len(args, kwargs) (line 623)
    len_call_result_107054 = invoke(stypy.reporting.localization.Localization(__file__, 623, 7), len_107051, *[shape_107052], **kwargs_107053)
    
    int_107055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 21), 'int')
    # Applying the binary operator '==' (line 623)
    result_eq_107056 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 7), '==', len_call_result_107054, int_107055)
    
    # Testing the type of an if condition (line 623)
    if_condition_107057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 623, 4), result_eq_107056)
    # Assigning a type to the variable 'if_condition_107057' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'if_condition_107057', if_condition_107057)
    # SSA begins for if statement (line 623)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 624):
    
    # Assigning a Num to a Name (line 624):
    int_107058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 16), 'int')
    # Assigning a type to the variable 'count' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'count', int_107058)
    # SSA branch for the else part of an if statement (line 623)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 626):
    
    # Assigning a Call to a Name (line 626):
    
    # Call to reduce(...): (line 626)
    # Processing the call arguments (line 626)
    # Getting the type of 'shape' (line 626)
    shape_107062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 38), 'shape', False)
    # Processing the call keyword arguments (line 626)
    kwargs_107063 = {}
    # Getting the type of 'numpy' (line 626)
    numpy_107059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 16), 'numpy', False)
    # Obtaining the member 'multiply' of a type (line 626)
    multiply_107060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 16), numpy_107059, 'multiply')
    # Obtaining the member 'reduce' of a type (line 626)
    reduce_107061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 16), multiply_107060, 'reduce')
    # Calling reduce(args, kwargs) (line 626)
    reduce_call_result_107064 = invoke(stypy.reporting.localization.Localization(__file__, 626, 16), reduce_107061, *[shape_107062], **kwargs_107063)
    
    # Assigning a type to the variable 'count' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'count', reduce_call_result_107064)
    # SSA join for if statement (line 623)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dtype' (line 629)
    dtype_107065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 7), 'dtype')
    # Obtaining the member 'hasobject' of a type (line 629)
    hasobject_107066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 7), dtype_107065, 'hasobject')
    # Testing the type of an if condition (line 629)
    if_condition_107067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 4), hasobject_107066)
    # Assigning a type to the variable 'if_condition_107067' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'if_condition_107067', if_condition_107067)
    # SSA begins for if statement (line 629)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'allow_pickle' (line 631)
    allow_pickle_107068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 15), 'allow_pickle')
    # Applying the 'not' unary operator (line 631)
    result_not__107069 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 11), 'not', allow_pickle_107068)
    
    # Testing the type of an if condition (line 631)
    if_condition_107070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 631, 8), result_not__107069)
    # Assigning a type to the variable 'if_condition_107070' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'if_condition_107070', if_condition_107070)
    # SSA begins for if statement (line 631)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 632)
    # Processing the call arguments (line 632)
    str_107072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 29), 'str', 'Object arrays cannot be loaded when allow_pickle=False')
    # Processing the call keyword arguments (line 632)
    kwargs_107073 = {}
    # Getting the type of 'ValueError' (line 632)
    ValueError_107071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 632)
    ValueError_call_result_107074 = invoke(stypy.reporting.localization.Localization(__file__, 632, 18), ValueError_107071, *[str_107072], **kwargs_107073)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 632, 12), ValueError_call_result_107074, 'raise parameter', BaseException)
    # SSA join for if statement (line 631)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 634)
    # Getting the type of 'pickle_kwargs' (line 634)
    pickle_kwargs_107075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 11), 'pickle_kwargs')
    # Getting the type of 'None' (line 634)
    None_107076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 28), 'None')
    
    (may_be_107077, more_types_in_union_107078) = may_be_none(pickle_kwargs_107075, None_107076)

    if may_be_107077:

        if more_types_in_union_107078:
            # Runtime conditional SSA (line 634)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 635):
        
        # Assigning a Dict to a Name (line 635):
        
        # Obtaining an instance of the builtin type 'dict' (line 635)
        dict_107079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 635)
        
        # Assigning a type to the variable 'pickle_kwargs' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'pickle_kwargs', dict_107079)

        if more_types_in_union_107078:
            # SSA join for if statement (line 634)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 636)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 637):
    
    # Assigning a Call to a Name (line 637):
    
    # Call to load(...): (line 637)
    # Processing the call arguments (line 637)
    # Getting the type of 'fp' (line 637)
    fp_107082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 32), 'fp', False)
    # Processing the call keyword arguments (line 637)
    # Getting the type of 'pickle_kwargs' (line 637)
    pickle_kwargs_107083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 38), 'pickle_kwargs', False)
    kwargs_107084 = {'pickle_kwargs_107083': pickle_kwargs_107083}
    # Getting the type of 'pickle' (line 637)
    pickle_107080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'pickle', False)
    # Obtaining the member 'load' of a type (line 637)
    load_107081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 20), pickle_107080, 'load')
    # Calling load(args, kwargs) (line 637)
    load_call_result_107085 = invoke(stypy.reporting.localization.Localization(__file__, 637, 20), load_107081, *[fp_107082], **kwargs_107084)
    
    # Assigning a type to the variable 'array' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'array', load_call_result_107085)
    # SSA branch for the except part of a try statement (line 636)
    # SSA branch for the except 'UnicodeError' branch of a try statement (line 636)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'UnicodeError' (line 638)
    UnicodeError_107086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 15), 'UnicodeError')
    # Assigning a type to the variable 'err' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'err', UnicodeError_107086)
    
    
    
    # Obtaining the type of the subscript
    int_107087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 32), 'int')
    # Getting the type of 'sys' (line 639)
    sys_107088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 15), 'sys')
    # Obtaining the member 'version_info' of a type (line 639)
    version_info_107089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 15), sys_107088, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 639)
    getitem___107090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 15), version_info_107089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 639)
    subscript_call_result_107091 = invoke(stypy.reporting.localization.Localization(__file__, 639, 15), getitem___107090, int_107087)
    
    int_107092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 38), 'int')
    # Applying the binary operator '>=' (line 639)
    result_ge_107093 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 15), '>=', subscript_call_result_107091, int_107092)
    
    # Testing the type of an if condition (line 639)
    if_condition_107094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 12), result_ge_107093)
    # Assigning a type to the variable 'if_condition_107094' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'if_condition_107094', if_condition_107094)
    # SSA begins for if statement (line 639)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to UnicodeError(...): (line 641)
    # Processing the call arguments (line 641)
    str_107096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 35), 'str', 'Unpickling a python object failed: %r\nYou may need to pass the encoding= option to numpy.load')
    
    # Obtaining an instance of the builtin type 'tuple' (line 643)
    tuple_107097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 643)
    # Adding element type (line 643)
    # Getting the type of 'err' (line 643)
    err_107098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 54), 'err', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 54), tuple_107097, err_107098)
    
    # Applying the binary operator '%' (line 641)
    result_mod_107099 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 35), '%', str_107096, tuple_107097)
    
    # Processing the call keyword arguments (line 641)
    kwargs_107100 = {}
    # Getting the type of 'UnicodeError' (line 641)
    UnicodeError_107095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 22), 'UnicodeError', False)
    # Calling UnicodeError(args, kwargs) (line 641)
    UnicodeError_call_result_107101 = invoke(stypy.reporting.localization.Localization(__file__, 641, 22), UnicodeError_107095, *[result_mod_107099], **kwargs_107100)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 641, 16), UnicodeError_call_result_107101, 'raise parameter', BaseException)
    # SSA join for if statement (line 639)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 636)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 629)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isfileobj(...): (line 646)
    # Processing the call arguments (line 646)
    # Getting the type of 'fp' (line 646)
    fp_107103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 21), 'fp', False)
    # Processing the call keyword arguments (line 646)
    kwargs_107104 = {}
    # Getting the type of 'isfileobj' (line 646)
    isfileobj_107102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 11), 'isfileobj', False)
    # Calling isfileobj(args, kwargs) (line 646)
    isfileobj_call_result_107105 = invoke(stypy.reporting.localization.Localization(__file__, 646, 11), isfileobj_107102, *[fp_107103], **kwargs_107104)
    
    # Testing the type of an if condition (line 646)
    if_condition_107106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 8), isfileobj_call_result_107105)
    # Assigning a type to the variable 'if_condition_107106' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'if_condition_107106', if_condition_107106)
    # SSA begins for if statement (line 646)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 648):
    
    # Assigning a Call to a Name (line 648):
    
    # Call to fromfile(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 'fp' (line 648)
    fp_107109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 35), 'fp', False)
    # Processing the call keyword arguments (line 648)
    # Getting the type of 'dtype' (line 648)
    dtype_107110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 45), 'dtype', False)
    keyword_107111 = dtype_107110
    # Getting the type of 'count' (line 648)
    count_107112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 58), 'count', False)
    keyword_107113 = count_107112
    kwargs_107114 = {'count': keyword_107113, 'dtype': keyword_107111}
    # Getting the type of 'numpy' (line 648)
    numpy_107107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 20), 'numpy', False)
    # Obtaining the member 'fromfile' of a type (line 648)
    fromfile_107108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 20), numpy_107107, 'fromfile')
    # Calling fromfile(args, kwargs) (line 648)
    fromfile_call_result_107115 = invoke(stypy.reporting.localization.Localization(__file__, 648, 20), fromfile_107108, *[fp_107109], **kwargs_107114)
    
    # Assigning a type to the variable 'array' (line 648)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 12), 'array', fromfile_call_result_107115)
    # SSA branch for the else part of an if statement (line 646)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 658):
    
    # Assigning a BinOp to a Name (line 658):
    # Getting the type of 'BUFFER_SIZE' (line 658)
    BUFFER_SIZE_107116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 29), 'BUFFER_SIZE')
    
    # Call to min(...): (line 658)
    # Processing the call arguments (line 658)
    # Getting the type of 'BUFFER_SIZE' (line 658)
    BUFFER_SIZE_107118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 48), 'BUFFER_SIZE', False)
    # Getting the type of 'dtype' (line 658)
    dtype_107119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 61), 'dtype', False)
    # Obtaining the member 'itemsize' of a type (line 658)
    itemsize_107120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 61), dtype_107119, 'itemsize')
    # Processing the call keyword arguments (line 658)
    kwargs_107121 = {}
    # Getting the type of 'min' (line 658)
    min_107117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 44), 'min', False)
    # Calling min(args, kwargs) (line 658)
    min_call_result_107122 = invoke(stypy.reporting.localization.Localization(__file__, 658, 44), min_107117, *[BUFFER_SIZE_107118, itemsize_107120], **kwargs_107121)
    
    # Applying the binary operator '//' (line 658)
    result_floordiv_107123 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 29), '//', BUFFER_SIZE_107116, min_call_result_107122)
    
    # Assigning a type to the variable 'max_read_count' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'max_read_count', result_floordiv_107123)
    
    # Assigning a Call to a Name (line 660):
    
    # Assigning a Call to a Name (line 660):
    
    # Call to empty(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'count' (line 660)
    count_107126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 32), 'count', False)
    # Processing the call keyword arguments (line 660)
    # Getting the type of 'dtype' (line 660)
    dtype_107127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 45), 'dtype', False)
    keyword_107128 = dtype_107127
    kwargs_107129 = {'dtype': keyword_107128}
    # Getting the type of 'numpy' (line 660)
    numpy_107124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 20), 'numpy', False)
    # Obtaining the member 'empty' of a type (line 660)
    empty_107125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 20), numpy_107124, 'empty')
    # Calling empty(args, kwargs) (line 660)
    empty_call_result_107130 = invoke(stypy.reporting.localization.Localization(__file__, 660, 20), empty_107125, *[count_107126], **kwargs_107129)
    
    # Assigning a type to the variable 'array' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'array', empty_call_result_107130)
    
    
    # Call to range(...): (line 661)
    # Processing the call arguments (line 661)
    int_107132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 27), 'int')
    # Getting the type of 'count' (line 661)
    count_107133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 30), 'count', False)
    # Getting the type of 'max_read_count' (line 661)
    max_read_count_107134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 37), 'max_read_count', False)
    # Processing the call keyword arguments (line 661)
    kwargs_107135 = {}
    # Getting the type of 'range' (line 661)
    range_107131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 21), 'range', False)
    # Calling range(args, kwargs) (line 661)
    range_call_result_107136 = invoke(stypy.reporting.localization.Localization(__file__, 661, 21), range_107131, *[int_107132, count_107133, max_read_count_107134], **kwargs_107135)
    
    # Testing the type of a for loop iterable (line 661)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 661, 12), range_call_result_107136)
    # Getting the type of the for loop variable (line 661)
    for_loop_var_107137 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 661, 12), range_call_result_107136)
    # Assigning a type to the variable 'i' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'i', for_loop_var_107137)
    # SSA begins for a for statement (line 661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 662):
    
    # Assigning a Call to a Name (line 662):
    
    # Call to min(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'max_read_count' (line 662)
    max_read_count_107139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 33), 'max_read_count', False)
    # Getting the type of 'count' (line 662)
    count_107140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 49), 'count', False)
    # Getting the type of 'i' (line 662)
    i_107141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 57), 'i', False)
    # Applying the binary operator '-' (line 662)
    result_sub_107142 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 49), '-', count_107140, i_107141)
    
    # Processing the call keyword arguments (line 662)
    kwargs_107143 = {}
    # Getting the type of 'min' (line 662)
    min_107138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 29), 'min', False)
    # Calling min(args, kwargs) (line 662)
    min_call_result_107144 = invoke(stypy.reporting.localization.Localization(__file__, 662, 29), min_107138, *[max_read_count_107139, result_sub_107142], **kwargs_107143)
    
    # Assigning a type to the variable 'read_count' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'read_count', min_call_result_107144)
    
    # Assigning a Call to a Name (line 663):
    
    # Assigning a Call to a Name (line 663):
    
    # Call to int(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 'read_count' (line 663)
    read_count_107146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 32), 'read_count', False)
    # Getting the type of 'dtype' (line 663)
    dtype_107147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 45), 'dtype', False)
    # Obtaining the member 'itemsize' of a type (line 663)
    itemsize_107148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 45), dtype_107147, 'itemsize')
    # Applying the binary operator '*' (line 663)
    result_mul_107149 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 32), '*', read_count_107146, itemsize_107148)
    
    # Processing the call keyword arguments (line 663)
    kwargs_107150 = {}
    # Getting the type of 'int' (line 663)
    int_107145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 28), 'int', False)
    # Calling int(args, kwargs) (line 663)
    int_call_result_107151 = invoke(stypy.reporting.localization.Localization(__file__, 663, 28), int_107145, *[result_mul_107149], **kwargs_107150)
    
    # Assigning a type to the variable 'read_size' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'read_size', int_call_result_107151)
    
    # Assigning a Call to a Name (line 664):
    
    # Assigning a Call to a Name (line 664):
    
    # Call to _read_bytes(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 'fp' (line 664)
    fp_107153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 35), 'fp', False)
    # Getting the type of 'read_size' (line 664)
    read_size_107154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 39), 'read_size', False)
    str_107155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 50), 'str', 'array data')
    # Processing the call keyword arguments (line 664)
    kwargs_107156 = {}
    # Getting the type of '_read_bytes' (line 664)
    _read_bytes_107152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 23), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 664)
    _read_bytes_call_result_107157 = invoke(stypy.reporting.localization.Localization(__file__, 664, 23), _read_bytes_107152, *[fp_107153, read_size_107154, str_107155], **kwargs_107156)
    
    # Assigning a type to the variable 'data' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'data', _read_bytes_call_result_107157)
    
    # Assigning a Call to a Subscript (line 665):
    
    # Assigning a Call to a Subscript (line 665):
    
    # Call to frombuffer(...): (line 665)
    # Processing the call arguments (line 665)
    # Getting the type of 'data' (line 665)
    data_107160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 57), 'data', False)
    # Processing the call keyword arguments (line 665)
    # Getting the type of 'dtype' (line 665)
    dtype_107161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 69), 'dtype', False)
    keyword_107162 = dtype_107161
    # Getting the type of 'read_count' (line 666)
    read_count_107163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 63), 'read_count', False)
    keyword_107164 = read_count_107163
    kwargs_107165 = {'count': keyword_107164, 'dtype': keyword_107162}
    # Getting the type of 'numpy' (line 665)
    numpy_107158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 40), 'numpy', False)
    # Obtaining the member 'frombuffer' of a type (line 665)
    frombuffer_107159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 40), numpy_107158, 'frombuffer')
    # Calling frombuffer(args, kwargs) (line 665)
    frombuffer_call_result_107166 = invoke(stypy.reporting.localization.Localization(__file__, 665, 40), frombuffer_107159, *[data_107160], **kwargs_107165)
    
    # Getting the type of 'array' (line 665)
    array_107167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'array')
    # Getting the type of 'i' (line 665)
    i_107168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 22), 'i')
    # Getting the type of 'i' (line 665)
    i_107169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 24), 'i')
    # Getting the type of 'read_count' (line 665)
    read_count_107170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 26), 'read_count')
    # Applying the binary operator '+' (line 665)
    result_add_107171 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 24), '+', i_107169, read_count_107170)
    
    slice_107172 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 665, 16), i_107168, result_add_107171, None)
    # Storing an element on a container (line 665)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 16), array_107167, (slice_107172, frombuffer_call_result_107166))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 646)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'fortran_order' (line 668)
    fortran_order_107173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 11), 'fortran_order')
    # Testing the type of an if condition (line 668)
    if_condition_107174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 8), fortran_order_107173)
    # Assigning a type to the variable 'if_condition_107174' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'if_condition_107174', if_condition_107174)
    # SSA begins for if statement (line 668)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Attribute (line 669):
    
    # Assigning a Subscript to a Attribute (line 669):
    
    # Obtaining the type of the subscript
    int_107175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 34), 'int')
    slice_107176 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 669, 26), None, None, int_107175)
    # Getting the type of 'shape' (line 669)
    shape_107177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 26), 'shape')
    # Obtaining the member '__getitem__' of a type (line 669)
    getitem___107178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 26), shape_107177, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 669)
    subscript_call_result_107179 = invoke(stypy.reporting.localization.Localization(__file__, 669, 26), getitem___107178, slice_107176)
    
    # Getting the type of 'array' (line 669)
    array_107180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'array')
    # Setting the type of the member 'shape' of a type (line 669)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 12), array_107180, 'shape', subscript_call_result_107179)
    
    # Assigning a Call to a Name (line 670):
    
    # Assigning a Call to a Name (line 670):
    
    # Call to transpose(...): (line 670)
    # Processing the call keyword arguments (line 670)
    kwargs_107183 = {}
    # Getting the type of 'array' (line 670)
    array_107181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 20), 'array', False)
    # Obtaining the member 'transpose' of a type (line 670)
    transpose_107182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 20), array_107181, 'transpose')
    # Calling transpose(args, kwargs) (line 670)
    transpose_call_result_107184 = invoke(stypy.reporting.localization.Localization(__file__, 670, 20), transpose_107182, *[], **kwargs_107183)
    
    # Assigning a type to the variable 'array' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'array', transpose_call_result_107184)
    # SSA branch for the else part of an if statement (line 668)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Attribute (line 672):
    
    # Assigning a Name to a Attribute (line 672):
    # Getting the type of 'shape' (line 672)
    shape_107185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 26), 'shape')
    # Getting the type of 'array' (line 672)
    array_107186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'array')
    # Setting the type of the member 'shape' of a type (line 672)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 12), array_107186, 'shape', shape_107185)
    # SSA join for if statement (line 668)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 629)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'array' (line 674)
    array_107187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 11), 'array')
    # Assigning a type to the variable 'stypy_return_type' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'stypy_return_type', array_107187)
    
    # ################# End of 'read_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_array' in the type store
    # Getting the type of 'stypy_return_type' (line 592)
    stypy_return_type_107188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_107188)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_array'
    return stypy_return_type_107188

# Assigning a type to the variable 'read_array' (line 592)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'read_array', read_array)

@norecursion
def open_memmap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_107189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 31), 'str', 'r+')
    # Getting the type of 'None' (line 677)
    None_107190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 43), 'None')
    # Getting the type of 'None' (line 677)
    None_107191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 55), 'None')
    # Getting the type of 'False' (line 678)
    False_107192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 30), 'False')
    # Getting the type of 'None' (line 678)
    None_107193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 45), 'None')
    defaults = [str_107189, None_107190, None_107191, False_107192, None_107193]
    # Create a new context for function 'open_memmap'
    module_type_store = module_type_store.open_function_context('open_memmap', 677, 0, False)
    
    # Passed parameters checking function
    open_memmap.stypy_localization = localization
    open_memmap.stypy_type_of_self = None
    open_memmap.stypy_type_store = module_type_store
    open_memmap.stypy_function_name = 'open_memmap'
    open_memmap.stypy_param_names_list = ['filename', 'mode', 'dtype', 'shape', 'fortran_order', 'version']
    open_memmap.stypy_varargs_param_name = None
    open_memmap.stypy_kwargs_param_name = None
    open_memmap.stypy_call_defaults = defaults
    open_memmap.stypy_call_varargs = varargs
    open_memmap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'open_memmap', ['filename', 'mode', 'dtype', 'shape', 'fortran_order', 'version'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'open_memmap', localization, ['filename', 'mode', 'dtype', 'shape', 'fortran_order', 'version'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'open_memmap(...)' code ##################

    str_107194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, (-1)), 'str', '\n    Open a .npy file as a memory-mapped array.\n\n    This may be used to read an existing file or create a new one.\n\n    Parameters\n    ----------\n    filename : str\n        The name of the file on disk.  This may *not* be a file-like\n        object.\n    mode : str, optional\n        The mode in which to open the file; the default is \'r+\'.  In\n        addition to the standard file modes, \'c\' is also accepted to mean\n        "copy on write."  See `memmap` for the available mode strings.\n    dtype : data-type, optional\n        The data type of the array if we are creating a new file in "write"\n        mode, if not, `dtype` is ignored.  The default value is None, which\n        results in a data-type of `float64`.\n    shape : tuple of int\n        The shape of the array if we are creating a new file in "write"\n        mode, in which case this parameter is required.  Otherwise, this\n        parameter is ignored and is thus optional.\n    fortran_order : bool, optional\n        Whether the array should be Fortran-contiguous (True) or\n        C-contiguous (False, the default) if we are creating a new file in\n        "write" mode.\n    version : tuple of int (major, minor) or None\n        If the mode is a "write" mode, then this is the version of the file\n        format used to create the file.  None means use the oldest\n        supported version that is able to store the data.  Default: None\n\n    Returns\n    -------\n    marray : memmap\n        The memory-mapped array.\n\n    Raises\n    ------\n    ValueError\n        If the data or the mode is invalid.\n    IOError\n        If the file is not found or cannot be opened correctly.\n\n    See Also\n    --------\n    memmap\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 727)
    # Getting the type of 'basestring' (line 727)
    basestring_107195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 32), 'basestring')
    # Getting the type of 'filename' (line 727)
    filename_107196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 22), 'filename')
    
    (may_be_107197, more_types_in_union_107198) = may_not_be_subtype(basestring_107195, filename_107196)

    if may_be_107197:

        if more_types_in_union_107198:
            # Runtime conditional SSA (line 727)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'filename' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'filename', remove_subtype_from_union(filename_107196, basestring))
        
        # Call to ValueError(...): (line 728)
        # Processing the call arguments (line 728)
        str_107200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 25), 'str', 'Filename must be a string.  Memmap cannot use existing file handles.')
        # Processing the call keyword arguments (line 728)
        kwargs_107201 = {}
        # Getting the type of 'ValueError' (line 728)
        ValueError_107199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 728)
        ValueError_call_result_107202 = invoke(stypy.reporting.localization.Localization(__file__, 728, 14), ValueError_107199, *[str_107200], **kwargs_107201)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 728, 8), ValueError_call_result_107202, 'raise parameter', BaseException)

        if more_types_in_union_107198:
            # SSA join for if statement (line 727)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    str_107203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 7), 'str', 'w')
    # Getting the type of 'mode' (line 731)
    mode_107204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 14), 'mode')
    # Applying the binary operator 'in' (line 731)
    result_contains_107205 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 7), 'in', str_107203, mode_107204)
    
    # Testing the type of an if condition (line 731)
    if_condition_107206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 731, 4), result_contains_107205)
    # Assigning a type to the variable 'if_condition_107206' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'if_condition_107206', if_condition_107206)
    # SSA begins for if statement (line 731)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _check_version(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of 'version' (line 734)
    version_107208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 23), 'version', False)
    # Processing the call keyword arguments (line 734)
    kwargs_107209 = {}
    # Getting the type of '_check_version' (line 734)
    _check_version_107207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), '_check_version', False)
    # Calling _check_version(args, kwargs) (line 734)
    _check_version_call_result_107210 = invoke(stypy.reporting.localization.Localization(__file__, 734, 8), _check_version_107207, *[version_107208], **kwargs_107209)
    
    
    # Assigning a Call to a Name (line 737):
    
    # Assigning a Call to a Name (line 737):
    
    # Call to dtype(...): (line 737)
    # Processing the call arguments (line 737)
    # Getting the type of 'dtype' (line 737)
    dtype_107213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 28), 'dtype', False)
    # Processing the call keyword arguments (line 737)
    kwargs_107214 = {}
    # Getting the type of 'numpy' (line 737)
    numpy_107211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 16), 'numpy', False)
    # Obtaining the member 'dtype' of a type (line 737)
    dtype_107212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 16), numpy_107211, 'dtype')
    # Calling dtype(args, kwargs) (line 737)
    dtype_call_result_107215 = invoke(stypy.reporting.localization.Localization(__file__, 737, 16), dtype_107212, *[dtype_107213], **kwargs_107214)
    
    # Assigning a type to the variable 'dtype' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'dtype', dtype_call_result_107215)
    
    # Getting the type of 'dtype' (line 738)
    dtype_107216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 11), 'dtype')
    # Obtaining the member 'hasobject' of a type (line 738)
    hasobject_107217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 11), dtype_107216, 'hasobject')
    # Testing the type of an if condition (line 738)
    if_condition_107218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 738, 8), hasobject_107217)
    # Assigning a type to the variable 'if_condition_107218' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'if_condition_107218', if_condition_107218)
    # SSA begins for if statement (line 738)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 739):
    
    # Assigning a Str to a Name (line 739):
    str_107219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 18), 'str', "Array can't be memory-mapped: Python objects in dtype.")
    # Assigning a type to the variable 'msg' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'msg', str_107219)
    
    # Call to ValueError(...): (line 740)
    # Processing the call arguments (line 740)
    # Getting the type of 'msg' (line 740)
    msg_107221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 29), 'msg', False)
    # Processing the call keyword arguments (line 740)
    kwargs_107222 = {}
    # Getting the type of 'ValueError' (line 740)
    ValueError_107220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 740)
    ValueError_call_result_107223 = invoke(stypy.reporting.localization.Localization(__file__, 740, 18), ValueError_107220, *[msg_107221], **kwargs_107222)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 740, 12), ValueError_call_result_107223, 'raise parameter', BaseException)
    # SSA join for if statement (line 738)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 741):
    
    # Assigning a Call to a Name (line 741):
    
    # Call to dict(...): (line 741)
    # Processing the call keyword arguments (line 741)
    
    # Call to dtype_to_descr(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'dtype' (line 742)
    dtype_107226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 33), 'dtype', False)
    # Processing the call keyword arguments (line 742)
    kwargs_107227 = {}
    # Getting the type of 'dtype_to_descr' (line 742)
    dtype_to_descr_107225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 18), 'dtype_to_descr', False)
    # Calling dtype_to_descr(args, kwargs) (line 742)
    dtype_to_descr_call_result_107228 = invoke(stypy.reporting.localization.Localization(__file__, 742, 18), dtype_to_descr_107225, *[dtype_107226], **kwargs_107227)
    
    keyword_107229 = dtype_to_descr_call_result_107228
    # Getting the type of 'fortran_order' (line 743)
    fortran_order_107230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 26), 'fortran_order', False)
    keyword_107231 = fortran_order_107230
    # Getting the type of 'shape' (line 744)
    shape_107232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 18), 'shape', False)
    keyword_107233 = shape_107232
    kwargs_107234 = {'shape': keyword_107233, 'fortran_order': keyword_107231, 'descr': keyword_107229}
    # Getting the type of 'dict' (line 741)
    dict_107224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'dict', False)
    # Calling dict(args, kwargs) (line 741)
    dict_call_result_107235 = invoke(stypy.reporting.localization.Localization(__file__, 741, 12), dict_107224, *[], **kwargs_107234)
    
    # Assigning a type to the variable 'd' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'd', dict_call_result_107235)
    
    # Assigning a Call to a Name (line 747):
    
    # Assigning a Call to a Name (line 747):
    
    # Call to open(...): (line 747)
    # Processing the call arguments (line 747)
    # Getting the type of 'filename' (line 747)
    filename_107237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 18), 'filename', False)
    # Getting the type of 'mode' (line 747)
    mode_107238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 28), 'mode', False)
    str_107239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 33), 'str', 'b')
    # Applying the binary operator '+' (line 747)
    result_add_107240 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 28), '+', mode_107238, str_107239)
    
    # Processing the call keyword arguments (line 747)
    kwargs_107241 = {}
    # Getting the type of 'open' (line 747)
    open_107236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 13), 'open', False)
    # Calling open(args, kwargs) (line 747)
    open_call_result_107242 = invoke(stypy.reporting.localization.Localization(__file__, 747, 13), open_107236, *[filename_107237, result_add_107240], **kwargs_107241)
    
    # Assigning a type to the variable 'fp' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'fp', open_call_result_107242)
    
    # Try-finally block (line 748)
    
    # Assigning a Call to a Name (line 749):
    
    # Assigning a Call to a Name (line 749):
    
    # Call to _write_array_header(...): (line 749)
    # Processing the call arguments (line 749)
    # Getting the type of 'fp' (line 749)
    fp_107244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 43), 'fp', False)
    # Getting the type of 'd' (line 749)
    d_107245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 47), 'd', False)
    # Getting the type of 'version' (line 749)
    version_107246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 50), 'version', False)
    # Processing the call keyword arguments (line 749)
    kwargs_107247 = {}
    # Getting the type of '_write_array_header' (line 749)
    _write_array_header_107243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 23), '_write_array_header', False)
    # Calling _write_array_header(args, kwargs) (line 749)
    _write_array_header_call_result_107248 = invoke(stypy.reporting.localization.Localization(__file__, 749, 23), _write_array_header_107243, *[fp_107244, d_107245, version_107246], **kwargs_107247)
    
    # Assigning a type to the variable 'used_ver' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 12), 'used_ver', _write_array_header_call_result_107248)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'version' (line 751)
    version_107249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 15), 'version')
    
    # Obtaining an instance of the builtin type 'tuple' (line 751)
    tuple_107250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 751)
    # Adding element type (line 751)
    int_107251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 27), tuple_107250, int_107251)
    # Adding element type (line 751)
    int_107252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 27), tuple_107250, int_107252)
    
    # Applying the binary operator '!=' (line 751)
    result_ne_107253 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 15), '!=', version_107249, tuple_107250)
    
    
    # Getting the type of 'used_ver' (line 751)
    used_ver_107254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 37), 'used_ver')
    
    # Obtaining an instance of the builtin type 'tuple' (line 751)
    tuple_107255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 751)
    # Adding element type (line 751)
    int_107256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 50), tuple_107255, int_107256)
    # Adding element type (line 751)
    int_107257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 50), tuple_107255, int_107257)
    
    # Applying the binary operator '==' (line 751)
    result_eq_107258 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 37), '==', used_ver_107254, tuple_107255)
    
    # Applying the binary operator 'and' (line 751)
    result_and_keyword_107259 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 15), 'and', result_ne_107253, result_eq_107258)
    
    # Testing the type of an if condition (line 751)
    if_condition_107260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 751, 12), result_and_keyword_107259)
    # Assigning a type to the variable 'if_condition_107260' (line 751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 12), 'if_condition_107260', if_condition_107260)
    # SSA begins for if statement (line 751)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 752)
    # Processing the call arguments (line 752)
    str_107263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 30), 'str', 'Stored array in format 2.0. It can only beread by NumPy >= 1.9')
    # Getting the type of 'UserWarning' (line 753)
    UserWarning_107264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 54), 'UserWarning', False)
    # Processing the call keyword arguments (line 752)
    kwargs_107265 = {}
    # Getting the type of 'warnings' (line 752)
    warnings_107261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 16), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 752)
    warn_107262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 16), warnings_107261, 'warn')
    # Calling warn(args, kwargs) (line 752)
    warn_call_result_107266 = invoke(stypy.reporting.localization.Localization(__file__, 752, 16), warn_107262, *[str_107263, UserWarning_107264], **kwargs_107265)
    
    # SSA join for if statement (line 751)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 754):
    
    # Assigning a Call to a Name (line 754):
    
    # Call to tell(...): (line 754)
    # Processing the call keyword arguments (line 754)
    kwargs_107269 = {}
    # Getting the type of 'fp' (line 754)
    fp_107267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 21), 'fp', False)
    # Obtaining the member 'tell' of a type (line 754)
    tell_107268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 21), fp_107267, 'tell')
    # Calling tell(args, kwargs) (line 754)
    tell_call_result_107270 = invoke(stypy.reporting.localization.Localization(__file__, 754, 21), tell_107268, *[], **kwargs_107269)
    
    # Assigning a type to the variable 'offset' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 12), 'offset', tell_call_result_107270)
    
    # finally branch of the try-finally block (line 748)
    
    # Call to close(...): (line 756)
    # Processing the call keyword arguments (line 756)
    kwargs_107273 = {}
    # Getting the type of 'fp' (line 756)
    fp_107271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'fp', False)
    # Obtaining the member 'close' of a type (line 756)
    close_107272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 12), fp_107271, 'close')
    # Calling close(args, kwargs) (line 756)
    close_call_result_107274 = invoke(stypy.reporting.localization.Localization(__file__, 756, 12), close_107272, *[], **kwargs_107273)
    
    
    # SSA branch for the else part of an if statement (line 731)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 759):
    
    # Assigning a Call to a Name (line 759):
    
    # Call to open(...): (line 759)
    # Processing the call arguments (line 759)
    # Getting the type of 'filename' (line 759)
    filename_107276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 18), 'filename', False)
    str_107277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 28), 'str', 'rb')
    # Processing the call keyword arguments (line 759)
    kwargs_107278 = {}
    # Getting the type of 'open' (line 759)
    open_107275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 13), 'open', False)
    # Calling open(args, kwargs) (line 759)
    open_call_result_107279 = invoke(stypy.reporting.localization.Localization(__file__, 759, 13), open_107275, *[filename_107276, str_107277], **kwargs_107278)
    
    # Assigning a type to the variable 'fp' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'fp', open_call_result_107279)
    
    # Try-finally block (line 760)
    
    # Assigning a Call to a Name (line 761):
    
    # Assigning a Call to a Name (line 761):
    
    # Call to read_magic(...): (line 761)
    # Processing the call arguments (line 761)
    # Getting the type of 'fp' (line 761)
    fp_107281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 33), 'fp', False)
    # Processing the call keyword arguments (line 761)
    kwargs_107282 = {}
    # Getting the type of 'read_magic' (line 761)
    read_magic_107280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 22), 'read_magic', False)
    # Calling read_magic(args, kwargs) (line 761)
    read_magic_call_result_107283 = invoke(stypy.reporting.localization.Localization(__file__, 761, 22), read_magic_107280, *[fp_107281], **kwargs_107282)
    
    # Assigning a type to the variable 'version' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'version', read_magic_call_result_107283)
    
    # Call to _check_version(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'version' (line 762)
    version_107285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 27), 'version', False)
    # Processing the call keyword arguments (line 762)
    kwargs_107286 = {}
    # Getting the type of '_check_version' (line 762)
    _check_version_107284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), '_check_version', False)
    # Calling _check_version(args, kwargs) (line 762)
    _check_version_call_result_107287 = invoke(stypy.reporting.localization.Localization(__file__, 762, 12), _check_version_107284, *[version_107285], **kwargs_107286)
    
    
    # Assigning a Call to a Tuple (line 764):
    
    # Assigning a Call to a Name:
    
    # Call to _read_array_header(...): (line 764)
    # Processing the call arguments (line 764)
    # Getting the type of 'fp' (line 764)
    fp_107289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 61), 'fp', False)
    # Getting the type of 'version' (line 764)
    version_107290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 65), 'version', False)
    # Processing the call keyword arguments (line 764)
    kwargs_107291 = {}
    # Getting the type of '_read_array_header' (line 764)
    _read_array_header_107288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 42), '_read_array_header', False)
    # Calling _read_array_header(args, kwargs) (line 764)
    _read_array_header_call_result_107292 = invoke(stypy.reporting.localization.Localization(__file__, 764, 42), _read_array_header_107288, *[fp_107289, version_107290], **kwargs_107291)
    
    # Assigning a type to the variable 'call_assignment_106194' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106194', _read_array_header_call_result_107292)
    
    # Assigning a Call to a Name (line 764):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_107295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 12), 'int')
    # Processing the call keyword arguments
    kwargs_107296 = {}
    # Getting the type of 'call_assignment_106194' (line 764)
    call_assignment_106194_107293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106194', False)
    # Obtaining the member '__getitem__' of a type (line 764)
    getitem___107294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), call_assignment_106194_107293, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_107297 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___107294, *[int_107295], **kwargs_107296)
    
    # Assigning a type to the variable 'call_assignment_106195' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106195', getitem___call_result_107297)
    
    # Assigning a Name to a Name (line 764):
    # Getting the type of 'call_assignment_106195' (line 764)
    call_assignment_106195_107298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106195')
    # Assigning a type to the variable 'shape' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'shape', call_assignment_106195_107298)
    
    # Assigning a Call to a Name (line 764):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_107301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 12), 'int')
    # Processing the call keyword arguments
    kwargs_107302 = {}
    # Getting the type of 'call_assignment_106194' (line 764)
    call_assignment_106194_107299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106194', False)
    # Obtaining the member '__getitem__' of a type (line 764)
    getitem___107300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), call_assignment_106194_107299, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_107303 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___107300, *[int_107301], **kwargs_107302)
    
    # Assigning a type to the variable 'call_assignment_106196' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106196', getitem___call_result_107303)
    
    # Assigning a Name to a Name (line 764):
    # Getting the type of 'call_assignment_106196' (line 764)
    call_assignment_106196_107304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106196')
    # Assigning a type to the variable 'fortran_order' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 19), 'fortran_order', call_assignment_106196_107304)
    
    # Assigning a Call to a Name (line 764):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_107307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 12), 'int')
    # Processing the call keyword arguments
    kwargs_107308 = {}
    # Getting the type of 'call_assignment_106194' (line 764)
    call_assignment_106194_107305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106194', False)
    # Obtaining the member '__getitem__' of a type (line 764)
    getitem___107306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), call_assignment_106194_107305, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_107309 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___107306, *[int_107307], **kwargs_107308)
    
    # Assigning a type to the variable 'call_assignment_106197' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106197', getitem___call_result_107309)
    
    # Assigning a Name to a Name (line 764):
    # Getting the type of 'call_assignment_106197' (line 764)
    call_assignment_106197_107310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'call_assignment_106197')
    # Assigning a type to the variable 'dtype' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 34), 'dtype', call_assignment_106197_107310)
    
    # Getting the type of 'dtype' (line 765)
    dtype_107311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 15), 'dtype')
    # Obtaining the member 'hasobject' of a type (line 765)
    hasobject_107312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 15), dtype_107311, 'hasobject')
    # Testing the type of an if condition (line 765)
    if_condition_107313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 765, 12), hasobject_107312)
    # Assigning a type to the variable 'if_condition_107313' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'if_condition_107313', if_condition_107313)
    # SSA begins for if statement (line 765)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 766):
    
    # Assigning a Str to a Name (line 766):
    str_107314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 22), 'str', "Array can't be memory-mapped: Python objects in dtype.")
    # Assigning a type to the variable 'msg' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 16), 'msg', str_107314)
    
    # Call to ValueError(...): (line 767)
    # Processing the call arguments (line 767)
    # Getting the type of 'msg' (line 767)
    msg_107316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 33), 'msg', False)
    # Processing the call keyword arguments (line 767)
    kwargs_107317 = {}
    # Getting the type of 'ValueError' (line 767)
    ValueError_107315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 767)
    ValueError_call_result_107318 = invoke(stypy.reporting.localization.Localization(__file__, 767, 22), ValueError_107315, *[msg_107316], **kwargs_107317)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 767, 16), ValueError_call_result_107318, 'raise parameter', BaseException)
    # SSA join for if statement (line 765)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 768):
    
    # Assigning a Call to a Name (line 768):
    
    # Call to tell(...): (line 768)
    # Processing the call keyword arguments (line 768)
    kwargs_107321 = {}
    # Getting the type of 'fp' (line 768)
    fp_107319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 21), 'fp', False)
    # Obtaining the member 'tell' of a type (line 768)
    tell_107320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 21), fp_107319, 'tell')
    # Calling tell(args, kwargs) (line 768)
    tell_call_result_107322 = invoke(stypy.reporting.localization.Localization(__file__, 768, 21), tell_107320, *[], **kwargs_107321)
    
    # Assigning a type to the variable 'offset' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 12), 'offset', tell_call_result_107322)
    
    # finally branch of the try-finally block (line 760)
    
    # Call to close(...): (line 770)
    # Processing the call keyword arguments (line 770)
    kwargs_107325 = {}
    # Getting the type of 'fp' (line 770)
    fp_107323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'fp', False)
    # Obtaining the member 'close' of a type (line 770)
    close_107324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 12), fp_107323, 'close')
    # Calling close(args, kwargs) (line 770)
    close_call_result_107326 = invoke(stypy.reporting.localization.Localization(__file__, 770, 12), close_107324, *[], **kwargs_107325)
    
    
    # SSA join for if statement (line 731)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'fortran_order' (line 772)
    fortran_order_107327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 7), 'fortran_order')
    # Testing the type of an if condition (line 772)
    if_condition_107328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 772, 4), fortran_order_107327)
    # Assigning a type to the variable 'if_condition_107328' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'if_condition_107328', if_condition_107328)
    # SSA begins for if statement (line 772)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 773):
    
    # Assigning a Str to a Name (line 773):
    str_107329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 16), 'str', 'F')
    # Assigning a type to the variable 'order' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'order', str_107329)
    # SSA branch for the else part of an if statement (line 772)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 775):
    
    # Assigning a Str to a Name (line 775):
    str_107330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 16), 'str', 'C')
    # Assigning a type to the variable 'order' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'order', str_107330)
    # SSA join for if statement (line 772)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 779)
    mode_107331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 7), 'mode')
    str_107332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 15), 'str', 'w+')
    # Applying the binary operator '==' (line 779)
    result_eq_107333 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 7), '==', mode_107331, str_107332)
    
    # Testing the type of an if condition (line 779)
    if_condition_107334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 779, 4), result_eq_107333)
    # Assigning a type to the variable 'if_condition_107334' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'if_condition_107334', if_condition_107334)
    # SSA begins for if statement (line 779)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 780):
    
    # Assigning a Str to a Name (line 780):
    str_107335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 15), 'str', 'r+')
    # Assigning a type to the variable 'mode' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'mode', str_107335)
    # SSA join for if statement (line 779)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 782):
    
    # Assigning a Call to a Name (line 782):
    
    # Call to memmap(...): (line 782)
    # Processing the call arguments (line 782)
    # Getting the type of 'filename' (line 782)
    filename_107338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 26), 'filename', False)
    # Processing the call keyword arguments (line 782)
    # Getting the type of 'dtype' (line 782)
    dtype_107339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 42), 'dtype', False)
    keyword_107340 = dtype_107339
    # Getting the type of 'shape' (line 782)
    shape_107341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 55), 'shape', False)
    keyword_107342 = shape_107341
    # Getting the type of 'order' (line 782)
    order_107343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 68), 'order', False)
    keyword_107344 = order_107343
    # Getting the type of 'mode' (line 783)
    mode_107345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 13), 'mode', False)
    keyword_107346 = mode_107345
    # Getting the type of 'offset' (line 783)
    offset_107347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 26), 'offset', False)
    keyword_107348 = offset_107347
    kwargs_107349 = {'dtype': keyword_107340, 'shape': keyword_107342, 'offset': keyword_107348, 'order': keyword_107344, 'mode': keyword_107346}
    # Getting the type of 'numpy' (line 782)
    numpy_107336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 13), 'numpy', False)
    # Obtaining the member 'memmap' of a type (line 782)
    memmap_107337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 13), numpy_107336, 'memmap')
    # Calling memmap(args, kwargs) (line 782)
    memmap_call_result_107350 = invoke(stypy.reporting.localization.Localization(__file__, 782, 13), memmap_107337, *[filename_107338], **kwargs_107349)
    
    # Assigning a type to the variable 'marray' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'marray', memmap_call_result_107350)
    # Getting the type of 'marray' (line 785)
    marray_107351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 11), 'marray')
    # Assigning a type to the variable 'stypy_return_type' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'stypy_return_type', marray_107351)
    
    # ################# End of 'open_memmap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'open_memmap' in the type store
    # Getting the type of 'stypy_return_type' (line 677)
    stypy_return_type_107352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_107352)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'open_memmap'
    return stypy_return_type_107352

# Assigning a type to the variable 'open_memmap' (line 677)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 0), 'open_memmap', open_memmap)

@norecursion
def _read_bytes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_107353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 41), 'str', 'ran out of data')
    defaults = [str_107353]
    # Create a new context for function '_read_bytes'
    module_type_store = module_type_store.open_function_context('_read_bytes', 788, 0, False)
    
    # Passed parameters checking function
    _read_bytes.stypy_localization = localization
    _read_bytes.stypy_type_of_self = None
    _read_bytes.stypy_type_store = module_type_store
    _read_bytes.stypy_function_name = '_read_bytes'
    _read_bytes.stypy_param_names_list = ['fp', 'size', 'error_template']
    _read_bytes.stypy_varargs_param_name = None
    _read_bytes.stypy_kwargs_param_name = None
    _read_bytes.stypy_call_defaults = defaults
    _read_bytes.stypy_call_varargs = varargs
    _read_bytes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_bytes', ['fp', 'size', 'error_template'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_bytes', localization, ['fp', 'size', 'error_template'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_bytes(...)' code ##################

    str_107354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, (-1)), 'str', '\n    Read from file-like object until size bytes are read.\n    Raises ValueError if not EOF is encountered before size bytes are read.\n    Non-blocking objects only supported if they derive from io objects.\n\n    Required as e.g. ZipExtFile in python 2.6 can return less data than\n    requested.\n    ')
    
    # Assigning a Call to a Name (line 797):
    
    # Assigning a Call to a Name (line 797):
    
    # Call to bytes(...): (line 797)
    # Processing the call keyword arguments (line 797)
    kwargs_107356 = {}
    # Getting the type of 'bytes' (line 797)
    bytes_107355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 11), 'bytes', False)
    # Calling bytes(args, kwargs) (line 797)
    bytes_call_result_107357 = invoke(stypy.reporting.localization.Localization(__file__, 797, 11), bytes_107355, *[], **kwargs_107356)
    
    # Assigning a type to the variable 'data' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'data', bytes_call_result_107357)
    
    # Getting the type of 'True' (line 798)
    True_107358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 10), 'True')
    # Testing the type of an if condition (line 798)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 798, 4), True_107358)
    # SSA begins for while statement (line 798)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # SSA begins for try-except statement (line 802)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 803):
    
    # Assigning a Call to a Name (line 803):
    
    # Call to read(...): (line 803)
    # Processing the call arguments (line 803)
    # Getting the type of 'size' (line 803)
    size_107361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 24), 'size', False)
    
    # Call to len(...): (line 803)
    # Processing the call arguments (line 803)
    # Getting the type of 'data' (line 803)
    data_107363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 35), 'data', False)
    # Processing the call keyword arguments (line 803)
    kwargs_107364 = {}
    # Getting the type of 'len' (line 803)
    len_107362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 31), 'len', False)
    # Calling len(args, kwargs) (line 803)
    len_call_result_107365 = invoke(stypy.reporting.localization.Localization(__file__, 803, 31), len_107362, *[data_107363], **kwargs_107364)
    
    # Applying the binary operator '-' (line 803)
    result_sub_107366 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 24), '-', size_107361, len_call_result_107365)
    
    # Processing the call keyword arguments (line 803)
    kwargs_107367 = {}
    # Getting the type of 'fp' (line 803)
    fp_107359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 16), 'fp', False)
    # Obtaining the member 'read' of a type (line 803)
    read_107360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 16), fp_107359, 'read')
    # Calling read(args, kwargs) (line 803)
    read_call_result_107368 = invoke(stypy.reporting.localization.Localization(__file__, 803, 16), read_107360, *[result_sub_107366], **kwargs_107367)
    
    # Assigning a type to the variable 'r' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 12), 'r', read_call_result_107368)
    
    # Getting the type of 'data' (line 804)
    data_107369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 12), 'data')
    # Getting the type of 'r' (line 804)
    r_107370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 20), 'r')
    # Applying the binary operator '+=' (line 804)
    result_iadd_107371 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 12), '+=', data_107369, r_107370)
    # Assigning a type to the variable 'data' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 12), 'data', result_iadd_107371)
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'r' (line 805)
    r_107373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 19), 'r', False)
    # Processing the call keyword arguments (line 805)
    kwargs_107374 = {}
    # Getting the type of 'len' (line 805)
    len_107372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 15), 'len', False)
    # Calling len(args, kwargs) (line 805)
    len_call_result_107375 = invoke(stypy.reporting.localization.Localization(__file__, 805, 15), len_107372, *[r_107373], **kwargs_107374)
    
    int_107376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 25), 'int')
    # Applying the binary operator '==' (line 805)
    result_eq_107377 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 15), '==', len_call_result_107375, int_107376)
    
    
    
    # Call to len(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'data' (line 805)
    data_107379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 34), 'data', False)
    # Processing the call keyword arguments (line 805)
    kwargs_107380 = {}
    # Getting the type of 'len' (line 805)
    len_107378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 30), 'len', False)
    # Calling len(args, kwargs) (line 805)
    len_call_result_107381 = invoke(stypy.reporting.localization.Localization(__file__, 805, 30), len_107378, *[data_107379], **kwargs_107380)
    
    # Getting the type of 'size' (line 805)
    size_107382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 43), 'size')
    # Applying the binary operator '==' (line 805)
    result_eq_107383 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 30), '==', len_call_result_107381, size_107382)
    
    # Applying the binary operator 'or' (line 805)
    result_or_keyword_107384 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 15), 'or', result_eq_107377, result_eq_107383)
    
    # Testing the type of an if condition (line 805)
    if_condition_107385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 805, 12), result_or_keyword_107384)
    # Assigning a type to the variable 'if_condition_107385' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 12), 'if_condition_107385', if_condition_107385)
    # SSA begins for if statement (line 805)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 805)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 802)
    # SSA branch for the except 'Attribute' branch of a try statement (line 802)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 802)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 798)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'data' (line 809)
    data_107387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 11), 'data', False)
    # Processing the call keyword arguments (line 809)
    kwargs_107388 = {}
    # Getting the type of 'len' (line 809)
    len_107386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 7), 'len', False)
    # Calling len(args, kwargs) (line 809)
    len_call_result_107389 = invoke(stypy.reporting.localization.Localization(__file__, 809, 7), len_107386, *[data_107387], **kwargs_107388)
    
    # Getting the type of 'size' (line 809)
    size_107390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 20), 'size')
    # Applying the binary operator '!=' (line 809)
    result_ne_107391 = python_operator(stypy.reporting.localization.Localization(__file__, 809, 7), '!=', len_call_result_107389, size_107390)
    
    # Testing the type of an if condition (line 809)
    if_condition_107392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 4), result_ne_107391)
    # Assigning a type to the variable 'if_condition_107392' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'if_condition_107392', if_condition_107392)
    # SSA begins for if statement (line 809)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 810):
    
    # Assigning a Str to a Name (line 810):
    str_107393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 14), 'str', 'EOF: reading %s, expected %d bytes got %d')
    # Assigning a type to the variable 'msg' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'msg', str_107393)
    
    # Call to ValueError(...): (line 811)
    # Processing the call arguments (line 811)
    # Getting the type of 'msg' (line 811)
    msg_107395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 25), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 811)
    tuple_107396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 811)
    # Adding element type (line 811)
    # Getting the type of 'error_template' (line 811)
    error_template_107397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 32), 'error_template', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 32), tuple_107396, error_template_107397)
    # Adding element type (line 811)
    # Getting the type of 'size' (line 811)
    size_107398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 48), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 32), tuple_107396, size_107398)
    # Adding element type (line 811)
    
    # Call to len(...): (line 811)
    # Processing the call arguments (line 811)
    # Getting the type of 'data' (line 811)
    data_107400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 58), 'data', False)
    # Processing the call keyword arguments (line 811)
    kwargs_107401 = {}
    # Getting the type of 'len' (line 811)
    len_107399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 54), 'len', False)
    # Calling len(args, kwargs) (line 811)
    len_call_result_107402 = invoke(stypy.reporting.localization.Localization(__file__, 811, 54), len_107399, *[data_107400], **kwargs_107401)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 32), tuple_107396, len_call_result_107402)
    
    # Applying the binary operator '%' (line 811)
    result_mod_107403 = python_operator(stypy.reporting.localization.Localization(__file__, 811, 25), '%', msg_107395, tuple_107396)
    
    # Processing the call keyword arguments (line 811)
    kwargs_107404 = {}
    # Getting the type of 'ValueError' (line 811)
    ValueError_107394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 811)
    ValueError_call_result_107405 = invoke(stypy.reporting.localization.Localization(__file__, 811, 14), ValueError_107394, *[result_mod_107403], **kwargs_107404)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 811, 8), ValueError_call_result_107405, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 809)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'data' (line 813)
    data_107406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 15), 'data')
    # Assigning a type to the variable 'stypy_return_type' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'stypy_return_type', data_107406)
    # SSA join for if statement (line 809)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_read_bytes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_bytes' in the type store
    # Getting the type of 'stypy_return_type' (line 788)
    stypy_return_type_107407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_107407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_bytes'
    return stypy_return_type_107407

# Assigning a type to the variable '_read_bytes' (line 788)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 0), '_read_bytes', _read_bytes)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
