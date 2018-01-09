
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Classes for read / write of matlab (TM) 5 files
2: 
3: The matfile specification last found here:
4: 
5: http://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf
6: 
7: (as of December 5 2008)
8: '''
9: from __future__ import division, print_function, absolute_import
10: 
11: '''
12: =================================
13:  Note on functions and mat files
14: =================================
15: 
16: The document above does not give any hints as to the storage of matlab
17: function handles, or anonymous function handles.  I had therefore to
18: guess the format of matlab arrays of ``mxFUNCTION_CLASS`` and
19: ``mxOPAQUE_CLASS`` by looking at example mat files.
20: 
21: ``mxFUNCTION_CLASS`` stores all types of matlab functions.  It seems to
22: contain a struct matrix with a set pattern of fields.  For anonymous
23: functions, a sub-fields of one of these fields seems to contain the
24: well-named ``mxOPAQUE_CLASS``. This seems to cotain:
25: 
26: * array flags as for any matlab matrix
27: * 3 int8 strings
28: * a matrix
29: 
30: It seems that, whenever the mat file contains a ``mxOPAQUE_CLASS``
31: instance, there is also an un-named matrix (name == '') at the end of
32: the mat file.  I'll call this the ``__function_workspace__`` matrix.
33: 
34: When I saved two anonymous functions in a mat file, or appended another
35: anonymous function to the mat file, there was still only one
36: ``__function_workspace__`` un-named matrix at the end, but larger than
37: that for a mat file with a single anonymous function, suggesting that
38: the workspaces for the two functions had been merged.
39: 
40: The ``__function_workspace__`` matrix appears to be of double class
41: (``mxCLASS_DOUBLE``), but stored as uint8, the memory for which is in
42: the format of a mini .mat file, without the first 124 bytes of the file
43: header (the description and the subsystem_offset), but with the version
44: U2 bytes, and the S2 endian test bytes.  There follow 4 zero bytes,
45: presumably for 8 byte padding, and then a series of ``miMATRIX``
46: entries, as in a standard mat file. The ``miMATRIX`` entries appear to
47: be series of un-named (name == '') matrices, and may also contain arrays
48: of this same mini-mat format.
49: 
50: I guess that:
51: 
52: * saving an anonymous function back to a mat file will need the
53:   associated ``__function_workspace__`` matrix saved as well for the
54:   anonymous function to work correctly.
55: * appending to a mat file that has a ``__function_workspace__`` would
56:   involve first pulling off this workspace, appending, checking whether
57:   there were any more anonymous functions appended, and then somehow
58:   merging the relevant workspaces, and saving at the end of the mat
59:   file.
60: 
61: The mat files I was playing with are in ``tests/data``:
62: 
63: * sqr.mat
64: * parabola.mat
65: * some_functions.mat
66: 
67: See ``tests/test_mio.py:test_mio_funcs.py`` for a debugging
68: script I was working with.
69: 
70: '''
71: 
72: # Small fragments of current code adapted from matfile.py by Heiko
73: # Henkelmann
74: 
75: import os
76: import time
77: import sys
78: import zlib
79: 
80: from io import BytesIO
81: 
82: import warnings
83: 
84: import numpy as np
85: from numpy.compat import asbytes, asstr
86: 
87: import scipy.sparse
88: 
89: from scipy._lib.six import string_types
90: 
91: from .byteordercodes import native_code, swapped_code
92: 
93: from .miobase import (MatFileReader, docfiller, matdims, read_dtype,
94:                       arr_to_chars, arr_dtype_number, MatWriteError,
95:                       MatReadError, MatReadWarning)
96: 
97: # Reader object for matlab 5 format variables
98: from .mio5_utils import VarReader5
99: 
100: # Constants and helper objects
101: from .mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
102:                           NP_TO_MXTYPES, miCOMPRESSED, miMATRIX, miINT8,
103:                           miUTF8, miUINT32, mxCELL_CLASS, mxSTRUCT_CLASS,
104:                           mxOBJECT_CLASS, mxCHAR_CLASS, mxSPARSE_CLASS,
105:                           mxDOUBLE_CLASS, mclass_info)
106: 
107: from .streams import ZlibInputStream
108: 
109: 
110: class MatFile5Reader(MatFileReader):
111:     ''' Reader for Mat 5 mat files
112:     Adds the following attribute to base class
113: 
114:     uint16_codec - char codec to use for uint16 char arrays
115:         (defaults to system default codec)
116: 
117:     Uses variable reader that has the following stardard interface (see
118:     abstract class in ``miobase``::
119: 
120:        __init__(self, file_reader)
121:        read_header(self)
122:        array_from_header(self)
123: 
124:     and added interface::
125: 
126:        set_stream(self, stream)
127:        read_full_tag(self)
128: 
129:     '''
130:     @docfiller
131:     def __init__(self,
132:                  mat_stream,
133:                  byte_order=None,
134:                  mat_dtype=False,
135:                  squeeze_me=False,
136:                  chars_as_strings=True,
137:                  matlab_compatible=False,
138:                  struct_as_record=True,
139:                  verify_compressed_data_integrity=True,
140:                  uint16_codec=None
141:                  ):
142:         '''Initializer for matlab 5 file format reader
143: 
144:     %(matstream_arg)s
145:     %(load_args)s
146:     %(struct_arg)s
147:     uint16_codec : {None, string}
148:         Set codec to use for uint16 char arrays (e.g. 'utf-8').
149:         Use system default codec if None
150:         '''
151:         super(MatFile5Reader, self).__init__(
152:             mat_stream,
153:             byte_order,
154:             mat_dtype,
155:             squeeze_me,
156:             chars_as_strings,
157:             matlab_compatible,
158:             struct_as_record,
159:             verify_compressed_data_integrity
160:             )
161:         # Set uint16 codec
162:         if not uint16_codec:
163:             uint16_codec = sys.getdefaultencoding()
164:         self.uint16_codec = uint16_codec
165:         # placeholders for readers - see initialize_read method
166:         self._file_reader = None
167:         self._matrix_reader = None
168: 
169:     def guess_byte_order(self):
170:         ''' Guess byte order.
171:         Sets stream pointer to 0 '''
172:         self.mat_stream.seek(126)
173:         mi = self.mat_stream.read(2)
174:         self.mat_stream.seek(0)
175:         return mi == b'IM' and '<' or '>'
176: 
177:     def read_file_header(self):
178:         ''' Read in mat 5 file header '''
179:         hdict = {}
180:         hdr_dtype = MDTYPES[self.byte_order]['dtypes']['file_header']
181:         hdr = read_dtype(self.mat_stream, hdr_dtype)
182:         hdict['__header__'] = hdr['description'].item().strip(b' \t\n\000')
183:         v_major = hdr['version'] >> 8
184:         v_minor = hdr['version'] & 0xFF
185:         hdict['__version__'] = '%d.%d' % (v_major, v_minor)
186:         return hdict
187: 
188:     def initialize_read(self):
189:         ''' Run when beginning read of variables
190: 
191:         Sets up readers from parameters in `self`
192:         '''
193:         # reader for top level stream.  We need this extra top-level
194:         # reader because we use the matrix_reader object to contain
195:         # compressed matrices (so they have their own stream)
196:         self._file_reader = VarReader5(self)
197:         # reader for matrix streams
198:         self._matrix_reader = VarReader5(self)
199: 
200:     def read_var_header(self):
201:         ''' Read header, return header, next position
202: 
203:         Header has to define at least .name and .is_global
204: 
205:         Parameters
206:         ----------
207:         None
208: 
209:         Returns
210:         -------
211:         header : object
212:            object that can be passed to self.read_var_array, and that
213:            has attributes .name and .is_global
214:         next_position : int
215:            position in stream of next variable
216:         '''
217:         mdtype, byte_count = self._file_reader.read_full_tag()
218:         if not byte_count > 0:
219:             raise ValueError("Did not read any bytes")
220:         next_pos = self.mat_stream.tell() + byte_count
221:         if mdtype == miCOMPRESSED:
222:             # Make new stream from compressed data
223:             stream = ZlibInputStream(self.mat_stream, byte_count)
224:             self._matrix_reader.set_stream(stream)
225:             check_stream_limit = self.verify_compressed_data_integrity
226:             mdtype, byte_count = self._matrix_reader.read_full_tag()
227:         else:
228:             check_stream_limit = False
229:             self._matrix_reader.set_stream(self.mat_stream)
230:         if not mdtype == miMATRIX:
231:             raise TypeError('Expecting miMATRIX type here, got %d' % mdtype)
232:         header = self._matrix_reader.read_header(check_stream_limit)
233:         return header, next_pos
234: 
235:     def read_var_array(self, header, process=True):
236:         ''' Read array, given `header`
237: 
238:         Parameters
239:         ----------
240:         header : header object
241:            object with fields defining variable header
242:         process : {True, False} bool, optional
243:            If True, apply recursive post-processing during loading of
244:            array.
245: 
246:         Returns
247:         -------
248:         arr : array
249:            array with post-processing applied or not according to
250:            `process`.
251:         '''
252:         return self._matrix_reader.array_from_header(header, process)
253: 
254:     def get_variables(self, variable_names=None):
255:         ''' get variables from stream as dictionary
256: 
257:         variable_names   - optional list of variable names to get
258: 
259:         If variable_names is None, then get all variables in file
260:         '''
261:         if isinstance(variable_names, string_types):
262:             variable_names = [variable_names]
263:         elif variable_names is not None:
264:             variable_names = list(variable_names)
265: 
266:         self.mat_stream.seek(0)
267:         # Here we pass all the parameters in self to the reading objects
268:         self.initialize_read()
269:         mdict = self.read_file_header()
270:         mdict['__globals__'] = []
271:         while not self.end_of_stream():
272:             hdr, next_position = self.read_var_header()
273:             name = asstr(hdr.name)
274:             if name in mdict:
275:                 warnings.warn('Duplicate variable name "%s" in stream'
276:                               ' - replacing previous with new\n'
277:                               'Consider mio5.varmats_from_mat to split '
278:                               'file into single variable files' % name,
279:                               MatReadWarning, stacklevel=2)
280:             if name == '':
281:                 # can only be a matlab 7 function workspace
282:                 name = '__function_workspace__'
283:                 # We want to keep this raw because mat_dtype processing
284:                 # will break the format (uint8 as mxDOUBLE_CLASS)
285:                 process = False
286:             else:
287:                 process = True
288:             if variable_names is not None and name not in variable_names:
289:                 self.mat_stream.seek(next_position)
290:                 continue
291:             try:
292:                 res = self.read_var_array(hdr, process)
293:             except MatReadError as err:
294:                 warnings.warn(
295:                     'Unreadable variable "%s", because "%s"' %
296:                     (name, err),
297:                     Warning, stacklevel=2)
298:                 res = "Read error: %s" % err
299:             self.mat_stream.seek(next_position)
300:             mdict[name] = res
301:             if hdr.is_global:
302:                 mdict['__globals__'].append(name)
303:             if variable_names is not None:
304:                 variable_names.remove(name)
305:                 if len(variable_names) == 0:
306:                     break
307:         return mdict
308: 
309:     def list_variables(self):
310:         ''' list variables from stream '''
311:         self.mat_stream.seek(0)
312:         # Here we pass all the parameters in self to the reading objects
313:         self.initialize_read()
314:         self.read_file_header()
315:         vars = []
316:         while not self.end_of_stream():
317:             hdr, next_position = self.read_var_header()
318:             name = asstr(hdr.name)
319:             if name == '':
320:                 # can only be a matlab 7 function workspace
321:                 name = '__function_workspace__'
322: 
323:             shape = self._matrix_reader.shape_from_header(hdr)
324:             if hdr.is_logical:
325:                 info = 'logical'
326:             else:
327:                 info = mclass_info.get(hdr.mclass, 'unknown')
328:             vars.append((name, shape, info))
329: 
330:             self.mat_stream.seek(next_position)
331:         return vars
332: 
333: 
334: def varmats_from_mat(file_obj):
335:     ''' Pull variables out of mat 5 file as a sequence of mat file objects
336: 
337:     This can be useful with a difficult mat file, containing unreadable
338:     variables.  This routine pulls the variables out in raw form and puts them,
339:     unread, back into a file stream for saving or reading.  Another use is the
340:     pathological case where there is more than one variable of the same name in
341:     the file; this routine returns the duplicates, whereas the standard reader
342:     will overwrite duplicates in the returned dictionary.
343: 
344:     The file pointer in `file_obj` will be undefined.  File pointers for the
345:     returned file-like objects are set at 0.
346: 
347:     Parameters
348:     ----------
349:     file_obj : file-like
350:         file object containing mat file
351: 
352:     Returns
353:     -------
354:     named_mats : list
355:         list contains tuples of (name, BytesIO) where BytesIO is a file-like
356:         object containing mat file contents as for a single variable.  The
357:         BytesIO contains a string with the original header and a single var. If
358:         ``var_file_obj`` is an individual BytesIO instance, then save as a mat
359:         file with something like ``open('test.mat',
360:         'wb').write(var_file_obj.read())``
361: 
362:     Examples
363:     --------
364:     >>> import scipy.io
365: 
366:     BytesIO is from the ``io`` module in python 3, and is ``cStringIO`` for
367:     python < 3.
368: 
369:     >>> mat_fileobj = BytesIO()
370:     >>> scipy.io.savemat(mat_fileobj, {'b': np.arange(10), 'a': 'a string'})
371:     >>> varmats = varmats_from_mat(mat_fileobj)
372:     >>> sorted([name for name, str_obj in varmats])
373:     ['a', 'b']
374:     '''
375:     rdr = MatFile5Reader(file_obj)
376:     file_obj.seek(0)
377:     # Raw read of top-level file header
378:     hdr_len = MDTYPES[native_code]['dtypes']['file_header'].itemsize
379:     raw_hdr = file_obj.read(hdr_len)
380:     # Initialize variable reading
381:     file_obj.seek(0)
382:     rdr.initialize_read()
383:     mdict = rdr.read_file_header()
384:     next_position = file_obj.tell()
385:     named_mats = []
386:     while not rdr.end_of_stream():
387:         start_position = next_position
388:         hdr, next_position = rdr.read_var_header()
389:         name = asstr(hdr.name)
390:         # Read raw variable string
391:         file_obj.seek(start_position)
392:         byte_count = next_position - start_position
393:         var_str = file_obj.read(byte_count)
394:         # write to stringio object
395:         out_obj = BytesIO()
396:         out_obj.write(raw_hdr)
397:         out_obj.write(var_str)
398:         out_obj.seek(0)
399:         named_mats.append((name, out_obj))
400:     return named_mats
401: 
402: 
403: class EmptyStructMarker(object):
404:     ''' Class to indicate presence of empty matlab struct on output '''
405: 
406: 
407: def to_writeable(source):
408:     ''' Convert input object ``source`` to something we can write
409: 
410:     Parameters
411:     ----------
412:     source : object
413: 
414:     Returns
415:     -------
416:     arr : None or ndarray or EmptyStructMarker
417:         If `source` cannot be converted to something we can write to a matfile,
418:         return None.  If `source` is equivalent to an empty dictionary, return
419:         ``EmptyStructMarker``.  Otherwise return `source` converted to an
420:         ndarray with contents for writing to matfile.
421:     '''
422:     if isinstance(source, np.ndarray):
423:         return source
424:     if source is None:
425:         return None
426:     # Objects that implement mappings
427:     is_mapping = (hasattr(source, 'keys') and hasattr(source, 'values') and
428:                   hasattr(source, 'items'))
429:     # Objects that don't implement mappings, but do have dicts
430:     if not is_mapping and hasattr(source, '__dict__'):
431:         source = dict((key, value) for key, value in source.__dict__.items()
432:                       if not key.startswith('_'))
433:         is_mapping = True
434:     if is_mapping:
435:         dtype = []
436:         values = []
437:         for field, value in source.items():
438:             if (isinstance(field, string_types) and
439:                     field[0] not in '_0123456789'):
440:                 dtype.append((field, object))
441:                 values.append(value)
442:         if dtype:
443:             return np.array([tuple(values)], dtype)
444:         else:
445:             return EmptyStructMarker
446:     # Next try and convert to an array
447:     narr = np.asanyarray(source)
448:     if narr.dtype.type in (object, np.object_) and \
449:        narr.shape == () and narr == source:
450:         # No interesting conversion possible
451:         return None
452:     return narr
453: 
454: 
455: # Native byte ordered dtypes for convenience for writers
456: NDT_FILE_HDR = MDTYPES[native_code]['dtypes']['file_header']
457: NDT_TAG_FULL = MDTYPES[native_code]['dtypes']['tag_full']
458: NDT_TAG_SMALL = MDTYPES[native_code]['dtypes']['tag_smalldata']
459: NDT_ARRAY_FLAGS = MDTYPES[native_code]['dtypes']['array_flags']
460: 
461: 
462: class VarWriter5(object):
463:     ''' Generic matlab matrix writing class '''
464:     mat_tag = np.zeros((), NDT_TAG_FULL)
465:     mat_tag['mdtype'] = miMATRIX
466: 
467:     def __init__(self, file_writer):
468:         self.file_stream = file_writer.file_stream
469:         self.unicode_strings = file_writer.unicode_strings
470:         self.long_field_names = file_writer.long_field_names
471:         self.oned_as = file_writer.oned_as
472:         # These are used for top level writes, and unset after
473:         self._var_name = None
474:         self._var_is_global = False
475: 
476:     def write_bytes(self, arr):
477:         self.file_stream.write(arr.tostring(order='F'))
478: 
479:     def write_string(self, s):
480:         self.file_stream.write(s)
481: 
482:     def write_element(self, arr, mdtype=None):
483:         ''' write tag and data '''
484:         if mdtype is None:
485:             mdtype = NP_TO_MTYPES[arr.dtype.str[1:]]
486:         # Array needs to be in native byte order
487:         if arr.dtype.byteorder == swapped_code:
488:             arr = arr.byteswap().newbyteorder()
489:         byte_count = arr.size*arr.itemsize
490:         if byte_count <= 4:
491:             self.write_smalldata_element(arr, mdtype, byte_count)
492:         else:
493:             self.write_regular_element(arr, mdtype, byte_count)
494: 
495:     def write_smalldata_element(self, arr, mdtype, byte_count):
496:         # write tag with embedded data
497:         tag = np.zeros((), NDT_TAG_SMALL)
498:         tag['byte_count_mdtype'] = (byte_count << 16) + mdtype
499:         # if arr.tostring is < 4, the element will be zero-padded as needed.
500:         tag['data'] = arr.tostring(order='F')
501:         self.write_bytes(tag)
502: 
503:     def write_regular_element(self, arr, mdtype, byte_count):
504:         # write tag, data
505:         tag = np.zeros((), NDT_TAG_FULL)
506:         tag['mdtype'] = mdtype
507:         tag['byte_count'] = byte_count
508:         self.write_bytes(tag)
509:         self.write_bytes(arr)
510:         # pad to next 64-bit boundary
511:         bc_mod_8 = byte_count % 8
512:         if bc_mod_8:
513:             self.file_stream.write(b'\x00' * (8-bc_mod_8))
514: 
515:     def write_header(self,
516:                      shape,
517:                      mclass,
518:                      is_complex=False,
519:                      is_logical=False,
520:                      nzmax=0):
521:         ''' Write header for given data options
522:         shape : sequence
523:            array shape
524:         mclass      - mat5 matrix class
525:         is_complex  - True if matrix is complex
526:         is_logical  - True if matrix is logical
527:         nzmax        - max non zero elements for sparse arrays
528: 
529:         We get the name and the global flag from the object, and reset
530:         them to defaults after we've used them
531:         '''
532:         # get name and is_global from one-shot object store
533:         name = self._var_name
534:         is_global = self._var_is_global
535:         # initialize the top-level matrix tag, store position
536:         self._mat_tag_pos = self.file_stream.tell()
537:         self.write_bytes(self.mat_tag)
538:         # write array flags (complex, global, logical, class, nzmax)
539:         af = np.zeros((), NDT_ARRAY_FLAGS)
540:         af['data_type'] = miUINT32
541:         af['byte_count'] = 8
542:         flags = is_complex << 3 | is_global << 2 | is_logical << 1
543:         af['flags_class'] = mclass | flags << 8
544:         af['nzmax'] = nzmax
545:         self.write_bytes(af)
546:         # shape
547:         self.write_element(np.array(shape, dtype='i4'))
548:         # write name
549:         name = np.asarray(name)
550:         if name == '':  # empty string zero-terminated
551:             self.write_smalldata_element(name, miINT8, 0)
552:         else:
553:             self.write_element(name, miINT8)
554:         # reset the one-shot store to defaults
555:         self._var_name = ''
556:         self._var_is_global = False
557: 
558:     def update_matrix_tag(self, start_pos):
559:         curr_pos = self.file_stream.tell()
560:         self.file_stream.seek(start_pos)
561:         byte_count = curr_pos - start_pos - 8
562:         if byte_count >= 2**32:
563:             raise MatWriteError("Matrix too large to save with Matlab "
564:                                 "5 format")
565:         self.mat_tag['byte_count'] = byte_count
566:         self.write_bytes(self.mat_tag)
567:         self.file_stream.seek(curr_pos)
568: 
569:     def write_top(self, arr, name, is_global):
570:         ''' Write variable at top level of mat file
571: 
572:         Parameters
573:         ----------
574:         arr : array_like
575:             array-like object to create writer for
576:         name : str, optional
577:             name as it will appear in matlab workspace
578:             default is empty string
579:         is_global : {False, True}, optional
580:             whether variable will be global on load into matlab
581:         '''
582:         # these are set before the top-level header write, and unset at
583:         # the end of the same write, because they do not apply for lower levels
584:         self._var_is_global = is_global
585:         self._var_name = name
586:         # write the header and data
587:         self.write(arr)
588: 
589:     def write(self, arr):
590:         ''' Write `arr` to stream at top and sub levels
591: 
592:         Parameters
593:         ----------
594:         arr : array_like
595:             array-like object to create writer for
596:         '''
597:         # store position, so we can update the matrix tag
598:         mat_tag_pos = self.file_stream.tell()
599:         # First check if these are sparse
600:         if scipy.sparse.issparse(arr):
601:             self.write_sparse(arr)
602:             self.update_matrix_tag(mat_tag_pos)
603:             return
604:         # Try to convert things that aren't arrays
605:         narr = to_writeable(arr)
606:         if narr is None:
607:             raise TypeError('Could not convert %s (type %s) to array'
608:                             % (arr, type(arr)))
609:         if isinstance(narr, MatlabObject):
610:             self.write_object(narr)
611:         elif isinstance(narr, MatlabFunction):
612:             raise MatWriteError('Cannot write matlab functions')
613:         elif narr is EmptyStructMarker:  # empty struct array
614:             self.write_empty_struct()
615:         elif narr.dtype.fields:  # struct array
616:             self.write_struct(narr)
617:         elif narr.dtype.hasobject:  # cell array
618:             self.write_cells(narr)
619:         elif narr.dtype.kind in ('U', 'S'):
620:             if self.unicode_strings:
621:                 codec = 'UTF8'
622:             else:
623:                 codec = 'ascii'
624:             self.write_char(narr, codec)
625:         else:
626:             self.write_numeric(narr)
627:         self.update_matrix_tag(mat_tag_pos)
628: 
629:     def write_numeric(self, arr):
630:         imagf = arr.dtype.kind == 'c'
631:         logif = arr.dtype.kind == 'b'
632:         try:
633:             mclass = NP_TO_MXTYPES[arr.dtype.str[1:]]
634:         except KeyError:
635:             # No matching matlab type, probably complex256 / float128 / float96
636:             # Cast data to complex128 / float64.
637:             if imagf:
638:                 arr = arr.astype('c128')
639:             elif logif:
640:                 arr = arr.astype('i1')  # Should only contain 0/1
641:             else:
642:                 arr = arr.astype('f8')
643:             mclass = mxDOUBLE_CLASS
644:         self.write_header(matdims(arr, self.oned_as),
645:                           mclass,
646:                           is_complex=imagf,
647:                           is_logical=logif)
648:         if imagf:
649:             self.write_element(arr.real)
650:             self.write_element(arr.imag)
651:         else:
652:             self.write_element(arr)
653: 
654:     def write_char(self, arr, codec='ascii'):
655:         ''' Write string array `arr` with given `codec`
656:         '''
657:         if arr.size == 0 or np.all(arr == ''):
658:             # This an empty string array or a string array containing
659:             # only empty strings.  Matlab cannot distiguish between a
660:             # string array that is empty, and a string array containing
661:             # only empty strings, because it stores strings as arrays of
662:             # char.  There is no way of having an array of char that is
663:             # not empty, but contains an empty string. We have to
664:             # special-case the array-with-empty-strings because even
665:             # empty strings have zero padding, which would otherwise
666:             # appear in matlab as a string with a space.
667:             shape = (0,) * np.max([arr.ndim, 2])
668:             self.write_header(shape, mxCHAR_CLASS)
669:             self.write_smalldata_element(arr, miUTF8, 0)
670:             return
671:         # non-empty string.
672:         #
673:         # Convert to char array
674:         arr = arr_to_chars(arr)
675:         # We have to write the shape directly, because we are going
676:         # recode the characters, and the resulting stream of chars
677:         # may have a different length
678:         shape = arr.shape
679:         self.write_header(shape, mxCHAR_CLASS)
680:         if arr.dtype.kind == 'U' and arr.size:
681:             # Make one long string from all the characters.  We need to
682:             # transpose here, because we're flattening the array, before
683:             # we write the bytes.  The bytes have to be written in
684:             # Fortran order.
685:             n_chars = np.product(shape)
686:             st_arr = np.ndarray(shape=(),
687:                                 dtype=arr_dtype_number(arr, n_chars),
688:                                 buffer=arr.T.copy())  # Fortran order
689:             # Recode with codec to give byte string
690:             st = st_arr.item().encode(codec)
691:             # Reconstruct as one-dimensional byte array
692:             arr = np.ndarray(shape=(len(st),),
693:                              dtype='S1',
694:                              buffer=st)
695:         self.write_element(arr, mdtype=miUTF8)
696: 
697:     def write_sparse(self, arr):
698:         ''' Sparse matrices are 2D
699:         '''
700:         A = arr.tocsc()  # convert to sparse CSC format
701:         A.sort_indices()     # MATLAB expects sorted row indices
702:         is_complex = (A.dtype.kind == 'c')
703:         is_logical = (A.dtype.kind == 'b')
704:         nz = A.nnz
705:         self.write_header(matdims(arr, self.oned_as),
706:                           mxSPARSE_CLASS,
707:                           is_complex=is_complex,
708:                           is_logical=is_logical,
709:                           # matlab won't load file with 0 nzmax
710:                           nzmax=1 if nz == 0 else nz)
711:         self.write_element(A.indices.astype('i4'))
712:         self.write_element(A.indptr.astype('i4'))
713:         self.write_element(A.data.real)
714:         if is_complex:
715:             self.write_element(A.data.imag)
716: 
717:     def write_cells(self, arr):
718:         self.write_header(matdims(arr, self.oned_as),
719:                           mxCELL_CLASS)
720:         # loop over data, column major
721:         A = np.atleast_2d(arr).flatten('F')
722:         for el in A:
723:             self.write(el)
724: 
725:     def write_empty_struct(self):
726:         self.write_header((1, 1), mxSTRUCT_CLASS)
727:         # max field name length set to 1 in an example matlab struct
728:         self.write_element(np.array(1, dtype=np.int32))
729:         # Field names element is empty
730:         self.write_element(np.array([], dtype=np.int8))
731: 
732:     def write_struct(self, arr):
733:         self.write_header(matdims(arr, self.oned_as),
734:                           mxSTRUCT_CLASS)
735:         self._write_items(arr)
736: 
737:     def _write_items(self, arr):
738:         # write fieldnames
739:         fieldnames = [f[0] for f in arr.dtype.descr]
740:         length = max([len(fieldname) for fieldname in fieldnames])+1
741:         max_length = (self.long_field_names and 64) or 32
742:         if length > max_length:
743:             raise ValueError("Field names are restricted to %d characters" %
744:                              (max_length-1))
745:         self.write_element(np.array([length], dtype='i4'))
746:         self.write_element(
747:             np.array(fieldnames, dtype='S%d' % (length)),
748:             mdtype=miINT8)
749:         A = np.atleast_2d(arr).flatten('F')
750:         for el in A:
751:             for f in fieldnames:
752:                 self.write(el[f])
753: 
754:     def write_object(self, arr):
755:         '''Same as writing structs, except different mx class, and extra
756:         classname element after header
757:         '''
758:         self.write_header(matdims(arr, self.oned_as),
759:                           mxOBJECT_CLASS)
760:         self.write_element(np.array(arr.classname, dtype='S'),
761:                            mdtype=miINT8)
762:         self._write_items(arr)
763: 
764: 
765: class MatFile5Writer(object):
766:     ''' Class for writing mat5 files '''
767: 
768:     @docfiller
769:     def __init__(self, file_stream,
770:                  do_compression=False,
771:                  unicode_strings=False,
772:                  global_vars=None,
773:                  long_field_names=False,
774:                  oned_as='row'):
775:         ''' Initialize writer for matlab 5 format files
776: 
777:         Parameters
778:         ----------
779:         %(do_compression)s
780:         %(unicode_strings)s
781:         global_vars : None or sequence of strings, optional
782:             Names of variables to be marked as global for matlab
783:         %(long_fields)s
784:         %(oned_as)s
785:         '''
786:         self.file_stream = file_stream
787:         self.do_compression = do_compression
788:         self.unicode_strings = unicode_strings
789:         if global_vars:
790:             self.global_vars = global_vars
791:         else:
792:             self.global_vars = []
793:         self.long_field_names = long_field_names
794:         self.oned_as = oned_as
795:         self._matrix_writer = None
796: 
797:     def write_file_header(self):
798:         # write header
799:         hdr = np.zeros((), NDT_FILE_HDR)
800:         hdr['description'] = 'MATLAB 5.0 MAT-file Platform: %s, Created on: %s' \
801:             % (os.name,time.asctime())
802:         hdr['version'] = 0x0100
803:         hdr['endian_test'] = np.ndarray(shape=(),
804:                                       dtype='S2',
805:                                       buffer=np.uint16(0x4d49))
806:         self.file_stream.write(hdr.tostring())
807: 
808:     def put_variables(self, mdict, write_header=None):
809:         ''' Write variables in `mdict` to stream
810: 
811:         Parameters
812:         ----------
813:         mdict : mapping
814:            mapping with method ``items`` returns name, contents pairs where
815:            ``name`` which will appear in the matlab workspace in file load, and
816:            ``contents`` is something writeable to a matlab file, such as a numpy
817:            array.
818:         write_header : {None, True, False}, optional
819:            If True, then write the matlab file header before writing the
820:            variables.  If None (the default) then write the file header
821:            if we are at position 0 in the stream.  By setting False
822:            here, and setting the stream position to the end of the file,
823:            you can append variables to a matlab file
824:         '''
825:         # write header if requested, or None and start of file
826:         if write_header is None:
827:             write_header = self.file_stream.tell() == 0
828:         if write_header:
829:             self.write_file_header()
830:         self._matrix_writer = VarWriter5(self)
831:         for name, var in mdict.items():
832:             if name[0] == '_':
833:                 continue
834:             is_global = name in self.global_vars
835:             if self.do_compression:
836:                 stream = BytesIO()
837:                 self._matrix_writer.file_stream = stream
838:                 self._matrix_writer.write_top(var, asbytes(name), is_global)
839:                 out_str = zlib.compress(stream.getvalue())
840:                 tag = np.empty((), NDT_TAG_FULL)
841:                 tag['mdtype'] = miCOMPRESSED
842:                 tag['byte_count'] = len(out_str)
843:                 self.file_stream.write(tag.tostring())
844:                 self.file_stream.write(out_str)
845:             else:  # not compressing
846:                 self._matrix_writer.write_top(var, asbytes(name), is_global)
847: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_135267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', ' Classes for read / write of matlab (TM) 5 files\n\nThe matfile specification last found here:\n\nhttp://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf\n\n(as of December 5 2008)\n')
str_135268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, (-1)), 'str', "\n=================================\n Note on functions and mat files\n=================================\n\nThe document above does not give any hints as to the storage of matlab\nfunction handles, or anonymous function handles.  I had therefore to\nguess the format of matlab arrays of ``mxFUNCTION_CLASS`` and\n``mxOPAQUE_CLASS`` by looking at example mat files.\n\n``mxFUNCTION_CLASS`` stores all types of matlab functions.  It seems to\ncontain a struct matrix with a set pattern of fields.  For anonymous\nfunctions, a sub-fields of one of these fields seems to contain the\nwell-named ``mxOPAQUE_CLASS``. This seems to cotain:\n\n* array flags as for any matlab matrix\n* 3 int8 strings\n* a matrix\n\nIt seems that, whenever the mat file contains a ``mxOPAQUE_CLASS``\ninstance, there is also an un-named matrix (name == '') at the end of\nthe mat file.  I'll call this the ``__function_workspace__`` matrix.\n\nWhen I saved two anonymous functions in a mat file, or appended another\nanonymous function to the mat file, there was still only one\n``__function_workspace__`` un-named matrix at the end, but larger than\nthat for a mat file with a single anonymous function, suggesting that\nthe workspaces for the two functions had been merged.\n\nThe ``__function_workspace__`` matrix appears to be of double class\n(``mxCLASS_DOUBLE``), but stored as uint8, the memory for which is in\nthe format of a mini .mat file, without the first 124 bytes of the file\nheader (the description and the subsystem_offset), but with the version\nU2 bytes, and the S2 endian test bytes.  There follow 4 zero bytes,\npresumably for 8 byte padding, and then a series of ``miMATRIX``\nentries, as in a standard mat file. The ``miMATRIX`` entries appear to\nbe series of un-named (name == '') matrices, and may also contain arrays\nof this same mini-mat format.\n\nI guess that:\n\n* saving an anonymous function back to a mat file will need the\n  associated ``__function_workspace__`` matrix saved as well for the\n  anonymous function to work correctly.\n* appending to a mat file that has a ``__function_workspace__`` would\n  involve first pulling off this workspace, appending, checking whether\n  there were any more anonymous functions appended, and then somehow\n  merging the relevant workspaces, and saving at the end of the mat\n  file.\n\nThe mat files I was playing with are in ``tests/data``:\n\n* sqr.mat\n* parabola.mat\n* some_functions.mat\n\nSee ``tests/test_mio.py:test_mio_funcs.py`` for a debugging\nscript I was working with.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 75, 0))

# 'import os' statement (line 75)
import os

import_module(stypy.reporting.localization.Localization(__file__, 75, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 76, 0))

# 'import time' statement (line 76)
import time

import_module(stypy.reporting.localization.Localization(__file__, 76, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 77, 0))

# 'import sys' statement (line 77)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 77, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 78, 0))

# 'import zlib' statement (line 78)
import zlib

import_module(stypy.reporting.localization.Localization(__file__, 78, 0), 'zlib', zlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 80, 0))

# 'from io import BytesIO' statement (line 80)
try:
    from io import BytesIO

except:
    BytesIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 80, 0), 'io', None, module_type_store, ['BytesIO'], [BytesIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 82, 0))

# 'import warnings' statement (line 82)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 82, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 84, 0))

# 'import numpy' statement (line 84)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135269 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 84, 0), 'numpy')

if (type(import_135269) is not StypyTypeError):

    if (import_135269 != 'pyd_module'):
        __import__(import_135269)
        sys_modules_135270 = sys.modules[import_135269]
        import_module(stypy.reporting.localization.Localization(__file__, 84, 0), 'np', sys_modules_135270.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 84, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'numpy', import_135269)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 85, 0))

# 'from numpy.compat import asbytes, asstr' statement (line 85)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135271 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 85, 0), 'numpy.compat')

if (type(import_135271) is not StypyTypeError):

    if (import_135271 != 'pyd_module'):
        __import__(import_135271)
        sys_modules_135272 = sys.modules[import_135271]
        import_from_module(stypy.reporting.localization.Localization(__file__, 85, 0), 'numpy.compat', sys_modules_135272.module_type_store, module_type_store, ['asbytes', 'asstr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 85, 0), __file__, sys_modules_135272, sys_modules_135272.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, asstr

        import_from_module(stypy.reporting.localization.Localization(__file__, 85, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'asstr'], [asbytes, asstr])

else:
    # Assigning a type to the variable 'numpy.compat' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'numpy.compat', import_135271)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 87, 0))

# 'import scipy.sparse' statement (line 87)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135273 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 87, 0), 'scipy.sparse')

if (type(import_135273) is not StypyTypeError):

    if (import_135273 != 'pyd_module'):
        __import__(import_135273)
        sys_modules_135274 = sys.modules[import_135273]
        import_module(stypy.reporting.localization.Localization(__file__, 87, 0), 'scipy.sparse', sys_modules_135274.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 87, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'scipy.sparse', import_135273)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 89, 0))

# 'from scipy._lib.six import string_types' statement (line 89)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135275 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy._lib.six')

if (type(import_135275) is not StypyTypeError):

    if (import_135275 != 'pyd_module'):
        __import__(import_135275)
        sys_modules_135276 = sys.modules[import_135275]
        import_from_module(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy._lib.six', sys_modules_135276.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 89, 0), __file__, sys_modules_135276, sys_modules_135276.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy._lib.six', import_135275)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 91, 0))

# 'from scipy.io.matlab.byteordercodes import native_code, swapped_code' statement (line 91)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135277 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.io.matlab.byteordercodes')

if (type(import_135277) is not StypyTypeError):

    if (import_135277 != 'pyd_module'):
        __import__(import_135277)
        sys_modules_135278 = sys.modules[import_135277]
        import_from_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.io.matlab.byteordercodes', sys_modules_135278.module_type_store, module_type_store, ['native_code', 'swapped_code'])
        nest_module(stypy.reporting.localization.Localization(__file__, 91, 0), __file__, sys_modules_135278, sys_modules_135278.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.byteordercodes import native_code, swapped_code

        import_from_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.io.matlab.byteordercodes', None, module_type_store, ['native_code', 'swapped_code'], [native_code, swapped_code])

else:
    # Assigning a type to the variable 'scipy.io.matlab.byteordercodes' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.io.matlab.byteordercodes', import_135277)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 93, 0))

# 'from scipy.io.matlab.miobase import MatFileReader, docfiller, matdims, read_dtype, arr_to_chars, arr_dtype_number, MatWriteError, MatReadError, MatReadWarning' statement (line 93)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135279 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.io.matlab.miobase')

if (type(import_135279) is not StypyTypeError):

    if (import_135279 != 'pyd_module'):
        __import__(import_135279)
        sys_modules_135280 = sys.modules[import_135279]
        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.io.matlab.miobase', sys_modules_135280.module_type_store, module_type_store, ['MatFileReader', 'docfiller', 'matdims', 'read_dtype', 'arr_to_chars', 'arr_dtype_number', 'MatWriteError', 'MatReadError', 'MatReadWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 93, 0), __file__, sys_modules_135280, sys_modules_135280.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.miobase import MatFileReader, docfiller, matdims, read_dtype, arr_to_chars, arr_dtype_number, MatWriteError, MatReadError, MatReadWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.io.matlab.miobase', None, module_type_store, ['MatFileReader', 'docfiller', 'matdims', 'read_dtype', 'arr_to_chars', 'arr_dtype_number', 'MatWriteError', 'MatReadError', 'MatReadWarning'], [MatFileReader, docfiller, matdims, read_dtype, arr_to_chars, arr_dtype_number, MatWriteError, MatReadError, MatReadWarning])

else:
    # Assigning a type to the variable 'scipy.io.matlab.miobase' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.io.matlab.miobase', import_135279)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 98, 0))

# 'from scipy.io.matlab.mio5_utils import VarReader5' statement (line 98)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135281 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.io.matlab.mio5_utils')

if (type(import_135281) is not StypyTypeError):

    if (import_135281 != 'pyd_module'):
        __import__(import_135281)
        sys_modules_135282 = sys.modules[import_135281]
        import_from_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.io.matlab.mio5_utils', sys_modules_135282.module_type_store, module_type_store, ['VarReader5'])
        nest_module(stypy.reporting.localization.Localization(__file__, 98, 0), __file__, sys_modules_135282, sys_modules_135282.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio5_utils import VarReader5

        import_from_module(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.io.matlab.mio5_utils', None, module_type_store, ['VarReader5'], [VarReader5])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio5_utils' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'scipy.io.matlab.mio5_utils', import_135281)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 101, 0))

# 'from scipy.io.matlab.mio5_params import MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES, NP_TO_MXTYPES, miCOMPRESSED, miMATRIX, miINT8, miUTF8, miUINT32, mxCELL_CLASS, mxSTRUCT_CLASS, mxOBJECT_CLASS, mxCHAR_CLASS, mxSPARSE_CLASS, mxDOUBLE_CLASS, mclass_info' statement (line 101)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135283 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.io.matlab.mio5_params')

if (type(import_135283) is not StypyTypeError):

    if (import_135283 != 'pyd_module'):
        __import__(import_135283)
        sys_modules_135284 = sys.modules[import_135283]
        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.io.matlab.mio5_params', sys_modules_135284.module_type_store, module_type_store, ['MatlabObject', 'MatlabFunction', 'MDTYPES', 'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'miCOMPRESSED', 'miMATRIX', 'miINT8', 'miUTF8', 'miUINT32', 'mxCELL_CLASS', 'mxSTRUCT_CLASS', 'mxOBJECT_CLASS', 'mxCHAR_CLASS', 'mxSPARSE_CLASS', 'mxDOUBLE_CLASS', 'mclass_info'])
        nest_module(stypy.reporting.localization.Localization(__file__, 101, 0), __file__, sys_modules_135284, sys_modules_135284.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio5_params import MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES, NP_TO_MXTYPES, miCOMPRESSED, miMATRIX, miINT8, miUTF8, miUINT32, mxCELL_CLASS, mxSTRUCT_CLASS, mxOBJECT_CLASS, mxCHAR_CLASS, mxSPARSE_CLASS, mxDOUBLE_CLASS, mclass_info

        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.io.matlab.mio5_params', None, module_type_store, ['MatlabObject', 'MatlabFunction', 'MDTYPES', 'NP_TO_MTYPES', 'NP_TO_MXTYPES', 'miCOMPRESSED', 'miMATRIX', 'miINT8', 'miUTF8', 'miUINT32', 'mxCELL_CLASS', 'mxSTRUCT_CLASS', 'mxOBJECT_CLASS', 'mxCHAR_CLASS', 'mxSPARSE_CLASS', 'mxDOUBLE_CLASS', 'mclass_info'], [MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES, NP_TO_MXTYPES, miCOMPRESSED, miMATRIX, miINT8, miUTF8, miUINT32, mxCELL_CLASS, mxSTRUCT_CLASS, mxOBJECT_CLASS, mxCHAR_CLASS, mxSPARSE_CLASS, mxDOUBLE_CLASS, mclass_info])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio5_params' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.io.matlab.mio5_params', import_135283)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 107, 0))

# 'from scipy.io.matlab.streams import ZlibInputStream' statement (line 107)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_135285 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.matlab.streams')

if (type(import_135285) is not StypyTypeError):

    if (import_135285 != 'pyd_module'):
        __import__(import_135285)
        sys_modules_135286 = sys.modules[import_135285]
        import_from_module(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.matlab.streams', sys_modules_135286.module_type_store, module_type_store, ['ZlibInputStream'])
        nest_module(stypy.reporting.localization.Localization(__file__, 107, 0), __file__, sys_modules_135286, sys_modules_135286.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.streams import ZlibInputStream

        import_from_module(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.matlab.streams', None, module_type_store, ['ZlibInputStream'], [ZlibInputStream])

else:
    # Assigning a type to the variable 'scipy.io.matlab.streams' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'scipy.io.matlab.streams', import_135285)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

# Declaration of the 'MatFile5Reader' class
# Getting the type of 'MatFileReader' (line 110)
MatFileReader_135287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'MatFileReader')

class MatFile5Reader(MatFileReader_135287, ):
    str_135288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', ' Reader for Mat 5 mat files\n    Adds the following attribute to base class\n\n    uint16_codec - char codec to use for uint16 char arrays\n        (defaults to system default codec)\n\n    Uses variable reader that has the following stardard interface (see\n    abstract class in ``miobase``::\n\n       __init__(self, file_reader)\n       read_header(self)\n       array_from_header(self)\n\n    and added interface::\n\n       set_stream(self, stream)\n       read_full_tag(self)\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 133)
        None_135289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'None')
        # Getting the type of 'False' (line 134)
        False_135290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'False')
        # Getting the type of 'False' (line 135)
        False_135291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'False')
        # Getting the type of 'True' (line 136)
        True_135292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 34), 'True')
        # Getting the type of 'False' (line 137)
        False_135293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 35), 'False')
        # Getting the type of 'True' (line 138)
        True_135294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'True')
        # Getting the type of 'True' (line 139)
        True_135295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'True')
        # Getting the type of 'None' (line 140)
        None_135296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'None')
        defaults = [None_135289, False_135290, False_135291, True_135292, False_135293, True_135294, True_135295, None_135296]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.__init__', ['mat_stream', 'byte_order', 'mat_dtype', 'squeeze_me', 'chars_as_strings', 'matlab_compatible', 'struct_as_record', 'verify_compressed_data_integrity', 'uint16_codec'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['mat_stream', 'byte_order', 'mat_dtype', 'squeeze_me', 'chars_as_strings', 'matlab_compatible', 'struct_as_record', 'verify_compressed_data_integrity', 'uint16_codec'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_135297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, (-1)), 'str', "Initializer for matlab 5 file format reader\n\n    %(matstream_arg)s\n    %(load_args)s\n    %(struct_arg)s\n    uint16_codec : {None, string}\n        Set codec to use for uint16 char arrays (e.g. 'utf-8').\n        Use system default codec if None\n        ")
        
        # Call to __init__(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'mat_stream' (line 152)
        mat_stream_135304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'mat_stream', False)
        # Getting the type of 'byte_order' (line 153)
        byte_order_135305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'byte_order', False)
        # Getting the type of 'mat_dtype' (line 154)
        mat_dtype_135306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'mat_dtype', False)
        # Getting the type of 'squeeze_me' (line 155)
        squeeze_me_135307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'squeeze_me', False)
        # Getting the type of 'chars_as_strings' (line 156)
        chars_as_strings_135308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'chars_as_strings', False)
        # Getting the type of 'matlab_compatible' (line 157)
        matlab_compatible_135309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'matlab_compatible', False)
        # Getting the type of 'struct_as_record' (line 158)
        struct_as_record_135310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'struct_as_record', False)
        # Getting the type of 'verify_compressed_data_integrity' (line 159)
        verify_compressed_data_integrity_135311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'verify_compressed_data_integrity', False)
        # Processing the call keyword arguments (line 151)
        kwargs_135312 = {}
        
        # Call to super(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'MatFile5Reader' (line 151)
        MatFile5Reader_135299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'MatFile5Reader', False)
        # Getting the type of 'self' (line 151)
        self_135300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'self', False)
        # Processing the call keyword arguments (line 151)
        kwargs_135301 = {}
        # Getting the type of 'super' (line 151)
        super_135298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'super', False)
        # Calling super(args, kwargs) (line 151)
        super_call_result_135302 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), super_135298, *[MatFile5Reader_135299, self_135300], **kwargs_135301)
        
        # Obtaining the member '__init__' of a type (line 151)
        init___135303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), super_call_result_135302, '__init__')
        # Calling __init__(args, kwargs) (line 151)
        init___call_result_135313 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), init___135303, *[mat_stream_135304, byte_order_135305, mat_dtype_135306, squeeze_me_135307, chars_as_strings_135308, matlab_compatible_135309, struct_as_record_135310, verify_compressed_data_integrity_135311], **kwargs_135312)
        
        
        
        # Getting the type of 'uint16_codec' (line 162)
        uint16_codec_135314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'uint16_codec')
        # Applying the 'not' unary operator (line 162)
        result_not__135315 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), 'not', uint16_codec_135314)
        
        # Testing the type of an if condition (line 162)
        if_condition_135316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), result_not__135315)
        # Assigning a type to the variable 'if_condition_135316' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_135316', if_condition_135316)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to getdefaultencoding(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_135319 = {}
        # Getting the type of 'sys' (line 163)
        sys_135317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'sys', False)
        # Obtaining the member 'getdefaultencoding' of a type (line 163)
        getdefaultencoding_135318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 27), sys_135317, 'getdefaultencoding')
        # Calling getdefaultencoding(args, kwargs) (line 163)
        getdefaultencoding_call_result_135320 = invoke(stypy.reporting.localization.Localization(__file__, 163, 27), getdefaultencoding_135318, *[], **kwargs_135319)
        
        # Assigning a type to the variable 'uint16_codec' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'uint16_codec', getdefaultencoding_call_result_135320)
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 164):
        
        # Assigning a Name to a Attribute (line 164):
        # Getting the type of 'uint16_codec' (line 164)
        uint16_codec_135321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'uint16_codec')
        # Getting the type of 'self' (line 164)
        self_135322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self')
        # Setting the type of the member 'uint16_codec' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_135322, 'uint16_codec', uint16_codec_135321)
        
        # Assigning a Name to a Attribute (line 166):
        
        # Assigning a Name to a Attribute (line 166):
        # Getting the type of 'None' (line 166)
        None_135323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'None')
        # Getting the type of 'self' (line 166)
        self_135324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member '_file_reader' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_135324, '_file_reader', None_135323)
        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'None' (line 167)
        None_135325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'None')
        # Getting the type of 'self' (line 167)
        self_135326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member '_matrix_reader' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_135326, '_matrix_reader', None_135325)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def guess_byte_order(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'guess_byte_order'
        module_type_store = module_type_store.open_function_context('guess_byte_order', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_function_name', 'MatFile5Reader.guess_byte_order')
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Reader.guess_byte_order.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.guess_byte_order', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'guess_byte_order', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'guess_byte_order(...)' code ##################

        str_135327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', ' Guess byte order.\n        Sets stream pointer to 0 ')
        
        # Call to seek(...): (line 172)
        # Processing the call arguments (line 172)
        int_135331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'int')
        # Processing the call keyword arguments (line 172)
        kwargs_135332 = {}
        # Getting the type of 'self' (line 172)
        self_135328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 172)
        mat_stream_135329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_135328, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 172)
        seek_135330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), mat_stream_135329, 'seek')
        # Calling seek(args, kwargs) (line 172)
        seek_call_result_135333 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), seek_135330, *[int_135331], **kwargs_135332)
        
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to read(...): (line 173)
        # Processing the call arguments (line 173)
        int_135337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 34), 'int')
        # Processing the call keyword arguments (line 173)
        kwargs_135338 = {}
        # Getting the type of 'self' (line 173)
        self_135334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 173)
        mat_stream_135335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 13), self_135334, 'mat_stream')
        # Obtaining the member 'read' of a type (line 173)
        read_135336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 13), mat_stream_135335, 'read')
        # Calling read(args, kwargs) (line 173)
        read_call_result_135339 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), read_135336, *[int_135337], **kwargs_135338)
        
        # Assigning a type to the variable 'mi' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'mi', read_call_result_135339)
        
        # Call to seek(...): (line 174)
        # Processing the call arguments (line 174)
        int_135343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'int')
        # Processing the call keyword arguments (line 174)
        kwargs_135344 = {}
        # Getting the type of 'self' (line 174)
        self_135340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 174)
        mat_stream_135341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_135340, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 174)
        seek_135342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), mat_stream_135341, 'seek')
        # Calling seek(args, kwargs) (line 174)
        seek_call_result_135345 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), seek_135342, *[int_135343], **kwargs_135344)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mi' (line 175)
        mi_135346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'mi')
        str_135347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 21), 'str', 'IM')
        # Applying the binary operator '==' (line 175)
        result_eq_135348 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 15), '==', mi_135346, str_135347)
        
        str_135349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 31), 'str', '<')
        # Applying the binary operator 'and' (line 175)
        result_and_keyword_135350 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 15), 'and', result_eq_135348, str_135349)
        
        str_135351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 38), 'str', '>')
        # Applying the binary operator 'or' (line 175)
        result_or_keyword_135352 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 15), 'or', result_and_keyword_135350, str_135351)
        
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', result_or_keyword_135352)
        
        # ################# End of 'guess_byte_order(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'guess_byte_order' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_135353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'guess_byte_order'
        return stypy_return_type_135353


    @norecursion
    def read_file_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_file_header'
        module_type_store = module_type_store.open_function_context('read_file_header', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_function_name', 'MatFile5Reader.read_file_header')
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Reader.read_file_header.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.read_file_header', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_file_header', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_file_header(...)' code ##################

        str_135354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 8), 'str', ' Read in mat 5 file header ')
        
        # Assigning a Dict to a Name (line 179):
        
        # Assigning a Dict to a Name (line 179):
        
        # Obtaining an instance of the builtin type 'dict' (line 179)
        dict_135355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 179)
        
        # Assigning a type to the variable 'hdict' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'hdict', dict_135355)
        
        # Assigning a Subscript to a Name (line 180):
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        str_135356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 55), 'str', 'file_header')
        
        # Obtaining the type of the subscript
        str_135357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 45), 'str', 'dtypes')
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 180)
        self_135358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'self')
        # Obtaining the member 'byte_order' of a type (line 180)
        byte_order_135359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 28), self_135358, 'byte_order')
        # Getting the type of 'MDTYPES' (line 180)
        MDTYPES_135360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'MDTYPES')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___135361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), MDTYPES_135360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_135362 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), getitem___135361, byte_order_135359)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___135363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), subscript_call_result_135362, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_135364 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), getitem___135363, str_135357)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___135365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), subscript_call_result_135364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_135366 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), getitem___135365, str_135356)
        
        # Assigning a type to the variable 'hdr_dtype' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'hdr_dtype', subscript_call_result_135366)
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to read_dtype(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'self' (line 181)
        self_135368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 181)
        mat_stream_135369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 25), self_135368, 'mat_stream')
        # Getting the type of 'hdr_dtype' (line 181)
        hdr_dtype_135370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'hdr_dtype', False)
        # Processing the call keyword arguments (line 181)
        kwargs_135371 = {}
        # Getting the type of 'read_dtype' (line 181)
        read_dtype_135367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'read_dtype', False)
        # Calling read_dtype(args, kwargs) (line 181)
        read_dtype_call_result_135372 = invoke(stypy.reporting.localization.Localization(__file__, 181, 14), read_dtype_135367, *[mat_stream_135369, hdr_dtype_135370], **kwargs_135371)
        
        # Assigning a type to the variable 'hdr' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'hdr', read_dtype_call_result_135372)
        
        # Assigning a Call to a Subscript (line 182):
        
        # Assigning a Call to a Subscript (line 182):
        
        # Call to strip(...): (line 182)
        # Processing the call arguments (line 182)
        str_135381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 62), 'str', ' \t\n\x00')
        # Processing the call keyword arguments (line 182)
        kwargs_135382 = {}
        
        # Call to item(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_135378 = {}
        
        # Obtaining the type of the subscript
        str_135373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 34), 'str', 'description')
        # Getting the type of 'hdr' (line 182)
        hdr_135374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'hdr', False)
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___135375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 30), hdr_135374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_135376 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), getitem___135375, str_135373)
        
        # Obtaining the member 'item' of a type (line 182)
        item_135377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 30), subscript_call_result_135376, 'item')
        # Calling item(args, kwargs) (line 182)
        item_call_result_135379 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), item_135377, *[], **kwargs_135378)
        
        # Obtaining the member 'strip' of a type (line 182)
        strip_135380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 30), item_call_result_135379, 'strip')
        # Calling strip(args, kwargs) (line 182)
        strip_call_result_135383 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), strip_135380, *[str_135381], **kwargs_135382)
        
        # Getting the type of 'hdict' (line 182)
        hdict_135384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'hdict')
        str_135385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 14), 'str', '__header__')
        # Storing an element on a container (line 182)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), hdict_135384, (str_135385, strip_call_result_135383))
        
        # Assigning a BinOp to a Name (line 183):
        
        # Assigning a BinOp to a Name (line 183):
        
        # Obtaining the type of the subscript
        str_135386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 22), 'str', 'version')
        # Getting the type of 'hdr' (line 183)
        hdr_135387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'hdr')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___135388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 18), hdr_135387, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_135389 = invoke(stypy.reporting.localization.Localization(__file__, 183, 18), getitem___135388, str_135386)
        
        int_135390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'int')
        # Applying the binary operator '>>' (line 183)
        result_rshift_135391 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 18), '>>', subscript_call_result_135389, int_135390)
        
        # Assigning a type to the variable 'v_major' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'v_major', result_rshift_135391)
        
        # Assigning a BinOp to a Name (line 184):
        
        # Assigning a BinOp to a Name (line 184):
        
        # Obtaining the type of the subscript
        str_135392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 22), 'str', 'version')
        # Getting the type of 'hdr' (line 184)
        hdr_135393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'hdr')
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___135394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), hdr_135393, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_135395 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), getitem___135394, str_135392)
        
        int_135396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 35), 'int')
        # Applying the binary operator '&' (line 184)
        result_and__135397 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 18), '&', subscript_call_result_135395, int_135396)
        
        # Assigning a type to the variable 'v_minor' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'v_minor', result_and__135397)
        
        # Assigning a BinOp to a Subscript (line 185):
        
        # Assigning a BinOp to a Subscript (line 185):
        str_135398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 31), 'str', '%d.%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_135399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        # Getting the type of 'v_major' (line 185)
        v_major_135400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 42), 'v_major')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), tuple_135399, v_major_135400)
        # Adding element type (line 185)
        # Getting the type of 'v_minor' (line 185)
        v_minor_135401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 51), 'v_minor')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 42), tuple_135399, v_minor_135401)
        
        # Applying the binary operator '%' (line 185)
        result_mod_135402 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 31), '%', str_135398, tuple_135399)
        
        # Getting the type of 'hdict' (line 185)
        hdict_135403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'hdict')
        str_135404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 14), 'str', '__version__')
        # Storing an element on a container (line 185)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 8), hdict_135403, (str_135404, result_mod_135402))
        # Getting the type of 'hdict' (line 186)
        hdict_135405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'hdict')
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', hdict_135405)
        
        # ################# End of 'read_file_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_file_header' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_135406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_file_header'
        return stypy_return_type_135406


    @norecursion
    def initialize_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_read'
        module_type_store = module_type_store.open_function_context('initialize_read', 188, 4, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_function_name', 'MatFile5Reader.initialize_read')
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Reader.initialize_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.initialize_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize_read(...)' code ##################

        str_135407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, (-1)), 'str', ' Run when beginning read of variables\n\n        Sets up readers from parameters in `self`\n        ')
        
        # Assigning a Call to a Attribute (line 196):
        
        # Assigning a Call to a Attribute (line 196):
        
        # Call to VarReader5(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'self' (line 196)
        self_135409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'self', False)
        # Processing the call keyword arguments (line 196)
        kwargs_135410 = {}
        # Getting the type of 'VarReader5' (line 196)
        VarReader5_135408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'VarReader5', False)
        # Calling VarReader5(args, kwargs) (line 196)
        VarReader5_call_result_135411 = invoke(stypy.reporting.localization.Localization(__file__, 196, 28), VarReader5_135408, *[self_135409], **kwargs_135410)
        
        # Getting the type of 'self' (line 196)
        self_135412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member '_file_reader' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_135412, '_file_reader', VarReader5_call_result_135411)
        
        # Assigning a Call to a Attribute (line 198):
        
        # Assigning a Call to a Attribute (line 198):
        
        # Call to VarReader5(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'self' (line 198)
        self_135414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 41), 'self', False)
        # Processing the call keyword arguments (line 198)
        kwargs_135415 = {}
        # Getting the type of 'VarReader5' (line 198)
        VarReader5_135413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'VarReader5', False)
        # Calling VarReader5(args, kwargs) (line 198)
        VarReader5_call_result_135416 = invoke(stypy.reporting.localization.Localization(__file__, 198, 30), VarReader5_135413, *[self_135414], **kwargs_135415)
        
        # Getting the type of 'self' (line 198)
        self_135417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self')
        # Setting the type of the member '_matrix_reader' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_135417, '_matrix_reader', VarReader5_call_result_135416)
        
        # ################# End of 'initialize_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_read' in the type store
        # Getting the type of 'stypy_return_type' (line 188)
        stypy_return_type_135418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_read'
        return stypy_return_type_135418


    @norecursion
    def read_var_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_var_header'
        module_type_store = module_type_store.open_function_context('read_var_header', 200, 4, False)
        # Assigning a type to the variable 'self' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_function_name', 'MatFile5Reader.read_var_header')
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Reader.read_var_header.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.read_var_header', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_var_header', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_var_header(...)' code ##################

        str_135419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, (-1)), 'str', ' Read header, return header, next position\n\n        Header has to define at least .name and .is_global\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        header : object\n           object that can be passed to self.read_var_array, and that\n           has attributes .name and .is_global\n        next_position : int\n           position in stream of next variable\n        ')
        
        # Assigning a Call to a Tuple (line 217):
        
        # Assigning a Subscript to a Name (line 217):
        
        # Obtaining the type of the subscript
        int_135420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
        
        # Call to read_full_tag(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_135424 = {}
        # Getting the type of 'self' (line 217)
        self_135421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'self', False)
        # Obtaining the member '_file_reader' of a type (line 217)
        _file_reader_135422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 29), self_135421, '_file_reader')
        # Obtaining the member 'read_full_tag' of a type (line 217)
        read_full_tag_135423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 29), _file_reader_135422, 'read_full_tag')
        # Calling read_full_tag(args, kwargs) (line 217)
        read_full_tag_call_result_135425 = invoke(stypy.reporting.localization.Localization(__file__, 217, 29), read_full_tag_135423, *[], **kwargs_135424)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___135426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), read_full_tag_call_result_135425, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_135427 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), getitem___135426, int_135420)
        
        # Assigning a type to the variable 'tuple_var_assignment_135257' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_135257', subscript_call_result_135427)
        
        # Assigning a Subscript to a Name (line 217):
        
        # Obtaining the type of the subscript
        int_135428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
        
        # Call to read_full_tag(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_135432 = {}
        # Getting the type of 'self' (line 217)
        self_135429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'self', False)
        # Obtaining the member '_file_reader' of a type (line 217)
        _file_reader_135430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 29), self_135429, '_file_reader')
        # Obtaining the member 'read_full_tag' of a type (line 217)
        read_full_tag_135431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 29), _file_reader_135430, 'read_full_tag')
        # Calling read_full_tag(args, kwargs) (line 217)
        read_full_tag_call_result_135433 = invoke(stypy.reporting.localization.Localization(__file__, 217, 29), read_full_tag_135431, *[], **kwargs_135432)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___135434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), read_full_tag_call_result_135433, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_135435 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), getitem___135434, int_135428)
        
        # Assigning a type to the variable 'tuple_var_assignment_135258' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_135258', subscript_call_result_135435)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'tuple_var_assignment_135257' (line 217)
        tuple_var_assignment_135257_135436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_135257')
        # Assigning a type to the variable 'mdtype' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'mdtype', tuple_var_assignment_135257_135436)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'tuple_var_assignment_135258' (line 217)
        tuple_var_assignment_135258_135437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_135258')
        # Assigning a type to the variable 'byte_count' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'byte_count', tuple_var_assignment_135258_135437)
        
        
        
        # Getting the type of 'byte_count' (line 218)
        byte_count_135438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'byte_count')
        int_135439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 28), 'int')
        # Applying the binary operator '>' (line 218)
        result_gt_135440 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 15), '>', byte_count_135438, int_135439)
        
        # Applying the 'not' unary operator (line 218)
        result_not__135441 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'not', result_gt_135440)
        
        # Testing the type of an if condition (line 218)
        if_condition_135442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_not__135441)
        # Assigning a type to the variable 'if_condition_135442' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_135442', if_condition_135442)
        # SSA begins for if statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 219)
        # Processing the call arguments (line 219)
        str_135444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 29), 'str', 'Did not read any bytes')
        # Processing the call keyword arguments (line 219)
        kwargs_135445 = {}
        # Getting the type of 'ValueError' (line 219)
        ValueError_135443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 219)
        ValueError_call_result_135446 = invoke(stypy.reporting.localization.Localization(__file__, 219, 18), ValueError_135443, *[str_135444], **kwargs_135445)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 219, 12), ValueError_call_result_135446, 'raise parameter', BaseException)
        # SSA join for if statement (line 218)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 220):
        
        # Assigning a BinOp to a Name (line 220):
        
        # Call to tell(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_135450 = {}
        # Getting the type of 'self' (line 220)
        self_135447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 220)
        mat_stream_135448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), self_135447, 'mat_stream')
        # Obtaining the member 'tell' of a type (line 220)
        tell_135449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), mat_stream_135448, 'tell')
        # Calling tell(args, kwargs) (line 220)
        tell_call_result_135451 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), tell_135449, *[], **kwargs_135450)
        
        # Getting the type of 'byte_count' (line 220)
        byte_count_135452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'byte_count')
        # Applying the binary operator '+' (line 220)
        result_add_135453 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 19), '+', tell_call_result_135451, byte_count_135452)
        
        # Assigning a type to the variable 'next_pos' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'next_pos', result_add_135453)
        
        
        # Getting the type of 'mdtype' (line 221)
        mdtype_135454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'mdtype')
        # Getting the type of 'miCOMPRESSED' (line 221)
        miCOMPRESSED_135455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'miCOMPRESSED')
        # Applying the binary operator '==' (line 221)
        result_eq_135456 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 11), '==', mdtype_135454, miCOMPRESSED_135455)
        
        # Testing the type of an if condition (line 221)
        if_condition_135457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), result_eq_135456)
        # Assigning a type to the variable 'if_condition_135457' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_135457', if_condition_135457)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to ZlibInputStream(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_135459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 37), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 223)
        mat_stream_135460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 37), self_135459, 'mat_stream')
        # Getting the type of 'byte_count' (line 223)
        byte_count_135461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 54), 'byte_count', False)
        # Processing the call keyword arguments (line 223)
        kwargs_135462 = {}
        # Getting the type of 'ZlibInputStream' (line 223)
        ZlibInputStream_135458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'ZlibInputStream', False)
        # Calling ZlibInputStream(args, kwargs) (line 223)
        ZlibInputStream_call_result_135463 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), ZlibInputStream_135458, *[mat_stream_135460, byte_count_135461], **kwargs_135462)
        
        # Assigning a type to the variable 'stream' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'stream', ZlibInputStream_call_result_135463)
        
        # Call to set_stream(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'stream' (line 224)
        stream_135467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'stream', False)
        # Processing the call keyword arguments (line 224)
        kwargs_135468 = {}
        # Getting the type of 'self' (line 224)
        self_135464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 224)
        _matrix_reader_135465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), self_135464, '_matrix_reader')
        # Obtaining the member 'set_stream' of a type (line 224)
        set_stream_135466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), _matrix_reader_135465, 'set_stream')
        # Calling set_stream(args, kwargs) (line 224)
        set_stream_call_result_135469 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), set_stream_135466, *[stream_135467], **kwargs_135468)
        
        
        # Assigning a Attribute to a Name (line 225):
        
        # Assigning a Attribute to a Name (line 225):
        # Getting the type of 'self' (line 225)
        self_135470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 33), 'self')
        # Obtaining the member 'verify_compressed_data_integrity' of a type (line 225)
        verify_compressed_data_integrity_135471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 33), self_135470, 'verify_compressed_data_integrity')
        # Assigning a type to the variable 'check_stream_limit' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'check_stream_limit', verify_compressed_data_integrity_135471)
        
        # Assigning a Call to a Tuple (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_135472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 12), 'int')
        
        # Call to read_full_tag(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_135476 = {}
        # Getting the type of 'self' (line 226)
        self_135473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 33), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 226)
        _matrix_reader_135474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 33), self_135473, '_matrix_reader')
        # Obtaining the member 'read_full_tag' of a type (line 226)
        read_full_tag_135475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 33), _matrix_reader_135474, 'read_full_tag')
        # Calling read_full_tag(args, kwargs) (line 226)
        read_full_tag_call_result_135477 = invoke(stypy.reporting.localization.Localization(__file__, 226, 33), read_full_tag_135475, *[], **kwargs_135476)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___135478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), read_full_tag_call_result_135477, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_135479 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), getitem___135478, int_135472)
        
        # Assigning a type to the variable 'tuple_var_assignment_135259' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'tuple_var_assignment_135259', subscript_call_result_135479)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_135480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 12), 'int')
        
        # Call to read_full_tag(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_135484 = {}
        # Getting the type of 'self' (line 226)
        self_135481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 33), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 226)
        _matrix_reader_135482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 33), self_135481, '_matrix_reader')
        # Obtaining the member 'read_full_tag' of a type (line 226)
        read_full_tag_135483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 33), _matrix_reader_135482, 'read_full_tag')
        # Calling read_full_tag(args, kwargs) (line 226)
        read_full_tag_call_result_135485 = invoke(stypy.reporting.localization.Localization(__file__, 226, 33), read_full_tag_135483, *[], **kwargs_135484)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___135486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), read_full_tag_call_result_135485, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_135487 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), getitem___135486, int_135480)
        
        # Assigning a type to the variable 'tuple_var_assignment_135260' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'tuple_var_assignment_135260', subscript_call_result_135487)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_135259' (line 226)
        tuple_var_assignment_135259_135488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'tuple_var_assignment_135259')
        # Assigning a type to the variable 'mdtype' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'mdtype', tuple_var_assignment_135259_135488)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_135260' (line 226)
        tuple_var_assignment_135260_135489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'tuple_var_assignment_135260')
        # Assigning a type to the variable 'byte_count' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'byte_count', tuple_var_assignment_135260_135489)
        # SSA branch for the else part of an if statement (line 221)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 228):
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'False' (line 228)
        False_135490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 33), 'False')
        # Assigning a type to the variable 'check_stream_limit' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'check_stream_limit', False_135490)
        
        # Call to set_stream(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'self' (line 229)
        self_135494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 43), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 229)
        mat_stream_135495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 43), self_135494, 'mat_stream')
        # Processing the call keyword arguments (line 229)
        kwargs_135496 = {}
        # Getting the type of 'self' (line 229)
        self_135491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 229)
        _matrix_reader_135492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), self_135491, '_matrix_reader')
        # Obtaining the member 'set_stream' of a type (line 229)
        set_stream_135493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), _matrix_reader_135492, 'set_stream')
        # Calling set_stream(args, kwargs) (line 229)
        set_stream_call_result_135497 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), set_stream_135493, *[mat_stream_135495], **kwargs_135496)
        
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Getting the type of 'mdtype' (line 230)
        mdtype_135498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'mdtype')
        # Getting the type of 'miMATRIX' (line 230)
        miMATRIX_135499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'miMATRIX')
        # Applying the binary operator '==' (line 230)
        result_eq_135500 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 15), '==', mdtype_135498, miMATRIX_135499)
        
        # Applying the 'not' unary operator (line 230)
        result_not__135501 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), 'not', result_eq_135500)
        
        # Testing the type of an if condition (line 230)
        if_condition_135502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_not__135501)
        # Assigning a type to the variable 'if_condition_135502' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_135502', if_condition_135502)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 231)
        # Processing the call arguments (line 231)
        str_135504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'str', 'Expecting miMATRIX type here, got %d')
        # Getting the type of 'mdtype' (line 231)
        mdtype_135505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 69), 'mdtype', False)
        # Applying the binary operator '%' (line 231)
        result_mod_135506 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 28), '%', str_135504, mdtype_135505)
        
        # Processing the call keyword arguments (line 231)
        kwargs_135507 = {}
        # Getting the type of 'TypeError' (line 231)
        TypeError_135503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 231)
        TypeError_call_result_135508 = invoke(stypy.reporting.localization.Localization(__file__, 231, 18), TypeError_135503, *[result_mod_135506], **kwargs_135507)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 231, 12), TypeError_call_result_135508, 'raise parameter', BaseException)
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to read_header(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'check_stream_limit' (line 232)
        check_stream_limit_135512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'check_stream_limit', False)
        # Processing the call keyword arguments (line 232)
        kwargs_135513 = {}
        # Getting the type of 'self' (line 232)
        self_135509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 232)
        _matrix_reader_135510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 17), self_135509, '_matrix_reader')
        # Obtaining the member 'read_header' of a type (line 232)
        read_header_135511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 17), _matrix_reader_135510, 'read_header')
        # Calling read_header(args, kwargs) (line 232)
        read_header_call_result_135514 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), read_header_135511, *[check_stream_limit_135512], **kwargs_135513)
        
        # Assigning a type to the variable 'header' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'header', read_header_call_result_135514)
        
        # Obtaining an instance of the builtin type 'tuple' (line 233)
        tuple_135515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 233)
        # Adding element type (line 233)
        # Getting the type of 'header' (line 233)
        header_135516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'header')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 15), tuple_135515, header_135516)
        # Adding element type (line 233)
        # Getting the type of 'next_pos' (line 233)
        next_pos_135517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 23), 'next_pos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 15), tuple_135515, next_pos_135517)
        
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'stypy_return_type', tuple_135515)
        
        # ################# End of 'read_var_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_var_header' in the type store
        # Getting the type of 'stypy_return_type' (line 200)
        stypy_return_type_135518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_var_header'
        return stypy_return_type_135518


    @norecursion
    def read_var_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 235)
        True_135519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 45), 'True')
        defaults = [True_135519]
        # Create a new context for function 'read_var_array'
        module_type_store = module_type_store.open_function_context('read_var_array', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_function_name', 'MatFile5Reader.read_var_array')
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_param_names_list', ['header', 'process'])
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Reader.read_var_array.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.read_var_array', ['header', 'process'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_var_array', localization, ['header', 'process'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_var_array(...)' code ##################

        str_135520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, (-1)), 'str', ' Read array, given `header`\n\n        Parameters\n        ----------\n        header : header object\n           object with fields defining variable header\n        process : {True, False} bool, optional\n           If True, apply recursive post-processing during loading of\n           array.\n\n        Returns\n        -------\n        arr : array\n           array with post-processing applied or not according to\n           `process`.\n        ')
        
        # Call to array_from_header(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'header' (line 252)
        header_135524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 53), 'header', False)
        # Getting the type of 'process' (line 252)
        process_135525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 61), 'process', False)
        # Processing the call keyword arguments (line 252)
        kwargs_135526 = {}
        # Getting the type of 'self' (line 252)
        self_135521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 252)
        _matrix_reader_135522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 15), self_135521, '_matrix_reader')
        # Obtaining the member 'array_from_header' of a type (line 252)
        array_from_header_135523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 15), _matrix_reader_135522, 'array_from_header')
        # Calling array_from_header(args, kwargs) (line 252)
        array_from_header_call_result_135527 = invoke(stypy.reporting.localization.Localization(__file__, 252, 15), array_from_header_135523, *[header_135524, process_135525], **kwargs_135526)
        
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'stypy_return_type', array_from_header_call_result_135527)
        
        # ################# End of 'read_var_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_var_array' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_135528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_var_array'
        return stypy_return_type_135528


    @norecursion
    def get_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 254)
        None_135529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 43), 'None')
        defaults = [None_135529]
        # Create a new context for function 'get_variables'
        module_type_store = module_type_store.open_function_context('get_variables', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_function_name', 'MatFile5Reader.get_variables')
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_param_names_list', ['variable_names'])
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Reader.get_variables.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.get_variables', ['variable_names'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_variables', localization, ['variable_names'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_variables(...)' code ##################

        str_135530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'str', ' get variables from stream as dictionary\n\n        variable_names   - optional list of variable names to get\n\n        If variable_names is None, then get all variables in file\n        ')
        
        
        # Call to isinstance(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'variable_names' (line 261)
        variable_names_135532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'variable_names', False)
        # Getting the type of 'string_types' (line 261)
        string_types_135533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 38), 'string_types', False)
        # Processing the call keyword arguments (line 261)
        kwargs_135534 = {}
        # Getting the type of 'isinstance' (line 261)
        isinstance_135531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 261)
        isinstance_call_result_135535 = invoke(stypy.reporting.localization.Localization(__file__, 261, 11), isinstance_135531, *[variable_names_135532, string_types_135533], **kwargs_135534)
        
        # Testing the type of an if condition (line 261)
        if_condition_135536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), isinstance_call_result_135535)
        # Assigning a type to the variable 'if_condition_135536' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_135536', if_condition_135536)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 262):
        
        # Assigning a List to a Name (line 262):
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_135537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        # Adding element type (line 262)
        # Getting the type of 'variable_names' (line 262)
        variable_names_135538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'variable_names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 29), list_135537, variable_names_135538)
        
        # Assigning a type to the variable 'variable_names' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'variable_names', list_135537)
        # SSA branch for the else part of an if statement (line 261)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 263)
        # Getting the type of 'variable_names' (line 263)
        variable_names_135539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'variable_names')
        # Getting the type of 'None' (line 263)
        None_135540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 35), 'None')
        
        (may_be_135541, more_types_in_union_135542) = may_not_be_none(variable_names_135539, None_135540)

        if may_be_135541:

            if more_types_in_union_135542:
                # Runtime conditional SSA (line 263)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 264):
            
            # Assigning a Call to a Name (line 264):
            
            # Call to list(...): (line 264)
            # Processing the call arguments (line 264)
            # Getting the type of 'variable_names' (line 264)
            variable_names_135544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 34), 'variable_names', False)
            # Processing the call keyword arguments (line 264)
            kwargs_135545 = {}
            # Getting the type of 'list' (line 264)
            list_135543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'list', False)
            # Calling list(args, kwargs) (line 264)
            list_call_result_135546 = invoke(stypy.reporting.localization.Localization(__file__, 264, 29), list_135543, *[variable_names_135544], **kwargs_135545)
            
            # Assigning a type to the variable 'variable_names' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'variable_names', list_call_result_135546)

            if more_types_in_union_135542:
                # SSA join for if statement (line 263)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to seek(...): (line 266)
        # Processing the call arguments (line 266)
        int_135550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 29), 'int')
        # Processing the call keyword arguments (line 266)
        kwargs_135551 = {}
        # Getting the type of 'self' (line 266)
        self_135547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 266)
        mat_stream_135548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_135547, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 266)
        seek_135549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), mat_stream_135548, 'seek')
        # Calling seek(args, kwargs) (line 266)
        seek_call_result_135552 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), seek_135549, *[int_135550], **kwargs_135551)
        
        
        # Call to initialize_read(...): (line 268)
        # Processing the call keyword arguments (line 268)
        kwargs_135555 = {}
        # Getting the type of 'self' (line 268)
        self_135553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self', False)
        # Obtaining the member 'initialize_read' of a type (line 268)
        initialize_read_135554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_135553, 'initialize_read')
        # Calling initialize_read(args, kwargs) (line 268)
        initialize_read_call_result_135556 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), initialize_read_135554, *[], **kwargs_135555)
        
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to read_file_header(...): (line 269)
        # Processing the call keyword arguments (line 269)
        kwargs_135559 = {}
        # Getting the type of 'self' (line 269)
        self_135557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'self', False)
        # Obtaining the member 'read_file_header' of a type (line 269)
        read_file_header_135558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), self_135557, 'read_file_header')
        # Calling read_file_header(args, kwargs) (line 269)
        read_file_header_call_result_135560 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), read_file_header_135558, *[], **kwargs_135559)
        
        # Assigning a type to the variable 'mdict' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'mdict', read_file_header_call_result_135560)
        
        # Assigning a List to a Subscript (line 270):
        
        # Assigning a List to a Subscript (line 270):
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_135561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        
        # Getting the type of 'mdict' (line 270)
        mdict_135562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'mdict')
        str_135563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 14), 'str', '__globals__')
        # Storing an element on a container (line 270)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), mdict_135562, (str_135563, list_135561))
        
        
        
        # Call to end_of_stream(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_135566 = {}
        # Getting the type of 'self' (line 271)
        self_135564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'self', False)
        # Obtaining the member 'end_of_stream' of a type (line 271)
        end_of_stream_135565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 18), self_135564, 'end_of_stream')
        # Calling end_of_stream(args, kwargs) (line 271)
        end_of_stream_call_result_135567 = invoke(stypy.reporting.localization.Localization(__file__, 271, 18), end_of_stream_135565, *[], **kwargs_135566)
        
        # Applying the 'not' unary operator (line 271)
        result_not__135568 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 14), 'not', end_of_stream_call_result_135567)
        
        # Testing the type of an if condition (line 271)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 8), result_not__135568)
        # SSA begins for while statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 272):
        
        # Assigning a Subscript to a Name (line 272):
        
        # Obtaining the type of the subscript
        int_135569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 12), 'int')
        
        # Call to read_var_header(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_135572 = {}
        # Getting the type of 'self' (line 272)
        self_135570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 272)
        read_var_header_135571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 33), self_135570, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 272)
        read_var_header_call_result_135573 = invoke(stypy.reporting.localization.Localization(__file__, 272, 33), read_var_header_135571, *[], **kwargs_135572)
        
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___135574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), read_var_header_call_result_135573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_135575 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), getitem___135574, int_135569)
        
        # Assigning a type to the variable 'tuple_var_assignment_135261' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'tuple_var_assignment_135261', subscript_call_result_135575)
        
        # Assigning a Subscript to a Name (line 272):
        
        # Obtaining the type of the subscript
        int_135576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 12), 'int')
        
        # Call to read_var_header(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_135579 = {}
        # Getting the type of 'self' (line 272)
        self_135577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 272)
        read_var_header_135578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 33), self_135577, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 272)
        read_var_header_call_result_135580 = invoke(stypy.reporting.localization.Localization(__file__, 272, 33), read_var_header_135578, *[], **kwargs_135579)
        
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___135581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), read_var_header_call_result_135580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_135582 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), getitem___135581, int_135576)
        
        # Assigning a type to the variable 'tuple_var_assignment_135262' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'tuple_var_assignment_135262', subscript_call_result_135582)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'tuple_var_assignment_135261' (line 272)
        tuple_var_assignment_135261_135583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'tuple_var_assignment_135261')
        # Assigning a type to the variable 'hdr' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'hdr', tuple_var_assignment_135261_135583)
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'tuple_var_assignment_135262' (line 272)
        tuple_var_assignment_135262_135584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'tuple_var_assignment_135262')
        # Assigning a type to the variable 'next_position' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'next_position', tuple_var_assignment_135262_135584)
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to asstr(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'hdr' (line 273)
        hdr_135586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'hdr', False)
        # Obtaining the member 'name' of a type (line 273)
        name_135587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 25), hdr_135586, 'name')
        # Processing the call keyword arguments (line 273)
        kwargs_135588 = {}
        # Getting the type of 'asstr' (line 273)
        asstr_135585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'asstr', False)
        # Calling asstr(args, kwargs) (line 273)
        asstr_call_result_135589 = invoke(stypy.reporting.localization.Localization(__file__, 273, 19), asstr_135585, *[name_135587], **kwargs_135588)
        
        # Assigning a type to the variable 'name' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'name', asstr_call_result_135589)
        
        
        # Getting the type of 'name' (line 274)
        name_135590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'name')
        # Getting the type of 'mdict' (line 274)
        mdict_135591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 23), 'mdict')
        # Applying the binary operator 'in' (line 274)
        result_contains_135592 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 15), 'in', name_135590, mdict_135591)
        
        # Testing the type of an if condition (line 274)
        if_condition_135593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 12), result_contains_135592)
        # Assigning a type to the variable 'if_condition_135593' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'if_condition_135593', if_condition_135593)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 275)
        # Processing the call arguments (line 275)
        str_135596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 30), 'str', 'Duplicate variable name "%s" in stream - replacing previous with new\nConsider mio5.varmats_from_mat to split file into single variable files')
        # Getting the type of 'name' (line 278)
        name_135597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 66), 'name', False)
        # Applying the binary operator '%' (line 275)
        result_mod_135598 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 30), '%', str_135596, name_135597)
        
        # Getting the type of 'MatReadWarning' (line 279)
        MatReadWarning_135599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'MatReadWarning', False)
        # Processing the call keyword arguments (line 275)
        int_135600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 57), 'int')
        keyword_135601 = int_135600
        kwargs_135602 = {'stacklevel': keyword_135601}
        # Getting the type of 'warnings' (line 275)
        warnings_135594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 275)
        warn_135595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), warnings_135594, 'warn')
        # Calling warn(args, kwargs) (line 275)
        warn_call_result_135603 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), warn_135595, *[result_mod_135598, MatReadWarning_135599], **kwargs_135602)
        
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'name' (line 280)
        name_135604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'name')
        str_135605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 23), 'str', '')
        # Applying the binary operator '==' (line 280)
        result_eq_135606 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), '==', name_135604, str_135605)
        
        # Testing the type of an if condition (line 280)
        if_condition_135607 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), result_eq_135606)
        # Assigning a type to the variable 'if_condition_135607' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_135607', if_condition_135607)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 282):
        
        # Assigning a Str to a Name (line 282):
        str_135608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 23), 'str', '__function_workspace__')
        # Assigning a type to the variable 'name' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'name', str_135608)
        
        # Assigning a Name to a Name (line 285):
        
        # Assigning a Name to a Name (line 285):
        # Getting the type of 'False' (line 285)
        False_135609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'False')
        # Assigning a type to the variable 'process' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'process', False_135609)
        # SSA branch for the else part of an if statement (line 280)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 287):
        
        # Assigning a Name to a Name (line 287):
        # Getting the type of 'True' (line 287)
        True_135610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'True')
        # Assigning a type to the variable 'process' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'process', True_135610)
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'variable_names' (line 288)
        variable_names_135611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'variable_names')
        # Getting the type of 'None' (line 288)
        None_135612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'None')
        # Applying the binary operator 'isnot' (line 288)
        result_is_not_135613 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 15), 'isnot', variable_names_135611, None_135612)
        
        
        # Getting the type of 'name' (line 288)
        name_135614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 46), 'name')
        # Getting the type of 'variable_names' (line 288)
        variable_names_135615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 58), 'variable_names')
        # Applying the binary operator 'notin' (line 288)
        result_contains_135616 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 46), 'notin', name_135614, variable_names_135615)
        
        # Applying the binary operator 'and' (line 288)
        result_and_keyword_135617 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 15), 'and', result_is_not_135613, result_contains_135616)
        
        # Testing the type of an if condition (line 288)
        if_condition_135618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 12), result_and_keyword_135617)
        # Assigning a type to the variable 'if_condition_135618' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'if_condition_135618', if_condition_135618)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to seek(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'next_position' (line 289)
        next_position_135622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 37), 'next_position', False)
        # Processing the call keyword arguments (line 289)
        kwargs_135623 = {}
        # Getting the type of 'self' (line 289)
        self_135619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 289)
        mat_stream_135620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), self_135619, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 289)
        seek_135621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), mat_stream_135620, 'seek')
        # Calling seek(args, kwargs) (line 289)
        seek_call_result_135624 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), seek_135621, *[next_position_135622], **kwargs_135623)
        
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 291)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to read_var_array(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'hdr' (line 292)
        hdr_135627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 42), 'hdr', False)
        # Getting the type of 'process' (line 292)
        process_135628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 47), 'process', False)
        # Processing the call keyword arguments (line 292)
        kwargs_135629 = {}
        # Getting the type of 'self' (line 292)
        self_135625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 22), 'self', False)
        # Obtaining the member 'read_var_array' of a type (line 292)
        read_var_array_135626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 22), self_135625, 'read_var_array')
        # Calling read_var_array(args, kwargs) (line 292)
        read_var_array_call_result_135630 = invoke(stypy.reporting.localization.Localization(__file__, 292, 22), read_var_array_135626, *[hdr_135627, process_135628], **kwargs_135629)
        
        # Assigning a type to the variable 'res' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'res', read_var_array_call_result_135630)
        # SSA branch for the except part of a try statement (line 291)
        # SSA branch for the except 'MatReadError' branch of a try statement (line 291)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'MatReadError' (line 293)
        MatReadError_135631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'MatReadError')
        # Assigning a type to the variable 'err' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'err', MatReadError_135631)
        
        # Call to warn(...): (line 294)
        # Processing the call arguments (line 294)
        str_135634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 20), 'str', 'Unreadable variable "%s", because "%s"')
        
        # Obtaining an instance of the builtin type 'tuple' (line 296)
        tuple_135635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 296)
        # Adding element type (line 296)
        # Getting the type of 'name' (line 296)
        name_135636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 21), tuple_135635, name_135636)
        # Adding element type (line 296)
        # Getting the type of 'err' (line 296)
        err_135637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'err', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 21), tuple_135635, err_135637)
        
        # Applying the binary operator '%' (line 295)
        result_mod_135638 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 20), '%', str_135634, tuple_135635)
        
        # Getting the type of 'Warning' (line 297)
        Warning_135639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 20), 'Warning', False)
        # Processing the call keyword arguments (line 294)
        int_135640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 40), 'int')
        keyword_135641 = int_135640
        kwargs_135642 = {'stacklevel': keyword_135641}
        # Getting the type of 'warnings' (line 294)
        warnings_135632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 294)
        warn_135633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 16), warnings_135632, 'warn')
        # Calling warn(args, kwargs) (line 294)
        warn_call_result_135643 = invoke(stypy.reporting.localization.Localization(__file__, 294, 16), warn_135633, *[result_mod_135638, Warning_135639], **kwargs_135642)
        
        
        # Assigning a BinOp to a Name (line 298):
        
        # Assigning a BinOp to a Name (line 298):
        str_135644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 22), 'str', 'Read error: %s')
        # Getting the type of 'err' (line 298)
        err_135645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'err')
        # Applying the binary operator '%' (line 298)
        result_mod_135646 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 22), '%', str_135644, err_135645)
        
        # Assigning a type to the variable 'res' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'res', result_mod_135646)
        # SSA join for try-except statement (line 291)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to seek(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'next_position' (line 299)
        next_position_135650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'next_position', False)
        # Processing the call keyword arguments (line 299)
        kwargs_135651 = {}
        # Getting the type of 'self' (line 299)
        self_135647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 299)
        mat_stream_135648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), self_135647, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 299)
        seek_135649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), mat_stream_135648, 'seek')
        # Calling seek(args, kwargs) (line 299)
        seek_call_result_135652 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), seek_135649, *[next_position_135650], **kwargs_135651)
        
        
        # Assigning a Name to a Subscript (line 300):
        
        # Assigning a Name to a Subscript (line 300):
        # Getting the type of 'res' (line 300)
        res_135653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 26), 'res')
        # Getting the type of 'mdict' (line 300)
        mdict_135654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'mdict')
        # Getting the type of 'name' (line 300)
        name_135655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 18), 'name')
        # Storing an element on a container (line 300)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 12), mdict_135654, (name_135655, res_135653))
        
        # Getting the type of 'hdr' (line 301)
        hdr_135656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'hdr')
        # Obtaining the member 'is_global' of a type (line 301)
        is_global_135657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 15), hdr_135656, 'is_global')
        # Testing the type of an if condition (line 301)
        if_condition_135658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 12), is_global_135657)
        # Assigning a type to the variable 'if_condition_135658' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'if_condition_135658', if_condition_135658)
        # SSA begins for if statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'name' (line 302)
        name_135664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 44), 'name', False)
        # Processing the call keyword arguments (line 302)
        kwargs_135665 = {}
        
        # Obtaining the type of the subscript
        str_135659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'str', '__globals__')
        # Getting the type of 'mdict' (line 302)
        mdict_135660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'mdict', False)
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___135661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), mdict_135660, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_135662 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), getitem___135661, str_135659)
        
        # Obtaining the member 'append' of a type (line 302)
        append_135663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), subscript_call_result_135662, 'append')
        # Calling append(args, kwargs) (line 302)
        append_call_result_135666 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), append_135663, *[name_135664], **kwargs_135665)
        
        # SSA join for if statement (line 301)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 303)
        # Getting the type of 'variable_names' (line 303)
        variable_names_135667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'variable_names')
        # Getting the type of 'None' (line 303)
        None_135668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 37), 'None')
        
        (may_be_135669, more_types_in_union_135670) = may_not_be_none(variable_names_135667, None_135668)

        if may_be_135669:

            if more_types_in_union_135670:
                # Runtime conditional SSA (line 303)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to remove(...): (line 304)
            # Processing the call arguments (line 304)
            # Getting the type of 'name' (line 304)
            name_135673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 38), 'name', False)
            # Processing the call keyword arguments (line 304)
            kwargs_135674 = {}
            # Getting the type of 'variable_names' (line 304)
            variable_names_135671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'variable_names', False)
            # Obtaining the member 'remove' of a type (line 304)
            remove_135672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 16), variable_names_135671, 'remove')
            # Calling remove(args, kwargs) (line 304)
            remove_call_result_135675 = invoke(stypy.reporting.localization.Localization(__file__, 304, 16), remove_135672, *[name_135673], **kwargs_135674)
            
            
            
            
            # Call to len(...): (line 305)
            # Processing the call arguments (line 305)
            # Getting the type of 'variable_names' (line 305)
            variable_names_135677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 23), 'variable_names', False)
            # Processing the call keyword arguments (line 305)
            kwargs_135678 = {}
            # Getting the type of 'len' (line 305)
            len_135676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'len', False)
            # Calling len(args, kwargs) (line 305)
            len_call_result_135679 = invoke(stypy.reporting.localization.Localization(__file__, 305, 19), len_135676, *[variable_names_135677], **kwargs_135678)
            
            int_135680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 42), 'int')
            # Applying the binary operator '==' (line 305)
            result_eq_135681 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 19), '==', len_call_result_135679, int_135680)
            
            # Testing the type of an if condition (line 305)
            if_condition_135682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 16), result_eq_135681)
            # Assigning a type to the variable 'if_condition_135682' (line 305)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'if_condition_135682', if_condition_135682)
            # SSA begins for if statement (line 305)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 305)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_135670:
                # SSA join for if statement (line 303)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for while statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'mdict' (line 307)
        mdict_135683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'mdict')
        # Assigning a type to the variable 'stypy_return_type' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', mdict_135683)
        
        # ################# End of 'get_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_135684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135684)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_variables'
        return stypy_return_type_135684


    @norecursion
    def list_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'list_variables'
        module_type_store = module_type_store.open_function_context('list_variables', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_function_name', 'MatFile5Reader.list_variables')
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Reader.list_variables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Reader.list_variables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'list_variables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'list_variables(...)' code ##################

        str_135685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'str', ' list variables from stream ')
        
        # Call to seek(...): (line 311)
        # Processing the call arguments (line 311)
        int_135689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 29), 'int')
        # Processing the call keyword arguments (line 311)
        kwargs_135690 = {}
        # Getting the type of 'self' (line 311)
        self_135686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 311)
        mat_stream_135687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_135686, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 311)
        seek_135688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), mat_stream_135687, 'seek')
        # Calling seek(args, kwargs) (line 311)
        seek_call_result_135691 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), seek_135688, *[int_135689], **kwargs_135690)
        
        
        # Call to initialize_read(...): (line 313)
        # Processing the call keyword arguments (line 313)
        kwargs_135694 = {}
        # Getting the type of 'self' (line 313)
        self_135692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'self', False)
        # Obtaining the member 'initialize_read' of a type (line 313)
        initialize_read_135693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), self_135692, 'initialize_read')
        # Calling initialize_read(args, kwargs) (line 313)
        initialize_read_call_result_135695 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), initialize_read_135693, *[], **kwargs_135694)
        
        
        # Call to read_file_header(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_135698 = {}
        # Getting the type of 'self' (line 314)
        self_135696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self', False)
        # Obtaining the member 'read_file_header' of a type (line 314)
        read_file_header_135697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_135696, 'read_file_header')
        # Calling read_file_header(args, kwargs) (line 314)
        read_file_header_call_result_135699 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), read_file_header_135697, *[], **kwargs_135698)
        
        
        # Assigning a List to a Name (line 315):
        
        # Assigning a List to a Name (line 315):
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_135700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        
        # Assigning a type to the variable 'vars' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'vars', list_135700)
        
        
        
        # Call to end_of_stream(...): (line 316)
        # Processing the call keyword arguments (line 316)
        kwargs_135703 = {}
        # Getting the type of 'self' (line 316)
        self_135701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'self', False)
        # Obtaining the member 'end_of_stream' of a type (line 316)
        end_of_stream_135702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 18), self_135701, 'end_of_stream')
        # Calling end_of_stream(args, kwargs) (line 316)
        end_of_stream_call_result_135704 = invoke(stypy.reporting.localization.Localization(__file__, 316, 18), end_of_stream_135702, *[], **kwargs_135703)
        
        # Applying the 'not' unary operator (line 316)
        result_not__135705 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 14), 'not', end_of_stream_call_result_135704)
        
        # Testing the type of an if condition (line 316)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 8), result_not__135705)
        # SSA begins for while statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 317):
        
        # Assigning a Subscript to a Name (line 317):
        
        # Obtaining the type of the subscript
        int_135706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 12), 'int')
        
        # Call to read_var_header(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_135709 = {}
        # Getting the type of 'self' (line 317)
        self_135707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 317)
        read_var_header_135708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 33), self_135707, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 317)
        read_var_header_call_result_135710 = invoke(stypy.reporting.localization.Localization(__file__, 317, 33), read_var_header_135708, *[], **kwargs_135709)
        
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___135711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), read_var_header_call_result_135710, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_135712 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), getitem___135711, int_135706)
        
        # Assigning a type to the variable 'tuple_var_assignment_135263' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'tuple_var_assignment_135263', subscript_call_result_135712)
        
        # Assigning a Subscript to a Name (line 317):
        
        # Obtaining the type of the subscript
        int_135713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 12), 'int')
        
        # Call to read_var_header(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_135716 = {}
        # Getting the type of 'self' (line 317)
        self_135714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 317)
        read_var_header_135715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 33), self_135714, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 317)
        read_var_header_call_result_135717 = invoke(stypy.reporting.localization.Localization(__file__, 317, 33), read_var_header_135715, *[], **kwargs_135716)
        
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___135718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), read_var_header_call_result_135717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_135719 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), getitem___135718, int_135713)
        
        # Assigning a type to the variable 'tuple_var_assignment_135264' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'tuple_var_assignment_135264', subscript_call_result_135719)
        
        # Assigning a Name to a Name (line 317):
        # Getting the type of 'tuple_var_assignment_135263' (line 317)
        tuple_var_assignment_135263_135720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'tuple_var_assignment_135263')
        # Assigning a type to the variable 'hdr' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'hdr', tuple_var_assignment_135263_135720)
        
        # Assigning a Name to a Name (line 317):
        # Getting the type of 'tuple_var_assignment_135264' (line 317)
        tuple_var_assignment_135264_135721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'tuple_var_assignment_135264')
        # Assigning a type to the variable 'next_position' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'next_position', tuple_var_assignment_135264_135721)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to asstr(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'hdr' (line 318)
        hdr_135723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'hdr', False)
        # Obtaining the member 'name' of a type (line 318)
        name_135724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 25), hdr_135723, 'name')
        # Processing the call keyword arguments (line 318)
        kwargs_135725 = {}
        # Getting the type of 'asstr' (line 318)
        asstr_135722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'asstr', False)
        # Calling asstr(args, kwargs) (line 318)
        asstr_call_result_135726 = invoke(stypy.reporting.localization.Localization(__file__, 318, 19), asstr_135722, *[name_135724], **kwargs_135725)
        
        # Assigning a type to the variable 'name' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'name', asstr_call_result_135726)
        
        
        # Getting the type of 'name' (line 319)
        name_135727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'name')
        str_135728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 23), 'str', '')
        # Applying the binary operator '==' (line 319)
        result_eq_135729 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 15), '==', name_135727, str_135728)
        
        # Testing the type of an if condition (line 319)
        if_condition_135730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 12), result_eq_135729)
        # Assigning a type to the variable 'if_condition_135730' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'if_condition_135730', if_condition_135730)
        # SSA begins for if statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 321):
        
        # Assigning a Str to a Name (line 321):
        str_135731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 23), 'str', '__function_workspace__')
        # Assigning a type to the variable 'name' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'name', str_135731)
        # SSA join for if statement (line 319)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to shape_from_header(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'hdr' (line 323)
        hdr_135735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 58), 'hdr', False)
        # Processing the call keyword arguments (line 323)
        kwargs_135736 = {}
        # Getting the type of 'self' (line 323)
        self_135732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 323)
        _matrix_reader_135733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 20), self_135732, '_matrix_reader')
        # Obtaining the member 'shape_from_header' of a type (line 323)
        shape_from_header_135734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 20), _matrix_reader_135733, 'shape_from_header')
        # Calling shape_from_header(args, kwargs) (line 323)
        shape_from_header_call_result_135737 = invoke(stypy.reporting.localization.Localization(__file__, 323, 20), shape_from_header_135734, *[hdr_135735], **kwargs_135736)
        
        # Assigning a type to the variable 'shape' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'shape', shape_from_header_call_result_135737)
        
        # Getting the type of 'hdr' (line 324)
        hdr_135738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'hdr')
        # Obtaining the member 'is_logical' of a type (line 324)
        is_logical_135739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 15), hdr_135738, 'is_logical')
        # Testing the type of an if condition (line 324)
        if_condition_135740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 12), is_logical_135739)
        # Assigning a type to the variable 'if_condition_135740' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'if_condition_135740', if_condition_135740)
        # SSA begins for if statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 325):
        
        # Assigning a Str to a Name (line 325):
        str_135741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'str', 'logical')
        # Assigning a type to the variable 'info' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'info', str_135741)
        # SSA branch for the else part of an if statement (line 324)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 327):
        
        # Assigning a Call to a Name (line 327):
        
        # Call to get(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'hdr' (line 327)
        hdr_135744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 39), 'hdr', False)
        # Obtaining the member 'mclass' of a type (line 327)
        mclass_135745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 39), hdr_135744, 'mclass')
        str_135746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 51), 'str', 'unknown')
        # Processing the call keyword arguments (line 327)
        kwargs_135747 = {}
        # Getting the type of 'mclass_info' (line 327)
        mclass_info_135742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 23), 'mclass_info', False)
        # Obtaining the member 'get' of a type (line 327)
        get_135743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 23), mclass_info_135742, 'get')
        # Calling get(args, kwargs) (line 327)
        get_call_result_135748 = invoke(stypy.reporting.localization.Localization(__file__, 327, 23), get_135743, *[mclass_135745, str_135746], **kwargs_135747)
        
        # Assigning a type to the variable 'info' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'info', get_call_result_135748)
        # SSA join for if statement (line 324)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 328)
        # Processing the call arguments (line 328)
        
        # Obtaining an instance of the builtin type 'tuple' (line 328)
        tuple_135751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 328)
        # Adding element type (line 328)
        # Getting the type of 'name' (line 328)
        name_135752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 25), tuple_135751, name_135752)
        # Adding element type (line 328)
        # Getting the type of 'shape' (line 328)
        shape_135753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 31), 'shape', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 25), tuple_135751, shape_135753)
        # Adding element type (line 328)
        # Getting the type of 'info' (line 328)
        info_135754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 38), 'info', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 25), tuple_135751, info_135754)
        
        # Processing the call keyword arguments (line 328)
        kwargs_135755 = {}
        # Getting the type of 'vars' (line 328)
        vars_135749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'vars', False)
        # Obtaining the member 'append' of a type (line 328)
        append_135750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), vars_135749, 'append')
        # Calling append(args, kwargs) (line 328)
        append_call_result_135756 = invoke(stypy.reporting.localization.Localization(__file__, 328, 12), append_135750, *[tuple_135751], **kwargs_135755)
        
        
        # Call to seek(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'next_position' (line 330)
        next_position_135760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 33), 'next_position', False)
        # Processing the call keyword arguments (line 330)
        kwargs_135761 = {}
        # Getting the type of 'self' (line 330)
        self_135757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 330)
        mat_stream_135758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), self_135757, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 330)
        seek_135759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), mat_stream_135758, 'seek')
        # Calling seek(args, kwargs) (line 330)
        seek_call_result_135762 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), seek_135759, *[next_position_135760], **kwargs_135761)
        
        # SSA join for while statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vars' (line 331)
        vars_135763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 15), 'vars')
        # Assigning a type to the variable 'stypy_return_type' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'stypy_return_type', vars_135763)
        
        # ################# End of 'list_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'list_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_135764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'list_variables'
        return stypy_return_type_135764


# Assigning a type to the variable 'MatFile5Reader' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'MatFile5Reader', MatFile5Reader)

@norecursion
def varmats_from_mat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'varmats_from_mat'
    module_type_store = module_type_store.open_function_context('varmats_from_mat', 334, 0, False)
    
    # Passed parameters checking function
    varmats_from_mat.stypy_localization = localization
    varmats_from_mat.stypy_type_of_self = None
    varmats_from_mat.stypy_type_store = module_type_store
    varmats_from_mat.stypy_function_name = 'varmats_from_mat'
    varmats_from_mat.stypy_param_names_list = ['file_obj']
    varmats_from_mat.stypy_varargs_param_name = None
    varmats_from_mat.stypy_kwargs_param_name = None
    varmats_from_mat.stypy_call_defaults = defaults
    varmats_from_mat.stypy_call_varargs = varargs
    varmats_from_mat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'varmats_from_mat', ['file_obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'varmats_from_mat', localization, ['file_obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'varmats_from_mat(...)' code ##################

    str_135765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, (-1)), 'str', " Pull variables out of mat 5 file as a sequence of mat file objects\n\n    This can be useful with a difficult mat file, containing unreadable\n    variables.  This routine pulls the variables out in raw form and puts them,\n    unread, back into a file stream for saving or reading.  Another use is the\n    pathological case where there is more than one variable of the same name in\n    the file; this routine returns the duplicates, whereas the standard reader\n    will overwrite duplicates in the returned dictionary.\n\n    The file pointer in `file_obj` will be undefined.  File pointers for the\n    returned file-like objects are set at 0.\n\n    Parameters\n    ----------\n    file_obj : file-like\n        file object containing mat file\n\n    Returns\n    -------\n    named_mats : list\n        list contains tuples of (name, BytesIO) where BytesIO is a file-like\n        object containing mat file contents as for a single variable.  The\n        BytesIO contains a string with the original header and a single var. If\n        ``var_file_obj`` is an individual BytesIO instance, then save as a mat\n        file with something like ``open('test.mat',\n        'wb').write(var_file_obj.read())``\n\n    Examples\n    --------\n    >>> import scipy.io\n\n    BytesIO is from the ``io`` module in python 3, and is ``cStringIO`` for\n    python < 3.\n\n    >>> mat_fileobj = BytesIO()\n    >>> scipy.io.savemat(mat_fileobj, {'b': np.arange(10), 'a': 'a string'})\n    >>> varmats = varmats_from_mat(mat_fileobj)\n    >>> sorted([name for name, str_obj in varmats])\n    ['a', 'b']\n    ")
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to MatFile5Reader(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'file_obj' (line 375)
    file_obj_135767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 25), 'file_obj', False)
    # Processing the call keyword arguments (line 375)
    kwargs_135768 = {}
    # Getting the type of 'MatFile5Reader' (line 375)
    MatFile5Reader_135766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 10), 'MatFile5Reader', False)
    # Calling MatFile5Reader(args, kwargs) (line 375)
    MatFile5Reader_call_result_135769 = invoke(stypy.reporting.localization.Localization(__file__, 375, 10), MatFile5Reader_135766, *[file_obj_135767], **kwargs_135768)
    
    # Assigning a type to the variable 'rdr' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'rdr', MatFile5Reader_call_result_135769)
    
    # Call to seek(...): (line 376)
    # Processing the call arguments (line 376)
    int_135772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 18), 'int')
    # Processing the call keyword arguments (line 376)
    kwargs_135773 = {}
    # Getting the type of 'file_obj' (line 376)
    file_obj_135770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'file_obj', False)
    # Obtaining the member 'seek' of a type (line 376)
    seek_135771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 4), file_obj_135770, 'seek')
    # Calling seek(args, kwargs) (line 376)
    seek_call_result_135774 = invoke(stypy.reporting.localization.Localization(__file__, 376, 4), seek_135771, *[int_135772], **kwargs_135773)
    
    
    # Assigning a Attribute to a Name (line 378):
    
    # Assigning a Attribute to a Name (line 378):
    
    # Obtaining the type of the subscript
    str_135775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 45), 'str', 'file_header')
    
    # Obtaining the type of the subscript
    str_135776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 35), 'str', 'dtypes')
    
    # Obtaining the type of the subscript
    # Getting the type of 'native_code' (line 378)
    native_code_135777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 22), 'native_code')
    # Getting the type of 'MDTYPES' (line 378)
    MDTYPES_135778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 14), 'MDTYPES')
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___135779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 14), MDTYPES_135778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 378)
    subscript_call_result_135780 = invoke(stypy.reporting.localization.Localization(__file__, 378, 14), getitem___135779, native_code_135777)
    
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___135781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 14), subscript_call_result_135780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 378)
    subscript_call_result_135782 = invoke(stypy.reporting.localization.Localization(__file__, 378, 14), getitem___135781, str_135776)
    
    # Obtaining the member '__getitem__' of a type (line 378)
    getitem___135783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 14), subscript_call_result_135782, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 378)
    subscript_call_result_135784 = invoke(stypy.reporting.localization.Localization(__file__, 378, 14), getitem___135783, str_135775)
    
    # Obtaining the member 'itemsize' of a type (line 378)
    itemsize_135785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 14), subscript_call_result_135784, 'itemsize')
    # Assigning a type to the variable 'hdr_len' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'hdr_len', itemsize_135785)
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to read(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'hdr_len' (line 379)
    hdr_len_135788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 28), 'hdr_len', False)
    # Processing the call keyword arguments (line 379)
    kwargs_135789 = {}
    # Getting the type of 'file_obj' (line 379)
    file_obj_135786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 14), 'file_obj', False)
    # Obtaining the member 'read' of a type (line 379)
    read_135787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 14), file_obj_135786, 'read')
    # Calling read(args, kwargs) (line 379)
    read_call_result_135790 = invoke(stypy.reporting.localization.Localization(__file__, 379, 14), read_135787, *[hdr_len_135788], **kwargs_135789)
    
    # Assigning a type to the variable 'raw_hdr' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'raw_hdr', read_call_result_135790)
    
    # Call to seek(...): (line 381)
    # Processing the call arguments (line 381)
    int_135793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 18), 'int')
    # Processing the call keyword arguments (line 381)
    kwargs_135794 = {}
    # Getting the type of 'file_obj' (line 381)
    file_obj_135791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'file_obj', False)
    # Obtaining the member 'seek' of a type (line 381)
    seek_135792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 4), file_obj_135791, 'seek')
    # Calling seek(args, kwargs) (line 381)
    seek_call_result_135795 = invoke(stypy.reporting.localization.Localization(__file__, 381, 4), seek_135792, *[int_135793], **kwargs_135794)
    
    
    # Call to initialize_read(...): (line 382)
    # Processing the call keyword arguments (line 382)
    kwargs_135798 = {}
    # Getting the type of 'rdr' (line 382)
    rdr_135796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'rdr', False)
    # Obtaining the member 'initialize_read' of a type (line 382)
    initialize_read_135797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 4), rdr_135796, 'initialize_read')
    # Calling initialize_read(args, kwargs) (line 382)
    initialize_read_call_result_135799 = invoke(stypy.reporting.localization.Localization(__file__, 382, 4), initialize_read_135797, *[], **kwargs_135798)
    
    
    # Assigning a Call to a Name (line 383):
    
    # Assigning a Call to a Name (line 383):
    
    # Call to read_file_header(...): (line 383)
    # Processing the call keyword arguments (line 383)
    kwargs_135802 = {}
    # Getting the type of 'rdr' (line 383)
    rdr_135800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'rdr', False)
    # Obtaining the member 'read_file_header' of a type (line 383)
    read_file_header_135801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), rdr_135800, 'read_file_header')
    # Calling read_file_header(args, kwargs) (line 383)
    read_file_header_call_result_135803 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), read_file_header_135801, *[], **kwargs_135802)
    
    # Assigning a type to the variable 'mdict' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'mdict', read_file_header_call_result_135803)
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 384):
    
    # Call to tell(...): (line 384)
    # Processing the call keyword arguments (line 384)
    kwargs_135806 = {}
    # Getting the type of 'file_obj' (line 384)
    file_obj_135804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'file_obj', False)
    # Obtaining the member 'tell' of a type (line 384)
    tell_135805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 20), file_obj_135804, 'tell')
    # Calling tell(args, kwargs) (line 384)
    tell_call_result_135807 = invoke(stypy.reporting.localization.Localization(__file__, 384, 20), tell_135805, *[], **kwargs_135806)
    
    # Assigning a type to the variable 'next_position' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'next_position', tell_call_result_135807)
    
    # Assigning a List to a Name (line 385):
    
    # Assigning a List to a Name (line 385):
    
    # Obtaining an instance of the builtin type 'list' (line 385)
    list_135808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 385)
    
    # Assigning a type to the variable 'named_mats' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'named_mats', list_135808)
    
    
    
    # Call to end_of_stream(...): (line 386)
    # Processing the call keyword arguments (line 386)
    kwargs_135811 = {}
    # Getting the type of 'rdr' (line 386)
    rdr_135809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 14), 'rdr', False)
    # Obtaining the member 'end_of_stream' of a type (line 386)
    end_of_stream_135810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 14), rdr_135809, 'end_of_stream')
    # Calling end_of_stream(args, kwargs) (line 386)
    end_of_stream_call_result_135812 = invoke(stypy.reporting.localization.Localization(__file__, 386, 14), end_of_stream_135810, *[], **kwargs_135811)
    
    # Applying the 'not' unary operator (line 386)
    result_not__135813 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 10), 'not', end_of_stream_call_result_135812)
    
    # Testing the type of an if condition (line 386)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 4), result_not__135813)
    # SSA begins for while statement (line 386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 387):
    
    # Assigning a Name to a Name (line 387):
    # Getting the type of 'next_position' (line 387)
    next_position_135814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'next_position')
    # Assigning a type to the variable 'start_position' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'start_position', next_position_135814)
    
    # Assigning a Call to a Tuple (line 388):
    
    # Assigning a Subscript to a Name (line 388):
    
    # Obtaining the type of the subscript
    int_135815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 8), 'int')
    
    # Call to read_var_header(...): (line 388)
    # Processing the call keyword arguments (line 388)
    kwargs_135818 = {}
    # Getting the type of 'rdr' (line 388)
    rdr_135816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'rdr', False)
    # Obtaining the member 'read_var_header' of a type (line 388)
    read_var_header_135817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 29), rdr_135816, 'read_var_header')
    # Calling read_var_header(args, kwargs) (line 388)
    read_var_header_call_result_135819 = invoke(stypy.reporting.localization.Localization(__file__, 388, 29), read_var_header_135817, *[], **kwargs_135818)
    
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___135820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), read_var_header_call_result_135819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_135821 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), getitem___135820, int_135815)
    
    # Assigning a type to the variable 'tuple_var_assignment_135265' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_135265', subscript_call_result_135821)
    
    # Assigning a Subscript to a Name (line 388):
    
    # Obtaining the type of the subscript
    int_135822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 8), 'int')
    
    # Call to read_var_header(...): (line 388)
    # Processing the call keyword arguments (line 388)
    kwargs_135825 = {}
    # Getting the type of 'rdr' (line 388)
    rdr_135823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'rdr', False)
    # Obtaining the member 'read_var_header' of a type (line 388)
    read_var_header_135824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 29), rdr_135823, 'read_var_header')
    # Calling read_var_header(args, kwargs) (line 388)
    read_var_header_call_result_135826 = invoke(stypy.reporting.localization.Localization(__file__, 388, 29), read_var_header_135824, *[], **kwargs_135825)
    
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___135827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), read_var_header_call_result_135826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_135828 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), getitem___135827, int_135822)
    
    # Assigning a type to the variable 'tuple_var_assignment_135266' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_135266', subscript_call_result_135828)
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'tuple_var_assignment_135265' (line 388)
    tuple_var_assignment_135265_135829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_135265')
    # Assigning a type to the variable 'hdr' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'hdr', tuple_var_assignment_135265_135829)
    
    # Assigning a Name to a Name (line 388):
    # Getting the type of 'tuple_var_assignment_135266' (line 388)
    tuple_var_assignment_135266_135830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_135266')
    # Assigning a type to the variable 'next_position' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'next_position', tuple_var_assignment_135266_135830)
    
    # Assigning a Call to a Name (line 389):
    
    # Assigning a Call to a Name (line 389):
    
    # Call to asstr(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'hdr' (line 389)
    hdr_135832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), 'hdr', False)
    # Obtaining the member 'name' of a type (line 389)
    name_135833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 21), hdr_135832, 'name')
    # Processing the call keyword arguments (line 389)
    kwargs_135834 = {}
    # Getting the type of 'asstr' (line 389)
    asstr_135831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'asstr', False)
    # Calling asstr(args, kwargs) (line 389)
    asstr_call_result_135835 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), asstr_135831, *[name_135833], **kwargs_135834)
    
    # Assigning a type to the variable 'name' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'name', asstr_call_result_135835)
    
    # Call to seek(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'start_position' (line 391)
    start_position_135838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 22), 'start_position', False)
    # Processing the call keyword arguments (line 391)
    kwargs_135839 = {}
    # Getting the type of 'file_obj' (line 391)
    file_obj_135836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'file_obj', False)
    # Obtaining the member 'seek' of a type (line 391)
    seek_135837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), file_obj_135836, 'seek')
    # Calling seek(args, kwargs) (line 391)
    seek_call_result_135840 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), seek_135837, *[start_position_135838], **kwargs_135839)
    
    
    # Assigning a BinOp to a Name (line 392):
    
    # Assigning a BinOp to a Name (line 392):
    # Getting the type of 'next_position' (line 392)
    next_position_135841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'next_position')
    # Getting the type of 'start_position' (line 392)
    start_position_135842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 37), 'start_position')
    # Applying the binary operator '-' (line 392)
    result_sub_135843 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 21), '-', next_position_135841, start_position_135842)
    
    # Assigning a type to the variable 'byte_count' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'byte_count', result_sub_135843)
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to read(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'byte_count' (line 393)
    byte_count_135846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 32), 'byte_count', False)
    # Processing the call keyword arguments (line 393)
    kwargs_135847 = {}
    # Getting the type of 'file_obj' (line 393)
    file_obj_135844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 18), 'file_obj', False)
    # Obtaining the member 'read' of a type (line 393)
    read_135845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 18), file_obj_135844, 'read')
    # Calling read(args, kwargs) (line 393)
    read_call_result_135848 = invoke(stypy.reporting.localization.Localization(__file__, 393, 18), read_135845, *[byte_count_135846], **kwargs_135847)
    
    # Assigning a type to the variable 'var_str' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'var_str', read_call_result_135848)
    
    # Assigning a Call to a Name (line 395):
    
    # Assigning a Call to a Name (line 395):
    
    # Call to BytesIO(...): (line 395)
    # Processing the call keyword arguments (line 395)
    kwargs_135850 = {}
    # Getting the type of 'BytesIO' (line 395)
    BytesIO_135849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 18), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 395)
    BytesIO_call_result_135851 = invoke(stypy.reporting.localization.Localization(__file__, 395, 18), BytesIO_135849, *[], **kwargs_135850)
    
    # Assigning a type to the variable 'out_obj' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'out_obj', BytesIO_call_result_135851)
    
    # Call to write(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'raw_hdr' (line 396)
    raw_hdr_135854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'raw_hdr', False)
    # Processing the call keyword arguments (line 396)
    kwargs_135855 = {}
    # Getting the type of 'out_obj' (line 396)
    out_obj_135852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'out_obj', False)
    # Obtaining the member 'write' of a type (line 396)
    write_135853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), out_obj_135852, 'write')
    # Calling write(args, kwargs) (line 396)
    write_call_result_135856 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), write_135853, *[raw_hdr_135854], **kwargs_135855)
    
    
    # Call to write(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'var_str' (line 397)
    var_str_135859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'var_str', False)
    # Processing the call keyword arguments (line 397)
    kwargs_135860 = {}
    # Getting the type of 'out_obj' (line 397)
    out_obj_135857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'out_obj', False)
    # Obtaining the member 'write' of a type (line 397)
    write_135858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), out_obj_135857, 'write')
    # Calling write(args, kwargs) (line 397)
    write_call_result_135861 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), write_135858, *[var_str_135859], **kwargs_135860)
    
    
    # Call to seek(...): (line 398)
    # Processing the call arguments (line 398)
    int_135864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 21), 'int')
    # Processing the call keyword arguments (line 398)
    kwargs_135865 = {}
    # Getting the type of 'out_obj' (line 398)
    out_obj_135862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'out_obj', False)
    # Obtaining the member 'seek' of a type (line 398)
    seek_135863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), out_obj_135862, 'seek')
    # Calling seek(args, kwargs) (line 398)
    seek_call_result_135866 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), seek_135863, *[int_135864], **kwargs_135865)
    
    
    # Call to append(...): (line 399)
    # Processing the call arguments (line 399)
    
    # Obtaining an instance of the builtin type 'tuple' (line 399)
    tuple_135869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 399)
    # Adding element type (line 399)
    # Getting the type of 'name' (line 399)
    name_135870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 27), tuple_135869, name_135870)
    # Adding element type (line 399)
    # Getting the type of 'out_obj' (line 399)
    out_obj_135871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 33), 'out_obj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 27), tuple_135869, out_obj_135871)
    
    # Processing the call keyword arguments (line 399)
    kwargs_135872 = {}
    # Getting the type of 'named_mats' (line 399)
    named_mats_135867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'named_mats', False)
    # Obtaining the member 'append' of a type (line 399)
    append_135868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), named_mats_135867, 'append')
    # Calling append(args, kwargs) (line 399)
    append_call_result_135873 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), append_135868, *[tuple_135869], **kwargs_135872)
    
    # SSA join for while statement (line 386)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'named_mats' (line 400)
    named_mats_135874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'named_mats')
    # Assigning a type to the variable 'stypy_return_type' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type', named_mats_135874)
    
    # ################# End of 'varmats_from_mat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'varmats_from_mat' in the type store
    # Getting the type of 'stypy_return_type' (line 334)
    stypy_return_type_135875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_135875)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'varmats_from_mat'
    return stypy_return_type_135875

# Assigning a type to the variable 'varmats_from_mat' (line 334)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'varmats_from_mat', varmats_from_mat)
# Declaration of the 'EmptyStructMarker' class

class EmptyStructMarker(object, ):
    str_135876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'str', ' Class to indicate presence of empty matlab struct on output ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 403, 0, False)
        # Assigning a type to the variable 'self' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EmptyStructMarker.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'EmptyStructMarker' (line 403)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'EmptyStructMarker', EmptyStructMarker)

@norecursion
def to_writeable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'to_writeable'
    module_type_store = module_type_store.open_function_context('to_writeable', 407, 0, False)
    
    # Passed parameters checking function
    to_writeable.stypy_localization = localization
    to_writeable.stypy_type_of_self = None
    to_writeable.stypy_type_store = module_type_store
    to_writeable.stypy_function_name = 'to_writeable'
    to_writeable.stypy_param_names_list = ['source']
    to_writeable.stypy_varargs_param_name = None
    to_writeable.stypy_kwargs_param_name = None
    to_writeable.stypy_call_defaults = defaults
    to_writeable.stypy_call_varargs = varargs
    to_writeable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'to_writeable', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'to_writeable', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'to_writeable(...)' code ##################

    str_135877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', ' Convert input object ``source`` to something we can write\n\n    Parameters\n    ----------\n    source : object\n\n    Returns\n    -------\n    arr : None or ndarray or EmptyStructMarker\n        If `source` cannot be converted to something we can write to a matfile,\n        return None.  If `source` is equivalent to an empty dictionary, return\n        ``EmptyStructMarker``.  Otherwise return `source` converted to an\n        ndarray with contents for writing to matfile.\n    ')
    
    
    # Call to isinstance(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'source' (line 422)
    source_135879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 18), 'source', False)
    # Getting the type of 'np' (line 422)
    np_135880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 26), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 422)
    ndarray_135881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 26), np_135880, 'ndarray')
    # Processing the call keyword arguments (line 422)
    kwargs_135882 = {}
    # Getting the type of 'isinstance' (line 422)
    isinstance_135878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 422)
    isinstance_call_result_135883 = invoke(stypy.reporting.localization.Localization(__file__, 422, 7), isinstance_135878, *[source_135879, ndarray_135881], **kwargs_135882)
    
    # Testing the type of an if condition (line 422)
    if_condition_135884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 4), isinstance_call_result_135883)
    # Assigning a type to the variable 'if_condition_135884' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'if_condition_135884', if_condition_135884)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'source' (line 423)
    source_135885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'source')
    # Assigning a type to the variable 'stypy_return_type' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'stypy_return_type', source_135885)
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 424)
    # Getting the type of 'source' (line 424)
    source_135886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 7), 'source')
    # Getting the type of 'None' (line 424)
    None_135887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 17), 'None')
    
    (may_be_135888, more_types_in_union_135889) = may_be_none(source_135886, None_135887)

    if may_be_135888:

        if more_types_in_union_135889:
            # Runtime conditional SSA (line 424)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'None' (line 425)
        None_135890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'stypy_return_type', None_135890)

        if more_types_in_union_135889:
            # SSA join for if statement (line 424)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BoolOp to a Name (line 427):
    
    # Assigning a BoolOp to a Name (line 427):
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'source' (line 427)
    source_135892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 26), 'source', False)
    str_135893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 34), 'str', 'keys')
    # Processing the call keyword arguments (line 427)
    kwargs_135894 = {}
    # Getting the type of 'hasattr' (line 427)
    hasattr_135891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 18), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 427)
    hasattr_call_result_135895 = invoke(stypy.reporting.localization.Localization(__file__, 427, 18), hasattr_135891, *[source_135892, str_135893], **kwargs_135894)
    
    
    # Call to hasattr(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'source' (line 427)
    source_135897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 54), 'source', False)
    str_135898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 62), 'str', 'values')
    # Processing the call keyword arguments (line 427)
    kwargs_135899 = {}
    # Getting the type of 'hasattr' (line 427)
    hasattr_135896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 46), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 427)
    hasattr_call_result_135900 = invoke(stypy.reporting.localization.Localization(__file__, 427, 46), hasattr_135896, *[source_135897, str_135898], **kwargs_135899)
    
    # Applying the binary operator 'and' (line 427)
    result_and_keyword_135901 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 18), 'and', hasattr_call_result_135895, hasattr_call_result_135900)
    
    # Call to hasattr(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'source' (line 428)
    source_135903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'source', False)
    str_135904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 34), 'str', 'items')
    # Processing the call keyword arguments (line 428)
    kwargs_135905 = {}
    # Getting the type of 'hasattr' (line 428)
    hasattr_135902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 18), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 428)
    hasattr_call_result_135906 = invoke(stypy.reporting.localization.Localization(__file__, 428, 18), hasattr_135902, *[source_135903, str_135904], **kwargs_135905)
    
    # Applying the binary operator 'and' (line 427)
    result_and_keyword_135907 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 18), 'and', result_and_keyword_135901, hasattr_call_result_135906)
    
    # Assigning a type to the variable 'is_mapping' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'is_mapping', result_and_keyword_135907)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'is_mapping' (line 430)
    is_mapping_135908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'is_mapping')
    # Applying the 'not' unary operator (line 430)
    result_not__135909 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 7), 'not', is_mapping_135908)
    
    
    # Call to hasattr(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'source' (line 430)
    source_135911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'source', False)
    str_135912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 42), 'str', '__dict__')
    # Processing the call keyword arguments (line 430)
    kwargs_135913 = {}
    # Getting the type of 'hasattr' (line 430)
    hasattr_135910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 26), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 430)
    hasattr_call_result_135914 = invoke(stypy.reporting.localization.Localization(__file__, 430, 26), hasattr_135910, *[source_135911, str_135912], **kwargs_135913)
    
    # Applying the binary operator 'and' (line 430)
    result_and_keyword_135915 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 7), 'and', result_not__135909, hasattr_call_result_135914)
    
    # Testing the type of an if condition (line 430)
    if_condition_135916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 4), result_and_keyword_135915)
    # Assigning a type to the variable 'if_condition_135916' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'if_condition_135916', if_condition_135916)
    # SSA begins for if statement (line 430)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 431):
    
    # Assigning a Call to a Name (line 431):
    
    # Call to dict(...): (line 431)
    # Processing the call arguments (line 431)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 431, 22, True)
    # Calculating comprehension expression
    
    # Call to items(...): (line 431)
    # Processing the call keyword arguments (line 431)
    kwargs_135930 = {}
    # Getting the type of 'source' (line 431)
    source_135927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 53), 'source', False)
    # Obtaining the member '__dict__' of a type (line 431)
    dict___135928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 53), source_135927, '__dict__')
    # Obtaining the member 'items' of a type (line 431)
    items_135929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 53), dict___135928, 'items')
    # Calling items(args, kwargs) (line 431)
    items_call_result_135931 = invoke(stypy.reporting.localization.Localization(__file__, 431, 53), items_135929, *[], **kwargs_135930)
    
    comprehension_135932 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 22), items_call_result_135931)
    # Assigning a type to the variable 'key' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 22), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 22), comprehension_135932))
    # Assigning a type to the variable 'value' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 22), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 22), comprehension_135932))
    
    
    # Call to startswith(...): (line 432)
    # Processing the call arguments (line 432)
    str_135923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 44), 'str', '_')
    # Processing the call keyword arguments (line 432)
    kwargs_135924 = {}
    # Getting the type of 'key' (line 432)
    key_135921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 29), 'key', False)
    # Obtaining the member 'startswith' of a type (line 432)
    startswith_135922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 29), key_135921, 'startswith')
    # Calling startswith(args, kwargs) (line 432)
    startswith_call_result_135925 = invoke(stypy.reporting.localization.Localization(__file__, 432, 29), startswith_135922, *[str_135923], **kwargs_135924)
    
    # Applying the 'not' unary operator (line 432)
    result_not__135926 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 25), 'not', startswith_call_result_135925)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 431)
    tuple_135918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 431)
    # Adding element type (line 431)
    # Getting the type of 'key' (line 431)
    key_135919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'key', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 23), tuple_135918, key_135919)
    # Adding element type (line 431)
    # Getting the type of 'value' (line 431)
    value_135920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 28), 'value', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 23), tuple_135918, value_135920)
    
    list_135933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 22), list_135933, tuple_135918)
    # Processing the call keyword arguments (line 431)
    kwargs_135934 = {}
    # Getting the type of 'dict' (line 431)
    dict_135917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 17), 'dict', False)
    # Calling dict(args, kwargs) (line 431)
    dict_call_result_135935 = invoke(stypy.reporting.localization.Localization(__file__, 431, 17), dict_135917, *[list_135933], **kwargs_135934)
    
    # Assigning a type to the variable 'source' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'source', dict_call_result_135935)
    
    # Assigning a Name to a Name (line 433):
    
    # Assigning a Name to a Name (line 433):
    # Getting the type of 'True' (line 433)
    True_135936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 21), 'True')
    # Assigning a type to the variable 'is_mapping' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'is_mapping', True_135936)
    # SSA join for if statement (line 430)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'is_mapping' (line 434)
    is_mapping_135937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 7), 'is_mapping')
    # Testing the type of an if condition (line 434)
    if_condition_135938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 4), is_mapping_135937)
    # Assigning a type to the variable 'if_condition_135938' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'if_condition_135938', if_condition_135938)
    # SSA begins for if statement (line 434)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 435):
    
    # Assigning a List to a Name (line 435):
    
    # Obtaining an instance of the builtin type 'list' (line 435)
    list_135939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 435)
    
    # Assigning a type to the variable 'dtype' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'dtype', list_135939)
    
    # Assigning a List to a Name (line 436):
    
    # Assigning a List to a Name (line 436):
    
    # Obtaining an instance of the builtin type 'list' (line 436)
    list_135940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 436)
    
    # Assigning a type to the variable 'values' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'values', list_135940)
    
    
    # Call to items(...): (line 437)
    # Processing the call keyword arguments (line 437)
    kwargs_135943 = {}
    # Getting the type of 'source' (line 437)
    source_135941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 28), 'source', False)
    # Obtaining the member 'items' of a type (line 437)
    items_135942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 28), source_135941, 'items')
    # Calling items(args, kwargs) (line 437)
    items_call_result_135944 = invoke(stypy.reporting.localization.Localization(__file__, 437, 28), items_135942, *[], **kwargs_135943)
    
    # Testing the type of a for loop iterable (line 437)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 437, 8), items_call_result_135944)
    # Getting the type of the for loop variable (line 437)
    for_loop_var_135945 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 437, 8), items_call_result_135944)
    # Assigning a type to the variable 'field' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'field', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 8), for_loop_var_135945))
    # Assigning a type to the variable 'value' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 8), for_loop_var_135945))
    # SSA begins for a for statement (line 437)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'field' (line 438)
    field_135947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'field', False)
    # Getting the type of 'string_types' (line 438)
    string_types_135948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 34), 'string_types', False)
    # Processing the call keyword arguments (line 438)
    kwargs_135949 = {}
    # Getting the type of 'isinstance' (line 438)
    isinstance_135946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 438)
    isinstance_call_result_135950 = invoke(stypy.reporting.localization.Localization(__file__, 438, 16), isinstance_135946, *[field_135947, string_types_135948], **kwargs_135949)
    
    
    
    # Obtaining the type of the subscript
    int_135951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 26), 'int')
    # Getting the type of 'field' (line 439)
    field_135952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'field')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___135953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 20), field_135952, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_135954 = invoke(stypy.reporting.localization.Localization(__file__, 439, 20), getitem___135953, int_135951)
    
    str_135955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 36), 'str', '_0123456789')
    # Applying the binary operator 'notin' (line 439)
    result_contains_135956 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 20), 'notin', subscript_call_result_135954, str_135955)
    
    # Applying the binary operator 'and' (line 438)
    result_and_keyword_135957 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 16), 'and', isinstance_call_result_135950, result_contains_135956)
    
    # Testing the type of an if condition (line 438)
    if_condition_135958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 12), result_and_keyword_135957)
    # Assigning a type to the variable 'if_condition_135958' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'if_condition_135958', if_condition_135958)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 440)
    # Processing the call arguments (line 440)
    
    # Obtaining an instance of the builtin type 'tuple' (line 440)
    tuple_135961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 440)
    # Adding element type (line 440)
    # Getting the type of 'field' (line 440)
    field_135962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 30), 'field', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 30), tuple_135961, field_135962)
    # Adding element type (line 440)
    # Getting the type of 'object' (line 440)
    object_135963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 37), 'object', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 30), tuple_135961, object_135963)
    
    # Processing the call keyword arguments (line 440)
    kwargs_135964 = {}
    # Getting the type of 'dtype' (line 440)
    dtype_135959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'dtype', False)
    # Obtaining the member 'append' of a type (line 440)
    append_135960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 16), dtype_135959, 'append')
    # Calling append(args, kwargs) (line 440)
    append_call_result_135965 = invoke(stypy.reporting.localization.Localization(__file__, 440, 16), append_135960, *[tuple_135961], **kwargs_135964)
    
    
    # Call to append(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'value' (line 441)
    value_135968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 30), 'value', False)
    # Processing the call keyword arguments (line 441)
    kwargs_135969 = {}
    # Getting the type of 'values' (line 441)
    values_135966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 16), 'values', False)
    # Obtaining the member 'append' of a type (line 441)
    append_135967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 16), values_135966, 'append')
    # Calling append(args, kwargs) (line 441)
    append_call_result_135970 = invoke(stypy.reporting.localization.Localization(__file__, 441, 16), append_135967, *[value_135968], **kwargs_135969)
    
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dtype' (line 442)
    dtype_135971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), 'dtype')
    # Testing the type of an if condition (line 442)
    if_condition_135972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 8), dtype_135971)
    # Assigning a type to the variable 'if_condition_135972' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'if_condition_135972', if_condition_135972)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 443)
    # Processing the call arguments (line 443)
    
    # Obtaining an instance of the builtin type 'list' (line 443)
    list_135975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 443)
    # Adding element type (line 443)
    
    # Call to tuple(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'values' (line 443)
    values_135977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 35), 'values', False)
    # Processing the call keyword arguments (line 443)
    kwargs_135978 = {}
    # Getting the type of 'tuple' (line 443)
    tuple_135976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 29), 'tuple', False)
    # Calling tuple(args, kwargs) (line 443)
    tuple_call_result_135979 = invoke(stypy.reporting.localization.Localization(__file__, 443, 29), tuple_135976, *[values_135977], **kwargs_135978)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 28), list_135975, tuple_call_result_135979)
    
    # Getting the type of 'dtype' (line 443)
    dtype_135980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 45), 'dtype', False)
    # Processing the call keyword arguments (line 443)
    kwargs_135981 = {}
    # Getting the type of 'np' (line 443)
    np_135973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 19), 'np', False)
    # Obtaining the member 'array' of a type (line 443)
    array_135974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 19), np_135973, 'array')
    # Calling array(args, kwargs) (line 443)
    array_call_result_135982 = invoke(stypy.reporting.localization.Localization(__file__, 443, 19), array_135974, *[list_135975, dtype_135980], **kwargs_135981)
    
    # Assigning a type to the variable 'stypy_return_type' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'stypy_return_type', array_call_result_135982)
    # SSA branch for the else part of an if statement (line 442)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'EmptyStructMarker' (line 445)
    EmptyStructMarker_135983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 19), 'EmptyStructMarker')
    # Assigning a type to the variable 'stypy_return_type' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'stypy_return_type', EmptyStructMarker_135983)
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 434)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to asanyarray(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'source' (line 447)
    source_135986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 25), 'source', False)
    # Processing the call keyword arguments (line 447)
    kwargs_135987 = {}
    # Getting the type of 'np' (line 447)
    np_135984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 447)
    asanyarray_135985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 11), np_135984, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 447)
    asanyarray_call_result_135988 = invoke(stypy.reporting.localization.Localization(__file__, 447, 11), asanyarray_135985, *[source_135986], **kwargs_135987)
    
    # Assigning a type to the variable 'narr' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'narr', asanyarray_call_result_135988)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'narr' (line 448)
    narr_135989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 7), 'narr')
    # Obtaining the member 'dtype' of a type (line 448)
    dtype_135990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 7), narr_135989, 'dtype')
    # Obtaining the member 'type' of a type (line 448)
    type_135991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 7), dtype_135990, 'type')
    
    # Obtaining an instance of the builtin type 'tuple' (line 448)
    tuple_135992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 448)
    # Adding element type (line 448)
    # Getting the type of 'object' (line 448)
    object_135993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 27), 'object')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 27), tuple_135992, object_135993)
    # Adding element type (line 448)
    # Getting the type of 'np' (line 448)
    np_135994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 35), 'np')
    # Obtaining the member 'object_' of a type (line 448)
    object__135995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 35), np_135994, 'object_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 27), tuple_135992, object__135995)
    
    # Applying the binary operator 'in' (line 448)
    result_contains_135996 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 7), 'in', type_135991, tuple_135992)
    
    
    # Getting the type of 'narr' (line 449)
    narr_135997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 7), 'narr')
    # Obtaining the member 'shape' of a type (line 449)
    shape_135998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 7), narr_135997, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 449)
    tuple_135999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 449)
    
    # Applying the binary operator '==' (line 449)
    result_eq_136000 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 7), '==', shape_135998, tuple_135999)
    
    # Applying the binary operator 'and' (line 448)
    result_and_keyword_136001 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 7), 'and', result_contains_135996, result_eq_136000)
    
    # Getting the type of 'narr' (line 449)
    narr_136002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 28), 'narr')
    # Getting the type of 'source' (line 449)
    source_136003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 36), 'source')
    # Applying the binary operator '==' (line 449)
    result_eq_136004 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 28), '==', narr_136002, source_136003)
    
    # Applying the binary operator 'and' (line 448)
    result_and_keyword_136005 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 7), 'and', result_and_keyword_136001, result_eq_136004)
    
    # Testing the type of an if condition (line 448)
    if_condition_136006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 4), result_and_keyword_136005)
    # Assigning a type to the variable 'if_condition_136006' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'if_condition_136006', if_condition_136006)
    # SSA begins for if statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 451)
    None_136007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'stypy_return_type', None_136007)
    # SSA join for if statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'narr' (line 452)
    narr_136008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 11), 'narr')
    # Assigning a type to the variable 'stypy_return_type' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type', narr_136008)
    
    # ################# End of 'to_writeable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_writeable' in the type store
    # Getting the type of 'stypy_return_type' (line 407)
    stypy_return_type_136009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_136009)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_writeable'
    return stypy_return_type_136009

# Assigning a type to the variable 'to_writeable' (line 407)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 0), 'to_writeable', to_writeable)

# Assigning a Subscript to a Name (line 456):

# Assigning a Subscript to a Name (line 456):

# Obtaining the type of the subscript
str_136010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 46), 'str', 'file_header')

# Obtaining the type of the subscript
str_136011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 36), 'str', 'dtypes')

# Obtaining the type of the subscript
# Getting the type of 'native_code' (line 456)
native_code_136012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'native_code')
# Getting the type of 'MDTYPES' (line 456)
MDTYPES_136013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'MDTYPES')
# Obtaining the member '__getitem__' of a type (line 456)
getitem___136014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 15), MDTYPES_136013, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 456)
subscript_call_result_136015 = invoke(stypy.reporting.localization.Localization(__file__, 456, 15), getitem___136014, native_code_136012)

# Obtaining the member '__getitem__' of a type (line 456)
getitem___136016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 15), subscript_call_result_136015, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 456)
subscript_call_result_136017 = invoke(stypy.reporting.localization.Localization(__file__, 456, 15), getitem___136016, str_136011)

# Obtaining the member '__getitem__' of a type (line 456)
getitem___136018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 15), subscript_call_result_136017, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 456)
subscript_call_result_136019 = invoke(stypy.reporting.localization.Localization(__file__, 456, 15), getitem___136018, str_136010)

# Assigning a type to the variable 'NDT_FILE_HDR' (line 456)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'NDT_FILE_HDR', subscript_call_result_136019)

# Assigning a Subscript to a Name (line 457):

# Assigning a Subscript to a Name (line 457):

# Obtaining the type of the subscript
str_136020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 46), 'str', 'tag_full')

# Obtaining the type of the subscript
str_136021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 36), 'str', 'dtypes')

# Obtaining the type of the subscript
# Getting the type of 'native_code' (line 457)
native_code_136022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'native_code')
# Getting the type of 'MDTYPES' (line 457)
MDTYPES_136023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'MDTYPES')
# Obtaining the member '__getitem__' of a type (line 457)
getitem___136024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), MDTYPES_136023, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 457)
subscript_call_result_136025 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), getitem___136024, native_code_136022)

# Obtaining the member '__getitem__' of a type (line 457)
getitem___136026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), subscript_call_result_136025, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 457)
subscript_call_result_136027 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), getitem___136026, str_136021)

# Obtaining the member '__getitem__' of a type (line 457)
getitem___136028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), subscript_call_result_136027, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 457)
subscript_call_result_136029 = invoke(stypy.reporting.localization.Localization(__file__, 457, 15), getitem___136028, str_136020)

# Assigning a type to the variable 'NDT_TAG_FULL' (line 457)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 0), 'NDT_TAG_FULL', subscript_call_result_136029)

# Assigning a Subscript to a Name (line 458):

# Assigning a Subscript to a Name (line 458):

# Obtaining the type of the subscript
str_136030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 47), 'str', 'tag_smalldata')

# Obtaining the type of the subscript
str_136031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 37), 'str', 'dtypes')

# Obtaining the type of the subscript
# Getting the type of 'native_code' (line 458)
native_code_136032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'native_code')
# Getting the type of 'MDTYPES' (line 458)
MDTYPES_136033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'MDTYPES')
# Obtaining the member '__getitem__' of a type (line 458)
getitem___136034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 16), MDTYPES_136033, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 458)
subscript_call_result_136035 = invoke(stypy.reporting.localization.Localization(__file__, 458, 16), getitem___136034, native_code_136032)

# Obtaining the member '__getitem__' of a type (line 458)
getitem___136036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 16), subscript_call_result_136035, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 458)
subscript_call_result_136037 = invoke(stypy.reporting.localization.Localization(__file__, 458, 16), getitem___136036, str_136031)

# Obtaining the member '__getitem__' of a type (line 458)
getitem___136038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 16), subscript_call_result_136037, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 458)
subscript_call_result_136039 = invoke(stypy.reporting.localization.Localization(__file__, 458, 16), getitem___136038, str_136030)

# Assigning a type to the variable 'NDT_TAG_SMALL' (line 458)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), 'NDT_TAG_SMALL', subscript_call_result_136039)

# Assigning a Subscript to a Name (line 459):

# Assigning a Subscript to a Name (line 459):

# Obtaining the type of the subscript
str_136040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 49), 'str', 'array_flags')

# Obtaining the type of the subscript
str_136041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 39), 'str', 'dtypes')

# Obtaining the type of the subscript
# Getting the type of 'native_code' (line 459)
native_code_136042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 26), 'native_code')
# Getting the type of 'MDTYPES' (line 459)
MDTYPES_136043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 18), 'MDTYPES')
# Obtaining the member '__getitem__' of a type (line 459)
getitem___136044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 18), MDTYPES_136043, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 459)
subscript_call_result_136045 = invoke(stypy.reporting.localization.Localization(__file__, 459, 18), getitem___136044, native_code_136042)

# Obtaining the member '__getitem__' of a type (line 459)
getitem___136046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 18), subscript_call_result_136045, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 459)
subscript_call_result_136047 = invoke(stypy.reporting.localization.Localization(__file__, 459, 18), getitem___136046, str_136041)

# Obtaining the member '__getitem__' of a type (line 459)
getitem___136048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 18), subscript_call_result_136047, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 459)
subscript_call_result_136049 = invoke(stypy.reporting.localization.Localization(__file__, 459, 18), getitem___136048, str_136040)

# Assigning a type to the variable 'NDT_ARRAY_FLAGS' (line 459)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 0), 'NDT_ARRAY_FLAGS', subscript_call_result_136049)
# Declaration of the 'VarWriter5' class

class VarWriter5(object, ):
    str_136050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 4), 'str', ' Generic matlab matrix writing class ')
    
    # Assigning a Call to a Name (line 464):
    
    # Assigning a Name to a Subscript (line 465):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.__init__', ['file_writer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_writer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 468):
        
        # Assigning a Attribute to a Attribute (line 468):
        # Getting the type of 'file_writer' (line 468)
        file_writer_136051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 27), 'file_writer')
        # Obtaining the member 'file_stream' of a type (line 468)
        file_stream_136052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 27), file_writer_136051, 'file_stream')
        # Getting the type of 'self' (line 468)
        self_136053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'self')
        # Setting the type of the member 'file_stream' of a type (line 468)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), self_136053, 'file_stream', file_stream_136052)
        
        # Assigning a Attribute to a Attribute (line 469):
        
        # Assigning a Attribute to a Attribute (line 469):
        # Getting the type of 'file_writer' (line 469)
        file_writer_136054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 31), 'file_writer')
        # Obtaining the member 'unicode_strings' of a type (line 469)
        unicode_strings_136055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 31), file_writer_136054, 'unicode_strings')
        # Getting the type of 'self' (line 469)
        self_136056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'self')
        # Setting the type of the member 'unicode_strings' of a type (line 469)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), self_136056, 'unicode_strings', unicode_strings_136055)
        
        # Assigning a Attribute to a Attribute (line 470):
        
        # Assigning a Attribute to a Attribute (line 470):
        # Getting the type of 'file_writer' (line 470)
        file_writer_136057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 32), 'file_writer')
        # Obtaining the member 'long_field_names' of a type (line 470)
        long_field_names_136058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 32), file_writer_136057, 'long_field_names')
        # Getting the type of 'self' (line 470)
        self_136059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'self')
        # Setting the type of the member 'long_field_names' of a type (line 470)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 8), self_136059, 'long_field_names', long_field_names_136058)
        
        # Assigning a Attribute to a Attribute (line 471):
        
        # Assigning a Attribute to a Attribute (line 471):
        # Getting the type of 'file_writer' (line 471)
        file_writer_136060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 23), 'file_writer')
        # Obtaining the member 'oned_as' of a type (line 471)
        oned_as_136061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 23), file_writer_136060, 'oned_as')
        # Getting the type of 'self' (line 471)
        self_136062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'self')
        # Setting the type of the member 'oned_as' of a type (line 471)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 8), self_136062, 'oned_as', oned_as_136061)
        
        # Assigning a Name to a Attribute (line 473):
        
        # Assigning a Name to a Attribute (line 473):
        # Getting the type of 'None' (line 473)
        None_136063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 25), 'None')
        # Getting the type of 'self' (line 473)
        self_136064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'self')
        # Setting the type of the member '_var_name' of a type (line 473)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), self_136064, '_var_name', None_136063)
        
        # Assigning a Name to a Attribute (line 474):
        
        # Assigning a Name to a Attribute (line 474):
        # Getting the type of 'False' (line 474)
        False_136065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 30), 'False')
        # Getting the type of 'self' (line 474)
        self_136066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'self')
        # Setting the type of the member '_var_is_global' of a type (line 474)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), self_136066, '_var_is_global', False_136065)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def write_bytes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_bytes'
        module_type_store = module_type_store.open_function_context('write_bytes', 476, 4, False)
        # Assigning a type to the variable 'self' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_bytes')
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_bytes.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_bytes', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_bytes', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_bytes(...)' code ##################

        
        # Call to write(...): (line 477)
        # Processing the call arguments (line 477)
        
        # Call to tostring(...): (line 477)
        # Processing the call keyword arguments (line 477)
        str_136072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 50), 'str', 'F')
        keyword_136073 = str_136072
        kwargs_136074 = {'order': keyword_136073}
        # Getting the type of 'arr' (line 477)
        arr_136070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 31), 'arr', False)
        # Obtaining the member 'tostring' of a type (line 477)
        tostring_136071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 31), arr_136070, 'tostring')
        # Calling tostring(args, kwargs) (line 477)
        tostring_call_result_136075 = invoke(stypy.reporting.localization.Localization(__file__, 477, 31), tostring_136071, *[], **kwargs_136074)
        
        # Processing the call keyword arguments (line 477)
        kwargs_136076 = {}
        # Getting the type of 'self' (line 477)
        self_136067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 477)
        file_stream_136068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), self_136067, 'file_stream')
        # Obtaining the member 'write' of a type (line 477)
        write_136069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), file_stream_136068, 'write')
        # Calling write(args, kwargs) (line 477)
        write_call_result_136077 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), write_136069, *[tostring_call_result_136075], **kwargs_136076)
        
        
        # ################# End of 'write_bytes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_bytes' in the type store
        # Getting the type of 'stypy_return_type' (line 476)
        stypy_return_type_136078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136078)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_bytes'
        return stypy_return_type_136078


    @norecursion
    def write_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_string'
        module_type_store = module_type_store.open_function_context('write_string', 479, 4, False)
        # Assigning a type to the variable 'self' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_string.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_string.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_string')
        VarWriter5.write_string.__dict__.__setitem__('stypy_param_names_list', ['s'])
        VarWriter5.write_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_string.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_string', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_string', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_string(...)' code ##################

        
        # Call to write(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 's' (line 480)
        s_136082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 31), 's', False)
        # Processing the call keyword arguments (line 480)
        kwargs_136083 = {}
        # Getting the type of 'self' (line 480)
        self_136079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 480)
        file_stream_136080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_136079, 'file_stream')
        # Obtaining the member 'write' of a type (line 480)
        write_136081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), file_stream_136080, 'write')
        # Calling write(args, kwargs) (line 480)
        write_call_result_136084 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), write_136081, *[s_136082], **kwargs_136083)
        
        
        # ################# End of 'write_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_string' in the type store
        # Getting the type of 'stypy_return_type' (line 479)
        stypy_return_type_136085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_string'
        return stypy_return_type_136085


    @norecursion
    def write_element(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 482)
        None_136086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 40), 'None')
        defaults = [None_136086]
        # Create a new context for function 'write_element'
        module_type_store = module_type_store.open_function_context('write_element', 482, 4, False)
        # Assigning a type to the variable 'self' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_element.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_element.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_element.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_element.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_element')
        VarWriter5.write_element.__dict__.__setitem__('stypy_param_names_list', ['arr', 'mdtype'])
        VarWriter5.write_element.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_element.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_element.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_element.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_element.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_element.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_element', ['arr', 'mdtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_element', localization, ['arr', 'mdtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_element(...)' code ##################

        str_136087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 8), 'str', ' write tag and data ')
        
        # Type idiom detected: calculating its left and rigth part (line 484)
        # Getting the type of 'mdtype' (line 484)
        mdtype_136088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'mdtype')
        # Getting the type of 'None' (line 484)
        None_136089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 21), 'None')
        
        (may_be_136090, more_types_in_union_136091) = may_be_none(mdtype_136088, None_136089)

        if may_be_136090:

            if more_types_in_union_136091:
                # Runtime conditional SSA (line 484)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 485):
            
            # Assigning a Subscript to a Name (line 485):
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_136092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 48), 'int')
            slice_136093 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 485, 34), int_136092, None, None)
            # Getting the type of 'arr' (line 485)
            arr_136094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 34), 'arr')
            # Obtaining the member 'dtype' of a type (line 485)
            dtype_136095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 34), arr_136094, 'dtype')
            # Obtaining the member 'str' of a type (line 485)
            str_136096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 34), dtype_136095, 'str')
            # Obtaining the member '__getitem__' of a type (line 485)
            getitem___136097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 34), str_136096, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 485)
            subscript_call_result_136098 = invoke(stypy.reporting.localization.Localization(__file__, 485, 34), getitem___136097, slice_136093)
            
            # Getting the type of 'NP_TO_MTYPES' (line 485)
            NP_TO_MTYPES_136099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 21), 'NP_TO_MTYPES')
            # Obtaining the member '__getitem__' of a type (line 485)
            getitem___136100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 21), NP_TO_MTYPES_136099, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 485)
            subscript_call_result_136101 = invoke(stypy.reporting.localization.Localization(__file__, 485, 21), getitem___136100, subscript_call_result_136098)
            
            # Assigning a type to the variable 'mdtype' (line 485)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'mdtype', subscript_call_result_136101)

            if more_types_in_union_136091:
                # SSA join for if statement (line 484)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'arr' (line 487)
        arr_136102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 11), 'arr')
        # Obtaining the member 'dtype' of a type (line 487)
        dtype_136103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 11), arr_136102, 'dtype')
        # Obtaining the member 'byteorder' of a type (line 487)
        byteorder_136104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 11), dtype_136103, 'byteorder')
        # Getting the type of 'swapped_code' (line 487)
        swapped_code_136105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 34), 'swapped_code')
        # Applying the binary operator '==' (line 487)
        result_eq_136106 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 11), '==', byteorder_136104, swapped_code_136105)
        
        # Testing the type of an if condition (line 487)
        if_condition_136107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 8), result_eq_136106)
        # Assigning a type to the variable 'if_condition_136107' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'if_condition_136107', if_condition_136107)
        # SSA begins for if statement (line 487)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 488):
        
        # Assigning a Call to a Name (line 488):
        
        # Call to newbyteorder(...): (line 488)
        # Processing the call keyword arguments (line 488)
        kwargs_136113 = {}
        
        # Call to byteswap(...): (line 488)
        # Processing the call keyword arguments (line 488)
        kwargs_136110 = {}
        # Getting the type of 'arr' (line 488)
        arr_136108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 18), 'arr', False)
        # Obtaining the member 'byteswap' of a type (line 488)
        byteswap_136109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 18), arr_136108, 'byteswap')
        # Calling byteswap(args, kwargs) (line 488)
        byteswap_call_result_136111 = invoke(stypy.reporting.localization.Localization(__file__, 488, 18), byteswap_136109, *[], **kwargs_136110)
        
        # Obtaining the member 'newbyteorder' of a type (line 488)
        newbyteorder_136112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 18), byteswap_call_result_136111, 'newbyteorder')
        # Calling newbyteorder(args, kwargs) (line 488)
        newbyteorder_call_result_136114 = invoke(stypy.reporting.localization.Localization(__file__, 488, 18), newbyteorder_136112, *[], **kwargs_136113)
        
        # Assigning a type to the variable 'arr' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'arr', newbyteorder_call_result_136114)
        # SSA join for if statement (line 487)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 489):
        
        # Assigning a BinOp to a Name (line 489):
        # Getting the type of 'arr' (line 489)
        arr_136115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 21), 'arr')
        # Obtaining the member 'size' of a type (line 489)
        size_136116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 21), arr_136115, 'size')
        # Getting the type of 'arr' (line 489)
        arr_136117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 30), 'arr')
        # Obtaining the member 'itemsize' of a type (line 489)
        itemsize_136118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 30), arr_136117, 'itemsize')
        # Applying the binary operator '*' (line 489)
        result_mul_136119 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 21), '*', size_136116, itemsize_136118)
        
        # Assigning a type to the variable 'byte_count' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'byte_count', result_mul_136119)
        
        
        # Getting the type of 'byte_count' (line 490)
        byte_count_136120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 11), 'byte_count')
        int_136121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 25), 'int')
        # Applying the binary operator '<=' (line 490)
        result_le_136122 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), '<=', byte_count_136120, int_136121)
        
        # Testing the type of an if condition (line 490)
        if_condition_136123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_le_136122)
        # Assigning a type to the variable 'if_condition_136123' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_136123', if_condition_136123)
        # SSA begins for if statement (line 490)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_smalldata_element(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'arr' (line 491)
        arr_136126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 41), 'arr', False)
        # Getting the type of 'mdtype' (line 491)
        mdtype_136127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 46), 'mdtype', False)
        # Getting the type of 'byte_count' (line 491)
        byte_count_136128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 54), 'byte_count', False)
        # Processing the call keyword arguments (line 491)
        kwargs_136129 = {}
        # Getting the type of 'self' (line 491)
        self_136124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'self', False)
        # Obtaining the member 'write_smalldata_element' of a type (line 491)
        write_smalldata_element_136125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 12), self_136124, 'write_smalldata_element')
        # Calling write_smalldata_element(args, kwargs) (line 491)
        write_smalldata_element_call_result_136130 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), write_smalldata_element_136125, *[arr_136126, mdtype_136127, byte_count_136128], **kwargs_136129)
        
        # SSA branch for the else part of an if statement (line 490)
        module_type_store.open_ssa_branch('else')
        
        # Call to write_regular_element(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'arr' (line 493)
        arr_136133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 39), 'arr', False)
        # Getting the type of 'mdtype' (line 493)
        mdtype_136134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 44), 'mdtype', False)
        # Getting the type of 'byte_count' (line 493)
        byte_count_136135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 52), 'byte_count', False)
        # Processing the call keyword arguments (line 493)
        kwargs_136136 = {}
        # Getting the type of 'self' (line 493)
        self_136131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'self', False)
        # Obtaining the member 'write_regular_element' of a type (line 493)
        write_regular_element_136132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), self_136131, 'write_regular_element')
        # Calling write_regular_element(args, kwargs) (line 493)
        write_regular_element_call_result_136137 = invoke(stypy.reporting.localization.Localization(__file__, 493, 12), write_regular_element_136132, *[arr_136133, mdtype_136134, byte_count_136135], **kwargs_136136)
        
        # SSA join for if statement (line 490)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_element(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_element' in the type store
        # Getting the type of 'stypy_return_type' (line 482)
        stypy_return_type_136138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136138)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_element'
        return stypy_return_type_136138


    @norecursion
    def write_smalldata_element(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_smalldata_element'
        module_type_store = module_type_store.open_function_context('write_smalldata_element', 495, 4, False)
        # Assigning a type to the variable 'self' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_smalldata_element')
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_param_names_list', ['arr', 'mdtype', 'byte_count'])
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_smalldata_element.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_smalldata_element', ['arr', 'mdtype', 'byte_count'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_smalldata_element', localization, ['arr', 'mdtype', 'byte_count'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_smalldata_element(...)' code ##################

        
        # Assigning a Call to a Name (line 497):
        
        # Assigning a Call to a Name (line 497):
        
        # Call to zeros(...): (line 497)
        # Processing the call arguments (line 497)
        
        # Obtaining an instance of the builtin type 'tuple' (line 497)
        tuple_136141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 497)
        
        # Getting the type of 'NDT_TAG_SMALL' (line 497)
        NDT_TAG_SMALL_136142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 27), 'NDT_TAG_SMALL', False)
        # Processing the call keyword arguments (line 497)
        kwargs_136143 = {}
        # Getting the type of 'np' (line 497)
        np_136139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 14), 'np', False)
        # Obtaining the member 'zeros' of a type (line 497)
        zeros_136140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 14), np_136139, 'zeros')
        # Calling zeros(args, kwargs) (line 497)
        zeros_call_result_136144 = invoke(stypy.reporting.localization.Localization(__file__, 497, 14), zeros_136140, *[tuple_136141, NDT_TAG_SMALL_136142], **kwargs_136143)
        
        # Assigning a type to the variable 'tag' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'tag', zeros_call_result_136144)
        
        # Assigning a BinOp to a Subscript (line 498):
        
        # Assigning a BinOp to a Subscript (line 498):
        # Getting the type of 'byte_count' (line 498)
        byte_count_136145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 36), 'byte_count')
        int_136146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 50), 'int')
        # Applying the binary operator '<<' (line 498)
        result_lshift_136147 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 36), '<<', byte_count_136145, int_136146)
        
        # Getting the type of 'mdtype' (line 498)
        mdtype_136148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 56), 'mdtype')
        # Applying the binary operator '+' (line 498)
        result_add_136149 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 35), '+', result_lshift_136147, mdtype_136148)
        
        # Getting the type of 'tag' (line 498)
        tag_136150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tag')
        str_136151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 12), 'str', 'byte_count_mdtype')
        # Storing an element on a container (line 498)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 8), tag_136150, (str_136151, result_add_136149))
        
        # Assigning a Call to a Subscript (line 500):
        
        # Assigning a Call to a Subscript (line 500):
        
        # Call to tostring(...): (line 500)
        # Processing the call keyword arguments (line 500)
        str_136154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 41), 'str', 'F')
        keyword_136155 = str_136154
        kwargs_136156 = {'order': keyword_136155}
        # Getting the type of 'arr' (line 500)
        arr_136152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 22), 'arr', False)
        # Obtaining the member 'tostring' of a type (line 500)
        tostring_136153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 22), arr_136152, 'tostring')
        # Calling tostring(args, kwargs) (line 500)
        tostring_call_result_136157 = invoke(stypy.reporting.localization.Localization(__file__, 500, 22), tostring_136153, *[], **kwargs_136156)
        
        # Getting the type of 'tag' (line 500)
        tag_136158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tag')
        str_136159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 12), 'str', 'data')
        # Storing an element on a container (line 500)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 8), tag_136158, (str_136159, tostring_call_result_136157))
        
        # Call to write_bytes(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'tag' (line 501)
        tag_136162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 25), 'tag', False)
        # Processing the call keyword arguments (line 501)
        kwargs_136163 = {}
        # Getting the type of 'self' (line 501)
        self_136160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 501)
        write_bytes_136161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), self_136160, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 501)
        write_bytes_call_result_136164 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), write_bytes_136161, *[tag_136162], **kwargs_136163)
        
        
        # ################# End of 'write_smalldata_element(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_smalldata_element' in the type store
        # Getting the type of 'stypy_return_type' (line 495)
        stypy_return_type_136165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_smalldata_element'
        return stypy_return_type_136165


    @norecursion
    def write_regular_element(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_regular_element'
        module_type_store = module_type_store.open_function_context('write_regular_element', 503, 4, False)
        # Assigning a type to the variable 'self' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_regular_element')
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_param_names_list', ['arr', 'mdtype', 'byte_count'])
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_regular_element.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_regular_element', ['arr', 'mdtype', 'byte_count'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_regular_element', localization, ['arr', 'mdtype', 'byte_count'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_regular_element(...)' code ##################

        
        # Assigning a Call to a Name (line 505):
        
        # Assigning a Call to a Name (line 505):
        
        # Call to zeros(...): (line 505)
        # Processing the call arguments (line 505)
        
        # Obtaining an instance of the builtin type 'tuple' (line 505)
        tuple_136168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 505)
        
        # Getting the type of 'NDT_TAG_FULL' (line 505)
        NDT_TAG_FULL_136169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 27), 'NDT_TAG_FULL', False)
        # Processing the call keyword arguments (line 505)
        kwargs_136170 = {}
        # Getting the type of 'np' (line 505)
        np_136166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 14), 'np', False)
        # Obtaining the member 'zeros' of a type (line 505)
        zeros_136167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 14), np_136166, 'zeros')
        # Calling zeros(args, kwargs) (line 505)
        zeros_call_result_136171 = invoke(stypy.reporting.localization.Localization(__file__, 505, 14), zeros_136167, *[tuple_136168, NDT_TAG_FULL_136169], **kwargs_136170)
        
        # Assigning a type to the variable 'tag' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'tag', zeros_call_result_136171)
        
        # Assigning a Name to a Subscript (line 506):
        
        # Assigning a Name to a Subscript (line 506):
        # Getting the type of 'mdtype' (line 506)
        mdtype_136172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 24), 'mdtype')
        # Getting the type of 'tag' (line 506)
        tag_136173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'tag')
        str_136174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 12), 'str', 'mdtype')
        # Storing an element on a container (line 506)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 8), tag_136173, (str_136174, mdtype_136172))
        
        # Assigning a Name to a Subscript (line 507):
        
        # Assigning a Name to a Subscript (line 507):
        # Getting the type of 'byte_count' (line 507)
        byte_count_136175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 28), 'byte_count')
        # Getting the type of 'tag' (line 507)
        tag_136176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'tag')
        str_136177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 12), 'str', 'byte_count')
        # Storing an element on a container (line 507)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 8), tag_136176, (str_136177, byte_count_136175))
        
        # Call to write_bytes(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'tag' (line 508)
        tag_136180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 25), 'tag', False)
        # Processing the call keyword arguments (line 508)
        kwargs_136181 = {}
        # Getting the type of 'self' (line 508)
        self_136178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 508)
        write_bytes_136179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), self_136178, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 508)
        write_bytes_call_result_136182 = invoke(stypy.reporting.localization.Localization(__file__, 508, 8), write_bytes_136179, *[tag_136180], **kwargs_136181)
        
        
        # Call to write_bytes(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'arr' (line 509)
        arr_136185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 25), 'arr', False)
        # Processing the call keyword arguments (line 509)
        kwargs_136186 = {}
        # Getting the type of 'self' (line 509)
        self_136183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 509)
        write_bytes_136184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), self_136183, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 509)
        write_bytes_call_result_136187 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), write_bytes_136184, *[arr_136185], **kwargs_136186)
        
        
        # Assigning a BinOp to a Name (line 511):
        
        # Assigning a BinOp to a Name (line 511):
        # Getting the type of 'byte_count' (line 511)
        byte_count_136188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 19), 'byte_count')
        int_136189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 32), 'int')
        # Applying the binary operator '%' (line 511)
        result_mod_136190 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 19), '%', byte_count_136188, int_136189)
        
        # Assigning a type to the variable 'bc_mod_8' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'bc_mod_8', result_mod_136190)
        
        # Getting the type of 'bc_mod_8' (line 512)
        bc_mod_8_136191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'bc_mod_8')
        # Testing the type of an if condition (line 512)
        if_condition_136192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 8), bc_mod_8_136191)
        # Assigning a type to the variable 'if_condition_136192' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'if_condition_136192', if_condition_136192)
        # SSA begins for if statement (line 512)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 513)
        # Processing the call arguments (line 513)
        str_136196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 35), 'str', '\x00')
        int_136197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 46), 'int')
        # Getting the type of 'bc_mod_8' (line 513)
        bc_mod_8_136198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 48), 'bc_mod_8', False)
        # Applying the binary operator '-' (line 513)
        result_sub_136199 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 46), '-', int_136197, bc_mod_8_136198)
        
        # Applying the binary operator '*' (line 513)
        result_mul_136200 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 35), '*', str_136196, result_sub_136199)
        
        # Processing the call keyword arguments (line 513)
        kwargs_136201 = {}
        # Getting the type of 'self' (line 513)
        self_136193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 513)
        file_stream_136194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), self_136193, 'file_stream')
        # Obtaining the member 'write' of a type (line 513)
        write_136195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), file_stream_136194, 'write')
        # Calling write(args, kwargs) (line 513)
        write_call_result_136202 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), write_136195, *[result_mul_136200], **kwargs_136201)
        
        # SSA join for if statement (line 512)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_regular_element(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_regular_element' in the type store
        # Getting the type of 'stypy_return_type' (line 503)
        stypy_return_type_136203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136203)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_regular_element'
        return stypy_return_type_136203


    @norecursion
    def write_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 518)
        False_136204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 32), 'False')
        # Getting the type of 'False' (line 519)
        False_136205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 32), 'False')
        int_136206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 27), 'int')
        defaults = [False_136204, False_136205, int_136206]
        # Create a new context for function 'write_header'
        module_type_store = module_type_store.open_function_context('write_header', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_header.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_header.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_header')
        VarWriter5.write_header.__dict__.__setitem__('stypy_param_names_list', ['shape', 'mclass', 'is_complex', 'is_logical', 'nzmax'])
        VarWriter5.write_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_header.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_header', ['shape', 'mclass', 'is_complex', 'is_logical', 'nzmax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_header', localization, ['shape', 'mclass', 'is_complex', 'is_logical', 'nzmax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_header(...)' code ##################

        str_136207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, (-1)), 'str', " Write header for given data options\n        shape : sequence\n           array shape\n        mclass      - mat5 matrix class\n        is_complex  - True if matrix is complex\n        is_logical  - True if matrix is logical\n        nzmax        - max non zero elements for sparse arrays\n\n        We get the name and the global flag from the object, and reset\n        them to defaults after we've used them\n        ")
        
        # Assigning a Attribute to a Name (line 533):
        
        # Assigning a Attribute to a Name (line 533):
        # Getting the type of 'self' (line 533)
        self_136208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'self')
        # Obtaining the member '_var_name' of a type (line 533)
        _var_name_136209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 15), self_136208, '_var_name')
        # Assigning a type to the variable 'name' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'name', _var_name_136209)
        
        # Assigning a Attribute to a Name (line 534):
        
        # Assigning a Attribute to a Name (line 534):
        # Getting the type of 'self' (line 534)
        self_136210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 20), 'self')
        # Obtaining the member '_var_is_global' of a type (line 534)
        _var_is_global_136211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 20), self_136210, '_var_is_global')
        # Assigning a type to the variable 'is_global' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'is_global', _var_is_global_136211)
        
        # Assigning a Call to a Attribute (line 536):
        
        # Assigning a Call to a Attribute (line 536):
        
        # Call to tell(...): (line 536)
        # Processing the call keyword arguments (line 536)
        kwargs_136215 = {}
        # Getting the type of 'self' (line 536)
        self_136212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 28), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 536)
        file_stream_136213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 28), self_136212, 'file_stream')
        # Obtaining the member 'tell' of a type (line 536)
        tell_136214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 28), file_stream_136213, 'tell')
        # Calling tell(args, kwargs) (line 536)
        tell_call_result_136216 = invoke(stypy.reporting.localization.Localization(__file__, 536, 28), tell_136214, *[], **kwargs_136215)
        
        # Getting the type of 'self' (line 536)
        self_136217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'self')
        # Setting the type of the member '_mat_tag_pos' of a type (line 536)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), self_136217, '_mat_tag_pos', tell_call_result_136216)
        
        # Call to write_bytes(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'self' (line 537)
        self_136220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 25), 'self', False)
        # Obtaining the member 'mat_tag' of a type (line 537)
        mat_tag_136221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 25), self_136220, 'mat_tag')
        # Processing the call keyword arguments (line 537)
        kwargs_136222 = {}
        # Getting the type of 'self' (line 537)
        self_136218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 537)
        write_bytes_136219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), self_136218, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 537)
        write_bytes_call_result_136223 = invoke(stypy.reporting.localization.Localization(__file__, 537, 8), write_bytes_136219, *[mat_tag_136221], **kwargs_136222)
        
        
        # Assigning a Call to a Name (line 539):
        
        # Assigning a Call to a Name (line 539):
        
        # Call to zeros(...): (line 539)
        # Processing the call arguments (line 539)
        
        # Obtaining an instance of the builtin type 'tuple' (line 539)
        tuple_136226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 539)
        
        # Getting the type of 'NDT_ARRAY_FLAGS' (line 539)
        NDT_ARRAY_FLAGS_136227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 26), 'NDT_ARRAY_FLAGS', False)
        # Processing the call keyword arguments (line 539)
        kwargs_136228 = {}
        # Getting the type of 'np' (line 539)
        np_136224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 539)
        zeros_136225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 13), np_136224, 'zeros')
        # Calling zeros(args, kwargs) (line 539)
        zeros_call_result_136229 = invoke(stypy.reporting.localization.Localization(__file__, 539, 13), zeros_136225, *[tuple_136226, NDT_ARRAY_FLAGS_136227], **kwargs_136228)
        
        # Assigning a type to the variable 'af' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'af', zeros_call_result_136229)
        
        # Assigning a Name to a Subscript (line 540):
        
        # Assigning a Name to a Subscript (line 540):
        # Getting the type of 'miUINT32' (line 540)
        miUINT32_136230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 26), 'miUINT32')
        # Getting the type of 'af' (line 540)
        af_136231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'af')
        str_136232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 11), 'str', 'data_type')
        # Storing an element on a container (line 540)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 8), af_136231, (str_136232, miUINT32_136230))
        
        # Assigning a Num to a Subscript (line 541):
        
        # Assigning a Num to a Subscript (line 541):
        int_136233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 27), 'int')
        # Getting the type of 'af' (line 541)
        af_136234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'af')
        str_136235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 11), 'str', 'byte_count')
        # Storing an element on a container (line 541)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 8), af_136234, (str_136235, int_136233))
        
        # Assigning a BinOp to a Name (line 542):
        
        # Assigning a BinOp to a Name (line 542):
        # Getting the type of 'is_complex' (line 542)
        is_complex_136236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'is_complex')
        int_136237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 30), 'int')
        # Applying the binary operator '<<' (line 542)
        result_lshift_136238 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 16), '<<', is_complex_136236, int_136237)
        
        # Getting the type of 'is_global' (line 542)
        is_global_136239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 34), 'is_global')
        int_136240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 47), 'int')
        # Applying the binary operator '<<' (line 542)
        result_lshift_136241 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 34), '<<', is_global_136239, int_136240)
        
        # Applying the binary operator '|' (line 542)
        result_or__136242 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 16), '|', result_lshift_136238, result_lshift_136241)
        
        # Getting the type of 'is_logical' (line 542)
        is_logical_136243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 51), 'is_logical')
        int_136244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 65), 'int')
        # Applying the binary operator '<<' (line 542)
        result_lshift_136245 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 51), '<<', is_logical_136243, int_136244)
        
        # Applying the binary operator '|' (line 542)
        result_or__136246 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 49), '|', result_or__136242, result_lshift_136245)
        
        # Assigning a type to the variable 'flags' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'flags', result_or__136246)
        
        # Assigning a BinOp to a Subscript (line 543):
        
        # Assigning a BinOp to a Subscript (line 543):
        # Getting the type of 'mclass' (line 543)
        mclass_136247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 28), 'mclass')
        # Getting the type of 'flags' (line 543)
        flags_136248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 37), 'flags')
        int_136249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 46), 'int')
        # Applying the binary operator '<<' (line 543)
        result_lshift_136250 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 37), '<<', flags_136248, int_136249)
        
        # Applying the binary operator '|' (line 543)
        result_or__136251 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 28), '|', mclass_136247, result_lshift_136250)
        
        # Getting the type of 'af' (line 543)
        af_136252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'af')
        str_136253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 11), 'str', 'flags_class')
        # Storing an element on a container (line 543)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 8), af_136252, (str_136253, result_or__136251))
        
        # Assigning a Name to a Subscript (line 544):
        
        # Assigning a Name to a Subscript (line 544):
        # Getting the type of 'nzmax' (line 544)
        nzmax_136254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 22), 'nzmax')
        # Getting the type of 'af' (line 544)
        af_136255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'af')
        str_136256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 11), 'str', 'nzmax')
        # Storing an element on a container (line 544)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 8), af_136255, (str_136256, nzmax_136254))
        
        # Call to write_bytes(...): (line 545)
        # Processing the call arguments (line 545)
        # Getting the type of 'af' (line 545)
        af_136259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 25), 'af', False)
        # Processing the call keyword arguments (line 545)
        kwargs_136260 = {}
        # Getting the type of 'self' (line 545)
        self_136257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 545)
        write_bytes_136258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 8), self_136257, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 545)
        write_bytes_call_result_136261 = invoke(stypy.reporting.localization.Localization(__file__, 545, 8), write_bytes_136258, *[af_136259], **kwargs_136260)
        
        
        # Call to write_element(...): (line 547)
        # Processing the call arguments (line 547)
        
        # Call to array(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'shape' (line 547)
        shape_136266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 36), 'shape', False)
        # Processing the call keyword arguments (line 547)
        str_136267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 49), 'str', 'i4')
        keyword_136268 = str_136267
        kwargs_136269 = {'dtype': keyword_136268}
        # Getting the type of 'np' (line 547)
        np_136264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 547)
        array_136265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 27), np_136264, 'array')
        # Calling array(args, kwargs) (line 547)
        array_call_result_136270 = invoke(stypy.reporting.localization.Localization(__file__, 547, 27), array_136265, *[shape_136266], **kwargs_136269)
        
        # Processing the call keyword arguments (line 547)
        kwargs_136271 = {}
        # Getting the type of 'self' (line 547)
        self_136262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 547)
        write_element_136263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 8), self_136262, 'write_element')
        # Calling write_element(args, kwargs) (line 547)
        write_element_call_result_136272 = invoke(stypy.reporting.localization.Localization(__file__, 547, 8), write_element_136263, *[array_call_result_136270], **kwargs_136271)
        
        
        # Assigning a Call to a Name (line 549):
        
        # Assigning a Call to a Name (line 549):
        
        # Call to asarray(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'name' (line 549)
        name_136275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 26), 'name', False)
        # Processing the call keyword arguments (line 549)
        kwargs_136276 = {}
        # Getting the type of 'np' (line 549)
        np_136273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 15), 'np', False)
        # Obtaining the member 'asarray' of a type (line 549)
        asarray_136274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 15), np_136273, 'asarray')
        # Calling asarray(args, kwargs) (line 549)
        asarray_call_result_136277 = invoke(stypy.reporting.localization.Localization(__file__, 549, 15), asarray_136274, *[name_136275], **kwargs_136276)
        
        # Assigning a type to the variable 'name' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'name', asarray_call_result_136277)
        
        
        # Getting the type of 'name' (line 550)
        name_136278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 11), 'name')
        str_136279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 19), 'str', '')
        # Applying the binary operator '==' (line 550)
        result_eq_136280 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 11), '==', name_136278, str_136279)
        
        # Testing the type of an if condition (line 550)
        if_condition_136281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 8), result_eq_136280)
        # Assigning a type to the variable 'if_condition_136281' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'if_condition_136281', if_condition_136281)
        # SSA begins for if statement (line 550)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_smalldata_element(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'name' (line 551)
        name_136284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 41), 'name', False)
        # Getting the type of 'miINT8' (line 551)
        miINT8_136285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 47), 'miINT8', False)
        int_136286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 55), 'int')
        # Processing the call keyword arguments (line 551)
        kwargs_136287 = {}
        # Getting the type of 'self' (line 551)
        self_136282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'self', False)
        # Obtaining the member 'write_smalldata_element' of a type (line 551)
        write_smalldata_element_136283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 12), self_136282, 'write_smalldata_element')
        # Calling write_smalldata_element(args, kwargs) (line 551)
        write_smalldata_element_call_result_136288 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), write_smalldata_element_136283, *[name_136284, miINT8_136285, int_136286], **kwargs_136287)
        
        # SSA branch for the else part of an if statement (line 550)
        module_type_store.open_ssa_branch('else')
        
        # Call to write_element(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'name' (line 553)
        name_136291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 31), 'name', False)
        # Getting the type of 'miINT8' (line 553)
        miINT8_136292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 37), 'miINT8', False)
        # Processing the call keyword arguments (line 553)
        kwargs_136293 = {}
        # Getting the type of 'self' (line 553)
        self_136289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'self', False)
        # Obtaining the member 'write_element' of a type (line 553)
        write_element_136290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 12), self_136289, 'write_element')
        # Calling write_element(args, kwargs) (line 553)
        write_element_call_result_136294 = invoke(stypy.reporting.localization.Localization(__file__, 553, 12), write_element_136290, *[name_136291, miINT8_136292], **kwargs_136293)
        
        # SSA join for if statement (line 550)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Attribute (line 555):
        
        # Assigning a Str to a Attribute (line 555):
        str_136295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 25), 'str', '')
        # Getting the type of 'self' (line 555)
        self_136296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'self')
        # Setting the type of the member '_var_name' of a type (line 555)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 8), self_136296, '_var_name', str_136295)
        
        # Assigning a Name to a Attribute (line 556):
        
        # Assigning a Name to a Attribute (line 556):
        # Getting the type of 'False' (line 556)
        False_136297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 30), 'False')
        # Getting the type of 'self' (line 556)
        self_136298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'self')
        # Setting the type of the member '_var_is_global' of a type (line 556)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), self_136298, '_var_is_global', False_136297)
        
        # ################# End of 'write_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_header' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_136299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_header'
        return stypy_return_type_136299


    @norecursion
    def update_matrix_tag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_matrix_tag'
        module_type_store = module_type_store.open_function_context('update_matrix_tag', 558, 4, False)
        # Assigning a type to the variable 'self' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_function_name', 'VarWriter5.update_matrix_tag')
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_param_names_list', ['start_pos'])
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.update_matrix_tag.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.update_matrix_tag', ['start_pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_matrix_tag', localization, ['start_pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_matrix_tag(...)' code ##################

        
        # Assigning a Call to a Name (line 559):
        
        # Assigning a Call to a Name (line 559):
        
        # Call to tell(...): (line 559)
        # Processing the call keyword arguments (line 559)
        kwargs_136303 = {}
        # Getting the type of 'self' (line 559)
        self_136300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 19), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 559)
        file_stream_136301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 19), self_136300, 'file_stream')
        # Obtaining the member 'tell' of a type (line 559)
        tell_136302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 19), file_stream_136301, 'tell')
        # Calling tell(args, kwargs) (line 559)
        tell_call_result_136304 = invoke(stypy.reporting.localization.Localization(__file__, 559, 19), tell_136302, *[], **kwargs_136303)
        
        # Assigning a type to the variable 'curr_pos' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'curr_pos', tell_call_result_136304)
        
        # Call to seek(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'start_pos' (line 560)
        start_pos_136308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 30), 'start_pos', False)
        # Processing the call keyword arguments (line 560)
        kwargs_136309 = {}
        # Getting the type of 'self' (line 560)
        self_136305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 560)
        file_stream_136306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), self_136305, 'file_stream')
        # Obtaining the member 'seek' of a type (line 560)
        seek_136307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), file_stream_136306, 'seek')
        # Calling seek(args, kwargs) (line 560)
        seek_call_result_136310 = invoke(stypy.reporting.localization.Localization(__file__, 560, 8), seek_136307, *[start_pos_136308], **kwargs_136309)
        
        
        # Assigning a BinOp to a Name (line 561):
        
        # Assigning a BinOp to a Name (line 561):
        # Getting the type of 'curr_pos' (line 561)
        curr_pos_136311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 21), 'curr_pos')
        # Getting the type of 'start_pos' (line 561)
        start_pos_136312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 32), 'start_pos')
        # Applying the binary operator '-' (line 561)
        result_sub_136313 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 21), '-', curr_pos_136311, start_pos_136312)
        
        int_136314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 44), 'int')
        # Applying the binary operator '-' (line 561)
        result_sub_136315 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 42), '-', result_sub_136313, int_136314)
        
        # Assigning a type to the variable 'byte_count' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'byte_count', result_sub_136315)
        
        
        # Getting the type of 'byte_count' (line 562)
        byte_count_136316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'byte_count')
        int_136317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 25), 'int')
        int_136318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 28), 'int')
        # Applying the binary operator '**' (line 562)
        result_pow_136319 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 25), '**', int_136317, int_136318)
        
        # Applying the binary operator '>=' (line 562)
        result_ge_136320 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 11), '>=', byte_count_136316, result_pow_136319)
        
        # Testing the type of an if condition (line 562)
        if_condition_136321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 8), result_ge_136320)
        # Assigning a type to the variable 'if_condition_136321' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'if_condition_136321', if_condition_136321)
        # SSA begins for if statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to MatWriteError(...): (line 563)
        # Processing the call arguments (line 563)
        str_136323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 32), 'str', 'Matrix too large to save with Matlab 5 format')
        # Processing the call keyword arguments (line 563)
        kwargs_136324 = {}
        # Getting the type of 'MatWriteError' (line 563)
        MatWriteError_136322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 18), 'MatWriteError', False)
        # Calling MatWriteError(args, kwargs) (line 563)
        MatWriteError_call_result_136325 = invoke(stypy.reporting.localization.Localization(__file__, 563, 18), MatWriteError_136322, *[str_136323], **kwargs_136324)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 563, 12), MatWriteError_call_result_136325, 'raise parameter', BaseException)
        # SSA join for if statement (line 562)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 565):
        
        # Assigning a Name to a Subscript (line 565):
        # Getting the type of 'byte_count' (line 565)
        byte_count_136326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 37), 'byte_count')
        # Getting the type of 'self' (line 565)
        self_136327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'self')
        # Obtaining the member 'mat_tag' of a type (line 565)
        mat_tag_136328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), self_136327, 'mat_tag')
        str_136329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 21), 'str', 'byte_count')
        # Storing an element on a container (line 565)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 8), mat_tag_136328, (str_136329, byte_count_136326))
        
        # Call to write_bytes(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'self' (line 566)
        self_136332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 25), 'self', False)
        # Obtaining the member 'mat_tag' of a type (line 566)
        mat_tag_136333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 25), self_136332, 'mat_tag')
        # Processing the call keyword arguments (line 566)
        kwargs_136334 = {}
        # Getting the type of 'self' (line 566)
        self_136330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 566)
        write_bytes_136331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), self_136330, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 566)
        write_bytes_call_result_136335 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), write_bytes_136331, *[mat_tag_136333], **kwargs_136334)
        
        
        # Call to seek(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'curr_pos' (line 567)
        curr_pos_136339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 30), 'curr_pos', False)
        # Processing the call keyword arguments (line 567)
        kwargs_136340 = {}
        # Getting the type of 'self' (line 567)
        self_136336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 567)
        file_stream_136337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), self_136336, 'file_stream')
        # Obtaining the member 'seek' of a type (line 567)
        seek_136338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), file_stream_136337, 'seek')
        # Calling seek(args, kwargs) (line 567)
        seek_call_result_136341 = invoke(stypy.reporting.localization.Localization(__file__, 567, 8), seek_136338, *[curr_pos_136339], **kwargs_136340)
        
        
        # ################# End of 'update_matrix_tag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_matrix_tag' in the type store
        # Getting the type of 'stypy_return_type' (line 558)
        stypy_return_type_136342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_matrix_tag'
        return stypy_return_type_136342


    @norecursion
    def write_top(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_top'
        module_type_store = module_type_store.open_function_context('write_top', 569, 4, False)
        # Assigning a type to the variable 'self' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_top.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_top.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_top.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_top.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_top')
        VarWriter5.write_top.__dict__.__setitem__('stypy_param_names_list', ['arr', 'name', 'is_global'])
        VarWriter5.write_top.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_top.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_top.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_top.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_top.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_top.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_top', ['arr', 'name', 'is_global'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_top', localization, ['arr', 'name', 'is_global'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_top(...)' code ##################

        str_136343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, (-1)), 'str', ' Write variable at top level of mat file\n\n        Parameters\n        ----------\n        arr : array_like\n            array-like object to create writer for\n        name : str, optional\n            name as it will appear in matlab workspace\n            default is empty string\n        is_global : {False, True}, optional\n            whether variable will be global on load into matlab\n        ')
        
        # Assigning a Name to a Attribute (line 584):
        
        # Assigning a Name to a Attribute (line 584):
        # Getting the type of 'is_global' (line 584)
        is_global_136344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 30), 'is_global')
        # Getting the type of 'self' (line 584)
        self_136345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'self')
        # Setting the type of the member '_var_is_global' of a type (line 584)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 8), self_136345, '_var_is_global', is_global_136344)
        
        # Assigning a Name to a Attribute (line 585):
        
        # Assigning a Name to a Attribute (line 585):
        # Getting the type of 'name' (line 585)
        name_136346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 25), 'name')
        # Getting the type of 'self' (line 585)
        self_136347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'self')
        # Setting the type of the member '_var_name' of a type (line 585)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 8), self_136347, '_var_name', name_136346)
        
        # Call to write(...): (line 587)
        # Processing the call arguments (line 587)
        # Getting the type of 'arr' (line 587)
        arr_136350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 19), 'arr', False)
        # Processing the call keyword arguments (line 587)
        kwargs_136351 = {}
        # Getting the type of 'self' (line 587)
        self_136348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 587)
        write_136349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 8), self_136348, 'write')
        # Calling write(args, kwargs) (line 587)
        write_call_result_136352 = invoke(stypy.reporting.localization.Localization(__file__, 587, 8), write_136349, *[arr_136350], **kwargs_136351)
        
        
        # ################# End of 'write_top(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_top' in the type store
        # Getting the type of 'stypy_return_type' (line 569)
        stypy_return_type_136353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_top'
        return stypy_return_type_136353


    @norecursion
    def write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 589, 4, False)
        # Assigning a type to the variable 'self' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write')
        VarWriter5.write.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5.write.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write(...)' code ##################

        str_136354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, (-1)), 'str', ' Write `arr` to stream at top and sub levels\n\n        Parameters\n        ----------\n        arr : array_like\n            array-like object to create writer for\n        ')
        
        # Assigning a Call to a Name (line 598):
        
        # Assigning a Call to a Name (line 598):
        
        # Call to tell(...): (line 598)
        # Processing the call keyword arguments (line 598)
        kwargs_136358 = {}
        # Getting the type of 'self' (line 598)
        self_136355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 22), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 598)
        file_stream_136356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 22), self_136355, 'file_stream')
        # Obtaining the member 'tell' of a type (line 598)
        tell_136357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 22), file_stream_136356, 'tell')
        # Calling tell(args, kwargs) (line 598)
        tell_call_result_136359 = invoke(stypy.reporting.localization.Localization(__file__, 598, 22), tell_136357, *[], **kwargs_136358)
        
        # Assigning a type to the variable 'mat_tag_pos' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'mat_tag_pos', tell_call_result_136359)
        
        
        # Call to issparse(...): (line 600)
        # Processing the call arguments (line 600)
        # Getting the type of 'arr' (line 600)
        arr_136363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 33), 'arr', False)
        # Processing the call keyword arguments (line 600)
        kwargs_136364 = {}
        # Getting the type of 'scipy' (line 600)
        scipy_136360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 11), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 600)
        sparse_136361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 11), scipy_136360, 'sparse')
        # Obtaining the member 'issparse' of a type (line 600)
        issparse_136362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 11), sparse_136361, 'issparse')
        # Calling issparse(args, kwargs) (line 600)
        issparse_call_result_136365 = invoke(stypy.reporting.localization.Localization(__file__, 600, 11), issparse_136362, *[arr_136363], **kwargs_136364)
        
        # Testing the type of an if condition (line 600)
        if_condition_136366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 8), issparse_call_result_136365)
        # Assigning a type to the variable 'if_condition_136366' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'if_condition_136366', if_condition_136366)
        # SSA begins for if statement (line 600)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_sparse(...): (line 601)
        # Processing the call arguments (line 601)
        # Getting the type of 'arr' (line 601)
        arr_136369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 30), 'arr', False)
        # Processing the call keyword arguments (line 601)
        kwargs_136370 = {}
        # Getting the type of 'self' (line 601)
        self_136367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'self', False)
        # Obtaining the member 'write_sparse' of a type (line 601)
        write_sparse_136368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), self_136367, 'write_sparse')
        # Calling write_sparse(args, kwargs) (line 601)
        write_sparse_call_result_136371 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), write_sparse_136368, *[arr_136369], **kwargs_136370)
        
        
        # Call to update_matrix_tag(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'mat_tag_pos' (line 602)
        mat_tag_pos_136374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 35), 'mat_tag_pos', False)
        # Processing the call keyword arguments (line 602)
        kwargs_136375 = {}
        # Getting the type of 'self' (line 602)
        self_136372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'self', False)
        # Obtaining the member 'update_matrix_tag' of a type (line 602)
        update_matrix_tag_136373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 12), self_136372, 'update_matrix_tag')
        # Calling update_matrix_tag(args, kwargs) (line 602)
        update_matrix_tag_call_result_136376 = invoke(stypy.reporting.localization.Localization(__file__, 602, 12), update_matrix_tag_136373, *[mat_tag_pos_136374], **kwargs_136375)
        
        # Assigning a type to the variable 'stypy_return_type' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 600)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 605):
        
        # Assigning a Call to a Name (line 605):
        
        # Call to to_writeable(...): (line 605)
        # Processing the call arguments (line 605)
        # Getting the type of 'arr' (line 605)
        arr_136378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 28), 'arr', False)
        # Processing the call keyword arguments (line 605)
        kwargs_136379 = {}
        # Getting the type of 'to_writeable' (line 605)
        to_writeable_136377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'to_writeable', False)
        # Calling to_writeable(args, kwargs) (line 605)
        to_writeable_call_result_136380 = invoke(stypy.reporting.localization.Localization(__file__, 605, 15), to_writeable_136377, *[arr_136378], **kwargs_136379)
        
        # Assigning a type to the variable 'narr' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'narr', to_writeable_call_result_136380)
        
        # Type idiom detected: calculating its left and rigth part (line 606)
        # Getting the type of 'narr' (line 606)
        narr_136381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 11), 'narr')
        # Getting the type of 'None' (line 606)
        None_136382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 19), 'None')
        
        (may_be_136383, more_types_in_union_136384) = may_be_none(narr_136381, None_136382)

        if may_be_136383:

            if more_types_in_union_136384:
                # Runtime conditional SSA (line 606)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 607)
            # Processing the call arguments (line 607)
            str_136386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 28), 'str', 'Could not convert %s (type %s) to array')
            
            # Obtaining an instance of the builtin type 'tuple' (line 608)
            tuple_136387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 608)
            # Adding element type (line 608)
            # Getting the type of 'arr' (line 608)
            arr_136388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 31), 'arr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 31), tuple_136387, arr_136388)
            # Adding element type (line 608)
            
            # Call to type(...): (line 608)
            # Processing the call arguments (line 608)
            # Getting the type of 'arr' (line 608)
            arr_136390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 41), 'arr', False)
            # Processing the call keyword arguments (line 608)
            kwargs_136391 = {}
            # Getting the type of 'type' (line 608)
            type_136389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 36), 'type', False)
            # Calling type(args, kwargs) (line 608)
            type_call_result_136392 = invoke(stypy.reporting.localization.Localization(__file__, 608, 36), type_136389, *[arr_136390], **kwargs_136391)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 31), tuple_136387, type_call_result_136392)
            
            # Applying the binary operator '%' (line 607)
            result_mod_136393 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 28), '%', str_136386, tuple_136387)
            
            # Processing the call keyword arguments (line 607)
            kwargs_136394 = {}
            # Getting the type of 'TypeError' (line 607)
            TypeError_136385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 607)
            TypeError_call_result_136395 = invoke(stypy.reporting.localization.Localization(__file__, 607, 18), TypeError_136385, *[result_mod_136393], **kwargs_136394)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 607, 12), TypeError_call_result_136395, 'raise parameter', BaseException)

            if more_types_in_union_136384:
                # SSA join for if statement (line 606)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to isinstance(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 'narr' (line 609)
        narr_136397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 22), 'narr', False)
        # Getting the type of 'MatlabObject' (line 609)
        MatlabObject_136398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 28), 'MatlabObject', False)
        # Processing the call keyword arguments (line 609)
        kwargs_136399 = {}
        # Getting the type of 'isinstance' (line 609)
        isinstance_136396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 609)
        isinstance_call_result_136400 = invoke(stypy.reporting.localization.Localization(__file__, 609, 11), isinstance_136396, *[narr_136397, MatlabObject_136398], **kwargs_136399)
        
        # Testing the type of an if condition (line 609)
        if_condition_136401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 8), isinstance_call_result_136400)
        # Assigning a type to the variable 'if_condition_136401' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'if_condition_136401', if_condition_136401)
        # SSA begins for if statement (line 609)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_object(...): (line 610)
        # Processing the call arguments (line 610)
        # Getting the type of 'narr' (line 610)
        narr_136404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 30), 'narr', False)
        # Processing the call keyword arguments (line 610)
        kwargs_136405 = {}
        # Getting the type of 'self' (line 610)
        self_136402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'self', False)
        # Obtaining the member 'write_object' of a type (line 610)
        write_object_136403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 12), self_136402, 'write_object')
        # Calling write_object(args, kwargs) (line 610)
        write_object_call_result_136406 = invoke(stypy.reporting.localization.Localization(__file__, 610, 12), write_object_136403, *[narr_136404], **kwargs_136405)
        
        # SSA branch for the else part of an if statement (line 609)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 611)
        # Processing the call arguments (line 611)
        # Getting the type of 'narr' (line 611)
        narr_136408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 24), 'narr', False)
        # Getting the type of 'MatlabFunction' (line 611)
        MatlabFunction_136409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 30), 'MatlabFunction', False)
        # Processing the call keyword arguments (line 611)
        kwargs_136410 = {}
        # Getting the type of 'isinstance' (line 611)
        isinstance_136407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 611)
        isinstance_call_result_136411 = invoke(stypy.reporting.localization.Localization(__file__, 611, 13), isinstance_136407, *[narr_136408, MatlabFunction_136409], **kwargs_136410)
        
        # Testing the type of an if condition (line 611)
        if_condition_136412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 13), isinstance_call_result_136411)
        # Assigning a type to the variable 'if_condition_136412' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 13), 'if_condition_136412', if_condition_136412)
        # SSA begins for if statement (line 611)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to MatWriteError(...): (line 612)
        # Processing the call arguments (line 612)
        str_136414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 32), 'str', 'Cannot write matlab functions')
        # Processing the call keyword arguments (line 612)
        kwargs_136415 = {}
        # Getting the type of 'MatWriteError' (line 612)
        MatWriteError_136413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 18), 'MatWriteError', False)
        # Calling MatWriteError(args, kwargs) (line 612)
        MatWriteError_call_result_136416 = invoke(stypy.reporting.localization.Localization(__file__, 612, 18), MatWriteError_136413, *[str_136414], **kwargs_136415)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 612, 12), MatWriteError_call_result_136416, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 611)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'narr' (line 613)
        narr_136417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 13), 'narr')
        # Getting the type of 'EmptyStructMarker' (line 613)
        EmptyStructMarker_136418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 21), 'EmptyStructMarker')
        # Applying the binary operator 'is' (line 613)
        result_is__136419 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 13), 'is', narr_136417, EmptyStructMarker_136418)
        
        # Testing the type of an if condition (line 613)
        if_condition_136420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 13), result_is__136419)
        # Assigning a type to the variable 'if_condition_136420' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 13), 'if_condition_136420', if_condition_136420)
        # SSA begins for if statement (line 613)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_empty_struct(...): (line 614)
        # Processing the call keyword arguments (line 614)
        kwargs_136423 = {}
        # Getting the type of 'self' (line 614)
        self_136421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'self', False)
        # Obtaining the member 'write_empty_struct' of a type (line 614)
        write_empty_struct_136422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 12), self_136421, 'write_empty_struct')
        # Calling write_empty_struct(args, kwargs) (line 614)
        write_empty_struct_call_result_136424 = invoke(stypy.reporting.localization.Localization(__file__, 614, 12), write_empty_struct_136422, *[], **kwargs_136423)
        
        # SSA branch for the else part of an if statement (line 613)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'narr' (line 615)
        narr_136425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 13), 'narr')
        # Obtaining the member 'dtype' of a type (line 615)
        dtype_136426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 13), narr_136425, 'dtype')
        # Obtaining the member 'fields' of a type (line 615)
        fields_136427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 13), dtype_136426, 'fields')
        # Testing the type of an if condition (line 615)
        if_condition_136428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 13), fields_136427)
        # Assigning a type to the variable 'if_condition_136428' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 13), 'if_condition_136428', if_condition_136428)
        # SSA begins for if statement (line 615)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_struct(...): (line 616)
        # Processing the call arguments (line 616)
        # Getting the type of 'narr' (line 616)
        narr_136431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 30), 'narr', False)
        # Processing the call keyword arguments (line 616)
        kwargs_136432 = {}
        # Getting the type of 'self' (line 616)
        self_136429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'self', False)
        # Obtaining the member 'write_struct' of a type (line 616)
        write_struct_136430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 12), self_136429, 'write_struct')
        # Calling write_struct(args, kwargs) (line 616)
        write_struct_call_result_136433 = invoke(stypy.reporting.localization.Localization(__file__, 616, 12), write_struct_136430, *[narr_136431], **kwargs_136432)
        
        # SSA branch for the else part of an if statement (line 615)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'narr' (line 617)
        narr_136434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 13), 'narr')
        # Obtaining the member 'dtype' of a type (line 617)
        dtype_136435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 13), narr_136434, 'dtype')
        # Obtaining the member 'hasobject' of a type (line 617)
        hasobject_136436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 13), dtype_136435, 'hasobject')
        # Testing the type of an if condition (line 617)
        if_condition_136437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 13), hasobject_136436)
        # Assigning a type to the variable 'if_condition_136437' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 13), 'if_condition_136437', if_condition_136437)
        # SSA begins for if statement (line 617)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_cells(...): (line 618)
        # Processing the call arguments (line 618)
        # Getting the type of 'narr' (line 618)
        narr_136440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 29), 'narr', False)
        # Processing the call keyword arguments (line 618)
        kwargs_136441 = {}
        # Getting the type of 'self' (line 618)
        self_136438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'self', False)
        # Obtaining the member 'write_cells' of a type (line 618)
        write_cells_136439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 12), self_136438, 'write_cells')
        # Calling write_cells(args, kwargs) (line 618)
        write_cells_call_result_136442 = invoke(stypy.reporting.localization.Localization(__file__, 618, 12), write_cells_136439, *[narr_136440], **kwargs_136441)
        
        # SSA branch for the else part of an if statement (line 617)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'narr' (line 619)
        narr_136443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 13), 'narr')
        # Obtaining the member 'dtype' of a type (line 619)
        dtype_136444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 13), narr_136443, 'dtype')
        # Obtaining the member 'kind' of a type (line 619)
        kind_136445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 13), dtype_136444, 'kind')
        
        # Obtaining an instance of the builtin type 'tuple' (line 619)
        tuple_136446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 619)
        # Adding element type (line 619)
        str_136447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 33), 'str', 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 33), tuple_136446, str_136447)
        # Adding element type (line 619)
        str_136448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 38), 'str', 'S')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 33), tuple_136446, str_136448)
        
        # Applying the binary operator 'in' (line 619)
        result_contains_136449 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 13), 'in', kind_136445, tuple_136446)
        
        # Testing the type of an if condition (line 619)
        if_condition_136450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 13), result_contains_136449)
        # Assigning a type to the variable 'if_condition_136450' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 13), 'if_condition_136450', if_condition_136450)
        # SSA begins for if statement (line 619)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 620)
        self_136451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'self')
        # Obtaining the member 'unicode_strings' of a type (line 620)
        unicode_strings_136452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), self_136451, 'unicode_strings')
        # Testing the type of an if condition (line 620)
        if_condition_136453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 12), unicode_strings_136452)
        # Assigning a type to the variable 'if_condition_136453' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'if_condition_136453', if_condition_136453)
        # SSA begins for if statement (line 620)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 621):
        
        # Assigning a Str to a Name (line 621):
        str_136454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 24), 'str', 'UTF8')
        # Assigning a type to the variable 'codec' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 16), 'codec', str_136454)
        # SSA branch for the else part of an if statement (line 620)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 623):
        
        # Assigning a Str to a Name (line 623):
        str_136455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 24), 'str', 'ascii')
        # Assigning a type to the variable 'codec' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'codec', str_136455)
        # SSA join for if statement (line 620)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_char(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 'narr' (line 624)
        narr_136458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 28), 'narr', False)
        # Getting the type of 'codec' (line 624)
        codec_136459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 34), 'codec', False)
        # Processing the call keyword arguments (line 624)
        kwargs_136460 = {}
        # Getting the type of 'self' (line 624)
        self_136456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'self', False)
        # Obtaining the member 'write_char' of a type (line 624)
        write_char_136457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 12), self_136456, 'write_char')
        # Calling write_char(args, kwargs) (line 624)
        write_char_call_result_136461 = invoke(stypy.reporting.localization.Localization(__file__, 624, 12), write_char_136457, *[narr_136458, codec_136459], **kwargs_136460)
        
        # SSA branch for the else part of an if statement (line 619)
        module_type_store.open_ssa_branch('else')
        
        # Call to write_numeric(...): (line 626)
        # Processing the call arguments (line 626)
        # Getting the type of 'narr' (line 626)
        narr_136464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 31), 'narr', False)
        # Processing the call keyword arguments (line 626)
        kwargs_136465 = {}
        # Getting the type of 'self' (line 626)
        self_136462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'self', False)
        # Obtaining the member 'write_numeric' of a type (line 626)
        write_numeric_136463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 12), self_136462, 'write_numeric')
        # Calling write_numeric(args, kwargs) (line 626)
        write_numeric_call_result_136466 = invoke(stypy.reporting.localization.Localization(__file__, 626, 12), write_numeric_136463, *[narr_136464], **kwargs_136465)
        
        # SSA join for if statement (line 619)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 617)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 615)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 613)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 611)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 609)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to update_matrix_tag(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 'mat_tag_pos' (line 627)
        mat_tag_pos_136469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 31), 'mat_tag_pos', False)
        # Processing the call keyword arguments (line 627)
        kwargs_136470 = {}
        # Getting the type of 'self' (line 627)
        self_136467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'self', False)
        # Obtaining the member 'update_matrix_tag' of a type (line 627)
        update_matrix_tag_136468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 8), self_136467, 'update_matrix_tag')
        # Calling update_matrix_tag(args, kwargs) (line 627)
        update_matrix_tag_call_result_136471 = invoke(stypy.reporting.localization.Localization(__file__, 627, 8), update_matrix_tag_136468, *[mat_tag_pos_136469], **kwargs_136470)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 589)
        stypy_return_type_136472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136472)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_136472


    @norecursion
    def write_numeric(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_numeric'
        module_type_store = module_type_store.open_function_context('write_numeric', 629, 4, False)
        # Assigning a type to the variable 'self' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_numeric')
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_numeric.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_numeric', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_numeric', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_numeric(...)' code ##################

        
        # Assigning a Compare to a Name (line 630):
        
        # Assigning a Compare to a Name (line 630):
        
        # Getting the type of 'arr' (line 630)
        arr_136473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 16), 'arr')
        # Obtaining the member 'dtype' of a type (line 630)
        dtype_136474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 16), arr_136473, 'dtype')
        # Obtaining the member 'kind' of a type (line 630)
        kind_136475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 16), dtype_136474, 'kind')
        str_136476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 34), 'str', 'c')
        # Applying the binary operator '==' (line 630)
        result_eq_136477 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 16), '==', kind_136475, str_136476)
        
        # Assigning a type to the variable 'imagf' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'imagf', result_eq_136477)
        
        # Assigning a Compare to a Name (line 631):
        
        # Assigning a Compare to a Name (line 631):
        
        # Getting the type of 'arr' (line 631)
        arr_136478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'arr')
        # Obtaining the member 'dtype' of a type (line 631)
        dtype_136479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 16), arr_136478, 'dtype')
        # Obtaining the member 'kind' of a type (line 631)
        kind_136480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 16), dtype_136479, 'kind')
        str_136481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 34), 'str', 'b')
        # Applying the binary operator '==' (line 631)
        result_eq_136482 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 16), '==', kind_136480, str_136481)
        
        # Assigning a type to the variable 'logif' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'logif', result_eq_136482)
        
        
        # SSA begins for try-except statement (line 632)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 633):
        
        # Assigning a Subscript to a Name (line 633):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_136483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 49), 'int')
        slice_136484 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 633, 35), int_136483, None, None)
        # Getting the type of 'arr' (line 633)
        arr_136485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 35), 'arr')
        # Obtaining the member 'dtype' of a type (line 633)
        dtype_136486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 35), arr_136485, 'dtype')
        # Obtaining the member 'str' of a type (line 633)
        str_136487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 35), dtype_136486, 'str')
        # Obtaining the member '__getitem__' of a type (line 633)
        getitem___136488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 35), str_136487, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 633)
        subscript_call_result_136489 = invoke(stypy.reporting.localization.Localization(__file__, 633, 35), getitem___136488, slice_136484)
        
        # Getting the type of 'NP_TO_MXTYPES' (line 633)
        NP_TO_MXTYPES_136490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 21), 'NP_TO_MXTYPES')
        # Obtaining the member '__getitem__' of a type (line 633)
        getitem___136491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 21), NP_TO_MXTYPES_136490, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 633)
        subscript_call_result_136492 = invoke(stypy.reporting.localization.Localization(__file__, 633, 21), getitem___136491, subscript_call_result_136489)
        
        # Assigning a type to the variable 'mclass' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 12), 'mclass', subscript_call_result_136492)
        # SSA branch for the except part of a try statement (line 632)
        # SSA branch for the except 'KeyError' branch of a try statement (line 632)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'imagf' (line 637)
        imagf_136493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 15), 'imagf')
        # Testing the type of an if condition (line 637)
        if_condition_136494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 637, 12), imagf_136493)
        # Assigning a type to the variable 'if_condition_136494' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'if_condition_136494', if_condition_136494)
        # SSA begins for if statement (line 637)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 638):
        
        # Assigning a Call to a Name (line 638):
        
        # Call to astype(...): (line 638)
        # Processing the call arguments (line 638)
        str_136497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 33), 'str', 'c128')
        # Processing the call keyword arguments (line 638)
        kwargs_136498 = {}
        # Getting the type of 'arr' (line 638)
        arr_136495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 22), 'arr', False)
        # Obtaining the member 'astype' of a type (line 638)
        astype_136496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 22), arr_136495, 'astype')
        # Calling astype(args, kwargs) (line 638)
        astype_call_result_136499 = invoke(stypy.reporting.localization.Localization(__file__, 638, 22), astype_136496, *[str_136497], **kwargs_136498)
        
        # Assigning a type to the variable 'arr' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 16), 'arr', astype_call_result_136499)
        # SSA branch for the else part of an if statement (line 637)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'logif' (line 639)
        logif_136500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 17), 'logif')
        # Testing the type of an if condition (line 639)
        if_condition_136501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 17), logif_136500)
        # Assigning a type to the variable 'if_condition_136501' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 17), 'if_condition_136501', if_condition_136501)
        # SSA begins for if statement (line 639)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 640):
        
        # Assigning a Call to a Name (line 640):
        
        # Call to astype(...): (line 640)
        # Processing the call arguments (line 640)
        str_136504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 33), 'str', 'i1')
        # Processing the call keyword arguments (line 640)
        kwargs_136505 = {}
        # Getting the type of 'arr' (line 640)
        arr_136502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 22), 'arr', False)
        # Obtaining the member 'astype' of a type (line 640)
        astype_136503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 22), arr_136502, 'astype')
        # Calling astype(args, kwargs) (line 640)
        astype_call_result_136506 = invoke(stypy.reporting.localization.Localization(__file__, 640, 22), astype_136503, *[str_136504], **kwargs_136505)
        
        # Assigning a type to the variable 'arr' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'arr', astype_call_result_136506)
        # SSA branch for the else part of an if statement (line 639)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 642):
        
        # Assigning a Call to a Name (line 642):
        
        # Call to astype(...): (line 642)
        # Processing the call arguments (line 642)
        str_136509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 33), 'str', 'f8')
        # Processing the call keyword arguments (line 642)
        kwargs_136510 = {}
        # Getting the type of 'arr' (line 642)
        arr_136507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 22), 'arr', False)
        # Obtaining the member 'astype' of a type (line 642)
        astype_136508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 22), arr_136507, 'astype')
        # Calling astype(args, kwargs) (line 642)
        astype_call_result_136511 = invoke(stypy.reporting.localization.Localization(__file__, 642, 22), astype_136508, *[str_136509], **kwargs_136510)
        
        # Assigning a type to the variable 'arr' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 16), 'arr', astype_call_result_136511)
        # SSA join for if statement (line 639)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 637)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 643):
        
        # Assigning a Name to a Name (line 643):
        # Getting the type of 'mxDOUBLE_CLASS' (line 643)
        mxDOUBLE_CLASS_136512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 21), 'mxDOUBLE_CLASS')
        # Assigning a type to the variable 'mclass' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'mclass', mxDOUBLE_CLASS_136512)
        # SSA join for try-except statement (line 632)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_header(...): (line 644)
        # Processing the call arguments (line 644)
        
        # Call to matdims(...): (line 644)
        # Processing the call arguments (line 644)
        # Getting the type of 'arr' (line 644)
        arr_136516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 34), 'arr', False)
        # Getting the type of 'self' (line 644)
        self_136517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 39), 'self', False)
        # Obtaining the member 'oned_as' of a type (line 644)
        oned_as_136518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 39), self_136517, 'oned_as')
        # Processing the call keyword arguments (line 644)
        kwargs_136519 = {}
        # Getting the type of 'matdims' (line 644)
        matdims_136515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 26), 'matdims', False)
        # Calling matdims(args, kwargs) (line 644)
        matdims_call_result_136520 = invoke(stypy.reporting.localization.Localization(__file__, 644, 26), matdims_136515, *[arr_136516, oned_as_136518], **kwargs_136519)
        
        # Getting the type of 'mclass' (line 645)
        mclass_136521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 26), 'mclass', False)
        # Processing the call keyword arguments (line 644)
        # Getting the type of 'imagf' (line 646)
        imagf_136522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 37), 'imagf', False)
        keyword_136523 = imagf_136522
        # Getting the type of 'logif' (line 647)
        logif_136524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 37), 'logif', False)
        keyword_136525 = logif_136524
        kwargs_136526 = {'is_logical': keyword_136525, 'is_complex': keyword_136523}
        # Getting the type of 'self' (line 644)
        self_136513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 644)
        write_header_136514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 8), self_136513, 'write_header')
        # Calling write_header(args, kwargs) (line 644)
        write_header_call_result_136527 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), write_header_136514, *[matdims_call_result_136520, mclass_136521], **kwargs_136526)
        
        
        # Getting the type of 'imagf' (line 648)
        imagf_136528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 11), 'imagf')
        # Testing the type of an if condition (line 648)
        if_condition_136529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 648, 8), imagf_136528)
        # Assigning a type to the variable 'if_condition_136529' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'if_condition_136529', if_condition_136529)
        # SSA begins for if statement (line 648)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_element(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'arr' (line 649)
        arr_136532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 31), 'arr', False)
        # Obtaining the member 'real' of a type (line 649)
        real_136533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 31), arr_136532, 'real')
        # Processing the call keyword arguments (line 649)
        kwargs_136534 = {}
        # Getting the type of 'self' (line 649)
        self_136530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 12), 'self', False)
        # Obtaining the member 'write_element' of a type (line 649)
        write_element_136531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 12), self_136530, 'write_element')
        # Calling write_element(args, kwargs) (line 649)
        write_element_call_result_136535 = invoke(stypy.reporting.localization.Localization(__file__, 649, 12), write_element_136531, *[real_136533], **kwargs_136534)
        
        
        # Call to write_element(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'arr' (line 650)
        arr_136538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 31), 'arr', False)
        # Obtaining the member 'imag' of a type (line 650)
        imag_136539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 31), arr_136538, 'imag')
        # Processing the call keyword arguments (line 650)
        kwargs_136540 = {}
        # Getting the type of 'self' (line 650)
        self_136536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'self', False)
        # Obtaining the member 'write_element' of a type (line 650)
        write_element_136537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 12), self_136536, 'write_element')
        # Calling write_element(args, kwargs) (line 650)
        write_element_call_result_136541 = invoke(stypy.reporting.localization.Localization(__file__, 650, 12), write_element_136537, *[imag_136539], **kwargs_136540)
        
        # SSA branch for the else part of an if statement (line 648)
        module_type_store.open_ssa_branch('else')
        
        # Call to write_element(...): (line 652)
        # Processing the call arguments (line 652)
        # Getting the type of 'arr' (line 652)
        arr_136544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 31), 'arr', False)
        # Processing the call keyword arguments (line 652)
        kwargs_136545 = {}
        # Getting the type of 'self' (line 652)
        self_136542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'self', False)
        # Obtaining the member 'write_element' of a type (line 652)
        write_element_136543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 12), self_136542, 'write_element')
        # Calling write_element(args, kwargs) (line 652)
        write_element_call_result_136546 = invoke(stypy.reporting.localization.Localization(__file__, 652, 12), write_element_136543, *[arr_136544], **kwargs_136545)
        
        # SSA join for if statement (line 648)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_numeric(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_numeric' in the type store
        # Getting the type of 'stypy_return_type' (line 629)
        stypy_return_type_136547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136547)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_numeric'
        return stypy_return_type_136547


    @norecursion
    def write_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_136548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 36), 'str', 'ascii')
        defaults = [str_136548]
        # Create a new context for function 'write_char'
        module_type_store = module_type_store.open_function_context('write_char', 654, 4, False)
        # Assigning a type to the variable 'self' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_char.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_char.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_char')
        VarWriter5.write_char.__dict__.__setitem__('stypy_param_names_list', ['arr', 'codec'])
        VarWriter5.write_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_char.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_char', ['arr', 'codec'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_char', localization, ['arr', 'codec'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_char(...)' code ##################

        str_136549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, (-1)), 'str', ' Write string array `arr` with given `codec`\n        ')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'arr' (line 657)
        arr_136550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 11), 'arr')
        # Obtaining the member 'size' of a type (line 657)
        size_136551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 11), arr_136550, 'size')
        int_136552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 23), 'int')
        # Applying the binary operator '==' (line 657)
        result_eq_136553 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 11), '==', size_136551, int_136552)
        
        
        # Call to all(...): (line 657)
        # Processing the call arguments (line 657)
        
        # Getting the type of 'arr' (line 657)
        arr_136556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 35), 'arr', False)
        str_136557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 42), 'str', '')
        # Applying the binary operator '==' (line 657)
        result_eq_136558 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 35), '==', arr_136556, str_136557)
        
        # Processing the call keyword arguments (line 657)
        kwargs_136559 = {}
        # Getting the type of 'np' (line 657)
        np_136554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 28), 'np', False)
        # Obtaining the member 'all' of a type (line 657)
        all_136555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 28), np_136554, 'all')
        # Calling all(args, kwargs) (line 657)
        all_call_result_136560 = invoke(stypy.reporting.localization.Localization(__file__, 657, 28), all_136555, *[result_eq_136558], **kwargs_136559)
        
        # Applying the binary operator 'or' (line 657)
        result_or_keyword_136561 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 11), 'or', result_eq_136553, all_call_result_136560)
        
        # Testing the type of an if condition (line 657)
        if_condition_136562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 8), result_or_keyword_136561)
        # Assigning a type to the variable 'if_condition_136562' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'if_condition_136562', if_condition_136562)
        # SSA begins for if statement (line 657)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 667):
        
        # Assigning a BinOp to a Name (line 667):
        
        # Obtaining an instance of the builtin type 'tuple' (line 667)
        tuple_136563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 667)
        # Adding element type (line 667)
        int_136564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 21), tuple_136563, int_136564)
        
        
        # Call to max(...): (line 667)
        # Processing the call arguments (line 667)
        
        # Obtaining an instance of the builtin type 'list' (line 667)
        list_136567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 667)
        # Adding element type (line 667)
        # Getting the type of 'arr' (line 667)
        arr_136568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 35), 'arr', False)
        # Obtaining the member 'ndim' of a type (line 667)
        ndim_136569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 35), arr_136568, 'ndim')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 34), list_136567, ndim_136569)
        # Adding element type (line 667)
        int_136570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 34), list_136567, int_136570)
        
        # Processing the call keyword arguments (line 667)
        kwargs_136571 = {}
        # Getting the type of 'np' (line 667)
        np_136565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 27), 'np', False)
        # Obtaining the member 'max' of a type (line 667)
        max_136566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 27), np_136565, 'max')
        # Calling max(args, kwargs) (line 667)
        max_call_result_136572 = invoke(stypy.reporting.localization.Localization(__file__, 667, 27), max_136566, *[list_136567], **kwargs_136571)
        
        # Applying the binary operator '*' (line 667)
        result_mul_136573 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 20), '*', tuple_136563, max_call_result_136572)
        
        # Assigning a type to the variable 'shape' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'shape', result_mul_136573)
        
        # Call to write_header(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'shape' (line 668)
        shape_136576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 30), 'shape', False)
        # Getting the type of 'mxCHAR_CLASS' (line 668)
        mxCHAR_CLASS_136577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 37), 'mxCHAR_CLASS', False)
        # Processing the call keyword arguments (line 668)
        kwargs_136578 = {}
        # Getting the type of 'self' (line 668)
        self_136574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'self', False)
        # Obtaining the member 'write_header' of a type (line 668)
        write_header_136575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 12), self_136574, 'write_header')
        # Calling write_header(args, kwargs) (line 668)
        write_header_call_result_136579 = invoke(stypy.reporting.localization.Localization(__file__, 668, 12), write_header_136575, *[shape_136576, mxCHAR_CLASS_136577], **kwargs_136578)
        
        
        # Call to write_smalldata_element(...): (line 669)
        # Processing the call arguments (line 669)
        # Getting the type of 'arr' (line 669)
        arr_136582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 41), 'arr', False)
        # Getting the type of 'miUTF8' (line 669)
        miUTF8_136583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 46), 'miUTF8', False)
        int_136584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 54), 'int')
        # Processing the call keyword arguments (line 669)
        kwargs_136585 = {}
        # Getting the type of 'self' (line 669)
        self_136580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'self', False)
        # Obtaining the member 'write_smalldata_element' of a type (line 669)
        write_smalldata_element_136581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 12), self_136580, 'write_smalldata_element')
        # Calling write_smalldata_element(args, kwargs) (line 669)
        write_smalldata_element_call_result_136586 = invoke(stypy.reporting.localization.Localization(__file__, 669, 12), write_smalldata_element_136581, *[arr_136582, miUTF8_136583, int_136584], **kwargs_136585)
        
        # Assigning a type to the variable 'stypy_return_type' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 657)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 674):
        
        # Assigning a Call to a Name (line 674):
        
        # Call to arr_to_chars(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'arr' (line 674)
        arr_136588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 27), 'arr', False)
        # Processing the call keyword arguments (line 674)
        kwargs_136589 = {}
        # Getting the type of 'arr_to_chars' (line 674)
        arr_to_chars_136587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 14), 'arr_to_chars', False)
        # Calling arr_to_chars(args, kwargs) (line 674)
        arr_to_chars_call_result_136590 = invoke(stypy.reporting.localization.Localization(__file__, 674, 14), arr_to_chars_136587, *[arr_136588], **kwargs_136589)
        
        # Assigning a type to the variable 'arr' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'arr', arr_to_chars_call_result_136590)
        
        # Assigning a Attribute to a Name (line 678):
        
        # Assigning a Attribute to a Name (line 678):
        # Getting the type of 'arr' (line 678)
        arr_136591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 16), 'arr')
        # Obtaining the member 'shape' of a type (line 678)
        shape_136592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 16), arr_136591, 'shape')
        # Assigning a type to the variable 'shape' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'shape', shape_136592)
        
        # Call to write_header(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'shape' (line 679)
        shape_136595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 26), 'shape', False)
        # Getting the type of 'mxCHAR_CLASS' (line 679)
        mxCHAR_CLASS_136596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 33), 'mxCHAR_CLASS', False)
        # Processing the call keyword arguments (line 679)
        kwargs_136597 = {}
        # Getting the type of 'self' (line 679)
        self_136593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 679)
        write_header_136594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 8), self_136593, 'write_header')
        # Calling write_header(args, kwargs) (line 679)
        write_header_call_result_136598 = invoke(stypy.reporting.localization.Localization(__file__, 679, 8), write_header_136594, *[shape_136595, mxCHAR_CLASS_136596], **kwargs_136597)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'arr' (line 680)
        arr_136599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 11), 'arr')
        # Obtaining the member 'dtype' of a type (line 680)
        dtype_136600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 11), arr_136599, 'dtype')
        # Obtaining the member 'kind' of a type (line 680)
        kind_136601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 11), dtype_136600, 'kind')
        str_136602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 29), 'str', 'U')
        # Applying the binary operator '==' (line 680)
        result_eq_136603 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 11), '==', kind_136601, str_136602)
        
        # Getting the type of 'arr' (line 680)
        arr_136604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 37), 'arr')
        # Obtaining the member 'size' of a type (line 680)
        size_136605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 37), arr_136604, 'size')
        # Applying the binary operator 'and' (line 680)
        result_and_keyword_136606 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 11), 'and', result_eq_136603, size_136605)
        
        # Testing the type of an if condition (line 680)
        if_condition_136607 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 680, 8), result_and_keyword_136606)
        # Assigning a type to the variable 'if_condition_136607' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'if_condition_136607', if_condition_136607)
        # SSA begins for if statement (line 680)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 685):
        
        # Assigning a Call to a Name (line 685):
        
        # Call to product(...): (line 685)
        # Processing the call arguments (line 685)
        # Getting the type of 'shape' (line 685)
        shape_136610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 33), 'shape', False)
        # Processing the call keyword arguments (line 685)
        kwargs_136611 = {}
        # Getting the type of 'np' (line 685)
        np_136608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 22), 'np', False)
        # Obtaining the member 'product' of a type (line 685)
        product_136609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 22), np_136608, 'product')
        # Calling product(args, kwargs) (line 685)
        product_call_result_136612 = invoke(stypy.reporting.localization.Localization(__file__, 685, 22), product_136609, *[shape_136610], **kwargs_136611)
        
        # Assigning a type to the variable 'n_chars' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 12), 'n_chars', product_call_result_136612)
        
        # Assigning a Call to a Name (line 686):
        
        # Assigning a Call to a Name (line 686):
        
        # Call to ndarray(...): (line 686)
        # Processing the call keyword arguments (line 686)
        
        # Obtaining an instance of the builtin type 'tuple' (line 686)
        tuple_136615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 686)
        
        keyword_136616 = tuple_136615
        
        # Call to arr_dtype_number(...): (line 687)
        # Processing the call arguments (line 687)
        # Getting the type of 'arr' (line 687)
        arr_136618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 55), 'arr', False)
        # Getting the type of 'n_chars' (line 687)
        n_chars_136619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 60), 'n_chars', False)
        # Processing the call keyword arguments (line 687)
        kwargs_136620 = {}
        # Getting the type of 'arr_dtype_number' (line 687)
        arr_dtype_number_136617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 38), 'arr_dtype_number', False)
        # Calling arr_dtype_number(args, kwargs) (line 687)
        arr_dtype_number_call_result_136621 = invoke(stypy.reporting.localization.Localization(__file__, 687, 38), arr_dtype_number_136617, *[arr_136618, n_chars_136619], **kwargs_136620)
        
        keyword_136622 = arr_dtype_number_call_result_136621
        
        # Call to copy(...): (line 688)
        # Processing the call keyword arguments (line 688)
        kwargs_136626 = {}
        # Getting the type of 'arr' (line 688)
        arr_136623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 39), 'arr', False)
        # Obtaining the member 'T' of a type (line 688)
        T_136624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 39), arr_136623, 'T')
        # Obtaining the member 'copy' of a type (line 688)
        copy_136625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 39), T_136624, 'copy')
        # Calling copy(args, kwargs) (line 688)
        copy_call_result_136627 = invoke(stypy.reporting.localization.Localization(__file__, 688, 39), copy_136625, *[], **kwargs_136626)
        
        keyword_136628 = copy_call_result_136627
        kwargs_136629 = {'buffer': keyword_136628, 'dtype': keyword_136622, 'shape': keyword_136616}
        # Getting the type of 'np' (line 686)
        np_136613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 21), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 686)
        ndarray_136614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 21), np_136613, 'ndarray')
        # Calling ndarray(args, kwargs) (line 686)
        ndarray_call_result_136630 = invoke(stypy.reporting.localization.Localization(__file__, 686, 21), ndarray_136614, *[], **kwargs_136629)
        
        # Assigning a type to the variable 'st_arr' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'st_arr', ndarray_call_result_136630)
        
        # Assigning a Call to a Name (line 690):
        
        # Assigning a Call to a Name (line 690):
        
        # Call to encode(...): (line 690)
        # Processing the call arguments (line 690)
        # Getting the type of 'codec' (line 690)
        codec_136636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 38), 'codec', False)
        # Processing the call keyword arguments (line 690)
        kwargs_136637 = {}
        
        # Call to item(...): (line 690)
        # Processing the call keyword arguments (line 690)
        kwargs_136633 = {}
        # Getting the type of 'st_arr' (line 690)
        st_arr_136631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 17), 'st_arr', False)
        # Obtaining the member 'item' of a type (line 690)
        item_136632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 17), st_arr_136631, 'item')
        # Calling item(args, kwargs) (line 690)
        item_call_result_136634 = invoke(stypy.reporting.localization.Localization(__file__, 690, 17), item_136632, *[], **kwargs_136633)
        
        # Obtaining the member 'encode' of a type (line 690)
        encode_136635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 17), item_call_result_136634, 'encode')
        # Calling encode(args, kwargs) (line 690)
        encode_call_result_136638 = invoke(stypy.reporting.localization.Localization(__file__, 690, 17), encode_136635, *[codec_136636], **kwargs_136637)
        
        # Assigning a type to the variable 'st' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 12), 'st', encode_call_result_136638)
        
        # Assigning a Call to a Name (line 692):
        
        # Assigning a Call to a Name (line 692):
        
        # Call to ndarray(...): (line 692)
        # Processing the call keyword arguments (line 692)
        
        # Obtaining an instance of the builtin type 'tuple' (line 692)
        tuple_136641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 692)
        # Adding element type (line 692)
        
        # Call to len(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'st' (line 692)
        st_136643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 40), 'st', False)
        # Processing the call keyword arguments (line 692)
        kwargs_136644 = {}
        # Getting the type of 'len' (line 692)
        len_136642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 36), 'len', False)
        # Calling len(args, kwargs) (line 692)
        len_call_result_136645 = invoke(stypy.reporting.localization.Localization(__file__, 692, 36), len_136642, *[st_136643], **kwargs_136644)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 36), tuple_136641, len_call_result_136645)
        
        keyword_136646 = tuple_136641
        str_136647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 35), 'str', 'S1')
        keyword_136648 = str_136647
        # Getting the type of 'st' (line 694)
        st_136649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 36), 'st', False)
        keyword_136650 = st_136649
        kwargs_136651 = {'buffer': keyword_136650, 'dtype': keyword_136648, 'shape': keyword_136646}
        # Getting the type of 'np' (line 692)
        np_136639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 692)
        ndarray_136640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 18), np_136639, 'ndarray')
        # Calling ndarray(args, kwargs) (line 692)
        ndarray_call_result_136652 = invoke(stypy.reporting.localization.Localization(__file__, 692, 18), ndarray_136640, *[], **kwargs_136651)
        
        # Assigning a type to the variable 'arr' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 12), 'arr', ndarray_call_result_136652)
        # SSA join for if statement (line 680)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_element(...): (line 695)
        # Processing the call arguments (line 695)
        # Getting the type of 'arr' (line 695)
        arr_136655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 27), 'arr', False)
        # Processing the call keyword arguments (line 695)
        # Getting the type of 'miUTF8' (line 695)
        miUTF8_136656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 39), 'miUTF8', False)
        keyword_136657 = miUTF8_136656
        kwargs_136658 = {'mdtype': keyword_136657}
        # Getting the type of 'self' (line 695)
        self_136653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 695)
        write_element_136654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 8), self_136653, 'write_element')
        # Calling write_element(args, kwargs) (line 695)
        write_element_call_result_136659 = invoke(stypy.reporting.localization.Localization(__file__, 695, 8), write_element_136654, *[arr_136655], **kwargs_136658)
        
        
        # ################# End of 'write_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_char' in the type store
        # Getting the type of 'stypy_return_type' (line 654)
        stypy_return_type_136660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_char'
        return stypy_return_type_136660


    @norecursion
    def write_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_sparse'
        module_type_store = module_type_store.open_function_context('write_sparse', 697, 4, False)
        # Assigning a type to the variable 'self' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_sparse')
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_sparse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_sparse', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_sparse', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_sparse(...)' code ##################

        str_136661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, (-1)), 'str', ' Sparse matrices are 2D\n        ')
        
        # Assigning a Call to a Name (line 700):
        
        # Assigning a Call to a Name (line 700):
        
        # Call to tocsc(...): (line 700)
        # Processing the call keyword arguments (line 700)
        kwargs_136664 = {}
        # Getting the type of 'arr' (line 700)
        arr_136662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'arr', False)
        # Obtaining the member 'tocsc' of a type (line 700)
        tocsc_136663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 12), arr_136662, 'tocsc')
        # Calling tocsc(args, kwargs) (line 700)
        tocsc_call_result_136665 = invoke(stypy.reporting.localization.Localization(__file__, 700, 12), tocsc_136663, *[], **kwargs_136664)
        
        # Assigning a type to the variable 'A' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'A', tocsc_call_result_136665)
        
        # Call to sort_indices(...): (line 701)
        # Processing the call keyword arguments (line 701)
        kwargs_136668 = {}
        # Getting the type of 'A' (line 701)
        A_136666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'A', False)
        # Obtaining the member 'sort_indices' of a type (line 701)
        sort_indices_136667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 8), A_136666, 'sort_indices')
        # Calling sort_indices(args, kwargs) (line 701)
        sort_indices_call_result_136669 = invoke(stypy.reporting.localization.Localization(__file__, 701, 8), sort_indices_136667, *[], **kwargs_136668)
        
        
        # Assigning a Compare to a Name (line 702):
        
        # Assigning a Compare to a Name (line 702):
        
        # Getting the type of 'A' (line 702)
        A_136670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 22), 'A')
        # Obtaining the member 'dtype' of a type (line 702)
        dtype_136671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 22), A_136670, 'dtype')
        # Obtaining the member 'kind' of a type (line 702)
        kind_136672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 22), dtype_136671, 'kind')
        str_136673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 38), 'str', 'c')
        # Applying the binary operator '==' (line 702)
        result_eq_136674 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 22), '==', kind_136672, str_136673)
        
        # Assigning a type to the variable 'is_complex' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'is_complex', result_eq_136674)
        
        # Assigning a Compare to a Name (line 703):
        
        # Assigning a Compare to a Name (line 703):
        
        # Getting the type of 'A' (line 703)
        A_136675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 22), 'A')
        # Obtaining the member 'dtype' of a type (line 703)
        dtype_136676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 22), A_136675, 'dtype')
        # Obtaining the member 'kind' of a type (line 703)
        kind_136677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 22), dtype_136676, 'kind')
        str_136678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 38), 'str', 'b')
        # Applying the binary operator '==' (line 703)
        result_eq_136679 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 22), '==', kind_136677, str_136678)
        
        # Assigning a type to the variable 'is_logical' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'is_logical', result_eq_136679)
        
        # Assigning a Attribute to a Name (line 704):
        
        # Assigning a Attribute to a Name (line 704):
        # Getting the type of 'A' (line 704)
        A_136680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 13), 'A')
        # Obtaining the member 'nnz' of a type (line 704)
        nnz_136681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 13), A_136680, 'nnz')
        # Assigning a type to the variable 'nz' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'nz', nnz_136681)
        
        # Call to write_header(...): (line 705)
        # Processing the call arguments (line 705)
        
        # Call to matdims(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'arr' (line 705)
        arr_136685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 34), 'arr', False)
        # Getting the type of 'self' (line 705)
        self_136686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 39), 'self', False)
        # Obtaining the member 'oned_as' of a type (line 705)
        oned_as_136687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 39), self_136686, 'oned_as')
        # Processing the call keyword arguments (line 705)
        kwargs_136688 = {}
        # Getting the type of 'matdims' (line 705)
        matdims_136684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 26), 'matdims', False)
        # Calling matdims(args, kwargs) (line 705)
        matdims_call_result_136689 = invoke(stypy.reporting.localization.Localization(__file__, 705, 26), matdims_136684, *[arr_136685, oned_as_136687], **kwargs_136688)
        
        # Getting the type of 'mxSPARSE_CLASS' (line 706)
        mxSPARSE_CLASS_136690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 26), 'mxSPARSE_CLASS', False)
        # Processing the call keyword arguments (line 705)
        # Getting the type of 'is_complex' (line 707)
        is_complex_136691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 37), 'is_complex', False)
        keyword_136692 = is_complex_136691
        # Getting the type of 'is_logical' (line 708)
        is_logical_136693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 37), 'is_logical', False)
        keyword_136694 = is_logical_136693
        
        
        # Getting the type of 'nz' (line 710)
        nz_136695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 37), 'nz', False)
        int_136696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 43), 'int')
        # Applying the binary operator '==' (line 710)
        result_eq_136697 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 37), '==', nz_136695, int_136696)
        
        # Testing the type of an if expression (line 710)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 710, 32), result_eq_136697)
        # SSA begins for if expression (line 710)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_136698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 32), 'int')
        # SSA branch for the else part of an if expression (line 710)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'nz' (line 710)
        nz_136699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 50), 'nz', False)
        # SSA join for if expression (line 710)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_136700 = union_type.UnionType.add(int_136698, nz_136699)
        
        keyword_136701 = if_exp_136700
        kwargs_136702 = {'nzmax': keyword_136701, 'is_logical': keyword_136694, 'is_complex': keyword_136692}
        # Getting the type of 'self' (line 705)
        self_136682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 705)
        write_header_136683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 8), self_136682, 'write_header')
        # Calling write_header(args, kwargs) (line 705)
        write_header_call_result_136703 = invoke(stypy.reporting.localization.Localization(__file__, 705, 8), write_header_136683, *[matdims_call_result_136689, mxSPARSE_CLASS_136690], **kwargs_136702)
        
        
        # Call to write_element(...): (line 711)
        # Processing the call arguments (line 711)
        
        # Call to astype(...): (line 711)
        # Processing the call arguments (line 711)
        str_136709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 44), 'str', 'i4')
        # Processing the call keyword arguments (line 711)
        kwargs_136710 = {}
        # Getting the type of 'A' (line 711)
        A_136706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 27), 'A', False)
        # Obtaining the member 'indices' of a type (line 711)
        indices_136707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 27), A_136706, 'indices')
        # Obtaining the member 'astype' of a type (line 711)
        astype_136708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 27), indices_136707, 'astype')
        # Calling astype(args, kwargs) (line 711)
        astype_call_result_136711 = invoke(stypy.reporting.localization.Localization(__file__, 711, 27), astype_136708, *[str_136709], **kwargs_136710)
        
        # Processing the call keyword arguments (line 711)
        kwargs_136712 = {}
        # Getting the type of 'self' (line 711)
        self_136704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 711)
        write_element_136705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 8), self_136704, 'write_element')
        # Calling write_element(args, kwargs) (line 711)
        write_element_call_result_136713 = invoke(stypy.reporting.localization.Localization(__file__, 711, 8), write_element_136705, *[astype_call_result_136711], **kwargs_136712)
        
        
        # Call to write_element(...): (line 712)
        # Processing the call arguments (line 712)
        
        # Call to astype(...): (line 712)
        # Processing the call arguments (line 712)
        str_136719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 43), 'str', 'i4')
        # Processing the call keyword arguments (line 712)
        kwargs_136720 = {}
        # Getting the type of 'A' (line 712)
        A_136716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 27), 'A', False)
        # Obtaining the member 'indptr' of a type (line 712)
        indptr_136717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 27), A_136716, 'indptr')
        # Obtaining the member 'astype' of a type (line 712)
        astype_136718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 27), indptr_136717, 'astype')
        # Calling astype(args, kwargs) (line 712)
        astype_call_result_136721 = invoke(stypy.reporting.localization.Localization(__file__, 712, 27), astype_136718, *[str_136719], **kwargs_136720)
        
        # Processing the call keyword arguments (line 712)
        kwargs_136722 = {}
        # Getting the type of 'self' (line 712)
        self_136714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 712)
        write_element_136715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), self_136714, 'write_element')
        # Calling write_element(args, kwargs) (line 712)
        write_element_call_result_136723 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), write_element_136715, *[astype_call_result_136721], **kwargs_136722)
        
        
        # Call to write_element(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'A' (line 713)
        A_136726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 27), 'A', False)
        # Obtaining the member 'data' of a type (line 713)
        data_136727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 27), A_136726, 'data')
        # Obtaining the member 'real' of a type (line 713)
        real_136728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 27), data_136727, 'real')
        # Processing the call keyword arguments (line 713)
        kwargs_136729 = {}
        # Getting the type of 'self' (line 713)
        self_136724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 713)
        write_element_136725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), self_136724, 'write_element')
        # Calling write_element(args, kwargs) (line 713)
        write_element_call_result_136730 = invoke(stypy.reporting.localization.Localization(__file__, 713, 8), write_element_136725, *[real_136728], **kwargs_136729)
        
        
        # Getting the type of 'is_complex' (line 714)
        is_complex_136731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 11), 'is_complex')
        # Testing the type of an if condition (line 714)
        if_condition_136732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 714, 8), is_complex_136731)
        # Assigning a type to the variable 'if_condition_136732' (line 714)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'if_condition_136732', if_condition_136732)
        # SSA begins for if statement (line 714)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_element(...): (line 715)
        # Processing the call arguments (line 715)
        # Getting the type of 'A' (line 715)
        A_136735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 31), 'A', False)
        # Obtaining the member 'data' of a type (line 715)
        data_136736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 31), A_136735, 'data')
        # Obtaining the member 'imag' of a type (line 715)
        imag_136737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 31), data_136736, 'imag')
        # Processing the call keyword arguments (line 715)
        kwargs_136738 = {}
        # Getting the type of 'self' (line 715)
        self_136733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 12), 'self', False)
        # Obtaining the member 'write_element' of a type (line 715)
        write_element_136734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 12), self_136733, 'write_element')
        # Calling write_element(args, kwargs) (line 715)
        write_element_call_result_136739 = invoke(stypy.reporting.localization.Localization(__file__, 715, 12), write_element_136734, *[imag_136737], **kwargs_136738)
        
        # SSA join for if statement (line 714)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 697)
        stypy_return_type_136740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136740)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_sparse'
        return stypy_return_type_136740


    @norecursion
    def write_cells(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_cells'
        module_type_store = module_type_store.open_function_context('write_cells', 717, 4, False)
        # Assigning a type to the variable 'self' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_cells.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_cells')
        VarWriter5.write_cells.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5.write_cells.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_cells.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_cells', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_cells', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_cells(...)' code ##################

        
        # Call to write_header(...): (line 718)
        # Processing the call arguments (line 718)
        
        # Call to matdims(...): (line 718)
        # Processing the call arguments (line 718)
        # Getting the type of 'arr' (line 718)
        arr_136744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 34), 'arr', False)
        # Getting the type of 'self' (line 718)
        self_136745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 39), 'self', False)
        # Obtaining the member 'oned_as' of a type (line 718)
        oned_as_136746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 39), self_136745, 'oned_as')
        # Processing the call keyword arguments (line 718)
        kwargs_136747 = {}
        # Getting the type of 'matdims' (line 718)
        matdims_136743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), 'matdims', False)
        # Calling matdims(args, kwargs) (line 718)
        matdims_call_result_136748 = invoke(stypy.reporting.localization.Localization(__file__, 718, 26), matdims_136743, *[arr_136744, oned_as_136746], **kwargs_136747)
        
        # Getting the type of 'mxCELL_CLASS' (line 719)
        mxCELL_CLASS_136749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 26), 'mxCELL_CLASS', False)
        # Processing the call keyword arguments (line 718)
        kwargs_136750 = {}
        # Getting the type of 'self' (line 718)
        self_136741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 718)
        write_header_136742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 8), self_136741, 'write_header')
        # Calling write_header(args, kwargs) (line 718)
        write_header_call_result_136751 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), write_header_136742, *[matdims_call_result_136748, mxCELL_CLASS_136749], **kwargs_136750)
        
        
        # Assigning a Call to a Name (line 721):
        
        # Assigning a Call to a Name (line 721):
        
        # Call to flatten(...): (line 721)
        # Processing the call arguments (line 721)
        str_136758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 39), 'str', 'F')
        # Processing the call keyword arguments (line 721)
        kwargs_136759 = {}
        
        # Call to atleast_2d(...): (line 721)
        # Processing the call arguments (line 721)
        # Getting the type of 'arr' (line 721)
        arr_136754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 26), 'arr', False)
        # Processing the call keyword arguments (line 721)
        kwargs_136755 = {}
        # Getting the type of 'np' (line 721)
        np_136752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 721)
        atleast_2d_136753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 12), np_136752, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 721)
        atleast_2d_call_result_136756 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), atleast_2d_136753, *[arr_136754], **kwargs_136755)
        
        # Obtaining the member 'flatten' of a type (line 721)
        flatten_136757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 12), atleast_2d_call_result_136756, 'flatten')
        # Calling flatten(args, kwargs) (line 721)
        flatten_call_result_136760 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), flatten_136757, *[str_136758], **kwargs_136759)
        
        # Assigning a type to the variable 'A' (line 721)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'A', flatten_call_result_136760)
        
        # Getting the type of 'A' (line 722)
        A_136761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 18), 'A')
        # Testing the type of a for loop iterable (line 722)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 722, 8), A_136761)
        # Getting the type of the for loop variable (line 722)
        for_loop_var_136762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 722, 8), A_136761)
        # Assigning a type to the variable 'el' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'el', for_loop_var_136762)
        # SSA begins for a for statement (line 722)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 723)
        # Processing the call arguments (line 723)
        # Getting the type of 'el' (line 723)
        el_136765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 23), 'el', False)
        # Processing the call keyword arguments (line 723)
        kwargs_136766 = {}
        # Getting the type of 'self' (line 723)
        self_136763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 12), 'self', False)
        # Obtaining the member 'write' of a type (line 723)
        write_136764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 12), self_136763, 'write')
        # Calling write(args, kwargs) (line 723)
        write_call_result_136767 = invoke(stypy.reporting.localization.Localization(__file__, 723, 12), write_136764, *[el_136765], **kwargs_136766)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_cells(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_cells' in the type store
        # Getting the type of 'stypy_return_type' (line 717)
        stypy_return_type_136768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_cells'
        return stypy_return_type_136768


    @norecursion
    def write_empty_struct(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_empty_struct'
        module_type_store = module_type_store.open_function_context('write_empty_struct', 725, 4, False)
        # Assigning a type to the variable 'self' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_empty_struct')
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_param_names_list', [])
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_empty_struct.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_empty_struct', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_empty_struct', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_empty_struct(...)' code ##################

        
        # Call to write_header(...): (line 726)
        # Processing the call arguments (line 726)
        
        # Obtaining an instance of the builtin type 'tuple' (line 726)
        tuple_136771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 726)
        # Adding element type (line 726)
        int_136772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 27), tuple_136771, int_136772)
        # Adding element type (line 726)
        int_136773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 27), tuple_136771, int_136773)
        
        # Getting the type of 'mxSTRUCT_CLASS' (line 726)
        mxSTRUCT_CLASS_136774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 34), 'mxSTRUCT_CLASS', False)
        # Processing the call keyword arguments (line 726)
        kwargs_136775 = {}
        # Getting the type of 'self' (line 726)
        self_136769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 726)
        write_header_136770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 8), self_136769, 'write_header')
        # Calling write_header(args, kwargs) (line 726)
        write_header_call_result_136776 = invoke(stypy.reporting.localization.Localization(__file__, 726, 8), write_header_136770, *[tuple_136771, mxSTRUCT_CLASS_136774], **kwargs_136775)
        
        
        # Call to write_element(...): (line 728)
        # Processing the call arguments (line 728)
        
        # Call to array(...): (line 728)
        # Processing the call arguments (line 728)
        int_136781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 36), 'int')
        # Processing the call keyword arguments (line 728)
        # Getting the type of 'np' (line 728)
        np_136782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 45), 'np', False)
        # Obtaining the member 'int32' of a type (line 728)
        int32_136783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 45), np_136782, 'int32')
        keyword_136784 = int32_136783
        kwargs_136785 = {'dtype': keyword_136784}
        # Getting the type of 'np' (line 728)
        np_136779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 728)
        array_136780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 27), np_136779, 'array')
        # Calling array(args, kwargs) (line 728)
        array_call_result_136786 = invoke(stypy.reporting.localization.Localization(__file__, 728, 27), array_136780, *[int_136781], **kwargs_136785)
        
        # Processing the call keyword arguments (line 728)
        kwargs_136787 = {}
        # Getting the type of 'self' (line 728)
        self_136777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 728)
        write_element_136778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 8), self_136777, 'write_element')
        # Calling write_element(args, kwargs) (line 728)
        write_element_call_result_136788 = invoke(stypy.reporting.localization.Localization(__file__, 728, 8), write_element_136778, *[array_call_result_136786], **kwargs_136787)
        
        
        # Call to write_element(...): (line 730)
        # Processing the call arguments (line 730)
        
        # Call to array(...): (line 730)
        # Processing the call arguments (line 730)
        
        # Obtaining an instance of the builtin type 'list' (line 730)
        list_136793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 730)
        
        # Processing the call keyword arguments (line 730)
        # Getting the type of 'np' (line 730)
        np_136794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 46), 'np', False)
        # Obtaining the member 'int8' of a type (line 730)
        int8_136795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 46), np_136794, 'int8')
        keyword_136796 = int8_136795
        kwargs_136797 = {'dtype': keyword_136796}
        # Getting the type of 'np' (line 730)
        np_136791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 730)
        array_136792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 27), np_136791, 'array')
        # Calling array(args, kwargs) (line 730)
        array_call_result_136798 = invoke(stypy.reporting.localization.Localization(__file__, 730, 27), array_136792, *[list_136793], **kwargs_136797)
        
        # Processing the call keyword arguments (line 730)
        kwargs_136799 = {}
        # Getting the type of 'self' (line 730)
        self_136789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 730)
        write_element_136790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 8), self_136789, 'write_element')
        # Calling write_element(args, kwargs) (line 730)
        write_element_call_result_136800 = invoke(stypy.reporting.localization.Localization(__file__, 730, 8), write_element_136790, *[array_call_result_136798], **kwargs_136799)
        
        
        # ################# End of 'write_empty_struct(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_empty_struct' in the type store
        # Getting the type of 'stypy_return_type' (line 725)
        stypy_return_type_136801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_empty_struct'
        return stypy_return_type_136801


    @norecursion
    def write_struct(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_struct'
        module_type_store = module_type_store.open_function_context('write_struct', 732, 4, False)
        # Assigning a type to the variable 'self' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_struct.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_struct')
        VarWriter5.write_struct.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5.write_struct.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_struct.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_struct', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_struct', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_struct(...)' code ##################

        
        # Call to write_header(...): (line 733)
        # Processing the call arguments (line 733)
        
        # Call to matdims(...): (line 733)
        # Processing the call arguments (line 733)
        # Getting the type of 'arr' (line 733)
        arr_136805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 34), 'arr', False)
        # Getting the type of 'self' (line 733)
        self_136806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 39), 'self', False)
        # Obtaining the member 'oned_as' of a type (line 733)
        oned_as_136807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 39), self_136806, 'oned_as')
        # Processing the call keyword arguments (line 733)
        kwargs_136808 = {}
        # Getting the type of 'matdims' (line 733)
        matdims_136804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 26), 'matdims', False)
        # Calling matdims(args, kwargs) (line 733)
        matdims_call_result_136809 = invoke(stypy.reporting.localization.Localization(__file__, 733, 26), matdims_136804, *[arr_136805, oned_as_136807], **kwargs_136808)
        
        # Getting the type of 'mxSTRUCT_CLASS' (line 734)
        mxSTRUCT_CLASS_136810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 26), 'mxSTRUCT_CLASS', False)
        # Processing the call keyword arguments (line 733)
        kwargs_136811 = {}
        # Getting the type of 'self' (line 733)
        self_136802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 733)
        write_header_136803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 8), self_136802, 'write_header')
        # Calling write_header(args, kwargs) (line 733)
        write_header_call_result_136812 = invoke(stypy.reporting.localization.Localization(__file__, 733, 8), write_header_136803, *[matdims_call_result_136809, mxSTRUCT_CLASS_136810], **kwargs_136811)
        
        
        # Call to _write_items(...): (line 735)
        # Processing the call arguments (line 735)
        # Getting the type of 'arr' (line 735)
        arr_136815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 26), 'arr', False)
        # Processing the call keyword arguments (line 735)
        kwargs_136816 = {}
        # Getting the type of 'self' (line 735)
        self_136813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'self', False)
        # Obtaining the member '_write_items' of a type (line 735)
        _write_items_136814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 8), self_136813, '_write_items')
        # Calling _write_items(args, kwargs) (line 735)
        _write_items_call_result_136817 = invoke(stypy.reporting.localization.Localization(__file__, 735, 8), _write_items_136814, *[arr_136815], **kwargs_136816)
        
        
        # ################# End of 'write_struct(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_struct' in the type store
        # Getting the type of 'stypy_return_type' (line 732)
        stypy_return_type_136818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_struct'
        return stypy_return_type_136818


    @norecursion
    def _write_items(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_items'
        module_type_store = module_type_store.open_function_context('_write_items', 737, 4, False)
        # Assigning a type to the variable 'self' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5._write_items.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5._write_items.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5._write_items.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5._write_items.__dict__.__setitem__('stypy_function_name', 'VarWriter5._write_items')
        VarWriter5._write_items.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5._write_items.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5._write_items.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5._write_items.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5._write_items.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5._write_items.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5._write_items.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5._write_items', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_items', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_items(...)' code ##################

        
        # Assigning a ListComp to a Name (line 739):
        
        # Assigning a ListComp to a Name (line 739):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'arr' (line 739)
        arr_136823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 36), 'arr')
        # Obtaining the member 'dtype' of a type (line 739)
        dtype_136824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 36), arr_136823, 'dtype')
        # Obtaining the member 'descr' of a type (line 739)
        descr_136825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 36), dtype_136824, 'descr')
        comprehension_136826 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 22), descr_136825)
        # Assigning a type to the variable 'f' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 22), 'f', comprehension_136826)
        
        # Obtaining the type of the subscript
        int_136819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 24), 'int')
        # Getting the type of 'f' (line 739)
        f_136820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 22), 'f')
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___136821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 22), f_136820, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_136822 = invoke(stypy.reporting.localization.Localization(__file__, 739, 22), getitem___136821, int_136819)
        
        list_136827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 22), list_136827, subscript_call_result_136822)
        # Assigning a type to the variable 'fieldnames' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'fieldnames', list_136827)
        
        # Assigning a BinOp to a Name (line 740):
        
        # Assigning a BinOp to a Name (line 740):
        
        # Call to max(...): (line 740)
        # Processing the call arguments (line 740)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'fieldnames' (line 740)
        fieldnames_136833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 54), 'fieldnames', False)
        comprehension_136834 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 22), fieldnames_136833)
        # Assigning a type to the variable 'fieldname' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 22), 'fieldname', comprehension_136834)
        
        # Call to len(...): (line 740)
        # Processing the call arguments (line 740)
        # Getting the type of 'fieldname' (line 740)
        fieldname_136830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 26), 'fieldname', False)
        # Processing the call keyword arguments (line 740)
        kwargs_136831 = {}
        # Getting the type of 'len' (line 740)
        len_136829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 22), 'len', False)
        # Calling len(args, kwargs) (line 740)
        len_call_result_136832 = invoke(stypy.reporting.localization.Localization(__file__, 740, 22), len_136829, *[fieldname_136830], **kwargs_136831)
        
        list_136835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 740, 22), list_136835, len_call_result_136832)
        # Processing the call keyword arguments (line 740)
        kwargs_136836 = {}
        # Getting the type of 'max' (line 740)
        max_136828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 17), 'max', False)
        # Calling max(args, kwargs) (line 740)
        max_call_result_136837 = invoke(stypy.reporting.localization.Localization(__file__, 740, 17), max_136828, *[list_136835], **kwargs_136836)
        
        int_136838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 67), 'int')
        # Applying the binary operator '+' (line 740)
        result_add_136839 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 17), '+', max_call_result_136837, int_136838)
        
        # Assigning a type to the variable 'length' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), 'length', result_add_136839)
        
        # Assigning a BoolOp to a Name (line 741):
        
        # Assigning a BoolOp to a Name (line 741):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 741)
        self_136840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 22), 'self')
        # Obtaining the member 'long_field_names' of a type (line 741)
        long_field_names_136841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 22), self_136840, 'long_field_names')
        int_136842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 48), 'int')
        # Applying the binary operator 'and' (line 741)
        result_and_keyword_136843 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 22), 'and', long_field_names_136841, int_136842)
        
        int_136844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 55), 'int')
        # Applying the binary operator 'or' (line 741)
        result_or_keyword_136845 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 21), 'or', result_and_keyword_136843, int_136844)
        
        # Assigning a type to the variable 'max_length' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'max_length', result_or_keyword_136845)
        
        
        # Getting the type of 'length' (line 742)
        length_136846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 11), 'length')
        # Getting the type of 'max_length' (line 742)
        max_length_136847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 20), 'max_length')
        # Applying the binary operator '>' (line 742)
        result_gt_136848 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 11), '>', length_136846, max_length_136847)
        
        # Testing the type of an if condition (line 742)
        if_condition_136849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 742, 8), result_gt_136848)
        # Assigning a type to the variable 'if_condition_136849' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'if_condition_136849', if_condition_136849)
        # SSA begins for if statement (line 742)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 743)
        # Processing the call arguments (line 743)
        str_136851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 29), 'str', 'Field names are restricted to %d characters')
        # Getting the type of 'max_length' (line 744)
        max_length_136852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 30), 'max_length', False)
        int_136853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 41), 'int')
        # Applying the binary operator '-' (line 744)
        result_sub_136854 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 30), '-', max_length_136852, int_136853)
        
        # Applying the binary operator '%' (line 743)
        result_mod_136855 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 29), '%', str_136851, result_sub_136854)
        
        # Processing the call keyword arguments (line 743)
        kwargs_136856 = {}
        # Getting the type of 'ValueError' (line 743)
        ValueError_136850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 743)
        ValueError_call_result_136857 = invoke(stypy.reporting.localization.Localization(__file__, 743, 18), ValueError_136850, *[result_mod_136855], **kwargs_136856)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 743, 12), ValueError_call_result_136857, 'raise parameter', BaseException)
        # SSA join for if statement (line 742)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_element(...): (line 745)
        # Processing the call arguments (line 745)
        
        # Call to array(...): (line 745)
        # Processing the call arguments (line 745)
        
        # Obtaining an instance of the builtin type 'list' (line 745)
        list_136862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 745)
        # Adding element type (line 745)
        # Getting the type of 'length' (line 745)
        length_136863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 37), 'length', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 36), list_136862, length_136863)
        
        # Processing the call keyword arguments (line 745)
        str_136864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 52), 'str', 'i4')
        keyword_136865 = str_136864
        kwargs_136866 = {'dtype': keyword_136865}
        # Getting the type of 'np' (line 745)
        np_136860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 745)
        array_136861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 27), np_136860, 'array')
        # Calling array(args, kwargs) (line 745)
        array_call_result_136867 = invoke(stypy.reporting.localization.Localization(__file__, 745, 27), array_136861, *[list_136862], **kwargs_136866)
        
        # Processing the call keyword arguments (line 745)
        kwargs_136868 = {}
        # Getting the type of 'self' (line 745)
        self_136858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 745)
        write_element_136859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 8), self_136858, 'write_element')
        # Calling write_element(args, kwargs) (line 745)
        write_element_call_result_136869 = invoke(stypy.reporting.localization.Localization(__file__, 745, 8), write_element_136859, *[array_call_result_136867], **kwargs_136868)
        
        
        # Call to write_element(...): (line 746)
        # Processing the call arguments (line 746)
        
        # Call to array(...): (line 747)
        # Processing the call arguments (line 747)
        # Getting the type of 'fieldnames' (line 747)
        fieldnames_136874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 21), 'fieldnames', False)
        # Processing the call keyword arguments (line 747)
        str_136875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 39), 'str', 'S%d')
        # Getting the type of 'length' (line 747)
        length_136876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 48), 'length', False)
        # Applying the binary operator '%' (line 747)
        result_mod_136877 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 39), '%', str_136875, length_136876)
        
        keyword_136878 = result_mod_136877
        kwargs_136879 = {'dtype': keyword_136878}
        # Getting the type of 'np' (line 747)
        np_136872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 747)
        array_136873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 12), np_136872, 'array')
        # Calling array(args, kwargs) (line 747)
        array_call_result_136880 = invoke(stypy.reporting.localization.Localization(__file__, 747, 12), array_136873, *[fieldnames_136874], **kwargs_136879)
        
        # Processing the call keyword arguments (line 746)
        # Getting the type of 'miINT8' (line 748)
        miINT8_136881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 19), 'miINT8', False)
        keyword_136882 = miINT8_136881
        kwargs_136883 = {'mdtype': keyword_136882}
        # Getting the type of 'self' (line 746)
        self_136870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 746)
        write_element_136871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 8), self_136870, 'write_element')
        # Calling write_element(args, kwargs) (line 746)
        write_element_call_result_136884 = invoke(stypy.reporting.localization.Localization(__file__, 746, 8), write_element_136871, *[array_call_result_136880], **kwargs_136883)
        
        
        # Assigning a Call to a Name (line 749):
        
        # Assigning a Call to a Name (line 749):
        
        # Call to flatten(...): (line 749)
        # Processing the call arguments (line 749)
        str_136891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 39), 'str', 'F')
        # Processing the call keyword arguments (line 749)
        kwargs_136892 = {}
        
        # Call to atleast_2d(...): (line 749)
        # Processing the call arguments (line 749)
        # Getting the type of 'arr' (line 749)
        arr_136887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 26), 'arr', False)
        # Processing the call keyword arguments (line 749)
        kwargs_136888 = {}
        # Getting the type of 'np' (line 749)
        np_136885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 12), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 749)
        atleast_2d_136886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 12), np_136885, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 749)
        atleast_2d_call_result_136889 = invoke(stypy.reporting.localization.Localization(__file__, 749, 12), atleast_2d_136886, *[arr_136887], **kwargs_136888)
        
        # Obtaining the member 'flatten' of a type (line 749)
        flatten_136890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 12), atleast_2d_call_result_136889, 'flatten')
        # Calling flatten(args, kwargs) (line 749)
        flatten_call_result_136893 = invoke(stypy.reporting.localization.Localization(__file__, 749, 12), flatten_136890, *[str_136891], **kwargs_136892)
        
        # Assigning a type to the variable 'A' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'A', flatten_call_result_136893)
        
        # Getting the type of 'A' (line 750)
        A_136894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 18), 'A')
        # Testing the type of a for loop iterable (line 750)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 750, 8), A_136894)
        # Getting the type of the for loop variable (line 750)
        for_loop_var_136895 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 750, 8), A_136894)
        # Assigning a type to the variable 'el' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 8), 'el', for_loop_var_136895)
        # SSA begins for a for statement (line 750)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'fieldnames' (line 751)
        fieldnames_136896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 21), 'fieldnames')
        # Testing the type of a for loop iterable (line 751)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 751, 12), fieldnames_136896)
        # Getting the type of the for loop variable (line 751)
        for_loop_var_136897 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 751, 12), fieldnames_136896)
        # Assigning a type to the variable 'f' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 12), 'f', for_loop_var_136897)
        # SSA begins for a for statement (line 751)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 752)
        # Processing the call arguments (line 752)
        
        # Obtaining the type of the subscript
        # Getting the type of 'f' (line 752)
        f_136900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 30), 'f', False)
        # Getting the type of 'el' (line 752)
        el_136901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 27), 'el', False)
        # Obtaining the member '__getitem__' of a type (line 752)
        getitem___136902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 27), el_136901, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 752)
        subscript_call_result_136903 = invoke(stypy.reporting.localization.Localization(__file__, 752, 27), getitem___136902, f_136900)
        
        # Processing the call keyword arguments (line 752)
        kwargs_136904 = {}
        # Getting the type of 'self' (line 752)
        self_136898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 16), 'self', False)
        # Obtaining the member 'write' of a type (line 752)
        write_136899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 16), self_136898, 'write')
        # Calling write(args, kwargs) (line 752)
        write_call_result_136905 = invoke(stypy.reporting.localization.Localization(__file__, 752, 16), write_136899, *[subscript_call_result_136903], **kwargs_136904)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_write_items(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_items' in the type store
        # Getting the type of 'stypy_return_type' (line 737)
        stypy_return_type_136906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136906)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_items'
        return stypy_return_type_136906


    @norecursion
    def write_object(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_object'
        module_type_store = module_type_store.open_function_context('write_object', 754, 4, False)
        # Assigning a type to the variable 'self' (line 755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter5.write_object.__dict__.__setitem__('stypy_localization', localization)
        VarWriter5.write_object.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter5.write_object.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter5.write_object.__dict__.__setitem__('stypy_function_name', 'VarWriter5.write_object')
        VarWriter5.write_object.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter5.write_object.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter5.write_object.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter5.write_object.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter5.write_object.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter5.write_object.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter5.write_object.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter5.write_object', ['arr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_object', localization, ['arr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_object(...)' code ##################

        str_136907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, (-1)), 'str', 'Same as writing structs, except different mx class, and extra\n        classname element after header\n        ')
        
        # Call to write_header(...): (line 758)
        # Processing the call arguments (line 758)
        
        # Call to matdims(...): (line 758)
        # Processing the call arguments (line 758)
        # Getting the type of 'arr' (line 758)
        arr_136911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 34), 'arr', False)
        # Getting the type of 'self' (line 758)
        self_136912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 39), 'self', False)
        # Obtaining the member 'oned_as' of a type (line 758)
        oned_as_136913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 39), self_136912, 'oned_as')
        # Processing the call keyword arguments (line 758)
        kwargs_136914 = {}
        # Getting the type of 'matdims' (line 758)
        matdims_136910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 26), 'matdims', False)
        # Calling matdims(args, kwargs) (line 758)
        matdims_call_result_136915 = invoke(stypy.reporting.localization.Localization(__file__, 758, 26), matdims_136910, *[arr_136911, oned_as_136913], **kwargs_136914)
        
        # Getting the type of 'mxOBJECT_CLASS' (line 759)
        mxOBJECT_CLASS_136916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 26), 'mxOBJECT_CLASS', False)
        # Processing the call keyword arguments (line 758)
        kwargs_136917 = {}
        # Getting the type of 'self' (line 758)
        self_136908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 758)
        write_header_136909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 8), self_136908, 'write_header')
        # Calling write_header(args, kwargs) (line 758)
        write_header_call_result_136918 = invoke(stypy.reporting.localization.Localization(__file__, 758, 8), write_header_136909, *[matdims_call_result_136915, mxOBJECT_CLASS_136916], **kwargs_136917)
        
        
        # Call to write_element(...): (line 760)
        # Processing the call arguments (line 760)
        
        # Call to array(...): (line 760)
        # Processing the call arguments (line 760)
        # Getting the type of 'arr' (line 760)
        arr_136923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 36), 'arr', False)
        # Obtaining the member 'classname' of a type (line 760)
        classname_136924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 36), arr_136923, 'classname')
        # Processing the call keyword arguments (line 760)
        str_136925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 57), 'str', 'S')
        keyword_136926 = str_136925
        kwargs_136927 = {'dtype': keyword_136926}
        # Getting the type of 'np' (line 760)
        np_136921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 760)
        array_136922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 27), np_136921, 'array')
        # Calling array(args, kwargs) (line 760)
        array_call_result_136928 = invoke(stypy.reporting.localization.Localization(__file__, 760, 27), array_136922, *[classname_136924], **kwargs_136927)
        
        # Processing the call keyword arguments (line 760)
        # Getting the type of 'miINT8' (line 761)
        miINT8_136929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 34), 'miINT8', False)
        keyword_136930 = miINT8_136929
        kwargs_136931 = {'mdtype': keyword_136930}
        # Getting the type of 'self' (line 760)
        self_136919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'self', False)
        # Obtaining the member 'write_element' of a type (line 760)
        write_element_136920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), self_136919, 'write_element')
        # Calling write_element(args, kwargs) (line 760)
        write_element_call_result_136932 = invoke(stypy.reporting.localization.Localization(__file__, 760, 8), write_element_136920, *[array_call_result_136928], **kwargs_136931)
        
        
        # Call to _write_items(...): (line 762)
        # Processing the call arguments (line 762)
        # Getting the type of 'arr' (line 762)
        arr_136935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 26), 'arr', False)
        # Processing the call keyword arguments (line 762)
        kwargs_136936 = {}
        # Getting the type of 'self' (line 762)
        self_136933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'self', False)
        # Obtaining the member '_write_items' of a type (line 762)
        _write_items_136934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 8), self_136933, '_write_items')
        # Calling _write_items(args, kwargs) (line 762)
        _write_items_call_result_136937 = invoke(stypy.reporting.localization.Localization(__file__, 762, 8), _write_items_136934, *[arr_136935], **kwargs_136936)
        
        
        # ################# End of 'write_object(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_object' in the type store
        # Getting the type of 'stypy_return_type' (line 754)
        stypy_return_type_136938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_object'
        return stypy_return_type_136938


# Assigning a type to the variable 'VarWriter5' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'VarWriter5', VarWriter5)

# Assigning a Call to a Name (line 464):

# Call to zeros(...): (line 464)
# Processing the call arguments (line 464)

# Obtaining an instance of the builtin type 'tuple' (line 464)
tuple_136941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 464)

# Getting the type of 'NDT_TAG_FULL' (line 464)
NDT_TAG_FULL_136942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 27), 'NDT_TAG_FULL', False)
# Processing the call keyword arguments (line 464)
kwargs_136943 = {}
# Getting the type of 'np' (line 464)
np_136939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 14), 'np', False)
# Obtaining the member 'zeros' of a type (line 464)
zeros_136940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 14), np_136939, 'zeros')
# Calling zeros(args, kwargs) (line 464)
zeros_call_result_136944 = invoke(stypy.reporting.localization.Localization(__file__, 464, 14), zeros_136940, *[tuple_136941, NDT_TAG_FULL_136942], **kwargs_136943)

# Getting the type of 'VarWriter5'
VarWriter5_136945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VarWriter5')
# Setting the type of the member 'mat_tag' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VarWriter5_136945, 'mat_tag', zeros_call_result_136944)

# Assigning a Name to a Subscript (line 465):
# Getting the type of 'miMATRIX' (line 465)
miMATRIX_136946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 24), 'miMATRIX')
# Getting the type of 'VarWriter5'
VarWriter5_136947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VarWriter5')
# Setting the type of the member 'mat_tag' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VarWriter5_136947, 'mat_tag', miMATRIX_136946)
# Declaration of the 'MatFile5Writer' class

class MatFile5Writer(object, ):
    str_136948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 4), 'str', ' Class for writing mat5 files ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 770)
        False_136949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 32), 'False')
        # Getting the type of 'False' (line 771)
        False_136950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 33), 'False')
        # Getting the type of 'None' (line 772)
        None_136951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 29), 'None')
        # Getting the type of 'False' (line 773)
        False_136952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 34), 'False')
        str_136953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 25), 'str', 'row')
        defaults = [False_136949, False_136950, None_136951, False_136952, str_136953]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 768, 4, False)
        # Assigning a type to the variable 'self' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Writer.__init__', ['file_stream', 'do_compression', 'unicode_strings', 'global_vars', 'long_field_names', 'oned_as'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_stream', 'do_compression', 'unicode_strings', 'global_vars', 'long_field_names', 'oned_as'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_136954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, (-1)), 'str', ' Initialize writer for matlab 5 format files\n\n        Parameters\n        ----------\n        %(do_compression)s\n        %(unicode_strings)s\n        global_vars : None or sequence of strings, optional\n            Names of variables to be marked as global for matlab\n        %(long_fields)s\n        %(oned_as)s\n        ')
        
        # Assigning a Name to a Attribute (line 786):
        
        # Assigning a Name to a Attribute (line 786):
        # Getting the type of 'file_stream' (line 786)
        file_stream_136955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 27), 'file_stream')
        # Getting the type of 'self' (line 786)
        self_136956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'self')
        # Setting the type of the member 'file_stream' of a type (line 786)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 8), self_136956, 'file_stream', file_stream_136955)
        
        # Assigning a Name to a Attribute (line 787):
        
        # Assigning a Name to a Attribute (line 787):
        # Getting the type of 'do_compression' (line 787)
        do_compression_136957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 30), 'do_compression')
        # Getting the type of 'self' (line 787)
        self_136958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'self')
        # Setting the type of the member 'do_compression' of a type (line 787)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 8), self_136958, 'do_compression', do_compression_136957)
        
        # Assigning a Name to a Attribute (line 788):
        
        # Assigning a Name to a Attribute (line 788):
        # Getting the type of 'unicode_strings' (line 788)
        unicode_strings_136959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 31), 'unicode_strings')
        # Getting the type of 'self' (line 788)
        self_136960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'self')
        # Setting the type of the member 'unicode_strings' of a type (line 788)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 8), self_136960, 'unicode_strings', unicode_strings_136959)
        
        # Getting the type of 'global_vars' (line 789)
        global_vars_136961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 11), 'global_vars')
        # Testing the type of an if condition (line 789)
        if_condition_136962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 789, 8), global_vars_136961)
        # Assigning a type to the variable 'if_condition_136962' (line 789)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'if_condition_136962', if_condition_136962)
        # SSA begins for if statement (line 789)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 790):
        
        # Assigning a Name to a Attribute (line 790):
        # Getting the type of 'global_vars' (line 790)
        global_vars_136963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 31), 'global_vars')
        # Getting the type of 'self' (line 790)
        self_136964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 12), 'self')
        # Setting the type of the member 'global_vars' of a type (line 790)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 12), self_136964, 'global_vars', global_vars_136963)
        # SSA branch for the else part of an if statement (line 789)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 792):
        
        # Assigning a List to a Attribute (line 792):
        
        # Obtaining an instance of the builtin type 'list' (line 792)
        list_136965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 792)
        
        # Getting the type of 'self' (line 792)
        self_136966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'self')
        # Setting the type of the member 'global_vars' of a type (line 792)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 12), self_136966, 'global_vars', list_136965)
        # SSA join for if statement (line 789)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 793):
        
        # Assigning a Name to a Attribute (line 793):
        # Getting the type of 'long_field_names' (line 793)
        long_field_names_136967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 32), 'long_field_names')
        # Getting the type of 'self' (line 793)
        self_136968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'self')
        # Setting the type of the member 'long_field_names' of a type (line 793)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), self_136968, 'long_field_names', long_field_names_136967)
        
        # Assigning a Name to a Attribute (line 794):
        
        # Assigning a Name to a Attribute (line 794):
        # Getting the type of 'oned_as' (line 794)
        oned_as_136969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 23), 'oned_as')
        # Getting the type of 'self' (line 794)
        self_136970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'self')
        # Setting the type of the member 'oned_as' of a type (line 794)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), self_136970, 'oned_as', oned_as_136969)
        
        # Assigning a Name to a Attribute (line 795):
        
        # Assigning a Name to a Attribute (line 795):
        # Getting the type of 'None' (line 795)
        None_136971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 30), 'None')
        # Getting the type of 'self' (line 795)
        self_136972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'self')
        # Setting the type of the member '_matrix_writer' of a type (line 795)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 8), self_136972, '_matrix_writer', None_136971)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def write_file_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_file_header'
        module_type_store = module_type_store.open_function_context('write_file_header', 797, 4, False)
        # Assigning a type to the variable 'self' (line 798)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_function_name', 'MatFile5Writer.write_file_header')
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Writer.write_file_header.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Writer.write_file_header', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_file_header', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_file_header(...)' code ##################

        
        # Assigning a Call to a Name (line 799):
        
        # Assigning a Call to a Name (line 799):
        
        # Call to zeros(...): (line 799)
        # Processing the call arguments (line 799)
        
        # Obtaining an instance of the builtin type 'tuple' (line 799)
        tuple_136975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 799)
        
        # Getting the type of 'NDT_FILE_HDR' (line 799)
        NDT_FILE_HDR_136976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 27), 'NDT_FILE_HDR', False)
        # Processing the call keyword arguments (line 799)
        kwargs_136977 = {}
        # Getting the type of 'np' (line 799)
        np_136973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 14), 'np', False)
        # Obtaining the member 'zeros' of a type (line 799)
        zeros_136974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 14), np_136973, 'zeros')
        # Calling zeros(args, kwargs) (line 799)
        zeros_call_result_136978 = invoke(stypy.reporting.localization.Localization(__file__, 799, 14), zeros_136974, *[tuple_136975, NDT_FILE_HDR_136976], **kwargs_136977)
        
        # Assigning a type to the variable 'hdr' (line 799)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'hdr', zeros_call_result_136978)
        
        # Assigning a BinOp to a Subscript (line 800):
        
        # Assigning a BinOp to a Subscript (line 800):
        str_136979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 29), 'str', 'MATLAB 5.0 MAT-file Platform: %s, Created on: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 801)
        tuple_136980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 801)
        # Adding element type (line 801)
        # Getting the type of 'os' (line 801)
        os_136981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 15), 'os')
        # Obtaining the member 'name' of a type (line 801)
        name_136982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 15), os_136981, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 15), tuple_136980, name_136982)
        # Adding element type (line 801)
        
        # Call to asctime(...): (line 801)
        # Processing the call keyword arguments (line 801)
        kwargs_136985 = {}
        # Getting the type of 'time' (line 801)
        time_136983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 23), 'time', False)
        # Obtaining the member 'asctime' of a type (line 801)
        asctime_136984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 23), time_136983, 'asctime')
        # Calling asctime(args, kwargs) (line 801)
        asctime_call_result_136986 = invoke(stypy.reporting.localization.Localization(__file__, 801, 23), asctime_136984, *[], **kwargs_136985)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 15), tuple_136980, asctime_call_result_136986)
        
        # Applying the binary operator '%' (line 800)
        result_mod_136987 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 29), '%', str_136979, tuple_136980)
        
        # Getting the type of 'hdr' (line 800)
        hdr_136988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'hdr')
        str_136989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 12), 'str', 'description')
        # Storing an element on a container (line 800)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 800, 8), hdr_136988, (str_136989, result_mod_136987))
        
        # Assigning a Num to a Subscript (line 802):
        
        # Assigning a Num to a Subscript (line 802):
        int_136990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 25), 'int')
        # Getting the type of 'hdr' (line 802)
        hdr_136991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'hdr')
        str_136992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 12), 'str', 'version')
        # Storing an element on a container (line 802)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 802, 8), hdr_136991, (str_136992, int_136990))
        
        # Assigning a Call to a Subscript (line 803):
        
        # Assigning a Call to a Subscript (line 803):
        
        # Call to ndarray(...): (line 803)
        # Processing the call keyword arguments (line 803)
        
        # Obtaining an instance of the builtin type 'tuple' (line 803)
        tuple_136995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 803)
        
        keyword_136996 = tuple_136995
        str_136997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 44), 'str', 'S2')
        keyword_136998 = str_136997
        
        # Call to uint16(...): (line 805)
        # Processing the call arguments (line 805)
        int_137001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 55), 'int')
        # Processing the call keyword arguments (line 805)
        kwargs_137002 = {}
        # Getting the type of 'np' (line 805)
        np_136999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 45), 'np', False)
        # Obtaining the member 'uint16' of a type (line 805)
        uint16_137000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 45), np_136999, 'uint16')
        # Calling uint16(args, kwargs) (line 805)
        uint16_call_result_137003 = invoke(stypy.reporting.localization.Localization(__file__, 805, 45), uint16_137000, *[int_137001], **kwargs_137002)
        
        keyword_137004 = uint16_call_result_137003
        kwargs_137005 = {'buffer': keyword_137004, 'dtype': keyword_136998, 'shape': keyword_136996}
        # Getting the type of 'np' (line 803)
        np_136993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 29), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 803)
        ndarray_136994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 29), np_136993, 'ndarray')
        # Calling ndarray(args, kwargs) (line 803)
        ndarray_call_result_137006 = invoke(stypy.reporting.localization.Localization(__file__, 803, 29), ndarray_136994, *[], **kwargs_137005)
        
        # Getting the type of 'hdr' (line 803)
        hdr_137007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'hdr')
        str_137008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 12), 'str', 'endian_test')
        # Storing an element on a container (line 803)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 803, 8), hdr_137007, (str_137008, ndarray_call_result_137006))
        
        # Call to write(...): (line 806)
        # Processing the call arguments (line 806)
        
        # Call to tostring(...): (line 806)
        # Processing the call keyword arguments (line 806)
        kwargs_137014 = {}
        # Getting the type of 'hdr' (line 806)
        hdr_137012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 31), 'hdr', False)
        # Obtaining the member 'tostring' of a type (line 806)
        tostring_137013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 31), hdr_137012, 'tostring')
        # Calling tostring(args, kwargs) (line 806)
        tostring_call_result_137015 = invoke(stypy.reporting.localization.Localization(__file__, 806, 31), tostring_137013, *[], **kwargs_137014)
        
        # Processing the call keyword arguments (line 806)
        kwargs_137016 = {}
        # Getting the type of 'self' (line 806)
        self_137009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 806)
        file_stream_137010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 8), self_137009, 'file_stream')
        # Obtaining the member 'write' of a type (line 806)
        write_137011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 8), file_stream_137010, 'write')
        # Calling write(args, kwargs) (line 806)
        write_call_result_137017 = invoke(stypy.reporting.localization.Localization(__file__, 806, 8), write_137011, *[tostring_call_result_137015], **kwargs_137016)
        
        
        # ################# End of 'write_file_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_file_header' in the type store
        # Getting the type of 'stypy_return_type' (line 797)
        stypy_return_type_137018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_file_header'
        return stypy_return_type_137018


    @norecursion
    def put_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 808)
        None_137019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 48), 'None')
        defaults = [None_137019]
        # Create a new context for function 'put_variables'
        module_type_store = module_type_store.open_function_context('put_variables', 808, 4, False)
        # Assigning a type to the variable 'self' (line 809)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_localization', localization)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_function_name', 'MatFile5Writer.put_variables')
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_param_names_list', ['mdict', 'write_header'])
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile5Writer.put_variables.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile5Writer.put_variables', ['mdict', 'write_header'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'put_variables', localization, ['mdict', 'write_header'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'put_variables(...)' code ##################

        str_137020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, (-1)), 'str', ' Write variables in `mdict` to stream\n\n        Parameters\n        ----------\n        mdict : mapping\n           mapping with method ``items`` returns name, contents pairs where\n           ``name`` which will appear in the matlab workspace in file load, and\n           ``contents`` is something writeable to a matlab file, such as a numpy\n           array.\n        write_header : {None, True, False}, optional\n           If True, then write the matlab file header before writing the\n           variables.  If None (the default) then write the file header\n           if we are at position 0 in the stream.  By setting False\n           here, and setting the stream position to the end of the file,\n           you can append variables to a matlab file\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 826)
        # Getting the type of 'write_header' (line 826)
        write_header_137021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 11), 'write_header')
        # Getting the type of 'None' (line 826)
        None_137022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 27), 'None')
        
        (may_be_137023, more_types_in_union_137024) = may_be_none(write_header_137021, None_137022)

        if may_be_137023:

            if more_types_in_union_137024:
                # Runtime conditional SSA (line 826)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Compare to a Name (line 827):
            
            # Assigning a Compare to a Name (line 827):
            
            
            # Call to tell(...): (line 827)
            # Processing the call keyword arguments (line 827)
            kwargs_137028 = {}
            # Getting the type of 'self' (line 827)
            self_137025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 27), 'self', False)
            # Obtaining the member 'file_stream' of a type (line 827)
            file_stream_137026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 27), self_137025, 'file_stream')
            # Obtaining the member 'tell' of a type (line 827)
            tell_137027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 27), file_stream_137026, 'tell')
            # Calling tell(args, kwargs) (line 827)
            tell_call_result_137029 = invoke(stypy.reporting.localization.Localization(__file__, 827, 27), tell_137027, *[], **kwargs_137028)
            
            int_137030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 54), 'int')
            # Applying the binary operator '==' (line 827)
            result_eq_137031 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 27), '==', tell_call_result_137029, int_137030)
            
            # Assigning a type to the variable 'write_header' (line 827)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 12), 'write_header', result_eq_137031)

            if more_types_in_union_137024:
                # SSA join for if statement (line 826)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'write_header' (line 828)
        write_header_137032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 11), 'write_header')
        # Testing the type of an if condition (line 828)
        if_condition_137033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 828, 8), write_header_137032)
        # Assigning a type to the variable 'if_condition_137033' (line 828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'if_condition_137033', if_condition_137033)
        # SSA begins for if statement (line 828)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_file_header(...): (line 829)
        # Processing the call keyword arguments (line 829)
        kwargs_137036 = {}
        # Getting the type of 'self' (line 829)
        self_137034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 12), 'self', False)
        # Obtaining the member 'write_file_header' of a type (line 829)
        write_file_header_137035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 12), self_137034, 'write_file_header')
        # Calling write_file_header(args, kwargs) (line 829)
        write_file_header_call_result_137037 = invoke(stypy.reporting.localization.Localization(__file__, 829, 12), write_file_header_137035, *[], **kwargs_137036)
        
        # SSA join for if statement (line 828)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 830):
        
        # Assigning a Call to a Attribute (line 830):
        
        # Call to VarWriter5(...): (line 830)
        # Processing the call arguments (line 830)
        # Getting the type of 'self' (line 830)
        self_137039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 41), 'self', False)
        # Processing the call keyword arguments (line 830)
        kwargs_137040 = {}
        # Getting the type of 'VarWriter5' (line 830)
        VarWriter5_137038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 30), 'VarWriter5', False)
        # Calling VarWriter5(args, kwargs) (line 830)
        VarWriter5_call_result_137041 = invoke(stypy.reporting.localization.Localization(__file__, 830, 30), VarWriter5_137038, *[self_137039], **kwargs_137040)
        
        # Getting the type of 'self' (line 830)
        self_137042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'self')
        # Setting the type of the member '_matrix_writer' of a type (line 830)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 8), self_137042, '_matrix_writer', VarWriter5_call_result_137041)
        
        
        # Call to items(...): (line 831)
        # Processing the call keyword arguments (line 831)
        kwargs_137045 = {}
        # Getting the type of 'mdict' (line 831)
        mdict_137043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 25), 'mdict', False)
        # Obtaining the member 'items' of a type (line 831)
        items_137044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 25), mdict_137043, 'items')
        # Calling items(args, kwargs) (line 831)
        items_call_result_137046 = invoke(stypy.reporting.localization.Localization(__file__, 831, 25), items_137044, *[], **kwargs_137045)
        
        # Testing the type of a for loop iterable (line 831)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 831, 8), items_call_result_137046)
        # Getting the type of the for loop variable (line 831)
        for_loop_var_137047 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 831, 8), items_call_result_137046)
        # Assigning a type to the variable 'name' (line 831)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 8), for_loop_var_137047))
        # Assigning a type to the variable 'var' (line 831)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'var', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 8), for_loop_var_137047))
        # SSA begins for a for statement (line 831)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        int_137048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 20), 'int')
        # Getting the type of 'name' (line 832)
        name_137049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 15), 'name')
        # Obtaining the member '__getitem__' of a type (line 832)
        getitem___137050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 15), name_137049, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 832)
        subscript_call_result_137051 = invoke(stypy.reporting.localization.Localization(__file__, 832, 15), getitem___137050, int_137048)
        
        str_137052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 26), 'str', '_')
        # Applying the binary operator '==' (line 832)
        result_eq_137053 = python_operator(stypy.reporting.localization.Localization(__file__, 832, 15), '==', subscript_call_result_137051, str_137052)
        
        # Testing the type of an if condition (line 832)
        if_condition_137054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 832, 12), result_eq_137053)
        # Assigning a type to the variable 'if_condition_137054' (line 832)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'if_condition_137054', if_condition_137054)
        # SSA begins for if statement (line 832)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 832)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Compare to a Name (line 834):
        
        # Assigning a Compare to a Name (line 834):
        
        # Getting the type of 'name' (line 834)
        name_137055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 24), 'name')
        # Getting the type of 'self' (line 834)
        self_137056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 32), 'self')
        # Obtaining the member 'global_vars' of a type (line 834)
        global_vars_137057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 32), self_137056, 'global_vars')
        # Applying the binary operator 'in' (line 834)
        result_contains_137058 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 24), 'in', name_137055, global_vars_137057)
        
        # Assigning a type to the variable 'is_global' (line 834)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 12), 'is_global', result_contains_137058)
        
        # Getting the type of 'self' (line 835)
        self_137059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 15), 'self')
        # Obtaining the member 'do_compression' of a type (line 835)
        do_compression_137060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 15), self_137059, 'do_compression')
        # Testing the type of an if condition (line 835)
        if_condition_137061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 835, 12), do_compression_137060)
        # Assigning a type to the variable 'if_condition_137061' (line 835)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 12), 'if_condition_137061', if_condition_137061)
        # SSA begins for if statement (line 835)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 836):
        
        # Assigning a Call to a Name (line 836):
        
        # Call to BytesIO(...): (line 836)
        # Processing the call keyword arguments (line 836)
        kwargs_137063 = {}
        # Getting the type of 'BytesIO' (line 836)
        BytesIO_137062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 25), 'BytesIO', False)
        # Calling BytesIO(args, kwargs) (line 836)
        BytesIO_call_result_137064 = invoke(stypy.reporting.localization.Localization(__file__, 836, 25), BytesIO_137062, *[], **kwargs_137063)
        
        # Assigning a type to the variable 'stream' (line 836)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 16), 'stream', BytesIO_call_result_137064)
        
        # Assigning a Name to a Attribute (line 837):
        
        # Assigning a Name to a Attribute (line 837):
        # Getting the type of 'stream' (line 837)
        stream_137065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 50), 'stream')
        # Getting the type of 'self' (line 837)
        self_137066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 16), 'self')
        # Obtaining the member '_matrix_writer' of a type (line 837)
        _matrix_writer_137067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 16), self_137066, '_matrix_writer')
        # Setting the type of the member 'file_stream' of a type (line 837)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 16), _matrix_writer_137067, 'file_stream', stream_137065)
        
        # Call to write_top(...): (line 838)
        # Processing the call arguments (line 838)
        # Getting the type of 'var' (line 838)
        var_137071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 46), 'var', False)
        
        # Call to asbytes(...): (line 838)
        # Processing the call arguments (line 838)
        # Getting the type of 'name' (line 838)
        name_137073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 59), 'name', False)
        # Processing the call keyword arguments (line 838)
        kwargs_137074 = {}
        # Getting the type of 'asbytes' (line 838)
        asbytes_137072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 51), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 838)
        asbytes_call_result_137075 = invoke(stypy.reporting.localization.Localization(__file__, 838, 51), asbytes_137072, *[name_137073], **kwargs_137074)
        
        # Getting the type of 'is_global' (line 838)
        is_global_137076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 66), 'is_global', False)
        # Processing the call keyword arguments (line 838)
        kwargs_137077 = {}
        # Getting the type of 'self' (line 838)
        self_137068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 16), 'self', False)
        # Obtaining the member '_matrix_writer' of a type (line 838)
        _matrix_writer_137069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 16), self_137068, '_matrix_writer')
        # Obtaining the member 'write_top' of a type (line 838)
        write_top_137070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 16), _matrix_writer_137069, 'write_top')
        # Calling write_top(args, kwargs) (line 838)
        write_top_call_result_137078 = invoke(stypy.reporting.localization.Localization(__file__, 838, 16), write_top_137070, *[var_137071, asbytes_call_result_137075, is_global_137076], **kwargs_137077)
        
        
        # Assigning a Call to a Name (line 839):
        
        # Assigning a Call to a Name (line 839):
        
        # Call to compress(...): (line 839)
        # Processing the call arguments (line 839)
        
        # Call to getvalue(...): (line 839)
        # Processing the call keyword arguments (line 839)
        kwargs_137083 = {}
        # Getting the type of 'stream' (line 839)
        stream_137081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 40), 'stream', False)
        # Obtaining the member 'getvalue' of a type (line 839)
        getvalue_137082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 40), stream_137081, 'getvalue')
        # Calling getvalue(args, kwargs) (line 839)
        getvalue_call_result_137084 = invoke(stypy.reporting.localization.Localization(__file__, 839, 40), getvalue_137082, *[], **kwargs_137083)
        
        # Processing the call keyword arguments (line 839)
        kwargs_137085 = {}
        # Getting the type of 'zlib' (line 839)
        zlib_137079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 26), 'zlib', False)
        # Obtaining the member 'compress' of a type (line 839)
        compress_137080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 26), zlib_137079, 'compress')
        # Calling compress(args, kwargs) (line 839)
        compress_call_result_137086 = invoke(stypy.reporting.localization.Localization(__file__, 839, 26), compress_137080, *[getvalue_call_result_137084], **kwargs_137085)
        
        # Assigning a type to the variable 'out_str' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 16), 'out_str', compress_call_result_137086)
        
        # Assigning a Call to a Name (line 840):
        
        # Assigning a Call to a Name (line 840):
        
        # Call to empty(...): (line 840)
        # Processing the call arguments (line 840)
        
        # Obtaining an instance of the builtin type 'tuple' (line 840)
        tuple_137089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 840)
        
        # Getting the type of 'NDT_TAG_FULL' (line 840)
        NDT_TAG_FULL_137090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 35), 'NDT_TAG_FULL', False)
        # Processing the call keyword arguments (line 840)
        kwargs_137091 = {}
        # Getting the type of 'np' (line 840)
        np_137087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 22), 'np', False)
        # Obtaining the member 'empty' of a type (line 840)
        empty_137088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 22), np_137087, 'empty')
        # Calling empty(args, kwargs) (line 840)
        empty_call_result_137092 = invoke(stypy.reporting.localization.Localization(__file__, 840, 22), empty_137088, *[tuple_137089, NDT_TAG_FULL_137090], **kwargs_137091)
        
        # Assigning a type to the variable 'tag' (line 840)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'tag', empty_call_result_137092)
        
        # Assigning a Name to a Subscript (line 841):
        
        # Assigning a Name to a Subscript (line 841):
        # Getting the type of 'miCOMPRESSED' (line 841)
        miCOMPRESSED_137093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 32), 'miCOMPRESSED')
        # Getting the type of 'tag' (line 841)
        tag_137094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 16), 'tag')
        str_137095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 20), 'str', 'mdtype')
        # Storing an element on a container (line 841)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 16), tag_137094, (str_137095, miCOMPRESSED_137093))
        
        # Assigning a Call to a Subscript (line 842):
        
        # Assigning a Call to a Subscript (line 842):
        
        # Call to len(...): (line 842)
        # Processing the call arguments (line 842)
        # Getting the type of 'out_str' (line 842)
        out_str_137097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 40), 'out_str', False)
        # Processing the call keyword arguments (line 842)
        kwargs_137098 = {}
        # Getting the type of 'len' (line 842)
        len_137096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 36), 'len', False)
        # Calling len(args, kwargs) (line 842)
        len_call_result_137099 = invoke(stypy.reporting.localization.Localization(__file__, 842, 36), len_137096, *[out_str_137097], **kwargs_137098)
        
        # Getting the type of 'tag' (line 842)
        tag_137100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 16), 'tag')
        str_137101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 20), 'str', 'byte_count')
        # Storing an element on a container (line 842)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 16), tag_137100, (str_137101, len_call_result_137099))
        
        # Call to write(...): (line 843)
        # Processing the call arguments (line 843)
        
        # Call to tostring(...): (line 843)
        # Processing the call keyword arguments (line 843)
        kwargs_137107 = {}
        # Getting the type of 'tag' (line 843)
        tag_137105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 39), 'tag', False)
        # Obtaining the member 'tostring' of a type (line 843)
        tostring_137106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 39), tag_137105, 'tostring')
        # Calling tostring(args, kwargs) (line 843)
        tostring_call_result_137108 = invoke(stypy.reporting.localization.Localization(__file__, 843, 39), tostring_137106, *[], **kwargs_137107)
        
        # Processing the call keyword arguments (line 843)
        kwargs_137109 = {}
        # Getting the type of 'self' (line 843)
        self_137102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 16), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 843)
        file_stream_137103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 16), self_137102, 'file_stream')
        # Obtaining the member 'write' of a type (line 843)
        write_137104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 16), file_stream_137103, 'write')
        # Calling write(args, kwargs) (line 843)
        write_call_result_137110 = invoke(stypy.reporting.localization.Localization(__file__, 843, 16), write_137104, *[tostring_call_result_137108], **kwargs_137109)
        
        
        # Call to write(...): (line 844)
        # Processing the call arguments (line 844)
        # Getting the type of 'out_str' (line 844)
        out_str_137114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 39), 'out_str', False)
        # Processing the call keyword arguments (line 844)
        kwargs_137115 = {}
        # Getting the type of 'self' (line 844)
        self_137111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 16), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 844)
        file_stream_137112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 16), self_137111, 'file_stream')
        # Obtaining the member 'write' of a type (line 844)
        write_137113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 16), file_stream_137112, 'write')
        # Calling write(args, kwargs) (line 844)
        write_call_result_137116 = invoke(stypy.reporting.localization.Localization(__file__, 844, 16), write_137113, *[out_str_137114], **kwargs_137115)
        
        # SSA branch for the else part of an if statement (line 835)
        module_type_store.open_ssa_branch('else')
        
        # Call to write_top(...): (line 846)
        # Processing the call arguments (line 846)
        # Getting the type of 'var' (line 846)
        var_137120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 46), 'var', False)
        
        # Call to asbytes(...): (line 846)
        # Processing the call arguments (line 846)
        # Getting the type of 'name' (line 846)
        name_137122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 59), 'name', False)
        # Processing the call keyword arguments (line 846)
        kwargs_137123 = {}
        # Getting the type of 'asbytes' (line 846)
        asbytes_137121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 51), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 846)
        asbytes_call_result_137124 = invoke(stypy.reporting.localization.Localization(__file__, 846, 51), asbytes_137121, *[name_137122], **kwargs_137123)
        
        # Getting the type of 'is_global' (line 846)
        is_global_137125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 66), 'is_global', False)
        # Processing the call keyword arguments (line 846)
        kwargs_137126 = {}
        # Getting the type of 'self' (line 846)
        self_137117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 16), 'self', False)
        # Obtaining the member '_matrix_writer' of a type (line 846)
        _matrix_writer_137118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 16), self_137117, '_matrix_writer')
        # Obtaining the member 'write_top' of a type (line 846)
        write_top_137119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 16), _matrix_writer_137118, 'write_top')
        # Calling write_top(args, kwargs) (line 846)
        write_top_call_result_137127 = invoke(stypy.reporting.localization.Localization(__file__, 846, 16), write_top_137119, *[var_137120, asbytes_call_result_137124, is_global_137125], **kwargs_137126)
        
        # SSA join for if statement (line 835)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'put_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'put_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 808)
        stypy_return_type_137128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'put_variables'
        return stypy_return_type_137128


# Assigning a type to the variable 'MatFile5Writer' (line 765)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 0), 'MatFile5Writer', MatFile5Writer)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
