
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Classes for read / write of matlab (TM) 4 files
2: '''
3: from __future__ import division, print_function, absolute_import
4: 
5: import sys
6: import warnings
7: 
8: import numpy as np
9: from numpy.compat import asbytes, asstr
10: 
11: import scipy.sparse
12: 
13: from scipy._lib.six import string_types
14: 
15: from .miobase import (MatFileReader, docfiller, matdims, read_dtype,
16:                       convert_dtypes, arr_to_chars, arr_dtype_number)
17: 
18: from .mio_utils import squeeze_element, chars_to_strings
19: from functools import reduce
20: 
21: 
22: SYS_LITTLE_ENDIAN = sys.byteorder == 'little'
23: 
24: miDOUBLE = 0
25: miSINGLE = 1
26: miINT32 = 2
27: miINT16 = 3
28: miUINT16 = 4
29: miUINT8 = 5
30: 
31: mdtypes_template = {
32:     miDOUBLE: 'f8',
33:     miSINGLE: 'f4',
34:     miINT32: 'i4',
35:     miINT16: 'i2',
36:     miUINT16: 'u2',
37:     miUINT8: 'u1',
38:     'header': [('mopt', 'i4'),
39:                ('mrows', 'i4'),
40:                ('ncols', 'i4'),
41:                ('imagf', 'i4'),
42:                ('namlen', 'i4')],
43:     'U1': 'U1',
44:     }
45: 
46: np_to_mtypes = {
47:     'f8': miDOUBLE,
48:     'c32': miDOUBLE,
49:     'c24': miDOUBLE,
50:     'c16': miDOUBLE,
51:     'f4': miSINGLE,
52:     'c8': miSINGLE,
53:     'i4': miINT32,
54:     'i2': miINT16,
55:     'u2': miUINT16,
56:     'u1': miUINT8,
57:     'S1': miUINT8,
58:     }
59: 
60: # matrix classes
61: mxFULL_CLASS = 0
62: mxCHAR_CLASS = 1
63: mxSPARSE_CLASS = 2
64: 
65: order_codes = {
66:     0: '<',
67:     1: '>',
68:     2: 'VAX D-float',  # !
69:     3: 'VAX G-float',
70:     4: 'Cray',  # !!
71:     }
72: 
73: mclass_info = {
74:     mxFULL_CLASS: 'double',
75:     mxCHAR_CLASS: 'char',
76:     mxSPARSE_CLASS: 'sparse',
77:     }
78: 
79: 
80: class VarHeader4(object):
81:     # Mat4 variables never logical or global
82:     is_logical = False
83:     is_global = False
84: 
85:     def __init__(self,
86:                  name,
87:                  dtype,
88:                  mclass,
89:                  dims,
90:                  is_complex):
91:         self.name = name
92:         self.dtype = dtype
93:         self.mclass = mclass
94:         self.dims = dims
95:         self.is_complex = is_complex
96: 
97: 
98: class VarReader4(object):
99:     ''' Class to read matlab 4 variables '''
100: 
101:     def __init__(self, file_reader):
102:         self.file_reader = file_reader
103:         self.mat_stream = file_reader.mat_stream
104:         self.dtypes = file_reader.dtypes
105:         self.chars_as_strings = file_reader.chars_as_strings
106:         self.squeeze_me = file_reader.squeeze_me
107: 
108:     def read_header(self):
109:         ''' Read and return header for variable '''
110:         data = read_dtype(self.mat_stream, self.dtypes['header'])
111:         name = self.mat_stream.read(int(data['namlen'])).strip(b'\x00')
112:         if data['mopt'] < 0 or data['mopt'] > 5000:
113:             raise ValueError('Mat 4 mopt wrong format, byteswapping problem?')
114:         M, rest = divmod(data['mopt'], 1000)  # order code
115:         if M not in (0, 1):
116:             warnings.warn("We do not support byte ordering '%s'; returned "
117:                           "data may be corrupt" % order_codes[M],
118:                           UserWarning)
119:         O, rest = divmod(rest, 100)  # unused, should be 0
120:         if O != 0:
121:             raise ValueError('O in MOPT integer should be 0, wrong format?')
122:         P, rest = divmod(rest, 10)  # data type code e.g miDOUBLE (see above)
123:         T = rest  # matrix type code e.g. mxFULL_CLASS (see above)
124:         dims = (data['mrows'], data['ncols'])
125:         is_complex = data['imagf'] == 1
126:         dtype = self.dtypes[P]
127:         return VarHeader4(
128:             name,
129:             dtype,
130:             T,
131:             dims,
132:             is_complex)
133: 
134:     def array_from_header(self, hdr, process=True):
135:         mclass = hdr.mclass
136:         if mclass == mxFULL_CLASS:
137:             arr = self.read_full_array(hdr)
138:         elif mclass == mxCHAR_CLASS:
139:             arr = self.read_char_array(hdr)
140:             if process and self.chars_as_strings:
141:                 arr = chars_to_strings(arr)
142:         elif mclass == mxSPARSE_CLASS:
143:             # no current processing (below) makes sense for sparse
144:             return self.read_sparse_array(hdr)
145:         else:
146:             raise TypeError('No reader for class code %s' % mclass)
147:         if process and self.squeeze_me:
148:             return squeeze_element(arr)
149:         return arr
150: 
151:     def read_sub_array(self, hdr, copy=True):
152:         ''' Mat4 read using header `hdr` dtype and dims
153: 
154:         Parameters
155:         ----------
156:         hdr : object
157:            object with attributes ``dtype``, ``dims``.  dtype is assumed to be
158:            the correct endianness
159:         copy : bool, optional
160:            copies array before return if True (default True)
161:            (buffer is usually read only)
162: 
163:         Returns
164:         -------
165:         arr : ndarray
166:             of dtype givem by `hdr` ``dtype`` and shape givem by `hdr` ``dims``
167:         '''
168:         dt = hdr.dtype
169:         dims = hdr.dims
170:         num_bytes = dt.itemsize
171:         for d in dims:
172:             num_bytes *= d
173:         buffer = self.mat_stream.read(int(num_bytes))
174:         if len(buffer) != num_bytes:
175:             raise ValueError("Not enough bytes to read matrix '%s'; is this "
176:                              "a badly-formed file? Consider listing matrices "
177:                              "with `whosmat` and loading named matrices with "
178:                              "`variable_names` kwarg to `loadmat`" % hdr.name)
179:         arr = np.ndarray(shape=dims,
180:                          dtype=dt,
181:                          buffer=buffer,
182:                          order='F')
183:         if copy:
184:             arr = arr.copy()
185:         return arr
186: 
187:     def read_full_array(self, hdr):
188:         ''' Full (rather than sparse) matrix getter
189: 
190:         Read matrix (array) can be real or complex
191: 
192:         Parameters
193:         ----------
194:         hdr : ``VarHeader4`` instance
195: 
196:         Returns
197:         -------
198:         arr : ndarray
199:             complex array if ``hdr.is_complex`` is True, otherwise a real
200:             numeric array
201:         '''
202:         if hdr.is_complex:
203:             # avoid array copy to save memory
204:             res = self.read_sub_array(hdr, copy=False)
205:             res_j = self.read_sub_array(hdr, copy=False)
206:             return res + (res_j * 1j)
207:         return self.read_sub_array(hdr)
208: 
209:     def read_char_array(self, hdr):
210:         ''' latin-1 text matrix (char matrix) reader
211: 
212:         Parameters
213:         ----------
214:         hdr : ``VarHeader4`` instance
215: 
216:         Returns
217:         -------
218:         arr : ndarray
219:             with dtype 'U1', shape given by `hdr` ``dims``
220:         '''
221:         arr = self.read_sub_array(hdr).astype(np.uint8)
222:         S = arr.tostring().decode('latin-1')
223:         return np.ndarray(shape=hdr.dims,
224:                           dtype=np.dtype('U1'),
225:                           buffer=np.array(S)).copy()
226: 
227:     def read_sparse_array(self, hdr):
228:         ''' Read and return sparse matrix type
229: 
230:         Parameters
231:         ----------
232:         hdr : ``VarHeader4`` instance
233: 
234:         Returns
235:         -------
236:         arr : ``scipy.sparse.coo_matrix``
237:             with dtype ``float`` and shape read from the sparse matrix data
238: 
239:         Notes
240:         -----
241:         MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where
242:         N is the number of non-zero values.  Column 1 values [0:N] are the
243:         (1-based) row indices of the each non-zero value, column 2 [0:N] are the
244:         column indices, column 3 [0:N] are the (real) values.  The last values
245:         [-1,0:2] of the rows, column indices are shape[0] and shape[1]
246:         respectively of the output matrix. The last value for the values column
247:         is a padding 0. mrows and ncols values from the header give the shape of
248:         the stored matrix, here [N+1, 3].  Complex data is saved as a 4 column
249:         matrix, where the fourth column contains the imaginary component; the
250:         last value is again 0.  Complex sparse data do *not* have the header
251:         ``imagf`` field set to True; the fact that the data are complex is only
252:         detectable because there are 4 storage columns
253:         '''
254:         res = self.read_sub_array(hdr)
255:         tmp = res[:-1,:]
256:         dims = res[-1,0:2]
257:         I = np.ascontiguousarray(tmp[:,0],dtype='intc')  # fixes byte order also
258:         J = np.ascontiguousarray(tmp[:,1],dtype='intc')
259:         I -= 1  # for 1-based indexing
260:         J -= 1
261:         if res.shape[1] == 3:
262:             V = np.ascontiguousarray(tmp[:,2],dtype='float')
263:         else:
264:             V = np.ascontiguousarray(tmp[:,2],dtype='complex')
265:             V.imag = tmp[:,3]
266:         return scipy.sparse.coo_matrix((V,(I,J)), dims)
267: 
268:     def shape_from_header(self, hdr):
269:         '''Read the shape of the array described by the header.
270:         The file position after this call is unspecified.
271:         '''
272:         mclass = hdr.mclass
273:         if mclass == mxFULL_CLASS:
274:             shape = tuple(map(int, hdr.dims))
275:         elif mclass == mxCHAR_CLASS:
276:             shape = tuple(map(int, hdr.dims))
277:             if self.chars_as_strings:
278:                 shape = shape[:-1]
279:         elif mclass == mxSPARSE_CLASS:
280:             dt = hdr.dtype
281:             dims = hdr.dims
282: 
283:             if not (len(dims) == 2 and dims[0] >= 1 and dims[1] >= 1):
284:                 return ()
285: 
286:             # Read only the row and column counts
287:             self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
288:             rows = np.ndarray(shape=(1,), dtype=dt,
289:                               buffer=self.mat_stream.read(dt.itemsize))
290:             self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
291:             cols = np.ndarray(shape=(1,), dtype=dt,
292:                               buffer=self.mat_stream.read(dt.itemsize))
293: 
294:             shape = (int(rows), int(cols))
295:         else:
296:             raise TypeError('No reader for class code %s' % mclass)
297: 
298:         if self.squeeze_me:
299:             shape = tuple([x for x in shape if x != 1])
300:         return shape
301: 
302: 
303: class MatFile4Reader(MatFileReader):
304:     ''' Reader for Mat4 files '''
305:     @docfiller
306:     def __init__(self, mat_stream, *args, **kwargs):
307:         ''' Initialize matlab 4 file reader
308: 
309:     %(matstream_arg)s
310:     %(load_args)s
311:         '''
312:         super(MatFile4Reader, self).__init__(mat_stream, *args, **kwargs)
313:         self._matrix_reader = None
314: 
315:     def guess_byte_order(self):
316:         self.mat_stream.seek(0)
317:         mopt = read_dtype(self.mat_stream, np.dtype('i4'))
318:         self.mat_stream.seek(0)
319:         if mopt == 0:
320:             return '<'
321:         if mopt < 0 or mopt > 5000:
322:             # Number must have been byteswapped
323:             return SYS_LITTLE_ENDIAN and '>' or '<'
324:         # Not byteswapped
325:         return SYS_LITTLE_ENDIAN and '<' or '>'
326: 
327:     def initialize_read(self):
328:         ''' Run when beginning read of variables
329: 
330:         Sets up readers from parameters in `self`
331:         '''
332:         self.dtypes = convert_dtypes(mdtypes_template, self.byte_order)
333:         self._matrix_reader = VarReader4(self)
334: 
335:     def read_var_header(self):
336:         ''' Read and return header, next position
337: 
338:         Parameters
339:         ----------
340:         None
341: 
342:         Returns
343:         -------
344:         header : object
345:            object that can be passed to self.read_var_array, and that
346:            has attributes ``name`` and ``is_global``
347:         next_position : int
348:            position in stream of next variable
349:         '''
350:         hdr = self._matrix_reader.read_header()
351:         n = reduce(lambda x, y: x*y, hdr.dims, 1)  # fast product
352:         remaining_bytes = hdr.dtype.itemsize * n
353:         if hdr.is_complex and not hdr.mclass == mxSPARSE_CLASS:
354:             remaining_bytes *= 2
355:         next_position = self.mat_stream.tell() + remaining_bytes
356:         return hdr, next_position
357: 
358:     def read_var_array(self, header, process=True):
359:         ''' Read array, given `header`
360: 
361:         Parameters
362:         ----------
363:         header : header object
364:            object with fields defining variable header
365:         process : {True, False}, optional
366:            If True, apply recursive post-processing during loading of array.
367: 
368:         Returns
369:         -------
370:         arr : array
371:            array with post-processing applied or not according to
372:            `process`.
373:         '''
374:         return self._matrix_reader.array_from_header(header, process)
375: 
376:     def get_variables(self, variable_names=None):
377:         ''' get variables from stream as dictionary
378: 
379:         Parameters
380:         ----------
381:         variable_names : None or str or sequence of str, optional
382:             variable name, or sequence of variable names to get from Mat file /
383:             file stream.  If None, then get all variables in file
384:         '''
385:         if isinstance(variable_names, string_types):
386:             variable_names = [variable_names]
387:         elif variable_names is not None:
388:             variable_names = list(variable_names)
389:         self.mat_stream.seek(0)
390:         # set up variable reader
391:         self.initialize_read()
392:         mdict = {}
393:         while not self.end_of_stream():
394:             hdr, next_position = self.read_var_header()
395:             name = asstr(hdr.name)
396:             if variable_names is not None and name not in variable_names:
397:                 self.mat_stream.seek(next_position)
398:                 continue
399:             mdict[name] = self.read_var_array(hdr)
400:             self.mat_stream.seek(next_position)
401:             if variable_names is not None:
402:                 variable_names.remove(name)
403:                 if len(variable_names) == 0:
404:                     break
405:         return mdict
406: 
407:     def list_variables(self):
408:         ''' list variables from stream '''
409:         self.mat_stream.seek(0)
410:         # set up variable reader
411:         self.initialize_read()
412:         vars = []
413:         while not self.end_of_stream():
414:             hdr, next_position = self.read_var_header()
415:             name = asstr(hdr.name)
416:             shape = self._matrix_reader.shape_from_header(hdr)
417:             info = mclass_info.get(hdr.mclass, 'unknown')
418:             vars.append((name, shape, info))
419: 
420:             self.mat_stream.seek(next_position)
421:         return vars
422: 
423: 
424: def arr_to_2d(arr, oned_as='row'):
425:     ''' Make ``arr`` exactly two dimensional
426: 
427:     If `arr` has more than 2 dimensions, raise a ValueError
428: 
429:     Parameters
430:     ----------
431:     arr : array
432:     oned_as : {'row', 'column'}, optional
433:        Whether to reshape 1D vectors as row vectors or column vectors.
434:        See documentation for ``matdims`` for more detail
435: 
436:     Returns
437:     -------
438:     arr2d : array
439:        2D version of the array
440:     '''
441:     dims = matdims(arr, oned_as)
442:     if len(dims) > 2:
443:         raise ValueError('Matlab 4 files cannot save arrays with more than '
444:                          '2 dimensions')
445:     return arr.reshape(dims)
446: 
447: 
448: class VarWriter4(object):
449:     def __init__(self, file_writer):
450:         self.file_stream = file_writer.file_stream
451:         self.oned_as = file_writer.oned_as
452: 
453:     def write_bytes(self, arr):
454:         self.file_stream.write(arr.tostring(order='F'))
455: 
456:     def write_string(self, s):
457:         self.file_stream.write(s)
458: 
459:     def write_header(self, name, shape, P=miDOUBLE, T=mxFULL_CLASS, imagf=0):
460:         ''' Write header for given data options
461: 
462:         Parameters
463:         ----------
464:         name : str
465:             name of variable
466:         shape : sequence
467:            Shape of array as it will be read in matlab
468:         P : int, optional
469:             code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,
470:             miINT16, miUINT16, miUINT8``
471:         T : int, optional
472:             code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,
473:             mxSPARSE_CLASS``
474:         imagf : int, optional
475:             flag indicating complex
476:         '''
477:         header = np.empty((), mdtypes_template['header'])
478:         M = not SYS_LITTLE_ENDIAN
479:         O = 0
480:         header['mopt'] = (M * 1000 +
481:                           O * 100 +
482:                           P * 10 +
483:                           T)
484:         header['mrows'] = shape[0]
485:         header['ncols'] = shape[1]
486:         header['imagf'] = imagf
487:         header['namlen'] = len(name) + 1
488:         self.write_bytes(header)
489:         self.write_string(asbytes(name + '\0'))
490: 
491:     def write(self, arr, name):
492:         ''' Write matrix `arr`, with name `name`
493: 
494:         Parameters
495:         ----------
496:         arr : array_like
497:            array to write
498:         name : str
499:            name in matlab workspace
500:         '''
501:         # we need to catch sparse first, because np.asarray returns an
502:         # an object array for scipy.sparse
503:         if scipy.sparse.issparse(arr):
504:             self.write_sparse(arr, name)
505:             return
506:         arr = np.asarray(arr)
507:         dt = arr.dtype
508:         if not dt.isnative:
509:             arr = arr.astype(dt.newbyteorder('='))
510:         dtt = dt.type
511:         if dtt is np.object_:
512:             raise TypeError('Cannot save object arrays in Mat4')
513:         elif dtt is np.void:
514:             raise TypeError('Cannot save void type arrays')
515:         elif dtt in (np.unicode_, np.string_):
516:             self.write_char(arr, name)
517:             return
518:         self.write_numeric(arr, name)
519: 
520:     def write_numeric(self, arr, name):
521:         arr = arr_to_2d(arr, self.oned_as)
522:         imagf = arr.dtype.kind == 'c'
523:         try:
524:             P = np_to_mtypes[arr.dtype.str[1:]]
525:         except KeyError:
526:             if imagf:
527:                 arr = arr.astype('c128')
528:             else:
529:                 arr = arr.astype('f8')
530:             P = miDOUBLE
531:         self.write_header(name,
532:                           arr.shape,
533:                           P=P,
534:                           T=mxFULL_CLASS,
535:                           imagf=imagf)
536:         if imagf:
537:             self.write_bytes(arr.real)
538:             self.write_bytes(arr.imag)
539:         else:
540:             self.write_bytes(arr)
541: 
542:     def write_char(self, arr, name):
543:         arr = arr_to_chars(arr)
544:         arr = arr_to_2d(arr, self.oned_as)
545:         dims = arr.shape
546:         self.write_header(
547:             name,
548:             dims,
549:             P=miUINT8,
550:             T=mxCHAR_CLASS)
551:         if arr.dtype.kind == 'U':
552:             # Recode unicode to latin1
553:             n_chars = np.product(dims)
554:             st_arr = np.ndarray(shape=(),
555:                                 dtype=arr_dtype_number(arr, n_chars),
556:                                 buffer=arr)
557:             st = st_arr.item().encode('latin-1')
558:             arr = np.ndarray(shape=dims, dtype='S1', buffer=st)
559:         self.write_bytes(arr)
560: 
561:     def write_sparse(self, arr, name):
562:         ''' Sparse matrices are 2D
563: 
564:         See docstring for VarReader4.read_sparse_array
565:         '''
566:         A = arr.tocoo()  # convert to sparse COO format (ijv)
567:         imagf = A.dtype.kind == 'c'
568:         ijv = np.zeros((A.nnz + 1, 3+imagf), dtype='f8')
569:         ijv[:-1,0] = A.row
570:         ijv[:-1,1] = A.col
571:         ijv[:-1,0:2] += 1  # 1 based indexing
572:         if imagf:
573:             ijv[:-1,2] = A.data.real
574:             ijv[:-1,3] = A.data.imag
575:         else:
576:             ijv[:-1,2] = A.data
577:         ijv[-1,0:2] = A.shape
578:         self.write_header(
579:             name,
580:             ijv.shape,
581:             P=miDOUBLE,
582:             T=mxSPARSE_CLASS)
583:         self.write_bytes(ijv)
584: 
585: 
586: class MatFile4Writer(object):
587:     ''' Class for writing matlab 4 format files '''
588:     def __init__(self, file_stream, oned_as=None):
589:         self.file_stream = file_stream
590:         if oned_as is None:
591:             oned_as = 'row'
592:         self.oned_as = oned_as
593:         self._matrix_writer = None
594: 
595:     def put_variables(self, mdict, write_header=None):
596:         ''' Write variables in `mdict` to stream
597: 
598:         Parameters
599:         ----------
600:         mdict : mapping
601:            mapping with method ``items`` return name, contents pairs
602:            where ``name`` which will appeak in the matlab workspace in
603:            file load, and ``contents`` is something writeable to a
604:            matlab file, such as a numpy array.
605:         write_header : {None, True, False}
606:            If True, then write the matlab file header before writing the
607:            variables.  If None (the default) then write the file header
608:            if we are at position 0 in the stream.  By setting False
609:            here, and setting the stream position to the end of the file,
610:            you can append variables to a matlab file
611:         '''
612:         # there is no header for a matlab 4 mat file, so we ignore the
613:         # ``write_header`` input argument.  It's there for compatibility
614:         # with the matlab 5 version of this method
615:         self._matrix_writer = VarWriter4(self)
616:         for name, var in mdict.items():
617:             self._matrix_writer.write(var, name)
618: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_133767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Classes for read / write of matlab (TM) 4 files\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133768 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_133768) is not StypyTypeError):

    if (import_133768 != 'pyd_module'):
        __import__(import_133768)
        sys_modules_133769 = sys.modules[import_133768]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_133769.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_133768)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.compat import asbytes, asstr' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133770 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat')

if (type(import_133770) is not StypyTypeError):

    if (import_133770 != 'pyd_module'):
        __import__(import_133770)
        sys_modules_133771 = sys.modules[import_133770]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat', sys_modules_133771.module_type_store, module_type_store, ['asbytes', 'asstr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_133771, sys_modules_133771.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, asstr

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'asstr'], [asbytes, asstr])

else:
    # Assigning a type to the variable 'numpy.compat' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat', import_133770)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import scipy.sparse' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133772 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse')

if (type(import_133772) is not StypyTypeError):

    if (import_133772 != 'pyd_module'):
        __import__(import_133772)
        sys_modules_133773 = sys.modules[import_133772]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', sys_modules_133773.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse', import_133772)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib.six import string_types' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133774 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six')

if (type(import_133774) is not StypyTypeError):

    if (import_133774 != 'pyd_module'):
        __import__(import_133774)
        sys_modules_133775 = sys.modules[import_133774]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', sys_modules_133775.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_133775, sys_modules_133775.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', import_133774)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.io.matlab.miobase import MatFileReader, docfiller, matdims, read_dtype, convert_dtypes, arr_to_chars, arr_dtype_number' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133776 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.io.matlab.miobase')

if (type(import_133776) is not StypyTypeError):

    if (import_133776 != 'pyd_module'):
        __import__(import_133776)
        sys_modules_133777 = sys.modules[import_133776]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.io.matlab.miobase', sys_modules_133777.module_type_store, module_type_store, ['MatFileReader', 'docfiller', 'matdims', 'read_dtype', 'convert_dtypes', 'arr_to_chars', 'arr_dtype_number'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_133777, sys_modules_133777.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.miobase import MatFileReader, docfiller, matdims, read_dtype, convert_dtypes, arr_to_chars, arr_dtype_number

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.io.matlab.miobase', None, module_type_store, ['MatFileReader', 'docfiller', 'matdims', 'read_dtype', 'convert_dtypes', 'arr_to_chars', 'arr_dtype_number'], [MatFileReader, docfiller, matdims, read_dtype, convert_dtypes, arr_to_chars, arr_dtype_number])

else:
    # Assigning a type to the variable 'scipy.io.matlab.miobase' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.io.matlab.miobase', import_133776)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.io.matlab.mio_utils import squeeze_element, chars_to_strings' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133778 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.matlab.mio_utils')

if (type(import_133778) is not StypyTypeError):

    if (import_133778 != 'pyd_module'):
        __import__(import_133778)
        sys_modules_133779 = sys.modules[import_133778]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.matlab.mio_utils', sys_modules_133779.module_type_store, module_type_store, ['squeeze_element', 'chars_to_strings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_133779, sys_modules_133779.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio_utils import squeeze_element, chars_to_strings

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.matlab.mio_utils', None, module_type_store, ['squeeze_element', 'chars_to_strings'], [squeeze_element, chars_to_strings])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio_utils' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.matlab.mio_utils', import_133778)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from functools import reduce' statement (line 19)
try:
    from functools import reduce

except:
    reduce = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'functools', None, module_type_store, ['reduce'], [reduce])


# Assigning a Compare to a Name (line 22):

# Assigning a Compare to a Name (line 22):

# Getting the type of 'sys' (line 22)
sys_133780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'sys')
# Obtaining the member 'byteorder' of a type (line 22)
byteorder_133781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 20), sys_133780, 'byteorder')
str_133782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 37), 'str', 'little')
# Applying the binary operator '==' (line 22)
result_eq_133783 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 20), '==', byteorder_133781, str_133782)

# Assigning a type to the variable 'SYS_LITTLE_ENDIAN' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'SYS_LITTLE_ENDIAN', result_eq_133783)

# Assigning a Num to a Name (line 24):

# Assigning a Num to a Name (line 24):
int_133784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'int')
# Assigning a type to the variable 'miDOUBLE' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'miDOUBLE', int_133784)

# Assigning a Num to a Name (line 25):

# Assigning a Num to a Name (line 25):
int_133785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'int')
# Assigning a type to the variable 'miSINGLE' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'miSINGLE', int_133785)

# Assigning a Num to a Name (line 26):

# Assigning a Num to a Name (line 26):
int_133786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'int')
# Assigning a type to the variable 'miINT32' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'miINT32', int_133786)

# Assigning a Num to a Name (line 27):

# Assigning a Num to a Name (line 27):
int_133787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'int')
# Assigning a type to the variable 'miINT16' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'miINT16', int_133787)

# Assigning a Num to a Name (line 28):

# Assigning a Num to a Name (line 28):
int_133788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'int')
# Assigning a type to the variable 'miUINT16' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'miUINT16', int_133788)

# Assigning a Num to a Name (line 29):

# Assigning a Num to a Name (line 29):
int_133789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'int')
# Assigning a type to the variable 'miUINT8' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'miUINT8', int_133789)

# Assigning a Dict to a Name (line 31):

# Assigning a Dict to a Name (line 31):

# Obtaining an instance of the builtin type 'dict' (line 31)
dict_133790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 31)
# Adding element type (key, value) (line 31)
# Getting the type of 'miDOUBLE' (line 32)
miDOUBLE_133791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'miDOUBLE')
str_133792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'str', 'f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (miDOUBLE_133791, str_133792))
# Adding element type (key, value) (line 31)
# Getting the type of 'miSINGLE' (line 33)
miSINGLE_133793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'miSINGLE')
str_133794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 14), 'str', 'f4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (miSINGLE_133793, str_133794))
# Adding element type (key, value) (line 31)
# Getting the type of 'miINT32' (line 34)
miINT32_133795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'miINT32')
str_133796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 13), 'str', 'i4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (miINT32_133795, str_133796))
# Adding element type (key, value) (line 31)
# Getting the type of 'miINT16' (line 35)
miINT16_133797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'miINT16')
str_133798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'str', 'i2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (miINT16_133797, str_133798))
# Adding element type (key, value) (line 31)
# Getting the type of 'miUINT16' (line 36)
miUINT16_133799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'miUINT16')
str_133800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'str', 'u2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (miUINT16_133799, str_133800))
# Adding element type (key, value) (line 31)
# Getting the type of 'miUINT8' (line 37)
miUINT8_133801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'miUINT8')
str_133802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'str', 'u1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (miUINT8_133801, str_133802))
# Adding element type (key, value) (line 31)
str_133803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', 'header')

# Obtaining an instance of the builtin type 'list' (line 38)
list_133804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)

# Obtaining an instance of the builtin type 'tuple' (line 38)
tuple_133805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 38)
# Adding element type (line 38)
str_133806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'str', 'mopt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 16), tuple_133805, str_133806)
# Adding element type (line 38)
str_133807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'str', 'i4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 16), tuple_133805, str_133807)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), list_133804, tuple_133805)
# Adding element type (line 38)

# Obtaining an instance of the builtin type 'tuple' (line 39)
tuple_133808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 39)
# Adding element type (line 39)
str_133809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 16), 'str', 'mrows')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 16), tuple_133808, str_133809)
# Adding element type (line 39)
str_133810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'str', 'i4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 16), tuple_133808, str_133810)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), list_133804, tuple_133808)
# Adding element type (line 38)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_133811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
str_133812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'str', 'ncols')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 16), tuple_133811, str_133812)
# Adding element type (line 40)
str_133813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'str', 'i4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 16), tuple_133811, str_133813)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), list_133804, tuple_133811)
# Adding element type (line 38)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_133814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
str_133815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'str', 'imagf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), tuple_133814, str_133815)
# Adding element type (line 41)
str_133816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', 'i4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), tuple_133814, str_133816)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), list_133804, tuple_133814)
# Adding element type (line 38)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_133817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_133818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'str', 'namlen')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 16), tuple_133817, str_133818)
# Adding element type (line 42)
str_133819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'str', 'i4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 16), tuple_133817, str_133819)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), list_133804, tuple_133817)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (str_133803, list_133804))
# Adding element type (key, value) (line 31)
str_133820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'U1')
str_133821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 10), 'str', 'U1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_133790, (str_133820, str_133821))

# Assigning a type to the variable 'mdtypes_template' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'mdtypes_template', dict_133790)

# Assigning a Dict to a Name (line 46):

# Assigning a Dict to a Name (line 46):

# Obtaining an instance of the builtin type 'dict' (line 46)
dict_133822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 46)
# Adding element type (key, value) (line 46)
str_133823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'str', 'f8')
# Getting the type of 'miDOUBLE' (line 47)
miDOUBLE_133824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133823, miDOUBLE_133824))
# Adding element type (key, value) (line 46)
str_133825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'str', 'c32')
# Getting the type of 'miDOUBLE' (line 48)
miDOUBLE_133826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133825, miDOUBLE_133826))
# Adding element type (key, value) (line 46)
str_133827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'c24')
# Getting the type of 'miDOUBLE' (line 49)
miDOUBLE_133828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133827, miDOUBLE_133828))
# Adding element type (key, value) (line 46)
str_133829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 4), 'str', 'c16')
# Getting the type of 'miDOUBLE' (line 50)
miDOUBLE_133830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133829, miDOUBLE_133830))
# Adding element type (key, value) (line 46)
str_133831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'str', 'f4')
# Getting the type of 'miSINGLE' (line 51)
miSINGLE_133832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'miSINGLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133831, miSINGLE_133832))
# Adding element type (key, value) (line 46)
str_133833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'str', 'c8')
# Getting the type of 'miSINGLE' (line 52)
miSINGLE_133834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'miSINGLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133833, miSINGLE_133834))
# Adding element type (key, value) (line 46)
str_133835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'str', 'i4')
# Getting the type of 'miINT32' (line 53)
miINT32_133836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 10), 'miINT32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133835, miINT32_133836))
# Adding element type (key, value) (line 46)
str_133837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'str', 'i2')
# Getting the type of 'miINT16' (line 54)
miINT16_133838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 10), 'miINT16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133837, miINT16_133838))
# Adding element type (key, value) (line 46)
str_133839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'str', 'u2')
# Getting the type of 'miUINT16' (line 55)
miUINT16_133840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 10), 'miUINT16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133839, miUINT16_133840))
# Adding element type (key, value) (line 46)
str_133841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'str', 'u1')
# Getting the type of 'miUINT8' (line 56)
miUINT8_133842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 10), 'miUINT8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133841, miUINT8_133842))
# Adding element type (key, value) (line 46)
str_133843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'str', 'S1')
# Getting the type of 'miUINT8' (line 57)
miUINT8_133844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 10), 'miUINT8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), dict_133822, (str_133843, miUINT8_133844))

# Assigning a type to the variable 'np_to_mtypes' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'np_to_mtypes', dict_133822)

# Assigning a Num to a Name (line 61):

# Assigning a Num to a Name (line 61):
int_133845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
# Assigning a type to the variable 'mxFULL_CLASS' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'mxFULL_CLASS', int_133845)

# Assigning a Num to a Name (line 62):

# Assigning a Num to a Name (line 62):
int_133846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'int')
# Assigning a type to the variable 'mxCHAR_CLASS' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'mxCHAR_CLASS', int_133846)

# Assigning a Num to a Name (line 63):

# Assigning a Num to a Name (line 63):
int_133847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 17), 'int')
# Assigning a type to the variable 'mxSPARSE_CLASS' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'mxSPARSE_CLASS', int_133847)

# Assigning a Dict to a Name (line 65):

# Assigning a Dict to a Name (line 65):

# Obtaining an instance of the builtin type 'dict' (line 65)
dict_133848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 65)
# Adding element type (key, value) (line 65)
int_133849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')
str_133850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 7), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 14), dict_133848, (int_133849, str_133850))
# Adding element type (key, value) (line 65)
int_133851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'int')
str_133852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 7), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 14), dict_133848, (int_133851, str_133852))
# Adding element type (key, value) (line 65)
int_133853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'int')
str_133854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 7), 'str', 'VAX D-float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 14), dict_133848, (int_133853, str_133854))
# Adding element type (key, value) (line 65)
int_133855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'int')
str_133856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 7), 'str', 'VAX G-float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 14), dict_133848, (int_133855, str_133856))
# Adding element type (key, value) (line 65)
int_133857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'int')
str_133858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 7), 'str', 'Cray')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 14), dict_133848, (int_133857, str_133858))

# Assigning a type to the variable 'order_codes' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'order_codes', dict_133848)

# Assigning a Dict to a Name (line 73):

# Assigning a Dict to a Name (line 73):

# Obtaining an instance of the builtin type 'dict' (line 73)
dict_133859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 73)
# Adding element type (key, value) (line 73)
# Getting the type of 'mxFULL_CLASS' (line 74)
mxFULL_CLASS_133860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'mxFULL_CLASS')
str_133861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'str', 'double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 14), dict_133859, (mxFULL_CLASS_133860, str_133861))
# Adding element type (key, value) (line 73)
# Getting the type of 'mxCHAR_CLASS' (line 75)
mxCHAR_CLASS_133862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'mxCHAR_CLASS')
str_133863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'str', 'char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 14), dict_133859, (mxCHAR_CLASS_133862, str_133863))
# Adding element type (key, value) (line 73)
# Getting the type of 'mxSPARSE_CLASS' (line 76)
mxSPARSE_CLASS_133864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'mxSPARSE_CLASS')
str_133865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'str', 'sparse')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 14), dict_133859, (mxSPARSE_CLASS_133864, str_133865))

# Assigning a type to the variable 'mclass_info' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'mclass_info', dict_133859)
# Declaration of the 'VarHeader4' class

class VarHeader4(object, ):
    
    # Assigning a Name to a Name (line 82):
    
    # Assigning a Name to a Name (line 83):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarHeader4.__init__', ['name', 'dtype', 'mclass', 'dims', 'is_complex'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'dtype', 'mclass', 'dims', 'is_complex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 91):
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'name' (line 91)
        name_133866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'name')
        # Getting the type of 'self' (line 91)
        self_133867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'name' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_133867, 'name', name_133866)
        
        # Assigning a Name to a Attribute (line 92):
        
        # Assigning a Name to a Attribute (line 92):
        # Getting the type of 'dtype' (line 92)
        dtype_133868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 21), 'dtype')
        # Getting the type of 'self' (line 92)
        self_133869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_133869, 'dtype', dtype_133868)
        
        # Assigning a Name to a Attribute (line 93):
        
        # Assigning a Name to a Attribute (line 93):
        # Getting the type of 'mclass' (line 93)
        mclass_133870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'mclass')
        # Getting the type of 'self' (line 93)
        self_133871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member 'mclass' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_133871, 'mclass', mclass_133870)
        
        # Assigning a Name to a Attribute (line 94):
        
        # Assigning a Name to a Attribute (line 94):
        # Getting the type of 'dims' (line 94)
        dims_133872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'dims')
        # Getting the type of 'self' (line 94)
        self_133873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'dims' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_133873, 'dims', dims_133872)
        
        # Assigning a Name to a Attribute (line 95):
        
        # Assigning a Name to a Attribute (line 95):
        # Getting the type of 'is_complex' (line 95)
        is_complex_133874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'is_complex')
        # Getting the type of 'self' (line 95)
        self_133875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'is_complex' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_133875, 'is_complex', is_complex_133874)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'VarHeader4' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'VarHeader4', VarHeader4)

# Assigning a Name to a Name (line 82):
# Getting the type of 'False' (line 82)
False_133876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'False')
# Getting the type of 'VarHeader4'
VarHeader4_133877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VarHeader4')
# Setting the type of the member 'is_logical' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VarHeader4_133877, 'is_logical', False_133876)

# Assigning a Name to a Name (line 83):
# Getting the type of 'False' (line 83)
False_133878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'False')
# Getting the type of 'VarHeader4'
VarHeader4_133879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VarHeader4')
# Setting the type of the member 'is_global' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VarHeader4_133879, 'is_global', False_133878)
# Declaration of the 'VarReader4' class

class VarReader4(object, ):
    str_133880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'str', ' Class to read matlab 4 variables ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.__init__', ['file_reader'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_reader'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'file_reader' (line 102)
        file_reader_133881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'file_reader')
        # Getting the type of 'self' (line 102)
        self_133882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'file_reader' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_133882, 'file_reader', file_reader_133881)
        
        # Assigning a Attribute to a Attribute (line 103):
        
        # Assigning a Attribute to a Attribute (line 103):
        # Getting the type of 'file_reader' (line 103)
        file_reader_133883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'file_reader')
        # Obtaining the member 'mat_stream' of a type (line 103)
        mat_stream_133884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), file_reader_133883, 'mat_stream')
        # Getting the type of 'self' (line 103)
        self_133885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member 'mat_stream' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_133885, 'mat_stream', mat_stream_133884)
        
        # Assigning a Attribute to a Attribute (line 104):
        
        # Assigning a Attribute to a Attribute (line 104):
        # Getting the type of 'file_reader' (line 104)
        file_reader_133886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'file_reader')
        # Obtaining the member 'dtypes' of a type (line 104)
        dtypes_133887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 22), file_reader_133886, 'dtypes')
        # Getting the type of 'self' (line 104)
        self_133888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
        # Setting the type of the member 'dtypes' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_133888, 'dtypes', dtypes_133887)
        
        # Assigning a Attribute to a Attribute (line 105):
        
        # Assigning a Attribute to a Attribute (line 105):
        # Getting the type of 'file_reader' (line 105)
        file_reader_133889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 32), 'file_reader')
        # Obtaining the member 'chars_as_strings' of a type (line 105)
        chars_as_strings_133890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 32), file_reader_133889, 'chars_as_strings')
        # Getting the type of 'self' (line 105)
        self_133891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self')
        # Setting the type of the member 'chars_as_strings' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_133891, 'chars_as_strings', chars_as_strings_133890)
        
        # Assigning a Attribute to a Attribute (line 106):
        
        # Assigning a Attribute to a Attribute (line 106):
        # Getting the type of 'file_reader' (line 106)
        file_reader_133892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'file_reader')
        # Obtaining the member 'squeeze_me' of a type (line 106)
        squeeze_me_133893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 26), file_reader_133892, 'squeeze_me')
        # Getting the type of 'self' (line 106)
        self_133894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member 'squeeze_me' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_133894, 'squeeze_me', squeeze_me_133893)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def read_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_header'
        module_type_store = module_type_store.open_function_context('read_header', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarReader4.read_header.__dict__.__setitem__('stypy_localization', localization)
        VarReader4.read_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarReader4.read_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarReader4.read_header.__dict__.__setitem__('stypy_function_name', 'VarReader4.read_header')
        VarReader4.read_header.__dict__.__setitem__('stypy_param_names_list', [])
        VarReader4.read_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarReader4.read_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarReader4.read_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarReader4.read_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarReader4.read_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarReader4.read_header.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.read_header', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_header', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_header(...)' code ##################

        str_133895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'str', ' Read and return header for variable ')
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to read_dtype(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_133897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 110)
        mat_stream_133898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 26), self_133897, 'mat_stream')
        
        # Obtaining the type of the subscript
        str_133899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 55), 'str', 'header')
        # Getting the type of 'self' (line 110)
        self_133900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'self', False)
        # Obtaining the member 'dtypes' of a type (line 110)
        dtypes_133901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 43), self_133900, 'dtypes')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___133902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 43), dtypes_133901, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_133903 = invoke(stypy.reporting.localization.Localization(__file__, 110, 43), getitem___133902, str_133899)
        
        # Processing the call keyword arguments (line 110)
        kwargs_133904 = {}
        # Getting the type of 'read_dtype' (line 110)
        read_dtype_133896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'read_dtype', False)
        # Calling read_dtype(args, kwargs) (line 110)
        read_dtype_call_result_133905 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), read_dtype_133896, *[mat_stream_133898, subscript_call_result_133903], **kwargs_133904)
        
        # Assigning a type to the variable 'data' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'data', read_dtype_call_result_133905)
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to strip(...): (line 111)
        # Processing the call arguments (line 111)
        str_133919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 63), 'str', '\x00')
        # Processing the call keyword arguments (line 111)
        kwargs_133920 = {}
        
        # Call to read(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to int(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining the type of the subscript
        str_133910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 45), 'str', 'namlen')
        # Getting the type of 'data' (line 111)
        data_133911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___133912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 40), data_133911, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_133913 = invoke(stypy.reporting.localization.Localization(__file__, 111, 40), getitem___133912, str_133910)
        
        # Processing the call keyword arguments (line 111)
        kwargs_133914 = {}
        # Getting the type of 'int' (line 111)
        int_133909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'int', False)
        # Calling int(args, kwargs) (line 111)
        int_call_result_133915 = invoke(stypy.reporting.localization.Localization(__file__, 111, 36), int_133909, *[subscript_call_result_133913], **kwargs_133914)
        
        # Processing the call keyword arguments (line 111)
        kwargs_133916 = {}
        # Getting the type of 'self' (line 111)
        self_133906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 111)
        mat_stream_133907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), self_133906, 'mat_stream')
        # Obtaining the member 'read' of a type (line 111)
        read_133908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), mat_stream_133907, 'read')
        # Calling read(args, kwargs) (line 111)
        read_call_result_133917 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), read_133908, *[int_call_result_133915], **kwargs_133916)
        
        # Obtaining the member 'strip' of a type (line 111)
        strip_133918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), read_call_result_133917, 'strip')
        # Calling strip(args, kwargs) (line 111)
        strip_call_result_133921 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), strip_133918, *[str_133919], **kwargs_133920)
        
        # Assigning a type to the variable 'name' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'name', strip_call_result_133921)
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        str_133922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 16), 'str', 'mopt')
        # Getting the type of 'data' (line 112)
        data_133923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'data')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___133924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 11), data_133923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_133925 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), getitem___133924, str_133922)
        
        int_133926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 26), 'int')
        # Applying the binary operator '<' (line 112)
        result_lt_133927 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), '<', subscript_call_result_133925, int_133926)
        
        
        
        # Obtaining the type of the subscript
        str_133928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 36), 'str', 'mopt')
        # Getting the type of 'data' (line 112)
        data_133929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'data')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___133930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), data_133929, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_133931 = invoke(stypy.reporting.localization.Localization(__file__, 112, 31), getitem___133930, str_133928)
        
        int_133932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 46), 'int')
        # Applying the binary operator '>' (line 112)
        result_gt_133933 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 31), '>', subscript_call_result_133931, int_133932)
        
        # Applying the binary operator 'or' (line 112)
        result_or_keyword_133934 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'or', result_lt_133927, result_gt_133933)
        
        # Testing the type of an if condition (line 112)
        if_condition_133935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_or_keyword_133934)
        # Assigning a type to the variable 'if_condition_133935' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_133935', if_condition_133935)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 113)
        # Processing the call arguments (line 113)
        str_133937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'str', 'Mat 4 mopt wrong format, byteswapping problem?')
        # Processing the call keyword arguments (line 113)
        kwargs_133938 = {}
        # Getting the type of 'ValueError' (line 113)
        ValueError_133936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 113)
        ValueError_call_result_133939 = invoke(stypy.reporting.localization.Localization(__file__, 113, 18), ValueError_133936, *[str_133937], **kwargs_133938)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 113, 12), ValueError_call_result_133939, 'raise parameter', BaseException)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 114):
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_133940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 8), 'int')
        
        # Call to divmod(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining the type of the subscript
        str_133942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'str', 'mopt')
        # Getting the type of 'data' (line 114)
        data_133943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___133944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 25), data_133943, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_133945 = invoke(stypy.reporting.localization.Localization(__file__, 114, 25), getitem___133944, str_133942)
        
        int_133946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 39), 'int')
        # Processing the call keyword arguments (line 114)
        kwargs_133947 = {}
        # Getting the type of 'divmod' (line 114)
        divmod_133941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'divmod', False)
        # Calling divmod(args, kwargs) (line 114)
        divmod_call_result_133948 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), divmod_133941, *[subscript_call_result_133945, int_133946], **kwargs_133947)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___133949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), divmod_call_result_133948, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_133950 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), getitem___133949, int_133940)
        
        # Assigning a type to the variable 'tuple_var_assignment_133757' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_133757', subscript_call_result_133950)
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_133951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 8), 'int')
        
        # Call to divmod(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining the type of the subscript
        str_133953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'str', 'mopt')
        # Getting the type of 'data' (line 114)
        data_133954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___133955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 25), data_133954, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_133956 = invoke(stypy.reporting.localization.Localization(__file__, 114, 25), getitem___133955, str_133953)
        
        int_133957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 39), 'int')
        # Processing the call keyword arguments (line 114)
        kwargs_133958 = {}
        # Getting the type of 'divmod' (line 114)
        divmod_133952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'divmod', False)
        # Calling divmod(args, kwargs) (line 114)
        divmod_call_result_133959 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), divmod_133952, *[subscript_call_result_133956, int_133957], **kwargs_133958)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___133960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), divmod_call_result_133959, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_133961 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), getitem___133960, int_133951)
        
        # Assigning a type to the variable 'tuple_var_assignment_133758' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_133758', subscript_call_result_133961)
        
        # Assigning a Name to a Name (line 114):
        # Getting the type of 'tuple_var_assignment_133757' (line 114)
        tuple_var_assignment_133757_133962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_133757')
        # Assigning a type to the variable 'M' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'M', tuple_var_assignment_133757_133962)
        
        # Assigning a Name to a Name (line 114):
        # Getting the type of 'tuple_var_assignment_133758' (line 114)
        tuple_var_assignment_133758_133963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_133758')
        # Assigning a type to the variable 'rest' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'rest', tuple_var_assignment_133758_133963)
        
        
        # Getting the type of 'M' (line 115)
        M_133964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'M')
        
        # Obtaining an instance of the builtin type 'tuple' (line 115)
        tuple_133965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 115)
        # Adding element type (line 115)
        int_133966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), tuple_133965, int_133966)
        # Adding element type (line 115)
        int_133967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 21), tuple_133965, int_133967)
        
        # Applying the binary operator 'notin' (line 115)
        result_contains_133968 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 11), 'notin', M_133964, tuple_133965)
        
        # Testing the type of an if condition (line 115)
        if_condition_133969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), result_contains_133968)
        # Assigning a type to the variable 'if_condition_133969' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_133969', if_condition_133969)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 116)
        # Processing the call arguments (line 116)
        str_133972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'str', "We do not support byte ordering '%s'; returned data may be corrupt")
        
        # Obtaining the type of the subscript
        # Getting the type of 'M' (line 117)
        M_133973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 62), 'M', False)
        # Getting the type of 'order_codes' (line 117)
        order_codes_133974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 50), 'order_codes', False)
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___133975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 50), order_codes_133974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_133976 = invoke(stypy.reporting.localization.Localization(__file__, 117, 50), getitem___133975, M_133973)
        
        # Applying the binary operator '%' (line 116)
        result_mod_133977 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 26), '%', str_133972, subscript_call_result_133976)
        
        # Getting the type of 'UserWarning' (line 118)
        UserWarning_133978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'UserWarning', False)
        # Processing the call keyword arguments (line 116)
        kwargs_133979 = {}
        # Getting the type of 'warnings' (line 116)
        warnings_133970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 116)
        warn_133971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), warnings_133970, 'warn')
        # Calling warn(args, kwargs) (line 116)
        warn_call_result_133980 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), warn_133971, *[result_mod_133977, UserWarning_133978], **kwargs_133979)
        
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 119):
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_133981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to divmod(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'rest' (line 119)
        rest_133983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'rest', False)
        int_133984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 31), 'int')
        # Processing the call keyword arguments (line 119)
        kwargs_133985 = {}
        # Getting the type of 'divmod' (line 119)
        divmod_133982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'divmod', False)
        # Calling divmod(args, kwargs) (line 119)
        divmod_call_result_133986 = invoke(stypy.reporting.localization.Localization(__file__, 119, 18), divmod_133982, *[rest_133983, int_133984], **kwargs_133985)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___133987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), divmod_call_result_133986, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_133988 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___133987, int_133981)
        
        # Assigning a type to the variable 'tuple_var_assignment_133759' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_133759', subscript_call_result_133988)
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_133989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to divmod(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'rest' (line 119)
        rest_133991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'rest', False)
        int_133992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 31), 'int')
        # Processing the call keyword arguments (line 119)
        kwargs_133993 = {}
        # Getting the type of 'divmod' (line 119)
        divmod_133990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'divmod', False)
        # Calling divmod(args, kwargs) (line 119)
        divmod_call_result_133994 = invoke(stypy.reporting.localization.Localization(__file__, 119, 18), divmod_133990, *[rest_133991, int_133992], **kwargs_133993)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___133995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), divmod_call_result_133994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_133996 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___133995, int_133989)
        
        # Assigning a type to the variable 'tuple_var_assignment_133760' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_133760', subscript_call_result_133996)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_133759' (line 119)
        tuple_var_assignment_133759_133997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_133759')
        # Assigning a type to the variable 'O' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'O', tuple_var_assignment_133759_133997)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_133760' (line 119)
        tuple_var_assignment_133760_133998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_133760')
        # Assigning a type to the variable 'rest' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'rest', tuple_var_assignment_133760_133998)
        
        
        # Getting the type of 'O' (line 120)
        O_133999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'O')
        int_134000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'int')
        # Applying the binary operator '!=' (line 120)
        result_ne_134001 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), '!=', O_133999, int_134000)
        
        # Testing the type of an if condition (line 120)
        if_condition_134002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), result_ne_134001)
        # Assigning a type to the variable 'if_condition_134002' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_134002', if_condition_134002)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 121)
        # Processing the call arguments (line 121)
        str_134004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'str', 'O in MOPT integer should be 0, wrong format?')
        # Processing the call keyword arguments (line 121)
        kwargs_134005 = {}
        # Getting the type of 'ValueError' (line 121)
        ValueError_134003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 121)
        ValueError_call_result_134006 = invoke(stypy.reporting.localization.Localization(__file__, 121, 18), ValueError_134003, *[str_134004], **kwargs_134005)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 121, 12), ValueError_call_result_134006, 'raise parameter', BaseException)
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 122):
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_134007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 8), 'int')
        
        # Call to divmod(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'rest' (line 122)
        rest_134009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'rest', False)
        int_134010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 31), 'int')
        # Processing the call keyword arguments (line 122)
        kwargs_134011 = {}
        # Getting the type of 'divmod' (line 122)
        divmod_134008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'divmod', False)
        # Calling divmod(args, kwargs) (line 122)
        divmod_call_result_134012 = invoke(stypy.reporting.localization.Localization(__file__, 122, 18), divmod_134008, *[rest_134009, int_134010], **kwargs_134011)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___134013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), divmod_call_result_134012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_134014 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), getitem___134013, int_134007)
        
        # Assigning a type to the variable 'tuple_var_assignment_133761' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_133761', subscript_call_result_134014)
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_134015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 8), 'int')
        
        # Call to divmod(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'rest' (line 122)
        rest_134017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'rest', False)
        int_134018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 31), 'int')
        # Processing the call keyword arguments (line 122)
        kwargs_134019 = {}
        # Getting the type of 'divmod' (line 122)
        divmod_134016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'divmod', False)
        # Calling divmod(args, kwargs) (line 122)
        divmod_call_result_134020 = invoke(stypy.reporting.localization.Localization(__file__, 122, 18), divmod_134016, *[rest_134017, int_134018], **kwargs_134019)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___134021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), divmod_call_result_134020, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_134022 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), getitem___134021, int_134015)
        
        # Assigning a type to the variable 'tuple_var_assignment_133762' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_133762', subscript_call_result_134022)
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'tuple_var_assignment_133761' (line 122)
        tuple_var_assignment_133761_134023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_133761')
        # Assigning a type to the variable 'P' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'P', tuple_var_assignment_133761_134023)
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'tuple_var_assignment_133762' (line 122)
        tuple_var_assignment_133762_134024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_133762')
        # Assigning a type to the variable 'rest' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'rest', tuple_var_assignment_133762_134024)
        
        # Assigning a Name to a Name (line 123):
        
        # Assigning a Name to a Name (line 123):
        # Getting the type of 'rest' (line 123)
        rest_134025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'rest')
        # Assigning a type to the variable 'T' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'T', rest_134025)
        
        # Assigning a Tuple to a Name (line 124):
        
        # Assigning a Tuple to a Name (line 124):
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_134026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        
        # Obtaining the type of the subscript
        str_134027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 21), 'str', 'mrows')
        # Getting the type of 'data' (line 124)
        data_134028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'data')
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___134029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), data_134028, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_134030 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), getitem___134029, str_134027)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 16), tuple_134026, subscript_call_result_134030)
        # Adding element type (line 124)
        
        # Obtaining the type of the subscript
        str_134031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 36), 'str', 'ncols')
        # Getting the type of 'data' (line 124)
        data_134032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'data')
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___134033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 31), data_134032, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_134034 = invoke(stypy.reporting.localization.Localization(__file__, 124, 31), getitem___134033, str_134031)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 16), tuple_134026, subscript_call_result_134034)
        
        # Assigning a type to the variable 'dims' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'dims', tuple_134026)
        
        # Assigning a Compare to a Name (line 125):
        
        # Assigning a Compare to a Name (line 125):
        
        
        # Obtaining the type of the subscript
        str_134035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 26), 'str', 'imagf')
        # Getting the type of 'data' (line 125)
        data_134036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'data')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___134037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 21), data_134036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_134038 = invoke(stypy.reporting.localization.Localization(__file__, 125, 21), getitem___134037, str_134035)
        
        int_134039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 38), 'int')
        # Applying the binary operator '==' (line 125)
        result_eq_134040 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 21), '==', subscript_call_result_134038, int_134039)
        
        # Assigning a type to the variable 'is_complex' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'is_complex', result_eq_134040)
        
        # Assigning a Subscript to a Name (line 126):
        
        # Assigning a Subscript to a Name (line 126):
        
        # Obtaining the type of the subscript
        # Getting the type of 'P' (line 126)
        P_134041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'P')
        # Getting the type of 'self' (line 126)
        self_134042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'self')
        # Obtaining the member 'dtypes' of a type (line 126)
        dtypes_134043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 16), self_134042, 'dtypes')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___134044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 16), dtypes_134043, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_134045 = invoke(stypy.reporting.localization.Localization(__file__, 126, 16), getitem___134044, P_134041)
        
        # Assigning a type to the variable 'dtype' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'dtype', subscript_call_result_134045)
        
        # Call to VarHeader4(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'name' (line 128)
        name_134047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'name', False)
        # Getting the type of 'dtype' (line 129)
        dtype_134048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'dtype', False)
        # Getting the type of 'T' (line 130)
        T_134049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'T', False)
        # Getting the type of 'dims' (line 131)
        dims_134050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'dims', False)
        # Getting the type of 'is_complex' (line 132)
        is_complex_134051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'is_complex', False)
        # Processing the call keyword arguments (line 127)
        kwargs_134052 = {}
        # Getting the type of 'VarHeader4' (line 127)
        VarHeader4_134046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'VarHeader4', False)
        # Calling VarHeader4(args, kwargs) (line 127)
        VarHeader4_call_result_134053 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), VarHeader4_134046, *[name_134047, dtype_134048, T_134049, dims_134050, is_complex_134051], **kwargs_134052)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', VarHeader4_call_result_134053)
        
        # ################# End of 'read_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_header' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_134054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_header'
        return stypy_return_type_134054


    @norecursion
    def array_from_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 134)
        True_134055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'True')
        defaults = [True_134055]
        # Create a new context for function 'array_from_header'
        module_type_store = module_type_store.open_function_context('array_from_header', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarReader4.array_from_header.__dict__.__setitem__('stypy_localization', localization)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_function_name', 'VarReader4.array_from_header')
        VarReader4.array_from_header.__dict__.__setitem__('stypy_param_names_list', ['hdr', 'process'])
        VarReader4.array_from_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarReader4.array_from_header.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.array_from_header', ['hdr', 'process'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'array_from_header', localization, ['hdr', 'process'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'array_from_header(...)' code ##################

        
        # Assigning a Attribute to a Name (line 135):
        
        # Assigning a Attribute to a Name (line 135):
        # Getting the type of 'hdr' (line 135)
        hdr_134056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'hdr')
        # Obtaining the member 'mclass' of a type (line 135)
        mclass_134057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), hdr_134056, 'mclass')
        # Assigning a type to the variable 'mclass' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'mclass', mclass_134057)
        
        
        # Getting the type of 'mclass' (line 136)
        mclass_134058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'mclass')
        # Getting the type of 'mxFULL_CLASS' (line 136)
        mxFULL_CLASS_134059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'mxFULL_CLASS')
        # Applying the binary operator '==' (line 136)
        result_eq_134060 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), '==', mclass_134058, mxFULL_CLASS_134059)
        
        # Testing the type of an if condition (line 136)
        if_condition_134061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 8), result_eq_134060)
        # Assigning a type to the variable 'if_condition_134061' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'if_condition_134061', if_condition_134061)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to read_full_array(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'hdr' (line 137)
        hdr_134064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'hdr', False)
        # Processing the call keyword arguments (line 137)
        kwargs_134065 = {}
        # Getting the type of 'self' (line 137)
        self_134062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 18), 'self', False)
        # Obtaining the member 'read_full_array' of a type (line 137)
        read_full_array_134063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 18), self_134062, 'read_full_array')
        # Calling read_full_array(args, kwargs) (line 137)
        read_full_array_call_result_134066 = invoke(stypy.reporting.localization.Localization(__file__, 137, 18), read_full_array_134063, *[hdr_134064], **kwargs_134065)
        
        # Assigning a type to the variable 'arr' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'arr', read_full_array_call_result_134066)
        # SSA branch for the else part of an if statement (line 136)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mclass' (line 138)
        mclass_134067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'mclass')
        # Getting the type of 'mxCHAR_CLASS' (line 138)
        mxCHAR_CLASS_134068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 23), 'mxCHAR_CLASS')
        # Applying the binary operator '==' (line 138)
        result_eq_134069 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 13), '==', mclass_134067, mxCHAR_CLASS_134068)
        
        # Testing the type of an if condition (line 138)
        if_condition_134070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 13), result_eq_134069)
        # Assigning a type to the variable 'if_condition_134070' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'if_condition_134070', if_condition_134070)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to read_char_array(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'hdr' (line 139)
        hdr_134073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 39), 'hdr', False)
        # Processing the call keyword arguments (line 139)
        kwargs_134074 = {}
        # Getting the type of 'self' (line 139)
        self_134071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'self', False)
        # Obtaining the member 'read_char_array' of a type (line 139)
        read_char_array_134072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 18), self_134071, 'read_char_array')
        # Calling read_char_array(args, kwargs) (line 139)
        read_char_array_call_result_134075 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), read_char_array_134072, *[hdr_134073], **kwargs_134074)
        
        # Assigning a type to the variable 'arr' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'arr', read_char_array_call_result_134075)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'process' (line 140)
        process_134076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'process')
        # Getting the type of 'self' (line 140)
        self_134077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'self')
        # Obtaining the member 'chars_as_strings' of a type (line 140)
        chars_as_strings_134078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 27), self_134077, 'chars_as_strings')
        # Applying the binary operator 'and' (line 140)
        result_and_keyword_134079 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 15), 'and', process_134076, chars_as_strings_134078)
        
        # Testing the type of an if condition (line 140)
        if_condition_134080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 12), result_and_keyword_134079)
        # Assigning a type to the variable 'if_condition_134080' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'if_condition_134080', if_condition_134080)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to chars_to_strings(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'arr' (line 141)
        arr_134082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 39), 'arr', False)
        # Processing the call keyword arguments (line 141)
        kwargs_134083 = {}
        # Getting the type of 'chars_to_strings' (line 141)
        chars_to_strings_134081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'chars_to_strings', False)
        # Calling chars_to_strings(args, kwargs) (line 141)
        chars_to_strings_call_result_134084 = invoke(stypy.reporting.localization.Localization(__file__, 141, 22), chars_to_strings_134081, *[arr_134082], **kwargs_134083)
        
        # Assigning a type to the variable 'arr' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'arr', chars_to_strings_call_result_134084)
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 138)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mclass' (line 142)
        mclass_134085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'mclass')
        # Getting the type of 'mxSPARSE_CLASS' (line 142)
        mxSPARSE_CLASS_134086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 23), 'mxSPARSE_CLASS')
        # Applying the binary operator '==' (line 142)
        result_eq_134087 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 13), '==', mclass_134085, mxSPARSE_CLASS_134086)
        
        # Testing the type of an if condition (line 142)
        if_condition_134088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 13), result_eq_134087)
        # Assigning a type to the variable 'if_condition_134088' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'if_condition_134088', if_condition_134088)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to read_sparse_array(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'hdr' (line 144)
        hdr_134091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 42), 'hdr', False)
        # Processing the call keyword arguments (line 144)
        kwargs_134092 = {}
        # Getting the type of 'self' (line 144)
        self_134089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'self', False)
        # Obtaining the member 'read_sparse_array' of a type (line 144)
        read_sparse_array_134090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), self_134089, 'read_sparse_array')
        # Calling read_sparse_array(args, kwargs) (line 144)
        read_sparse_array_call_result_134093 = invoke(stypy.reporting.localization.Localization(__file__, 144, 19), read_sparse_array_134090, *[hdr_134091], **kwargs_134092)
        
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'stypy_return_type', read_sparse_array_call_result_134093)
        # SSA branch for the else part of an if statement (line 142)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 146)
        # Processing the call arguments (line 146)
        str_134095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'str', 'No reader for class code %s')
        # Getting the type of 'mclass' (line 146)
        mclass_134096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 60), 'mclass', False)
        # Applying the binary operator '%' (line 146)
        result_mod_134097 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 28), '%', str_134095, mclass_134096)
        
        # Processing the call keyword arguments (line 146)
        kwargs_134098 = {}
        # Getting the type of 'TypeError' (line 146)
        TypeError_134094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 146)
        TypeError_call_result_134099 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), TypeError_134094, *[result_mod_134097], **kwargs_134098)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 12), TypeError_call_result_134099, 'raise parameter', BaseException)
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'process' (line 147)
        process_134100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'process')
        # Getting the type of 'self' (line 147)
        self_134101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'self')
        # Obtaining the member 'squeeze_me' of a type (line 147)
        squeeze_me_134102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 23), self_134101, 'squeeze_me')
        # Applying the binary operator 'and' (line 147)
        result_and_keyword_134103 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 11), 'and', process_134100, squeeze_me_134102)
        
        # Testing the type of an if condition (line 147)
        if_condition_134104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), result_and_keyword_134103)
        # Assigning a type to the variable 'if_condition_134104' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_134104', if_condition_134104)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to squeeze_element(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'arr' (line 148)
        arr_134106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'arr', False)
        # Processing the call keyword arguments (line 148)
        kwargs_134107 = {}
        # Getting the type of 'squeeze_element' (line 148)
        squeeze_element_134105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'squeeze_element', False)
        # Calling squeeze_element(args, kwargs) (line 148)
        squeeze_element_call_result_134108 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), squeeze_element_134105, *[arr_134106], **kwargs_134107)
        
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'stypy_return_type', squeeze_element_call_result_134108)
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'arr' (line 149)
        arr_134109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'arr')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', arr_134109)
        
        # ################# End of 'array_from_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'array_from_header' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_134110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'array_from_header'
        return stypy_return_type_134110


    @norecursion
    def read_sub_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 151)
        True_134111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'True')
        defaults = [True_134111]
        # Create a new context for function 'read_sub_array'
        module_type_store = module_type_store.open_function_context('read_sub_array', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_localization', localization)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_function_name', 'VarReader4.read_sub_array')
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_param_names_list', ['hdr', 'copy'])
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarReader4.read_sub_array.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.read_sub_array', ['hdr', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_sub_array', localization, ['hdr', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_sub_array(...)' code ##################

        str_134112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, (-1)), 'str', ' Mat4 read using header `hdr` dtype and dims\n\n        Parameters\n        ----------\n        hdr : object\n           object with attributes ``dtype``, ``dims``.  dtype is assumed to be\n           the correct endianness\n        copy : bool, optional\n           copies array before return if True (default True)\n           (buffer is usually read only)\n\n        Returns\n        -------\n        arr : ndarray\n            of dtype givem by `hdr` ``dtype`` and shape givem by `hdr` ``dims``\n        ')
        
        # Assigning a Attribute to a Name (line 168):
        
        # Assigning a Attribute to a Name (line 168):
        # Getting the type of 'hdr' (line 168)
        hdr_134113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'hdr')
        # Obtaining the member 'dtype' of a type (line 168)
        dtype_134114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 13), hdr_134113, 'dtype')
        # Assigning a type to the variable 'dt' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'dt', dtype_134114)
        
        # Assigning a Attribute to a Name (line 169):
        
        # Assigning a Attribute to a Name (line 169):
        # Getting the type of 'hdr' (line 169)
        hdr_134115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'hdr')
        # Obtaining the member 'dims' of a type (line 169)
        dims_134116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 15), hdr_134115, 'dims')
        # Assigning a type to the variable 'dims' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'dims', dims_134116)
        
        # Assigning a Attribute to a Name (line 170):
        
        # Assigning a Attribute to a Name (line 170):
        # Getting the type of 'dt' (line 170)
        dt_134117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'dt')
        # Obtaining the member 'itemsize' of a type (line 170)
        itemsize_134118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 20), dt_134117, 'itemsize')
        # Assigning a type to the variable 'num_bytes' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'num_bytes', itemsize_134118)
        
        # Getting the type of 'dims' (line 171)
        dims_134119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'dims')
        # Testing the type of a for loop iterable (line 171)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 8), dims_134119)
        # Getting the type of the for loop variable (line 171)
        for_loop_var_134120 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 8), dims_134119)
        # Assigning a type to the variable 'd' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'd', for_loop_var_134120)
        # SSA begins for a for statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'num_bytes' (line 172)
        num_bytes_134121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'num_bytes')
        # Getting the type of 'd' (line 172)
        d_134122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'd')
        # Applying the binary operator '*=' (line 172)
        result_imul_134123 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 12), '*=', num_bytes_134121, d_134122)
        # Assigning a type to the variable 'num_bytes' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'num_bytes', result_imul_134123)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to read(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Call to int(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'num_bytes' (line 173)
        num_bytes_134128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 42), 'num_bytes', False)
        # Processing the call keyword arguments (line 173)
        kwargs_134129 = {}
        # Getting the type of 'int' (line 173)
        int_134127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'int', False)
        # Calling int(args, kwargs) (line 173)
        int_call_result_134130 = invoke(stypy.reporting.localization.Localization(__file__, 173, 38), int_134127, *[num_bytes_134128], **kwargs_134129)
        
        # Processing the call keyword arguments (line 173)
        kwargs_134131 = {}
        # Getting the type of 'self' (line 173)
        self_134124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 173)
        mat_stream_134125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 17), self_134124, 'mat_stream')
        # Obtaining the member 'read' of a type (line 173)
        read_134126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 17), mat_stream_134125, 'read')
        # Calling read(args, kwargs) (line 173)
        read_call_result_134132 = invoke(stypy.reporting.localization.Localization(__file__, 173, 17), read_134126, *[int_call_result_134130], **kwargs_134131)
        
        # Assigning a type to the variable 'buffer' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'buffer', read_call_result_134132)
        
        
        
        # Call to len(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'buffer' (line 174)
        buffer_134134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'buffer', False)
        # Processing the call keyword arguments (line 174)
        kwargs_134135 = {}
        # Getting the type of 'len' (line 174)
        len_134133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'len', False)
        # Calling len(args, kwargs) (line 174)
        len_call_result_134136 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), len_134133, *[buffer_134134], **kwargs_134135)
        
        # Getting the type of 'num_bytes' (line 174)
        num_bytes_134137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'num_bytes')
        # Applying the binary operator '!=' (line 174)
        result_ne_134138 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 11), '!=', len_call_result_134136, num_bytes_134137)
        
        # Testing the type of an if condition (line 174)
        if_condition_134139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 8), result_ne_134138)
        # Assigning a type to the variable 'if_condition_134139' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'if_condition_134139', if_condition_134139)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 175)
        # Processing the call arguments (line 175)
        str_134141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'str', "Not enough bytes to read matrix '%s'; is this a badly-formed file? Consider listing matrices with `whosmat` and loading named matrices with `variable_names` kwarg to `loadmat`")
        # Getting the type of 'hdr' (line 178)
        hdr_134142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 69), 'hdr', False)
        # Obtaining the member 'name' of a type (line 178)
        name_134143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 69), hdr_134142, 'name')
        # Applying the binary operator '%' (line 175)
        result_mod_134144 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 29), '%', str_134141, name_134143)
        
        # Processing the call keyword arguments (line 175)
        kwargs_134145 = {}
        # Getting the type of 'ValueError' (line 175)
        ValueError_134140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 175)
        ValueError_call_result_134146 = invoke(stypy.reporting.localization.Localization(__file__, 175, 18), ValueError_134140, *[result_mod_134144], **kwargs_134145)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 12), ValueError_call_result_134146, 'raise parameter', BaseException)
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to ndarray(...): (line 179)
        # Processing the call keyword arguments (line 179)
        # Getting the type of 'dims' (line 179)
        dims_134149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'dims', False)
        keyword_134150 = dims_134149
        # Getting the type of 'dt' (line 180)
        dt_134151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'dt', False)
        keyword_134152 = dt_134151
        # Getting the type of 'buffer' (line 181)
        buffer_134153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 32), 'buffer', False)
        keyword_134154 = buffer_134153
        str_134155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 31), 'str', 'F')
        keyword_134156 = str_134155
        kwargs_134157 = {'buffer': keyword_134154, 'dtype': keyword_134152, 'shape': keyword_134150, 'order': keyword_134156}
        # Getting the type of 'np' (line 179)
        np_134147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 179)
        ndarray_134148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 14), np_134147, 'ndarray')
        # Calling ndarray(args, kwargs) (line 179)
        ndarray_call_result_134158 = invoke(stypy.reporting.localization.Localization(__file__, 179, 14), ndarray_134148, *[], **kwargs_134157)
        
        # Assigning a type to the variable 'arr' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'arr', ndarray_call_result_134158)
        
        # Getting the type of 'copy' (line 183)
        copy_134159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'copy')
        # Testing the type of an if condition (line 183)
        if_condition_134160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), copy_134159)
        # Assigning a type to the variable 'if_condition_134160' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'if_condition_134160', if_condition_134160)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 184):
        
        # Assigning a Call to a Name (line 184):
        
        # Call to copy(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_134163 = {}
        # Getting the type of 'arr' (line 184)
        arr_134161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'arr', False)
        # Obtaining the member 'copy' of a type (line 184)
        copy_134162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), arr_134161, 'copy')
        # Calling copy(args, kwargs) (line 184)
        copy_call_result_134164 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), copy_134162, *[], **kwargs_134163)
        
        # Assigning a type to the variable 'arr' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'arr', copy_call_result_134164)
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'arr' (line 185)
        arr_134165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'arr')
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', arr_134165)
        
        # ################# End of 'read_sub_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_sub_array' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_134166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134166)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_sub_array'
        return stypy_return_type_134166


    @norecursion
    def read_full_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_full_array'
        module_type_store = module_type_store.open_function_context('read_full_array', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarReader4.read_full_array.__dict__.__setitem__('stypy_localization', localization)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_function_name', 'VarReader4.read_full_array')
        VarReader4.read_full_array.__dict__.__setitem__('stypy_param_names_list', ['hdr'])
        VarReader4.read_full_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarReader4.read_full_array.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.read_full_array', ['hdr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_full_array', localization, ['hdr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_full_array(...)' code ##################

        str_134167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', ' Full (rather than sparse) matrix getter\n\n        Read matrix (array) can be real or complex\n\n        Parameters\n        ----------\n        hdr : ``VarHeader4`` instance\n\n        Returns\n        -------\n        arr : ndarray\n            complex array if ``hdr.is_complex`` is True, otherwise a real\n            numeric array\n        ')
        
        # Getting the type of 'hdr' (line 202)
        hdr_134168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'hdr')
        # Obtaining the member 'is_complex' of a type (line 202)
        is_complex_134169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 11), hdr_134168, 'is_complex')
        # Testing the type of an if condition (line 202)
        if_condition_134170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), is_complex_134169)
        # Assigning a type to the variable 'if_condition_134170' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_134170', if_condition_134170)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to read_sub_array(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'hdr' (line 204)
        hdr_134173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), 'hdr', False)
        # Processing the call keyword arguments (line 204)
        # Getting the type of 'False' (line 204)
        False_134174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 48), 'False', False)
        keyword_134175 = False_134174
        kwargs_134176 = {'copy': keyword_134175}
        # Getting the type of 'self' (line 204)
        self_134171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'self', False)
        # Obtaining the member 'read_sub_array' of a type (line 204)
        read_sub_array_134172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), self_134171, 'read_sub_array')
        # Calling read_sub_array(args, kwargs) (line 204)
        read_sub_array_call_result_134177 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), read_sub_array_134172, *[hdr_134173], **kwargs_134176)
        
        # Assigning a type to the variable 'res' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'res', read_sub_array_call_result_134177)
        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to read_sub_array(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'hdr' (line 205)
        hdr_134180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 40), 'hdr', False)
        # Processing the call keyword arguments (line 205)
        # Getting the type of 'False' (line 205)
        False_134181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 50), 'False', False)
        keyword_134182 = False_134181
        kwargs_134183 = {'copy': keyword_134182}
        # Getting the type of 'self' (line 205)
        self_134178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'self', False)
        # Obtaining the member 'read_sub_array' of a type (line 205)
        read_sub_array_134179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 20), self_134178, 'read_sub_array')
        # Calling read_sub_array(args, kwargs) (line 205)
        read_sub_array_call_result_134184 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), read_sub_array_134179, *[hdr_134180], **kwargs_134183)
        
        # Assigning a type to the variable 'res_j' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'res_j', read_sub_array_call_result_134184)
        # Getting the type of 'res' (line 206)
        res_134185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'res')
        # Getting the type of 'res_j' (line 206)
        res_j_134186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'res_j')
        complex_134187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 34), 'complex')
        # Applying the binary operator '*' (line 206)
        result_mul_134188 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 26), '*', res_j_134186, complex_134187)
        
        # Applying the binary operator '+' (line 206)
        result_add_134189 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 19), '+', res_134185, result_mul_134188)
        
        # Assigning a type to the variable 'stypy_return_type' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type', result_add_134189)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to read_sub_array(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'hdr' (line 207)
        hdr_134192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'hdr', False)
        # Processing the call keyword arguments (line 207)
        kwargs_134193 = {}
        # Getting the type of 'self' (line 207)
        self_134190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'self', False)
        # Obtaining the member 'read_sub_array' of a type (line 207)
        read_sub_array_134191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 15), self_134190, 'read_sub_array')
        # Calling read_sub_array(args, kwargs) (line 207)
        read_sub_array_call_result_134194 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), read_sub_array_134191, *[hdr_134192], **kwargs_134193)
        
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', read_sub_array_call_result_134194)
        
        # ################# End of 'read_full_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_full_array' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_134195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134195)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_full_array'
        return stypy_return_type_134195


    @norecursion
    def read_char_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_char_array'
        module_type_store = module_type_store.open_function_context('read_char_array', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarReader4.read_char_array.__dict__.__setitem__('stypy_localization', localization)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_function_name', 'VarReader4.read_char_array')
        VarReader4.read_char_array.__dict__.__setitem__('stypy_param_names_list', ['hdr'])
        VarReader4.read_char_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarReader4.read_char_array.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.read_char_array', ['hdr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_char_array', localization, ['hdr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_char_array(...)' code ##################

        str_134196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'str', " latin-1 text matrix (char matrix) reader\n\n        Parameters\n        ----------\n        hdr : ``VarHeader4`` instance\n\n        Returns\n        -------\n        arr : ndarray\n            with dtype 'U1', shape given by `hdr` ``dims``\n        ")
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to astype(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'np' (line 221)
        np_134203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 46), 'np', False)
        # Obtaining the member 'uint8' of a type (line 221)
        uint8_134204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 46), np_134203, 'uint8')
        # Processing the call keyword arguments (line 221)
        kwargs_134205 = {}
        
        # Call to read_sub_array(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'hdr' (line 221)
        hdr_134199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 34), 'hdr', False)
        # Processing the call keyword arguments (line 221)
        kwargs_134200 = {}
        # Getting the type of 'self' (line 221)
        self_134197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'self', False)
        # Obtaining the member 'read_sub_array' of a type (line 221)
        read_sub_array_134198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 14), self_134197, 'read_sub_array')
        # Calling read_sub_array(args, kwargs) (line 221)
        read_sub_array_call_result_134201 = invoke(stypy.reporting.localization.Localization(__file__, 221, 14), read_sub_array_134198, *[hdr_134199], **kwargs_134200)
        
        # Obtaining the member 'astype' of a type (line 221)
        astype_134202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 14), read_sub_array_call_result_134201, 'astype')
        # Calling astype(args, kwargs) (line 221)
        astype_call_result_134206 = invoke(stypy.reporting.localization.Localization(__file__, 221, 14), astype_134202, *[uint8_134204], **kwargs_134205)
        
        # Assigning a type to the variable 'arr' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'arr', astype_call_result_134206)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to decode(...): (line 222)
        # Processing the call arguments (line 222)
        str_134212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 34), 'str', 'latin-1')
        # Processing the call keyword arguments (line 222)
        kwargs_134213 = {}
        
        # Call to tostring(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_134209 = {}
        # Getting the type of 'arr' (line 222)
        arr_134207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'arr', False)
        # Obtaining the member 'tostring' of a type (line 222)
        tostring_134208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), arr_134207, 'tostring')
        # Calling tostring(args, kwargs) (line 222)
        tostring_call_result_134210 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), tostring_134208, *[], **kwargs_134209)
        
        # Obtaining the member 'decode' of a type (line 222)
        decode_134211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), tostring_call_result_134210, 'decode')
        # Calling decode(args, kwargs) (line 222)
        decode_call_result_134214 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), decode_134211, *[str_134212], **kwargs_134213)
        
        # Assigning a type to the variable 'S' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'S', decode_call_result_134214)
        
        # Call to copy(...): (line 223)
        # Processing the call keyword arguments (line 223)
        kwargs_134235 = {}
        
        # Call to ndarray(...): (line 223)
        # Processing the call keyword arguments (line 223)
        # Getting the type of 'hdr' (line 223)
        hdr_134217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'hdr', False)
        # Obtaining the member 'dims' of a type (line 223)
        dims_134218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 32), hdr_134217, 'dims')
        keyword_134219 = dims_134218
        
        # Call to dtype(...): (line 224)
        # Processing the call arguments (line 224)
        str_134222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 41), 'str', 'U1')
        # Processing the call keyword arguments (line 224)
        kwargs_134223 = {}
        # Getting the type of 'np' (line 224)
        np_134220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 32), 'np', False)
        # Obtaining the member 'dtype' of a type (line 224)
        dtype_134221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 32), np_134220, 'dtype')
        # Calling dtype(args, kwargs) (line 224)
        dtype_call_result_134224 = invoke(stypy.reporting.localization.Localization(__file__, 224, 32), dtype_134221, *[str_134222], **kwargs_134223)
        
        keyword_134225 = dtype_call_result_134224
        
        # Call to array(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'S' (line 225)
        S_134228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 42), 'S', False)
        # Processing the call keyword arguments (line 225)
        kwargs_134229 = {}
        # Getting the type of 'np' (line 225)
        np_134226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 33), 'np', False)
        # Obtaining the member 'array' of a type (line 225)
        array_134227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 33), np_134226, 'array')
        # Calling array(args, kwargs) (line 225)
        array_call_result_134230 = invoke(stypy.reporting.localization.Localization(__file__, 225, 33), array_134227, *[S_134228], **kwargs_134229)
        
        keyword_134231 = array_call_result_134230
        kwargs_134232 = {'buffer': keyword_134231, 'dtype': keyword_134225, 'shape': keyword_134219}
        # Getting the type of 'np' (line 223)
        np_134215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 223)
        ndarray_134216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), np_134215, 'ndarray')
        # Calling ndarray(args, kwargs) (line 223)
        ndarray_call_result_134233 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), ndarray_134216, *[], **kwargs_134232)
        
        # Obtaining the member 'copy' of a type (line 223)
        copy_134234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), ndarray_call_result_134233, 'copy')
        # Calling copy(args, kwargs) (line 223)
        copy_call_result_134236 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), copy_134234, *[], **kwargs_134235)
        
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', copy_call_result_134236)
        
        # ################# End of 'read_char_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_char_array' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_134237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_char_array'
        return stypy_return_type_134237


    @norecursion
    def read_sparse_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_sparse_array'
        module_type_store = module_type_store.open_function_context('read_sparse_array', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_localization', localization)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_function_name', 'VarReader4.read_sparse_array')
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_param_names_list', ['hdr'])
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarReader4.read_sparse_array.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.read_sparse_array', ['hdr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_sparse_array', localization, ['hdr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_sparse_array(...)' code ##################

        str_134238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', ' Read and return sparse matrix type\n\n        Parameters\n        ----------\n        hdr : ``VarHeader4`` instance\n\n        Returns\n        -------\n        arr : ``scipy.sparse.coo_matrix``\n            with dtype ``float`` and shape read from the sparse matrix data\n\n        Notes\n        -----\n        MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where\n        N is the number of non-zero values.  Column 1 values [0:N] are the\n        (1-based) row indices of the each non-zero value, column 2 [0:N] are the\n        column indices, column 3 [0:N] are the (real) values.  The last values\n        [-1,0:2] of the rows, column indices are shape[0] and shape[1]\n        respectively of the output matrix. The last value for the values column\n        is a padding 0. mrows and ncols values from the header give the shape of\n        the stored matrix, here [N+1, 3].  Complex data is saved as a 4 column\n        matrix, where the fourth column contains the imaginary component; the\n        last value is again 0.  Complex sparse data do *not* have the header\n        ``imagf`` field set to True; the fact that the data are complex is only\n        detectable because there are 4 storage columns\n        ')
        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to read_sub_array(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'hdr' (line 254)
        hdr_134241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'hdr', False)
        # Processing the call keyword arguments (line 254)
        kwargs_134242 = {}
        # Getting the type of 'self' (line 254)
        self_134239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 14), 'self', False)
        # Obtaining the member 'read_sub_array' of a type (line 254)
        read_sub_array_134240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 14), self_134239, 'read_sub_array')
        # Calling read_sub_array(args, kwargs) (line 254)
        read_sub_array_call_result_134243 = invoke(stypy.reporting.localization.Localization(__file__, 254, 14), read_sub_array_134240, *[hdr_134241], **kwargs_134242)
        
        # Assigning a type to the variable 'res' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'res', read_sub_array_call_result_134243)
        
        # Assigning a Subscript to a Name (line 255):
        
        # Assigning a Subscript to a Name (line 255):
        
        # Obtaining the type of the subscript
        int_134244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 19), 'int')
        slice_134245 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 255, 14), None, int_134244, None)
        slice_134246 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 255, 14), None, None, None)
        # Getting the type of 'res' (line 255)
        res_134247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 14), 'res')
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___134248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 14), res_134247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_134249 = invoke(stypy.reporting.localization.Localization(__file__, 255, 14), getitem___134248, (slice_134245, slice_134246))
        
        # Assigning a type to the variable 'tmp' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'tmp', subscript_call_result_134249)
        
        # Assigning a Subscript to a Name (line 256):
        
        # Assigning a Subscript to a Name (line 256):
        
        # Obtaining the type of the subscript
        int_134250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'int')
        int_134251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 22), 'int')
        int_134252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'int')
        slice_134253 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 256, 15), int_134251, int_134252, None)
        # Getting the type of 'res' (line 256)
        res_134254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'res')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___134255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 15), res_134254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_134256 = invoke(stypy.reporting.localization.Localization(__file__, 256, 15), getitem___134255, (int_134250, slice_134253))
        
        # Assigning a type to the variable 'dims' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'dims', subscript_call_result_134256)
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to ascontiguousarray(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Obtaining the type of the subscript
        slice_134259 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 257, 33), None, None, None)
        int_134260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 39), 'int')
        # Getting the type of 'tmp' (line 257)
        tmp_134261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 33), 'tmp', False)
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___134262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 33), tmp_134261, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_134263 = invoke(stypy.reporting.localization.Localization(__file__, 257, 33), getitem___134262, (slice_134259, int_134260))
        
        # Processing the call keyword arguments (line 257)
        str_134264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 48), 'str', 'intc')
        keyword_134265 = str_134264
        kwargs_134266 = {'dtype': keyword_134265}
        # Getting the type of 'np' (line 257)
        np_134257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 257)
        ascontiguousarray_134258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), np_134257, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 257)
        ascontiguousarray_call_result_134267 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), ascontiguousarray_134258, *[subscript_call_result_134263], **kwargs_134266)
        
        # Assigning a type to the variable 'I' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'I', ascontiguousarray_call_result_134267)
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to ascontiguousarray(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Obtaining the type of the subscript
        slice_134270 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 258, 33), None, None, None)
        int_134271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 39), 'int')
        # Getting the type of 'tmp' (line 258)
        tmp_134272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 'tmp', False)
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___134273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 33), tmp_134272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_134274 = invoke(stypy.reporting.localization.Localization(__file__, 258, 33), getitem___134273, (slice_134270, int_134271))
        
        # Processing the call keyword arguments (line 258)
        str_134275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 48), 'str', 'intc')
        keyword_134276 = str_134275
        kwargs_134277 = {'dtype': keyword_134276}
        # Getting the type of 'np' (line 258)
        np_134268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 258)
        ascontiguousarray_134269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), np_134268, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 258)
        ascontiguousarray_call_result_134278 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), ascontiguousarray_134269, *[subscript_call_result_134274], **kwargs_134277)
        
        # Assigning a type to the variable 'J' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'J', ascontiguousarray_call_result_134278)
        
        # Getting the type of 'I' (line 259)
        I_134279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'I')
        int_134280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 13), 'int')
        # Applying the binary operator '-=' (line 259)
        result_isub_134281 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 8), '-=', I_134279, int_134280)
        # Assigning a type to the variable 'I' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'I', result_isub_134281)
        
        
        # Getting the type of 'J' (line 260)
        J_134282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'J')
        int_134283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 13), 'int')
        # Applying the binary operator '-=' (line 260)
        result_isub_134284 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 8), '-=', J_134282, int_134283)
        # Assigning a type to the variable 'J' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'J', result_isub_134284)
        
        
        
        
        # Obtaining the type of the subscript
        int_134285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 21), 'int')
        # Getting the type of 'res' (line 261)
        res_134286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'res')
        # Obtaining the member 'shape' of a type (line 261)
        shape_134287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 11), res_134286, 'shape')
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___134288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 11), shape_134287, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_134289 = invoke(stypy.reporting.localization.Localization(__file__, 261, 11), getitem___134288, int_134285)
        
        int_134290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 27), 'int')
        # Applying the binary operator '==' (line 261)
        result_eq_134291 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), '==', subscript_call_result_134289, int_134290)
        
        # Testing the type of an if condition (line 261)
        if_condition_134292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_eq_134291)
        # Assigning a type to the variable 'if_condition_134292' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_134292', if_condition_134292)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Call to ascontiguousarray(...): (line 262)
        # Processing the call arguments (line 262)
        
        # Obtaining the type of the subscript
        slice_134295 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 262, 37), None, None, None)
        int_134296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 43), 'int')
        # Getting the type of 'tmp' (line 262)
        tmp_134297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 37), 'tmp', False)
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___134298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 37), tmp_134297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_134299 = invoke(stypy.reporting.localization.Localization(__file__, 262, 37), getitem___134298, (slice_134295, int_134296))
        
        # Processing the call keyword arguments (line 262)
        str_134300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 52), 'str', 'float')
        keyword_134301 = str_134300
        kwargs_134302 = {'dtype': keyword_134301}
        # Getting the type of 'np' (line 262)
        np_134293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 262)
        ascontiguousarray_134294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 16), np_134293, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 262)
        ascontiguousarray_call_result_134303 = invoke(stypy.reporting.localization.Localization(__file__, 262, 16), ascontiguousarray_134294, *[subscript_call_result_134299], **kwargs_134302)
        
        # Assigning a type to the variable 'V' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'V', ascontiguousarray_call_result_134303)
        # SSA branch for the else part of an if statement (line 261)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to ascontiguousarray(...): (line 264)
        # Processing the call arguments (line 264)
        
        # Obtaining the type of the subscript
        slice_134306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 37), None, None, None)
        int_134307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 43), 'int')
        # Getting the type of 'tmp' (line 264)
        tmp_134308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 37), 'tmp', False)
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___134309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 37), tmp_134308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_134310 = invoke(stypy.reporting.localization.Localization(__file__, 264, 37), getitem___134309, (slice_134306, int_134307))
        
        # Processing the call keyword arguments (line 264)
        str_134311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 52), 'str', 'complex')
        keyword_134312 = str_134311
        kwargs_134313 = {'dtype': keyword_134312}
        # Getting the type of 'np' (line 264)
        np_134304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 264)
        ascontiguousarray_134305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), np_134304, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 264)
        ascontiguousarray_call_result_134314 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), ascontiguousarray_134305, *[subscript_call_result_134310], **kwargs_134313)
        
        # Assigning a type to the variable 'V' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'V', ascontiguousarray_call_result_134314)
        
        # Assigning a Subscript to a Attribute (line 265):
        
        # Assigning a Subscript to a Attribute (line 265):
        
        # Obtaining the type of the subscript
        slice_134315 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 21), None, None, None)
        int_134316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'int')
        # Getting the type of 'tmp' (line 265)
        tmp_134317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'tmp')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___134318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 21), tmp_134317, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_134319 = invoke(stypy.reporting.localization.Localization(__file__, 265, 21), getitem___134318, (slice_134315, int_134316))
        
        # Getting the type of 'V' (line 265)
        V_134320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'V')
        # Setting the type of the member 'imag' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), V_134320, 'imag', subscript_call_result_134319)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to coo_matrix(...): (line 266)
        # Processing the call arguments (line 266)
        
        # Obtaining an instance of the builtin type 'tuple' (line 266)
        tuple_134324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 266)
        # Adding element type (line 266)
        # Getting the type of 'V' (line 266)
        V_134325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 40), 'V', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 40), tuple_134324, V_134325)
        # Adding element type (line 266)
        
        # Obtaining an instance of the builtin type 'tuple' (line 266)
        tuple_134326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 266)
        # Adding element type (line 266)
        # Getting the type of 'I' (line 266)
        I_134327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 43), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 43), tuple_134326, I_134327)
        # Adding element type (line 266)
        # Getting the type of 'J' (line 266)
        J_134328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 45), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 43), tuple_134326, J_134328)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 40), tuple_134324, tuple_134326)
        
        # Getting the type of 'dims' (line 266)
        dims_134329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 50), 'dims', False)
        # Processing the call keyword arguments (line 266)
        kwargs_134330 = {}
        # Getting the type of 'scipy' (line 266)
        scipy_134321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 266)
        sparse_134322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 15), scipy_134321, 'sparse')
        # Obtaining the member 'coo_matrix' of a type (line 266)
        coo_matrix_134323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 15), sparse_134322, 'coo_matrix')
        # Calling coo_matrix(args, kwargs) (line 266)
        coo_matrix_call_result_134331 = invoke(stypy.reporting.localization.Localization(__file__, 266, 15), coo_matrix_134323, *[tuple_134324, dims_134329], **kwargs_134330)
        
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type', coo_matrix_call_result_134331)
        
        # ################# End of 'read_sparse_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_sparse_array' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_134332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134332)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_sparse_array'
        return stypy_return_type_134332


    @norecursion
    def shape_from_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shape_from_header'
        module_type_store = module_type_store.open_function_context('shape_from_header', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_localization', localization)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_function_name', 'VarReader4.shape_from_header')
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_param_names_list', ['hdr'])
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarReader4.shape_from_header.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarReader4.shape_from_header', ['hdr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shape_from_header', localization, ['hdr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shape_from_header(...)' code ##################

        str_134333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, (-1)), 'str', 'Read the shape of the array described by the header.\n        The file position after this call is unspecified.\n        ')
        
        # Assigning a Attribute to a Name (line 272):
        
        # Assigning a Attribute to a Name (line 272):
        # Getting the type of 'hdr' (line 272)
        hdr_134334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'hdr')
        # Obtaining the member 'mclass' of a type (line 272)
        mclass_134335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 17), hdr_134334, 'mclass')
        # Assigning a type to the variable 'mclass' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'mclass', mclass_134335)
        
        
        # Getting the type of 'mclass' (line 273)
        mclass_134336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'mclass')
        # Getting the type of 'mxFULL_CLASS' (line 273)
        mxFULL_CLASS_134337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'mxFULL_CLASS')
        # Applying the binary operator '==' (line 273)
        result_eq_134338 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), '==', mclass_134336, mxFULL_CLASS_134337)
        
        # Testing the type of an if condition (line 273)
        if_condition_134339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_134338)
        # Assigning a type to the variable 'if_condition_134339' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_134339', if_condition_134339)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to tuple(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Call to map(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'int' (line 274)
        int_134342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 30), 'int', False)
        # Getting the type of 'hdr' (line 274)
        hdr_134343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'hdr', False)
        # Obtaining the member 'dims' of a type (line 274)
        dims_134344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 35), hdr_134343, 'dims')
        # Processing the call keyword arguments (line 274)
        kwargs_134345 = {}
        # Getting the type of 'map' (line 274)
        map_134341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 26), 'map', False)
        # Calling map(args, kwargs) (line 274)
        map_call_result_134346 = invoke(stypy.reporting.localization.Localization(__file__, 274, 26), map_134341, *[int_134342, dims_134344], **kwargs_134345)
        
        # Processing the call keyword arguments (line 274)
        kwargs_134347 = {}
        # Getting the type of 'tuple' (line 274)
        tuple_134340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 274)
        tuple_call_result_134348 = invoke(stypy.reporting.localization.Localization(__file__, 274, 20), tuple_134340, *[map_call_result_134346], **kwargs_134347)
        
        # Assigning a type to the variable 'shape' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'shape', tuple_call_result_134348)
        # SSA branch for the else part of an if statement (line 273)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mclass' (line 275)
        mclass_134349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'mclass')
        # Getting the type of 'mxCHAR_CLASS' (line 275)
        mxCHAR_CLASS_134350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'mxCHAR_CLASS')
        # Applying the binary operator '==' (line 275)
        result_eq_134351 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 13), '==', mclass_134349, mxCHAR_CLASS_134350)
        
        # Testing the type of an if condition (line 275)
        if_condition_134352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 13), result_eq_134351)
        # Assigning a type to the variable 'if_condition_134352' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'if_condition_134352', if_condition_134352)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to tuple(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Call to map(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'int' (line 276)
        int_134355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 30), 'int', False)
        # Getting the type of 'hdr' (line 276)
        hdr_134356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'hdr', False)
        # Obtaining the member 'dims' of a type (line 276)
        dims_134357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 35), hdr_134356, 'dims')
        # Processing the call keyword arguments (line 276)
        kwargs_134358 = {}
        # Getting the type of 'map' (line 276)
        map_134354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'map', False)
        # Calling map(args, kwargs) (line 276)
        map_call_result_134359 = invoke(stypy.reporting.localization.Localization(__file__, 276, 26), map_134354, *[int_134355, dims_134357], **kwargs_134358)
        
        # Processing the call keyword arguments (line 276)
        kwargs_134360 = {}
        # Getting the type of 'tuple' (line 276)
        tuple_134353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 276)
        tuple_call_result_134361 = invoke(stypy.reporting.localization.Localization(__file__, 276, 20), tuple_134353, *[map_call_result_134359], **kwargs_134360)
        
        # Assigning a type to the variable 'shape' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'shape', tuple_call_result_134361)
        
        # Getting the type of 'self' (line 277)
        self_134362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'self')
        # Obtaining the member 'chars_as_strings' of a type (line 277)
        chars_as_strings_134363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 15), self_134362, 'chars_as_strings')
        # Testing the type of an if condition (line 277)
        if_condition_134364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 12), chars_as_strings_134363)
        # Assigning a type to the variable 'if_condition_134364' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'if_condition_134364', if_condition_134364)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 278):
        
        # Assigning a Subscript to a Name (line 278):
        
        # Obtaining the type of the subscript
        int_134365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
        slice_134366 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 278, 24), None, int_134365, None)
        # Getting the type of 'shape' (line 278)
        shape_134367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'shape')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___134368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 24), shape_134367, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_134369 = invoke(stypy.reporting.localization.Localization(__file__, 278, 24), getitem___134368, slice_134366)
        
        # Assigning a type to the variable 'shape' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'shape', subscript_call_result_134369)
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 275)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mclass' (line 279)
        mclass_134370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 13), 'mclass')
        # Getting the type of 'mxSPARSE_CLASS' (line 279)
        mxSPARSE_CLASS_134371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'mxSPARSE_CLASS')
        # Applying the binary operator '==' (line 279)
        result_eq_134372 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 13), '==', mclass_134370, mxSPARSE_CLASS_134371)
        
        # Testing the type of an if condition (line 279)
        if_condition_134373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 13), result_eq_134372)
        # Assigning a type to the variable 'if_condition_134373' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 13), 'if_condition_134373', if_condition_134373)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 280):
        
        # Assigning a Attribute to a Name (line 280):
        # Getting the type of 'hdr' (line 280)
        hdr_134374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 17), 'hdr')
        # Obtaining the member 'dtype' of a type (line 280)
        dtype_134375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 17), hdr_134374, 'dtype')
        # Assigning a type to the variable 'dt' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'dt', dtype_134375)
        
        # Assigning a Attribute to a Name (line 281):
        
        # Assigning a Attribute to a Name (line 281):
        # Getting the type of 'hdr' (line 281)
        hdr_134376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'hdr')
        # Obtaining the member 'dims' of a type (line 281)
        dims_134377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), hdr_134376, 'dims')
        # Assigning a type to the variable 'dims' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'dims', dims_134377)
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'dims' (line 283)
        dims_134379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'dims', False)
        # Processing the call keyword arguments (line 283)
        kwargs_134380 = {}
        # Getting the type of 'len' (line 283)
        len_134378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'len', False)
        # Calling len(args, kwargs) (line 283)
        len_call_result_134381 = invoke(stypy.reporting.localization.Localization(__file__, 283, 20), len_134378, *[dims_134379], **kwargs_134380)
        
        int_134382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 33), 'int')
        # Applying the binary operator '==' (line 283)
        result_eq_134383 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 20), '==', len_call_result_134381, int_134382)
        
        
        
        # Obtaining the type of the subscript
        int_134384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 44), 'int')
        # Getting the type of 'dims' (line 283)
        dims_134385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 39), 'dims')
        # Obtaining the member '__getitem__' of a type (line 283)
        getitem___134386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 39), dims_134385, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 283)
        subscript_call_result_134387 = invoke(stypy.reporting.localization.Localization(__file__, 283, 39), getitem___134386, int_134384)
        
        int_134388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 50), 'int')
        # Applying the binary operator '>=' (line 283)
        result_ge_134389 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 39), '>=', subscript_call_result_134387, int_134388)
        
        # Applying the binary operator 'and' (line 283)
        result_and_keyword_134390 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 20), 'and', result_eq_134383, result_ge_134389)
        
        
        # Obtaining the type of the subscript
        int_134391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 61), 'int')
        # Getting the type of 'dims' (line 283)
        dims_134392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 56), 'dims')
        # Obtaining the member '__getitem__' of a type (line 283)
        getitem___134393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 56), dims_134392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 283)
        subscript_call_result_134394 = invoke(stypy.reporting.localization.Localization(__file__, 283, 56), getitem___134393, int_134391)
        
        int_134395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 67), 'int')
        # Applying the binary operator '>=' (line 283)
        result_ge_134396 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 56), '>=', subscript_call_result_134394, int_134395)
        
        # Applying the binary operator 'and' (line 283)
        result_and_keyword_134397 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 20), 'and', result_and_keyword_134390, result_ge_134396)
        
        # Applying the 'not' unary operator (line 283)
        result_not__134398 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), 'not', result_and_keyword_134397)
        
        # Testing the type of an if condition (line 283)
        if_condition_134399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 12), result_not__134398)
        # Assigning a type to the variable 'if_condition_134399' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'if_condition_134399', if_condition_134399)
        # SSA begins for if statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_134400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        
        # Assigning a type to the variable 'stypy_return_type' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'stypy_return_type', tuple_134400)
        # SSA join for if statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to seek(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'dt' (line 287)
        dt_134404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 33), 'dt', False)
        # Obtaining the member 'itemsize' of a type (line 287)
        itemsize_134405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 33), dt_134404, 'itemsize')
        
        # Obtaining the type of the subscript
        int_134406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 53), 'int')
        # Getting the type of 'dims' (line 287)
        dims_134407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 48), 'dims', False)
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___134408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 48), dims_134407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_134409 = invoke(stypy.reporting.localization.Localization(__file__, 287, 48), getitem___134408, int_134406)
        
        int_134410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 58), 'int')
        # Applying the binary operator '-' (line 287)
        result_sub_134411 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 48), '-', subscript_call_result_134409, int_134410)
        
        # Applying the binary operator '*' (line 287)
        result_mul_134412 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 33), '*', itemsize_134405, result_sub_134411)
        
        int_134413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 62), 'int')
        # Processing the call keyword arguments (line 287)
        kwargs_134414 = {}
        # Getting the type of 'self' (line 287)
        self_134401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 287)
        mat_stream_134402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), self_134401, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 287)
        seek_134403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), mat_stream_134402, 'seek')
        # Calling seek(args, kwargs) (line 287)
        seek_call_result_134415 = invoke(stypy.reporting.localization.Localization(__file__, 287, 12), seek_134403, *[result_mul_134412, int_134413], **kwargs_134414)
        
        
        # Assigning a Call to a Name (line 288):
        
        # Assigning a Call to a Name (line 288):
        
        # Call to ndarray(...): (line 288)
        # Processing the call keyword arguments (line 288)
        
        # Obtaining an instance of the builtin type 'tuple' (line 288)
        tuple_134418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 288)
        # Adding element type (line 288)
        int_134419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 37), tuple_134418, int_134419)
        
        keyword_134420 = tuple_134418
        # Getting the type of 'dt' (line 288)
        dt_134421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 48), 'dt', False)
        keyword_134422 = dt_134421
        
        # Call to read(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'dt' (line 289)
        dt_134426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 58), 'dt', False)
        # Obtaining the member 'itemsize' of a type (line 289)
        itemsize_134427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 58), dt_134426, 'itemsize')
        # Processing the call keyword arguments (line 289)
        kwargs_134428 = {}
        # Getting the type of 'self' (line 289)
        self_134423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 37), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 289)
        mat_stream_134424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 37), self_134423, 'mat_stream')
        # Obtaining the member 'read' of a type (line 289)
        read_134425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 37), mat_stream_134424, 'read')
        # Calling read(args, kwargs) (line 289)
        read_call_result_134429 = invoke(stypy.reporting.localization.Localization(__file__, 289, 37), read_134425, *[itemsize_134427], **kwargs_134428)
        
        keyword_134430 = read_call_result_134429
        kwargs_134431 = {'buffer': keyword_134430, 'dtype': keyword_134422, 'shape': keyword_134420}
        # Getting the type of 'np' (line 288)
        np_134416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 288)
        ndarray_134417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 19), np_134416, 'ndarray')
        # Calling ndarray(args, kwargs) (line 288)
        ndarray_call_result_134432 = invoke(stypy.reporting.localization.Localization(__file__, 288, 19), ndarray_134417, *[], **kwargs_134431)
        
        # Assigning a type to the variable 'rows' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'rows', ndarray_call_result_134432)
        
        # Call to seek(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'dt' (line 290)
        dt_134436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 33), 'dt', False)
        # Obtaining the member 'itemsize' of a type (line 290)
        itemsize_134437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 33), dt_134436, 'itemsize')
        
        # Obtaining the type of the subscript
        int_134438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 53), 'int')
        # Getting the type of 'dims' (line 290)
        dims_134439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 48), 'dims', False)
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___134440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 48), dims_134439, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_134441 = invoke(stypy.reporting.localization.Localization(__file__, 290, 48), getitem___134440, int_134438)
        
        int_134442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 58), 'int')
        # Applying the binary operator '-' (line 290)
        result_sub_134443 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 48), '-', subscript_call_result_134441, int_134442)
        
        # Applying the binary operator '*' (line 290)
        result_mul_134444 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 33), '*', itemsize_134437, result_sub_134443)
        
        int_134445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 62), 'int')
        # Processing the call keyword arguments (line 290)
        kwargs_134446 = {}
        # Getting the type of 'self' (line 290)
        self_134433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 290)
        mat_stream_134434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), self_134433, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 290)
        seek_134435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), mat_stream_134434, 'seek')
        # Calling seek(args, kwargs) (line 290)
        seek_call_result_134447 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), seek_134435, *[result_mul_134444, int_134445], **kwargs_134446)
        
        
        # Assigning a Call to a Name (line 291):
        
        # Assigning a Call to a Name (line 291):
        
        # Call to ndarray(...): (line 291)
        # Processing the call keyword arguments (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_134450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        int_134451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 37), tuple_134450, int_134451)
        
        keyword_134452 = tuple_134450
        # Getting the type of 'dt' (line 291)
        dt_134453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 48), 'dt', False)
        keyword_134454 = dt_134453
        
        # Call to read(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'dt' (line 292)
        dt_134458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 58), 'dt', False)
        # Obtaining the member 'itemsize' of a type (line 292)
        itemsize_134459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 58), dt_134458, 'itemsize')
        # Processing the call keyword arguments (line 292)
        kwargs_134460 = {}
        # Getting the type of 'self' (line 292)
        self_134455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 37), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 292)
        mat_stream_134456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 37), self_134455, 'mat_stream')
        # Obtaining the member 'read' of a type (line 292)
        read_134457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 37), mat_stream_134456, 'read')
        # Calling read(args, kwargs) (line 292)
        read_call_result_134461 = invoke(stypy.reporting.localization.Localization(__file__, 292, 37), read_134457, *[itemsize_134459], **kwargs_134460)
        
        keyword_134462 = read_call_result_134461
        kwargs_134463 = {'buffer': keyword_134462, 'dtype': keyword_134454, 'shape': keyword_134452}
        # Getting the type of 'np' (line 291)
        np_134448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 19), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 291)
        ndarray_134449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 19), np_134448, 'ndarray')
        # Calling ndarray(args, kwargs) (line 291)
        ndarray_call_result_134464 = invoke(stypy.reporting.localization.Localization(__file__, 291, 19), ndarray_134449, *[], **kwargs_134463)
        
        # Assigning a type to the variable 'cols' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'cols', ndarray_call_result_134464)
        
        # Assigning a Tuple to a Name (line 294):
        
        # Assigning a Tuple to a Name (line 294):
        
        # Obtaining an instance of the builtin type 'tuple' (line 294)
        tuple_134465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 294)
        # Adding element type (line 294)
        
        # Call to int(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'rows' (line 294)
        rows_134467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 'rows', False)
        # Processing the call keyword arguments (line 294)
        kwargs_134468 = {}
        # Getting the type of 'int' (line 294)
        int_134466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'int', False)
        # Calling int(args, kwargs) (line 294)
        int_call_result_134469 = invoke(stypy.reporting.localization.Localization(__file__, 294, 21), int_134466, *[rows_134467], **kwargs_134468)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), tuple_134465, int_call_result_134469)
        # Adding element type (line 294)
        
        # Call to int(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'cols' (line 294)
        cols_134471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 36), 'cols', False)
        # Processing the call keyword arguments (line 294)
        kwargs_134472 = {}
        # Getting the type of 'int' (line 294)
        int_134470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 32), 'int', False)
        # Calling int(args, kwargs) (line 294)
        int_call_result_134473 = invoke(stypy.reporting.localization.Localization(__file__, 294, 32), int_134470, *[cols_134471], **kwargs_134472)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), tuple_134465, int_call_result_134473)
        
        # Assigning a type to the variable 'shape' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'shape', tuple_134465)
        # SSA branch for the else part of an if statement (line 279)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 296)
        # Processing the call arguments (line 296)
        str_134475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 28), 'str', 'No reader for class code %s')
        # Getting the type of 'mclass' (line 296)
        mclass_134476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 60), 'mclass', False)
        # Applying the binary operator '%' (line 296)
        result_mod_134477 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 28), '%', str_134475, mclass_134476)
        
        # Processing the call keyword arguments (line 296)
        kwargs_134478 = {}
        # Getting the type of 'TypeError' (line 296)
        TypeError_134474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 296)
        TypeError_call_result_134479 = invoke(stypy.reporting.localization.Localization(__file__, 296, 18), TypeError_134474, *[result_mod_134477], **kwargs_134478)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 296, 12), TypeError_call_result_134479, 'raise parameter', BaseException)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 298)
        self_134480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'self')
        # Obtaining the member 'squeeze_me' of a type (line 298)
        squeeze_me_134481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), self_134480, 'squeeze_me')
        # Testing the type of an if condition (line 298)
        if_condition_134482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), squeeze_me_134481)
        # Assigning a type to the variable 'if_condition_134482' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_134482', if_condition_134482)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to tuple(...): (line 299)
        # Processing the call arguments (line 299)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'shape' (line 299)
        shape_134488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 38), 'shape', False)
        comprehension_134489 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 27), shape_134488)
        # Assigning a type to the variable 'x' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'x', comprehension_134489)
        
        # Getting the type of 'x' (line 299)
        x_134485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 47), 'x', False)
        int_134486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 52), 'int')
        # Applying the binary operator '!=' (line 299)
        result_ne_134487 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 47), '!=', x_134485, int_134486)
        
        # Getting the type of 'x' (line 299)
        x_134484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'x', False)
        list_134490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 27), list_134490, x_134484)
        # Processing the call keyword arguments (line 299)
        kwargs_134491 = {}
        # Getting the type of 'tuple' (line 299)
        tuple_134483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 299)
        tuple_call_result_134492 = invoke(stypy.reporting.localization.Localization(__file__, 299, 20), tuple_134483, *[list_134490], **kwargs_134491)
        
        # Assigning a type to the variable 'shape' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'shape', tuple_call_result_134492)
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'shape' (line 300)
        shape_134493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'shape')
        # Assigning a type to the variable 'stypy_return_type' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'stypy_return_type', shape_134493)
        
        # ################# End of 'shape_from_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shape_from_header' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_134494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shape_from_header'
        return stypy_return_type_134494


# Assigning a type to the variable 'VarReader4' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'VarReader4', VarReader4)
# Declaration of the 'MatFile4Reader' class
# Getting the type of 'MatFileReader' (line 303)
MatFileReader_134495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 'MatFileReader')

class MatFile4Reader(MatFileReader_134495, ):
    str_134496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 4), 'str', ' Reader for Mat4 files ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Reader.__init__', ['mat_stream'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['mat_stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_134497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, (-1)), 'str', ' Initialize matlab 4 file reader\n\n    %(matstream_arg)s\n    %(load_args)s\n        ')
        
        # Call to __init__(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'mat_stream' (line 312)
        mat_stream_134504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 45), 'mat_stream', False)
        # Getting the type of 'args' (line 312)
        args_134505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 58), 'args', False)
        # Processing the call keyword arguments (line 312)
        # Getting the type of 'kwargs' (line 312)
        kwargs_134506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 66), 'kwargs', False)
        kwargs_134507 = {'kwargs_134506': kwargs_134506}
        
        # Call to super(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'MatFile4Reader' (line 312)
        MatFile4Reader_134499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 14), 'MatFile4Reader', False)
        # Getting the type of 'self' (line 312)
        self_134500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'self', False)
        # Processing the call keyword arguments (line 312)
        kwargs_134501 = {}
        # Getting the type of 'super' (line 312)
        super_134498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'super', False)
        # Calling super(args, kwargs) (line 312)
        super_call_result_134502 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), super_134498, *[MatFile4Reader_134499, self_134500], **kwargs_134501)
        
        # Obtaining the member '__init__' of a type (line 312)
        init___134503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), super_call_result_134502, '__init__')
        # Calling __init__(args, kwargs) (line 312)
        init___call_result_134508 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), init___134503, *[mat_stream_134504, args_134505], **kwargs_134507)
        
        
        # Assigning a Name to a Attribute (line 313):
        
        # Assigning a Name to a Attribute (line 313):
        # Getting the type of 'None' (line 313)
        None_134509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 'None')
        # Getting the type of 'self' (line 313)
        self_134510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'self')
        # Setting the type of the member '_matrix_reader' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), self_134510, '_matrix_reader', None_134509)
        
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
        module_type_store = module_type_store.open_function_context('guess_byte_order', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_localization', localization)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_function_name', 'MatFile4Reader.guess_byte_order')
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile4Reader.guess_byte_order.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Reader.guess_byte_order', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to seek(...): (line 316)
        # Processing the call arguments (line 316)
        int_134514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 29), 'int')
        # Processing the call keyword arguments (line 316)
        kwargs_134515 = {}
        # Getting the type of 'self' (line 316)
        self_134511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 316)
        mat_stream_134512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_134511, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 316)
        seek_134513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), mat_stream_134512, 'seek')
        # Calling seek(args, kwargs) (line 316)
        seek_call_result_134516 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), seek_134513, *[int_134514], **kwargs_134515)
        
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to read_dtype(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'self' (line 317)
        self_134518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 26), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 317)
        mat_stream_134519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 26), self_134518, 'mat_stream')
        
        # Call to dtype(...): (line 317)
        # Processing the call arguments (line 317)
        str_134522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 52), 'str', 'i4')
        # Processing the call keyword arguments (line 317)
        kwargs_134523 = {}
        # Getting the type of 'np' (line 317)
        np_134520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 43), 'np', False)
        # Obtaining the member 'dtype' of a type (line 317)
        dtype_134521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 43), np_134520, 'dtype')
        # Calling dtype(args, kwargs) (line 317)
        dtype_call_result_134524 = invoke(stypy.reporting.localization.Localization(__file__, 317, 43), dtype_134521, *[str_134522], **kwargs_134523)
        
        # Processing the call keyword arguments (line 317)
        kwargs_134525 = {}
        # Getting the type of 'read_dtype' (line 317)
        read_dtype_134517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'read_dtype', False)
        # Calling read_dtype(args, kwargs) (line 317)
        read_dtype_call_result_134526 = invoke(stypy.reporting.localization.Localization(__file__, 317, 15), read_dtype_134517, *[mat_stream_134519, dtype_call_result_134524], **kwargs_134525)
        
        # Assigning a type to the variable 'mopt' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'mopt', read_dtype_call_result_134526)
        
        # Call to seek(...): (line 318)
        # Processing the call arguments (line 318)
        int_134530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 29), 'int')
        # Processing the call keyword arguments (line 318)
        kwargs_134531 = {}
        # Getting the type of 'self' (line 318)
        self_134527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 318)
        mat_stream_134528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), self_134527, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 318)
        seek_134529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), mat_stream_134528, 'seek')
        # Calling seek(args, kwargs) (line 318)
        seek_call_result_134532 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), seek_134529, *[int_134530], **kwargs_134531)
        
        
        
        # Getting the type of 'mopt' (line 319)
        mopt_134533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'mopt')
        int_134534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 19), 'int')
        # Applying the binary operator '==' (line 319)
        result_eq_134535 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 11), '==', mopt_134533, int_134534)
        
        # Testing the type of an if condition (line 319)
        if_condition_134536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 8), result_eq_134535)
        # Assigning a type to the variable 'if_condition_134536' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'if_condition_134536', if_condition_134536)
        # SSA begins for if statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_134537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 19), 'str', '<')
        # Assigning a type to the variable 'stypy_return_type' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'stypy_return_type', str_134537)
        # SSA join for if statement (line 319)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mopt' (line 321)
        mopt_134538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'mopt')
        int_134539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 18), 'int')
        # Applying the binary operator '<' (line 321)
        result_lt_134540 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 11), '<', mopt_134538, int_134539)
        
        
        # Getting the type of 'mopt' (line 321)
        mopt_134541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'mopt')
        int_134542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 30), 'int')
        # Applying the binary operator '>' (line 321)
        result_gt_134543 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 23), '>', mopt_134541, int_134542)
        
        # Applying the binary operator 'or' (line 321)
        result_or_keyword_134544 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 11), 'or', result_lt_134540, result_gt_134543)
        
        # Testing the type of an if condition (line 321)
        if_condition_134545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), result_or_keyword_134544)
        # Assigning a type to the variable 'if_condition_134545' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_134545', if_condition_134545)
        # SSA begins for if statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'SYS_LITTLE_ENDIAN' (line 323)
        SYS_LITTLE_ENDIAN_134546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'SYS_LITTLE_ENDIAN')
        str_134547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 41), 'str', '>')
        # Applying the binary operator 'and' (line 323)
        result_and_keyword_134548 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 19), 'and', SYS_LITTLE_ENDIAN_134546, str_134547)
        
        str_134549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 48), 'str', '<')
        # Applying the binary operator 'or' (line 323)
        result_or_keyword_134550 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 19), 'or', result_and_keyword_134548, str_134549)
        
        # Assigning a type to the variable 'stypy_return_type' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'stypy_return_type', result_or_keyword_134550)
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'SYS_LITTLE_ENDIAN' (line 325)
        SYS_LITTLE_ENDIAN_134551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'SYS_LITTLE_ENDIAN')
        str_134552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 37), 'str', '<')
        # Applying the binary operator 'and' (line 325)
        result_and_keyword_134553 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 15), 'and', SYS_LITTLE_ENDIAN_134551, str_134552)
        
        str_134554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 44), 'str', '>')
        # Applying the binary operator 'or' (line 325)
        result_or_keyword_134555 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 15), 'or', result_and_keyword_134553, str_134554)
        
        # Assigning a type to the variable 'stypy_return_type' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'stypy_return_type', result_or_keyword_134555)
        
        # ################# End of 'guess_byte_order(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'guess_byte_order' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_134556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134556)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'guess_byte_order'
        return stypy_return_type_134556


    @norecursion
    def initialize_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_read'
        module_type_store = module_type_store.open_function_context('initialize_read', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_localization', localization)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_function_name', 'MatFile4Reader.initialize_read')
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile4Reader.initialize_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Reader.initialize_read', [], None, None, defaults, varargs, kwargs)

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

        str_134557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, (-1)), 'str', ' Run when beginning read of variables\n\n        Sets up readers from parameters in `self`\n        ')
        
        # Assigning a Call to a Attribute (line 332):
        
        # Assigning a Call to a Attribute (line 332):
        
        # Call to convert_dtypes(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'mdtypes_template' (line 332)
        mdtypes_template_134559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'mdtypes_template', False)
        # Getting the type of 'self' (line 332)
        self_134560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 55), 'self', False)
        # Obtaining the member 'byte_order' of a type (line 332)
        byte_order_134561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 55), self_134560, 'byte_order')
        # Processing the call keyword arguments (line 332)
        kwargs_134562 = {}
        # Getting the type of 'convert_dtypes' (line 332)
        convert_dtypes_134558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'convert_dtypes', False)
        # Calling convert_dtypes(args, kwargs) (line 332)
        convert_dtypes_call_result_134563 = invoke(stypy.reporting.localization.Localization(__file__, 332, 22), convert_dtypes_134558, *[mdtypes_template_134559, byte_order_134561], **kwargs_134562)
        
        # Getting the type of 'self' (line 332)
        self_134564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self')
        # Setting the type of the member 'dtypes' of a type (line 332)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_134564, 'dtypes', convert_dtypes_call_result_134563)
        
        # Assigning a Call to a Attribute (line 333):
        
        # Assigning a Call to a Attribute (line 333):
        
        # Call to VarReader4(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'self' (line 333)
        self_134566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 41), 'self', False)
        # Processing the call keyword arguments (line 333)
        kwargs_134567 = {}
        # Getting the type of 'VarReader4' (line 333)
        VarReader4_134565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 30), 'VarReader4', False)
        # Calling VarReader4(args, kwargs) (line 333)
        VarReader4_call_result_134568 = invoke(stypy.reporting.localization.Localization(__file__, 333, 30), VarReader4_134565, *[self_134566], **kwargs_134567)
        
        # Getting the type of 'self' (line 333)
        self_134569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
        # Setting the type of the member '_matrix_reader' of a type (line 333)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_134569, '_matrix_reader', VarReader4_call_result_134568)
        
        # ################# End of 'initialize_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_read' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_134570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134570)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_read'
        return stypy_return_type_134570


    @norecursion
    def read_var_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_var_header'
        module_type_store = module_type_store.open_function_context('read_var_header', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_localization', localization)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_function_name', 'MatFile4Reader.read_var_header')
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile4Reader.read_var_header.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Reader.read_var_header', [], None, None, defaults, varargs, kwargs)

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

        str_134571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, (-1)), 'str', ' Read and return header, next position\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        header : object\n           object that can be passed to self.read_var_array, and that\n           has attributes ``name`` and ``is_global``\n        next_position : int\n           position in stream of next variable\n        ')
        
        # Assigning a Call to a Name (line 350):
        
        # Assigning a Call to a Name (line 350):
        
        # Call to read_header(...): (line 350)
        # Processing the call keyword arguments (line 350)
        kwargs_134575 = {}
        # Getting the type of 'self' (line 350)
        self_134572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 14), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 350)
        _matrix_reader_134573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 14), self_134572, '_matrix_reader')
        # Obtaining the member 'read_header' of a type (line 350)
        read_header_134574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 14), _matrix_reader_134573, 'read_header')
        # Calling read_header(args, kwargs) (line 350)
        read_header_call_result_134576 = invoke(stypy.reporting.localization.Localization(__file__, 350, 14), read_header_134574, *[], **kwargs_134575)
        
        # Assigning a type to the variable 'hdr' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'hdr', read_header_call_result_134576)
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to reduce(...): (line 351)
        # Processing the call arguments (line 351)

        @norecursion
        def _stypy_temp_lambda_88(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_88'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_88', 351, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_88.stypy_localization = localization
            _stypy_temp_lambda_88.stypy_type_of_self = None
            _stypy_temp_lambda_88.stypy_type_store = module_type_store
            _stypy_temp_lambda_88.stypy_function_name = '_stypy_temp_lambda_88'
            _stypy_temp_lambda_88.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_88.stypy_varargs_param_name = None
            _stypy_temp_lambda_88.stypy_kwargs_param_name = None
            _stypy_temp_lambda_88.stypy_call_defaults = defaults
            _stypy_temp_lambda_88.stypy_call_varargs = varargs
            _stypy_temp_lambda_88.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_88', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_88', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 351)
            x_134578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'x', False)
            # Getting the type of 'y' (line 351)
            y_134579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 34), 'y', False)
            # Applying the binary operator '*' (line 351)
            result_mul_134580 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 32), '*', x_134578, y_134579)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'stypy_return_type', result_mul_134580)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_88' in the type store
            # Getting the type of 'stypy_return_type' (line 351)
            stypy_return_type_134581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_134581)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_88'
            return stypy_return_type_134581

        # Assigning a type to the variable '_stypy_temp_lambda_88' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), '_stypy_temp_lambda_88', _stypy_temp_lambda_88)
        # Getting the type of '_stypy_temp_lambda_88' (line 351)
        _stypy_temp_lambda_88_134582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), '_stypy_temp_lambda_88')
        # Getting the type of 'hdr' (line 351)
        hdr_134583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 37), 'hdr', False)
        # Obtaining the member 'dims' of a type (line 351)
        dims_134584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 37), hdr_134583, 'dims')
        int_134585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 47), 'int')
        # Processing the call keyword arguments (line 351)
        kwargs_134586 = {}
        # Getting the type of 'reduce' (line 351)
        reduce_134577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'reduce', False)
        # Calling reduce(args, kwargs) (line 351)
        reduce_call_result_134587 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), reduce_134577, *[_stypy_temp_lambda_88_134582, dims_134584, int_134585], **kwargs_134586)
        
        # Assigning a type to the variable 'n' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'n', reduce_call_result_134587)
        
        # Assigning a BinOp to a Name (line 352):
        
        # Assigning a BinOp to a Name (line 352):
        # Getting the type of 'hdr' (line 352)
        hdr_134588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 26), 'hdr')
        # Obtaining the member 'dtype' of a type (line 352)
        dtype_134589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 26), hdr_134588, 'dtype')
        # Obtaining the member 'itemsize' of a type (line 352)
        itemsize_134590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 26), dtype_134589, 'itemsize')
        # Getting the type of 'n' (line 352)
        n_134591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 47), 'n')
        # Applying the binary operator '*' (line 352)
        result_mul_134592 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 26), '*', itemsize_134590, n_134591)
        
        # Assigning a type to the variable 'remaining_bytes' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'remaining_bytes', result_mul_134592)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'hdr' (line 353)
        hdr_134593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'hdr')
        # Obtaining the member 'is_complex' of a type (line 353)
        is_complex_134594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 11), hdr_134593, 'is_complex')
        
        
        # Getting the type of 'hdr' (line 353)
        hdr_134595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 'hdr')
        # Obtaining the member 'mclass' of a type (line 353)
        mclass_134596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 34), hdr_134595, 'mclass')
        # Getting the type of 'mxSPARSE_CLASS' (line 353)
        mxSPARSE_CLASS_134597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 48), 'mxSPARSE_CLASS')
        # Applying the binary operator '==' (line 353)
        result_eq_134598 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 34), '==', mclass_134596, mxSPARSE_CLASS_134597)
        
        # Applying the 'not' unary operator (line 353)
        result_not__134599 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 30), 'not', result_eq_134598)
        
        # Applying the binary operator 'and' (line 353)
        result_and_keyword_134600 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 11), 'and', is_complex_134594, result_not__134599)
        
        # Testing the type of an if condition (line 353)
        if_condition_134601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 8), result_and_keyword_134600)
        # Assigning a type to the variable 'if_condition_134601' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'if_condition_134601', if_condition_134601)
        # SSA begins for if statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'remaining_bytes' (line 354)
        remaining_bytes_134602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'remaining_bytes')
        int_134603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 31), 'int')
        # Applying the binary operator '*=' (line 354)
        result_imul_134604 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 12), '*=', remaining_bytes_134602, int_134603)
        # Assigning a type to the variable 'remaining_bytes' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'remaining_bytes', result_imul_134604)
        
        # SSA join for if statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 355):
        
        # Assigning a BinOp to a Name (line 355):
        
        # Call to tell(...): (line 355)
        # Processing the call keyword arguments (line 355)
        kwargs_134608 = {}
        # Getting the type of 'self' (line 355)
        self_134605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 355)
        mat_stream_134606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 24), self_134605, 'mat_stream')
        # Obtaining the member 'tell' of a type (line 355)
        tell_134607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 24), mat_stream_134606, 'tell')
        # Calling tell(args, kwargs) (line 355)
        tell_call_result_134609 = invoke(stypy.reporting.localization.Localization(__file__, 355, 24), tell_134607, *[], **kwargs_134608)
        
        # Getting the type of 'remaining_bytes' (line 355)
        remaining_bytes_134610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 49), 'remaining_bytes')
        # Applying the binary operator '+' (line 355)
        result_add_134611 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 24), '+', tell_call_result_134609, remaining_bytes_134610)
        
        # Assigning a type to the variable 'next_position' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'next_position', result_add_134611)
        
        # Obtaining an instance of the builtin type 'tuple' (line 356)
        tuple_134612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 356)
        # Adding element type (line 356)
        # Getting the type of 'hdr' (line 356)
        hdr_134613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'hdr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 15), tuple_134612, hdr_134613)
        # Adding element type (line 356)
        # Getting the type of 'next_position' (line 356)
        next_position_134614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 20), 'next_position')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 15), tuple_134612, next_position_134614)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type', tuple_134612)
        
        # ################# End of 'read_var_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_var_header' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_134615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_var_header'
        return stypy_return_type_134615


    @norecursion
    def read_var_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 358)
        True_134616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 45), 'True')
        defaults = [True_134616]
        # Create a new context for function 'read_var_array'
        module_type_store = module_type_store.open_function_context('read_var_array', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_localization', localization)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_function_name', 'MatFile4Reader.read_var_array')
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_param_names_list', ['header', 'process'])
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile4Reader.read_var_array.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Reader.read_var_array', ['header', 'process'], None, None, defaults, varargs, kwargs)

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

        str_134617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, (-1)), 'str', ' Read array, given `header`\n\n        Parameters\n        ----------\n        header : header object\n           object with fields defining variable header\n        process : {True, False}, optional\n           If True, apply recursive post-processing during loading of array.\n\n        Returns\n        -------\n        arr : array\n           array with post-processing applied or not according to\n           `process`.\n        ')
        
        # Call to array_from_header(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'header' (line 374)
        header_134621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 53), 'header', False)
        # Getting the type of 'process' (line 374)
        process_134622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 61), 'process', False)
        # Processing the call keyword arguments (line 374)
        kwargs_134623 = {}
        # Getting the type of 'self' (line 374)
        self_134618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 374)
        _matrix_reader_134619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 15), self_134618, '_matrix_reader')
        # Obtaining the member 'array_from_header' of a type (line 374)
        array_from_header_134620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 15), _matrix_reader_134619, 'array_from_header')
        # Calling array_from_header(args, kwargs) (line 374)
        array_from_header_call_result_134624 = invoke(stypy.reporting.localization.Localization(__file__, 374, 15), array_from_header_134620, *[header_134621, process_134622], **kwargs_134623)
        
        # Assigning a type to the variable 'stypy_return_type' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'stypy_return_type', array_from_header_call_result_134624)
        
        # ################# End of 'read_var_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_var_array' in the type store
        # Getting the type of 'stypy_return_type' (line 358)
        stypy_return_type_134625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_var_array'
        return stypy_return_type_134625


    @norecursion
    def get_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 376)
        None_134626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 43), 'None')
        defaults = [None_134626]
        # Create a new context for function 'get_variables'
        module_type_store = module_type_store.open_function_context('get_variables', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_localization', localization)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_function_name', 'MatFile4Reader.get_variables')
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_param_names_list', ['variable_names'])
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile4Reader.get_variables.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Reader.get_variables', ['variable_names'], None, None, defaults, varargs, kwargs)

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

        str_134627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, (-1)), 'str', ' get variables from stream as dictionary\n\n        Parameters\n        ----------\n        variable_names : None or str or sequence of str, optional\n            variable name, or sequence of variable names to get from Mat file /\n            file stream.  If None, then get all variables in file\n        ')
        
        
        # Call to isinstance(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'variable_names' (line 385)
        variable_names_134629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 22), 'variable_names', False)
        # Getting the type of 'string_types' (line 385)
        string_types_134630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 38), 'string_types', False)
        # Processing the call keyword arguments (line 385)
        kwargs_134631 = {}
        # Getting the type of 'isinstance' (line 385)
        isinstance_134628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 385)
        isinstance_call_result_134632 = invoke(stypy.reporting.localization.Localization(__file__, 385, 11), isinstance_134628, *[variable_names_134629, string_types_134630], **kwargs_134631)
        
        # Testing the type of an if condition (line 385)
        if_condition_134633 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 8), isinstance_call_result_134632)
        # Assigning a type to the variable 'if_condition_134633' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'if_condition_134633', if_condition_134633)
        # SSA begins for if statement (line 385)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 386):
        
        # Assigning a List to a Name (line 386):
        
        # Obtaining an instance of the builtin type 'list' (line 386)
        list_134634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 386)
        # Adding element type (line 386)
        # Getting the type of 'variable_names' (line 386)
        variable_names_134635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 30), 'variable_names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 29), list_134634, variable_names_134635)
        
        # Assigning a type to the variable 'variable_names' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'variable_names', list_134634)
        # SSA branch for the else part of an if statement (line 385)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 387)
        # Getting the type of 'variable_names' (line 387)
        variable_names_134636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 13), 'variable_names')
        # Getting the type of 'None' (line 387)
        None_134637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 35), 'None')
        
        (may_be_134638, more_types_in_union_134639) = may_not_be_none(variable_names_134636, None_134637)

        if may_be_134638:

            if more_types_in_union_134639:
                # Runtime conditional SSA (line 387)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 388):
            
            # Assigning a Call to a Name (line 388):
            
            # Call to list(...): (line 388)
            # Processing the call arguments (line 388)
            # Getting the type of 'variable_names' (line 388)
            variable_names_134641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 34), 'variable_names', False)
            # Processing the call keyword arguments (line 388)
            kwargs_134642 = {}
            # Getting the type of 'list' (line 388)
            list_134640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'list', False)
            # Calling list(args, kwargs) (line 388)
            list_call_result_134643 = invoke(stypy.reporting.localization.Localization(__file__, 388, 29), list_134640, *[variable_names_134641], **kwargs_134642)
            
            # Assigning a type to the variable 'variable_names' (line 388)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'variable_names', list_call_result_134643)

            if more_types_in_union_134639:
                # SSA join for if statement (line 387)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 385)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to seek(...): (line 389)
        # Processing the call arguments (line 389)
        int_134647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 29), 'int')
        # Processing the call keyword arguments (line 389)
        kwargs_134648 = {}
        # Getting the type of 'self' (line 389)
        self_134644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 389)
        mat_stream_134645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_134644, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 389)
        seek_134646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), mat_stream_134645, 'seek')
        # Calling seek(args, kwargs) (line 389)
        seek_call_result_134649 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), seek_134646, *[int_134647], **kwargs_134648)
        
        
        # Call to initialize_read(...): (line 391)
        # Processing the call keyword arguments (line 391)
        kwargs_134652 = {}
        # Getting the type of 'self' (line 391)
        self_134650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self', False)
        # Obtaining the member 'initialize_read' of a type (line 391)
        initialize_read_134651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_134650, 'initialize_read')
        # Calling initialize_read(args, kwargs) (line 391)
        initialize_read_call_result_134653 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), initialize_read_134651, *[], **kwargs_134652)
        
        
        # Assigning a Dict to a Name (line 392):
        
        # Assigning a Dict to a Name (line 392):
        
        # Obtaining an instance of the builtin type 'dict' (line 392)
        dict_134654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 392)
        
        # Assigning a type to the variable 'mdict' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'mdict', dict_134654)
        
        
        
        # Call to end_of_stream(...): (line 393)
        # Processing the call keyword arguments (line 393)
        kwargs_134657 = {}
        # Getting the type of 'self' (line 393)
        self_134655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 18), 'self', False)
        # Obtaining the member 'end_of_stream' of a type (line 393)
        end_of_stream_134656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 18), self_134655, 'end_of_stream')
        # Calling end_of_stream(args, kwargs) (line 393)
        end_of_stream_call_result_134658 = invoke(stypy.reporting.localization.Localization(__file__, 393, 18), end_of_stream_134656, *[], **kwargs_134657)
        
        # Applying the 'not' unary operator (line 393)
        result_not__134659 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 14), 'not', end_of_stream_call_result_134658)
        
        # Testing the type of an if condition (line 393)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 8), result_not__134659)
        # SSA begins for while statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 394):
        
        # Assigning a Subscript to a Name (line 394):
        
        # Obtaining the type of the subscript
        int_134660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 12), 'int')
        
        # Call to read_var_header(...): (line 394)
        # Processing the call keyword arguments (line 394)
        kwargs_134663 = {}
        # Getting the type of 'self' (line 394)
        self_134661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 394)
        read_var_header_134662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 33), self_134661, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 394)
        read_var_header_call_result_134664 = invoke(stypy.reporting.localization.Localization(__file__, 394, 33), read_var_header_134662, *[], **kwargs_134663)
        
        # Obtaining the member '__getitem__' of a type (line 394)
        getitem___134665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), read_var_header_call_result_134664, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 394)
        subscript_call_result_134666 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), getitem___134665, int_134660)
        
        # Assigning a type to the variable 'tuple_var_assignment_133763' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'tuple_var_assignment_133763', subscript_call_result_134666)
        
        # Assigning a Subscript to a Name (line 394):
        
        # Obtaining the type of the subscript
        int_134667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 12), 'int')
        
        # Call to read_var_header(...): (line 394)
        # Processing the call keyword arguments (line 394)
        kwargs_134670 = {}
        # Getting the type of 'self' (line 394)
        self_134668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 394)
        read_var_header_134669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 33), self_134668, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 394)
        read_var_header_call_result_134671 = invoke(stypy.reporting.localization.Localization(__file__, 394, 33), read_var_header_134669, *[], **kwargs_134670)
        
        # Obtaining the member '__getitem__' of a type (line 394)
        getitem___134672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), read_var_header_call_result_134671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 394)
        subscript_call_result_134673 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), getitem___134672, int_134667)
        
        # Assigning a type to the variable 'tuple_var_assignment_133764' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'tuple_var_assignment_133764', subscript_call_result_134673)
        
        # Assigning a Name to a Name (line 394):
        # Getting the type of 'tuple_var_assignment_133763' (line 394)
        tuple_var_assignment_133763_134674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'tuple_var_assignment_133763')
        # Assigning a type to the variable 'hdr' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'hdr', tuple_var_assignment_133763_134674)
        
        # Assigning a Name to a Name (line 394):
        # Getting the type of 'tuple_var_assignment_133764' (line 394)
        tuple_var_assignment_133764_134675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'tuple_var_assignment_133764')
        # Assigning a type to the variable 'next_position' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 17), 'next_position', tuple_var_assignment_133764_134675)
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to asstr(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'hdr' (line 395)
        hdr_134677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 25), 'hdr', False)
        # Obtaining the member 'name' of a type (line 395)
        name_134678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 25), hdr_134677, 'name')
        # Processing the call keyword arguments (line 395)
        kwargs_134679 = {}
        # Getting the type of 'asstr' (line 395)
        asstr_134676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'asstr', False)
        # Calling asstr(args, kwargs) (line 395)
        asstr_call_result_134680 = invoke(stypy.reporting.localization.Localization(__file__, 395, 19), asstr_134676, *[name_134678], **kwargs_134679)
        
        # Assigning a type to the variable 'name' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'name', asstr_call_result_134680)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'variable_names' (line 396)
        variable_names_134681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 15), 'variable_names')
        # Getting the type of 'None' (line 396)
        None_134682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 37), 'None')
        # Applying the binary operator 'isnot' (line 396)
        result_is_not_134683 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 15), 'isnot', variable_names_134681, None_134682)
        
        
        # Getting the type of 'name' (line 396)
        name_134684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 46), 'name')
        # Getting the type of 'variable_names' (line 396)
        variable_names_134685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 58), 'variable_names')
        # Applying the binary operator 'notin' (line 396)
        result_contains_134686 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 46), 'notin', name_134684, variable_names_134685)
        
        # Applying the binary operator 'and' (line 396)
        result_and_keyword_134687 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 15), 'and', result_is_not_134683, result_contains_134686)
        
        # Testing the type of an if condition (line 396)
        if_condition_134688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 12), result_and_keyword_134687)
        # Assigning a type to the variable 'if_condition_134688' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'if_condition_134688', if_condition_134688)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to seek(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'next_position' (line 397)
        next_position_134692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 37), 'next_position', False)
        # Processing the call keyword arguments (line 397)
        kwargs_134693 = {}
        # Getting the type of 'self' (line 397)
        self_134689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 397)
        mat_stream_134690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 16), self_134689, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 397)
        seek_134691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 16), mat_stream_134690, 'seek')
        # Calling seek(args, kwargs) (line 397)
        seek_call_result_134694 = invoke(stypy.reporting.localization.Localization(__file__, 397, 16), seek_134691, *[next_position_134692], **kwargs_134693)
        
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 399):
        
        # Assigning a Call to a Subscript (line 399):
        
        # Call to read_var_array(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'hdr' (line 399)
        hdr_134697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 46), 'hdr', False)
        # Processing the call keyword arguments (line 399)
        kwargs_134698 = {}
        # Getting the type of 'self' (line 399)
        self_134695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 26), 'self', False)
        # Obtaining the member 'read_var_array' of a type (line 399)
        read_var_array_134696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 26), self_134695, 'read_var_array')
        # Calling read_var_array(args, kwargs) (line 399)
        read_var_array_call_result_134699 = invoke(stypy.reporting.localization.Localization(__file__, 399, 26), read_var_array_134696, *[hdr_134697], **kwargs_134698)
        
        # Getting the type of 'mdict' (line 399)
        mdict_134700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'mdict')
        # Getting the type of 'name' (line 399)
        name_134701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 18), 'name')
        # Storing an element on a container (line 399)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 12), mdict_134700, (name_134701, read_var_array_call_result_134699))
        
        # Call to seek(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'next_position' (line 400)
        next_position_134705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 33), 'next_position', False)
        # Processing the call keyword arguments (line 400)
        kwargs_134706 = {}
        # Getting the type of 'self' (line 400)
        self_134702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 400)
        mat_stream_134703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), self_134702, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 400)
        seek_134704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), mat_stream_134703, 'seek')
        # Calling seek(args, kwargs) (line 400)
        seek_call_result_134707 = invoke(stypy.reporting.localization.Localization(__file__, 400, 12), seek_134704, *[next_position_134705], **kwargs_134706)
        
        
        # Type idiom detected: calculating its left and rigth part (line 401)
        # Getting the type of 'variable_names' (line 401)
        variable_names_134708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'variable_names')
        # Getting the type of 'None' (line 401)
        None_134709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 37), 'None')
        
        (may_be_134710, more_types_in_union_134711) = may_not_be_none(variable_names_134708, None_134709)

        if may_be_134710:

            if more_types_in_union_134711:
                # Runtime conditional SSA (line 401)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to remove(...): (line 402)
            # Processing the call arguments (line 402)
            # Getting the type of 'name' (line 402)
            name_134714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 38), 'name', False)
            # Processing the call keyword arguments (line 402)
            kwargs_134715 = {}
            # Getting the type of 'variable_names' (line 402)
            variable_names_134712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'variable_names', False)
            # Obtaining the member 'remove' of a type (line 402)
            remove_134713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 16), variable_names_134712, 'remove')
            # Calling remove(args, kwargs) (line 402)
            remove_call_result_134716 = invoke(stypy.reporting.localization.Localization(__file__, 402, 16), remove_134713, *[name_134714], **kwargs_134715)
            
            
            
            
            # Call to len(...): (line 403)
            # Processing the call arguments (line 403)
            # Getting the type of 'variable_names' (line 403)
            variable_names_134718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 23), 'variable_names', False)
            # Processing the call keyword arguments (line 403)
            kwargs_134719 = {}
            # Getting the type of 'len' (line 403)
            len_134717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'len', False)
            # Calling len(args, kwargs) (line 403)
            len_call_result_134720 = invoke(stypy.reporting.localization.Localization(__file__, 403, 19), len_134717, *[variable_names_134718], **kwargs_134719)
            
            int_134721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 42), 'int')
            # Applying the binary operator '==' (line 403)
            result_eq_134722 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 19), '==', len_call_result_134720, int_134721)
            
            # Testing the type of an if condition (line 403)
            if_condition_134723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 16), result_eq_134722)
            # Assigning a type to the variable 'if_condition_134723' (line 403)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'if_condition_134723', if_condition_134723)
            # SSA begins for if statement (line 403)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 403)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_134711:
                # SSA join for if statement (line 401)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for while statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'mdict' (line 405)
        mdict_134724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'mdict')
        # Assigning a type to the variable 'stypy_return_type' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'stypy_return_type', mdict_134724)
        
        # ################# End of 'get_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_134725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_variables'
        return stypy_return_type_134725


    @norecursion
    def list_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'list_variables'
        module_type_store = module_type_store.open_function_context('list_variables', 407, 4, False)
        # Assigning a type to the variable 'self' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_localization', localization)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_function_name', 'MatFile4Reader.list_variables')
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_param_names_list', [])
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile4Reader.list_variables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Reader.list_variables', [], None, None, defaults, varargs, kwargs)

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

        str_134726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 8), 'str', ' list variables from stream ')
        
        # Call to seek(...): (line 409)
        # Processing the call arguments (line 409)
        int_134730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 29), 'int')
        # Processing the call keyword arguments (line 409)
        kwargs_134731 = {}
        # Getting the type of 'self' (line 409)
        self_134727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 409)
        mat_stream_134728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), self_134727, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 409)
        seek_134729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), mat_stream_134728, 'seek')
        # Calling seek(args, kwargs) (line 409)
        seek_call_result_134732 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), seek_134729, *[int_134730], **kwargs_134731)
        
        
        # Call to initialize_read(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_134735 = {}
        # Getting the type of 'self' (line 411)
        self_134733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self', False)
        # Obtaining the member 'initialize_read' of a type (line 411)
        initialize_read_134734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_134733, 'initialize_read')
        # Calling initialize_read(args, kwargs) (line 411)
        initialize_read_call_result_134736 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), initialize_read_134734, *[], **kwargs_134735)
        
        
        # Assigning a List to a Name (line 412):
        
        # Assigning a List to a Name (line 412):
        
        # Obtaining an instance of the builtin type 'list' (line 412)
        list_134737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 412)
        
        # Assigning a type to the variable 'vars' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'vars', list_134737)
        
        
        
        # Call to end_of_stream(...): (line 413)
        # Processing the call keyword arguments (line 413)
        kwargs_134740 = {}
        # Getting the type of 'self' (line 413)
        self_134738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 18), 'self', False)
        # Obtaining the member 'end_of_stream' of a type (line 413)
        end_of_stream_134739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 18), self_134738, 'end_of_stream')
        # Calling end_of_stream(args, kwargs) (line 413)
        end_of_stream_call_result_134741 = invoke(stypy.reporting.localization.Localization(__file__, 413, 18), end_of_stream_134739, *[], **kwargs_134740)
        
        # Applying the 'not' unary operator (line 413)
        result_not__134742 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 14), 'not', end_of_stream_call_result_134741)
        
        # Testing the type of an if condition (line 413)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 8), result_not__134742)
        # SSA begins for while statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 414):
        
        # Assigning a Subscript to a Name (line 414):
        
        # Obtaining the type of the subscript
        int_134743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 12), 'int')
        
        # Call to read_var_header(...): (line 414)
        # Processing the call keyword arguments (line 414)
        kwargs_134746 = {}
        # Getting the type of 'self' (line 414)
        self_134744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 414)
        read_var_header_134745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 33), self_134744, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 414)
        read_var_header_call_result_134747 = invoke(stypy.reporting.localization.Localization(__file__, 414, 33), read_var_header_134745, *[], **kwargs_134746)
        
        # Obtaining the member '__getitem__' of a type (line 414)
        getitem___134748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), read_var_header_call_result_134747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 414)
        subscript_call_result_134749 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), getitem___134748, int_134743)
        
        # Assigning a type to the variable 'tuple_var_assignment_133765' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'tuple_var_assignment_133765', subscript_call_result_134749)
        
        # Assigning a Subscript to a Name (line 414):
        
        # Obtaining the type of the subscript
        int_134750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 12), 'int')
        
        # Call to read_var_header(...): (line 414)
        # Processing the call keyword arguments (line 414)
        kwargs_134753 = {}
        # Getting the type of 'self' (line 414)
        self_134751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 33), 'self', False)
        # Obtaining the member 'read_var_header' of a type (line 414)
        read_var_header_134752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 33), self_134751, 'read_var_header')
        # Calling read_var_header(args, kwargs) (line 414)
        read_var_header_call_result_134754 = invoke(stypy.reporting.localization.Localization(__file__, 414, 33), read_var_header_134752, *[], **kwargs_134753)
        
        # Obtaining the member '__getitem__' of a type (line 414)
        getitem___134755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), read_var_header_call_result_134754, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 414)
        subscript_call_result_134756 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), getitem___134755, int_134750)
        
        # Assigning a type to the variable 'tuple_var_assignment_133766' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'tuple_var_assignment_133766', subscript_call_result_134756)
        
        # Assigning a Name to a Name (line 414):
        # Getting the type of 'tuple_var_assignment_133765' (line 414)
        tuple_var_assignment_133765_134757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'tuple_var_assignment_133765')
        # Assigning a type to the variable 'hdr' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'hdr', tuple_var_assignment_133765_134757)
        
        # Assigning a Name to a Name (line 414):
        # Getting the type of 'tuple_var_assignment_133766' (line 414)
        tuple_var_assignment_133766_134758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'tuple_var_assignment_133766')
        # Assigning a type to the variable 'next_position' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 17), 'next_position', tuple_var_assignment_133766_134758)
        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to asstr(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'hdr' (line 415)
        hdr_134760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'hdr', False)
        # Obtaining the member 'name' of a type (line 415)
        name_134761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 25), hdr_134760, 'name')
        # Processing the call keyword arguments (line 415)
        kwargs_134762 = {}
        # Getting the type of 'asstr' (line 415)
        asstr_134759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 19), 'asstr', False)
        # Calling asstr(args, kwargs) (line 415)
        asstr_call_result_134763 = invoke(stypy.reporting.localization.Localization(__file__, 415, 19), asstr_134759, *[name_134761], **kwargs_134762)
        
        # Assigning a type to the variable 'name' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'name', asstr_call_result_134763)
        
        # Assigning a Call to a Name (line 416):
        
        # Assigning a Call to a Name (line 416):
        
        # Call to shape_from_header(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'hdr' (line 416)
        hdr_134767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 58), 'hdr', False)
        # Processing the call keyword arguments (line 416)
        kwargs_134768 = {}
        # Getting the type of 'self' (line 416)
        self_134764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'self', False)
        # Obtaining the member '_matrix_reader' of a type (line 416)
        _matrix_reader_134765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 20), self_134764, '_matrix_reader')
        # Obtaining the member 'shape_from_header' of a type (line 416)
        shape_from_header_134766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 20), _matrix_reader_134765, 'shape_from_header')
        # Calling shape_from_header(args, kwargs) (line 416)
        shape_from_header_call_result_134769 = invoke(stypy.reporting.localization.Localization(__file__, 416, 20), shape_from_header_134766, *[hdr_134767], **kwargs_134768)
        
        # Assigning a type to the variable 'shape' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'shape', shape_from_header_call_result_134769)
        
        # Assigning a Call to a Name (line 417):
        
        # Assigning a Call to a Name (line 417):
        
        # Call to get(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'hdr' (line 417)
        hdr_134772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 35), 'hdr', False)
        # Obtaining the member 'mclass' of a type (line 417)
        mclass_134773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 35), hdr_134772, 'mclass')
        str_134774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 47), 'str', 'unknown')
        # Processing the call keyword arguments (line 417)
        kwargs_134775 = {}
        # Getting the type of 'mclass_info' (line 417)
        mclass_info_134770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 'mclass_info', False)
        # Obtaining the member 'get' of a type (line 417)
        get_134771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 19), mclass_info_134770, 'get')
        # Calling get(args, kwargs) (line 417)
        get_call_result_134776 = invoke(stypy.reporting.localization.Localization(__file__, 417, 19), get_134771, *[mclass_134773, str_134774], **kwargs_134775)
        
        # Assigning a type to the variable 'info' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'info', get_call_result_134776)
        
        # Call to append(...): (line 418)
        # Processing the call arguments (line 418)
        
        # Obtaining an instance of the builtin type 'tuple' (line 418)
        tuple_134779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 418)
        # Adding element type (line 418)
        # Getting the type of 'name' (line 418)
        name_134780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 25), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 25), tuple_134779, name_134780)
        # Adding element type (line 418)
        # Getting the type of 'shape' (line 418)
        shape_134781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 31), 'shape', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 25), tuple_134779, shape_134781)
        # Adding element type (line 418)
        # Getting the type of 'info' (line 418)
        info_134782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 38), 'info', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 25), tuple_134779, info_134782)
        
        # Processing the call keyword arguments (line 418)
        kwargs_134783 = {}
        # Getting the type of 'vars' (line 418)
        vars_134777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'vars', False)
        # Obtaining the member 'append' of a type (line 418)
        append_134778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 12), vars_134777, 'append')
        # Calling append(args, kwargs) (line 418)
        append_call_result_134784 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), append_134778, *[tuple_134779], **kwargs_134783)
        
        
        # Call to seek(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'next_position' (line 420)
        next_position_134788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 33), 'next_position', False)
        # Processing the call keyword arguments (line 420)
        kwargs_134789 = {}
        # Getting the type of 'self' (line 420)
        self_134785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 420)
        mat_stream_134786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 12), self_134785, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 420)
        seek_134787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 12), mat_stream_134786, 'seek')
        # Calling seek(args, kwargs) (line 420)
        seek_call_result_134790 = invoke(stypy.reporting.localization.Localization(__file__, 420, 12), seek_134787, *[next_position_134788], **kwargs_134789)
        
        # SSA join for while statement (line 413)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vars' (line 421)
        vars_134791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'vars')
        # Assigning a type to the variable 'stypy_return_type' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'stypy_return_type', vars_134791)
        
        # ################# End of 'list_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'list_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 407)
        stypy_return_type_134792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'list_variables'
        return stypy_return_type_134792


# Assigning a type to the variable 'MatFile4Reader' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'MatFile4Reader', MatFile4Reader)

@norecursion
def arr_to_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_134793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 27), 'str', 'row')
    defaults = [str_134793]
    # Create a new context for function 'arr_to_2d'
    module_type_store = module_type_store.open_function_context('arr_to_2d', 424, 0, False)
    
    # Passed parameters checking function
    arr_to_2d.stypy_localization = localization
    arr_to_2d.stypy_type_of_self = None
    arr_to_2d.stypy_type_store = module_type_store
    arr_to_2d.stypy_function_name = 'arr_to_2d'
    arr_to_2d.stypy_param_names_list = ['arr', 'oned_as']
    arr_to_2d.stypy_varargs_param_name = None
    arr_to_2d.stypy_kwargs_param_name = None
    arr_to_2d.stypy_call_defaults = defaults
    arr_to_2d.stypy_call_varargs = varargs
    arr_to_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arr_to_2d', ['arr', 'oned_as'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arr_to_2d', localization, ['arr', 'oned_as'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arr_to_2d(...)' code ##################

    str_134794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, (-1)), 'str', " Make ``arr`` exactly two dimensional\n\n    If `arr` has more than 2 dimensions, raise a ValueError\n\n    Parameters\n    ----------\n    arr : array\n    oned_as : {'row', 'column'}, optional\n       Whether to reshape 1D vectors as row vectors or column vectors.\n       See documentation for ``matdims`` for more detail\n\n    Returns\n    -------\n    arr2d : array\n       2D version of the array\n    ")
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to matdims(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'arr' (line 441)
    arr_134796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 19), 'arr', False)
    # Getting the type of 'oned_as' (line 441)
    oned_as_134797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 24), 'oned_as', False)
    # Processing the call keyword arguments (line 441)
    kwargs_134798 = {}
    # Getting the type of 'matdims' (line 441)
    matdims_134795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'matdims', False)
    # Calling matdims(args, kwargs) (line 441)
    matdims_call_result_134799 = invoke(stypy.reporting.localization.Localization(__file__, 441, 11), matdims_134795, *[arr_134796, oned_as_134797], **kwargs_134798)
    
    # Assigning a type to the variable 'dims' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'dims', matdims_call_result_134799)
    
    
    
    # Call to len(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'dims' (line 442)
    dims_134801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), 'dims', False)
    # Processing the call keyword arguments (line 442)
    kwargs_134802 = {}
    # Getting the type of 'len' (line 442)
    len_134800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 7), 'len', False)
    # Calling len(args, kwargs) (line 442)
    len_call_result_134803 = invoke(stypy.reporting.localization.Localization(__file__, 442, 7), len_134800, *[dims_134801], **kwargs_134802)
    
    int_134804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'int')
    # Applying the binary operator '>' (line 442)
    result_gt_134805 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 7), '>', len_call_result_134803, int_134804)
    
    # Testing the type of an if condition (line 442)
    if_condition_134806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 4), result_gt_134805)
    # Assigning a type to the variable 'if_condition_134806' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'if_condition_134806', if_condition_134806)
    # SSA begins for if statement (line 442)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 443)
    # Processing the call arguments (line 443)
    str_134808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 25), 'str', 'Matlab 4 files cannot save arrays with more than 2 dimensions')
    # Processing the call keyword arguments (line 443)
    kwargs_134809 = {}
    # Getting the type of 'ValueError' (line 443)
    ValueError_134807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 443)
    ValueError_call_result_134810 = invoke(stypy.reporting.localization.Localization(__file__, 443, 14), ValueError_134807, *[str_134808], **kwargs_134809)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 443, 8), ValueError_call_result_134810, 'raise parameter', BaseException)
    # SSA join for if statement (line 442)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reshape(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'dims' (line 445)
    dims_134813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 23), 'dims', False)
    # Processing the call keyword arguments (line 445)
    kwargs_134814 = {}
    # Getting the type of 'arr' (line 445)
    arr_134811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'arr', False)
    # Obtaining the member 'reshape' of a type (line 445)
    reshape_134812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 11), arr_134811, 'reshape')
    # Calling reshape(args, kwargs) (line 445)
    reshape_call_result_134815 = invoke(stypy.reporting.localization.Localization(__file__, 445, 11), reshape_134812, *[dims_134813], **kwargs_134814)
    
    # Assigning a type to the variable 'stypy_return_type' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type', reshape_call_result_134815)
    
    # ################# End of 'arr_to_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arr_to_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 424)
    stypy_return_type_134816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_134816)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arr_to_2d'
    return stypy_return_type_134816

# Assigning a type to the variable 'arr_to_2d' (line 424)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 0), 'arr_to_2d', arr_to_2d)
# Declaration of the 'VarWriter4' class

class VarWriter4(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 449, 4, False)
        # Assigning a type to the variable 'self' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.__init__', ['file_writer'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 450):
        
        # Assigning a Attribute to a Attribute (line 450):
        # Getting the type of 'file_writer' (line 450)
        file_writer_134817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'file_writer')
        # Obtaining the member 'file_stream' of a type (line 450)
        file_stream_134818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 27), file_writer_134817, 'file_stream')
        # Getting the type of 'self' (line 450)
        self_134819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'self')
        # Setting the type of the member 'file_stream' of a type (line 450)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), self_134819, 'file_stream', file_stream_134818)
        
        # Assigning a Attribute to a Attribute (line 451):
        
        # Assigning a Attribute to a Attribute (line 451):
        # Getting the type of 'file_writer' (line 451)
        file_writer_134820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'file_writer')
        # Obtaining the member 'oned_as' of a type (line 451)
        oned_as_134821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 23), file_writer_134820, 'oned_as')
        # Getting the type of 'self' (line 451)
        self_134822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'self')
        # Setting the type of the member 'oned_as' of a type (line 451)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), self_134822, 'oned_as', oned_as_134821)
        
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
        module_type_store = module_type_store.open_function_context('write_bytes', 453, 4, False)
        # Assigning a type to the variable 'self' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_localization', localization)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_function_name', 'VarWriter4.write_bytes')
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_param_names_list', ['arr'])
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter4.write_bytes.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.write_bytes', ['arr'], None, None, defaults, varargs, kwargs)

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

        
        # Call to write(...): (line 454)
        # Processing the call arguments (line 454)
        
        # Call to tostring(...): (line 454)
        # Processing the call keyword arguments (line 454)
        str_134828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 50), 'str', 'F')
        keyword_134829 = str_134828
        kwargs_134830 = {'order': keyword_134829}
        # Getting the type of 'arr' (line 454)
        arr_134826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 31), 'arr', False)
        # Obtaining the member 'tostring' of a type (line 454)
        tostring_134827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 31), arr_134826, 'tostring')
        # Calling tostring(args, kwargs) (line 454)
        tostring_call_result_134831 = invoke(stypy.reporting.localization.Localization(__file__, 454, 31), tostring_134827, *[], **kwargs_134830)
        
        # Processing the call keyword arguments (line 454)
        kwargs_134832 = {}
        # Getting the type of 'self' (line 454)
        self_134823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 454)
        file_stream_134824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_134823, 'file_stream')
        # Obtaining the member 'write' of a type (line 454)
        write_134825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), file_stream_134824, 'write')
        # Calling write(args, kwargs) (line 454)
        write_call_result_134833 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), write_134825, *[tostring_call_result_134831], **kwargs_134832)
        
        
        # ################# End of 'write_bytes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_bytes' in the type store
        # Getting the type of 'stypy_return_type' (line 453)
        stypy_return_type_134834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_bytes'
        return stypy_return_type_134834


    @norecursion
    def write_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_string'
        module_type_store = module_type_store.open_function_context('write_string', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter4.write_string.__dict__.__setitem__('stypy_localization', localization)
        VarWriter4.write_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter4.write_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter4.write_string.__dict__.__setitem__('stypy_function_name', 'VarWriter4.write_string')
        VarWriter4.write_string.__dict__.__setitem__('stypy_param_names_list', ['s'])
        VarWriter4.write_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter4.write_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter4.write_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter4.write_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter4.write_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter4.write_string.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.write_string', ['s'], None, None, defaults, varargs, kwargs)

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

        
        # Call to write(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 's' (line 457)
        s_134838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 31), 's', False)
        # Processing the call keyword arguments (line 457)
        kwargs_134839 = {}
        # Getting the type of 'self' (line 457)
        self_134835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'self', False)
        # Obtaining the member 'file_stream' of a type (line 457)
        file_stream_134836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), self_134835, 'file_stream')
        # Obtaining the member 'write' of a type (line 457)
        write_134837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), file_stream_134836, 'write')
        # Calling write(args, kwargs) (line 457)
        write_call_result_134840 = invoke(stypy.reporting.localization.Localization(__file__, 457, 8), write_134837, *[s_134838], **kwargs_134839)
        
        
        # ################# End of 'write_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_string' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_134841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134841)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_string'
        return stypy_return_type_134841


    @norecursion
    def write_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'miDOUBLE' (line 459)
        miDOUBLE_134842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 42), 'miDOUBLE')
        # Getting the type of 'mxFULL_CLASS' (line 459)
        mxFULL_CLASS_134843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 54), 'mxFULL_CLASS')
        int_134844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 74), 'int')
        defaults = [miDOUBLE_134842, mxFULL_CLASS_134843, int_134844]
        # Create a new context for function 'write_header'
        module_type_store = module_type_store.open_function_context('write_header', 459, 4, False)
        # Assigning a type to the variable 'self' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter4.write_header.__dict__.__setitem__('stypy_localization', localization)
        VarWriter4.write_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter4.write_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter4.write_header.__dict__.__setitem__('stypy_function_name', 'VarWriter4.write_header')
        VarWriter4.write_header.__dict__.__setitem__('stypy_param_names_list', ['name', 'shape', 'P', 'T', 'imagf'])
        VarWriter4.write_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter4.write_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter4.write_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter4.write_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter4.write_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter4.write_header.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.write_header', ['name', 'shape', 'P', 'T', 'imagf'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_header', localization, ['name', 'shape', 'P', 'T', 'imagf'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_header(...)' code ##################

        str_134845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, (-1)), 'str', ' Write header for given data options\n\n        Parameters\n        ----------\n        name : str\n            name of variable\n        shape : sequence\n           Shape of array as it will be read in matlab\n        P : int, optional\n            code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,\n            miINT16, miUINT16, miUINT8``\n        T : int, optional\n            code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,\n            mxSPARSE_CLASS``\n        imagf : int, optional\n            flag indicating complex\n        ')
        
        # Assigning a Call to a Name (line 477):
        
        # Assigning a Call to a Name (line 477):
        
        # Call to empty(...): (line 477)
        # Processing the call arguments (line 477)
        
        # Obtaining an instance of the builtin type 'tuple' (line 477)
        tuple_134848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 477)
        
        
        # Obtaining the type of the subscript
        str_134849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 47), 'str', 'header')
        # Getting the type of 'mdtypes_template' (line 477)
        mdtypes_template_134850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 30), 'mdtypes_template', False)
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___134851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 30), mdtypes_template_134850, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_134852 = invoke(stypy.reporting.localization.Localization(__file__, 477, 30), getitem___134851, str_134849)
        
        # Processing the call keyword arguments (line 477)
        kwargs_134853 = {}
        # Getting the type of 'np' (line 477)
        np_134846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 477)
        empty_134847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), np_134846, 'empty')
        # Calling empty(args, kwargs) (line 477)
        empty_call_result_134854 = invoke(stypy.reporting.localization.Localization(__file__, 477, 17), empty_134847, *[tuple_134848, subscript_call_result_134852], **kwargs_134853)
        
        # Assigning a type to the variable 'header' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'header', empty_call_result_134854)
        
        # Assigning a UnaryOp to a Name (line 478):
        
        # Assigning a UnaryOp to a Name (line 478):
        
        # Getting the type of 'SYS_LITTLE_ENDIAN' (line 478)
        SYS_LITTLE_ENDIAN_134855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 16), 'SYS_LITTLE_ENDIAN')
        # Applying the 'not' unary operator (line 478)
        result_not__134856 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 12), 'not', SYS_LITTLE_ENDIAN_134855)
        
        # Assigning a type to the variable 'M' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'M', result_not__134856)
        
        # Assigning a Num to a Name (line 479):
        
        # Assigning a Num to a Name (line 479):
        int_134857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 12), 'int')
        # Assigning a type to the variable 'O' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'O', int_134857)
        
        # Assigning a BinOp to a Subscript (line 480):
        
        # Assigning a BinOp to a Subscript (line 480):
        # Getting the type of 'M' (line 480)
        M_134858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 26), 'M')
        int_134859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 30), 'int')
        # Applying the binary operator '*' (line 480)
        result_mul_134860 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 26), '*', M_134858, int_134859)
        
        # Getting the type of 'O' (line 481)
        O_134861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 26), 'O')
        int_134862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 30), 'int')
        # Applying the binary operator '*' (line 481)
        result_mul_134863 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 26), '*', O_134861, int_134862)
        
        # Applying the binary operator '+' (line 480)
        result_add_134864 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 26), '+', result_mul_134860, result_mul_134863)
        
        # Getting the type of 'P' (line 482)
        P_134865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 26), 'P')
        int_134866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 30), 'int')
        # Applying the binary operator '*' (line 482)
        result_mul_134867 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 26), '*', P_134865, int_134866)
        
        # Applying the binary operator '+' (line 481)
        result_add_134868 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 34), '+', result_add_134864, result_mul_134867)
        
        # Getting the type of 'T' (line 483)
        T_134869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 26), 'T')
        # Applying the binary operator '+' (line 482)
        result_add_134870 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 33), '+', result_add_134868, T_134869)
        
        # Getting the type of 'header' (line 480)
        header_134871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'header')
        str_134872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 15), 'str', 'mopt')
        # Storing an element on a container (line 480)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 8), header_134871, (str_134872, result_add_134870))
        
        # Assigning a Subscript to a Subscript (line 484):
        
        # Assigning a Subscript to a Subscript (line 484):
        
        # Obtaining the type of the subscript
        int_134873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 32), 'int')
        # Getting the type of 'shape' (line 484)
        shape_134874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 26), 'shape')
        # Obtaining the member '__getitem__' of a type (line 484)
        getitem___134875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 26), shape_134874, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 484)
        subscript_call_result_134876 = invoke(stypy.reporting.localization.Localization(__file__, 484, 26), getitem___134875, int_134873)
        
        # Getting the type of 'header' (line 484)
        header_134877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'header')
        str_134878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 15), 'str', 'mrows')
        # Storing an element on a container (line 484)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 8), header_134877, (str_134878, subscript_call_result_134876))
        
        # Assigning a Subscript to a Subscript (line 485):
        
        # Assigning a Subscript to a Subscript (line 485):
        
        # Obtaining the type of the subscript
        int_134879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 32), 'int')
        # Getting the type of 'shape' (line 485)
        shape_134880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 26), 'shape')
        # Obtaining the member '__getitem__' of a type (line 485)
        getitem___134881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 26), shape_134880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 485)
        subscript_call_result_134882 = invoke(stypy.reporting.localization.Localization(__file__, 485, 26), getitem___134881, int_134879)
        
        # Getting the type of 'header' (line 485)
        header_134883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'header')
        str_134884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 15), 'str', 'ncols')
        # Storing an element on a container (line 485)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 8), header_134883, (str_134884, subscript_call_result_134882))
        
        # Assigning a Name to a Subscript (line 486):
        
        # Assigning a Name to a Subscript (line 486):
        # Getting the type of 'imagf' (line 486)
        imagf_134885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 26), 'imagf')
        # Getting the type of 'header' (line 486)
        header_134886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'header')
        str_134887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 15), 'str', 'imagf')
        # Storing an element on a container (line 486)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 486, 8), header_134886, (str_134887, imagf_134885))
        
        # Assigning a BinOp to a Subscript (line 487):
        
        # Assigning a BinOp to a Subscript (line 487):
        
        # Call to len(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'name' (line 487)
        name_134889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 31), 'name', False)
        # Processing the call keyword arguments (line 487)
        kwargs_134890 = {}
        # Getting the type of 'len' (line 487)
        len_134888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 27), 'len', False)
        # Calling len(args, kwargs) (line 487)
        len_call_result_134891 = invoke(stypy.reporting.localization.Localization(__file__, 487, 27), len_134888, *[name_134889], **kwargs_134890)
        
        int_134892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 39), 'int')
        # Applying the binary operator '+' (line 487)
        result_add_134893 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 27), '+', len_call_result_134891, int_134892)
        
        # Getting the type of 'header' (line 487)
        header_134894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'header')
        str_134895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 15), 'str', 'namlen')
        # Storing an element on a container (line 487)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 8), header_134894, (str_134895, result_add_134893))
        
        # Call to write_bytes(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'header' (line 488)
        header_134898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 25), 'header', False)
        # Processing the call keyword arguments (line 488)
        kwargs_134899 = {}
        # Getting the type of 'self' (line 488)
        self_134896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 488)
        write_bytes_134897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), self_134896, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 488)
        write_bytes_call_result_134900 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), write_bytes_134897, *[header_134898], **kwargs_134899)
        
        
        # Call to write_string(...): (line 489)
        # Processing the call arguments (line 489)
        
        # Call to asbytes(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'name' (line 489)
        name_134904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 34), 'name', False)
        str_134905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 41), 'str', '\x00')
        # Applying the binary operator '+' (line 489)
        result_add_134906 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 34), '+', name_134904, str_134905)
        
        # Processing the call keyword arguments (line 489)
        kwargs_134907 = {}
        # Getting the type of 'asbytes' (line 489)
        asbytes_134903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 26), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 489)
        asbytes_call_result_134908 = invoke(stypy.reporting.localization.Localization(__file__, 489, 26), asbytes_134903, *[result_add_134906], **kwargs_134907)
        
        # Processing the call keyword arguments (line 489)
        kwargs_134909 = {}
        # Getting the type of 'self' (line 489)
        self_134901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'self', False)
        # Obtaining the member 'write_string' of a type (line 489)
        write_string_134902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), self_134901, 'write_string')
        # Calling write_string(args, kwargs) (line 489)
        write_string_call_result_134910 = invoke(stypy.reporting.localization.Localization(__file__, 489, 8), write_string_134902, *[asbytes_call_result_134908], **kwargs_134909)
        
        
        # ################# End of 'write_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_header' in the type store
        # Getting the type of 'stypy_return_type' (line 459)
        stypy_return_type_134911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_header'
        return stypy_return_type_134911


    @norecursion
    def write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 491, 4, False)
        # Assigning a type to the variable 'self' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter4.write.__dict__.__setitem__('stypy_localization', localization)
        VarWriter4.write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter4.write.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter4.write.__dict__.__setitem__('stypy_function_name', 'VarWriter4.write')
        VarWriter4.write.__dict__.__setitem__('stypy_param_names_list', ['arr', 'name'])
        VarWriter4.write.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter4.write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter4.write.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter4.write.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter4.write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter4.write.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.write', ['arr', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write', localization, ['arr', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write(...)' code ##################

        str_134912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, (-1)), 'str', ' Write matrix `arr`, with name `name`\n\n        Parameters\n        ----------\n        arr : array_like\n           array to write\n        name : str\n           name in matlab workspace\n        ')
        
        
        # Call to issparse(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'arr' (line 503)
        arr_134916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 33), 'arr', False)
        # Processing the call keyword arguments (line 503)
        kwargs_134917 = {}
        # Getting the type of 'scipy' (line 503)
        scipy_134913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 503)
        sparse_134914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 11), scipy_134913, 'sparse')
        # Obtaining the member 'issparse' of a type (line 503)
        issparse_134915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 11), sparse_134914, 'issparse')
        # Calling issparse(args, kwargs) (line 503)
        issparse_call_result_134918 = invoke(stypy.reporting.localization.Localization(__file__, 503, 11), issparse_134915, *[arr_134916], **kwargs_134917)
        
        # Testing the type of an if condition (line 503)
        if_condition_134919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 503, 8), issparse_call_result_134918)
        # Assigning a type to the variable 'if_condition_134919' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'if_condition_134919', if_condition_134919)
        # SSA begins for if statement (line 503)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_sparse(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'arr' (line 504)
        arr_134922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 30), 'arr', False)
        # Getting the type of 'name' (line 504)
        name_134923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 35), 'name', False)
        # Processing the call keyword arguments (line 504)
        kwargs_134924 = {}
        # Getting the type of 'self' (line 504)
        self_134920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'self', False)
        # Obtaining the member 'write_sparse' of a type (line 504)
        write_sparse_134921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), self_134920, 'write_sparse')
        # Calling write_sparse(args, kwargs) (line 504)
        write_sparse_call_result_134925 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), write_sparse_134921, *[arr_134922, name_134923], **kwargs_134924)
        
        # Assigning a type to the variable 'stypy_return_type' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 503)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 506):
        
        # Assigning a Call to a Name (line 506):
        
        # Call to asarray(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'arr' (line 506)
        arr_134928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 25), 'arr', False)
        # Processing the call keyword arguments (line 506)
        kwargs_134929 = {}
        # Getting the type of 'np' (line 506)
        np_134926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 506)
        asarray_134927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 14), np_134926, 'asarray')
        # Calling asarray(args, kwargs) (line 506)
        asarray_call_result_134930 = invoke(stypy.reporting.localization.Localization(__file__, 506, 14), asarray_134927, *[arr_134928], **kwargs_134929)
        
        # Assigning a type to the variable 'arr' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'arr', asarray_call_result_134930)
        
        # Assigning a Attribute to a Name (line 507):
        
        # Assigning a Attribute to a Name (line 507):
        # Getting the type of 'arr' (line 507)
        arr_134931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 13), 'arr')
        # Obtaining the member 'dtype' of a type (line 507)
        dtype_134932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 13), arr_134931, 'dtype')
        # Assigning a type to the variable 'dt' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'dt', dtype_134932)
        
        
        # Getting the type of 'dt' (line 508)
        dt_134933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'dt')
        # Obtaining the member 'isnative' of a type (line 508)
        isnative_134934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 15), dt_134933, 'isnative')
        # Applying the 'not' unary operator (line 508)
        result_not__134935 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 11), 'not', isnative_134934)
        
        # Testing the type of an if condition (line 508)
        if_condition_134936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 508, 8), result_not__134935)
        # Assigning a type to the variable 'if_condition_134936' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'if_condition_134936', if_condition_134936)
        # SSA begins for if statement (line 508)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to astype(...): (line 509)
        # Processing the call arguments (line 509)
        
        # Call to newbyteorder(...): (line 509)
        # Processing the call arguments (line 509)
        str_134941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 45), 'str', '=')
        # Processing the call keyword arguments (line 509)
        kwargs_134942 = {}
        # Getting the type of 'dt' (line 509)
        dt_134939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'dt', False)
        # Obtaining the member 'newbyteorder' of a type (line 509)
        newbyteorder_134940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 29), dt_134939, 'newbyteorder')
        # Calling newbyteorder(args, kwargs) (line 509)
        newbyteorder_call_result_134943 = invoke(stypy.reporting.localization.Localization(__file__, 509, 29), newbyteorder_134940, *[str_134941], **kwargs_134942)
        
        # Processing the call keyword arguments (line 509)
        kwargs_134944 = {}
        # Getting the type of 'arr' (line 509)
        arr_134937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 18), 'arr', False)
        # Obtaining the member 'astype' of a type (line 509)
        astype_134938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 18), arr_134937, 'astype')
        # Calling astype(args, kwargs) (line 509)
        astype_call_result_134945 = invoke(stypy.reporting.localization.Localization(__file__, 509, 18), astype_134938, *[newbyteorder_call_result_134943], **kwargs_134944)
        
        # Assigning a type to the variable 'arr' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'arr', astype_call_result_134945)
        # SSA join for if statement (line 508)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 510):
        
        # Assigning a Attribute to a Name (line 510):
        # Getting the type of 'dt' (line 510)
        dt_134946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 14), 'dt')
        # Obtaining the member 'type' of a type (line 510)
        type_134947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 14), dt_134946, 'type')
        # Assigning a type to the variable 'dtt' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'dtt', type_134947)
        
        
        # Getting the type of 'dtt' (line 511)
        dtt_134948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 11), 'dtt')
        # Getting the type of 'np' (line 511)
        np_134949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 18), 'np')
        # Obtaining the member 'object_' of a type (line 511)
        object__134950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 18), np_134949, 'object_')
        # Applying the binary operator 'is' (line 511)
        result_is__134951 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 11), 'is', dtt_134948, object__134950)
        
        # Testing the type of an if condition (line 511)
        if_condition_134952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 8), result_is__134951)
        # Assigning a type to the variable 'if_condition_134952' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'if_condition_134952', if_condition_134952)
        # SSA begins for if statement (line 511)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 512)
        # Processing the call arguments (line 512)
        str_134954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 28), 'str', 'Cannot save object arrays in Mat4')
        # Processing the call keyword arguments (line 512)
        kwargs_134955 = {}
        # Getting the type of 'TypeError' (line 512)
        TypeError_134953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 512)
        TypeError_call_result_134956 = invoke(stypy.reporting.localization.Localization(__file__, 512, 18), TypeError_134953, *[str_134954], **kwargs_134955)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 512, 12), TypeError_call_result_134956, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 511)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'dtt' (line 513)
        dtt_134957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 13), 'dtt')
        # Getting the type of 'np' (line 513)
        np_134958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'np')
        # Obtaining the member 'void' of a type (line 513)
        void_134959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 20), np_134958, 'void')
        # Applying the binary operator 'is' (line 513)
        result_is__134960 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 13), 'is', dtt_134957, void_134959)
        
        # Testing the type of an if condition (line 513)
        if_condition_134961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 13), result_is__134960)
        # Assigning a type to the variable 'if_condition_134961' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 13), 'if_condition_134961', if_condition_134961)
        # SSA begins for if statement (line 513)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 514)
        # Processing the call arguments (line 514)
        str_134963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 28), 'str', 'Cannot save void type arrays')
        # Processing the call keyword arguments (line 514)
        kwargs_134964 = {}
        # Getting the type of 'TypeError' (line 514)
        TypeError_134962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 514)
        TypeError_call_result_134965 = invoke(stypy.reporting.localization.Localization(__file__, 514, 18), TypeError_134962, *[str_134963], **kwargs_134964)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 514, 12), TypeError_call_result_134965, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 513)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'dtt' (line 515)
        dtt_134966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 13), 'dtt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 515)
        tuple_134967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 515)
        # Adding element type (line 515)
        # Getting the type of 'np' (line 515)
        np_134968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 21), 'np')
        # Obtaining the member 'unicode_' of a type (line 515)
        unicode__134969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 21), np_134968, 'unicode_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 21), tuple_134967, unicode__134969)
        # Adding element type (line 515)
        # Getting the type of 'np' (line 515)
        np_134970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 34), 'np')
        # Obtaining the member 'string_' of a type (line 515)
        string__134971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 34), np_134970, 'string_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 21), tuple_134967, string__134971)
        
        # Applying the binary operator 'in' (line 515)
        result_contains_134972 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 13), 'in', dtt_134966, tuple_134967)
        
        # Testing the type of an if condition (line 515)
        if_condition_134973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 13), result_contains_134972)
        # Assigning a type to the variable 'if_condition_134973' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 13), 'if_condition_134973', if_condition_134973)
        # SSA begins for if statement (line 515)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_char(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'arr' (line 516)
        arr_134976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 28), 'arr', False)
        # Getting the type of 'name' (line 516)
        name_134977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 33), 'name', False)
        # Processing the call keyword arguments (line 516)
        kwargs_134978 = {}
        # Getting the type of 'self' (line 516)
        self_134974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'self', False)
        # Obtaining the member 'write_char' of a type (line 516)
        write_char_134975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), self_134974, 'write_char')
        # Calling write_char(args, kwargs) (line 516)
        write_char_call_result_134979 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), write_char_134975, *[arr_134976, name_134977], **kwargs_134978)
        
        # Assigning a type to the variable 'stypy_return_type' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 515)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 513)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 511)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_numeric(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'arr' (line 518)
        arr_134982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 27), 'arr', False)
        # Getting the type of 'name' (line 518)
        name_134983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 32), 'name', False)
        # Processing the call keyword arguments (line 518)
        kwargs_134984 = {}
        # Getting the type of 'self' (line 518)
        self_134980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'self', False)
        # Obtaining the member 'write_numeric' of a type (line 518)
        write_numeric_134981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), self_134980, 'write_numeric')
        # Calling write_numeric(args, kwargs) (line 518)
        write_numeric_call_result_134985 = invoke(stypy.reporting.localization.Localization(__file__, 518, 8), write_numeric_134981, *[arr_134982, name_134983], **kwargs_134984)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 491)
        stypy_return_type_134986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134986)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_134986


    @norecursion
    def write_numeric(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_numeric'
        module_type_store = module_type_store.open_function_context('write_numeric', 520, 4, False)
        # Assigning a type to the variable 'self' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_localization', localization)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_function_name', 'VarWriter4.write_numeric')
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_param_names_list', ['arr', 'name'])
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter4.write_numeric.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.write_numeric', ['arr', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_numeric', localization, ['arr', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_numeric(...)' code ##################

        
        # Assigning a Call to a Name (line 521):
        
        # Assigning a Call to a Name (line 521):
        
        # Call to arr_to_2d(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'arr' (line 521)
        arr_134988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 'arr', False)
        # Getting the type of 'self' (line 521)
        self_134989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 29), 'self', False)
        # Obtaining the member 'oned_as' of a type (line 521)
        oned_as_134990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 29), self_134989, 'oned_as')
        # Processing the call keyword arguments (line 521)
        kwargs_134991 = {}
        # Getting the type of 'arr_to_2d' (line 521)
        arr_to_2d_134987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 14), 'arr_to_2d', False)
        # Calling arr_to_2d(args, kwargs) (line 521)
        arr_to_2d_call_result_134992 = invoke(stypy.reporting.localization.Localization(__file__, 521, 14), arr_to_2d_134987, *[arr_134988, oned_as_134990], **kwargs_134991)
        
        # Assigning a type to the variable 'arr' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'arr', arr_to_2d_call_result_134992)
        
        # Assigning a Compare to a Name (line 522):
        
        # Assigning a Compare to a Name (line 522):
        
        # Getting the type of 'arr' (line 522)
        arr_134993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), 'arr')
        # Obtaining the member 'dtype' of a type (line 522)
        dtype_134994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 16), arr_134993, 'dtype')
        # Obtaining the member 'kind' of a type (line 522)
        kind_134995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 16), dtype_134994, 'kind')
        str_134996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 34), 'str', 'c')
        # Applying the binary operator '==' (line 522)
        result_eq_134997 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 16), '==', kind_134995, str_134996)
        
        # Assigning a type to the variable 'imagf' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'imagf', result_eq_134997)
        
        
        # SSA begins for try-except statement (line 523)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 524):
        
        # Assigning a Subscript to a Name (line 524):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_134998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 43), 'int')
        slice_134999 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 524, 29), int_134998, None, None)
        # Getting the type of 'arr' (line 524)
        arr_135000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 29), 'arr')
        # Obtaining the member 'dtype' of a type (line 524)
        dtype_135001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 29), arr_135000, 'dtype')
        # Obtaining the member 'str' of a type (line 524)
        str_135002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 29), dtype_135001, 'str')
        # Obtaining the member '__getitem__' of a type (line 524)
        getitem___135003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 29), str_135002, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 524)
        subscript_call_result_135004 = invoke(stypy.reporting.localization.Localization(__file__, 524, 29), getitem___135003, slice_134999)
        
        # Getting the type of 'np_to_mtypes' (line 524)
        np_to_mtypes_135005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'np_to_mtypes')
        # Obtaining the member '__getitem__' of a type (line 524)
        getitem___135006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 16), np_to_mtypes_135005, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 524)
        subscript_call_result_135007 = invoke(stypy.reporting.localization.Localization(__file__, 524, 16), getitem___135006, subscript_call_result_135004)
        
        # Assigning a type to the variable 'P' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'P', subscript_call_result_135007)
        # SSA branch for the except part of a try statement (line 523)
        # SSA branch for the except 'KeyError' branch of a try statement (line 523)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'imagf' (line 526)
        imagf_135008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'imagf')
        # Testing the type of an if condition (line 526)
        if_condition_135009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 526, 12), imagf_135008)
        # Assigning a type to the variable 'if_condition_135009' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'if_condition_135009', if_condition_135009)
        # SSA begins for if statement (line 526)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 527):
        
        # Assigning a Call to a Name (line 527):
        
        # Call to astype(...): (line 527)
        # Processing the call arguments (line 527)
        str_135012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 33), 'str', 'c128')
        # Processing the call keyword arguments (line 527)
        kwargs_135013 = {}
        # Getting the type of 'arr' (line 527)
        arr_135010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 22), 'arr', False)
        # Obtaining the member 'astype' of a type (line 527)
        astype_135011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 22), arr_135010, 'astype')
        # Calling astype(args, kwargs) (line 527)
        astype_call_result_135014 = invoke(stypy.reporting.localization.Localization(__file__, 527, 22), astype_135011, *[str_135012], **kwargs_135013)
        
        # Assigning a type to the variable 'arr' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 16), 'arr', astype_call_result_135014)
        # SSA branch for the else part of an if statement (line 526)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 529):
        
        # Assigning a Call to a Name (line 529):
        
        # Call to astype(...): (line 529)
        # Processing the call arguments (line 529)
        str_135017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 33), 'str', 'f8')
        # Processing the call keyword arguments (line 529)
        kwargs_135018 = {}
        # Getting the type of 'arr' (line 529)
        arr_135015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 22), 'arr', False)
        # Obtaining the member 'astype' of a type (line 529)
        astype_135016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 22), arr_135015, 'astype')
        # Calling astype(args, kwargs) (line 529)
        astype_call_result_135019 = invoke(stypy.reporting.localization.Localization(__file__, 529, 22), astype_135016, *[str_135017], **kwargs_135018)
        
        # Assigning a type to the variable 'arr' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 16), 'arr', astype_call_result_135019)
        # SSA join for if statement (line 526)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 530):
        
        # Assigning a Name to a Name (line 530):
        # Getting the type of 'miDOUBLE' (line 530)
        miDOUBLE_135020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'miDOUBLE')
        # Assigning a type to the variable 'P' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'P', miDOUBLE_135020)
        # SSA join for try-except statement (line 523)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_header(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'name' (line 531)
        name_135023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 26), 'name', False)
        # Getting the type of 'arr' (line 532)
        arr_135024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 26), 'arr', False)
        # Obtaining the member 'shape' of a type (line 532)
        shape_135025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 26), arr_135024, 'shape')
        # Processing the call keyword arguments (line 531)
        # Getting the type of 'P' (line 533)
        P_135026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 28), 'P', False)
        keyword_135027 = P_135026
        # Getting the type of 'mxFULL_CLASS' (line 534)
        mxFULL_CLASS_135028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 28), 'mxFULL_CLASS', False)
        keyword_135029 = mxFULL_CLASS_135028
        # Getting the type of 'imagf' (line 535)
        imagf_135030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 32), 'imagf', False)
        keyword_135031 = imagf_135030
        kwargs_135032 = {'P': keyword_135027, 'T': keyword_135029, 'imagf': keyword_135031}
        # Getting the type of 'self' (line 531)
        self_135021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 531)
        write_header_135022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), self_135021, 'write_header')
        # Calling write_header(args, kwargs) (line 531)
        write_header_call_result_135033 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), write_header_135022, *[name_135023, shape_135025], **kwargs_135032)
        
        
        # Getting the type of 'imagf' (line 536)
        imagf_135034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'imagf')
        # Testing the type of an if condition (line 536)
        if_condition_135035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), imagf_135034)
        # Assigning a type to the variable 'if_condition_135035' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_135035', if_condition_135035)
        # SSA begins for if statement (line 536)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_bytes(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'arr' (line 537)
        arr_135038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 29), 'arr', False)
        # Obtaining the member 'real' of a type (line 537)
        real_135039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 29), arr_135038, 'real')
        # Processing the call keyword arguments (line 537)
        kwargs_135040 = {}
        # Getting the type of 'self' (line 537)
        self_135036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 537)
        write_bytes_135037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), self_135036, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 537)
        write_bytes_call_result_135041 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), write_bytes_135037, *[real_135039], **kwargs_135040)
        
        
        # Call to write_bytes(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'arr' (line 538)
        arr_135044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 29), 'arr', False)
        # Obtaining the member 'imag' of a type (line 538)
        imag_135045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 29), arr_135044, 'imag')
        # Processing the call keyword arguments (line 538)
        kwargs_135046 = {}
        # Getting the type of 'self' (line 538)
        self_135042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 538)
        write_bytes_135043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), self_135042, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 538)
        write_bytes_call_result_135047 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), write_bytes_135043, *[imag_135045], **kwargs_135046)
        
        # SSA branch for the else part of an if statement (line 536)
        module_type_store.open_ssa_branch('else')
        
        # Call to write_bytes(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'arr' (line 540)
        arr_135050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 29), 'arr', False)
        # Processing the call keyword arguments (line 540)
        kwargs_135051 = {}
        # Getting the type of 'self' (line 540)
        self_135048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 540)
        write_bytes_135049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 12), self_135048, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 540)
        write_bytes_call_result_135052 = invoke(stypy.reporting.localization.Localization(__file__, 540, 12), write_bytes_135049, *[arr_135050], **kwargs_135051)
        
        # SSA join for if statement (line 536)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_numeric(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_numeric' in the type store
        # Getting the type of 'stypy_return_type' (line 520)
        stypy_return_type_135053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135053)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_numeric'
        return stypy_return_type_135053


    @norecursion
    def write_char(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_char'
        module_type_store = module_type_store.open_function_context('write_char', 542, 4, False)
        # Assigning a type to the variable 'self' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter4.write_char.__dict__.__setitem__('stypy_localization', localization)
        VarWriter4.write_char.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter4.write_char.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter4.write_char.__dict__.__setitem__('stypy_function_name', 'VarWriter4.write_char')
        VarWriter4.write_char.__dict__.__setitem__('stypy_param_names_list', ['arr', 'name'])
        VarWriter4.write_char.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter4.write_char.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter4.write_char.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter4.write_char.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter4.write_char.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter4.write_char.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.write_char', ['arr', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_char', localization, ['arr', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_char(...)' code ##################

        
        # Assigning a Call to a Name (line 543):
        
        # Assigning a Call to a Name (line 543):
        
        # Call to arr_to_chars(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'arr' (line 543)
        arr_135055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 27), 'arr', False)
        # Processing the call keyword arguments (line 543)
        kwargs_135056 = {}
        # Getting the type of 'arr_to_chars' (line 543)
        arr_to_chars_135054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 14), 'arr_to_chars', False)
        # Calling arr_to_chars(args, kwargs) (line 543)
        arr_to_chars_call_result_135057 = invoke(stypy.reporting.localization.Localization(__file__, 543, 14), arr_to_chars_135054, *[arr_135055], **kwargs_135056)
        
        # Assigning a type to the variable 'arr' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'arr', arr_to_chars_call_result_135057)
        
        # Assigning a Call to a Name (line 544):
        
        # Assigning a Call to a Name (line 544):
        
        # Call to arr_to_2d(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'arr' (line 544)
        arr_135059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 24), 'arr', False)
        # Getting the type of 'self' (line 544)
        self_135060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 29), 'self', False)
        # Obtaining the member 'oned_as' of a type (line 544)
        oned_as_135061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 29), self_135060, 'oned_as')
        # Processing the call keyword arguments (line 544)
        kwargs_135062 = {}
        # Getting the type of 'arr_to_2d' (line 544)
        arr_to_2d_135058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 14), 'arr_to_2d', False)
        # Calling arr_to_2d(args, kwargs) (line 544)
        arr_to_2d_call_result_135063 = invoke(stypy.reporting.localization.Localization(__file__, 544, 14), arr_to_2d_135058, *[arr_135059, oned_as_135061], **kwargs_135062)
        
        # Assigning a type to the variable 'arr' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'arr', arr_to_2d_call_result_135063)
        
        # Assigning a Attribute to a Name (line 545):
        
        # Assigning a Attribute to a Name (line 545):
        # Getting the type of 'arr' (line 545)
        arr_135064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 15), 'arr')
        # Obtaining the member 'shape' of a type (line 545)
        shape_135065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 15), arr_135064, 'shape')
        # Assigning a type to the variable 'dims' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'dims', shape_135065)
        
        # Call to write_header(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'name' (line 547)
        name_135068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'name', False)
        # Getting the type of 'dims' (line 548)
        dims_135069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'dims', False)
        # Processing the call keyword arguments (line 546)
        # Getting the type of 'miUINT8' (line 549)
        miUINT8_135070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 14), 'miUINT8', False)
        keyword_135071 = miUINT8_135070
        # Getting the type of 'mxCHAR_CLASS' (line 550)
        mxCHAR_CLASS_135072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 14), 'mxCHAR_CLASS', False)
        keyword_135073 = mxCHAR_CLASS_135072
        kwargs_135074 = {'P': keyword_135071, 'T': keyword_135073}
        # Getting the type of 'self' (line 546)
        self_135066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 546)
        write_header_135067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 8), self_135066, 'write_header')
        # Calling write_header(args, kwargs) (line 546)
        write_header_call_result_135075 = invoke(stypy.reporting.localization.Localization(__file__, 546, 8), write_header_135067, *[name_135068, dims_135069], **kwargs_135074)
        
        
        
        # Getting the type of 'arr' (line 551)
        arr_135076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'arr')
        # Obtaining the member 'dtype' of a type (line 551)
        dtype_135077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 11), arr_135076, 'dtype')
        # Obtaining the member 'kind' of a type (line 551)
        kind_135078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 11), dtype_135077, 'kind')
        str_135079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 29), 'str', 'U')
        # Applying the binary operator '==' (line 551)
        result_eq_135080 = python_operator(stypy.reporting.localization.Localization(__file__, 551, 11), '==', kind_135078, str_135079)
        
        # Testing the type of an if condition (line 551)
        if_condition_135081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 551, 8), result_eq_135080)
        # Assigning a type to the variable 'if_condition_135081' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'if_condition_135081', if_condition_135081)
        # SSA begins for if statement (line 551)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Call to product(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'dims' (line 553)
        dims_135084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 33), 'dims', False)
        # Processing the call keyword arguments (line 553)
        kwargs_135085 = {}
        # Getting the type of 'np' (line 553)
        np_135082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 22), 'np', False)
        # Obtaining the member 'product' of a type (line 553)
        product_135083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 22), np_135082, 'product')
        # Calling product(args, kwargs) (line 553)
        product_call_result_135086 = invoke(stypy.reporting.localization.Localization(__file__, 553, 22), product_135083, *[dims_135084], **kwargs_135085)
        
        # Assigning a type to the variable 'n_chars' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'n_chars', product_call_result_135086)
        
        # Assigning a Call to a Name (line 554):
        
        # Assigning a Call to a Name (line 554):
        
        # Call to ndarray(...): (line 554)
        # Processing the call keyword arguments (line 554)
        
        # Obtaining an instance of the builtin type 'tuple' (line 554)
        tuple_135089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 554)
        
        keyword_135090 = tuple_135089
        
        # Call to arr_dtype_number(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 'arr' (line 555)
        arr_135092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 55), 'arr', False)
        # Getting the type of 'n_chars' (line 555)
        n_chars_135093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 60), 'n_chars', False)
        # Processing the call keyword arguments (line 555)
        kwargs_135094 = {}
        # Getting the type of 'arr_dtype_number' (line 555)
        arr_dtype_number_135091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 38), 'arr_dtype_number', False)
        # Calling arr_dtype_number(args, kwargs) (line 555)
        arr_dtype_number_call_result_135095 = invoke(stypy.reporting.localization.Localization(__file__, 555, 38), arr_dtype_number_135091, *[arr_135092, n_chars_135093], **kwargs_135094)
        
        keyword_135096 = arr_dtype_number_call_result_135095
        # Getting the type of 'arr' (line 556)
        arr_135097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 39), 'arr', False)
        keyword_135098 = arr_135097
        kwargs_135099 = {'buffer': keyword_135098, 'dtype': keyword_135096, 'shape': keyword_135090}
        # Getting the type of 'np' (line 554)
        np_135087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 554)
        ndarray_135088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 21), np_135087, 'ndarray')
        # Calling ndarray(args, kwargs) (line 554)
        ndarray_call_result_135100 = invoke(stypy.reporting.localization.Localization(__file__, 554, 21), ndarray_135088, *[], **kwargs_135099)
        
        # Assigning a type to the variable 'st_arr' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'st_arr', ndarray_call_result_135100)
        
        # Assigning a Call to a Name (line 557):
        
        # Assigning a Call to a Name (line 557):
        
        # Call to encode(...): (line 557)
        # Processing the call arguments (line 557)
        str_135106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 38), 'str', 'latin-1')
        # Processing the call keyword arguments (line 557)
        kwargs_135107 = {}
        
        # Call to item(...): (line 557)
        # Processing the call keyword arguments (line 557)
        kwargs_135103 = {}
        # Getting the type of 'st_arr' (line 557)
        st_arr_135101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 17), 'st_arr', False)
        # Obtaining the member 'item' of a type (line 557)
        item_135102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 17), st_arr_135101, 'item')
        # Calling item(args, kwargs) (line 557)
        item_call_result_135104 = invoke(stypy.reporting.localization.Localization(__file__, 557, 17), item_135102, *[], **kwargs_135103)
        
        # Obtaining the member 'encode' of a type (line 557)
        encode_135105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 17), item_call_result_135104, 'encode')
        # Calling encode(args, kwargs) (line 557)
        encode_call_result_135108 = invoke(stypy.reporting.localization.Localization(__file__, 557, 17), encode_135105, *[str_135106], **kwargs_135107)
        
        # Assigning a type to the variable 'st' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'st', encode_call_result_135108)
        
        # Assigning a Call to a Name (line 558):
        
        # Assigning a Call to a Name (line 558):
        
        # Call to ndarray(...): (line 558)
        # Processing the call keyword arguments (line 558)
        # Getting the type of 'dims' (line 558)
        dims_135111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 35), 'dims', False)
        keyword_135112 = dims_135111
        str_135113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 47), 'str', 'S1')
        keyword_135114 = str_135113
        # Getting the type of 'st' (line 558)
        st_135115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 60), 'st', False)
        keyword_135116 = st_135115
        kwargs_135117 = {'buffer': keyword_135116, 'dtype': keyword_135114, 'shape': keyword_135112}
        # Getting the type of 'np' (line 558)
        np_135109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 558)
        ndarray_135110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 18), np_135109, 'ndarray')
        # Calling ndarray(args, kwargs) (line 558)
        ndarray_call_result_135118 = invoke(stypy.reporting.localization.Localization(__file__, 558, 18), ndarray_135110, *[], **kwargs_135117)
        
        # Assigning a type to the variable 'arr' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'arr', ndarray_call_result_135118)
        # SSA join for if statement (line 551)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_bytes(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'arr' (line 559)
        arr_135121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 25), 'arr', False)
        # Processing the call keyword arguments (line 559)
        kwargs_135122 = {}
        # Getting the type of 'self' (line 559)
        self_135119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 559)
        write_bytes_135120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), self_135119, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 559)
        write_bytes_call_result_135123 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), write_bytes_135120, *[arr_135121], **kwargs_135122)
        
        
        # ################# End of 'write_char(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_char' in the type store
        # Getting the type of 'stypy_return_type' (line 542)
        stypy_return_type_135124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135124)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_char'
        return stypy_return_type_135124


    @norecursion
    def write_sparse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_sparse'
        module_type_store = module_type_store.open_function_context('write_sparse', 561, 4, False)
        # Assigning a type to the variable 'self' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_localization', localization)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_type_store', module_type_store)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_function_name', 'VarWriter4.write_sparse')
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_param_names_list', ['arr', 'name'])
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_varargs_param_name', None)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_call_defaults', defaults)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_call_varargs', varargs)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VarWriter4.write_sparse.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VarWriter4.write_sparse', ['arr', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_sparse', localization, ['arr', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_sparse(...)' code ##################

        str_135125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, (-1)), 'str', ' Sparse matrices are 2D\n\n        See docstring for VarReader4.read_sparse_array\n        ')
        
        # Assigning a Call to a Name (line 566):
        
        # Assigning a Call to a Name (line 566):
        
        # Call to tocoo(...): (line 566)
        # Processing the call keyword arguments (line 566)
        kwargs_135128 = {}
        # Getting the type of 'arr' (line 566)
        arr_135126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'arr', False)
        # Obtaining the member 'tocoo' of a type (line 566)
        tocoo_135127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 12), arr_135126, 'tocoo')
        # Calling tocoo(args, kwargs) (line 566)
        tocoo_call_result_135129 = invoke(stypy.reporting.localization.Localization(__file__, 566, 12), tocoo_135127, *[], **kwargs_135128)
        
        # Assigning a type to the variable 'A' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'A', tocoo_call_result_135129)
        
        # Assigning a Compare to a Name (line 567):
        
        # Assigning a Compare to a Name (line 567):
        
        # Getting the type of 'A' (line 567)
        A_135130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 16), 'A')
        # Obtaining the member 'dtype' of a type (line 567)
        dtype_135131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 16), A_135130, 'dtype')
        # Obtaining the member 'kind' of a type (line 567)
        kind_135132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 16), dtype_135131, 'kind')
        str_135133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 32), 'str', 'c')
        # Applying the binary operator '==' (line 567)
        result_eq_135134 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 16), '==', kind_135132, str_135133)
        
        # Assigning a type to the variable 'imagf' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'imagf', result_eq_135134)
        
        # Assigning a Call to a Name (line 568):
        
        # Assigning a Call to a Name (line 568):
        
        # Call to zeros(...): (line 568)
        # Processing the call arguments (line 568)
        
        # Obtaining an instance of the builtin type 'tuple' (line 568)
        tuple_135137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 568)
        # Adding element type (line 568)
        # Getting the type of 'A' (line 568)
        A_135138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 24), 'A', False)
        # Obtaining the member 'nnz' of a type (line 568)
        nnz_135139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 24), A_135138, 'nnz')
        int_135140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 32), 'int')
        # Applying the binary operator '+' (line 568)
        result_add_135141 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 24), '+', nnz_135139, int_135140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 24), tuple_135137, result_add_135141)
        # Adding element type (line 568)
        int_135142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 35), 'int')
        # Getting the type of 'imagf' (line 568)
        imagf_135143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 37), 'imagf', False)
        # Applying the binary operator '+' (line 568)
        result_add_135144 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 35), '+', int_135142, imagf_135143)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 24), tuple_135137, result_add_135144)
        
        # Processing the call keyword arguments (line 568)
        str_135145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 51), 'str', 'f8')
        keyword_135146 = str_135145
        kwargs_135147 = {'dtype': keyword_135146}
        # Getting the type of 'np' (line 568)
        np_135135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 14), 'np', False)
        # Obtaining the member 'zeros' of a type (line 568)
        zeros_135136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 14), np_135135, 'zeros')
        # Calling zeros(args, kwargs) (line 568)
        zeros_call_result_135148 = invoke(stypy.reporting.localization.Localization(__file__, 568, 14), zeros_135136, *[tuple_135137], **kwargs_135147)
        
        # Assigning a type to the variable 'ijv' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'ijv', zeros_call_result_135148)
        
        # Assigning a Attribute to a Subscript (line 569):
        
        # Assigning a Attribute to a Subscript (line 569):
        # Getting the type of 'A' (line 569)
        A_135149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 21), 'A')
        # Obtaining the member 'row' of a type (line 569)
        row_135150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 21), A_135149, 'row')
        # Getting the type of 'ijv' (line 569)
        ijv_135151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'ijv')
        int_135152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 13), 'int')
        slice_135153 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 569, 8), None, int_135152, None)
        int_135154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 16), 'int')
        # Storing an element on a container (line 569)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 8), ijv_135151, ((slice_135153, int_135154), row_135150))
        
        # Assigning a Attribute to a Subscript (line 570):
        
        # Assigning a Attribute to a Subscript (line 570):
        # Getting the type of 'A' (line 570)
        A_135155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 21), 'A')
        # Obtaining the member 'col' of a type (line 570)
        col_135156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 21), A_135155, 'col')
        # Getting the type of 'ijv' (line 570)
        ijv_135157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'ijv')
        int_135158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 13), 'int')
        slice_135159 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 570, 8), None, int_135158, None)
        int_135160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 16), 'int')
        # Storing an element on a container (line 570)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 8), ijv_135157, ((slice_135159, int_135160), col_135156))
        
        # Getting the type of 'ijv' (line 571)
        ijv_135161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'ijv')
        
        # Obtaining the type of the subscript
        int_135162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 13), 'int')
        slice_135163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 8), None, int_135162, None)
        int_135164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 16), 'int')
        int_135165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 18), 'int')
        slice_135166 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 8), int_135164, int_135165, None)
        # Getting the type of 'ijv' (line 571)
        ijv_135167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'ijv')
        # Obtaining the member '__getitem__' of a type (line 571)
        getitem___135168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 8), ijv_135167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 571)
        subscript_call_result_135169 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), getitem___135168, (slice_135163, slice_135166))
        
        int_135170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 24), 'int')
        # Applying the binary operator '+=' (line 571)
        result_iadd_135171 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 8), '+=', subscript_call_result_135169, int_135170)
        # Getting the type of 'ijv' (line 571)
        ijv_135172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'ijv')
        int_135173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 13), 'int')
        slice_135174 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 8), None, int_135173, None)
        int_135175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 16), 'int')
        int_135176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 18), 'int')
        slice_135177 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 8), int_135175, int_135176, None)
        # Storing an element on a container (line 571)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 8), ijv_135172, ((slice_135174, slice_135177), result_iadd_135171))
        
        
        # Getting the type of 'imagf' (line 572)
        imagf_135178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 11), 'imagf')
        # Testing the type of an if condition (line 572)
        if_condition_135179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 572, 8), imagf_135178)
        # Assigning a type to the variable 'if_condition_135179' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'if_condition_135179', if_condition_135179)
        # SSA begins for if statement (line 572)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 573):
        
        # Assigning a Attribute to a Subscript (line 573):
        # Getting the type of 'A' (line 573)
        A_135180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 25), 'A')
        # Obtaining the member 'data' of a type (line 573)
        data_135181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 25), A_135180, 'data')
        # Obtaining the member 'real' of a type (line 573)
        real_135182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 25), data_135181, 'real')
        # Getting the type of 'ijv' (line 573)
        ijv_135183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'ijv')
        int_135184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 17), 'int')
        slice_135185 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 573, 12), None, int_135184, None)
        int_135186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 20), 'int')
        # Storing an element on a container (line 573)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 12), ijv_135183, ((slice_135185, int_135186), real_135182))
        
        # Assigning a Attribute to a Subscript (line 574):
        
        # Assigning a Attribute to a Subscript (line 574):
        # Getting the type of 'A' (line 574)
        A_135187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 25), 'A')
        # Obtaining the member 'data' of a type (line 574)
        data_135188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 25), A_135187, 'data')
        # Obtaining the member 'imag' of a type (line 574)
        imag_135189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 25), data_135188, 'imag')
        # Getting the type of 'ijv' (line 574)
        ijv_135190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'ijv')
        int_135191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 17), 'int')
        slice_135192 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 574, 12), None, int_135191, None)
        int_135193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 20), 'int')
        # Storing an element on a container (line 574)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 12), ijv_135190, ((slice_135192, int_135193), imag_135189))
        # SSA branch for the else part of an if statement (line 572)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Subscript (line 576):
        
        # Assigning a Attribute to a Subscript (line 576):
        # Getting the type of 'A' (line 576)
        A_135194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 25), 'A')
        # Obtaining the member 'data' of a type (line 576)
        data_135195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 25), A_135194, 'data')
        # Getting the type of 'ijv' (line 576)
        ijv_135196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'ijv')
        int_135197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 17), 'int')
        slice_135198 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 576, 12), None, int_135197, None)
        int_135199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 20), 'int')
        # Storing an element on a container (line 576)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 12), ijv_135196, ((slice_135198, int_135199), data_135195))
        # SSA join for if statement (line 572)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Subscript (line 577):
        
        # Assigning a Attribute to a Subscript (line 577):
        # Getting the type of 'A' (line 577)
        A_135200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 22), 'A')
        # Obtaining the member 'shape' of a type (line 577)
        shape_135201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 22), A_135200, 'shape')
        # Getting the type of 'ijv' (line 577)
        ijv_135202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'ijv')
        int_135203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 12), 'int')
        int_135204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 15), 'int')
        int_135205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 17), 'int')
        slice_135206 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 577, 8), int_135204, int_135205, None)
        # Storing an element on a container (line 577)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 8), ijv_135202, ((int_135203, slice_135206), shape_135201))
        
        # Call to write_header(...): (line 578)
        # Processing the call arguments (line 578)
        # Getting the type of 'name' (line 579)
        name_135209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'name', False)
        # Getting the type of 'ijv' (line 580)
        ijv_135210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'ijv', False)
        # Obtaining the member 'shape' of a type (line 580)
        shape_135211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 12), ijv_135210, 'shape')
        # Processing the call keyword arguments (line 578)
        # Getting the type of 'miDOUBLE' (line 581)
        miDOUBLE_135212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 14), 'miDOUBLE', False)
        keyword_135213 = miDOUBLE_135212
        # Getting the type of 'mxSPARSE_CLASS' (line 582)
        mxSPARSE_CLASS_135214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 14), 'mxSPARSE_CLASS', False)
        keyword_135215 = mxSPARSE_CLASS_135214
        kwargs_135216 = {'P': keyword_135213, 'T': keyword_135215}
        # Getting the type of 'self' (line 578)
        self_135207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'self', False)
        # Obtaining the member 'write_header' of a type (line 578)
        write_header_135208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 8), self_135207, 'write_header')
        # Calling write_header(args, kwargs) (line 578)
        write_header_call_result_135217 = invoke(stypy.reporting.localization.Localization(__file__, 578, 8), write_header_135208, *[name_135209, shape_135211], **kwargs_135216)
        
        
        # Call to write_bytes(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'ijv' (line 583)
        ijv_135220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 25), 'ijv', False)
        # Processing the call keyword arguments (line 583)
        kwargs_135221 = {}
        # Getting the type of 'self' (line 583)
        self_135218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'self', False)
        # Obtaining the member 'write_bytes' of a type (line 583)
        write_bytes_135219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 8), self_135218, 'write_bytes')
        # Calling write_bytes(args, kwargs) (line 583)
        write_bytes_call_result_135222 = invoke(stypy.reporting.localization.Localization(__file__, 583, 8), write_bytes_135219, *[ijv_135220], **kwargs_135221)
        
        
        # ################# End of 'write_sparse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_sparse' in the type store
        # Getting the type of 'stypy_return_type' (line 561)
        stypy_return_type_135223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135223)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_sparse'
        return stypy_return_type_135223


# Assigning a type to the variable 'VarWriter4' (line 448)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'VarWriter4', VarWriter4)
# Declaration of the 'MatFile4Writer' class

class MatFile4Writer(object, ):
    str_135224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 4), 'str', ' Class for writing matlab 4 format files ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 588)
        None_135225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 44), 'None')
        defaults = [None_135225]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 588, 4, False)
        # Assigning a type to the variable 'self' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Writer.__init__', ['file_stream', 'oned_as'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_stream', 'oned_as'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 589):
        
        # Assigning a Name to a Attribute (line 589):
        # Getting the type of 'file_stream' (line 589)
        file_stream_135226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 27), 'file_stream')
        # Getting the type of 'self' (line 589)
        self_135227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'self')
        # Setting the type of the member 'file_stream' of a type (line 589)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 8), self_135227, 'file_stream', file_stream_135226)
        
        # Type idiom detected: calculating its left and rigth part (line 590)
        # Getting the type of 'oned_as' (line 590)
        oned_as_135228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 11), 'oned_as')
        # Getting the type of 'None' (line 590)
        None_135229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 22), 'None')
        
        (may_be_135230, more_types_in_union_135231) = may_be_none(oned_as_135228, None_135229)

        if may_be_135230:

            if more_types_in_union_135231:
                # Runtime conditional SSA (line 590)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 591):
            
            # Assigning a Str to a Name (line 591):
            str_135232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 22), 'str', 'row')
            # Assigning a type to the variable 'oned_as' (line 591)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'oned_as', str_135232)

            if more_types_in_union_135231:
                # SSA join for if statement (line 590)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 592):
        
        # Assigning a Name to a Attribute (line 592):
        # Getting the type of 'oned_as' (line 592)
        oned_as_135233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 23), 'oned_as')
        # Getting the type of 'self' (line 592)
        self_135234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'self')
        # Setting the type of the member 'oned_as' of a type (line 592)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 8), self_135234, 'oned_as', oned_as_135233)
        
        # Assigning a Name to a Attribute (line 593):
        
        # Assigning a Name to a Attribute (line 593):
        # Getting the type of 'None' (line 593)
        None_135235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 30), 'None')
        # Getting the type of 'self' (line 593)
        self_135236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'self')
        # Setting the type of the member '_matrix_writer' of a type (line 593)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), self_135236, '_matrix_writer', None_135235)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def put_variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 595)
        None_135237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 48), 'None')
        defaults = [None_135237]
        # Create a new context for function 'put_variables'
        module_type_store = module_type_store.open_function_context('put_variables', 595, 4, False)
        # Assigning a type to the variable 'self' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_localization', localization)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_function_name', 'MatFile4Writer.put_variables')
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_param_names_list', ['mdict', 'write_header'])
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFile4Writer.put_variables.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFile4Writer.put_variables', ['mdict', 'write_header'], None, None, defaults, varargs, kwargs)

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

        str_135238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, (-1)), 'str', ' Write variables in `mdict` to stream\n\n        Parameters\n        ----------\n        mdict : mapping\n           mapping with method ``items`` return name, contents pairs\n           where ``name`` which will appeak in the matlab workspace in\n           file load, and ``contents`` is something writeable to a\n           matlab file, such as a numpy array.\n        write_header : {None, True, False}\n           If True, then write the matlab file header before writing the\n           variables.  If None (the default) then write the file header\n           if we are at position 0 in the stream.  By setting False\n           here, and setting the stream position to the end of the file,\n           you can append variables to a matlab file\n        ')
        
        # Assigning a Call to a Attribute (line 615):
        
        # Assigning a Call to a Attribute (line 615):
        
        # Call to VarWriter4(...): (line 615)
        # Processing the call arguments (line 615)
        # Getting the type of 'self' (line 615)
        self_135240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 41), 'self', False)
        # Processing the call keyword arguments (line 615)
        kwargs_135241 = {}
        # Getting the type of 'VarWriter4' (line 615)
        VarWriter4_135239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 30), 'VarWriter4', False)
        # Calling VarWriter4(args, kwargs) (line 615)
        VarWriter4_call_result_135242 = invoke(stypy.reporting.localization.Localization(__file__, 615, 30), VarWriter4_135239, *[self_135240], **kwargs_135241)
        
        # Getting the type of 'self' (line 615)
        self_135243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'self')
        # Setting the type of the member '_matrix_writer' of a type (line 615)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 8), self_135243, '_matrix_writer', VarWriter4_call_result_135242)
        
        
        # Call to items(...): (line 616)
        # Processing the call keyword arguments (line 616)
        kwargs_135246 = {}
        # Getting the type of 'mdict' (line 616)
        mdict_135244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 25), 'mdict', False)
        # Obtaining the member 'items' of a type (line 616)
        items_135245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 25), mdict_135244, 'items')
        # Calling items(args, kwargs) (line 616)
        items_call_result_135247 = invoke(stypy.reporting.localization.Localization(__file__, 616, 25), items_135245, *[], **kwargs_135246)
        
        # Testing the type of a for loop iterable (line 616)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 616, 8), items_call_result_135247)
        # Getting the type of the for loop variable (line 616)
        for_loop_var_135248 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 616, 8), items_call_result_135247)
        # Assigning a type to the variable 'name' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 8), for_loop_var_135248))
        # Assigning a type to the variable 'var' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'var', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 8), for_loop_var_135248))
        # SSA begins for a for statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'var' (line 617)
        var_135252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 38), 'var', False)
        # Getting the type of 'name' (line 617)
        name_135253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 43), 'name', False)
        # Processing the call keyword arguments (line 617)
        kwargs_135254 = {}
        # Getting the type of 'self' (line 617)
        self_135249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'self', False)
        # Obtaining the member '_matrix_writer' of a type (line 617)
        _matrix_writer_135250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 12), self_135249, '_matrix_writer')
        # Obtaining the member 'write' of a type (line 617)
        write_135251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 12), _matrix_writer_135250, 'write')
        # Calling write(args, kwargs) (line 617)
        write_call_result_135255 = invoke(stypy.reporting.localization.Localization(__file__, 617, 12), write_135251, *[var_135252, name_135253], **kwargs_135254)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'put_variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'put_variables' in the type store
        # Getting the type of 'stypy_return_type' (line 595)
        stypy_return_type_135256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'put_variables'
        return stypy_return_type_135256


# Assigning a type to the variable 'MatFile4Writer' (line 586)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 0), 'MatFile4Writer', MatFile4Writer)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
