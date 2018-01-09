
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Implementation of Harwell-Boeing read/write.
3: 
4: At the moment not the full Harwell-Boeing format is supported. Supported
5: features are:
6: 
7:     - assembled, non-symmetric, real matrices
8:     - integer for pointer/indices
9:     - exponential format for float values, and int format
10: 
11: '''
12: from __future__ import division, print_function, absolute_import
13: 
14: # TODO:
15: #   - Add more support (symmetric/complex matrices, non-assembled matrices ?)
16: 
17: # XXX: reading is reasonably efficient (>= 85 % is in numpy.fromstring), but
18: # takes a lot of memory. Being faster would require compiled code.
19: # write is not efficient. Although not a terribly exciting task,
20: # having reusable facilities to efficiently read/write fortran-formatted files
21: # would be useful outside this module.
22: 
23: import warnings
24: 
25: import numpy as np
26: from scipy.sparse import csc_matrix
27: from scipy.io.harwell_boeing._fortran_format_parser import \
28:         FortranFormatParser, IntFormat, ExpFormat
29: 
30: from scipy._lib.six import string_types
31: 
32: __all__ = ["MalformedHeader", "hb_read", "hb_write", "HBInfo", "HBFile",
33:            "HBMatrixType"]
34: 
35: 
36: class MalformedHeader(Exception):
37:     pass
38: 
39: 
40: class LineOverflow(Warning):
41:     pass
42: 
43: 
44: def _nbytes_full(fmt, nlines):
45:     '''Return the number of bytes to read to get every full lines for the
46:     given parsed fortran format.'''
47:     return (fmt.repeat * fmt.width + 1) * (nlines - 1)
48: 
49: 
50: class HBInfo(object):
51:     @classmethod
52:     def from_data(cls, m, title="Default title", key="0", mxtype=None, fmt=None):
53:         '''Create a HBInfo instance from an existing sparse matrix.
54: 
55:         Parameters
56:         ----------
57:         m : sparse matrix
58:             the HBInfo instance will derive its parameters from m
59:         title : str
60:             Title to put in the HB header
61:         key : str
62:             Key
63:         mxtype : HBMatrixType
64:             type of the input matrix
65:         fmt : dict
66:             not implemented
67: 
68:         Returns
69:         -------
70:         hb_info : HBInfo instance
71:         '''
72:         pointer = m.indptr
73:         indices = m.indices
74:         values = m.data
75: 
76:         nrows, ncols = m.shape
77:         nnon_zeros = m.nnz
78: 
79:         if fmt is None:
80:             # +1 because HB use one-based indexing (Fortran), and we will write
81:             # the indices /pointer as such
82:             pointer_fmt = IntFormat.from_number(np.max(pointer+1))
83:             indices_fmt = IntFormat.from_number(np.max(indices+1))
84: 
85:             if values.dtype.kind in np.typecodes["AllFloat"]:
86:                 values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
87:             elif values.dtype.kind in np.typecodes["AllInteger"]:
88:                 values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
89:             else:
90:                 raise NotImplementedError("type %s not implemented yet" % values.dtype.kind)
91:         else:
92:             raise NotImplementedError("fmt argument not supported yet.")
93: 
94:         if mxtype is None:
95:             if not np.isrealobj(values):
96:                 raise ValueError("Complex values not supported yet")
97:             if values.dtype.kind in np.typecodes["AllInteger"]:
98:                 tp = "integer"
99:             elif values.dtype.kind in np.typecodes["AllFloat"]:
100:                 tp = "real"
101:             else:
102:                 raise NotImplementedError("type %s for values not implemented"
103:                                           % values.dtype)
104:             mxtype = HBMatrixType(tp, "unsymmetric", "assembled")
105:         else:
106:             raise ValueError("mxtype argument not handled yet.")
107: 
108:         def _nlines(fmt, size):
109:             nlines = size // fmt.repeat
110:             if nlines * fmt.repeat != size:
111:                 nlines += 1
112:             return nlines
113: 
114:         pointer_nlines = _nlines(pointer_fmt, pointer.size)
115:         indices_nlines = _nlines(indices_fmt, indices.size)
116:         values_nlines = _nlines(values_fmt, values.size)
117: 
118:         total_nlines = pointer_nlines + indices_nlines + values_nlines
119: 
120:         return cls(title, key,
121:             total_nlines, pointer_nlines, indices_nlines, values_nlines,
122:             mxtype, nrows, ncols, nnon_zeros,
123:             pointer_fmt.fortran_format, indices_fmt.fortran_format,
124:             values_fmt.fortran_format)
125: 
126:     @classmethod
127:     def from_file(cls, fid):
128:         '''Create a HBInfo instance from a file object containg a matrix in the
129:         HB format.
130: 
131:         Parameters
132:         ----------
133:         fid : file-like matrix
134:             File or file-like object containing a matrix in the HB format.
135: 
136:         Returns
137:         -------
138:         hb_info : HBInfo instance
139:         '''
140:         # First line
141:         line = fid.readline().strip("\n")
142:         if not len(line) > 72:
143:             raise ValueError("Expected at least 72 characters for first line, "
144:                              "got: \n%s" % line)
145:         title = line[:72]
146:         key = line[72:]
147: 
148:         # Second line
149:         line = fid.readline().strip("\n")
150:         if not len(line.rstrip()) >= 56:
151:             raise ValueError("Expected at least 56 characters for second line, "
152:                              "got: \n%s" % line)
153:         total_nlines = _expect_int(line[:14])
154:         pointer_nlines = _expect_int(line[14:28])
155:         indices_nlines = _expect_int(line[28:42])
156:         values_nlines = _expect_int(line[42:56])
157: 
158:         rhs_nlines = line[56:72].strip()
159:         if rhs_nlines == '':
160:             rhs_nlines = 0
161:         else:
162:             rhs_nlines = _expect_int(rhs_nlines)
163:         if not rhs_nlines == 0:
164:             raise ValueError("Only files without right hand side supported for "
165:                              "now.")
166: 
167:         # Third line
168:         line = fid.readline().strip("\n")
169:         if not len(line) >= 70:
170:             raise ValueError("Expected at least 72 character for third line, got:\n"
171:                              "%s" % line)
172: 
173:         mxtype_s = line[:3].upper()
174:         if not len(mxtype_s) == 3:
175:             raise ValueError("mxtype expected to be 3 characters long")
176: 
177:         mxtype = HBMatrixType.from_fortran(mxtype_s)
178:         if mxtype.value_type not in ["real", "integer"]:
179:             raise ValueError("Only real or integer matrices supported for "
180:                              "now (detected %s)" % mxtype)
181:         if not mxtype.structure == "unsymmetric":
182:             raise ValueError("Only unsymmetric matrices supported for "
183:                              "now (detected %s)" % mxtype)
184:         if not mxtype.storage == "assembled":
185:             raise ValueError("Only assembled matrices supported for now")
186: 
187:         if not line[3:14] == " " * 11:
188:             raise ValueError("Malformed data for third line: %s" % line)
189: 
190:         nrows = _expect_int(line[14:28])
191:         ncols = _expect_int(line[28:42])
192:         nnon_zeros = _expect_int(line[42:56])
193:         nelementals = _expect_int(line[56:70])
194:         if not nelementals == 0:
195:             raise ValueError("Unexpected value %d for nltvl (last entry of line 3)"
196:                              % nelementals)
197: 
198:         # Fourth line
199:         line = fid.readline().strip("\n")
200: 
201:         ct = line.split()
202:         if not len(ct) == 3:
203:             raise ValueError("Expected 3 formats, got %s" % ct)
204: 
205:         return cls(title, key,
206:                    total_nlines, pointer_nlines, indices_nlines, values_nlines,
207:                    mxtype, nrows, ncols, nnon_zeros,
208:                    ct[0], ct[1], ct[2],
209:                    rhs_nlines, nelementals)
210: 
211:     def __init__(self, title, key,
212:             total_nlines, pointer_nlines, indices_nlines, values_nlines,
213:             mxtype, nrows, ncols, nnon_zeros,
214:             pointer_format_str, indices_format_str, values_format_str,
215:             right_hand_sides_nlines=0, nelementals=0):
216:         '''Do not use this directly, but the class ctrs (from_* functions).'''
217:         self.title = title
218:         self.key = key
219:         if title is None:
220:             title = "No Title"
221:         if len(title) > 72:
222:             raise ValueError("title cannot be > 72 characters")
223: 
224:         if key is None:
225:             key = "|No Key"
226:         if len(key) > 8:
227:             warnings.warn("key is > 8 characters (key is %s)" % key, LineOverflow)
228: 
229:         self.total_nlines = total_nlines
230:         self.pointer_nlines = pointer_nlines
231:         self.indices_nlines = indices_nlines
232:         self.values_nlines = values_nlines
233: 
234:         parser = FortranFormatParser()
235:         pointer_format = parser.parse(pointer_format_str)
236:         if not isinstance(pointer_format, IntFormat):
237:             raise ValueError("Expected int format for pointer format, got %s"
238:                              % pointer_format)
239: 
240:         indices_format = parser.parse(indices_format_str)
241:         if not isinstance(indices_format, IntFormat):
242:             raise ValueError("Expected int format for indices format, got %s" %
243:                              indices_format)
244: 
245:         values_format = parser.parse(values_format_str)
246:         if isinstance(values_format, ExpFormat):
247:             if mxtype.value_type not in ["real", "complex"]:
248:                 raise ValueError("Inconsistency between matrix type %s and "
249:                                  "value type %s" % (mxtype, values_format))
250:             values_dtype = np.float64
251:         elif isinstance(values_format, IntFormat):
252:             if mxtype.value_type not in ["integer"]:
253:                 raise ValueError("Inconsistency between matrix type %s and "
254:                                  "value type %s" % (mxtype, values_format))
255:             # XXX: fortran int -> dtype association ?
256:             values_dtype = int
257:         else:
258:             raise ValueError("Unsupported format for values %r" % (values_format,))
259: 
260:         self.pointer_format = pointer_format
261:         self.indices_format = indices_format
262:         self.values_format = values_format
263: 
264:         self.pointer_dtype = np.int32
265:         self.indices_dtype = np.int32
266:         self.values_dtype = values_dtype
267: 
268:         self.pointer_nlines = pointer_nlines
269:         self.pointer_nbytes_full = _nbytes_full(pointer_format, pointer_nlines)
270: 
271:         self.indices_nlines = indices_nlines
272:         self.indices_nbytes_full = _nbytes_full(indices_format, indices_nlines)
273: 
274:         self.values_nlines = values_nlines
275:         self.values_nbytes_full = _nbytes_full(values_format, values_nlines)
276: 
277:         self.nrows = nrows
278:         self.ncols = ncols
279:         self.nnon_zeros = nnon_zeros
280:         self.nelementals = nelementals
281:         self.mxtype = mxtype
282: 
283:     def dump(self):
284:         '''Gives the header corresponding to this instance as a string.'''
285:         header = [self.title.ljust(72) + self.key.ljust(8)]
286: 
287:         header.append("%14d%14d%14d%14d" %
288:                       (self.total_nlines, self.pointer_nlines,
289:                        self.indices_nlines, self.values_nlines))
290:         header.append("%14s%14d%14d%14d%14d" %
291:                       (self.mxtype.fortran_format.ljust(14), self.nrows,
292:                        self.ncols, self.nnon_zeros, 0))
293: 
294:         pffmt = self.pointer_format.fortran_format
295:         iffmt = self.indices_format.fortran_format
296:         vffmt = self.values_format.fortran_format
297:         header.append("%16s%16s%20s" %
298:                       (pffmt.ljust(16), iffmt.ljust(16), vffmt.ljust(20)))
299:         return "\n".join(header)
300: 
301: 
302: def _expect_int(value, msg=None):
303:     try:
304:         return int(value)
305:     except ValueError:
306:         if msg is None:
307:             msg = "Expected an int, got %s"
308:         raise ValueError(msg % value)
309: 
310: 
311: def _read_hb_data(content, header):
312:     # XXX: look at a way to reduce memory here (big string creation)
313:     ptr_string = "".join([content.read(header.pointer_nbytes_full),
314:                            content.readline()])
315:     ptr = np.fromstring(ptr_string,
316:             dtype=int, sep=' ')
317: 
318:     ind_string = "".join([content.read(header.indices_nbytes_full),
319:                        content.readline()])
320:     ind = np.fromstring(ind_string,
321:             dtype=int, sep=' ')
322: 
323:     val_string = "".join([content.read(header.values_nbytes_full),
324:                           content.readline()])
325:     val = np.fromstring(val_string,
326:             dtype=header.values_dtype, sep=' ')
327: 
328:     try:
329:         return csc_matrix((val, ind-1, ptr-1),
330:                           shape=(header.nrows, header.ncols))
331:     except ValueError as e:
332:         raise e
333: 
334: 
335: def _write_data(m, fid, header):
336:     def write_array(f, ar, nlines, fmt):
337:         # ar_nlines is the number of full lines, n is the number of items per
338:         # line, ffmt the fortran format
339:         pyfmt = fmt.python_format
340:         pyfmt_full = pyfmt * fmt.repeat
341: 
342:         # for each array to write, we first write the full lines, and special
343:         # case for partial line
344:         full = ar[:(nlines - 1) * fmt.repeat]
345:         for row in full.reshape((nlines-1, fmt.repeat)):
346:             f.write(pyfmt_full % tuple(row) + "\n")
347:         nremain = ar.size - full.size
348:         if nremain > 0:
349:             f.write((pyfmt * nremain) % tuple(ar[ar.size - nremain:]) + "\n")
350: 
351:     fid.write(header.dump())
352:     fid.write("\n")
353:     # +1 is for fortran one-based indexing
354:     write_array(fid, m.indptr+1, header.pointer_nlines,
355:                 header.pointer_format)
356:     write_array(fid, m.indices+1, header.indices_nlines,
357:                 header.indices_format)
358:     write_array(fid, m.data, header.values_nlines,
359:                 header.values_format)
360: 
361: 
362: class HBMatrixType(object):
363:     '''Class to hold the matrix type.'''
364:     # q2f* translates qualified names to fortran character
365:     _q2f_type = {
366:         "real": "R",
367:         "complex": "C",
368:         "pattern": "P",
369:         "integer": "I",
370:     }
371:     _q2f_structure = {
372:             "symmetric": "S",
373:             "unsymmetric": "U",
374:             "hermitian": "H",
375:             "skewsymmetric": "Z",
376:             "rectangular": "R"
377:     }
378:     _q2f_storage = {
379:         "assembled": "A",
380:         "elemental": "E",
381:     }
382: 
383:     _f2q_type = dict([(j, i) for i, j in _q2f_type.items()])
384:     _f2q_structure = dict([(j, i) for i, j in _q2f_structure.items()])
385:     _f2q_storage = dict([(j, i) for i, j in _q2f_storage.items()])
386: 
387:     @classmethod
388:     def from_fortran(cls, fmt):
389:         if not len(fmt) == 3:
390:             raise ValueError("Fortran format for matrix type should be 3 "
391:                              "characters long")
392:         try:
393:             value_type = cls._f2q_type[fmt[0]]
394:             structure = cls._f2q_structure[fmt[1]]
395:             storage = cls._f2q_storage[fmt[2]]
396:             return cls(value_type, structure, storage)
397:         except KeyError:
398:             raise ValueError("Unrecognized format %s" % fmt)
399: 
400:     def __init__(self, value_type, structure, storage="assembled"):
401:         self.value_type = value_type
402:         self.structure = structure
403:         self.storage = storage
404: 
405:         if value_type not in self._q2f_type:
406:             raise ValueError("Unrecognized type %s" % value_type)
407:         if structure not in self._q2f_structure:
408:             raise ValueError("Unrecognized structure %s" % structure)
409:         if storage not in self._q2f_storage:
410:             raise ValueError("Unrecognized storage %s" % storage)
411: 
412:     @property
413:     def fortran_format(self):
414:         return self._q2f_type[self.value_type] + \
415:                self._q2f_structure[self.structure] + \
416:                self._q2f_storage[self.storage]
417: 
418:     def __repr__(self):
419:         return "HBMatrixType(%s, %s, %s)" % \
420:                (self.value_type, self.structure, self.storage)
421: 
422: 
423: class HBFile(object):
424:     def __init__(self, file, hb_info=None):
425:         '''Create a HBFile instance.
426: 
427:         Parameters
428:         ----------
429:         file : file-object
430:             StringIO work as well
431:         hb_info : HBInfo, optional
432:             Should be given as an argument for writing, in which case the file
433:             should be writable.
434:         '''
435:         self._fid = file
436:         if hb_info is None:
437:             self._hb_info = HBInfo.from_file(file)
438:         else:
439:             #raise IOError("file %s is not writable, and hb_info "
440:             #              "was given." % file)
441:             self._hb_info = hb_info
442: 
443:     @property
444:     def title(self):
445:         return self._hb_info.title
446: 
447:     @property
448:     def key(self):
449:         return self._hb_info.key
450: 
451:     @property
452:     def type(self):
453:         return self._hb_info.mxtype.value_type
454: 
455:     @property
456:     def structure(self):
457:         return self._hb_info.mxtype.structure
458: 
459:     @property
460:     def storage(self):
461:         return self._hb_info.mxtype.storage
462: 
463:     def read_matrix(self):
464:         return _read_hb_data(self._fid, self._hb_info)
465: 
466:     def write_matrix(self, m):
467:         return _write_data(m, self._fid, self._hb_info)
468: 
469: 
470: def hb_read(path_or_open_file):
471:     '''Read HB-format file.
472: 
473:     Parameters
474:     ----------
475:     path_or_open_file : path-like or file-like
476:         If a file-like object, it is used as-is. Otherwise it is opened
477:         before reading.
478: 
479:     Returns
480:     -------
481:     data : scipy.sparse.csc_matrix instance
482:         The data read from the HB file as a sparse matrix.
483: 
484:     Notes
485:     -----
486:     At the moment not the full Harwell-Boeing format is supported. Supported
487:     features are:
488: 
489:         - assembled, non-symmetric, real matrices
490:         - integer for pointer/indices
491:         - exponential format for float values, and int format
492: 
493:     '''
494:     def _get_matrix(fid):
495:         hb = HBFile(fid)
496:         return hb.read_matrix()
497: 
498:     if hasattr(path_or_open_file, 'read'):
499:         return _get_matrix(path_or_open_file)
500:     else:
501:         with open(path_or_open_file) as f:
502:             return _get_matrix(f)
503: 
504: 
505: def hb_write(path_or_open_file, m, hb_info=None):
506:     '''Write HB-format file.
507: 
508:     Parameters
509:     ----------
510:     path_or_open_file : path-like or file-like
511:         If a file-like object, it is used as-is. Otherwise it is opened
512:         before writing.
513:     m : sparse-matrix
514:         the sparse matrix to write
515:     hb_info : HBInfo
516:         contains the meta-data for write
517: 
518:     Returns
519:     -------
520:     None
521: 
522:     Notes
523:     -----
524:     At the moment not the full Harwell-Boeing format is supported. Supported
525:     features are:
526: 
527:         - assembled, non-symmetric, real matrices
528:         - integer for pointer/indices
529:         - exponential format for float values, and int format
530: 
531:     '''
532:     if hb_info is None:
533:         hb_info = HBInfo.from_data(m)
534: 
535:     def _set_matrix(fid):
536:         hb = HBFile(fid, hb_info)
537:         return hb.write_matrix(m)
538: 
539:     if hasattr(path_or_open_file, 'write'):
540:         return _set_matrix(path_or_open_file)
541:     else:
542:         with open(path_or_open_file, 'w') as f:
543:             return _set_matrix(f)
544: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_130779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nImplementation of Harwell-Boeing read/write.\n\nAt the moment not the full Harwell-Boeing format is supported. Supported\nfeatures are:\n\n    - assembled, non-symmetric, real matrices\n    - integer for pointer/indices\n    - exponential format for float values, and int format\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import warnings' statement (line 23)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import numpy' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')
import_130780 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy')

if (type(import_130780) is not StypyTypeError):

    if (import_130780 != 'pyd_module'):
        __import__(import_130780)
        sys_modules_130781 = sys.modules[import_130780]
        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'np', sys_modules_130781.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy', import_130780)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.sparse import csc_matrix' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')
import_130782 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse')

if (type(import_130782) is not StypyTypeError):

    if (import_130782 != 'pyd_module'):
        __import__(import_130782)
        sys_modules_130783 = sys.modules[import_130782]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', sys_modules_130783.module_type_store, module_type_store, ['csc_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_130783, sys_modules_130783.module_type_store, module_type_store)
    else:
        from scipy.sparse import csc_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', None, module_type_store, ['csc_matrix'], [csc_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.sparse', import_130782)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from scipy.io.harwell_boeing._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')
import_130784 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.io.harwell_boeing._fortran_format_parser')

if (type(import_130784) is not StypyTypeError):

    if (import_130784 != 'pyd_module'):
        __import__(import_130784)
        sys_modules_130785 = sys.modules[import_130784]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.io.harwell_boeing._fortran_format_parser', sys_modules_130785.module_type_store, module_type_store, ['FortranFormatParser', 'IntFormat', 'ExpFormat'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_130785, sys_modules_130785.module_type_store, module_type_store)
    else:
        from scipy.io.harwell_boeing._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.io.harwell_boeing._fortran_format_parser', None, module_type_store, ['FortranFormatParser', 'IntFormat', 'ExpFormat'], [FortranFormatParser, IntFormat, ExpFormat])

else:
    # Assigning a type to the variable 'scipy.io.harwell_boeing._fortran_format_parser' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy.io.harwell_boeing._fortran_format_parser', import_130784)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from scipy._lib.six import string_types' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')
import_130786 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'scipy._lib.six')

if (type(import_130786) is not StypyTypeError):

    if (import_130786 != 'pyd_module'):
        __import__(import_130786)
        sys_modules_130787 = sys.modules[import_130786]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'scipy._lib.six', sys_modules_130787.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_130787, sys_modules_130787.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'scipy._lib.six', import_130786)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')


# Assigning a List to a Name (line 32):

# Assigning a List to a Name (line 32):
__all__ = ['MalformedHeader', 'hb_read', 'hb_write', 'HBInfo', 'HBFile', 'HBMatrixType']
module_type_store.set_exportable_members(['MalformedHeader', 'hb_read', 'hb_write', 'HBInfo', 'HBFile', 'HBMatrixType'])

# Obtaining an instance of the builtin type 'list' (line 32)
list_130788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
str_130789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 11), 'str', 'MalformedHeader')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_130788, str_130789)
# Adding element type (line 32)
str_130790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 30), 'str', 'hb_read')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_130788, str_130790)
# Adding element type (line 32)
str_130791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 41), 'str', 'hb_write')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_130788, str_130791)
# Adding element type (line 32)
str_130792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 53), 'str', 'HBInfo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_130788, str_130792)
# Adding element type (line 32)
str_130793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 63), 'str', 'HBFile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_130788, str_130793)
# Adding element type (line 32)
str_130794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 11), 'str', 'HBMatrixType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_130788, str_130794)

# Assigning a type to the variable '__all__' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '__all__', list_130788)
# Declaration of the 'MalformedHeader' class
# Getting the type of 'Exception' (line 36)
Exception_130795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'Exception')

class MalformedHeader(Exception_130795, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 36, 0, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MalformedHeader.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MalformedHeader' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'MalformedHeader', MalformedHeader)
# Declaration of the 'LineOverflow' class
# Getting the type of 'Warning' (line 40)
Warning_130796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'Warning')

class LineOverflow(Warning_130796, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 0, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LineOverflow.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LineOverflow' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'LineOverflow', LineOverflow)

@norecursion
def _nbytes_full(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_nbytes_full'
    module_type_store = module_type_store.open_function_context('_nbytes_full', 44, 0, False)
    
    # Passed parameters checking function
    _nbytes_full.stypy_localization = localization
    _nbytes_full.stypy_type_of_self = None
    _nbytes_full.stypy_type_store = module_type_store
    _nbytes_full.stypy_function_name = '_nbytes_full'
    _nbytes_full.stypy_param_names_list = ['fmt', 'nlines']
    _nbytes_full.stypy_varargs_param_name = None
    _nbytes_full.stypy_kwargs_param_name = None
    _nbytes_full.stypy_call_defaults = defaults
    _nbytes_full.stypy_call_varargs = varargs
    _nbytes_full.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nbytes_full', ['fmt', 'nlines'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nbytes_full', localization, ['fmt', 'nlines'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nbytes_full(...)' code ##################

    str_130797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', 'Return the number of bytes to read to get every full lines for the\n    given parsed fortran format.')
    # Getting the type of 'fmt' (line 47)
    fmt_130798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'fmt')
    # Obtaining the member 'repeat' of a type (line 47)
    repeat_130799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), fmt_130798, 'repeat')
    # Getting the type of 'fmt' (line 47)
    fmt_130800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'fmt')
    # Obtaining the member 'width' of a type (line 47)
    width_130801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), fmt_130800, 'width')
    # Applying the binary operator '*' (line 47)
    result_mul_130802 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 12), '*', repeat_130799, width_130801)
    
    int_130803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'int')
    # Applying the binary operator '+' (line 47)
    result_add_130804 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 12), '+', result_mul_130802, int_130803)
    
    # Getting the type of 'nlines' (line 47)
    nlines_130805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 43), 'nlines')
    int_130806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 52), 'int')
    # Applying the binary operator '-' (line 47)
    result_sub_130807 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 43), '-', nlines_130805, int_130806)
    
    # Applying the binary operator '*' (line 47)
    result_mul_130808 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '*', result_add_130804, result_sub_130807)
    
    # Assigning a type to the variable 'stypy_return_type' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type', result_mul_130808)
    
    # ################# End of '_nbytes_full(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nbytes_full' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_130809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_130809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nbytes_full'
    return stypy_return_type_130809

# Assigning a type to the variable '_nbytes_full' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), '_nbytes_full', _nbytes_full)
# Declaration of the 'HBInfo' class

class HBInfo(object, ):

    @norecursion
    def from_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_130810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'str', 'Default title')
        str_130811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 53), 'str', '0')
        # Getting the type of 'None' (line 52)
        None_130812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 65), 'None')
        # Getting the type of 'None' (line 52)
        None_130813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 75), 'None')
        defaults = [str_130810, str_130811, None_130812, None_130813]
        # Create a new context for function 'from_data'
        module_type_store = module_type_store.open_function_context('from_data', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBInfo.from_data.__dict__.__setitem__('stypy_localization', localization)
        HBInfo.from_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBInfo.from_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBInfo.from_data.__dict__.__setitem__('stypy_function_name', 'HBInfo.from_data')
        HBInfo.from_data.__dict__.__setitem__('stypy_param_names_list', ['m', 'title', 'key', 'mxtype', 'fmt'])
        HBInfo.from_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBInfo.from_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBInfo.from_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBInfo.from_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBInfo.from_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBInfo.from_data.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBInfo.from_data', ['m', 'title', 'key', 'mxtype', 'fmt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_data', localization, ['m', 'title', 'key', 'mxtype', 'fmt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_data(...)' code ##################

        str_130814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', 'Create a HBInfo instance from an existing sparse matrix.\n\n        Parameters\n        ----------\n        m : sparse matrix\n            the HBInfo instance will derive its parameters from m\n        title : str\n            Title to put in the HB header\n        key : str\n            Key\n        mxtype : HBMatrixType\n            type of the input matrix\n        fmt : dict\n            not implemented\n\n        Returns\n        -------\n        hb_info : HBInfo instance\n        ')
        
        # Assigning a Attribute to a Name (line 72):
        
        # Assigning a Attribute to a Name (line 72):
        # Getting the type of 'm' (line 72)
        m_130815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'm')
        # Obtaining the member 'indptr' of a type (line 72)
        indptr_130816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 18), m_130815, 'indptr')
        # Assigning a type to the variable 'pointer' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'pointer', indptr_130816)
        
        # Assigning a Attribute to a Name (line 73):
        
        # Assigning a Attribute to a Name (line 73):
        # Getting the type of 'm' (line 73)
        m_130817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'm')
        # Obtaining the member 'indices' of a type (line 73)
        indices_130818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), m_130817, 'indices')
        # Assigning a type to the variable 'indices' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'indices', indices_130818)
        
        # Assigning a Attribute to a Name (line 74):
        
        # Assigning a Attribute to a Name (line 74):
        # Getting the type of 'm' (line 74)
        m_130819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'm')
        # Obtaining the member 'data' of a type (line 74)
        data_130820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), m_130819, 'data')
        # Assigning a type to the variable 'values' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'values', data_130820)
        
        # Assigning a Attribute to a Tuple (line 76):
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_130821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
        # Getting the type of 'm' (line 76)
        m_130822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'm')
        # Obtaining the member 'shape' of a type (line 76)
        shape_130823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), m_130822, 'shape')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___130824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), shape_130823, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_130825 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___130824, int_130821)
        
        # Assigning a type to the variable 'tuple_var_assignment_130777' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_130777', subscript_call_result_130825)
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_130826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
        # Getting the type of 'm' (line 76)
        m_130827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'm')
        # Obtaining the member 'shape' of a type (line 76)
        shape_130828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 23), m_130827, 'shape')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___130829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), shape_130828, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_130830 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___130829, int_130826)
        
        # Assigning a type to the variable 'tuple_var_assignment_130778' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_130778', subscript_call_result_130830)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_130777' (line 76)
        tuple_var_assignment_130777_130831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_130777')
        # Assigning a type to the variable 'nrows' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'nrows', tuple_var_assignment_130777_130831)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_130778' (line 76)
        tuple_var_assignment_130778_130832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_130778')
        # Assigning a type to the variable 'ncols' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'ncols', tuple_var_assignment_130778_130832)
        
        # Assigning a Attribute to a Name (line 77):
        
        # Assigning a Attribute to a Name (line 77):
        # Getting the type of 'm' (line 77)
        m_130833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'm')
        # Obtaining the member 'nnz' of a type (line 77)
        nnz_130834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 21), m_130833, 'nnz')
        # Assigning a type to the variable 'nnon_zeros' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'nnon_zeros', nnz_130834)
        
        # Type idiom detected: calculating its left and rigth part (line 79)
        # Getting the type of 'fmt' (line 79)
        fmt_130835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'fmt')
        # Getting the type of 'None' (line 79)
        None_130836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'None')
        
        (may_be_130837, more_types_in_union_130838) = may_be_none(fmt_130835, None_130836)

        if may_be_130837:

            if more_types_in_union_130838:
                # Runtime conditional SSA (line 79)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 82):
            
            # Assigning a Call to a Name (line 82):
            
            # Call to from_number(...): (line 82)
            # Processing the call arguments (line 82)
            
            # Call to max(...): (line 82)
            # Processing the call arguments (line 82)
            # Getting the type of 'pointer' (line 82)
            pointer_130843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'pointer', False)
            int_130844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 63), 'int')
            # Applying the binary operator '+' (line 82)
            result_add_130845 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 55), '+', pointer_130843, int_130844)
            
            # Processing the call keyword arguments (line 82)
            kwargs_130846 = {}
            # Getting the type of 'np' (line 82)
            np_130841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'np', False)
            # Obtaining the member 'max' of a type (line 82)
            max_130842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 48), np_130841, 'max')
            # Calling max(args, kwargs) (line 82)
            max_call_result_130847 = invoke(stypy.reporting.localization.Localization(__file__, 82, 48), max_130842, *[result_add_130845], **kwargs_130846)
            
            # Processing the call keyword arguments (line 82)
            kwargs_130848 = {}
            # Getting the type of 'IntFormat' (line 82)
            IntFormat_130839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'IntFormat', False)
            # Obtaining the member 'from_number' of a type (line 82)
            from_number_130840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), IntFormat_130839, 'from_number')
            # Calling from_number(args, kwargs) (line 82)
            from_number_call_result_130849 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), from_number_130840, *[max_call_result_130847], **kwargs_130848)
            
            # Assigning a type to the variable 'pointer_fmt' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'pointer_fmt', from_number_call_result_130849)
            
            # Assigning a Call to a Name (line 83):
            
            # Assigning a Call to a Name (line 83):
            
            # Call to from_number(...): (line 83)
            # Processing the call arguments (line 83)
            
            # Call to max(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'indices' (line 83)
            indices_130854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 55), 'indices', False)
            int_130855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 63), 'int')
            # Applying the binary operator '+' (line 83)
            result_add_130856 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 55), '+', indices_130854, int_130855)
            
            # Processing the call keyword arguments (line 83)
            kwargs_130857 = {}
            # Getting the type of 'np' (line 83)
            np_130852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 48), 'np', False)
            # Obtaining the member 'max' of a type (line 83)
            max_130853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 48), np_130852, 'max')
            # Calling max(args, kwargs) (line 83)
            max_call_result_130858 = invoke(stypy.reporting.localization.Localization(__file__, 83, 48), max_130853, *[result_add_130856], **kwargs_130857)
            
            # Processing the call keyword arguments (line 83)
            kwargs_130859 = {}
            # Getting the type of 'IntFormat' (line 83)
            IntFormat_130850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'IntFormat', False)
            # Obtaining the member 'from_number' of a type (line 83)
            from_number_130851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 26), IntFormat_130850, 'from_number')
            # Calling from_number(args, kwargs) (line 83)
            from_number_call_result_130860 = invoke(stypy.reporting.localization.Localization(__file__, 83, 26), from_number_130851, *[max_call_result_130858], **kwargs_130859)
            
            # Assigning a type to the variable 'indices_fmt' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'indices_fmt', from_number_call_result_130860)
            
            
            # Getting the type of 'values' (line 85)
            values_130861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'values')
            # Obtaining the member 'dtype' of a type (line 85)
            dtype_130862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), values_130861, 'dtype')
            # Obtaining the member 'kind' of a type (line 85)
            kind_130863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), dtype_130862, 'kind')
            
            # Obtaining the type of the subscript
            str_130864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 49), 'str', 'AllFloat')
            # Getting the type of 'np' (line 85)
            np_130865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'np')
            # Obtaining the member 'typecodes' of a type (line 85)
            typecodes_130866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 36), np_130865, 'typecodes')
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___130867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 36), typecodes_130866, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_130868 = invoke(stypy.reporting.localization.Localization(__file__, 85, 36), getitem___130867, str_130864)
            
            # Applying the binary operator 'in' (line 85)
            result_contains_130869 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 15), 'in', kind_130863, subscript_call_result_130868)
            
            # Testing the type of an if condition (line 85)
            if_condition_130870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 12), result_contains_130869)
            # Assigning a type to the variable 'if_condition_130870' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'if_condition_130870', if_condition_130870)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 86):
            
            # Assigning a Call to a Name (line 86):
            
            # Call to from_number(...): (line 86)
            # Processing the call arguments (line 86)
            
            
            # Call to max(...): (line 86)
            # Processing the call arguments (line 86)
            
            # Call to abs(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'values' (line 86)
            values_130877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 66), 'values', False)
            # Processing the call keyword arguments (line 86)
            kwargs_130878 = {}
            # Getting the type of 'np' (line 86)
            np_130875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 59), 'np', False)
            # Obtaining the member 'abs' of a type (line 86)
            abs_130876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 59), np_130875, 'abs')
            # Calling abs(args, kwargs) (line 86)
            abs_call_result_130879 = invoke(stypy.reporting.localization.Localization(__file__, 86, 59), abs_130876, *[values_130877], **kwargs_130878)
            
            # Processing the call keyword arguments (line 86)
            kwargs_130880 = {}
            # Getting the type of 'np' (line 86)
            np_130873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 52), 'np', False)
            # Obtaining the member 'max' of a type (line 86)
            max_130874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 52), np_130873, 'max')
            # Calling max(args, kwargs) (line 86)
            max_call_result_130881 = invoke(stypy.reporting.localization.Localization(__file__, 86, 52), max_130874, *[abs_call_result_130879], **kwargs_130880)
            
            # Applying the 'usub' unary operator (line 86)
            result___neg___130882 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 51), 'usub', max_call_result_130881)
            
            # Processing the call keyword arguments (line 86)
            kwargs_130883 = {}
            # Getting the type of 'ExpFormat' (line 86)
            ExpFormat_130871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'ExpFormat', False)
            # Obtaining the member 'from_number' of a type (line 86)
            from_number_130872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 29), ExpFormat_130871, 'from_number')
            # Calling from_number(args, kwargs) (line 86)
            from_number_call_result_130884 = invoke(stypy.reporting.localization.Localization(__file__, 86, 29), from_number_130872, *[result___neg___130882], **kwargs_130883)
            
            # Assigning a type to the variable 'values_fmt' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'values_fmt', from_number_call_result_130884)
            # SSA branch for the else part of an if statement (line 85)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'values' (line 87)
            values_130885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'values')
            # Obtaining the member 'dtype' of a type (line 87)
            dtype_130886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 17), values_130885, 'dtype')
            # Obtaining the member 'kind' of a type (line 87)
            kind_130887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 17), dtype_130886, 'kind')
            
            # Obtaining the type of the subscript
            str_130888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 51), 'str', 'AllInteger')
            # Getting the type of 'np' (line 87)
            np_130889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 38), 'np')
            # Obtaining the member 'typecodes' of a type (line 87)
            typecodes_130890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 38), np_130889, 'typecodes')
            # Obtaining the member '__getitem__' of a type (line 87)
            getitem___130891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 38), typecodes_130890, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 87)
            subscript_call_result_130892 = invoke(stypy.reporting.localization.Localization(__file__, 87, 38), getitem___130891, str_130888)
            
            # Applying the binary operator 'in' (line 87)
            result_contains_130893 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 17), 'in', kind_130887, subscript_call_result_130892)
            
            # Testing the type of an if condition (line 87)
            if_condition_130894 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 17), result_contains_130893)
            # Assigning a type to the variable 'if_condition_130894' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'if_condition_130894', if_condition_130894)
            # SSA begins for if statement (line 87)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 88):
            
            # Assigning a Call to a Name (line 88):
            
            # Call to from_number(...): (line 88)
            # Processing the call arguments (line 88)
            
            
            # Call to max(...): (line 88)
            # Processing the call arguments (line 88)
            
            # Call to abs(...): (line 88)
            # Processing the call arguments (line 88)
            # Getting the type of 'values' (line 88)
            values_130901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 66), 'values', False)
            # Processing the call keyword arguments (line 88)
            kwargs_130902 = {}
            # Getting the type of 'np' (line 88)
            np_130899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'np', False)
            # Obtaining the member 'abs' of a type (line 88)
            abs_130900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 59), np_130899, 'abs')
            # Calling abs(args, kwargs) (line 88)
            abs_call_result_130903 = invoke(stypy.reporting.localization.Localization(__file__, 88, 59), abs_130900, *[values_130901], **kwargs_130902)
            
            # Processing the call keyword arguments (line 88)
            kwargs_130904 = {}
            # Getting the type of 'np' (line 88)
            np_130897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 52), 'np', False)
            # Obtaining the member 'max' of a type (line 88)
            max_130898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 52), np_130897, 'max')
            # Calling max(args, kwargs) (line 88)
            max_call_result_130905 = invoke(stypy.reporting.localization.Localization(__file__, 88, 52), max_130898, *[abs_call_result_130903], **kwargs_130904)
            
            # Applying the 'usub' unary operator (line 88)
            result___neg___130906 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 51), 'usub', max_call_result_130905)
            
            # Processing the call keyword arguments (line 88)
            kwargs_130907 = {}
            # Getting the type of 'IntFormat' (line 88)
            IntFormat_130895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'IntFormat', False)
            # Obtaining the member 'from_number' of a type (line 88)
            from_number_130896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 29), IntFormat_130895, 'from_number')
            # Calling from_number(args, kwargs) (line 88)
            from_number_call_result_130908 = invoke(stypy.reporting.localization.Localization(__file__, 88, 29), from_number_130896, *[result___neg___130906], **kwargs_130907)
            
            # Assigning a type to the variable 'values_fmt' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'values_fmt', from_number_call_result_130908)
            # SSA branch for the else part of an if statement (line 87)
            module_type_store.open_ssa_branch('else')
            
            # Call to NotImplementedError(...): (line 90)
            # Processing the call arguments (line 90)
            str_130910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 42), 'str', 'type %s not implemented yet')
            # Getting the type of 'values' (line 90)
            values_130911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 74), 'values', False)
            # Obtaining the member 'dtype' of a type (line 90)
            dtype_130912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 74), values_130911, 'dtype')
            # Obtaining the member 'kind' of a type (line 90)
            kind_130913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 74), dtype_130912, 'kind')
            # Applying the binary operator '%' (line 90)
            result_mod_130914 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 42), '%', str_130910, kind_130913)
            
            # Processing the call keyword arguments (line 90)
            kwargs_130915 = {}
            # Getting the type of 'NotImplementedError' (line 90)
            NotImplementedError_130909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'NotImplementedError', False)
            # Calling NotImplementedError(args, kwargs) (line 90)
            NotImplementedError_call_result_130916 = invoke(stypy.reporting.localization.Localization(__file__, 90, 22), NotImplementedError_130909, *[result_mod_130914], **kwargs_130915)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 90, 16), NotImplementedError_call_result_130916, 'raise parameter', BaseException)
            # SSA join for if statement (line 87)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_130838:
                # Runtime conditional SSA for else branch (line 79)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_130837) or more_types_in_union_130838):
            
            # Call to NotImplementedError(...): (line 92)
            # Processing the call arguments (line 92)
            str_130918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 38), 'str', 'fmt argument not supported yet.')
            # Processing the call keyword arguments (line 92)
            kwargs_130919 = {}
            # Getting the type of 'NotImplementedError' (line 92)
            NotImplementedError_130917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'NotImplementedError', False)
            # Calling NotImplementedError(args, kwargs) (line 92)
            NotImplementedError_call_result_130920 = invoke(stypy.reporting.localization.Localization(__file__, 92, 18), NotImplementedError_130917, *[str_130918], **kwargs_130919)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 92, 12), NotImplementedError_call_result_130920, 'raise parameter', BaseException)

            if (may_be_130837 and more_types_in_union_130838):
                # SSA join for if statement (line 79)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 94)
        # Getting the type of 'mxtype' (line 94)
        mxtype_130921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'mxtype')
        # Getting the type of 'None' (line 94)
        None_130922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'None')
        
        (may_be_130923, more_types_in_union_130924) = may_be_none(mxtype_130921, None_130922)

        if may_be_130923:

            if more_types_in_union_130924:
                # Runtime conditional SSA (line 94)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            
            # Call to isrealobj(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'values' (line 95)
            values_130927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'values', False)
            # Processing the call keyword arguments (line 95)
            kwargs_130928 = {}
            # Getting the type of 'np' (line 95)
            np_130925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'np', False)
            # Obtaining the member 'isrealobj' of a type (line 95)
            isrealobj_130926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 19), np_130925, 'isrealobj')
            # Calling isrealobj(args, kwargs) (line 95)
            isrealobj_call_result_130929 = invoke(stypy.reporting.localization.Localization(__file__, 95, 19), isrealobj_130926, *[values_130927], **kwargs_130928)
            
            # Applying the 'not' unary operator (line 95)
            result_not__130930 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), 'not', isrealobj_call_result_130929)
            
            # Testing the type of an if condition (line 95)
            if_condition_130931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 12), result_not__130930)
            # Assigning a type to the variable 'if_condition_130931' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'if_condition_130931', if_condition_130931)
            # SSA begins for if statement (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 96)
            # Processing the call arguments (line 96)
            str_130933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 33), 'str', 'Complex values not supported yet')
            # Processing the call keyword arguments (line 96)
            kwargs_130934 = {}
            # Getting the type of 'ValueError' (line 96)
            ValueError_130932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 96)
            ValueError_call_result_130935 = invoke(stypy.reporting.localization.Localization(__file__, 96, 22), ValueError_130932, *[str_130933], **kwargs_130934)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 96, 16), ValueError_call_result_130935, 'raise parameter', BaseException)
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'values' (line 97)
            values_130936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'values')
            # Obtaining the member 'dtype' of a type (line 97)
            dtype_130937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), values_130936, 'dtype')
            # Obtaining the member 'kind' of a type (line 97)
            kind_130938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), dtype_130937, 'kind')
            
            # Obtaining the type of the subscript
            str_130939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 49), 'str', 'AllInteger')
            # Getting the type of 'np' (line 97)
            np_130940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'np')
            # Obtaining the member 'typecodes' of a type (line 97)
            typecodes_130941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 36), np_130940, 'typecodes')
            # Obtaining the member '__getitem__' of a type (line 97)
            getitem___130942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 36), typecodes_130941, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 97)
            subscript_call_result_130943 = invoke(stypy.reporting.localization.Localization(__file__, 97, 36), getitem___130942, str_130939)
            
            # Applying the binary operator 'in' (line 97)
            result_contains_130944 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), 'in', kind_130938, subscript_call_result_130943)
            
            # Testing the type of an if condition (line 97)
            if_condition_130945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), result_contains_130944)
            # Assigning a type to the variable 'if_condition_130945' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_130945', if_condition_130945)
            # SSA begins for if statement (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 98):
            
            # Assigning a Str to a Name (line 98):
            str_130946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 21), 'str', 'integer')
            # Assigning a type to the variable 'tp' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'tp', str_130946)
            # SSA branch for the else part of an if statement (line 97)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'values' (line 99)
            values_130947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'values')
            # Obtaining the member 'dtype' of a type (line 99)
            dtype_130948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), values_130947, 'dtype')
            # Obtaining the member 'kind' of a type (line 99)
            kind_130949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), dtype_130948, 'kind')
            
            # Obtaining the type of the subscript
            str_130950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 51), 'str', 'AllFloat')
            # Getting the type of 'np' (line 99)
            np_130951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'np')
            # Obtaining the member 'typecodes' of a type (line 99)
            typecodes_130952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), np_130951, 'typecodes')
            # Obtaining the member '__getitem__' of a type (line 99)
            getitem___130953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), typecodes_130952, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 99)
            subscript_call_result_130954 = invoke(stypy.reporting.localization.Localization(__file__, 99, 38), getitem___130953, str_130950)
            
            # Applying the binary operator 'in' (line 99)
            result_contains_130955 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 17), 'in', kind_130949, subscript_call_result_130954)
            
            # Testing the type of an if condition (line 99)
            if_condition_130956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 17), result_contains_130955)
            # Assigning a type to the variable 'if_condition_130956' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'if_condition_130956', if_condition_130956)
            # SSA begins for if statement (line 99)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 100):
            
            # Assigning a Str to a Name (line 100):
            str_130957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'str', 'real')
            # Assigning a type to the variable 'tp' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'tp', str_130957)
            # SSA branch for the else part of an if statement (line 99)
            module_type_store.open_ssa_branch('else')
            
            # Call to NotImplementedError(...): (line 102)
            # Processing the call arguments (line 102)
            str_130959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'str', 'type %s for values not implemented')
            # Getting the type of 'values' (line 103)
            values_130960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 44), 'values', False)
            # Obtaining the member 'dtype' of a type (line 103)
            dtype_130961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 44), values_130960, 'dtype')
            # Applying the binary operator '%' (line 102)
            result_mod_130962 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 42), '%', str_130959, dtype_130961)
            
            # Processing the call keyword arguments (line 102)
            kwargs_130963 = {}
            # Getting the type of 'NotImplementedError' (line 102)
            NotImplementedError_130958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'NotImplementedError', False)
            # Calling NotImplementedError(args, kwargs) (line 102)
            NotImplementedError_call_result_130964 = invoke(stypy.reporting.localization.Localization(__file__, 102, 22), NotImplementedError_130958, *[result_mod_130962], **kwargs_130963)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 102, 16), NotImplementedError_call_result_130964, 'raise parameter', BaseException)
            # SSA join for if statement (line 99)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 104):
            
            # Assigning a Call to a Name (line 104):
            
            # Call to HBMatrixType(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'tp' (line 104)
            tp_130966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'tp', False)
            str_130967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 38), 'str', 'unsymmetric')
            str_130968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 53), 'str', 'assembled')
            # Processing the call keyword arguments (line 104)
            kwargs_130969 = {}
            # Getting the type of 'HBMatrixType' (line 104)
            HBMatrixType_130965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'HBMatrixType', False)
            # Calling HBMatrixType(args, kwargs) (line 104)
            HBMatrixType_call_result_130970 = invoke(stypy.reporting.localization.Localization(__file__, 104, 21), HBMatrixType_130965, *[tp_130966, str_130967, str_130968], **kwargs_130969)
            
            # Assigning a type to the variable 'mxtype' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'mxtype', HBMatrixType_call_result_130970)

            if more_types_in_union_130924:
                # Runtime conditional SSA for else branch (line 94)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_130923) or more_types_in_union_130924):
            
            # Call to ValueError(...): (line 106)
            # Processing the call arguments (line 106)
            str_130972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 29), 'str', 'mxtype argument not handled yet.')
            # Processing the call keyword arguments (line 106)
            kwargs_130973 = {}
            # Getting the type of 'ValueError' (line 106)
            ValueError_130971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 106)
            ValueError_call_result_130974 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), ValueError_130971, *[str_130972], **kwargs_130973)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 106, 12), ValueError_call_result_130974, 'raise parameter', BaseException)

            if (may_be_130923 and more_types_in_union_130924):
                # SSA join for if statement (line 94)
                module_type_store = module_type_store.join_ssa_context()


        

        @norecursion
        def _nlines(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_nlines'
            module_type_store = module_type_store.open_function_context('_nlines', 108, 8, False)
            
            # Passed parameters checking function
            _nlines.stypy_localization = localization
            _nlines.stypy_type_of_self = None
            _nlines.stypy_type_store = module_type_store
            _nlines.stypy_function_name = '_nlines'
            _nlines.stypy_param_names_list = ['fmt', 'size']
            _nlines.stypy_varargs_param_name = None
            _nlines.stypy_kwargs_param_name = None
            _nlines.stypy_call_defaults = defaults
            _nlines.stypy_call_varargs = varargs
            _nlines.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_nlines', ['fmt', 'size'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_nlines', localization, ['fmt', 'size'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_nlines(...)' code ##################

            
            # Assigning a BinOp to a Name (line 109):
            
            # Assigning a BinOp to a Name (line 109):
            # Getting the type of 'size' (line 109)
            size_130975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'size')
            # Getting the type of 'fmt' (line 109)
            fmt_130976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 29), 'fmt')
            # Obtaining the member 'repeat' of a type (line 109)
            repeat_130977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 29), fmt_130976, 'repeat')
            # Applying the binary operator '//' (line 109)
            result_floordiv_130978 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 21), '//', size_130975, repeat_130977)
            
            # Assigning a type to the variable 'nlines' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'nlines', result_floordiv_130978)
            
            
            # Getting the type of 'nlines' (line 110)
            nlines_130979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'nlines')
            # Getting the type of 'fmt' (line 110)
            fmt_130980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'fmt')
            # Obtaining the member 'repeat' of a type (line 110)
            repeat_130981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 24), fmt_130980, 'repeat')
            # Applying the binary operator '*' (line 110)
            result_mul_130982 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), '*', nlines_130979, repeat_130981)
            
            # Getting the type of 'size' (line 110)
            size_130983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 38), 'size')
            # Applying the binary operator '!=' (line 110)
            result_ne_130984 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), '!=', result_mul_130982, size_130983)
            
            # Testing the type of an if condition (line 110)
            if_condition_130985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 12), result_ne_130984)
            # Assigning a type to the variable 'if_condition_130985' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'if_condition_130985', if_condition_130985)
            # SSA begins for if statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'nlines' (line 111)
            nlines_130986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'nlines')
            int_130987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 26), 'int')
            # Applying the binary operator '+=' (line 111)
            result_iadd_130988 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), '+=', nlines_130986, int_130987)
            # Assigning a type to the variable 'nlines' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'nlines', result_iadd_130988)
            
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'nlines' (line 112)
            nlines_130989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'nlines')
            # Assigning a type to the variable 'stypy_return_type' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'stypy_return_type', nlines_130989)
            
            # ################# End of '_nlines(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_nlines' in the type store
            # Getting the type of 'stypy_return_type' (line 108)
            stypy_return_type_130990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_130990)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_nlines'
            return stypy_return_type_130990

        # Assigning a type to the variable '_nlines' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), '_nlines', _nlines)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to _nlines(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'pointer_fmt' (line 114)
        pointer_fmt_130992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'pointer_fmt', False)
        # Getting the type of 'pointer' (line 114)
        pointer_130993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'pointer', False)
        # Obtaining the member 'size' of a type (line 114)
        size_130994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 46), pointer_130993, 'size')
        # Processing the call keyword arguments (line 114)
        kwargs_130995 = {}
        # Getting the type of '_nlines' (line 114)
        _nlines_130991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), '_nlines', False)
        # Calling _nlines(args, kwargs) (line 114)
        _nlines_call_result_130996 = invoke(stypy.reporting.localization.Localization(__file__, 114, 25), _nlines_130991, *[pointer_fmt_130992, size_130994], **kwargs_130995)
        
        # Assigning a type to the variable 'pointer_nlines' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'pointer_nlines', _nlines_call_result_130996)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to _nlines(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'indices_fmt' (line 115)
        indices_fmt_130998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 33), 'indices_fmt', False)
        # Getting the type of 'indices' (line 115)
        indices_130999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 46), 'indices', False)
        # Obtaining the member 'size' of a type (line 115)
        size_131000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 46), indices_130999, 'size')
        # Processing the call keyword arguments (line 115)
        kwargs_131001 = {}
        # Getting the type of '_nlines' (line 115)
        _nlines_130997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), '_nlines', False)
        # Calling _nlines(args, kwargs) (line 115)
        _nlines_call_result_131002 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), _nlines_130997, *[indices_fmt_130998, size_131000], **kwargs_131001)
        
        # Assigning a type to the variable 'indices_nlines' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'indices_nlines', _nlines_call_result_131002)
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to _nlines(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'values_fmt' (line 116)
        values_fmt_131004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 32), 'values_fmt', False)
        # Getting the type of 'values' (line 116)
        values_131005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 44), 'values', False)
        # Obtaining the member 'size' of a type (line 116)
        size_131006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 44), values_131005, 'size')
        # Processing the call keyword arguments (line 116)
        kwargs_131007 = {}
        # Getting the type of '_nlines' (line 116)
        _nlines_131003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), '_nlines', False)
        # Calling _nlines(args, kwargs) (line 116)
        _nlines_call_result_131008 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), _nlines_131003, *[values_fmt_131004, size_131006], **kwargs_131007)
        
        # Assigning a type to the variable 'values_nlines' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'values_nlines', _nlines_call_result_131008)
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        # Getting the type of 'pointer_nlines' (line 118)
        pointer_nlines_131009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'pointer_nlines')
        # Getting the type of 'indices_nlines' (line 118)
        indices_nlines_131010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'indices_nlines')
        # Applying the binary operator '+' (line 118)
        result_add_131011 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 23), '+', pointer_nlines_131009, indices_nlines_131010)
        
        # Getting the type of 'values_nlines' (line 118)
        values_nlines_131012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 57), 'values_nlines')
        # Applying the binary operator '+' (line 118)
        result_add_131013 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 55), '+', result_add_131011, values_nlines_131012)
        
        # Assigning a type to the variable 'total_nlines' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'total_nlines', result_add_131013)
        
        # Call to cls(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'title' (line 120)
        title_131015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'title', False)
        # Getting the type of 'key' (line 120)
        key_131016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'key', False)
        # Getting the type of 'total_nlines' (line 121)
        total_nlines_131017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'total_nlines', False)
        # Getting the type of 'pointer_nlines' (line 121)
        pointer_nlines_131018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'pointer_nlines', False)
        # Getting the type of 'indices_nlines' (line 121)
        indices_nlines_131019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 'indices_nlines', False)
        # Getting the type of 'values_nlines' (line 121)
        values_nlines_131020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 58), 'values_nlines', False)
        # Getting the type of 'mxtype' (line 122)
        mxtype_131021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'mxtype', False)
        # Getting the type of 'nrows' (line 122)
        nrows_131022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'nrows', False)
        # Getting the type of 'ncols' (line 122)
        ncols_131023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'ncols', False)
        # Getting the type of 'nnon_zeros' (line 122)
        nnon_zeros_131024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 34), 'nnon_zeros', False)
        # Getting the type of 'pointer_fmt' (line 123)
        pointer_fmt_131025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'pointer_fmt', False)
        # Obtaining the member 'fortran_format' of a type (line 123)
        fortran_format_131026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), pointer_fmt_131025, 'fortran_format')
        # Getting the type of 'indices_fmt' (line 123)
        indices_fmt_131027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 40), 'indices_fmt', False)
        # Obtaining the member 'fortran_format' of a type (line 123)
        fortran_format_131028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 40), indices_fmt_131027, 'fortran_format')
        # Getting the type of 'values_fmt' (line 124)
        values_fmt_131029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'values_fmt', False)
        # Obtaining the member 'fortran_format' of a type (line 124)
        fortran_format_131030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), values_fmt_131029, 'fortran_format')
        # Processing the call keyword arguments (line 120)
        kwargs_131031 = {}
        # Getting the type of 'cls' (line 120)
        cls_131014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 120)
        cls_call_result_131032 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), cls_131014, *[title_131015, key_131016, total_nlines_131017, pointer_nlines_131018, indices_nlines_131019, values_nlines_131020, mxtype_131021, nrows_131022, ncols_131023, nnon_zeros_131024, fortran_format_131026, fortran_format_131028, fortran_format_131030], **kwargs_131031)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', cls_call_result_131032)
        
        # ################# End of 'from_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_data' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_131033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131033)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_data'
        return stypy_return_type_131033


    @norecursion
    def from_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'from_file'
        module_type_store = module_type_store.open_function_context('from_file', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBInfo.from_file.__dict__.__setitem__('stypy_localization', localization)
        HBInfo.from_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBInfo.from_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBInfo.from_file.__dict__.__setitem__('stypy_function_name', 'HBInfo.from_file')
        HBInfo.from_file.__dict__.__setitem__('stypy_param_names_list', ['fid'])
        HBInfo.from_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBInfo.from_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBInfo.from_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBInfo.from_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBInfo.from_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBInfo.from_file.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBInfo.from_file', ['fid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_file', localization, ['fid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_file(...)' code ##################

        str_131034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', 'Create a HBInfo instance from a file object containg a matrix in the\n        HB format.\n\n        Parameters\n        ----------\n        fid : file-like matrix\n            File or file-like object containing a matrix in the HB format.\n\n        Returns\n        -------\n        hb_info : HBInfo instance\n        ')
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to strip(...): (line 141)
        # Processing the call arguments (line 141)
        str_131040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 36), 'str', '\n')
        # Processing the call keyword arguments (line 141)
        kwargs_131041 = {}
        
        # Call to readline(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_131037 = {}
        # Getting the type of 'fid' (line 141)
        fid_131035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'fid', False)
        # Obtaining the member 'readline' of a type (line 141)
        readline_131036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), fid_131035, 'readline')
        # Calling readline(args, kwargs) (line 141)
        readline_call_result_131038 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), readline_131036, *[], **kwargs_131037)
        
        # Obtaining the member 'strip' of a type (line 141)
        strip_131039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), readline_call_result_131038, 'strip')
        # Calling strip(args, kwargs) (line 141)
        strip_call_result_131042 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), strip_131039, *[str_131040], **kwargs_131041)
        
        # Assigning a type to the variable 'line' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'line', strip_call_result_131042)
        
        
        
        
        # Call to len(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'line' (line 142)
        line_131044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'line', False)
        # Processing the call keyword arguments (line 142)
        kwargs_131045 = {}
        # Getting the type of 'len' (line 142)
        len_131043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'len', False)
        # Calling len(args, kwargs) (line 142)
        len_call_result_131046 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), len_131043, *[line_131044], **kwargs_131045)
        
        int_131047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'int')
        # Applying the binary operator '>' (line 142)
        result_gt_131048 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 15), '>', len_call_result_131046, int_131047)
        
        # Applying the 'not' unary operator (line 142)
        result_not__131049 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), 'not', result_gt_131048)
        
        # Testing the type of an if condition (line 142)
        if_condition_131050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), result_not__131049)
        # Assigning a type to the variable 'if_condition_131050' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_131050', if_condition_131050)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 143)
        # Processing the call arguments (line 143)
        str_131052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'str', 'Expected at least 72 characters for first line, got: \n%s')
        # Getting the type of 'line' (line 144)
        line_131053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 43), 'line', False)
        # Applying the binary operator '%' (line 143)
        result_mod_131054 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 29), '%', str_131052, line_131053)
        
        # Processing the call keyword arguments (line 143)
        kwargs_131055 = {}
        # Getting the type of 'ValueError' (line 143)
        ValueError_131051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 143)
        ValueError_call_result_131056 = invoke(stypy.reporting.localization.Localization(__file__, 143, 18), ValueError_131051, *[result_mod_131054], **kwargs_131055)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 143, 12), ValueError_call_result_131056, 'raise parameter', BaseException)
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 145):
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        int_131057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 22), 'int')
        slice_131058 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 145, 16), None, int_131057, None)
        # Getting the type of 'line' (line 145)
        line_131059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'line')
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___131060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 16), line_131059, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_131061 = invoke(stypy.reporting.localization.Localization(__file__, 145, 16), getitem___131060, slice_131058)
        
        # Assigning a type to the variable 'title' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'title', subscript_call_result_131061)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_131062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
        slice_131063 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 14), int_131062, None, None)
        # Getting the type of 'line' (line 146)
        line_131064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 14), 'line')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___131065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 14), line_131064, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_131066 = invoke(stypy.reporting.localization.Localization(__file__, 146, 14), getitem___131065, slice_131063)
        
        # Assigning a type to the variable 'key' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'key', subscript_call_result_131066)
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to strip(...): (line 149)
        # Processing the call arguments (line 149)
        str_131072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 36), 'str', '\n')
        # Processing the call keyword arguments (line 149)
        kwargs_131073 = {}
        
        # Call to readline(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_131069 = {}
        # Getting the type of 'fid' (line 149)
        fid_131067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'fid', False)
        # Obtaining the member 'readline' of a type (line 149)
        readline_131068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), fid_131067, 'readline')
        # Calling readline(args, kwargs) (line 149)
        readline_call_result_131070 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), readline_131068, *[], **kwargs_131069)
        
        # Obtaining the member 'strip' of a type (line 149)
        strip_131071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), readline_call_result_131070, 'strip')
        # Calling strip(args, kwargs) (line 149)
        strip_call_result_131074 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), strip_131071, *[str_131072], **kwargs_131073)
        
        # Assigning a type to the variable 'line' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'line', strip_call_result_131074)
        
        
        
        
        # Call to len(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Call to rstrip(...): (line 150)
        # Processing the call keyword arguments (line 150)
        kwargs_131078 = {}
        # Getting the type of 'line' (line 150)
        line_131076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'line', False)
        # Obtaining the member 'rstrip' of a type (line 150)
        rstrip_131077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 19), line_131076, 'rstrip')
        # Calling rstrip(args, kwargs) (line 150)
        rstrip_call_result_131079 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), rstrip_131077, *[], **kwargs_131078)
        
        # Processing the call keyword arguments (line 150)
        kwargs_131080 = {}
        # Getting the type of 'len' (line 150)
        len_131075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'len', False)
        # Calling len(args, kwargs) (line 150)
        len_call_result_131081 = invoke(stypy.reporting.localization.Localization(__file__, 150, 15), len_131075, *[rstrip_call_result_131079], **kwargs_131080)
        
        int_131082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 37), 'int')
        # Applying the binary operator '>=' (line 150)
        result_ge_131083 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '>=', len_call_result_131081, int_131082)
        
        # Applying the 'not' unary operator (line 150)
        result_not__131084 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), 'not', result_ge_131083)
        
        # Testing the type of an if condition (line 150)
        if_condition_131085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_not__131084)
        # Assigning a type to the variable 'if_condition_131085' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_131085', if_condition_131085)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 151)
        # Processing the call arguments (line 151)
        str_131087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'str', 'Expected at least 56 characters for second line, got: \n%s')
        # Getting the type of 'line' (line 152)
        line_131088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 43), 'line', False)
        # Applying the binary operator '%' (line 151)
        result_mod_131089 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 29), '%', str_131087, line_131088)
        
        # Processing the call keyword arguments (line 151)
        kwargs_131090 = {}
        # Getting the type of 'ValueError' (line 151)
        ValueError_131086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 151)
        ValueError_call_result_131091 = invoke(stypy.reporting.localization.Localization(__file__, 151, 18), ValueError_131086, *[result_mod_131089], **kwargs_131090)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 151, 12), ValueError_call_result_131091, 'raise parameter', BaseException)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to _expect_int(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Obtaining the type of the subscript
        int_131093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 41), 'int')
        slice_131094 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 35), None, int_131093, None)
        # Getting the type of 'line' (line 153)
        line_131095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 35), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___131096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 35), line_131095, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_131097 = invoke(stypy.reporting.localization.Localization(__file__, 153, 35), getitem___131096, slice_131094)
        
        # Processing the call keyword arguments (line 153)
        kwargs_131098 = {}
        # Getting the type of '_expect_int' (line 153)
        _expect_int_131092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 153)
        _expect_int_call_result_131099 = invoke(stypy.reporting.localization.Localization(__file__, 153, 23), _expect_int_131092, *[subscript_call_result_131097], **kwargs_131098)
        
        # Assigning a type to the variable 'total_nlines' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'total_nlines', _expect_int_call_result_131099)
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to _expect_int(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining the type of the subscript
        int_131101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 42), 'int')
        int_131102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 45), 'int')
        slice_131103 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 154, 37), int_131101, int_131102, None)
        # Getting the type of 'line' (line 154)
        line_131104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___131105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 37), line_131104, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_131106 = invoke(stypy.reporting.localization.Localization(__file__, 154, 37), getitem___131105, slice_131103)
        
        # Processing the call keyword arguments (line 154)
        kwargs_131107 = {}
        # Getting the type of '_expect_int' (line 154)
        _expect_int_131100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 154)
        _expect_int_call_result_131108 = invoke(stypy.reporting.localization.Localization(__file__, 154, 25), _expect_int_131100, *[subscript_call_result_131106], **kwargs_131107)
        
        # Assigning a type to the variable 'pointer_nlines' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'pointer_nlines', _expect_int_call_result_131108)
        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to _expect_int(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining the type of the subscript
        int_131110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 42), 'int')
        int_131111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 45), 'int')
        slice_131112 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 155, 37), int_131110, int_131111, None)
        # Getting the type of 'line' (line 155)
        line_131113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 37), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___131114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 37), line_131113, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_131115 = invoke(stypy.reporting.localization.Localization(__file__, 155, 37), getitem___131114, slice_131112)
        
        # Processing the call keyword arguments (line 155)
        kwargs_131116 = {}
        # Getting the type of '_expect_int' (line 155)
        _expect_int_131109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 155)
        _expect_int_call_result_131117 = invoke(stypy.reporting.localization.Localization(__file__, 155, 25), _expect_int_131109, *[subscript_call_result_131115], **kwargs_131116)
        
        # Assigning a type to the variable 'indices_nlines' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'indices_nlines', _expect_int_call_result_131117)
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to _expect_int(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining the type of the subscript
        int_131119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 41), 'int')
        int_131120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'int')
        slice_131121 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 36), int_131119, int_131120, None)
        # Getting the type of 'line' (line 156)
        line_131122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 36), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___131123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 36), line_131122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_131124 = invoke(stypy.reporting.localization.Localization(__file__, 156, 36), getitem___131123, slice_131121)
        
        # Processing the call keyword arguments (line 156)
        kwargs_131125 = {}
        # Getting the type of '_expect_int' (line 156)
        _expect_int_131118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 156)
        _expect_int_call_result_131126 = invoke(stypy.reporting.localization.Localization(__file__, 156, 24), _expect_int_131118, *[subscript_call_result_131124], **kwargs_131125)
        
        # Assigning a type to the variable 'values_nlines' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'values_nlines', _expect_int_call_result_131126)
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to strip(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_131134 = {}
        
        # Obtaining the type of the subscript
        int_131127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'int')
        int_131128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 29), 'int')
        slice_131129 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 21), int_131127, int_131128, None)
        # Getting the type of 'line' (line 158)
        line_131130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___131131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 21), line_131130, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_131132 = invoke(stypy.reporting.localization.Localization(__file__, 158, 21), getitem___131131, slice_131129)
        
        # Obtaining the member 'strip' of a type (line 158)
        strip_131133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 21), subscript_call_result_131132, 'strip')
        # Calling strip(args, kwargs) (line 158)
        strip_call_result_131135 = invoke(stypy.reporting.localization.Localization(__file__, 158, 21), strip_131133, *[], **kwargs_131134)
        
        # Assigning a type to the variable 'rhs_nlines' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'rhs_nlines', strip_call_result_131135)
        
        
        # Getting the type of 'rhs_nlines' (line 159)
        rhs_nlines_131136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'rhs_nlines')
        str_131137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'str', '')
        # Applying the binary operator '==' (line 159)
        result_eq_131138 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 11), '==', rhs_nlines_131136, str_131137)
        
        # Testing the type of an if condition (line 159)
        if_condition_131139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), result_eq_131138)
        # Assigning a type to the variable 'if_condition_131139' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_131139', if_condition_131139)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 160):
        
        # Assigning a Num to a Name (line 160):
        int_131140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'int')
        # Assigning a type to the variable 'rhs_nlines' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'rhs_nlines', int_131140)
        # SSA branch for the else part of an if statement (line 159)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to _expect_int(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'rhs_nlines' (line 162)
        rhs_nlines_131142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'rhs_nlines', False)
        # Processing the call keyword arguments (line 162)
        kwargs_131143 = {}
        # Getting the type of '_expect_int' (line 162)
        _expect_int_131141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 162)
        _expect_int_call_result_131144 = invoke(stypy.reporting.localization.Localization(__file__, 162, 25), _expect_int_131141, *[rhs_nlines_131142], **kwargs_131143)
        
        # Assigning a type to the variable 'rhs_nlines' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'rhs_nlines', _expect_int_call_result_131144)
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Getting the type of 'rhs_nlines' (line 163)
        rhs_nlines_131145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'rhs_nlines')
        int_131146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 29), 'int')
        # Applying the binary operator '==' (line 163)
        result_eq_131147 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 15), '==', rhs_nlines_131145, int_131146)
        
        # Applying the 'not' unary operator (line 163)
        result_not__131148 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), 'not', result_eq_131147)
        
        # Testing the type of an if condition (line 163)
        if_condition_131149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), result_not__131148)
        # Assigning a type to the variable 'if_condition_131149' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_131149', if_condition_131149)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 164)
        # Processing the call arguments (line 164)
        str_131151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'str', 'Only files without right hand side supported for now.')
        # Processing the call keyword arguments (line 164)
        kwargs_131152 = {}
        # Getting the type of 'ValueError' (line 164)
        ValueError_131150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 164)
        ValueError_call_result_131153 = invoke(stypy.reporting.localization.Localization(__file__, 164, 18), ValueError_131150, *[str_131151], **kwargs_131152)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 12), ValueError_call_result_131153, 'raise parameter', BaseException)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to strip(...): (line 168)
        # Processing the call arguments (line 168)
        str_131159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 36), 'str', '\n')
        # Processing the call keyword arguments (line 168)
        kwargs_131160 = {}
        
        # Call to readline(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_131156 = {}
        # Getting the type of 'fid' (line 168)
        fid_131154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'fid', False)
        # Obtaining the member 'readline' of a type (line 168)
        readline_131155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), fid_131154, 'readline')
        # Calling readline(args, kwargs) (line 168)
        readline_call_result_131157 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), readline_131155, *[], **kwargs_131156)
        
        # Obtaining the member 'strip' of a type (line 168)
        strip_131158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), readline_call_result_131157, 'strip')
        # Calling strip(args, kwargs) (line 168)
        strip_call_result_131161 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), strip_131158, *[str_131159], **kwargs_131160)
        
        # Assigning a type to the variable 'line' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'line', strip_call_result_131161)
        
        
        
        
        # Call to len(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'line' (line 169)
        line_131163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'line', False)
        # Processing the call keyword arguments (line 169)
        kwargs_131164 = {}
        # Getting the type of 'len' (line 169)
        len_131162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'len', False)
        # Calling len(args, kwargs) (line 169)
        len_call_result_131165 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), len_131162, *[line_131163], **kwargs_131164)
        
        int_131166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 28), 'int')
        # Applying the binary operator '>=' (line 169)
        result_ge_131167 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), '>=', len_call_result_131165, int_131166)
        
        # Applying the 'not' unary operator (line 169)
        result_not__131168 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'not', result_ge_131167)
        
        # Testing the type of an if condition (line 169)
        if_condition_131169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 8), result_not__131168)
        # Assigning a type to the variable 'if_condition_131169' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'if_condition_131169', if_condition_131169)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 170)
        # Processing the call arguments (line 170)
        str_131171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 29), 'str', 'Expected at least 72 character for third line, got:\n%s')
        # Getting the type of 'line' (line 171)
        line_131172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'line', False)
        # Applying the binary operator '%' (line 170)
        result_mod_131173 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 29), '%', str_131171, line_131172)
        
        # Processing the call keyword arguments (line 170)
        kwargs_131174 = {}
        # Getting the type of 'ValueError' (line 170)
        ValueError_131170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 170)
        ValueError_call_result_131175 = invoke(stypy.reporting.localization.Localization(__file__, 170, 18), ValueError_131170, *[result_mod_131173], **kwargs_131174)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 170, 12), ValueError_call_result_131175, 'raise parameter', BaseException)
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to upper(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_131182 = {}
        
        # Obtaining the type of the subscript
        int_131176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'int')
        slice_131177 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 173, 19), None, int_131176, None)
        # Getting the type of 'line' (line 173)
        line_131178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___131179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 19), line_131178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_131180 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), getitem___131179, slice_131177)
        
        # Obtaining the member 'upper' of a type (line 173)
        upper_131181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 19), subscript_call_result_131180, 'upper')
        # Calling upper(args, kwargs) (line 173)
        upper_call_result_131183 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), upper_131181, *[], **kwargs_131182)
        
        # Assigning a type to the variable 'mxtype_s' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'mxtype_s', upper_call_result_131183)
        
        
        
        
        # Call to len(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'mxtype_s' (line 174)
        mxtype_s_131185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 19), 'mxtype_s', False)
        # Processing the call keyword arguments (line 174)
        kwargs_131186 = {}
        # Getting the type of 'len' (line 174)
        len_131184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'len', False)
        # Calling len(args, kwargs) (line 174)
        len_call_result_131187 = invoke(stypy.reporting.localization.Localization(__file__, 174, 15), len_131184, *[mxtype_s_131185], **kwargs_131186)
        
        int_131188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'int')
        # Applying the binary operator '==' (line 174)
        result_eq_131189 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '==', len_call_result_131187, int_131188)
        
        # Applying the 'not' unary operator (line 174)
        result_not__131190 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 11), 'not', result_eq_131189)
        
        # Testing the type of an if condition (line 174)
        if_condition_131191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 8), result_not__131190)
        # Assigning a type to the variable 'if_condition_131191' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'if_condition_131191', if_condition_131191)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 175)
        # Processing the call arguments (line 175)
        str_131193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'str', 'mxtype expected to be 3 characters long')
        # Processing the call keyword arguments (line 175)
        kwargs_131194 = {}
        # Getting the type of 'ValueError' (line 175)
        ValueError_131192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 175)
        ValueError_call_result_131195 = invoke(stypy.reporting.localization.Localization(__file__, 175, 18), ValueError_131192, *[str_131193], **kwargs_131194)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 12), ValueError_call_result_131195, 'raise parameter', BaseException)
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to from_fortran(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'mxtype_s' (line 177)
        mxtype_s_131198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), 'mxtype_s', False)
        # Processing the call keyword arguments (line 177)
        kwargs_131199 = {}
        # Getting the type of 'HBMatrixType' (line 177)
        HBMatrixType_131196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'HBMatrixType', False)
        # Obtaining the member 'from_fortran' of a type (line 177)
        from_fortran_131197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 17), HBMatrixType_131196, 'from_fortran')
        # Calling from_fortran(args, kwargs) (line 177)
        from_fortran_call_result_131200 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), from_fortran_131197, *[mxtype_s_131198], **kwargs_131199)
        
        # Assigning a type to the variable 'mxtype' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'mxtype', from_fortran_call_result_131200)
        
        
        # Getting the type of 'mxtype' (line 178)
        mxtype_131201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'mxtype')
        # Obtaining the member 'value_type' of a type (line 178)
        value_type_131202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 11), mxtype_131201, 'value_type')
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_131203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        str_131204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 37), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 36), list_131203, str_131204)
        # Adding element type (line 178)
        str_131205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 45), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 36), list_131203, str_131205)
        
        # Applying the binary operator 'notin' (line 178)
        result_contains_131206 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 11), 'notin', value_type_131202, list_131203)
        
        # Testing the type of an if condition (line 178)
        if_condition_131207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 8), result_contains_131206)
        # Assigning a type to the variable 'if_condition_131207' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'if_condition_131207', if_condition_131207)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 179)
        # Processing the call arguments (line 179)
        str_131209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'str', 'Only real or integer matrices supported for now (detected %s)')
        # Getting the type of 'mxtype' (line 180)
        mxtype_131210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 51), 'mxtype', False)
        # Applying the binary operator '%' (line 179)
        result_mod_131211 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 29), '%', str_131209, mxtype_131210)
        
        # Processing the call keyword arguments (line 179)
        kwargs_131212 = {}
        # Getting the type of 'ValueError' (line 179)
        ValueError_131208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 179)
        ValueError_call_result_131213 = invoke(stypy.reporting.localization.Localization(__file__, 179, 18), ValueError_131208, *[result_mod_131211], **kwargs_131212)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 179, 12), ValueError_call_result_131213, 'raise parameter', BaseException)
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Getting the type of 'mxtype' (line 181)
        mxtype_131214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'mxtype')
        # Obtaining the member 'structure' of a type (line 181)
        structure_131215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), mxtype_131214, 'structure')
        str_131216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 35), 'str', 'unsymmetric')
        # Applying the binary operator '==' (line 181)
        result_eq_131217 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 15), '==', structure_131215, str_131216)
        
        # Applying the 'not' unary operator (line 181)
        result_not__131218 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'not', result_eq_131217)
        
        # Testing the type of an if condition (line 181)
        if_condition_131219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_not__131218)
        # Assigning a type to the variable 'if_condition_131219' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_131219', if_condition_131219)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 182)
        # Processing the call arguments (line 182)
        str_131221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 29), 'str', 'Only unsymmetric matrices supported for now (detected %s)')
        # Getting the type of 'mxtype' (line 183)
        mxtype_131222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 51), 'mxtype', False)
        # Applying the binary operator '%' (line 182)
        result_mod_131223 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 29), '%', str_131221, mxtype_131222)
        
        # Processing the call keyword arguments (line 182)
        kwargs_131224 = {}
        # Getting the type of 'ValueError' (line 182)
        ValueError_131220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 182)
        ValueError_call_result_131225 = invoke(stypy.reporting.localization.Localization(__file__, 182, 18), ValueError_131220, *[result_mod_131223], **kwargs_131224)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 182, 12), ValueError_call_result_131225, 'raise parameter', BaseException)
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Getting the type of 'mxtype' (line 184)
        mxtype_131226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'mxtype')
        # Obtaining the member 'storage' of a type (line 184)
        storage_131227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 15), mxtype_131226, 'storage')
        str_131228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 33), 'str', 'assembled')
        # Applying the binary operator '==' (line 184)
        result_eq_131229 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), '==', storage_131227, str_131228)
        
        # Applying the 'not' unary operator (line 184)
        result_not__131230 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), 'not', result_eq_131229)
        
        # Testing the type of an if condition (line 184)
        if_condition_131231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), result_not__131230)
        # Assigning a type to the variable 'if_condition_131231' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_131231', if_condition_131231)
        # SSA begins for if statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 185)
        # Processing the call arguments (line 185)
        str_131233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 29), 'str', 'Only assembled matrices supported for now')
        # Processing the call keyword arguments (line 185)
        kwargs_131234 = {}
        # Getting the type of 'ValueError' (line 185)
        ValueError_131232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 185)
        ValueError_call_result_131235 = invoke(stypy.reporting.localization.Localization(__file__, 185, 18), ValueError_131232, *[str_131233], **kwargs_131234)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 185, 12), ValueError_call_result_131235, 'raise parameter', BaseException)
        # SSA join for if statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        
        # Obtaining the type of the subscript
        int_131236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'int')
        int_131237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 22), 'int')
        slice_131238 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 15), int_131236, int_131237, None)
        # Getting the type of 'line' (line 187)
        line_131239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'line')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___131240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), line_131239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_131241 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), getitem___131240, slice_131238)
        
        str_131242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 29), 'str', ' ')
        int_131243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 35), 'int')
        # Applying the binary operator '*' (line 187)
        result_mul_131244 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 29), '*', str_131242, int_131243)
        
        # Applying the binary operator '==' (line 187)
        result_eq_131245 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 15), '==', subscript_call_result_131241, result_mul_131244)
        
        # Applying the 'not' unary operator (line 187)
        result_not__131246 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 11), 'not', result_eq_131245)
        
        # Testing the type of an if condition (line 187)
        if_condition_131247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), result_not__131246)
        # Assigning a type to the variable 'if_condition_131247' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_131247', if_condition_131247)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 188)
        # Processing the call arguments (line 188)
        str_131249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'str', 'Malformed data for third line: %s')
        # Getting the type of 'line' (line 188)
        line_131250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 67), 'line', False)
        # Applying the binary operator '%' (line 188)
        result_mod_131251 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 29), '%', str_131249, line_131250)
        
        # Processing the call keyword arguments (line 188)
        kwargs_131252 = {}
        # Getting the type of 'ValueError' (line 188)
        ValueError_131248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 188)
        ValueError_call_result_131253 = invoke(stypy.reporting.localization.Localization(__file__, 188, 18), ValueError_131248, *[result_mod_131251], **kwargs_131252)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 188, 12), ValueError_call_result_131253, 'raise parameter', BaseException)
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to _expect_int(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Obtaining the type of the subscript
        int_131255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 33), 'int')
        int_131256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 36), 'int')
        slice_131257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 190, 28), int_131255, int_131256, None)
        # Getting the type of 'line' (line 190)
        line_131258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 28), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___131259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 28), line_131258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_131260 = invoke(stypy.reporting.localization.Localization(__file__, 190, 28), getitem___131259, slice_131257)
        
        # Processing the call keyword arguments (line 190)
        kwargs_131261 = {}
        # Getting the type of '_expect_int' (line 190)
        _expect_int_131254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 190)
        _expect_int_call_result_131262 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), _expect_int_131254, *[subscript_call_result_131260], **kwargs_131261)
        
        # Assigning a type to the variable 'nrows' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'nrows', _expect_int_call_result_131262)
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to _expect_int(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Obtaining the type of the subscript
        int_131264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 33), 'int')
        int_131265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 36), 'int')
        slice_131266 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 191, 28), int_131264, int_131265, None)
        # Getting the type of 'line' (line 191)
        line_131267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___131268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 28), line_131267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_131269 = invoke(stypy.reporting.localization.Localization(__file__, 191, 28), getitem___131268, slice_131266)
        
        # Processing the call keyword arguments (line 191)
        kwargs_131270 = {}
        # Getting the type of '_expect_int' (line 191)
        _expect_int_131263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 191)
        _expect_int_call_result_131271 = invoke(stypy.reporting.localization.Localization(__file__, 191, 16), _expect_int_131263, *[subscript_call_result_131269], **kwargs_131270)
        
        # Assigning a type to the variable 'ncols' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'ncols', _expect_int_call_result_131271)
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to _expect_int(...): (line 192)
        # Processing the call arguments (line 192)
        
        # Obtaining the type of the subscript
        int_131273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 38), 'int')
        int_131274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 41), 'int')
        slice_131275 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 192, 33), int_131273, int_131274, None)
        # Getting the type of 'line' (line 192)
        line_131276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 33), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___131277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 33), line_131276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_131278 = invoke(stypy.reporting.localization.Localization(__file__, 192, 33), getitem___131277, slice_131275)
        
        # Processing the call keyword arguments (line 192)
        kwargs_131279 = {}
        # Getting the type of '_expect_int' (line 192)
        _expect_int_131272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 192)
        _expect_int_call_result_131280 = invoke(stypy.reporting.localization.Localization(__file__, 192, 21), _expect_int_131272, *[subscript_call_result_131278], **kwargs_131279)
        
        # Assigning a type to the variable 'nnon_zeros' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'nnon_zeros', _expect_int_call_result_131280)
        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to _expect_int(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Obtaining the type of the subscript
        int_131282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 39), 'int')
        int_131283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 42), 'int')
        slice_131284 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 193, 34), int_131282, int_131283, None)
        # Getting the type of 'line' (line 193)
        line_131285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 34), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___131286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 34), line_131285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_131287 = invoke(stypy.reporting.localization.Localization(__file__, 193, 34), getitem___131286, slice_131284)
        
        # Processing the call keyword arguments (line 193)
        kwargs_131288 = {}
        # Getting the type of '_expect_int' (line 193)
        _expect_int_131281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), '_expect_int', False)
        # Calling _expect_int(args, kwargs) (line 193)
        _expect_int_call_result_131289 = invoke(stypy.reporting.localization.Localization(__file__, 193, 22), _expect_int_131281, *[subscript_call_result_131287], **kwargs_131288)
        
        # Assigning a type to the variable 'nelementals' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'nelementals', _expect_int_call_result_131289)
        
        
        
        # Getting the type of 'nelementals' (line 194)
        nelementals_131290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'nelementals')
        int_131291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 30), 'int')
        # Applying the binary operator '==' (line 194)
        result_eq_131292 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 15), '==', nelementals_131290, int_131291)
        
        # Applying the 'not' unary operator (line 194)
        result_not__131293 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), 'not', result_eq_131292)
        
        # Testing the type of an if condition (line 194)
        if_condition_131294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), result_not__131293)
        # Assigning a type to the variable 'if_condition_131294' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_131294', if_condition_131294)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 195)
        # Processing the call arguments (line 195)
        str_131296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 29), 'str', 'Unexpected value %d for nltvl (last entry of line 3)')
        # Getting the type of 'nelementals' (line 196)
        nelementals_131297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'nelementals', False)
        # Applying the binary operator '%' (line 195)
        result_mod_131298 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 29), '%', str_131296, nelementals_131297)
        
        # Processing the call keyword arguments (line 195)
        kwargs_131299 = {}
        # Getting the type of 'ValueError' (line 195)
        ValueError_131295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 195)
        ValueError_call_result_131300 = invoke(stypy.reporting.localization.Localization(__file__, 195, 18), ValueError_131295, *[result_mod_131298], **kwargs_131299)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 195, 12), ValueError_call_result_131300, 'raise parameter', BaseException)
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to strip(...): (line 199)
        # Processing the call arguments (line 199)
        str_131306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'str', '\n')
        # Processing the call keyword arguments (line 199)
        kwargs_131307 = {}
        
        # Call to readline(...): (line 199)
        # Processing the call keyword arguments (line 199)
        kwargs_131303 = {}
        # Getting the type of 'fid' (line 199)
        fid_131301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'fid', False)
        # Obtaining the member 'readline' of a type (line 199)
        readline_131302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 15), fid_131301, 'readline')
        # Calling readline(args, kwargs) (line 199)
        readline_call_result_131304 = invoke(stypy.reporting.localization.Localization(__file__, 199, 15), readline_131302, *[], **kwargs_131303)
        
        # Obtaining the member 'strip' of a type (line 199)
        strip_131305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 15), readline_call_result_131304, 'strip')
        # Calling strip(args, kwargs) (line 199)
        strip_call_result_131308 = invoke(stypy.reporting.localization.Localization(__file__, 199, 15), strip_131305, *[str_131306], **kwargs_131307)
        
        # Assigning a type to the variable 'line' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'line', strip_call_result_131308)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to split(...): (line 201)
        # Processing the call keyword arguments (line 201)
        kwargs_131311 = {}
        # Getting the type of 'line' (line 201)
        line_131309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'line', False)
        # Obtaining the member 'split' of a type (line 201)
        split_131310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 13), line_131309, 'split')
        # Calling split(args, kwargs) (line 201)
        split_call_result_131312 = invoke(stypy.reporting.localization.Localization(__file__, 201, 13), split_131310, *[], **kwargs_131311)
        
        # Assigning a type to the variable 'ct' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'ct', split_call_result_131312)
        
        
        
        
        # Call to len(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'ct' (line 202)
        ct_131314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'ct', False)
        # Processing the call keyword arguments (line 202)
        kwargs_131315 = {}
        # Getting the type of 'len' (line 202)
        len_131313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'len', False)
        # Calling len(args, kwargs) (line 202)
        len_call_result_131316 = invoke(stypy.reporting.localization.Localization(__file__, 202, 15), len_131313, *[ct_131314], **kwargs_131315)
        
        int_131317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 26), 'int')
        # Applying the binary operator '==' (line 202)
        result_eq_131318 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 15), '==', len_call_result_131316, int_131317)
        
        # Applying the 'not' unary operator (line 202)
        result_not__131319 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 11), 'not', result_eq_131318)
        
        # Testing the type of an if condition (line 202)
        if_condition_131320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), result_not__131319)
        # Assigning a type to the variable 'if_condition_131320' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_131320', if_condition_131320)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 203)
        # Processing the call arguments (line 203)
        str_131322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 29), 'str', 'Expected 3 formats, got %s')
        # Getting the type of 'ct' (line 203)
        ct_131323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 60), 'ct', False)
        # Applying the binary operator '%' (line 203)
        result_mod_131324 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 29), '%', str_131322, ct_131323)
        
        # Processing the call keyword arguments (line 203)
        kwargs_131325 = {}
        # Getting the type of 'ValueError' (line 203)
        ValueError_131321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 203)
        ValueError_call_result_131326 = invoke(stypy.reporting.localization.Localization(__file__, 203, 18), ValueError_131321, *[result_mod_131324], **kwargs_131325)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 203, 12), ValueError_call_result_131326, 'raise parameter', BaseException)
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cls(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'title' (line 205)
        title_131328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'title', False)
        # Getting the type of 'key' (line 205)
        key_131329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'key', False)
        # Getting the type of 'total_nlines' (line 206)
        total_nlines_131330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'total_nlines', False)
        # Getting the type of 'pointer_nlines' (line 206)
        pointer_nlines_131331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'pointer_nlines', False)
        # Getting the type of 'indices_nlines' (line 206)
        indices_nlines_131332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 49), 'indices_nlines', False)
        # Getting the type of 'values_nlines' (line 206)
        values_nlines_131333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 65), 'values_nlines', False)
        # Getting the type of 'mxtype' (line 207)
        mxtype_131334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'mxtype', False)
        # Getting the type of 'nrows' (line 207)
        nrows_131335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'nrows', False)
        # Getting the type of 'ncols' (line 207)
        ncols_131336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 34), 'ncols', False)
        # Getting the type of 'nnon_zeros' (line 207)
        nnon_zeros_131337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 41), 'nnon_zeros', False)
        
        # Obtaining the type of the subscript
        int_131338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'int')
        # Getting the type of 'ct' (line 208)
        ct_131339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 'ct', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___131340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 19), ct_131339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_131341 = invoke(stypy.reporting.localization.Localization(__file__, 208, 19), getitem___131340, int_131338)
        
        
        # Obtaining the type of the subscript
        int_131342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'int')
        # Getting the type of 'ct' (line 208)
        ct_131343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'ct', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___131344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 26), ct_131343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_131345 = invoke(stypy.reporting.localization.Localization(__file__, 208, 26), getitem___131344, int_131342)
        
        
        # Obtaining the type of the subscript
        int_131346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 36), 'int')
        # Getting the type of 'ct' (line 208)
        ct_131347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 33), 'ct', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___131348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 33), ct_131347, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_131349 = invoke(stypy.reporting.localization.Localization(__file__, 208, 33), getitem___131348, int_131346)
        
        # Getting the type of 'rhs_nlines' (line 209)
        rhs_nlines_131350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'rhs_nlines', False)
        # Getting the type of 'nelementals' (line 209)
        nelementals_131351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 31), 'nelementals', False)
        # Processing the call keyword arguments (line 205)
        kwargs_131352 = {}
        # Getting the type of 'cls' (line 205)
        cls_131327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 205)
        cls_call_result_131353 = invoke(stypy.reporting.localization.Localization(__file__, 205, 15), cls_131327, *[title_131328, key_131329, total_nlines_131330, pointer_nlines_131331, indices_nlines_131332, values_nlines_131333, mxtype_131334, nrows_131335, ncols_131336, nnon_zeros_131337, subscript_call_result_131341, subscript_call_result_131345, subscript_call_result_131349, rhs_nlines_131350, nelementals_131351], **kwargs_131352)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', cls_call_result_131353)
        
        # ################# End of 'from_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_file' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_131354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_file'
        return stypy_return_type_131354


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_131355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 36), 'int')
        int_131356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 51), 'int')
        defaults = [int_131355, int_131356]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBInfo.__init__', ['title', 'key', 'total_nlines', 'pointer_nlines', 'indices_nlines', 'values_nlines', 'mxtype', 'nrows', 'ncols', 'nnon_zeros', 'pointer_format_str', 'indices_format_str', 'values_format_str', 'right_hand_sides_nlines', 'nelementals'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['title', 'key', 'total_nlines', 'pointer_nlines', 'indices_nlines', 'values_nlines', 'mxtype', 'nrows', 'ncols', 'nnon_zeros', 'pointer_format_str', 'indices_format_str', 'values_format_str', 'right_hand_sides_nlines', 'nelementals'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_131357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'str', 'Do not use this directly, but the class ctrs (from_* functions).')
        
        # Assigning a Name to a Attribute (line 217):
        
        # Assigning a Name to a Attribute (line 217):
        # Getting the type of 'title' (line 217)
        title_131358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'title')
        # Getting the type of 'self' (line 217)
        self_131359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'self')
        # Setting the type of the member 'title' of a type (line 217)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), self_131359, 'title', title_131358)
        
        # Assigning a Name to a Attribute (line 218):
        
        # Assigning a Name to a Attribute (line 218):
        # Getting the type of 'key' (line 218)
        key_131360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'key')
        # Getting the type of 'self' (line 218)
        self_131361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self')
        # Setting the type of the member 'key' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_131361, 'key', key_131360)
        
        # Type idiom detected: calculating its left and rigth part (line 219)
        # Getting the type of 'title' (line 219)
        title_131362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'title')
        # Getting the type of 'None' (line 219)
        None_131363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'None')
        
        (may_be_131364, more_types_in_union_131365) = may_be_none(title_131362, None_131363)

        if may_be_131364:

            if more_types_in_union_131365:
                # Runtime conditional SSA (line 219)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 220):
            
            # Assigning a Str to a Name (line 220):
            str_131366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'str', 'No Title')
            # Assigning a type to the variable 'title' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'title', str_131366)

            if more_types_in_union_131365:
                # SSA join for if statement (line 219)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to len(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'title' (line 221)
        title_131368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'title', False)
        # Processing the call keyword arguments (line 221)
        kwargs_131369 = {}
        # Getting the type of 'len' (line 221)
        len_131367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'len', False)
        # Calling len(args, kwargs) (line 221)
        len_call_result_131370 = invoke(stypy.reporting.localization.Localization(__file__, 221, 11), len_131367, *[title_131368], **kwargs_131369)
        
        int_131371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 24), 'int')
        # Applying the binary operator '>' (line 221)
        result_gt_131372 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 11), '>', len_call_result_131370, int_131371)
        
        # Testing the type of an if condition (line 221)
        if_condition_131373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), result_gt_131372)
        # Assigning a type to the variable 'if_condition_131373' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_131373', if_condition_131373)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 222)
        # Processing the call arguments (line 222)
        str_131375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 29), 'str', 'title cannot be > 72 characters')
        # Processing the call keyword arguments (line 222)
        kwargs_131376 = {}
        # Getting the type of 'ValueError' (line 222)
        ValueError_131374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 222)
        ValueError_call_result_131377 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), ValueError_131374, *[str_131375], **kwargs_131376)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 222, 12), ValueError_call_result_131377, 'raise parameter', BaseException)
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 224)
        # Getting the type of 'key' (line 224)
        key_131378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'key')
        # Getting the type of 'None' (line 224)
        None_131379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'None')
        
        (may_be_131380, more_types_in_union_131381) = may_be_none(key_131378, None_131379)

        if may_be_131380:

            if more_types_in_union_131381:
                # Runtime conditional SSA (line 224)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 225):
            
            # Assigning a Str to a Name (line 225):
            str_131382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 18), 'str', '|No Key')
            # Assigning a type to the variable 'key' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'key', str_131382)

            if more_types_in_union_131381:
                # SSA join for if statement (line 224)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to len(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'key' (line 226)
        key_131384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'key', False)
        # Processing the call keyword arguments (line 226)
        kwargs_131385 = {}
        # Getting the type of 'len' (line 226)
        len_131383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'len', False)
        # Calling len(args, kwargs) (line 226)
        len_call_result_131386 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), len_131383, *[key_131384], **kwargs_131385)
        
        int_131387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 22), 'int')
        # Applying the binary operator '>' (line 226)
        result_gt_131388 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 11), '>', len_call_result_131386, int_131387)
        
        # Testing the type of an if condition (line 226)
        if_condition_131389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), result_gt_131388)
        # Assigning a type to the variable 'if_condition_131389' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_131389', if_condition_131389)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 227)
        # Processing the call arguments (line 227)
        str_131392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 26), 'str', 'key is > 8 characters (key is %s)')
        # Getting the type of 'key' (line 227)
        key_131393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 64), 'key', False)
        # Applying the binary operator '%' (line 227)
        result_mod_131394 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 26), '%', str_131392, key_131393)
        
        # Getting the type of 'LineOverflow' (line 227)
        LineOverflow_131395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 69), 'LineOverflow', False)
        # Processing the call keyword arguments (line 227)
        kwargs_131396 = {}
        # Getting the type of 'warnings' (line 227)
        warnings_131390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 227)
        warn_131391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), warnings_131390, 'warn')
        # Calling warn(args, kwargs) (line 227)
        warn_call_result_131397 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), warn_131391, *[result_mod_131394, LineOverflow_131395], **kwargs_131396)
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 229):
        
        # Assigning a Name to a Attribute (line 229):
        # Getting the type of 'total_nlines' (line 229)
        total_nlines_131398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 28), 'total_nlines')
        # Getting the type of 'self' (line 229)
        self_131399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self')
        # Setting the type of the member 'total_nlines' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_131399, 'total_nlines', total_nlines_131398)
        
        # Assigning a Name to a Attribute (line 230):
        
        # Assigning a Name to a Attribute (line 230):
        # Getting the type of 'pointer_nlines' (line 230)
        pointer_nlines_131400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 'pointer_nlines')
        # Getting the type of 'self' (line 230)
        self_131401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
        # Setting the type of the member 'pointer_nlines' of a type (line 230)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_131401, 'pointer_nlines', pointer_nlines_131400)
        
        # Assigning a Name to a Attribute (line 231):
        
        # Assigning a Name to a Attribute (line 231):
        # Getting the type of 'indices_nlines' (line 231)
        indices_nlines_131402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'indices_nlines')
        # Getting the type of 'self' (line 231)
        self_131403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'self')
        # Setting the type of the member 'indices_nlines' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), self_131403, 'indices_nlines', indices_nlines_131402)
        
        # Assigning a Name to a Attribute (line 232):
        
        # Assigning a Name to a Attribute (line 232):
        # Getting the type of 'values_nlines' (line 232)
        values_nlines_131404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 29), 'values_nlines')
        # Getting the type of 'self' (line 232)
        self_131405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self')
        # Setting the type of the member 'values_nlines' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_131405, 'values_nlines', values_nlines_131404)
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to FortranFormatParser(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_131407 = {}
        # Getting the type of 'FortranFormatParser' (line 234)
        FortranFormatParser_131406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'FortranFormatParser', False)
        # Calling FortranFormatParser(args, kwargs) (line 234)
        FortranFormatParser_call_result_131408 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), FortranFormatParser_131406, *[], **kwargs_131407)
        
        # Assigning a type to the variable 'parser' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'parser', FortranFormatParser_call_result_131408)
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to parse(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'pointer_format_str' (line 235)
        pointer_format_str_131411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 38), 'pointer_format_str', False)
        # Processing the call keyword arguments (line 235)
        kwargs_131412 = {}
        # Getting the type of 'parser' (line 235)
        parser_131409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'parser', False)
        # Obtaining the member 'parse' of a type (line 235)
        parse_131410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 25), parser_131409, 'parse')
        # Calling parse(args, kwargs) (line 235)
        parse_call_result_131413 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), parse_131410, *[pointer_format_str_131411], **kwargs_131412)
        
        # Assigning a type to the variable 'pointer_format' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'pointer_format', parse_call_result_131413)
        
        
        
        # Call to isinstance(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'pointer_format' (line 236)
        pointer_format_131415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 26), 'pointer_format', False)
        # Getting the type of 'IntFormat' (line 236)
        IntFormat_131416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 42), 'IntFormat', False)
        # Processing the call keyword arguments (line 236)
        kwargs_131417 = {}
        # Getting the type of 'isinstance' (line 236)
        isinstance_131414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 236)
        isinstance_call_result_131418 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), isinstance_131414, *[pointer_format_131415, IntFormat_131416], **kwargs_131417)
        
        # Applying the 'not' unary operator (line 236)
        result_not__131419 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), 'not', isinstance_call_result_131418)
        
        # Testing the type of an if condition (line 236)
        if_condition_131420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), result_not__131419)
        # Assigning a type to the variable 'if_condition_131420' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_131420', if_condition_131420)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 237)
        # Processing the call arguments (line 237)
        str_131422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 29), 'str', 'Expected int format for pointer format, got %s')
        # Getting the type of 'pointer_format' (line 238)
        pointer_format_131423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 31), 'pointer_format', False)
        # Applying the binary operator '%' (line 237)
        result_mod_131424 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 29), '%', str_131422, pointer_format_131423)
        
        # Processing the call keyword arguments (line 237)
        kwargs_131425 = {}
        # Getting the type of 'ValueError' (line 237)
        ValueError_131421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 237)
        ValueError_call_result_131426 = invoke(stypy.reporting.localization.Localization(__file__, 237, 18), ValueError_131421, *[result_mod_131424], **kwargs_131425)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 237, 12), ValueError_call_result_131426, 'raise parameter', BaseException)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 240):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to parse(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'indices_format_str' (line 240)
        indices_format_str_131429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 38), 'indices_format_str', False)
        # Processing the call keyword arguments (line 240)
        kwargs_131430 = {}
        # Getting the type of 'parser' (line 240)
        parser_131427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'parser', False)
        # Obtaining the member 'parse' of a type (line 240)
        parse_131428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 25), parser_131427, 'parse')
        # Calling parse(args, kwargs) (line 240)
        parse_call_result_131431 = invoke(stypy.reporting.localization.Localization(__file__, 240, 25), parse_131428, *[indices_format_str_131429], **kwargs_131430)
        
        # Assigning a type to the variable 'indices_format' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'indices_format', parse_call_result_131431)
        
        
        
        # Call to isinstance(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'indices_format' (line 241)
        indices_format_131433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'indices_format', False)
        # Getting the type of 'IntFormat' (line 241)
        IntFormat_131434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 42), 'IntFormat', False)
        # Processing the call keyword arguments (line 241)
        kwargs_131435 = {}
        # Getting the type of 'isinstance' (line 241)
        isinstance_131432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 241)
        isinstance_call_result_131436 = invoke(stypy.reporting.localization.Localization(__file__, 241, 15), isinstance_131432, *[indices_format_131433, IntFormat_131434], **kwargs_131435)
        
        # Applying the 'not' unary operator (line 241)
        result_not__131437 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 11), 'not', isinstance_call_result_131436)
        
        # Testing the type of an if condition (line 241)
        if_condition_131438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), result_not__131437)
        # Assigning a type to the variable 'if_condition_131438' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_131438', if_condition_131438)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 242)
        # Processing the call arguments (line 242)
        str_131440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 29), 'str', 'Expected int format for indices format, got %s')
        # Getting the type of 'indices_format' (line 243)
        indices_format_131441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'indices_format', False)
        # Applying the binary operator '%' (line 242)
        result_mod_131442 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 29), '%', str_131440, indices_format_131441)
        
        # Processing the call keyword arguments (line 242)
        kwargs_131443 = {}
        # Getting the type of 'ValueError' (line 242)
        ValueError_131439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 242)
        ValueError_call_result_131444 = invoke(stypy.reporting.localization.Localization(__file__, 242, 18), ValueError_131439, *[result_mod_131442], **kwargs_131443)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 242, 12), ValueError_call_result_131444, 'raise parameter', BaseException)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to parse(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'values_format_str' (line 245)
        values_format_str_131447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 37), 'values_format_str', False)
        # Processing the call keyword arguments (line 245)
        kwargs_131448 = {}
        # Getting the type of 'parser' (line 245)
        parser_131445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'parser', False)
        # Obtaining the member 'parse' of a type (line 245)
        parse_131446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 24), parser_131445, 'parse')
        # Calling parse(args, kwargs) (line 245)
        parse_call_result_131449 = invoke(stypy.reporting.localization.Localization(__file__, 245, 24), parse_131446, *[values_format_str_131447], **kwargs_131448)
        
        # Assigning a type to the variable 'values_format' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'values_format', parse_call_result_131449)
        
        
        # Call to isinstance(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'values_format' (line 246)
        values_format_131451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'values_format', False)
        # Getting the type of 'ExpFormat' (line 246)
        ExpFormat_131452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'ExpFormat', False)
        # Processing the call keyword arguments (line 246)
        kwargs_131453 = {}
        # Getting the type of 'isinstance' (line 246)
        isinstance_131450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 246)
        isinstance_call_result_131454 = invoke(stypy.reporting.localization.Localization(__file__, 246, 11), isinstance_131450, *[values_format_131451, ExpFormat_131452], **kwargs_131453)
        
        # Testing the type of an if condition (line 246)
        if_condition_131455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), isinstance_call_result_131454)
        # Assigning a type to the variable 'if_condition_131455' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_131455', if_condition_131455)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'mxtype' (line 247)
        mxtype_131456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'mxtype')
        # Obtaining the member 'value_type' of a type (line 247)
        value_type_131457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), mxtype_131456, 'value_type')
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_131458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        str_131459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 41), 'str', 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 40), list_131458, str_131459)
        # Adding element type (line 247)
        str_131460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 49), 'str', 'complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 40), list_131458, str_131460)
        
        # Applying the binary operator 'notin' (line 247)
        result_contains_131461 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 15), 'notin', value_type_131457, list_131458)
        
        # Testing the type of an if condition (line 247)
        if_condition_131462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 12), result_contains_131461)
        # Assigning a type to the variable 'if_condition_131462' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'if_condition_131462', if_condition_131462)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 248)
        # Processing the call arguments (line 248)
        str_131464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 33), 'str', 'Inconsistency between matrix type %s and value type %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 249)
        tuple_131465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 249)
        # Adding element type (line 249)
        # Getting the type of 'mxtype' (line 249)
        mxtype_131466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 52), 'mxtype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 52), tuple_131465, mxtype_131466)
        # Adding element type (line 249)
        # Getting the type of 'values_format' (line 249)
        values_format_131467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 60), 'values_format', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 52), tuple_131465, values_format_131467)
        
        # Applying the binary operator '%' (line 248)
        result_mod_131468 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 33), '%', str_131464, tuple_131465)
        
        # Processing the call keyword arguments (line 248)
        kwargs_131469 = {}
        # Getting the type of 'ValueError' (line 248)
        ValueError_131463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 248)
        ValueError_call_result_131470 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), ValueError_131463, *[result_mod_131468], **kwargs_131469)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 248, 16), ValueError_call_result_131470, 'raise parameter', BaseException)
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 250):
        
        # Assigning a Attribute to a Name (line 250):
        # Getting the type of 'np' (line 250)
        np_131471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'np')
        # Obtaining the member 'float64' of a type (line 250)
        float64_131472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 27), np_131471, 'float64')
        # Assigning a type to the variable 'values_dtype' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'values_dtype', float64_131472)
        # SSA branch for the else part of an if statement (line 246)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'values_format' (line 251)
        values_format_131474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'values_format', False)
        # Getting the type of 'IntFormat' (line 251)
        IntFormat_131475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 39), 'IntFormat', False)
        # Processing the call keyword arguments (line 251)
        kwargs_131476 = {}
        # Getting the type of 'isinstance' (line 251)
        isinstance_131473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 251)
        isinstance_call_result_131477 = invoke(stypy.reporting.localization.Localization(__file__, 251, 13), isinstance_131473, *[values_format_131474, IntFormat_131475], **kwargs_131476)
        
        # Testing the type of an if condition (line 251)
        if_condition_131478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 13), isinstance_call_result_131477)
        # Assigning a type to the variable 'if_condition_131478' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'if_condition_131478', if_condition_131478)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'mxtype' (line 252)
        mxtype_131479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'mxtype')
        # Obtaining the member 'value_type' of a type (line 252)
        value_type_131480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 15), mxtype_131479, 'value_type')
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_131481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        str_131482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 41), 'str', 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 40), list_131481, str_131482)
        
        # Applying the binary operator 'notin' (line 252)
        result_contains_131483 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 15), 'notin', value_type_131480, list_131481)
        
        # Testing the type of an if condition (line 252)
        if_condition_131484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 12), result_contains_131483)
        # Assigning a type to the variable 'if_condition_131484' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'if_condition_131484', if_condition_131484)
        # SSA begins for if statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 253)
        # Processing the call arguments (line 253)
        str_131486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 33), 'str', 'Inconsistency between matrix type %s and value type %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 254)
        tuple_131487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 254)
        # Adding element type (line 254)
        # Getting the type of 'mxtype' (line 254)
        mxtype_131488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 52), 'mxtype', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 52), tuple_131487, mxtype_131488)
        # Adding element type (line 254)
        # Getting the type of 'values_format' (line 254)
        values_format_131489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 60), 'values_format', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 52), tuple_131487, values_format_131489)
        
        # Applying the binary operator '%' (line 253)
        result_mod_131490 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 33), '%', str_131486, tuple_131487)
        
        # Processing the call keyword arguments (line 253)
        kwargs_131491 = {}
        # Getting the type of 'ValueError' (line 253)
        ValueError_131485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 253)
        ValueError_call_result_131492 = invoke(stypy.reporting.localization.Localization(__file__, 253, 22), ValueError_131485, *[result_mod_131490], **kwargs_131491)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 253, 16), ValueError_call_result_131492, 'raise parameter', BaseException)
        # SSA join for if statement (line 252)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 256):
        
        # Assigning a Name to a Name (line 256):
        # Getting the type of 'int' (line 256)
        int_131493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'int')
        # Assigning a type to the variable 'values_dtype' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'values_dtype', int_131493)
        # SSA branch for the else part of an if statement (line 251)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 258)
        # Processing the call arguments (line 258)
        str_131495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'str', 'Unsupported format for values %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 258)
        tuple_131496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 258)
        # Adding element type (line 258)
        # Getting the type of 'values_format' (line 258)
        values_format_131497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 67), 'values_format', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 67), tuple_131496, values_format_131497)
        
        # Applying the binary operator '%' (line 258)
        result_mod_131498 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 29), '%', str_131495, tuple_131496)
        
        # Processing the call keyword arguments (line 258)
        kwargs_131499 = {}
        # Getting the type of 'ValueError' (line 258)
        ValueError_131494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 258)
        ValueError_call_result_131500 = invoke(stypy.reporting.localization.Localization(__file__, 258, 18), ValueError_131494, *[result_mod_131498], **kwargs_131499)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 258, 12), ValueError_call_result_131500, 'raise parameter', BaseException)
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 260):
        
        # Assigning a Name to a Attribute (line 260):
        # Getting the type of 'pointer_format' (line 260)
        pointer_format_131501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 30), 'pointer_format')
        # Getting the type of 'self' (line 260)
        self_131502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'self')
        # Setting the type of the member 'pointer_format' of a type (line 260)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), self_131502, 'pointer_format', pointer_format_131501)
        
        # Assigning a Name to a Attribute (line 261):
        
        # Assigning a Name to a Attribute (line 261):
        # Getting the type of 'indices_format' (line 261)
        indices_format_131503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 30), 'indices_format')
        # Getting the type of 'self' (line 261)
        self_131504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self')
        # Setting the type of the member 'indices_format' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_131504, 'indices_format', indices_format_131503)
        
        # Assigning a Name to a Attribute (line 262):
        
        # Assigning a Name to a Attribute (line 262):
        # Getting the type of 'values_format' (line 262)
        values_format_131505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 29), 'values_format')
        # Getting the type of 'self' (line 262)
        self_131506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self')
        # Setting the type of the member 'values_format' of a type (line 262)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_131506, 'values_format', values_format_131505)
        
        # Assigning a Attribute to a Attribute (line 264):
        
        # Assigning a Attribute to a Attribute (line 264):
        # Getting the type of 'np' (line 264)
        np_131507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'np')
        # Obtaining the member 'int32' of a type (line 264)
        int32_131508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 29), np_131507, 'int32')
        # Getting the type of 'self' (line 264)
        self_131509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
        # Setting the type of the member 'pointer_dtype' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_131509, 'pointer_dtype', int32_131508)
        
        # Assigning a Attribute to a Attribute (line 265):
        
        # Assigning a Attribute to a Attribute (line 265):
        # Getting the type of 'np' (line 265)
        np_131510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 29), 'np')
        # Obtaining the member 'int32' of a type (line 265)
        int32_131511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 29), np_131510, 'int32')
        # Getting the type of 'self' (line 265)
        self_131512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self')
        # Setting the type of the member 'indices_dtype' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_131512, 'indices_dtype', int32_131511)
        
        # Assigning a Name to a Attribute (line 266):
        
        # Assigning a Name to a Attribute (line 266):
        # Getting the type of 'values_dtype' (line 266)
        values_dtype_131513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'values_dtype')
        # Getting the type of 'self' (line 266)
        self_131514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self')
        # Setting the type of the member 'values_dtype' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_131514, 'values_dtype', values_dtype_131513)
        
        # Assigning a Name to a Attribute (line 268):
        
        # Assigning a Name to a Attribute (line 268):
        # Getting the type of 'pointer_nlines' (line 268)
        pointer_nlines_131515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'pointer_nlines')
        # Getting the type of 'self' (line 268)
        self_131516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self')
        # Setting the type of the member 'pointer_nlines' of a type (line 268)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_131516, 'pointer_nlines', pointer_nlines_131515)
        
        # Assigning a Call to a Attribute (line 269):
        
        # Assigning a Call to a Attribute (line 269):
        
        # Call to _nbytes_full(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'pointer_format' (line 269)
        pointer_format_131518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 48), 'pointer_format', False)
        # Getting the type of 'pointer_nlines' (line 269)
        pointer_nlines_131519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 64), 'pointer_nlines', False)
        # Processing the call keyword arguments (line 269)
        kwargs_131520 = {}
        # Getting the type of '_nbytes_full' (line 269)
        _nbytes_full_131517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 35), '_nbytes_full', False)
        # Calling _nbytes_full(args, kwargs) (line 269)
        _nbytes_full_call_result_131521 = invoke(stypy.reporting.localization.Localization(__file__, 269, 35), _nbytes_full_131517, *[pointer_format_131518, pointer_nlines_131519], **kwargs_131520)
        
        # Getting the type of 'self' (line 269)
        self_131522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Setting the type of the member 'pointer_nbytes_full' of a type (line 269)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_131522, 'pointer_nbytes_full', _nbytes_full_call_result_131521)
        
        # Assigning a Name to a Attribute (line 271):
        
        # Assigning a Name to a Attribute (line 271):
        # Getting the type of 'indices_nlines' (line 271)
        indices_nlines_131523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 30), 'indices_nlines')
        # Getting the type of 'self' (line 271)
        self_131524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'self')
        # Setting the type of the member 'indices_nlines' of a type (line 271)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), self_131524, 'indices_nlines', indices_nlines_131523)
        
        # Assigning a Call to a Attribute (line 272):
        
        # Assigning a Call to a Attribute (line 272):
        
        # Call to _nbytes_full(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'indices_format' (line 272)
        indices_format_131526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 48), 'indices_format', False)
        # Getting the type of 'indices_nlines' (line 272)
        indices_nlines_131527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 64), 'indices_nlines', False)
        # Processing the call keyword arguments (line 272)
        kwargs_131528 = {}
        # Getting the type of '_nbytes_full' (line 272)
        _nbytes_full_131525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), '_nbytes_full', False)
        # Calling _nbytes_full(args, kwargs) (line 272)
        _nbytes_full_call_result_131529 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), _nbytes_full_131525, *[indices_format_131526, indices_nlines_131527], **kwargs_131528)
        
        # Getting the type of 'self' (line 272)
        self_131530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'indices_nbytes_full' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_131530, 'indices_nbytes_full', _nbytes_full_call_result_131529)
        
        # Assigning a Name to a Attribute (line 274):
        
        # Assigning a Name to a Attribute (line 274):
        # Getting the type of 'values_nlines' (line 274)
        values_nlines_131531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'values_nlines')
        # Getting the type of 'self' (line 274)
        self_131532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self')
        # Setting the type of the member 'values_nlines' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_131532, 'values_nlines', values_nlines_131531)
        
        # Assigning a Call to a Attribute (line 275):
        
        # Assigning a Call to a Attribute (line 275):
        
        # Call to _nbytes_full(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'values_format' (line 275)
        values_format_131534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 47), 'values_format', False)
        # Getting the type of 'values_nlines' (line 275)
        values_nlines_131535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 62), 'values_nlines', False)
        # Processing the call keyword arguments (line 275)
        kwargs_131536 = {}
        # Getting the type of '_nbytes_full' (line 275)
        _nbytes_full_131533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 34), '_nbytes_full', False)
        # Calling _nbytes_full(args, kwargs) (line 275)
        _nbytes_full_call_result_131537 = invoke(stypy.reporting.localization.Localization(__file__, 275, 34), _nbytes_full_131533, *[values_format_131534, values_nlines_131535], **kwargs_131536)
        
        # Getting the type of 'self' (line 275)
        self_131538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Setting the type of the member 'values_nbytes_full' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_131538, 'values_nbytes_full', _nbytes_full_call_result_131537)
        
        # Assigning a Name to a Attribute (line 277):
        
        # Assigning a Name to a Attribute (line 277):
        # Getting the type of 'nrows' (line 277)
        nrows_131539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'nrows')
        # Getting the type of 'self' (line 277)
        self_131540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'self')
        # Setting the type of the member 'nrows' of a type (line 277)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), self_131540, 'nrows', nrows_131539)
        
        # Assigning a Name to a Attribute (line 278):
        
        # Assigning a Name to a Attribute (line 278):
        # Getting the type of 'ncols' (line 278)
        ncols_131541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'ncols')
        # Getting the type of 'self' (line 278)
        self_131542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'self')
        # Setting the type of the member 'ncols' of a type (line 278)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), self_131542, 'ncols', ncols_131541)
        
        # Assigning a Name to a Attribute (line 279):
        
        # Assigning a Name to a Attribute (line 279):
        # Getting the type of 'nnon_zeros' (line 279)
        nnon_zeros_131543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 26), 'nnon_zeros')
        # Getting the type of 'self' (line 279)
        self_131544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'self')
        # Setting the type of the member 'nnon_zeros' of a type (line 279)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), self_131544, 'nnon_zeros', nnon_zeros_131543)
        
        # Assigning a Name to a Attribute (line 280):
        
        # Assigning a Name to a Attribute (line 280):
        # Getting the type of 'nelementals' (line 280)
        nelementals_131545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'nelementals')
        # Getting the type of 'self' (line 280)
        self_131546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'self')
        # Setting the type of the member 'nelementals' of a type (line 280)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), self_131546, 'nelementals', nelementals_131545)
        
        # Assigning a Name to a Attribute (line 281):
        
        # Assigning a Name to a Attribute (line 281):
        # Getting the type of 'mxtype' (line 281)
        mxtype_131547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'mxtype')
        # Getting the type of 'self' (line 281)
        self_131548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'self')
        # Setting the type of the member 'mxtype' of a type (line 281)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), self_131548, 'mxtype', mxtype_131547)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def dump(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump'
        module_type_store = module_type_store.open_function_context('dump', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBInfo.dump.__dict__.__setitem__('stypy_localization', localization)
        HBInfo.dump.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBInfo.dump.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBInfo.dump.__dict__.__setitem__('stypy_function_name', 'HBInfo.dump')
        HBInfo.dump.__dict__.__setitem__('stypy_param_names_list', [])
        HBInfo.dump.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBInfo.dump.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBInfo.dump.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBInfo.dump.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBInfo.dump.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBInfo.dump.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBInfo.dump', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump(...)' code ##################

        str_131549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 8), 'str', 'Gives the header corresponding to this instance as a string.')
        
        # Assigning a List to a Name (line 285):
        
        # Assigning a List to a Name (line 285):
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_131550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        
        # Call to ljust(...): (line 285)
        # Processing the call arguments (line 285)
        int_131554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 35), 'int')
        # Processing the call keyword arguments (line 285)
        kwargs_131555 = {}
        # Getting the type of 'self' (line 285)
        self_131551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 18), 'self', False)
        # Obtaining the member 'title' of a type (line 285)
        title_131552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 18), self_131551, 'title')
        # Obtaining the member 'ljust' of a type (line 285)
        ljust_131553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 18), title_131552, 'ljust')
        # Calling ljust(args, kwargs) (line 285)
        ljust_call_result_131556 = invoke(stypy.reporting.localization.Localization(__file__, 285, 18), ljust_131553, *[int_131554], **kwargs_131555)
        
        
        # Call to ljust(...): (line 285)
        # Processing the call arguments (line 285)
        int_131560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 56), 'int')
        # Processing the call keyword arguments (line 285)
        kwargs_131561 = {}
        # Getting the type of 'self' (line 285)
        self_131557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 41), 'self', False)
        # Obtaining the member 'key' of a type (line 285)
        key_131558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 41), self_131557, 'key')
        # Obtaining the member 'ljust' of a type (line 285)
        ljust_131559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 41), key_131558, 'ljust')
        # Calling ljust(args, kwargs) (line 285)
        ljust_call_result_131562 = invoke(stypy.reporting.localization.Localization(__file__, 285, 41), ljust_131559, *[int_131560], **kwargs_131561)
        
        # Applying the binary operator '+' (line 285)
        result_add_131563 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 18), '+', ljust_call_result_131556, ljust_call_result_131562)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 17), list_131550, result_add_131563)
        
        # Assigning a type to the variable 'header' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'header', list_131550)
        
        # Call to append(...): (line 287)
        # Processing the call arguments (line 287)
        str_131566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 22), 'str', '%14d%14d%14d%14d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 288)
        tuple_131567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 288)
        # Adding element type (line 288)
        # Getting the type of 'self' (line 288)
        self_131568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'self', False)
        # Obtaining the member 'total_nlines' of a type (line 288)
        total_nlines_131569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 23), self_131568, 'total_nlines')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), tuple_131567, total_nlines_131569)
        # Adding element type (line 288)
        # Getting the type of 'self' (line 288)
        self_131570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 42), 'self', False)
        # Obtaining the member 'pointer_nlines' of a type (line 288)
        pointer_nlines_131571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 42), self_131570, 'pointer_nlines')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), tuple_131567, pointer_nlines_131571)
        # Adding element type (line 288)
        # Getting the type of 'self' (line 289)
        self_131572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'self', False)
        # Obtaining the member 'indices_nlines' of a type (line 289)
        indices_nlines_131573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 23), self_131572, 'indices_nlines')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), tuple_131567, indices_nlines_131573)
        # Adding element type (line 288)
        # Getting the type of 'self' (line 289)
        self_131574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 44), 'self', False)
        # Obtaining the member 'values_nlines' of a type (line 289)
        values_nlines_131575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 44), self_131574, 'values_nlines')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 23), tuple_131567, values_nlines_131575)
        
        # Applying the binary operator '%' (line 287)
        result_mod_131576 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 22), '%', str_131566, tuple_131567)
        
        # Processing the call keyword arguments (line 287)
        kwargs_131577 = {}
        # Getting the type of 'header' (line 287)
        header_131564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'header', False)
        # Obtaining the member 'append' of a type (line 287)
        append_131565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), header_131564, 'append')
        # Calling append(args, kwargs) (line 287)
        append_call_result_131578 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), append_131565, *[result_mod_131576], **kwargs_131577)
        
        
        # Call to append(...): (line 290)
        # Processing the call arguments (line 290)
        str_131581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 22), 'str', '%14s%14d%14d%14d%14d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_131582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        
        # Call to ljust(...): (line 291)
        # Processing the call arguments (line 291)
        int_131587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 56), 'int')
        # Processing the call keyword arguments (line 291)
        kwargs_131588 = {}
        # Getting the type of 'self' (line 291)
        self_131583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'self', False)
        # Obtaining the member 'mxtype' of a type (line 291)
        mxtype_131584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 23), self_131583, 'mxtype')
        # Obtaining the member 'fortran_format' of a type (line 291)
        fortran_format_131585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 23), mxtype_131584, 'fortran_format')
        # Obtaining the member 'ljust' of a type (line 291)
        ljust_131586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 23), fortran_format_131585, 'ljust')
        # Calling ljust(args, kwargs) (line 291)
        ljust_call_result_131589 = invoke(stypy.reporting.localization.Localization(__file__, 291, 23), ljust_131586, *[int_131587], **kwargs_131588)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 23), tuple_131582, ljust_call_result_131589)
        # Adding element type (line 291)
        # Getting the type of 'self' (line 291)
        self_131590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 61), 'self', False)
        # Obtaining the member 'nrows' of a type (line 291)
        nrows_131591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 61), self_131590, 'nrows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 23), tuple_131582, nrows_131591)
        # Adding element type (line 291)
        # Getting the type of 'self' (line 292)
        self_131592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'self', False)
        # Obtaining the member 'ncols' of a type (line 292)
        ncols_131593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 23), self_131592, 'ncols')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 23), tuple_131582, ncols_131593)
        # Adding element type (line 291)
        # Getting the type of 'self' (line 292)
        self_131594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 35), 'self', False)
        # Obtaining the member 'nnon_zeros' of a type (line 292)
        nnon_zeros_131595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 35), self_131594, 'nnon_zeros')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 23), tuple_131582, nnon_zeros_131595)
        # Adding element type (line 291)
        int_131596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 23), tuple_131582, int_131596)
        
        # Applying the binary operator '%' (line 290)
        result_mod_131597 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 22), '%', str_131581, tuple_131582)
        
        # Processing the call keyword arguments (line 290)
        kwargs_131598 = {}
        # Getting the type of 'header' (line 290)
        header_131579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'header', False)
        # Obtaining the member 'append' of a type (line 290)
        append_131580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), header_131579, 'append')
        # Calling append(args, kwargs) (line 290)
        append_call_result_131599 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), append_131580, *[result_mod_131597], **kwargs_131598)
        
        
        # Assigning a Attribute to a Name (line 294):
        
        # Assigning a Attribute to a Name (line 294):
        # Getting the type of 'self' (line 294)
        self_131600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'self')
        # Obtaining the member 'pointer_format' of a type (line 294)
        pointer_format_131601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 16), self_131600, 'pointer_format')
        # Obtaining the member 'fortran_format' of a type (line 294)
        fortran_format_131602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 16), pointer_format_131601, 'fortran_format')
        # Assigning a type to the variable 'pffmt' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'pffmt', fortran_format_131602)
        
        # Assigning a Attribute to a Name (line 295):
        
        # Assigning a Attribute to a Name (line 295):
        # Getting the type of 'self' (line 295)
        self_131603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'self')
        # Obtaining the member 'indices_format' of a type (line 295)
        indices_format_131604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), self_131603, 'indices_format')
        # Obtaining the member 'fortran_format' of a type (line 295)
        fortran_format_131605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), indices_format_131604, 'fortran_format')
        # Assigning a type to the variable 'iffmt' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'iffmt', fortran_format_131605)
        
        # Assigning a Attribute to a Name (line 296):
        
        # Assigning a Attribute to a Name (line 296):
        # Getting the type of 'self' (line 296)
        self_131606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'self')
        # Obtaining the member 'values_format' of a type (line 296)
        values_format_131607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), self_131606, 'values_format')
        # Obtaining the member 'fortran_format' of a type (line 296)
        fortran_format_131608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), values_format_131607, 'fortran_format')
        # Assigning a type to the variable 'vffmt' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'vffmt', fortran_format_131608)
        
        # Call to append(...): (line 297)
        # Processing the call arguments (line 297)
        str_131611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 22), 'str', '%16s%16s%20s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_131612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        
        # Call to ljust(...): (line 298)
        # Processing the call arguments (line 298)
        int_131615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 35), 'int')
        # Processing the call keyword arguments (line 298)
        kwargs_131616 = {}
        # Getting the type of 'pffmt' (line 298)
        pffmt_131613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 'pffmt', False)
        # Obtaining the member 'ljust' of a type (line 298)
        ljust_131614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 23), pffmt_131613, 'ljust')
        # Calling ljust(args, kwargs) (line 298)
        ljust_call_result_131617 = invoke(stypy.reporting.localization.Localization(__file__, 298, 23), ljust_131614, *[int_131615], **kwargs_131616)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 23), tuple_131612, ljust_call_result_131617)
        # Adding element type (line 298)
        
        # Call to ljust(...): (line 298)
        # Processing the call arguments (line 298)
        int_131620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 52), 'int')
        # Processing the call keyword arguments (line 298)
        kwargs_131621 = {}
        # Getting the type of 'iffmt' (line 298)
        iffmt_131618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 40), 'iffmt', False)
        # Obtaining the member 'ljust' of a type (line 298)
        ljust_131619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 40), iffmt_131618, 'ljust')
        # Calling ljust(args, kwargs) (line 298)
        ljust_call_result_131622 = invoke(stypy.reporting.localization.Localization(__file__, 298, 40), ljust_131619, *[int_131620], **kwargs_131621)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 23), tuple_131612, ljust_call_result_131622)
        # Adding element type (line 298)
        
        # Call to ljust(...): (line 298)
        # Processing the call arguments (line 298)
        int_131625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 69), 'int')
        # Processing the call keyword arguments (line 298)
        kwargs_131626 = {}
        # Getting the type of 'vffmt' (line 298)
        vffmt_131623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 57), 'vffmt', False)
        # Obtaining the member 'ljust' of a type (line 298)
        ljust_131624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 57), vffmt_131623, 'ljust')
        # Calling ljust(args, kwargs) (line 298)
        ljust_call_result_131627 = invoke(stypy.reporting.localization.Localization(__file__, 298, 57), ljust_131624, *[int_131625], **kwargs_131626)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 23), tuple_131612, ljust_call_result_131627)
        
        # Applying the binary operator '%' (line 297)
        result_mod_131628 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 22), '%', str_131611, tuple_131612)
        
        # Processing the call keyword arguments (line 297)
        kwargs_131629 = {}
        # Getting the type of 'header' (line 297)
        header_131609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'header', False)
        # Obtaining the member 'append' of a type (line 297)
        append_131610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), header_131609, 'append')
        # Calling append(args, kwargs) (line 297)
        append_call_result_131630 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), append_131610, *[result_mod_131628], **kwargs_131629)
        
        
        # Call to join(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'header' (line 299)
        header_131633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 25), 'header', False)
        # Processing the call keyword arguments (line 299)
        kwargs_131634 = {}
        str_131631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 299)
        join_131632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 15), str_131631, 'join')
        # Calling join(args, kwargs) (line 299)
        join_call_result_131635 = invoke(stypy.reporting.localization.Localization(__file__, 299, 15), join_131632, *[header_131633], **kwargs_131634)
        
        # Assigning a type to the variable 'stypy_return_type' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'stypy_return_type', join_call_result_131635)
        
        # ################# End of 'dump(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_131636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131636)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump'
        return stypy_return_type_131636


# Assigning a type to the variable 'HBInfo' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'HBInfo', HBInfo)

@norecursion
def _expect_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 302)
    None_131637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'None')
    defaults = [None_131637]
    # Create a new context for function '_expect_int'
    module_type_store = module_type_store.open_function_context('_expect_int', 302, 0, False)
    
    # Passed parameters checking function
    _expect_int.stypy_localization = localization
    _expect_int.stypy_type_of_self = None
    _expect_int.stypy_type_store = module_type_store
    _expect_int.stypy_function_name = '_expect_int'
    _expect_int.stypy_param_names_list = ['value', 'msg']
    _expect_int.stypy_varargs_param_name = None
    _expect_int.stypy_kwargs_param_name = None
    _expect_int.stypy_call_defaults = defaults
    _expect_int.stypy_call_varargs = varargs
    _expect_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_expect_int', ['value', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_expect_int', localization, ['value', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_expect_int(...)' code ##################

    
    
    # SSA begins for try-except statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to int(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'value' (line 304)
    value_131639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'value', False)
    # Processing the call keyword arguments (line 304)
    kwargs_131640 = {}
    # Getting the type of 'int' (line 304)
    int_131638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'int', False)
    # Calling int(args, kwargs) (line 304)
    int_call_result_131641 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), int_131638, *[value_131639], **kwargs_131640)
    
    # Assigning a type to the variable 'stypy_return_type' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', int_call_result_131641)
    # SSA branch for the except part of a try statement (line 303)
    # SSA branch for the except 'ValueError' branch of a try statement (line 303)
    module_type_store.open_ssa_branch('except')
    
    # Type idiom detected: calculating its left and rigth part (line 306)
    # Getting the type of 'msg' (line 306)
    msg_131642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'msg')
    # Getting the type of 'None' (line 306)
    None_131643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 18), 'None')
    
    (may_be_131644, more_types_in_union_131645) = may_be_none(msg_131642, None_131643)

    if may_be_131644:

        if more_types_in_union_131645:
            # Runtime conditional SSA (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 307):
        
        # Assigning a Str to a Name (line 307):
        str_131646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 18), 'str', 'Expected an int, got %s')
        # Assigning a type to the variable 'msg' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'msg', str_131646)

        if more_types_in_union_131645:
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to ValueError(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'msg' (line 308)
    msg_131648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'msg', False)
    # Getting the type of 'value' (line 308)
    value_131649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'value', False)
    # Applying the binary operator '%' (line 308)
    result_mod_131650 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 25), '%', msg_131648, value_131649)
    
    # Processing the call keyword arguments (line 308)
    kwargs_131651 = {}
    # Getting the type of 'ValueError' (line 308)
    ValueError_131647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 308)
    ValueError_call_result_131652 = invoke(stypy.reporting.localization.Localization(__file__, 308, 14), ValueError_131647, *[result_mod_131650], **kwargs_131651)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 308, 8), ValueError_call_result_131652, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_expect_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_expect_int' in the type store
    # Getting the type of 'stypy_return_type' (line 302)
    stypy_return_type_131653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131653)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_expect_int'
    return stypy_return_type_131653

# Assigning a type to the variable '_expect_int' (line 302)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), '_expect_int', _expect_int)

@norecursion
def _read_hb_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_hb_data'
    module_type_store = module_type_store.open_function_context('_read_hb_data', 311, 0, False)
    
    # Passed parameters checking function
    _read_hb_data.stypy_localization = localization
    _read_hb_data.stypy_type_of_self = None
    _read_hb_data.stypy_type_store = module_type_store
    _read_hb_data.stypy_function_name = '_read_hb_data'
    _read_hb_data.stypy_param_names_list = ['content', 'header']
    _read_hb_data.stypy_varargs_param_name = None
    _read_hb_data.stypy_kwargs_param_name = None
    _read_hb_data.stypy_call_defaults = defaults
    _read_hb_data.stypy_call_varargs = varargs
    _read_hb_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_hb_data', ['content', 'header'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_hb_data', localization, ['content', 'header'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_hb_data(...)' code ##################

    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to join(...): (line 313)
    # Processing the call arguments (line 313)
    
    # Obtaining an instance of the builtin type 'list' (line 313)
    list_131656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 313)
    # Adding element type (line 313)
    
    # Call to read(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'header' (line 313)
    header_131659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 39), 'header', False)
    # Obtaining the member 'pointer_nbytes_full' of a type (line 313)
    pointer_nbytes_full_131660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 39), header_131659, 'pointer_nbytes_full')
    # Processing the call keyword arguments (line 313)
    kwargs_131661 = {}
    # Getting the type of 'content' (line 313)
    content_131657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'content', False)
    # Obtaining the member 'read' of a type (line 313)
    read_131658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 26), content_131657, 'read')
    # Calling read(args, kwargs) (line 313)
    read_call_result_131662 = invoke(stypy.reporting.localization.Localization(__file__, 313, 26), read_131658, *[pointer_nbytes_full_131660], **kwargs_131661)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 25), list_131656, read_call_result_131662)
    # Adding element type (line 313)
    
    # Call to readline(...): (line 314)
    # Processing the call keyword arguments (line 314)
    kwargs_131665 = {}
    # Getting the type of 'content' (line 314)
    content_131663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 27), 'content', False)
    # Obtaining the member 'readline' of a type (line 314)
    readline_131664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 27), content_131663, 'readline')
    # Calling readline(args, kwargs) (line 314)
    readline_call_result_131666 = invoke(stypy.reporting.localization.Localization(__file__, 314, 27), readline_131664, *[], **kwargs_131665)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 25), list_131656, readline_call_result_131666)
    
    # Processing the call keyword arguments (line 313)
    kwargs_131667 = {}
    str_131654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 17), 'str', '')
    # Obtaining the member 'join' of a type (line 313)
    join_131655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 17), str_131654, 'join')
    # Calling join(args, kwargs) (line 313)
    join_call_result_131668 = invoke(stypy.reporting.localization.Localization(__file__, 313, 17), join_131655, *[list_131656], **kwargs_131667)
    
    # Assigning a type to the variable 'ptr_string' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'ptr_string', join_call_result_131668)
    
    # Assigning a Call to a Name (line 315):
    
    # Assigning a Call to a Name (line 315):
    
    # Call to fromstring(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'ptr_string' (line 315)
    ptr_string_131671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 'ptr_string', False)
    # Processing the call keyword arguments (line 315)
    # Getting the type of 'int' (line 316)
    int_131672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'int', False)
    keyword_131673 = int_131672
    str_131674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 27), 'str', ' ')
    keyword_131675 = str_131674
    kwargs_131676 = {'dtype': keyword_131673, 'sep': keyword_131675}
    # Getting the type of 'np' (line 315)
    np_131669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 10), 'np', False)
    # Obtaining the member 'fromstring' of a type (line 315)
    fromstring_131670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 10), np_131669, 'fromstring')
    # Calling fromstring(args, kwargs) (line 315)
    fromstring_call_result_131677 = invoke(stypy.reporting.localization.Localization(__file__, 315, 10), fromstring_131670, *[ptr_string_131671], **kwargs_131676)
    
    # Assigning a type to the variable 'ptr' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'ptr', fromstring_call_result_131677)
    
    # Assigning a Call to a Name (line 318):
    
    # Assigning a Call to a Name (line 318):
    
    # Call to join(...): (line 318)
    # Processing the call arguments (line 318)
    
    # Obtaining an instance of the builtin type 'list' (line 318)
    list_131680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 318)
    # Adding element type (line 318)
    
    # Call to read(...): (line 318)
    # Processing the call arguments (line 318)
    # Getting the type of 'header' (line 318)
    header_131683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 39), 'header', False)
    # Obtaining the member 'indices_nbytes_full' of a type (line 318)
    indices_nbytes_full_131684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 39), header_131683, 'indices_nbytes_full')
    # Processing the call keyword arguments (line 318)
    kwargs_131685 = {}
    # Getting the type of 'content' (line 318)
    content_131681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'content', False)
    # Obtaining the member 'read' of a type (line 318)
    read_131682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 26), content_131681, 'read')
    # Calling read(args, kwargs) (line 318)
    read_call_result_131686 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), read_131682, *[indices_nbytes_full_131684], **kwargs_131685)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 25), list_131680, read_call_result_131686)
    # Adding element type (line 318)
    
    # Call to readline(...): (line 319)
    # Processing the call keyword arguments (line 319)
    kwargs_131689 = {}
    # Getting the type of 'content' (line 319)
    content_131687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 23), 'content', False)
    # Obtaining the member 'readline' of a type (line 319)
    readline_131688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 23), content_131687, 'readline')
    # Calling readline(args, kwargs) (line 319)
    readline_call_result_131690 = invoke(stypy.reporting.localization.Localization(__file__, 319, 23), readline_131688, *[], **kwargs_131689)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 25), list_131680, readline_call_result_131690)
    
    # Processing the call keyword arguments (line 318)
    kwargs_131691 = {}
    str_131678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 17), 'str', '')
    # Obtaining the member 'join' of a type (line 318)
    join_131679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 17), str_131678, 'join')
    # Calling join(args, kwargs) (line 318)
    join_call_result_131692 = invoke(stypy.reporting.localization.Localization(__file__, 318, 17), join_131679, *[list_131680], **kwargs_131691)
    
    # Assigning a type to the variable 'ind_string' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'ind_string', join_call_result_131692)
    
    # Assigning a Call to a Name (line 320):
    
    # Assigning a Call to a Name (line 320):
    
    # Call to fromstring(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'ind_string' (line 320)
    ind_string_131695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'ind_string', False)
    # Processing the call keyword arguments (line 320)
    # Getting the type of 'int' (line 321)
    int_131696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'int', False)
    keyword_131697 = int_131696
    str_131698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 27), 'str', ' ')
    keyword_131699 = str_131698
    kwargs_131700 = {'dtype': keyword_131697, 'sep': keyword_131699}
    # Getting the type of 'np' (line 320)
    np_131693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 10), 'np', False)
    # Obtaining the member 'fromstring' of a type (line 320)
    fromstring_131694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 10), np_131693, 'fromstring')
    # Calling fromstring(args, kwargs) (line 320)
    fromstring_call_result_131701 = invoke(stypy.reporting.localization.Localization(__file__, 320, 10), fromstring_131694, *[ind_string_131695], **kwargs_131700)
    
    # Assigning a type to the variable 'ind' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'ind', fromstring_call_result_131701)
    
    # Assigning a Call to a Name (line 323):
    
    # Assigning a Call to a Name (line 323):
    
    # Call to join(...): (line 323)
    # Processing the call arguments (line 323)
    
    # Obtaining an instance of the builtin type 'list' (line 323)
    list_131704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 323)
    # Adding element type (line 323)
    
    # Call to read(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'header' (line 323)
    header_131707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 39), 'header', False)
    # Obtaining the member 'values_nbytes_full' of a type (line 323)
    values_nbytes_full_131708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 39), header_131707, 'values_nbytes_full')
    # Processing the call keyword arguments (line 323)
    kwargs_131709 = {}
    # Getting the type of 'content' (line 323)
    content_131705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'content', False)
    # Obtaining the member 'read' of a type (line 323)
    read_131706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 26), content_131705, 'read')
    # Calling read(args, kwargs) (line 323)
    read_call_result_131710 = invoke(stypy.reporting.localization.Localization(__file__, 323, 26), read_131706, *[values_nbytes_full_131708], **kwargs_131709)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 25), list_131704, read_call_result_131710)
    # Adding element type (line 323)
    
    # Call to readline(...): (line 324)
    # Processing the call keyword arguments (line 324)
    kwargs_131713 = {}
    # Getting the type of 'content' (line 324)
    content_131711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'content', False)
    # Obtaining the member 'readline' of a type (line 324)
    readline_131712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 26), content_131711, 'readline')
    # Calling readline(args, kwargs) (line 324)
    readline_call_result_131714 = invoke(stypy.reporting.localization.Localization(__file__, 324, 26), readline_131712, *[], **kwargs_131713)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 25), list_131704, readline_call_result_131714)
    
    # Processing the call keyword arguments (line 323)
    kwargs_131715 = {}
    str_131702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 17), 'str', '')
    # Obtaining the member 'join' of a type (line 323)
    join_131703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 17), str_131702, 'join')
    # Calling join(args, kwargs) (line 323)
    join_call_result_131716 = invoke(stypy.reporting.localization.Localization(__file__, 323, 17), join_131703, *[list_131704], **kwargs_131715)
    
    # Assigning a type to the variable 'val_string' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'val_string', join_call_result_131716)
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to fromstring(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'val_string' (line 325)
    val_string_131719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 24), 'val_string', False)
    # Processing the call keyword arguments (line 325)
    # Getting the type of 'header' (line 326)
    header_131720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 18), 'header', False)
    # Obtaining the member 'values_dtype' of a type (line 326)
    values_dtype_131721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 18), header_131720, 'values_dtype')
    keyword_131722 = values_dtype_131721
    str_131723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 43), 'str', ' ')
    keyword_131724 = str_131723
    kwargs_131725 = {'dtype': keyword_131722, 'sep': keyword_131724}
    # Getting the type of 'np' (line 325)
    np_131717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 10), 'np', False)
    # Obtaining the member 'fromstring' of a type (line 325)
    fromstring_131718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 10), np_131717, 'fromstring')
    # Calling fromstring(args, kwargs) (line 325)
    fromstring_call_result_131726 = invoke(stypy.reporting.localization.Localization(__file__, 325, 10), fromstring_131718, *[val_string_131719], **kwargs_131725)
    
    # Assigning a type to the variable 'val' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'val', fromstring_call_result_131726)
    
    
    # SSA begins for try-except statement (line 328)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to csc_matrix(...): (line 329)
    # Processing the call arguments (line 329)
    
    # Obtaining an instance of the builtin type 'tuple' (line 329)
    tuple_131728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 329)
    # Adding element type (line 329)
    # Getting the type of 'val' (line 329)
    val_131729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'val', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 27), tuple_131728, val_131729)
    # Adding element type (line 329)
    # Getting the type of 'ind' (line 329)
    ind_131730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 32), 'ind', False)
    int_131731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 36), 'int')
    # Applying the binary operator '-' (line 329)
    result_sub_131732 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 32), '-', ind_131730, int_131731)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 27), tuple_131728, result_sub_131732)
    # Adding element type (line 329)
    # Getting the type of 'ptr' (line 329)
    ptr_131733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 39), 'ptr', False)
    int_131734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 43), 'int')
    # Applying the binary operator '-' (line 329)
    result_sub_131735 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 39), '-', ptr_131733, int_131734)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 27), tuple_131728, result_sub_131735)
    
    # Processing the call keyword arguments (line 329)
    
    # Obtaining an instance of the builtin type 'tuple' (line 330)
    tuple_131736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 330)
    # Adding element type (line 330)
    # Getting the type of 'header' (line 330)
    header_131737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 33), 'header', False)
    # Obtaining the member 'nrows' of a type (line 330)
    nrows_131738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 33), header_131737, 'nrows')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 33), tuple_131736, nrows_131738)
    # Adding element type (line 330)
    # Getting the type of 'header' (line 330)
    header_131739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 47), 'header', False)
    # Obtaining the member 'ncols' of a type (line 330)
    ncols_131740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 47), header_131739, 'ncols')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 33), tuple_131736, ncols_131740)
    
    keyword_131741 = tuple_131736
    kwargs_131742 = {'shape': keyword_131741}
    # Getting the type of 'csc_matrix' (line 329)
    csc_matrix_131727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 329)
    csc_matrix_call_result_131743 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), csc_matrix_131727, *[tuple_131728], **kwargs_131742)
    
    # Assigning a type to the variable 'stypy_return_type' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'stypy_return_type', csc_matrix_call_result_131743)
    # SSA branch for the except part of a try statement (line 328)
    # SSA branch for the except 'ValueError' branch of a try statement (line 328)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'ValueError' (line 331)
    ValueError_131744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'ValueError')
    # Assigning a type to the variable 'e' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'e', ValueError_131744)
    # Getting the type of 'e' (line 332)
    e_131745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'e')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 332, 8), e_131745, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 328)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_read_hb_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_hb_data' in the type store
    # Getting the type of 'stypy_return_type' (line 311)
    stypy_return_type_131746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131746)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_hb_data'
    return stypy_return_type_131746

# Assigning a type to the variable '_read_hb_data' (line 311)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), '_read_hb_data', _read_hb_data)

@norecursion
def _write_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_write_data'
    module_type_store = module_type_store.open_function_context('_write_data', 335, 0, False)
    
    # Passed parameters checking function
    _write_data.stypy_localization = localization
    _write_data.stypy_type_of_self = None
    _write_data.stypy_type_store = module_type_store
    _write_data.stypy_function_name = '_write_data'
    _write_data.stypy_param_names_list = ['m', 'fid', 'header']
    _write_data.stypy_varargs_param_name = None
    _write_data.stypy_kwargs_param_name = None
    _write_data.stypy_call_defaults = defaults
    _write_data.stypy_call_varargs = varargs
    _write_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_write_data', ['m', 'fid', 'header'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_write_data', localization, ['m', 'fid', 'header'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_write_data(...)' code ##################


    @norecursion
    def write_array(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_array'
        module_type_store = module_type_store.open_function_context('write_array', 336, 4, False)
        
        # Passed parameters checking function
        write_array.stypy_localization = localization
        write_array.stypy_type_of_self = None
        write_array.stypy_type_store = module_type_store
        write_array.stypy_function_name = 'write_array'
        write_array.stypy_param_names_list = ['f', 'ar', 'nlines', 'fmt']
        write_array.stypy_varargs_param_name = None
        write_array.stypy_kwargs_param_name = None
        write_array.stypy_call_defaults = defaults
        write_array.stypy_call_varargs = varargs
        write_array.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'write_array', ['f', 'ar', 'nlines', 'fmt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_array', localization, ['f', 'ar', 'nlines', 'fmt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_array(...)' code ##################

        
        # Assigning a Attribute to a Name (line 339):
        
        # Assigning a Attribute to a Name (line 339):
        # Getting the type of 'fmt' (line 339)
        fmt_131747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'fmt')
        # Obtaining the member 'python_format' of a type (line 339)
        python_format_131748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 16), fmt_131747, 'python_format')
        # Assigning a type to the variable 'pyfmt' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'pyfmt', python_format_131748)
        
        # Assigning a BinOp to a Name (line 340):
        
        # Assigning a BinOp to a Name (line 340):
        # Getting the type of 'pyfmt' (line 340)
        pyfmt_131749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 21), 'pyfmt')
        # Getting the type of 'fmt' (line 340)
        fmt_131750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 29), 'fmt')
        # Obtaining the member 'repeat' of a type (line 340)
        repeat_131751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 29), fmt_131750, 'repeat')
        # Applying the binary operator '*' (line 340)
        result_mul_131752 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 21), '*', pyfmt_131749, repeat_131751)
        
        # Assigning a type to the variable 'pyfmt_full' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'pyfmt_full', result_mul_131752)
        
        # Assigning a Subscript to a Name (line 344):
        
        # Assigning a Subscript to a Name (line 344):
        
        # Obtaining the type of the subscript
        # Getting the type of 'nlines' (line 344)
        nlines_131753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'nlines')
        int_131754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 29), 'int')
        # Applying the binary operator '-' (line 344)
        result_sub_131755 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 20), '-', nlines_131753, int_131754)
        
        # Getting the type of 'fmt' (line 344)
        fmt_131756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 34), 'fmt')
        # Obtaining the member 'repeat' of a type (line 344)
        repeat_131757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 34), fmt_131756, 'repeat')
        # Applying the binary operator '*' (line 344)
        result_mul_131758 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 19), '*', result_sub_131755, repeat_131757)
        
        slice_131759 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 15), None, result_mul_131758, None)
        # Getting the type of 'ar' (line 344)
        ar_131760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'ar')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___131761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 15), ar_131760, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_131762 = invoke(stypy.reporting.localization.Localization(__file__, 344, 15), getitem___131761, slice_131759)
        
        # Assigning a type to the variable 'full' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'full', subscript_call_result_131762)
        
        
        # Call to reshape(...): (line 345)
        # Processing the call arguments (line 345)
        
        # Obtaining an instance of the builtin type 'tuple' (line 345)
        tuple_131765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 345)
        # Adding element type (line 345)
        # Getting the type of 'nlines' (line 345)
        nlines_131766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 33), 'nlines', False)
        int_131767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 40), 'int')
        # Applying the binary operator '-' (line 345)
        result_sub_131768 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 33), '-', nlines_131766, int_131767)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 33), tuple_131765, result_sub_131768)
        # Adding element type (line 345)
        # Getting the type of 'fmt' (line 345)
        fmt_131769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 43), 'fmt', False)
        # Obtaining the member 'repeat' of a type (line 345)
        repeat_131770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 43), fmt_131769, 'repeat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 33), tuple_131765, repeat_131770)
        
        # Processing the call keyword arguments (line 345)
        kwargs_131771 = {}
        # Getting the type of 'full' (line 345)
        full_131763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'full', False)
        # Obtaining the member 'reshape' of a type (line 345)
        reshape_131764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 19), full_131763, 'reshape')
        # Calling reshape(args, kwargs) (line 345)
        reshape_call_result_131772 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), reshape_131764, *[tuple_131765], **kwargs_131771)
        
        # Testing the type of a for loop iterable (line 345)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 345, 8), reshape_call_result_131772)
        # Getting the type of the for loop variable (line 345)
        for_loop_var_131773 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 345, 8), reshape_call_result_131772)
        # Assigning a type to the variable 'row' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'row', for_loop_var_131773)
        # SSA begins for a for statement (line 345)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'pyfmt_full' (line 346)
        pyfmt_full_131776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'pyfmt_full', False)
        
        # Call to tuple(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'row' (line 346)
        row_131778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 39), 'row', False)
        # Processing the call keyword arguments (line 346)
        kwargs_131779 = {}
        # Getting the type of 'tuple' (line 346)
        tuple_131777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 33), 'tuple', False)
        # Calling tuple(args, kwargs) (line 346)
        tuple_call_result_131780 = invoke(stypy.reporting.localization.Localization(__file__, 346, 33), tuple_131777, *[row_131778], **kwargs_131779)
        
        # Applying the binary operator '%' (line 346)
        result_mod_131781 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 20), '%', pyfmt_full_131776, tuple_call_result_131780)
        
        str_131782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 46), 'str', '\n')
        # Applying the binary operator '+' (line 346)
        result_add_131783 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 20), '+', result_mod_131781, str_131782)
        
        # Processing the call keyword arguments (line 346)
        kwargs_131784 = {}
        # Getting the type of 'f' (line 346)
        f_131774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 346)
        write_131775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 12), f_131774, 'write')
        # Calling write(args, kwargs) (line 346)
        write_call_result_131785 = invoke(stypy.reporting.localization.Localization(__file__, 346, 12), write_131775, *[result_add_131783], **kwargs_131784)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 347):
        
        # Assigning a BinOp to a Name (line 347):
        # Getting the type of 'ar' (line 347)
        ar_131786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'ar')
        # Obtaining the member 'size' of a type (line 347)
        size_131787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), ar_131786, 'size')
        # Getting the type of 'full' (line 347)
        full_131788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 'full')
        # Obtaining the member 'size' of a type (line 347)
        size_131789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 28), full_131788, 'size')
        # Applying the binary operator '-' (line 347)
        result_sub_131790 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 18), '-', size_131787, size_131789)
        
        # Assigning a type to the variable 'nremain' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'nremain', result_sub_131790)
        
        
        # Getting the type of 'nremain' (line 348)
        nremain_131791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 'nremain')
        int_131792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 21), 'int')
        # Applying the binary operator '>' (line 348)
        result_gt_131793 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 11), '>', nremain_131791, int_131792)
        
        # Testing the type of an if condition (line 348)
        if_condition_131794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 8), result_gt_131793)
        # Assigning a type to the variable 'if_condition_131794' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'if_condition_131794', if_condition_131794)
        # SSA begins for if statement (line 348)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'pyfmt' (line 349)
        pyfmt_131797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'pyfmt', False)
        # Getting the type of 'nremain' (line 349)
        nremain_131798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'nremain', False)
        # Applying the binary operator '*' (line 349)
        result_mul_131799 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 21), '*', pyfmt_131797, nremain_131798)
        
        
        # Call to tuple(...): (line 349)
        # Processing the call arguments (line 349)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ar' (line 349)
        ar_131801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 49), 'ar', False)
        # Obtaining the member 'size' of a type (line 349)
        size_131802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 49), ar_131801, 'size')
        # Getting the type of 'nremain' (line 349)
        nremain_131803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 59), 'nremain', False)
        # Applying the binary operator '-' (line 349)
        result_sub_131804 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 49), '-', size_131802, nremain_131803)
        
        slice_131805 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 349, 46), result_sub_131804, None, None)
        # Getting the type of 'ar' (line 349)
        ar_131806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 46), 'ar', False)
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___131807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 46), ar_131806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_131808 = invoke(stypy.reporting.localization.Localization(__file__, 349, 46), getitem___131807, slice_131805)
        
        # Processing the call keyword arguments (line 349)
        kwargs_131809 = {}
        # Getting the type of 'tuple' (line 349)
        tuple_131800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 40), 'tuple', False)
        # Calling tuple(args, kwargs) (line 349)
        tuple_call_result_131810 = invoke(stypy.reporting.localization.Localization(__file__, 349, 40), tuple_131800, *[subscript_call_result_131808], **kwargs_131809)
        
        # Applying the binary operator '%' (line 349)
        result_mod_131811 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 20), '%', result_mul_131799, tuple_call_result_131810)
        
        str_131812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 72), 'str', '\n')
        # Applying the binary operator '+' (line 349)
        result_add_131813 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 20), '+', result_mod_131811, str_131812)
        
        # Processing the call keyword arguments (line 349)
        kwargs_131814 = {}
        # Getting the type of 'f' (line 349)
        f_131795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 349)
        write_131796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 12), f_131795, 'write')
        # Calling write(args, kwargs) (line 349)
        write_call_result_131815 = invoke(stypy.reporting.localization.Localization(__file__, 349, 12), write_131796, *[result_add_131813], **kwargs_131814)
        
        # SSA join for if statement (line 348)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_array' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_131816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131816)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_array'
        return stypy_return_type_131816

    # Assigning a type to the variable 'write_array' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'write_array', write_array)
    
    # Call to write(...): (line 351)
    # Processing the call arguments (line 351)
    
    # Call to dump(...): (line 351)
    # Processing the call keyword arguments (line 351)
    kwargs_131821 = {}
    # Getting the type of 'header' (line 351)
    header_131819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 14), 'header', False)
    # Obtaining the member 'dump' of a type (line 351)
    dump_131820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 14), header_131819, 'dump')
    # Calling dump(args, kwargs) (line 351)
    dump_call_result_131822 = invoke(stypy.reporting.localization.Localization(__file__, 351, 14), dump_131820, *[], **kwargs_131821)
    
    # Processing the call keyword arguments (line 351)
    kwargs_131823 = {}
    # Getting the type of 'fid' (line 351)
    fid_131817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'fid', False)
    # Obtaining the member 'write' of a type (line 351)
    write_131818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 4), fid_131817, 'write')
    # Calling write(args, kwargs) (line 351)
    write_call_result_131824 = invoke(stypy.reporting.localization.Localization(__file__, 351, 4), write_131818, *[dump_call_result_131822], **kwargs_131823)
    
    
    # Call to write(...): (line 352)
    # Processing the call arguments (line 352)
    str_131827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 14), 'str', '\n')
    # Processing the call keyword arguments (line 352)
    kwargs_131828 = {}
    # Getting the type of 'fid' (line 352)
    fid_131825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'fid', False)
    # Obtaining the member 'write' of a type (line 352)
    write_131826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 4), fid_131825, 'write')
    # Calling write(args, kwargs) (line 352)
    write_call_result_131829 = invoke(stypy.reporting.localization.Localization(__file__, 352, 4), write_131826, *[str_131827], **kwargs_131828)
    
    
    # Call to write_array(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'fid' (line 354)
    fid_131831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'fid', False)
    # Getting the type of 'm' (line 354)
    m_131832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 21), 'm', False)
    # Obtaining the member 'indptr' of a type (line 354)
    indptr_131833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 21), m_131832, 'indptr')
    int_131834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 30), 'int')
    # Applying the binary operator '+' (line 354)
    result_add_131835 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 21), '+', indptr_131833, int_131834)
    
    # Getting the type of 'header' (line 354)
    header_131836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'header', False)
    # Obtaining the member 'pointer_nlines' of a type (line 354)
    pointer_nlines_131837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 33), header_131836, 'pointer_nlines')
    # Getting the type of 'header' (line 355)
    header_131838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'header', False)
    # Obtaining the member 'pointer_format' of a type (line 355)
    pointer_format_131839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), header_131838, 'pointer_format')
    # Processing the call keyword arguments (line 354)
    kwargs_131840 = {}
    # Getting the type of 'write_array' (line 354)
    write_array_131830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'write_array', False)
    # Calling write_array(args, kwargs) (line 354)
    write_array_call_result_131841 = invoke(stypy.reporting.localization.Localization(__file__, 354, 4), write_array_131830, *[fid_131831, result_add_131835, pointer_nlines_131837, pointer_format_131839], **kwargs_131840)
    
    
    # Call to write_array(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'fid' (line 356)
    fid_131843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'fid', False)
    # Getting the type of 'm' (line 356)
    m_131844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'm', False)
    # Obtaining the member 'indices' of a type (line 356)
    indices_131845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 21), m_131844, 'indices')
    int_131846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 31), 'int')
    # Applying the binary operator '+' (line 356)
    result_add_131847 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 21), '+', indices_131845, int_131846)
    
    # Getting the type of 'header' (line 356)
    header_131848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'header', False)
    # Obtaining the member 'indices_nlines' of a type (line 356)
    indices_nlines_131849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 34), header_131848, 'indices_nlines')
    # Getting the type of 'header' (line 357)
    header_131850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 16), 'header', False)
    # Obtaining the member 'indices_format' of a type (line 357)
    indices_format_131851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 16), header_131850, 'indices_format')
    # Processing the call keyword arguments (line 356)
    kwargs_131852 = {}
    # Getting the type of 'write_array' (line 356)
    write_array_131842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'write_array', False)
    # Calling write_array(args, kwargs) (line 356)
    write_array_call_result_131853 = invoke(stypy.reporting.localization.Localization(__file__, 356, 4), write_array_131842, *[fid_131843, result_add_131847, indices_nlines_131849, indices_format_131851], **kwargs_131852)
    
    
    # Call to write_array(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'fid' (line 358)
    fid_131855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'fid', False)
    # Getting the type of 'm' (line 358)
    m_131856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'm', False)
    # Obtaining the member 'data' of a type (line 358)
    data_131857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 21), m_131856, 'data')
    # Getting the type of 'header' (line 358)
    header_131858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'header', False)
    # Obtaining the member 'values_nlines' of a type (line 358)
    values_nlines_131859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 29), header_131858, 'values_nlines')
    # Getting the type of 'header' (line 359)
    header_131860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'header', False)
    # Obtaining the member 'values_format' of a type (line 359)
    values_format_131861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 16), header_131860, 'values_format')
    # Processing the call keyword arguments (line 358)
    kwargs_131862 = {}
    # Getting the type of 'write_array' (line 358)
    write_array_131854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'write_array', False)
    # Calling write_array(args, kwargs) (line 358)
    write_array_call_result_131863 = invoke(stypy.reporting.localization.Localization(__file__, 358, 4), write_array_131854, *[fid_131855, data_131857, values_nlines_131859, values_format_131861], **kwargs_131862)
    
    
    # ################# End of '_write_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_write_data' in the type store
    # Getting the type of 'stypy_return_type' (line 335)
    stypy_return_type_131864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131864)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_write_data'
    return stypy_return_type_131864

# Assigning a type to the variable '_write_data' (line 335)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), '_write_data', _write_data)
# Declaration of the 'HBMatrixType' class

class HBMatrixType(object, ):
    str_131865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 4), 'str', 'Class to hold the matrix type.')
    
    # Assigning a Dict to a Name (line 365):
    
    # Assigning a Dict to a Name (line 371):
    
    # Assigning a Dict to a Name (line 378):
    
    # Assigning a Call to a Name (line 383):
    
    # Assigning a Call to a Name (line 384):
    
    # Assigning a Call to a Name (line 385):

    @norecursion
    def from_fortran(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'from_fortran'
        module_type_store = module_type_store.open_function_context('from_fortran', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_localization', localization)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_function_name', 'HBMatrixType.from_fortran')
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_param_names_list', ['fmt'])
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBMatrixType.from_fortran.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBMatrixType.from_fortran', ['fmt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_fortran', localization, ['fmt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_fortran(...)' code ##################

        
        
        
        
        # Call to len(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'fmt' (line 389)
        fmt_131867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'fmt', False)
        # Processing the call keyword arguments (line 389)
        kwargs_131868 = {}
        # Getting the type of 'len' (line 389)
        len_131866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'len', False)
        # Calling len(args, kwargs) (line 389)
        len_call_result_131869 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), len_131866, *[fmt_131867], **kwargs_131868)
        
        int_131870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 27), 'int')
        # Applying the binary operator '==' (line 389)
        result_eq_131871 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 15), '==', len_call_result_131869, int_131870)
        
        # Applying the 'not' unary operator (line 389)
        result_not__131872 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 11), 'not', result_eq_131871)
        
        # Testing the type of an if condition (line 389)
        if_condition_131873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), result_not__131872)
        # Assigning a type to the variable 'if_condition_131873' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_131873', if_condition_131873)
        # SSA begins for if statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 390)
        # Processing the call arguments (line 390)
        str_131875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 29), 'str', 'Fortran format for matrix type should be 3 characters long')
        # Processing the call keyword arguments (line 390)
        kwargs_131876 = {}
        # Getting the type of 'ValueError' (line 390)
        ValueError_131874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 390)
        ValueError_call_result_131877 = invoke(stypy.reporting.localization.Localization(__file__, 390, 18), ValueError_131874, *[str_131875], **kwargs_131876)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 390, 12), ValueError_call_result_131877, 'raise parameter', BaseException)
        # SSA join for if statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 393):
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_131878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 43), 'int')
        # Getting the type of 'fmt' (line 393)
        fmt_131879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 39), 'fmt')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___131880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 39), fmt_131879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_131881 = invoke(stypy.reporting.localization.Localization(__file__, 393, 39), getitem___131880, int_131878)
        
        # Getting the type of 'cls' (line 393)
        cls_131882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'cls')
        # Obtaining the member '_f2q_type' of a type (line 393)
        _f2q_type_131883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 25), cls_131882, '_f2q_type')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___131884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 25), _f2q_type_131883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_131885 = invoke(stypy.reporting.localization.Localization(__file__, 393, 25), getitem___131884, subscript_call_result_131881)
        
        # Assigning a type to the variable 'value_type' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'value_type', subscript_call_result_131885)
        
        # Assigning a Subscript to a Name (line 394):
        
        # Assigning a Subscript to a Name (line 394):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_131886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 47), 'int')
        # Getting the type of 'fmt' (line 394)
        fmt_131887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'fmt')
        # Obtaining the member '__getitem__' of a type (line 394)
        getitem___131888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 43), fmt_131887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 394)
        subscript_call_result_131889 = invoke(stypy.reporting.localization.Localization(__file__, 394, 43), getitem___131888, int_131886)
        
        # Getting the type of 'cls' (line 394)
        cls_131890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 24), 'cls')
        # Obtaining the member '_f2q_structure' of a type (line 394)
        _f2q_structure_131891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 24), cls_131890, '_f2q_structure')
        # Obtaining the member '__getitem__' of a type (line 394)
        getitem___131892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 24), _f2q_structure_131891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 394)
        subscript_call_result_131893 = invoke(stypy.reporting.localization.Localization(__file__, 394, 24), getitem___131892, subscript_call_result_131889)
        
        # Assigning a type to the variable 'structure' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'structure', subscript_call_result_131893)
        
        # Assigning a Subscript to a Name (line 395):
        
        # Assigning a Subscript to a Name (line 395):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_131894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 43), 'int')
        # Getting the type of 'fmt' (line 395)
        fmt_131895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 39), 'fmt')
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___131896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 39), fmt_131895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_131897 = invoke(stypy.reporting.localization.Localization(__file__, 395, 39), getitem___131896, int_131894)
        
        # Getting the type of 'cls' (line 395)
        cls_131898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 22), 'cls')
        # Obtaining the member '_f2q_storage' of a type (line 395)
        _f2q_storage_131899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 22), cls_131898, '_f2q_storage')
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___131900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 22), _f2q_storage_131899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_131901 = invoke(stypy.reporting.localization.Localization(__file__, 395, 22), getitem___131900, subscript_call_result_131897)
        
        # Assigning a type to the variable 'storage' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'storage', subscript_call_result_131901)
        
        # Call to cls(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'value_type' (line 396)
        value_type_131903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'value_type', False)
        # Getting the type of 'structure' (line 396)
        structure_131904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 35), 'structure', False)
        # Getting the type of 'storage' (line 396)
        storage_131905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 46), 'storage', False)
        # Processing the call keyword arguments (line 396)
        kwargs_131906 = {}
        # Getting the type of 'cls' (line 396)
        cls_131902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'cls', False)
        # Calling cls(args, kwargs) (line 396)
        cls_call_result_131907 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), cls_131902, *[value_type_131903, structure_131904, storage_131905], **kwargs_131906)
        
        # Assigning a type to the variable 'stypy_return_type' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', cls_call_result_131907)
        # SSA branch for the except part of a try statement (line 392)
        # SSA branch for the except 'KeyError' branch of a try statement (line 392)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 398)
        # Processing the call arguments (line 398)
        str_131909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 29), 'str', 'Unrecognized format %s')
        # Getting the type of 'fmt' (line 398)
        fmt_131910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 56), 'fmt', False)
        # Applying the binary operator '%' (line 398)
        result_mod_131911 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 29), '%', str_131909, fmt_131910)
        
        # Processing the call keyword arguments (line 398)
        kwargs_131912 = {}
        # Getting the type of 'ValueError' (line 398)
        ValueError_131908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 398)
        ValueError_call_result_131913 = invoke(stypy.reporting.localization.Localization(__file__, 398, 18), ValueError_131908, *[result_mod_131911], **kwargs_131912)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 398, 12), ValueError_call_result_131913, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 392)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'from_fortran(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_fortran' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_131914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131914)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_fortran'
        return stypy_return_type_131914


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_131915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 54), 'str', 'assembled')
        defaults = [str_131915]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 400, 4, False)
        # Assigning a type to the variable 'self' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBMatrixType.__init__', ['value_type', 'structure', 'storage'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['value_type', 'structure', 'storage'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 401):
        
        # Assigning a Name to a Attribute (line 401):
        # Getting the type of 'value_type' (line 401)
        value_type_131916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 26), 'value_type')
        # Getting the type of 'self' (line 401)
        self_131917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'self')
        # Setting the type of the member 'value_type' of a type (line 401)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), self_131917, 'value_type', value_type_131916)
        
        # Assigning a Name to a Attribute (line 402):
        
        # Assigning a Name to a Attribute (line 402):
        # Getting the type of 'structure' (line 402)
        structure_131918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 25), 'structure')
        # Getting the type of 'self' (line 402)
        self_131919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self')
        # Setting the type of the member 'structure' of a type (line 402)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_131919, 'structure', structure_131918)
        
        # Assigning a Name to a Attribute (line 403):
        
        # Assigning a Name to a Attribute (line 403):
        # Getting the type of 'storage' (line 403)
        storage_131920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 23), 'storage')
        # Getting the type of 'self' (line 403)
        self_131921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self')
        # Setting the type of the member 'storage' of a type (line 403)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_131921, 'storage', storage_131920)
        
        
        # Getting the type of 'value_type' (line 405)
        value_type_131922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 11), 'value_type')
        # Getting the type of 'self' (line 405)
        self_131923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 29), 'self')
        # Obtaining the member '_q2f_type' of a type (line 405)
        _q2f_type_131924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 29), self_131923, '_q2f_type')
        # Applying the binary operator 'notin' (line 405)
        result_contains_131925 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 11), 'notin', value_type_131922, _q2f_type_131924)
        
        # Testing the type of an if condition (line 405)
        if_condition_131926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 8), result_contains_131925)
        # Assigning a type to the variable 'if_condition_131926' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'if_condition_131926', if_condition_131926)
        # SSA begins for if statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 406)
        # Processing the call arguments (line 406)
        str_131928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 29), 'str', 'Unrecognized type %s')
        # Getting the type of 'value_type' (line 406)
        value_type_131929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 54), 'value_type', False)
        # Applying the binary operator '%' (line 406)
        result_mod_131930 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 29), '%', str_131928, value_type_131929)
        
        # Processing the call keyword arguments (line 406)
        kwargs_131931 = {}
        # Getting the type of 'ValueError' (line 406)
        ValueError_131927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 406)
        ValueError_call_result_131932 = invoke(stypy.reporting.localization.Localization(__file__, 406, 18), ValueError_131927, *[result_mod_131930], **kwargs_131931)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 406, 12), ValueError_call_result_131932, 'raise parameter', BaseException)
        # SSA join for if statement (line 405)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'structure' (line 407)
        structure_131933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'structure')
        # Getting the type of 'self' (line 407)
        self_131934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'self')
        # Obtaining the member '_q2f_structure' of a type (line 407)
        _q2f_structure_131935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 28), self_131934, '_q2f_structure')
        # Applying the binary operator 'notin' (line 407)
        result_contains_131936 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 11), 'notin', structure_131933, _q2f_structure_131935)
        
        # Testing the type of an if condition (line 407)
        if_condition_131937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 8), result_contains_131936)
        # Assigning a type to the variable 'if_condition_131937' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'if_condition_131937', if_condition_131937)
        # SSA begins for if statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 408)
        # Processing the call arguments (line 408)
        str_131939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 29), 'str', 'Unrecognized structure %s')
        # Getting the type of 'structure' (line 408)
        structure_131940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 59), 'structure', False)
        # Applying the binary operator '%' (line 408)
        result_mod_131941 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 29), '%', str_131939, structure_131940)
        
        # Processing the call keyword arguments (line 408)
        kwargs_131942 = {}
        # Getting the type of 'ValueError' (line 408)
        ValueError_131938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 408)
        ValueError_call_result_131943 = invoke(stypy.reporting.localization.Localization(__file__, 408, 18), ValueError_131938, *[result_mod_131941], **kwargs_131942)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 408, 12), ValueError_call_result_131943, 'raise parameter', BaseException)
        # SSA join for if statement (line 407)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'storage' (line 409)
        storage_131944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'storage')
        # Getting the type of 'self' (line 409)
        self_131945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 26), 'self')
        # Obtaining the member '_q2f_storage' of a type (line 409)
        _q2f_storage_131946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 26), self_131945, '_q2f_storage')
        # Applying the binary operator 'notin' (line 409)
        result_contains_131947 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 11), 'notin', storage_131944, _q2f_storage_131946)
        
        # Testing the type of an if condition (line 409)
        if_condition_131948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 8), result_contains_131947)
        # Assigning a type to the variable 'if_condition_131948' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'if_condition_131948', if_condition_131948)
        # SSA begins for if statement (line 409)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 410)
        # Processing the call arguments (line 410)
        str_131950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 29), 'str', 'Unrecognized storage %s')
        # Getting the type of 'storage' (line 410)
        storage_131951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 57), 'storage', False)
        # Applying the binary operator '%' (line 410)
        result_mod_131952 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 29), '%', str_131950, storage_131951)
        
        # Processing the call keyword arguments (line 410)
        kwargs_131953 = {}
        # Getting the type of 'ValueError' (line 410)
        ValueError_131949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 410)
        ValueError_call_result_131954 = invoke(stypy.reporting.localization.Localization(__file__, 410, 18), ValueError_131949, *[result_mod_131952], **kwargs_131953)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 410, 12), ValueError_call_result_131954, 'raise parameter', BaseException)
        # SSA join for if statement (line 409)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def fortran_format(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fortran_format'
        module_type_store = module_type_store.open_function_context('fortran_format', 412, 4, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_localization', localization)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_function_name', 'HBMatrixType.fortran_format')
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_param_names_list', [])
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBMatrixType.fortran_format.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBMatrixType.fortran_format', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fortran_format', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fortran_format(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 414)
        self_131955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 30), 'self')
        # Obtaining the member 'value_type' of a type (line 414)
        value_type_131956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 30), self_131955, 'value_type')
        # Getting the type of 'self' (line 414)
        self_131957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'self')
        # Obtaining the member '_q2f_type' of a type (line 414)
        _q2f_type_131958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 15), self_131957, '_q2f_type')
        # Obtaining the member '__getitem__' of a type (line 414)
        getitem___131959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 15), _q2f_type_131958, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 414)
        subscript_call_result_131960 = invoke(stypy.reporting.localization.Localization(__file__, 414, 15), getitem___131959, value_type_131956)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 415)
        self_131961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 35), 'self')
        # Obtaining the member 'structure' of a type (line 415)
        structure_131962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 35), self_131961, 'structure')
        # Getting the type of 'self' (line 415)
        self_131963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'self')
        # Obtaining the member '_q2f_structure' of a type (line 415)
        _q2f_structure_131964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 15), self_131963, '_q2f_structure')
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___131965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 15), _q2f_structure_131964, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_131966 = invoke(stypy.reporting.localization.Localization(__file__, 415, 15), getitem___131965, structure_131962)
        
        # Applying the binary operator '+' (line 414)
        result_add_131967 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 15), '+', subscript_call_result_131960, subscript_call_result_131966)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 416)
        self_131968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 33), 'self')
        # Obtaining the member 'storage' of a type (line 416)
        storage_131969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 33), self_131968, 'storage')
        # Getting the type of 'self' (line 416)
        self_131970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'self')
        # Obtaining the member '_q2f_storage' of a type (line 416)
        _q2f_storage_131971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), self_131970, '_q2f_storage')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___131972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), _q2f_storage_131971, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_131973 = invoke(stypy.reporting.localization.Localization(__file__, 416, 15), getitem___131972, storage_131969)
        
        # Applying the binary operator '+' (line 415)
        result_add_131974 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 51), '+', result_add_131967, subscript_call_result_131973)
        
        # Assigning a type to the variable 'stypy_return_type' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'stypy_return_type', result_add_131974)
        
        # ################# End of 'fortran_format(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fortran_format' in the type store
        # Getting the type of 'stypy_return_type' (line 412)
        stypy_return_type_131975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fortran_format'
        return stypy_return_type_131975


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 418, 4, False)
        # Assigning a type to the variable 'self' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'HBMatrixType.stypy__repr__')
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBMatrixType.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBMatrixType.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_131976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 15), 'str', 'HBMatrixType(%s, %s, %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 420)
        tuple_131977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 420)
        # Adding element type (line 420)
        # Getting the type of 'self' (line 420)
        self_131978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'self')
        # Obtaining the member 'value_type' of a type (line 420)
        value_type_131979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), self_131978, 'value_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 16), tuple_131977, value_type_131979)
        # Adding element type (line 420)
        # Getting the type of 'self' (line 420)
        self_131980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 33), 'self')
        # Obtaining the member 'structure' of a type (line 420)
        structure_131981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 33), self_131980, 'structure')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 16), tuple_131977, structure_131981)
        # Adding element type (line 420)
        # Getting the type of 'self' (line 420)
        self_131982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 49), 'self')
        # Obtaining the member 'storage' of a type (line 420)
        storage_131983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 49), self_131982, 'storage')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 16), tuple_131977, storage_131983)
        
        # Applying the binary operator '%' (line 419)
        result_mod_131984 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 15), '%', str_131976, tuple_131977)
        
        # Assigning a type to the variable 'stypy_return_type' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'stypy_return_type', result_mod_131984)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 418)
        stypy_return_type_131985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131985)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_131985


# Assigning a type to the variable 'HBMatrixType' (line 362)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 0), 'HBMatrixType', HBMatrixType)

# Assigning a Dict to a Name (line 365):

# Obtaining an instance of the builtin type 'dict' (line 365)
dict_131986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 365)
# Adding element type (key, value) (line 365)
str_131987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 8), 'str', 'real')
str_131988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 16), 'str', 'R')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 16), dict_131986, (str_131987, str_131988))
# Adding element type (key, value) (line 365)
str_131989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 8), 'str', 'complex')
str_131990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 19), 'str', 'C')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 16), dict_131986, (str_131989, str_131990))
# Adding element type (key, value) (line 365)
str_131991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'str', 'pattern')
str_131992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 19), 'str', 'P')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 16), dict_131986, (str_131991, str_131992))
# Adding element type (key, value) (line 365)
str_131993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 8), 'str', 'integer')
str_131994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 19), 'str', 'I')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 16), dict_131986, (str_131993, str_131994))

# Getting the type of 'HBMatrixType'
HBMatrixType_131995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType')
# Setting the type of the member '_q2f_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_131995, '_q2f_type', dict_131986)

# Assigning a Dict to a Name (line 371):

# Obtaining an instance of the builtin type 'dict' (line 371)
dict_131996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 371)
# Adding element type (key, value) (line 371)
str_131997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 12), 'str', 'symmetric')
str_131998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 25), 'str', 'S')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 21), dict_131996, (str_131997, str_131998))
# Adding element type (key, value) (line 371)
str_131999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 12), 'str', 'unsymmetric')
str_132000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 27), 'str', 'U')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 21), dict_131996, (str_131999, str_132000))
# Adding element type (key, value) (line 371)
str_132001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 12), 'str', 'hermitian')
str_132002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 25), 'str', 'H')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 21), dict_131996, (str_132001, str_132002))
# Adding element type (key, value) (line 371)
str_132003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 12), 'str', 'skewsymmetric')
str_132004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 29), 'str', 'Z')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 21), dict_131996, (str_132003, str_132004))
# Adding element type (key, value) (line 371)
str_132005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 12), 'str', 'rectangular')
str_132006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 27), 'str', 'R')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 21), dict_131996, (str_132005, str_132006))

# Getting the type of 'HBMatrixType'
HBMatrixType_132007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType')
# Setting the type of the member '_q2f_structure' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132007, '_q2f_structure', dict_131996)

# Assigning a Dict to a Name (line 378):

# Obtaining an instance of the builtin type 'dict' (line 378)
dict_132008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 378)
# Adding element type (key, value) (line 378)
str_132009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 8), 'str', 'assembled')
str_132010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 21), 'str', 'A')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 19), dict_132008, (str_132009, str_132010))
# Adding element type (key, value) (line 378)
str_132011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 8), 'str', 'elemental')
str_132012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 21), 'str', 'E')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 19), dict_132008, (str_132011, str_132012))

# Getting the type of 'HBMatrixType'
HBMatrixType_132013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType')
# Setting the type of the member '_q2f_storage' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132013, '_q2f_storage', dict_132008)

# Assigning a Call to a Name (line 383):

# Call to dict(...): (line 383)
# Processing the call arguments (line 383)
# Calculating list comprehension
# Calculating comprehension expression

# Call to items(...): (line 383)
# Processing the call keyword arguments (line 383)
kwargs_132021 = {}
# Getting the type of 'HBMatrixType'
HBMatrixType_132018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType', False)
# Obtaining the member '_q2f_type' of a type
_q2f_type_132019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132018, '_q2f_type')
# Obtaining the member 'items' of a type (line 383)
items_132020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 41), _q2f_type_132019, 'items')
# Calling items(args, kwargs) (line 383)
items_call_result_132022 = invoke(stypy.reporting.localization.Localization(__file__, 383, 41), items_132020, *[], **kwargs_132021)

comprehension_132023 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 22), items_call_result_132022)
# Assigning a type to the variable 'i' (line 383)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 22), comprehension_132023))
# Assigning a type to the variable 'j' (line 383)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 22), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 22), comprehension_132023))

# Obtaining an instance of the builtin type 'tuple' (line 383)
tuple_132015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 383)
# Adding element type (line 383)
# Getting the type of 'j' (line 383)
j_132016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 23), 'j', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 23), tuple_132015, j_132016)
# Adding element type (line 383)
# Getting the type of 'i' (line 383)
i_132017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 26), 'i', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 23), tuple_132015, i_132017)

list_132024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 22), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 22), list_132024, tuple_132015)
# Processing the call keyword arguments (line 383)
kwargs_132025 = {}
# Getting the type of 'dict' (line 383)
dict_132014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'dict', False)
# Calling dict(args, kwargs) (line 383)
dict_call_result_132026 = invoke(stypy.reporting.localization.Localization(__file__, 383, 16), dict_132014, *[list_132024], **kwargs_132025)

# Getting the type of 'HBMatrixType'
HBMatrixType_132027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType')
# Setting the type of the member '_f2q_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132027, '_f2q_type', dict_call_result_132026)

# Assigning a Call to a Name (line 384):

# Call to dict(...): (line 384)
# Processing the call arguments (line 384)
# Calculating list comprehension
# Calculating comprehension expression

# Call to items(...): (line 384)
# Processing the call keyword arguments (line 384)
kwargs_132035 = {}
# Getting the type of 'HBMatrixType'
HBMatrixType_132032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType', False)
# Obtaining the member '_q2f_structure' of a type
_q2f_structure_132033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132032, '_q2f_structure')
# Obtaining the member 'items' of a type (line 384)
items_132034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 46), _q2f_structure_132033, 'items')
# Calling items(args, kwargs) (line 384)
items_call_result_132036 = invoke(stypy.reporting.localization.Localization(__file__, 384, 46), items_132034, *[], **kwargs_132035)

comprehension_132037 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 27), items_call_result_132036)
# Assigning a type to the variable 'i' (line 384)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 27), comprehension_132037))
# Assigning a type to the variable 'j' (line 384)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 27), comprehension_132037))

# Obtaining an instance of the builtin type 'tuple' (line 384)
tuple_132029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 384)
# Adding element type (line 384)
# Getting the type of 'j' (line 384)
j_132030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'j', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 28), tuple_132029, j_132030)
# Adding element type (line 384)
# Getting the type of 'i' (line 384)
i_132031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 31), 'i', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 28), tuple_132029, i_132031)

list_132038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 27), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 27), list_132038, tuple_132029)
# Processing the call keyword arguments (line 384)
kwargs_132039 = {}
# Getting the type of 'dict' (line 384)
dict_132028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 21), 'dict', False)
# Calling dict(args, kwargs) (line 384)
dict_call_result_132040 = invoke(stypy.reporting.localization.Localization(__file__, 384, 21), dict_132028, *[list_132038], **kwargs_132039)

# Getting the type of 'HBMatrixType'
HBMatrixType_132041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType')
# Setting the type of the member '_f2q_structure' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132041, '_f2q_structure', dict_call_result_132040)

# Assigning a Call to a Name (line 385):

# Call to dict(...): (line 385)
# Processing the call arguments (line 385)
# Calculating list comprehension
# Calculating comprehension expression

# Call to items(...): (line 385)
# Processing the call keyword arguments (line 385)
kwargs_132049 = {}
# Getting the type of 'HBMatrixType'
HBMatrixType_132046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType', False)
# Obtaining the member '_q2f_storage' of a type
_q2f_storage_132047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132046, '_q2f_storage')
# Obtaining the member 'items' of a type (line 385)
items_132048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 44), _q2f_storage_132047, 'items')
# Calling items(args, kwargs) (line 385)
items_call_result_132050 = invoke(stypy.reporting.localization.Localization(__file__, 385, 44), items_132048, *[], **kwargs_132049)

comprehension_132051 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), items_call_result_132050)
# Assigning a type to the variable 'i' (line 385)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 25), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), comprehension_132051))
# Assigning a type to the variable 'j' (line 385)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 25), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), comprehension_132051))

# Obtaining an instance of the builtin type 'tuple' (line 385)
tuple_132043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 385)
# Adding element type (line 385)
# Getting the type of 'j' (line 385)
j_132044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 26), 'j', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 26), tuple_132043, j_132044)
# Adding element type (line 385)
# Getting the type of 'i' (line 385)
i_132045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 29), 'i', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 26), tuple_132043, i_132045)

list_132052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 25), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), list_132052, tuple_132043)
# Processing the call keyword arguments (line 385)
kwargs_132053 = {}
# Getting the type of 'dict' (line 385)
dict_132042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'dict', False)
# Calling dict(args, kwargs) (line 385)
dict_call_result_132054 = invoke(stypy.reporting.localization.Localization(__file__, 385, 19), dict_132042, *[list_132052], **kwargs_132053)

# Getting the type of 'HBMatrixType'
HBMatrixType_132055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HBMatrixType')
# Setting the type of the member '_f2q_storage' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HBMatrixType_132055, '_f2q_storage', dict_call_result_132054)
# Declaration of the 'HBFile' class

class HBFile(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 424)
        None_132056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 37), 'None')
        defaults = [None_132056]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 424, 4, False)
        # Assigning a type to the variable 'self' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.__init__', ['file', 'hb_info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file', 'hb_info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_132057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, (-1)), 'str', 'Create a HBFile instance.\n\n        Parameters\n        ----------\n        file : file-object\n            StringIO work as well\n        hb_info : HBInfo, optional\n            Should be given as an argument for writing, in which case the file\n            should be writable.\n        ')
        
        # Assigning a Name to a Attribute (line 435):
        
        # Assigning a Name to a Attribute (line 435):
        # Getting the type of 'file' (line 435)
        file_132058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'file')
        # Getting the type of 'self' (line 435)
        self_132059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'self')
        # Setting the type of the member '_fid' of a type (line 435)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 8), self_132059, '_fid', file_132058)
        
        # Type idiom detected: calculating its left and rigth part (line 436)
        # Getting the type of 'hb_info' (line 436)
        hb_info_132060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'hb_info')
        # Getting the type of 'None' (line 436)
        None_132061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'None')
        
        (may_be_132062, more_types_in_union_132063) = may_be_none(hb_info_132060, None_132061)

        if may_be_132062:

            if more_types_in_union_132063:
                # Runtime conditional SSA (line 436)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 437):
            
            # Assigning a Call to a Attribute (line 437):
            
            # Call to from_file(...): (line 437)
            # Processing the call arguments (line 437)
            # Getting the type of 'file' (line 437)
            file_132066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 45), 'file', False)
            # Processing the call keyword arguments (line 437)
            kwargs_132067 = {}
            # Getting the type of 'HBInfo' (line 437)
            HBInfo_132064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 28), 'HBInfo', False)
            # Obtaining the member 'from_file' of a type (line 437)
            from_file_132065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 28), HBInfo_132064, 'from_file')
            # Calling from_file(args, kwargs) (line 437)
            from_file_call_result_132068 = invoke(stypy.reporting.localization.Localization(__file__, 437, 28), from_file_132065, *[file_132066], **kwargs_132067)
            
            # Getting the type of 'self' (line 437)
            self_132069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'self')
            # Setting the type of the member '_hb_info' of a type (line 437)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 12), self_132069, '_hb_info', from_file_call_result_132068)

            if more_types_in_union_132063:
                # Runtime conditional SSA for else branch (line 436)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_132062) or more_types_in_union_132063):
            
            # Assigning a Name to a Attribute (line 441):
            
            # Assigning a Name to a Attribute (line 441):
            # Getting the type of 'hb_info' (line 441)
            hb_info_132070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 28), 'hb_info')
            # Getting the type of 'self' (line 441)
            self_132071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'self')
            # Setting the type of the member '_hb_info' of a type (line 441)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), self_132071, '_hb_info', hb_info_132070)

            if (may_be_132062 and more_types_in_union_132063):
                # SSA join for if statement (line 436)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'title'
        module_type_store = module_type_store.open_function_context('title', 443, 4, False)
        # Assigning a type to the variable 'self' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBFile.title.__dict__.__setitem__('stypy_localization', localization)
        HBFile.title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBFile.title.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBFile.title.__dict__.__setitem__('stypy_function_name', 'HBFile.title')
        HBFile.title.__dict__.__setitem__('stypy_param_names_list', [])
        HBFile.title.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBFile.title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBFile.title.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBFile.title.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBFile.title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBFile.title.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.title', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'title', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'title(...)' code ##################

        # Getting the type of 'self' (line 445)
        self_132072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 15), 'self')
        # Obtaining the member '_hb_info' of a type (line 445)
        _hb_info_132073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), self_132072, '_hb_info')
        # Obtaining the member 'title' of a type (line 445)
        title_132074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), _hb_info_132073, 'title')
        # Assigning a type to the variable 'stypy_return_type' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'stypy_return_type', title_132074)
        
        # ################# End of 'title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'title' in the type store
        # Getting the type of 'stypy_return_type' (line 443)
        stypy_return_type_132075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'title'
        return stypy_return_type_132075


    @norecursion
    def key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'key'
        module_type_store = module_type_store.open_function_context('key', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBFile.key.__dict__.__setitem__('stypy_localization', localization)
        HBFile.key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBFile.key.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBFile.key.__dict__.__setitem__('stypy_function_name', 'HBFile.key')
        HBFile.key.__dict__.__setitem__('stypy_param_names_list', [])
        HBFile.key.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBFile.key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBFile.key.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBFile.key.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBFile.key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBFile.key.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.key', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'key', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'key(...)' code ##################

        # Getting the type of 'self' (line 449)
        self_132076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'self')
        # Obtaining the member '_hb_info' of a type (line 449)
        _hb_info_132077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 15), self_132076, '_hb_info')
        # Obtaining the member 'key' of a type (line 449)
        key_132078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 15), _hb_info_132077, 'key')
        # Assigning a type to the variable 'stypy_return_type' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'stypy_return_type', key_132078)
        
        # ################# End of 'key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'key' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_132079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132079)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'key'
        return stypy_return_type_132079


    @norecursion
    def type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'type'
        module_type_store = module_type_store.open_function_context('type', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBFile.type.__dict__.__setitem__('stypy_localization', localization)
        HBFile.type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBFile.type.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBFile.type.__dict__.__setitem__('stypy_function_name', 'HBFile.type')
        HBFile.type.__dict__.__setitem__('stypy_param_names_list', [])
        HBFile.type.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBFile.type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBFile.type.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBFile.type.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBFile.type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBFile.type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'type(...)' code ##################

        # Getting the type of 'self' (line 453)
        self_132080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'self')
        # Obtaining the member '_hb_info' of a type (line 453)
        _hb_info_132081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 15), self_132080, '_hb_info')
        # Obtaining the member 'mxtype' of a type (line 453)
        mxtype_132082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 15), _hb_info_132081, 'mxtype')
        # Obtaining the member 'value_type' of a type (line 453)
        value_type_132083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 15), mxtype_132082, 'value_type')
        # Assigning a type to the variable 'stypy_return_type' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'stypy_return_type', value_type_132083)
        
        # ################# End of 'type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'type' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_132084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'type'
        return stypy_return_type_132084


    @norecursion
    def structure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'structure'
        module_type_store = module_type_store.open_function_context('structure', 455, 4, False)
        # Assigning a type to the variable 'self' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBFile.structure.__dict__.__setitem__('stypy_localization', localization)
        HBFile.structure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBFile.structure.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBFile.structure.__dict__.__setitem__('stypy_function_name', 'HBFile.structure')
        HBFile.structure.__dict__.__setitem__('stypy_param_names_list', [])
        HBFile.structure.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBFile.structure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBFile.structure.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBFile.structure.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBFile.structure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBFile.structure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.structure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'structure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'structure(...)' code ##################

        # Getting the type of 'self' (line 457)
        self_132085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'self')
        # Obtaining the member '_hb_info' of a type (line 457)
        _hb_info_132086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), self_132085, '_hb_info')
        # Obtaining the member 'mxtype' of a type (line 457)
        mxtype_132087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), _hb_info_132086, 'mxtype')
        # Obtaining the member 'structure' of a type (line 457)
        structure_132088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 15), mxtype_132087, 'structure')
        # Assigning a type to the variable 'stypy_return_type' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'stypy_return_type', structure_132088)
        
        # ################# End of 'structure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'structure' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_132089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132089)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'structure'
        return stypy_return_type_132089


    @norecursion
    def storage(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'storage'
        module_type_store = module_type_store.open_function_context('storage', 459, 4, False)
        # Assigning a type to the variable 'self' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBFile.storage.__dict__.__setitem__('stypy_localization', localization)
        HBFile.storage.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBFile.storage.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBFile.storage.__dict__.__setitem__('stypy_function_name', 'HBFile.storage')
        HBFile.storage.__dict__.__setitem__('stypy_param_names_list', [])
        HBFile.storage.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBFile.storage.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBFile.storage.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBFile.storage.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBFile.storage.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBFile.storage.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.storage', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'storage', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'storage(...)' code ##################

        # Getting the type of 'self' (line 461)
        self_132090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 15), 'self')
        # Obtaining the member '_hb_info' of a type (line 461)
        _hb_info_132091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 15), self_132090, '_hb_info')
        # Obtaining the member 'mxtype' of a type (line 461)
        mxtype_132092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 15), _hb_info_132091, 'mxtype')
        # Obtaining the member 'storage' of a type (line 461)
        storage_132093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 15), mxtype_132092, 'storage')
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'stypy_return_type', storage_132093)
        
        # ################# End of 'storage(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'storage' in the type store
        # Getting the type of 'stypy_return_type' (line 459)
        stypy_return_type_132094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'storage'
        return stypy_return_type_132094


    @norecursion
    def read_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_matrix'
        module_type_store = module_type_store.open_function_context('read_matrix', 463, 4, False)
        # Assigning a type to the variable 'self' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBFile.read_matrix.__dict__.__setitem__('stypy_localization', localization)
        HBFile.read_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBFile.read_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBFile.read_matrix.__dict__.__setitem__('stypy_function_name', 'HBFile.read_matrix')
        HBFile.read_matrix.__dict__.__setitem__('stypy_param_names_list', [])
        HBFile.read_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBFile.read_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBFile.read_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBFile.read_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBFile.read_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBFile.read_matrix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.read_matrix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_matrix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_matrix(...)' code ##################

        
        # Call to _read_hb_data(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'self' (line 464)
        self_132096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 29), 'self', False)
        # Obtaining the member '_fid' of a type (line 464)
        _fid_132097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 29), self_132096, '_fid')
        # Getting the type of 'self' (line 464)
        self_132098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 40), 'self', False)
        # Obtaining the member '_hb_info' of a type (line 464)
        _hb_info_132099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 40), self_132098, '_hb_info')
        # Processing the call keyword arguments (line 464)
        kwargs_132100 = {}
        # Getting the type of '_read_hb_data' (line 464)
        _read_hb_data_132095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), '_read_hb_data', False)
        # Calling _read_hb_data(args, kwargs) (line 464)
        _read_hb_data_call_result_132101 = invoke(stypy.reporting.localization.Localization(__file__, 464, 15), _read_hb_data_132095, *[_fid_132097, _hb_info_132099], **kwargs_132100)
        
        # Assigning a type to the variable 'stypy_return_type' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'stypy_return_type', _read_hb_data_call_result_132101)
        
        # ################# End of 'read_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 463)
        stypy_return_type_132102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_matrix'
        return stypy_return_type_132102


    @norecursion
    def write_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_matrix'
        module_type_store = module_type_store.open_function_context('write_matrix', 466, 4, False)
        # Assigning a type to the variable 'self' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HBFile.write_matrix.__dict__.__setitem__('stypy_localization', localization)
        HBFile.write_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HBFile.write_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        HBFile.write_matrix.__dict__.__setitem__('stypy_function_name', 'HBFile.write_matrix')
        HBFile.write_matrix.__dict__.__setitem__('stypy_param_names_list', ['m'])
        HBFile.write_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        HBFile.write_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HBFile.write_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        HBFile.write_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        HBFile.write_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HBFile.write_matrix.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HBFile.write_matrix', ['m'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_matrix', localization, ['m'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_matrix(...)' code ##################

        
        # Call to _write_data(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'm' (line 467)
        m_132104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 27), 'm', False)
        # Getting the type of 'self' (line 467)
        self_132105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 30), 'self', False)
        # Obtaining the member '_fid' of a type (line 467)
        _fid_132106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 30), self_132105, '_fid')
        # Getting the type of 'self' (line 467)
        self_132107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 41), 'self', False)
        # Obtaining the member '_hb_info' of a type (line 467)
        _hb_info_132108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 41), self_132107, '_hb_info')
        # Processing the call keyword arguments (line 467)
        kwargs_132109 = {}
        # Getting the type of '_write_data' (line 467)
        _write_data_132103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), '_write_data', False)
        # Calling _write_data(args, kwargs) (line 467)
        _write_data_call_result_132110 = invoke(stypy.reporting.localization.Localization(__file__, 467, 15), _write_data_132103, *[m_132104, _fid_132106, _hb_info_132108], **kwargs_132109)
        
        # Assigning a type to the variable 'stypy_return_type' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'stypy_return_type', _write_data_call_result_132110)
        
        # ################# End of 'write_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 466)
        stypy_return_type_132111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_matrix'
        return stypy_return_type_132111


# Assigning a type to the variable 'HBFile' (line 423)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'HBFile', HBFile)

@norecursion
def hb_read(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hb_read'
    module_type_store = module_type_store.open_function_context('hb_read', 470, 0, False)
    
    # Passed parameters checking function
    hb_read.stypy_localization = localization
    hb_read.stypy_type_of_self = None
    hb_read.stypy_type_store = module_type_store
    hb_read.stypy_function_name = 'hb_read'
    hb_read.stypy_param_names_list = ['path_or_open_file']
    hb_read.stypy_varargs_param_name = None
    hb_read.stypy_kwargs_param_name = None
    hb_read.stypy_call_defaults = defaults
    hb_read.stypy_call_varargs = varargs
    hb_read.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hb_read', ['path_or_open_file'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hb_read', localization, ['path_or_open_file'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hb_read(...)' code ##################

    str_132112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, (-1)), 'str', 'Read HB-format file.\n\n    Parameters\n    ----------\n    path_or_open_file : path-like or file-like\n        If a file-like object, it is used as-is. Otherwise it is opened\n        before reading.\n\n    Returns\n    -------\n    data : scipy.sparse.csc_matrix instance\n        The data read from the HB file as a sparse matrix.\n\n    Notes\n    -----\n    At the moment not the full Harwell-Boeing format is supported. Supported\n    features are:\n\n        - assembled, non-symmetric, real matrices\n        - integer for pointer/indices\n        - exponential format for float values, and int format\n\n    ')

    @norecursion
    def _get_matrix(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_matrix'
        module_type_store = module_type_store.open_function_context('_get_matrix', 494, 4, False)
        
        # Passed parameters checking function
        _get_matrix.stypy_localization = localization
        _get_matrix.stypy_type_of_self = None
        _get_matrix.stypy_type_store = module_type_store
        _get_matrix.stypy_function_name = '_get_matrix'
        _get_matrix.stypy_param_names_list = ['fid']
        _get_matrix.stypy_varargs_param_name = None
        _get_matrix.stypy_kwargs_param_name = None
        _get_matrix.stypy_call_defaults = defaults
        _get_matrix.stypy_call_varargs = varargs
        _get_matrix.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_get_matrix', ['fid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_matrix', localization, ['fid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_matrix(...)' code ##################

        
        # Assigning a Call to a Name (line 495):
        
        # Assigning a Call to a Name (line 495):
        
        # Call to HBFile(...): (line 495)
        # Processing the call arguments (line 495)
        # Getting the type of 'fid' (line 495)
        fid_132114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 20), 'fid', False)
        # Processing the call keyword arguments (line 495)
        kwargs_132115 = {}
        # Getting the type of 'HBFile' (line 495)
        HBFile_132113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 13), 'HBFile', False)
        # Calling HBFile(args, kwargs) (line 495)
        HBFile_call_result_132116 = invoke(stypy.reporting.localization.Localization(__file__, 495, 13), HBFile_132113, *[fid_132114], **kwargs_132115)
        
        # Assigning a type to the variable 'hb' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'hb', HBFile_call_result_132116)
        
        # Call to read_matrix(...): (line 496)
        # Processing the call keyword arguments (line 496)
        kwargs_132119 = {}
        # Getting the type of 'hb' (line 496)
        hb_132117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'hb', False)
        # Obtaining the member 'read_matrix' of a type (line 496)
        read_matrix_132118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 15), hb_132117, 'read_matrix')
        # Calling read_matrix(args, kwargs) (line 496)
        read_matrix_call_result_132120 = invoke(stypy.reporting.localization.Localization(__file__, 496, 15), read_matrix_132118, *[], **kwargs_132119)
        
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', read_matrix_call_result_132120)
        
        # ################# End of '_get_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 494)
        stypy_return_type_132121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_matrix'
        return stypy_return_type_132121

    # Assigning a type to the variable '_get_matrix' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), '_get_matrix', _get_matrix)
    
    # Type idiom detected: calculating its left and rigth part (line 498)
    str_132122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 34), 'str', 'read')
    # Getting the type of 'path_or_open_file' (line 498)
    path_or_open_file_132123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 15), 'path_or_open_file')
    
    (may_be_132124, more_types_in_union_132125) = may_provide_member(str_132122, path_or_open_file_132123)

    if may_be_132124:

        if more_types_in_union_132125:
            # Runtime conditional SSA (line 498)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'path_or_open_file' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'path_or_open_file', remove_not_member_provider_from_union(path_or_open_file_132123, 'read'))
        
        # Call to _get_matrix(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'path_or_open_file' (line 499)
        path_or_open_file_132127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 27), 'path_or_open_file', False)
        # Processing the call keyword arguments (line 499)
        kwargs_132128 = {}
        # Getting the type of '_get_matrix' (line 499)
        _get_matrix_132126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), '_get_matrix', False)
        # Calling _get_matrix(args, kwargs) (line 499)
        _get_matrix_call_result_132129 = invoke(stypy.reporting.localization.Localization(__file__, 499, 15), _get_matrix_132126, *[path_or_open_file_132127], **kwargs_132128)
        
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'stypy_return_type', _get_matrix_call_result_132129)

        if more_types_in_union_132125:
            # Runtime conditional SSA for else branch (line 498)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_132124) or more_types_in_union_132125):
        # Assigning a type to the variable 'path_or_open_file' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'path_or_open_file', remove_member_provider_from_union(path_or_open_file_132123, 'read'))
        
        # Call to open(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'path_or_open_file' (line 501)
        path_or_open_file_132131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 18), 'path_or_open_file', False)
        # Processing the call keyword arguments (line 501)
        kwargs_132132 = {}
        # Getting the type of 'open' (line 501)
        open_132130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 13), 'open', False)
        # Calling open(args, kwargs) (line 501)
        open_call_result_132133 = invoke(stypy.reporting.localization.Localization(__file__, 501, 13), open_132130, *[path_or_open_file_132131], **kwargs_132132)
        
        with_132134 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 501, 13), open_call_result_132133, 'with parameter', '__enter__', '__exit__')

        if with_132134:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 501)
            enter___132135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 13), open_call_result_132133, '__enter__')
            with_enter_132136 = invoke(stypy.reporting.localization.Localization(__file__, 501, 13), enter___132135)
            # Assigning a type to the variable 'f' (line 501)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 13), 'f', with_enter_132136)
            
            # Call to _get_matrix(...): (line 502)
            # Processing the call arguments (line 502)
            # Getting the type of 'f' (line 502)
            f_132138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 31), 'f', False)
            # Processing the call keyword arguments (line 502)
            kwargs_132139 = {}
            # Getting the type of '_get_matrix' (line 502)
            _get_matrix_132137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), '_get_matrix', False)
            # Calling _get_matrix(args, kwargs) (line 502)
            _get_matrix_call_result_132140 = invoke(stypy.reporting.localization.Localization(__file__, 502, 19), _get_matrix_132137, *[f_132138], **kwargs_132139)
            
            # Assigning a type to the variable 'stypy_return_type' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'stypy_return_type', _get_matrix_call_result_132140)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 501)
            exit___132141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 13), open_call_result_132133, '__exit__')
            with_exit_132142 = invoke(stypy.reporting.localization.Localization(__file__, 501, 13), exit___132141, None, None, None)


        if (may_be_132124 and more_types_in_union_132125):
            # SSA join for if statement (line 498)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'hb_read(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hb_read' in the type store
    # Getting the type of 'stypy_return_type' (line 470)
    stypy_return_type_132143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132143)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hb_read'
    return stypy_return_type_132143

# Assigning a type to the variable 'hb_read' (line 470)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 0), 'hb_read', hb_read)

@norecursion
def hb_write(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 505)
    None_132144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 43), 'None')
    defaults = [None_132144]
    # Create a new context for function 'hb_write'
    module_type_store = module_type_store.open_function_context('hb_write', 505, 0, False)
    
    # Passed parameters checking function
    hb_write.stypy_localization = localization
    hb_write.stypy_type_of_self = None
    hb_write.stypy_type_store = module_type_store
    hb_write.stypy_function_name = 'hb_write'
    hb_write.stypy_param_names_list = ['path_or_open_file', 'm', 'hb_info']
    hb_write.stypy_varargs_param_name = None
    hb_write.stypy_kwargs_param_name = None
    hb_write.stypy_call_defaults = defaults
    hb_write.stypy_call_varargs = varargs
    hb_write.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hb_write', ['path_or_open_file', 'm', 'hb_info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hb_write', localization, ['path_or_open_file', 'm', 'hb_info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hb_write(...)' code ##################

    str_132145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, (-1)), 'str', 'Write HB-format file.\n\n    Parameters\n    ----------\n    path_or_open_file : path-like or file-like\n        If a file-like object, it is used as-is. Otherwise it is opened\n        before writing.\n    m : sparse-matrix\n        the sparse matrix to write\n    hb_info : HBInfo\n        contains the meta-data for write\n\n    Returns\n    -------\n    None\n\n    Notes\n    -----\n    At the moment not the full Harwell-Boeing format is supported. Supported\n    features are:\n\n        - assembled, non-symmetric, real matrices\n        - integer for pointer/indices\n        - exponential format for float values, and int format\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 532)
    # Getting the type of 'hb_info' (line 532)
    hb_info_132146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 7), 'hb_info')
    # Getting the type of 'None' (line 532)
    None_132147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 18), 'None')
    
    (may_be_132148, more_types_in_union_132149) = may_be_none(hb_info_132146, None_132147)

    if may_be_132148:

        if more_types_in_union_132149:
            # Runtime conditional SSA (line 532)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 533):
        
        # Assigning a Call to a Name (line 533):
        
        # Call to from_data(...): (line 533)
        # Processing the call arguments (line 533)
        # Getting the type of 'm' (line 533)
        m_132152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 35), 'm', False)
        # Processing the call keyword arguments (line 533)
        kwargs_132153 = {}
        # Getting the type of 'HBInfo' (line 533)
        HBInfo_132150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 18), 'HBInfo', False)
        # Obtaining the member 'from_data' of a type (line 533)
        from_data_132151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 18), HBInfo_132150, 'from_data')
        # Calling from_data(args, kwargs) (line 533)
        from_data_call_result_132154 = invoke(stypy.reporting.localization.Localization(__file__, 533, 18), from_data_132151, *[m_132152], **kwargs_132153)
        
        # Assigning a type to the variable 'hb_info' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'hb_info', from_data_call_result_132154)

        if more_types_in_union_132149:
            # SSA join for if statement (line 532)
            module_type_store = module_type_store.join_ssa_context()


    

    @norecursion
    def _set_matrix(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_matrix'
        module_type_store = module_type_store.open_function_context('_set_matrix', 535, 4, False)
        
        # Passed parameters checking function
        _set_matrix.stypy_localization = localization
        _set_matrix.stypy_type_of_self = None
        _set_matrix.stypy_type_store = module_type_store
        _set_matrix.stypy_function_name = '_set_matrix'
        _set_matrix.stypy_param_names_list = ['fid']
        _set_matrix.stypy_varargs_param_name = None
        _set_matrix.stypy_kwargs_param_name = None
        _set_matrix.stypy_call_defaults = defaults
        _set_matrix.stypy_call_varargs = varargs
        _set_matrix.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_set_matrix', ['fid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_matrix', localization, ['fid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_matrix(...)' code ##################

        
        # Assigning a Call to a Name (line 536):
        
        # Assigning a Call to a Name (line 536):
        
        # Call to HBFile(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'fid' (line 536)
        fid_132156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 20), 'fid', False)
        # Getting the type of 'hb_info' (line 536)
        hb_info_132157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 25), 'hb_info', False)
        # Processing the call keyword arguments (line 536)
        kwargs_132158 = {}
        # Getting the type of 'HBFile' (line 536)
        HBFile_132155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 13), 'HBFile', False)
        # Calling HBFile(args, kwargs) (line 536)
        HBFile_call_result_132159 = invoke(stypy.reporting.localization.Localization(__file__, 536, 13), HBFile_132155, *[fid_132156, hb_info_132157], **kwargs_132158)
        
        # Assigning a type to the variable 'hb' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'hb', HBFile_call_result_132159)
        
        # Call to write_matrix(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'm' (line 537)
        m_132162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 31), 'm', False)
        # Processing the call keyword arguments (line 537)
        kwargs_132163 = {}
        # Getting the type of 'hb' (line 537)
        hb_132160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 15), 'hb', False)
        # Obtaining the member 'write_matrix' of a type (line 537)
        write_matrix_132161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 15), hb_132160, 'write_matrix')
        # Calling write_matrix(args, kwargs) (line 537)
        write_matrix_call_result_132164 = invoke(stypy.reporting.localization.Localization(__file__, 537, 15), write_matrix_132161, *[m_132162], **kwargs_132163)
        
        # Assigning a type to the variable 'stypy_return_type' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'stypy_return_type', write_matrix_call_result_132164)
        
        # ################# End of '_set_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_132165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_matrix'
        return stypy_return_type_132165

    # Assigning a type to the variable '_set_matrix' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), '_set_matrix', _set_matrix)
    
    # Type idiom detected: calculating its left and rigth part (line 539)
    str_132166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 34), 'str', 'write')
    # Getting the type of 'path_or_open_file' (line 539)
    path_or_open_file_132167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'path_or_open_file')
    
    (may_be_132168, more_types_in_union_132169) = may_provide_member(str_132166, path_or_open_file_132167)

    if may_be_132168:

        if more_types_in_union_132169:
            # Runtime conditional SSA (line 539)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'path_or_open_file' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'path_or_open_file', remove_not_member_provider_from_union(path_or_open_file_132167, 'write'))
        
        # Call to _set_matrix(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'path_or_open_file' (line 540)
        path_or_open_file_132171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'path_or_open_file', False)
        # Processing the call keyword arguments (line 540)
        kwargs_132172 = {}
        # Getting the type of '_set_matrix' (line 540)
        _set_matrix_132170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), '_set_matrix', False)
        # Calling _set_matrix(args, kwargs) (line 540)
        _set_matrix_call_result_132173 = invoke(stypy.reporting.localization.Localization(__file__, 540, 15), _set_matrix_132170, *[path_or_open_file_132171], **kwargs_132172)
        
        # Assigning a type to the variable 'stypy_return_type' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'stypy_return_type', _set_matrix_call_result_132173)

        if more_types_in_union_132169:
            # Runtime conditional SSA for else branch (line 539)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_132168) or more_types_in_union_132169):
        # Assigning a type to the variable 'path_or_open_file' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'path_or_open_file', remove_member_provider_from_union(path_or_open_file_132167, 'write'))
        
        # Call to open(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'path_or_open_file' (line 542)
        path_or_open_file_132175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 18), 'path_or_open_file', False)
        str_132176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 37), 'str', 'w')
        # Processing the call keyword arguments (line 542)
        kwargs_132177 = {}
        # Getting the type of 'open' (line 542)
        open_132174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 13), 'open', False)
        # Calling open(args, kwargs) (line 542)
        open_call_result_132178 = invoke(stypy.reporting.localization.Localization(__file__, 542, 13), open_132174, *[path_or_open_file_132175, str_132176], **kwargs_132177)
        
        with_132179 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 542, 13), open_call_result_132178, 'with parameter', '__enter__', '__exit__')

        if with_132179:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 542)
            enter___132180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 13), open_call_result_132178, '__enter__')
            with_enter_132181 = invoke(stypy.reporting.localization.Localization(__file__, 542, 13), enter___132180)
            # Assigning a type to the variable 'f' (line 542)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 13), 'f', with_enter_132181)
            
            # Call to _set_matrix(...): (line 543)
            # Processing the call arguments (line 543)
            # Getting the type of 'f' (line 543)
            f_132183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 31), 'f', False)
            # Processing the call keyword arguments (line 543)
            kwargs_132184 = {}
            # Getting the type of '_set_matrix' (line 543)
            _set_matrix_132182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 19), '_set_matrix', False)
            # Calling _set_matrix(args, kwargs) (line 543)
            _set_matrix_call_result_132185 = invoke(stypy.reporting.localization.Localization(__file__, 543, 19), _set_matrix_132182, *[f_132183], **kwargs_132184)
            
            # Assigning a type to the variable 'stypy_return_type' (line 543)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'stypy_return_type', _set_matrix_call_result_132185)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 542)
            exit___132186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 13), open_call_result_132178, '__exit__')
            with_exit_132187 = invoke(stypy.reporting.localization.Localization(__file__, 542, 13), exit___132186, None, None, None)


        if (may_be_132168 and more_types_in_union_132169):
            # SSA join for if statement (line 539)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'hb_write(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hb_write' in the type store
    # Getting the type of 'stypy_return_type' (line 505)
    stypy_return_type_132188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_132188)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hb_write'
    return stypy_return_type_132188

# Assigning a type to the variable 'hb_write' (line 505)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 0), 'hb_write', hb_write)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
