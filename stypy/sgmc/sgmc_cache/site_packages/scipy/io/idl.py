
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # IDLSave - a python module to read IDL 'save' files
2: # Copyright (c) 2010 Thomas P. Robitaille
3: 
4: # Many thanks to Craig Markwardt for publishing the Unofficial Format
5: # Specification for IDL .sav files, without which this Python module would not
6: # exist (http://cow.physics.wisc.edu/~craigm/idl/savefmt).
7: 
8: # This code was developed by with permission from ITT Visual Information
9: # Systems. IDL(r) is a registered trademark of ITT Visual Information Systems,
10: # Inc. for their Interactive Data Language software.
11: 
12: # Permission is hereby granted, free of charge, to any person obtaining a
13: # copy of this software and associated documentation files (the "Software"),
14: # to deal in the Software without restriction, including without limitation
15: # the rights to use, copy, modify, merge, publish, distribute, sublicense,
16: # and/or sell copies of the Software, and to permit persons to whom the
17: # Software is furnished to do so, subject to the following conditions:
18: 
19: # The above copyright notice and this permission notice shall be included in
20: # all copies or substantial portions of the Software.
21: 
22: # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
23: # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
24: # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
25: # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
26: # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
27: # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
28: # DEALINGS IN THE SOFTWARE.
29: 
30: from __future__ import division, print_function, absolute_import
31: 
32: import struct
33: import numpy as np
34: from numpy.compat import asstr
35: import tempfile
36: import zlib
37: import warnings
38: 
39: # Define the different data types that can be found in an IDL save file
40: DTYPE_DICT = {1: '>u1',
41:               2: '>i2',
42:               3: '>i4',
43:               4: '>f4',
44:               5: '>f8',
45:               6: '>c8',
46:               7: '|O',
47:               8: '|O',
48:               9: '>c16',
49:               10: '|O',
50:               11: '|O',
51:               12: '>u2',
52:               13: '>u4',
53:               14: '>i8',
54:               15: '>u8'}
55: 
56: # Define the different record types that can be found in an IDL save file
57: RECTYPE_DICT = {0: "START_MARKER",
58:                 1: "COMMON_VARIABLE",
59:                 2: "VARIABLE",
60:                 3: "SYSTEM_VARIABLE",
61:                 6: "END_MARKER",
62:                 10: "TIMESTAMP",
63:                 12: "COMPILED",
64:                 13: "IDENTIFICATION",
65:                 14: "VERSION",
66:                 15: "HEAP_HEADER",
67:                 16: "HEAP_DATA",
68:                 17: "PROMOTE64",
69:                 19: "NOTICE",
70:                 20: "DESCRIPTION"}
71: 
72: # Define a dictionary to contain structure definitions
73: STRUCT_DICT = {}
74: 
75: 
76: def _align_32(f):
77:     '''Align to the next 32-bit position in a file'''
78: 
79:     pos = f.tell()
80:     if pos % 4 != 0:
81:         f.seek(pos + 4 - pos % 4)
82:     return
83: 
84: 
85: def _skip_bytes(f, n):
86:     '''Skip `n` bytes'''
87:     f.read(n)
88:     return
89: 
90: 
91: def _read_bytes(f, n):
92:     '''Read the next `n` bytes'''
93:     return f.read(n)
94: 
95: 
96: def _read_byte(f):
97:     '''Read a single byte'''
98:     return np.uint8(struct.unpack('>B', f.read(4)[:1])[0])
99: 
100: 
101: def _read_long(f):
102:     '''Read a signed 32-bit integer'''
103:     return np.int32(struct.unpack('>l', f.read(4))[0])
104: 
105: 
106: def _read_int16(f):
107:     '''Read a signed 16-bit integer'''
108:     return np.int16(struct.unpack('>h', f.read(4)[2:4])[0])
109: 
110: 
111: def _read_int32(f):
112:     '''Read a signed 32-bit integer'''
113:     return np.int32(struct.unpack('>i', f.read(4))[0])
114: 
115: 
116: def _read_int64(f):
117:     '''Read a signed 64-bit integer'''
118:     return np.int64(struct.unpack('>q', f.read(8))[0])
119: 
120: 
121: def _read_uint16(f):
122:     '''Read an unsigned 16-bit integer'''
123:     return np.uint16(struct.unpack('>H', f.read(4)[2:4])[0])
124: 
125: 
126: def _read_uint32(f):
127:     '''Read an unsigned 32-bit integer'''
128:     return np.uint32(struct.unpack('>I', f.read(4))[0])
129: 
130: 
131: def _read_uint64(f):
132:     '''Read an unsigned 64-bit integer'''
133:     return np.uint64(struct.unpack('>Q', f.read(8))[0])
134: 
135: 
136: def _read_float32(f):
137:     '''Read a 32-bit float'''
138:     return np.float32(struct.unpack('>f', f.read(4))[0])
139: 
140: 
141: def _read_float64(f):
142:     '''Read a 64-bit float'''
143:     return np.float64(struct.unpack('>d', f.read(8))[0])
144: 
145: 
146: class Pointer(object):
147:     '''Class used to define pointers'''
148: 
149:     def __init__(self, index):
150:         self.index = index
151:         return
152: 
153: 
154: class ObjectPointer(Pointer):
155:     '''Class used to define object pointers'''
156:     pass
157: 
158: 
159: def _read_string(f):
160:     '''Read a string'''
161:     length = _read_long(f)
162:     if length > 0:
163:         chars = _read_bytes(f, length)
164:         _align_32(f)
165:         chars = asstr(chars)
166:     else:
167:         chars = ''
168:     return chars
169: 
170: 
171: def _read_string_data(f):
172:     '''Read a data string (length is specified twice)'''
173:     length = _read_long(f)
174:     if length > 0:
175:         length = _read_long(f)
176:         string_data = _read_bytes(f, length)
177:         _align_32(f)
178:     else:
179:         string_data = ''
180:     return string_data
181: 
182: 
183: def _read_data(f, dtype):
184:     '''Read a variable with a specified data type'''
185:     if dtype == 1:
186:         if _read_int32(f) != 1:
187:             raise Exception("Error occurred while reading byte variable")
188:         return _read_byte(f)
189:     elif dtype == 2:
190:         return _read_int16(f)
191:     elif dtype == 3:
192:         return _read_int32(f)
193:     elif dtype == 4:
194:         return _read_float32(f)
195:     elif dtype == 5:
196:         return _read_float64(f)
197:     elif dtype == 6:
198:         real = _read_float32(f)
199:         imag = _read_float32(f)
200:         return np.complex64(real + imag * 1j)
201:     elif dtype == 7:
202:         return _read_string_data(f)
203:     elif dtype == 8:
204:         raise Exception("Should not be here - please report this")
205:     elif dtype == 9:
206:         real = _read_float64(f)
207:         imag = _read_float64(f)
208:         return np.complex128(real + imag * 1j)
209:     elif dtype == 10:
210:         return Pointer(_read_int32(f))
211:     elif dtype == 11:
212:         return ObjectPointer(_read_int32(f))
213:     elif dtype == 12:
214:         return _read_uint16(f)
215:     elif dtype == 13:
216:         return _read_uint32(f)
217:     elif dtype == 14:
218:         return _read_int64(f)
219:     elif dtype == 15:
220:         return _read_uint64(f)
221:     else:
222:         raise Exception("Unknown IDL type: %i - please report this" % dtype)
223: 
224: 
225: def _read_structure(f, array_desc, struct_desc):
226:     '''
227:     Read a structure, with the array and structure descriptors given as
228:     `array_desc` and `structure_desc` respectively.
229:     '''
230: 
231:     nrows = array_desc['nelements']
232:     columns = struct_desc['tagtable']
233: 
234:     dtype = []
235:     for col in columns:
236:         if col['structure'] or col['array']:
237:             dtype.append(((col['name'].lower(), col['name']), np.object_))
238:         else:
239:             if col['typecode'] in DTYPE_DICT:
240:                 dtype.append(((col['name'].lower(), col['name']),
241:                                     DTYPE_DICT[col['typecode']]))
242:             else:
243:                 raise Exception("Variable type %i not implemented" %
244:                                                             col['typecode'])
245: 
246:     structure = np.recarray((nrows, ), dtype=dtype)
247: 
248:     for i in range(nrows):
249:         for col in columns:
250:             dtype = col['typecode']
251:             if col['structure']:
252:                 structure[col['name']][i] = _read_structure(f,
253:                                       struct_desc['arrtable'][col['name']],
254:                                       struct_desc['structtable'][col['name']])
255:             elif col['array']:
256:                 structure[col['name']][i] = _read_array(f, dtype,
257:                                       struct_desc['arrtable'][col['name']])
258:             else:
259:                 structure[col['name']][i] = _read_data(f, dtype)
260: 
261:     # Reshape structure if needed
262:     if array_desc['ndims'] > 1:
263:         dims = array_desc['dims'][:int(array_desc['ndims'])]
264:         dims.reverse()
265:         structure = structure.reshape(dims)
266: 
267:     return structure
268: 
269: 
270: def _read_array(f, typecode, array_desc):
271:     '''
272:     Read an array of type `typecode`, with the array descriptor given as
273:     `array_desc`.
274:     '''
275: 
276:     if typecode in [1, 3, 4, 5, 6, 9, 13, 14, 15]:
277: 
278:         if typecode == 1:
279:             nbytes = _read_int32(f)
280:             if nbytes != array_desc['nbytes']:
281:                 warnings.warn("Not able to verify number of bytes from header")
282: 
283:         # Read bytes as numpy array
284:         array = np.fromstring(f.read(array_desc['nbytes']),
285:                                 dtype=DTYPE_DICT[typecode])
286: 
287:     elif typecode in [2, 12]:
288: 
289:         # These are 2 byte types, need to skip every two as they are not packed
290: 
291:         array = np.fromstring(f.read(array_desc['nbytes']*2),
292:                                 dtype=DTYPE_DICT[typecode])[1::2]
293: 
294:     else:
295: 
296:         # Read bytes into list
297:         array = []
298:         for i in range(array_desc['nelements']):
299:             dtype = typecode
300:             data = _read_data(f, dtype)
301:             array.append(data)
302: 
303:         array = np.array(array, dtype=np.object_)
304: 
305:     # Reshape array if needed
306:     if array_desc['ndims'] > 1:
307:         dims = array_desc['dims'][:int(array_desc['ndims'])]
308:         dims.reverse()
309:         array = array.reshape(dims)
310: 
311:     # Go to next alignment position
312:     _align_32(f)
313: 
314:     return array
315: 
316: 
317: def _read_record(f):
318:     '''Function to read in a full record'''
319: 
320:     record = {'rectype': _read_long(f)}
321: 
322:     nextrec = _read_uint32(f)
323:     nextrec += _read_uint32(f) * 2**32
324: 
325:     _skip_bytes(f, 4)
326: 
327:     if not record['rectype'] in RECTYPE_DICT:
328:         raise Exception("Unknown RECTYPE: %i" % record['rectype'])
329: 
330:     record['rectype'] = RECTYPE_DICT[record['rectype']]
331: 
332:     if record['rectype'] in ["VARIABLE", "HEAP_DATA"]:
333: 
334:         if record['rectype'] == "VARIABLE":
335:             record['varname'] = _read_string(f)
336:         else:
337:             record['heap_index'] = _read_long(f)
338:             _skip_bytes(f, 4)
339: 
340:         rectypedesc = _read_typedesc(f)
341: 
342:         if rectypedesc['typecode'] == 0:
343: 
344:             if nextrec == f.tell():
345:                 record['data'] = None  # Indicates NULL value
346:             else:
347:                 raise ValueError("Unexpected type code: 0")
348: 
349:         else:
350: 
351:             varstart = _read_long(f)
352:             if varstart != 7:
353:                 raise Exception("VARSTART is not 7")
354: 
355:             if rectypedesc['structure']:
356:                 record['data'] = _read_structure(f, rectypedesc['array_desc'],
357:                                                     rectypedesc['struct_desc'])
358:             elif rectypedesc['array']:
359:                 record['data'] = _read_array(f, rectypedesc['typecode'],
360:                                                 rectypedesc['array_desc'])
361:             else:
362:                 dtype = rectypedesc['typecode']
363:                 record['data'] = _read_data(f, dtype)
364: 
365:     elif record['rectype'] == "TIMESTAMP":
366: 
367:         _skip_bytes(f, 4*256)
368:         record['date'] = _read_string(f)
369:         record['user'] = _read_string(f)
370:         record['host'] = _read_string(f)
371: 
372:     elif record['rectype'] == "VERSION":
373: 
374:         record['format'] = _read_long(f)
375:         record['arch'] = _read_string(f)
376:         record['os'] = _read_string(f)
377:         record['release'] = _read_string(f)
378: 
379:     elif record['rectype'] == "IDENTIFICATON":
380: 
381:         record['author'] = _read_string(f)
382:         record['title'] = _read_string(f)
383:         record['idcode'] = _read_string(f)
384: 
385:     elif record['rectype'] == "NOTICE":
386: 
387:         record['notice'] = _read_string(f)
388: 
389:     elif record['rectype'] == "DESCRIPTION":
390: 
391:         record['description'] = _read_string_data(f)
392: 
393:     elif record['rectype'] == "HEAP_HEADER":
394: 
395:         record['nvalues'] = _read_long(f)
396:         record['indices'] = []
397:         for i in range(record['nvalues']):
398:             record['indices'].append(_read_long(f))
399: 
400:     elif record['rectype'] == "COMMONBLOCK":
401: 
402:         record['nvars'] = _read_long(f)
403:         record['name'] = _read_string(f)
404:         record['varnames'] = []
405:         for i in range(record['nvars']):
406:             record['varnames'].append(_read_string(f))
407: 
408:     elif record['rectype'] == "END_MARKER":
409: 
410:         record['end'] = True
411: 
412:     elif record['rectype'] == "UNKNOWN":
413: 
414:         warnings.warn("Skipping UNKNOWN record")
415: 
416:     elif record['rectype'] == "SYSTEM_VARIABLE":
417: 
418:         warnings.warn("Skipping SYSTEM_VARIABLE record")
419: 
420:     else:
421: 
422:         raise Exception("record['rectype']=%s not implemented" %
423:                                                             record['rectype'])
424: 
425:     f.seek(nextrec)
426: 
427:     return record
428: 
429: 
430: def _read_typedesc(f):
431:     '''Function to read in a type descriptor'''
432: 
433:     typedesc = {'typecode': _read_long(f), 'varflags': _read_long(f)}
434: 
435:     if typedesc['varflags'] & 2 == 2:
436:         raise Exception("System variables not implemented")
437: 
438:     typedesc['array'] = typedesc['varflags'] & 4 == 4
439:     typedesc['structure'] = typedesc['varflags'] & 32 == 32
440: 
441:     if typedesc['structure']:
442:         typedesc['array_desc'] = _read_arraydesc(f)
443:         typedesc['struct_desc'] = _read_structdesc(f)
444:     elif typedesc['array']:
445:         typedesc['array_desc'] = _read_arraydesc(f)
446: 
447:     return typedesc
448: 
449: 
450: def _read_arraydesc(f):
451:     '''Function to read in an array descriptor'''
452: 
453:     arraydesc = {'arrstart': _read_long(f)}
454: 
455:     if arraydesc['arrstart'] == 8:
456: 
457:         _skip_bytes(f, 4)
458: 
459:         arraydesc['nbytes'] = _read_long(f)
460:         arraydesc['nelements'] = _read_long(f)
461:         arraydesc['ndims'] = _read_long(f)
462: 
463:         _skip_bytes(f, 8)
464: 
465:         arraydesc['nmax'] = _read_long(f)
466: 
467:         arraydesc['dims'] = []
468:         for d in range(arraydesc['nmax']):
469:             arraydesc['dims'].append(_read_long(f))
470: 
471:     elif arraydesc['arrstart'] == 18:
472: 
473:         warnings.warn("Using experimental 64-bit array read")
474: 
475:         _skip_bytes(f, 8)
476: 
477:         arraydesc['nbytes'] = _read_uint64(f)
478:         arraydesc['nelements'] = _read_uint64(f)
479:         arraydesc['ndims'] = _read_long(f)
480: 
481:         _skip_bytes(f, 8)
482: 
483:         arraydesc['nmax'] = 8
484: 
485:         arraydesc['dims'] = []
486:         for d in range(arraydesc['nmax']):
487:             v = _read_long(f)
488:             if v != 0:
489:                 raise Exception("Expected a zero in ARRAY_DESC")
490:             arraydesc['dims'].append(_read_long(f))
491: 
492:     else:
493: 
494:         raise Exception("Unknown ARRSTART: %i" % arraydesc['arrstart'])
495: 
496:     return arraydesc
497: 
498: 
499: def _read_structdesc(f):
500:     '''Function to read in a structure descriptor'''
501: 
502:     structdesc = {}
503: 
504:     structstart = _read_long(f)
505:     if structstart != 9:
506:         raise Exception("STRUCTSTART should be 9")
507: 
508:     structdesc['name'] = _read_string(f)
509:     predef = _read_long(f)
510:     structdesc['ntags'] = _read_long(f)
511:     structdesc['nbytes'] = _read_long(f)
512: 
513:     structdesc['predef'] = predef & 1
514:     structdesc['inherits'] = predef & 2
515:     structdesc['is_super'] = predef & 4
516: 
517:     if not structdesc['predef']:
518: 
519:         structdesc['tagtable'] = []
520:         for t in range(structdesc['ntags']):
521:             structdesc['tagtable'].append(_read_tagdesc(f))
522: 
523:         for tag in structdesc['tagtable']:
524:             tag['name'] = _read_string(f)
525: 
526:         structdesc['arrtable'] = {}
527:         for tag in structdesc['tagtable']:
528:             if tag['array']:
529:                 structdesc['arrtable'][tag['name']] = _read_arraydesc(f)
530: 
531:         structdesc['structtable'] = {}
532:         for tag in structdesc['tagtable']:
533:             if tag['structure']:
534:                 structdesc['structtable'][tag['name']] = _read_structdesc(f)
535: 
536:         if structdesc['inherits'] or structdesc['is_super']:
537:             structdesc['classname'] = _read_string(f)
538:             structdesc['nsupclasses'] = _read_long(f)
539:             structdesc['supclassnames'] = []
540:             for s in range(structdesc['nsupclasses']):
541:                 structdesc['supclassnames'].append(_read_string(f))
542:             structdesc['supclasstable'] = []
543:             for s in range(structdesc['nsupclasses']):
544:                 structdesc['supclasstable'].append(_read_structdesc(f))
545: 
546:         STRUCT_DICT[structdesc['name']] = structdesc
547: 
548:     else:
549: 
550:         if not structdesc['name'] in STRUCT_DICT:
551:             raise Exception("PREDEF=1 but can't find definition")
552: 
553:         structdesc = STRUCT_DICT[structdesc['name']]
554: 
555:     return structdesc
556: 
557: 
558: def _read_tagdesc(f):
559:     '''Function to read in a tag descriptor'''
560: 
561:     tagdesc = {'offset': _read_long(f)}
562: 
563:     if tagdesc['offset'] == -1:
564:         tagdesc['offset'] = _read_uint64(f)
565: 
566:     tagdesc['typecode'] = _read_long(f)
567:     tagflags = _read_long(f)
568: 
569:     tagdesc['array'] = tagflags & 4 == 4
570:     tagdesc['structure'] = tagflags & 32 == 32
571:     tagdesc['scalar'] = tagdesc['typecode'] in DTYPE_DICT
572:     # Assume '10'x is scalar
573: 
574:     return tagdesc
575: 
576: 
577: def _replace_heap(variable, heap):
578: 
579:     if isinstance(variable, Pointer):
580: 
581:         while isinstance(variable, Pointer):
582: 
583:             if variable.index == 0:
584:                 variable = None
585:             else:
586:                 if variable.index in heap:
587:                     variable = heap[variable.index]
588:                 else:
589:                     warnings.warn("Variable referenced by pointer not found "
590:                                   "in heap: variable will be set to None")
591:                     variable = None
592: 
593:         replace, new = _replace_heap(variable, heap)
594: 
595:         if replace:
596:             variable = new
597: 
598:         return True, variable
599: 
600:     elif isinstance(variable, np.core.records.recarray):
601: 
602:         # Loop over records
603:         for ir, record in enumerate(variable):
604: 
605:             replace, new = _replace_heap(record, heap)
606: 
607:             if replace:
608:                 variable[ir] = new
609: 
610:         return False, variable
611: 
612:     elif isinstance(variable, np.core.records.record):
613: 
614:         # Loop over values
615:         for iv, value in enumerate(variable):
616: 
617:             replace, new = _replace_heap(value, heap)
618: 
619:             if replace:
620:                 variable[iv] = new
621: 
622:         return False, variable
623: 
624:     elif isinstance(variable, np.ndarray):
625: 
626:         # Loop over values if type is np.object_
627:         if variable.dtype.type is np.object_:
628: 
629:             for iv in range(variable.size):
630: 
631:                 replace, new = _replace_heap(variable.item(iv), heap)
632: 
633:                 if replace:
634:                     variable.itemset(iv, new)
635: 
636:         return False, variable
637: 
638:     else:
639: 
640:         return False, variable
641: 
642: 
643: class AttrDict(dict):
644:     '''
645:     A case-insensitive dictionary with access via item, attribute, and call
646:     notations:
647: 
648:         >>> d = AttrDict()
649:         >>> d['Variable'] = 123
650:         >>> d['Variable']
651:         123
652:         >>> d.Variable
653:         123
654:         >>> d.variable
655:         123
656:         >>> d('VARIABLE')
657:         123
658:     '''
659: 
660:     def __init__(self, init={}):
661:         dict.__init__(self, init)
662: 
663:     def __getitem__(self, name):
664:         return super(AttrDict, self).__getitem__(name.lower())
665: 
666:     def __setitem__(self, key, value):
667:         return super(AttrDict, self).__setitem__(key.lower(), value)
668: 
669:     __getattr__ = __getitem__
670:     __setattr__ = __setitem__
671:     __call__ = __getitem__
672: 
673: 
674: def readsav(file_name, idict=None, python_dict=False,
675:             uncompressed_file_name=None, verbose=False):
676:     '''
677:     Read an IDL .sav file.
678: 
679:     Parameters
680:     ----------
681:     file_name : str
682:         Name of the IDL save file.
683:     idict : dict, optional
684:         Dictionary in which to insert .sav file variables.
685:     python_dict : bool, optional
686:         By default, the object return is not a Python dictionary, but a
687:         case-insensitive dictionary with item, attribute, and call access
688:         to variables. To get a standard Python dictionary, set this option
689:         to True.
690:     uncompressed_file_name : str, optional
691:         This option only has an effect for .sav files written with the
692:         /compress option. If a file name is specified, compressed .sav
693:         files are uncompressed to this file. Otherwise, readsav will use
694:         the `tempfile` module to determine a temporary filename
695:         automatically, and will remove the temporary file upon successfully
696:         reading it in.
697:     verbose : bool, optional
698:         Whether to print out information about the save file, including
699:         the records read, and available variables.
700: 
701:     Returns
702:     -------
703:     idl_dict : AttrDict or dict
704:         If `python_dict` is set to False (default), this function returns a
705:         case-insensitive dictionary with item, attribute, and call access
706:         to variables. If `python_dict` is set to True, this function
707:         returns a Python dictionary with all variable names in lowercase.
708:         If `idict` was specified, then variables are written to the
709:         dictionary specified, and the updated dictionary is returned.
710: 
711:     '''
712: 
713:     # Initialize record and variable holders
714:     records = []
715:     if python_dict or idict:
716:         variables = {}
717:     else:
718:         variables = AttrDict()
719: 
720:     # Open the IDL file
721:     f = open(file_name, 'rb')
722: 
723:     # Read the signature, which should be 'SR'
724:     signature = _read_bytes(f, 2)
725:     if signature != b'SR':
726:         raise Exception("Invalid SIGNATURE: %s" % signature)
727: 
728:     # Next, the record format, which is '\x00\x04' for normal .sav
729:     # files, and '\x00\x06' for compressed .sav files.
730:     recfmt = _read_bytes(f, 2)
731: 
732:     if recfmt == b'\x00\x04':
733:         pass
734: 
735:     elif recfmt == b'\x00\x06':
736: 
737:         if verbose:
738:             print("IDL Save file is compressed")
739: 
740:         if uncompressed_file_name:
741:             fout = open(uncompressed_file_name, 'w+b')
742:         else:
743:             fout = tempfile.NamedTemporaryFile(suffix='.sav')
744: 
745:         if verbose:
746:             print(" -> expanding to %s" % fout.name)
747: 
748:         # Write header
749:         fout.write(b'SR\x00\x04')
750: 
751:         # Cycle through records
752:         while True:
753: 
754:             # Read record type
755:             rectype = _read_long(f)
756:             fout.write(struct.pack('>l', int(rectype)))
757: 
758:             # Read position of next record and return as int
759:             nextrec = _read_uint32(f)
760:             nextrec += _read_uint32(f) * 2**32
761: 
762:             # Read the unknown 4 bytes
763:             unknown = f.read(4)
764: 
765:             # Check if the end of the file has been reached
766:             if RECTYPE_DICT[rectype] == 'END_MARKER':
767:                 fout.write(struct.pack('>I', int(nextrec) % 2**32))
768:                 fout.write(struct.pack('>I', int((nextrec - (nextrec % 2**32)) / 2**32)))
769:                 fout.write(unknown)
770:                 break
771: 
772:             # Find current position
773:             pos = f.tell()
774: 
775:             # Decompress record
776:             rec_string = zlib.decompress(f.read(nextrec-pos))
777: 
778:             # Find new position of next record
779:             nextrec = fout.tell() + len(rec_string) + 12
780: 
781:             # Write out record
782:             fout.write(struct.pack('>I', int(nextrec % 2**32)))
783:             fout.write(struct.pack('>I', int((nextrec - (nextrec % 2**32)) / 2**32)))
784:             fout.write(unknown)
785:             fout.write(rec_string)
786: 
787:         # Close the original compressed file
788:         f.close()
789: 
790:         # Set f to be the decompressed file, and skip the first four bytes
791:         f = fout
792:         f.seek(4)
793: 
794:     else:
795:         raise Exception("Invalid RECFMT: %s" % recfmt)
796: 
797:     # Loop through records, and add them to the list
798:     while True:
799:         r = _read_record(f)
800:         records.append(r)
801:         if 'end' in r:
802:             if r['end']:
803:                 break
804: 
805:     # Close the file
806:     f.close()
807: 
808:     # Find heap data variables
809:     heap = {}
810:     for r in records:
811:         if r['rectype'] == "HEAP_DATA":
812:             heap[r['heap_index']] = r['data']
813: 
814:     # Find all variables
815:     for r in records:
816:         if r['rectype'] == "VARIABLE":
817:             replace, new = _replace_heap(r['data'], heap)
818:             if replace:
819:                 r['data'] = new
820:             variables[r['varname'].lower()] = r['data']
821: 
822:     if verbose:
823: 
824:         # Print out timestamp info about the file
825:         for record in records:
826:             if record['rectype'] == "TIMESTAMP":
827:                 print("-"*50)
828:                 print("Date: %s" % record['date'])
829:                 print("User: %s" % record['user'])
830:                 print("Host: %s" % record['host'])
831:                 break
832: 
833:         # Print out version info about the file
834:         for record in records:
835:             if record['rectype'] == "VERSION":
836:                 print("-"*50)
837:                 print("Format: %s" % record['format'])
838:                 print("Architecture: %s" % record['arch'])
839:                 print("Operating System: %s" % record['os'])
840:                 print("IDL Version: %s" % record['release'])
841:                 break
842: 
843:         # Print out identification info about the file
844:         for record in records:
845:             if record['rectype'] == "IDENTIFICATON":
846:                 print("-"*50)
847:                 print("Author: %s" % record['author'])
848:                 print("Title: %s" % record['title'])
849:                 print("ID Code: %s" % record['idcode'])
850:                 break
851: 
852:         # Print out descriptions saved with the file
853:         for record in records:
854:             if record['rectype'] == "DESCRIPTION":
855:                 print("-"*50)
856:                 print("Description: %s" % record['description'])
857:                 break
858: 
859:         print("-"*50)
860:         print("Successfully read %i records of which:" %
861:                                             (len(records)))
862: 
863:         # Create convenience list of record types
864:         rectypes = [r['rectype'] for r in records]
865: 
866:         for rt in set(rectypes):
867:             if rt != 'END_MARKER':
868:                 print(" - %i are of type %s" % (rectypes.count(rt), rt))
869:         print("-"*50)
870: 
871:         if 'VARIABLE' in rectypes:
872:             print("Available variables:")
873:             for var in variables:
874:                 print(" - %s [%s]" % (var, type(variables[var])))
875:             print("-"*50)
876: 
877:     if idict:
878:         for var in variables:
879:             idict[var] = variables[var]
880:         return idict
881:     else:
882:         return variables
883: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'import struct' statement (line 32)
import struct

import_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'struct', struct, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import numpy' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_119612 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy')

if (type(import_119612) is not StypyTypeError):

    if (import_119612 != 'pyd_module'):
        __import__(import_119612)
        sys_modules_119613 = sys.modules[import_119612]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'np', sys_modules_119613.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', import_119612)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from numpy.compat import asstr' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_119614 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.compat')

if (type(import_119614) is not StypyTypeError):

    if (import_119614 != 'pyd_module'):
        __import__(import_119614)
        sys_modules_119615 = sys.modules[import_119614]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.compat', sys_modules_119615.module_type_store, module_type_store, ['asstr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_119615, sys_modules_119615.module_type_store, module_type_store)
    else:
        from numpy.compat import asstr

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.compat', None, module_type_store, ['asstr'], [asstr])

else:
    # Assigning a type to the variable 'numpy.compat' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.compat', import_119614)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'import tempfile' statement (line 35)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'import zlib' statement (line 36)
import zlib

import_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'zlib', zlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'import warnings' statement (line 37)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'warnings', warnings, module_type_store)


# Assigning a Dict to a Name (line 40):

# Assigning a Dict to a Name (line 40):

# Obtaining an instance of the builtin type 'dict' (line 40)
dict_119616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 40)
# Adding element type (key, value) (line 40)
int_119617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'int')
str_119618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 17), 'str', '>u1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119617, str_119618))
# Adding element type (key, value) (line 40)
int_119619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'int')
str_119620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 17), 'str', '>i2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119619, str_119620))
# Adding element type (key, value) (line 40)
int_119621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'int')
str_119622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'str', '>i4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119621, str_119622))
# Adding element type (key, value) (line 40)
int_119623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'int')
str_119624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'str', '>f4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119623, str_119624))
# Adding element type (key, value) (line 40)
int_119625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 14), 'int')
str_119626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'str', '>f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119625, str_119626))
# Adding element type (key, value) (line 40)
int_119627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'int')
str_119628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'str', '>c8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119627, str_119628))
# Adding element type (key, value) (line 40)
int_119629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'int')
str_119630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'str', '|O')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119629, str_119630))
# Adding element type (key, value) (line 40)
int_119631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'int')
str_119632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 17), 'str', '|O')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119631, str_119632))
# Adding element type (key, value) (line 40)
int_119633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 14), 'int')
str_119634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'str', '>c16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119633, str_119634))
# Adding element type (key, value) (line 40)
int_119635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 14), 'int')
str_119636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'str', '|O')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119635, str_119636))
# Adding element type (key, value) (line 40)
int_119637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 14), 'int')
str_119638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'str', '|O')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119637, str_119638))
# Adding element type (key, value) (line 40)
int_119639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'int')
str_119640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 18), 'str', '>u2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119639, str_119640))
# Adding element type (key, value) (line 40)
int_119641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'int')
str_119642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'str', '>u4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119641, str_119642))
# Adding element type (key, value) (line 40)
int_119643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 14), 'int')
str_119644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'str', '>i8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119643, str_119644))
# Adding element type (key, value) (line 40)
int_119645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'int')
str_119646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'str', '>u8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), dict_119616, (int_119645, str_119646))

# Assigning a type to the variable 'DTYPE_DICT' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'DTYPE_DICT', dict_119616)

# Assigning a Dict to a Name (line 57):

# Assigning a Dict to a Name (line 57):

# Obtaining an instance of the builtin type 'dict' (line 57)
dict_119647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 57)
# Adding element type (key, value) (line 57)
int_119648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'int')
str_119649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'str', 'START_MARKER')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119648, str_119649))
# Adding element type (key, value) (line 57)
int_119650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 16), 'int')
str_119651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'str', 'COMMON_VARIABLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119650, str_119651))
# Adding element type (key, value) (line 57)
int_119652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'int')
str_119653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'str', 'VARIABLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119652, str_119653))
# Adding element type (key, value) (line 57)
int_119654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'int')
str_119655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'str', 'SYSTEM_VARIABLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119654, str_119655))
# Adding element type (key, value) (line 57)
int_119656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'int')
str_119657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'str', 'END_MARKER')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119656, str_119657))
# Adding element type (key, value) (line 57)
int_119658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'int')
str_119659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'str', 'TIMESTAMP')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119658, str_119659))
# Adding element type (key, value) (line 57)
int_119660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 16), 'int')
str_119661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'str', 'COMPILED')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119660, str_119661))
# Adding element type (key, value) (line 57)
int_119662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'int')
str_119663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'str', 'IDENTIFICATION')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119662, str_119663))
# Adding element type (key, value) (line 57)
int_119664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 16), 'int')
str_119665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'str', 'VERSION')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119664, str_119665))
# Adding element type (key, value) (line 57)
int_119666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'int')
str_119667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'str', 'HEAP_HEADER')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119666, str_119667))
# Adding element type (key, value) (line 57)
int_119668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 16), 'int')
str_119669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'str', 'HEAP_DATA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119668, str_119669))
# Adding element type (key, value) (line 57)
int_119670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'int')
str_119671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'str', 'PROMOTE64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119670, str_119671))
# Adding element type (key, value) (line 57)
int_119672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 16), 'int')
str_119673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'str', 'NOTICE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119672, str_119673))
# Adding element type (key, value) (line 57)
int_119674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 16), 'int')
str_119675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'str', 'DESCRIPTION')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), dict_119647, (int_119674, str_119675))

# Assigning a type to the variable 'RECTYPE_DICT' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'RECTYPE_DICT', dict_119647)

# Assigning a Dict to a Name (line 73):

# Assigning a Dict to a Name (line 73):

# Obtaining an instance of the builtin type 'dict' (line 73)
dict_119676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 73)

# Assigning a type to the variable 'STRUCT_DICT' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'STRUCT_DICT', dict_119676)

@norecursion
def _align_32(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_align_32'
    module_type_store = module_type_store.open_function_context('_align_32', 76, 0, False)
    
    # Passed parameters checking function
    _align_32.stypy_localization = localization
    _align_32.stypy_type_of_self = None
    _align_32.stypy_type_store = module_type_store
    _align_32.stypy_function_name = '_align_32'
    _align_32.stypy_param_names_list = ['f']
    _align_32.stypy_varargs_param_name = None
    _align_32.stypy_kwargs_param_name = None
    _align_32.stypy_call_defaults = defaults
    _align_32.stypy_call_varargs = varargs
    _align_32.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_align_32', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_align_32', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_align_32(...)' code ##################

    str_119677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'Align to the next 32-bit position in a file')
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to tell(...): (line 79)
    # Processing the call keyword arguments (line 79)
    kwargs_119680 = {}
    # Getting the type of 'f' (line 79)
    f_119678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 10), 'f', False)
    # Obtaining the member 'tell' of a type (line 79)
    tell_119679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 10), f_119678, 'tell')
    # Calling tell(args, kwargs) (line 79)
    tell_call_result_119681 = invoke(stypy.reporting.localization.Localization(__file__, 79, 10), tell_119679, *[], **kwargs_119680)
    
    # Assigning a type to the variable 'pos' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'pos', tell_call_result_119681)
    
    
    # Getting the type of 'pos' (line 80)
    pos_119682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'pos')
    int_119683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'int')
    # Applying the binary operator '%' (line 80)
    result_mod_119684 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), '%', pos_119682, int_119683)
    
    int_119685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'int')
    # Applying the binary operator '!=' (line 80)
    result_ne_119686 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), '!=', result_mod_119684, int_119685)
    
    # Testing the type of an if condition (line 80)
    if_condition_119687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_ne_119686)
    # Assigning a type to the variable 'if_condition_119687' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_119687', if_condition_119687)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to seek(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'pos' (line 81)
    pos_119690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'pos', False)
    int_119691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'int')
    # Applying the binary operator '+' (line 81)
    result_add_119692 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '+', pos_119690, int_119691)
    
    # Getting the type of 'pos' (line 81)
    pos_119693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'pos', False)
    int_119694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'int')
    # Applying the binary operator '%' (line 81)
    result_mod_119695 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 25), '%', pos_119693, int_119694)
    
    # Applying the binary operator '-' (line 81)
    result_sub_119696 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 23), '-', result_add_119692, result_mod_119695)
    
    # Processing the call keyword arguments (line 81)
    kwargs_119697 = {}
    # Getting the type of 'f' (line 81)
    f_119688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'f', False)
    # Obtaining the member 'seek' of a type (line 81)
    seek_119689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), f_119688, 'seek')
    # Calling seek(args, kwargs) (line 81)
    seek_call_result_119698 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), seek_119689, *[result_sub_119696], **kwargs_119697)
    
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of '_align_32(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_align_32' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_119699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119699)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_align_32'
    return stypy_return_type_119699

# Assigning a type to the variable '_align_32' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), '_align_32', _align_32)

@norecursion
def _skip_bytes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_skip_bytes'
    module_type_store = module_type_store.open_function_context('_skip_bytes', 85, 0, False)
    
    # Passed parameters checking function
    _skip_bytes.stypy_localization = localization
    _skip_bytes.stypy_type_of_self = None
    _skip_bytes.stypy_type_store = module_type_store
    _skip_bytes.stypy_function_name = '_skip_bytes'
    _skip_bytes.stypy_param_names_list = ['f', 'n']
    _skip_bytes.stypy_varargs_param_name = None
    _skip_bytes.stypy_kwargs_param_name = None
    _skip_bytes.stypy_call_defaults = defaults
    _skip_bytes.stypy_call_varargs = varargs
    _skip_bytes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_skip_bytes', ['f', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_skip_bytes', localization, ['f', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_skip_bytes(...)' code ##################

    str_119700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'str', 'Skip `n` bytes')
    
    # Call to read(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'n' (line 87)
    n_119703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'n', False)
    # Processing the call keyword arguments (line 87)
    kwargs_119704 = {}
    # Getting the type of 'f' (line 87)
    f_119701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'f', False)
    # Obtaining the member 'read' of a type (line 87)
    read_119702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 4), f_119701, 'read')
    # Calling read(args, kwargs) (line 87)
    read_call_result_119705 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), read_119702, *[n_119703], **kwargs_119704)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of '_skip_bytes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_skip_bytes' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_119706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119706)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_skip_bytes'
    return stypy_return_type_119706

# Assigning a type to the variable '_skip_bytes' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), '_skip_bytes', _skip_bytes)

@norecursion
def _read_bytes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_bytes'
    module_type_store = module_type_store.open_function_context('_read_bytes', 91, 0, False)
    
    # Passed parameters checking function
    _read_bytes.stypy_localization = localization
    _read_bytes.stypy_type_of_self = None
    _read_bytes.stypy_type_store = module_type_store
    _read_bytes.stypy_function_name = '_read_bytes'
    _read_bytes.stypy_param_names_list = ['f', 'n']
    _read_bytes.stypy_varargs_param_name = None
    _read_bytes.stypy_kwargs_param_name = None
    _read_bytes.stypy_call_defaults = defaults
    _read_bytes.stypy_call_varargs = varargs
    _read_bytes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_bytes', ['f', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_bytes', localization, ['f', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_bytes(...)' code ##################

    str_119707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'str', 'Read the next `n` bytes')
    
    # Call to read(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'n' (line 93)
    n_119710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'n', False)
    # Processing the call keyword arguments (line 93)
    kwargs_119711 = {}
    # Getting the type of 'f' (line 93)
    f_119708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'f', False)
    # Obtaining the member 'read' of a type (line 93)
    read_119709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), f_119708, 'read')
    # Calling read(args, kwargs) (line 93)
    read_call_result_119712 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), read_119709, *[n_119710], **kwargs_119711)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type', read_call_result_119712)
    
    # ################# End of '_read_bytes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_bytes' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_119713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_bytes'
    return stypy_return_type_119713

# Assigning a type to the variable '_read_bytes' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), '_read_bytes', _read_bytes)

@norecursion
def _read_byte(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_byte'
    module_type_store = module_type_store.open_function_context('_read_byte', 96, 0, False)
    
    # Passed parameters checking function
    _read_byte.stypy_localization = localization
    _read_byte.stypy_type_of_self = None
    _read_byte.stypy_type_store = module_type_store
    _read_byte.stypy_function_name = '_read_byte'
    _read_byte.stypy_param_names_list = ['f']
    _read_byte.stypy_varargs_param_name = None
    _read_byte.stypy_kwargs_param_name = None
    _read_byte.stypy_call_defaults = defaults
    _read_byte.stypy_call_varargs = varargs
    _read_byte.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_byte', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_byte', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_byte(...)' code ##################

    str_119714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'str', 'Read a single byte')
    
    # Call to uint8(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Obtaining the type of the subscript
    int_119717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 55), 'int')
    
    # Call to unpack(...): (line 98)
    # Processing the call arguments (line 98)
    str_119720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 34), 'str', '>B')
    
    # Obtaining the type of the subscript
    int_119721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 51), 'int')
    slice_119722 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 40), None, int_119721, None)
    
    # Call to read(...): (line 98)
    # Processing the call arguments (line 98)
    int_119725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 47), 'int')
    # Processing the call keyword arguments (line 98)
    kwargs_119726 = {}
    # Getting the type of 'f' (line 98)
    f_119723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'f', False)
    # Obtaining the member 'read' of a type (line 98)
    read_119724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 40), f_119723, 'read')
    # Calling read(args, kwargs) (line 98)
    read_call_result_119727 = invoke(stypy.reporting.localization.Localization(__file__, 98, 40), read_119724, *[int_119725], **kwargs_119726)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___119728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 40), read_call_result_119727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_119729 = invoke(stypy.reporting.localization.Localization(__file__, 98, 40), getitem___119728, slice_119722)
    
    # Processing the call keyword arguments (line 98)
    kwargs_119730 = {}
    # Getting the type of 'struct' (line 98)
    struct_119718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 98)
    unpack_119719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), struct_119718, 'unpack')
    # Calling unpack(args, kwargs) (line 98)
    unpack_call_result_119731 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), unpack_119719, *[str_119720, subscript_call_result_119729], **kwargs_119730)
    
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___119732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), unpack_call_result_119731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_119733 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), getitem___119732, int_119717)
    
    # Processing the call keyword arguments (line 98)
    kwargs_119734 = {}
    # Getting the type of 'np' (line 98)
    np_119715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'np', False)
    # Obtaining the member 'uint8' of a type (line 98)
    uint8_119716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 11), np_119715, 'uint8')
    # Calling uint8(args, kwargs) (line 98)
    uint8_call_result_119735 = invoke(stypy.reporting.localization.Localization(__file__, 98, 11), uint8_119716, *[subscript_call_result_119733], **kwargs_119734)
    
    # Assigning a type to the variable 'stypy_return_type' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type', uint8_call_result_119735)
    
    # ################# End of '_read_byte(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_byte' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_119736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119736)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_byte'
    return stypy_return_type_119736

# Assigning a type to the variable '_read_byte' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), '_read_byte', _read_byte)

@norecursion
def _read_long(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_long'
    module_type_store = module_type_store.open_function_context('_read_long', 101, 0, False)
    
    # Passed parameters checking function
    _read_long.stypy_localization = localization
    _read_long.stypy_type_of_self = None
    _read_long.stypy_type_store = module_type_store
    _read_long.stypy_function_name = '_read_long'
    _read_long.stypy_param_names_list = ['f']
    _read_long.stypy_varargs_param_name = None
    _read_long.stypy_kwargs_param_name = None
    _read_long.stypy_call_defaults = defaults
    _read_long.stypy_call_varargs = varargs
    _read_long.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_long', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_long', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_long(...)' code ##################

    str_119737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'str', 'Read a signed 32-bit integer')
    
    # Call to int32(...): (line 103)
    # Processing the call arguments (line 103)
    
    # Obtaining the type of the subscript
    int_119740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 51), 'int')
    
    # Call to unpack(...): (line 103)
    # Processing the call arguments (line 103)
    str_119743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'str', '>l')
    
    # Call to read(...): (line 103)
    # Processing the call arguments (line 103)
    int_119746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 47), 'int')
    # Processing the call keyword arguments (line 103)
    kwargs_119747 = {}
    # Getting the type of 'f' (line 103)
    f_119744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'f', False)
    # Obtaining the member 'read' of a type (line 103)
    read_119745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 40), f_119744, 'read')
    # Calling read(args, kwargs) (line 103)
    read_call_result_119748 = invoke(stypy.reporting.localization.Localization(__file__, 103, 40), read_119745, *[int_119746], **kwargs_119747)
    
    # Processing the call keyword arguments (line 103)
    kwargs_119749 = {}
    # Getting the type of 'struct' (line 103)
    struct_119741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 103)
    unpack_119742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), struct_119741, 'unpack')
    # Calling unpack(args, kwargs) (line 103)
    unpack_call_result_119750 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), unpack_119742, *[str_119743, read_call_result_119748], **kwargs_119749)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___119751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), unpack_call_result_119750, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_119752 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), getitem___119751, int_119740)
    
    # Processing the call keyword arguments (line 103)
    kwargs_119753 = {}
    # Getting the type of 'np' (line 103)
    np_119738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'np', False)
    # Obtaining the member 'int32' of a type (line 103)
    int32_119739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 11), np_119738, 'int32')
    # Calling int32(args, kwargs) (line 103)
    int32_call_result_119754 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), int32_119739, *[subscript_call_result_119752], **kwargs_119753)
    
    # Assigning a type to the variable 'stypy_return_type' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type', int32_call_result_119754)
    
    # ################# End of '_read_long(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_long' in the type store
    # Getting the type of 'stypy_return_type' (line 101)
    stypy_return_type_119755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119755)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_long'
    return stypy_return_type_119755

# Assigning a type to the variable '_read_long' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '_read_long', _read_long)

@norecursion
def _read_int16(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_int16'
    module_type_store = module_type_store.open_function_context('_read_int16', 106, 0, False)
    
    # Passed parameters checking function
    _read_int16.stypy_localization = localization
    _read_int16.stypy_type_of_self = None
    _read_int16.stypy_type_store = module_type_store
    _read_int16.stypy_function_name = '_read_int16'
    _read_int16.stypy_param_names_list = ['f']
    _read_int16.stypy_varargs_param_name = None
    _read_int16.stypy_kwargs_param_name = None
    _read_int16.stypy_call_defaults = defaults
    _read_int16.stypy_call_varargs = varargs
    _read_int16.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_int16', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_int16', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_int16(...)' code ##################

    str_119756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'str', 'Read a signed 16-bit integer')
    
    # Call to int16(...): (line 108)
    # Processing the call arguments (line 108)
    
    # Obtaining the type of the subscript
    int_119759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 56), 'int')
    
    # Call to unpack(...): (line 108)
    # Processing the call arguments (line 108)
    str_119762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'str', '>h')
    
    # Obtaining the type of the subscript
    int_119763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 50), 'int')
    int_119764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 52), 'int')
    slice_119765 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 108, 40), int_119763, int_119764, None)
    
    # Call to read(...): (line 108)
    # Processing the call arguments (line 108)
    int_119768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 47), 'int')
    # Processing the call keyword arguments (line 108)
    kwargs_119769 = {}
    # Getting the type of 'f' (line 108)
    f_119766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'f', False)
    # Obtaining the member 'read' of a type (line 108)
    read_119767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 40), f_119766, 'read')
    # Calling read(args, kwargs) (line 108)
    read_call_result_119770 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), read_119767, *[int_119768], **kwargs_119769)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___119771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 40), read_call_result_119770, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_119772 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), getitem___119771, slice_119765)
    
    # Processing the call keyword arguments (line 108)
    kwargs_119773 = {}
    # Getting the type of 'struct' (line 108)
    struct_119760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 108)
    unpack_119761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), struct_119760, 'unpack')
    # Calling unpack(args, kwargs) (line 108)
    unpack_call_result_119774 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), unpack_119761, *[str_119762, subscript_call_result_119772], **kwargs_119773)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___119775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), unpack_call_result_119774, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_119776 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), getitem___119775, int_119759)
    
    # Processing the call keyword arguments (line 108)
    kwargs_119777 = {}
    # Getting the type of 'np' (line 108)
    np_119757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'np', False)
    # Obtaining the member 'int16' of a type (line 108)
    int16_119758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), np_119757, 'int16')
    # Calling int16(args, kwargs) (line 108)
    int16_call_result_119778 = invoke(stypy.reporting.localization.Localization(__file__, 108, 11), int16_119758, *[subscript_call_result_119776], **kwargs_119777)
    
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', int16_call_result_119778)
    
    # ################# End of '_read_int16(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_int16' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_119779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_int16'
    return stypy_return_type_119779

# Assigning a type to the variable '_read_int16' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), '_read_int16', _read_int16)

@norecursion
def _read_int32(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_int32'
    module_type_store = module_type_store.open_function_context('_read_int32', 111, 0, False)
    
    # Passed parameters checking function
    _read_int32.stypy_localization = localization
    _read_int32.stypy_type_of_self = None
    _read_int32.stypy_type_store = module_type_store
    _read_int32.stypy_function_name = '_read_int32'
    _read_int32.stypy_param_names_list = ['f']
    _read_int32.stypy_varargs_param_name = None
    _read_int32.stypy_kwargs_param_name = None
    _read_int32.stypy_call_defaults = defaults
    _read_int32.stypy_call_varargs = varargs
    _read_int32.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_int32', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_int32', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_int32(...)' code ##################

    str_119780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'str', 'Read a signed 32-bit integer')
    
    # Call to int32(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    int_119783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 51), 'int')
    
    # Call to unpack(...): (line 113)
    # Processing the call arguments (line 113)
    str_119786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 34), 'str', '>i')
    
    # Call to read(...): (line 113)
    # Processing the call arguments (line 113)
    int_119789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 47), 'int')
    # Processing the call keyword arguments (line 113)
    kwargs_119790 = {}
    # Getting the type of 'f' (line 113)
    f_119787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'f', False)
    # Obtaining the member 'read' of a type (line 113)
    read_119788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 40), f_119787, 'read')
    # Calling read(args, kwargs) (line 113)
    read_call_result_119791 = invoke(stypy.reporting.localization.Localization(__file__, 113, 40), read_119788, *[int_119789], **kwargs_119790)
    
    # Processing the call keyword arguments (line 113)
    kwargs_119792 = {}
    # Getting the type of 'struct' (line 113)
    struct_119784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 113)
    unpack_119785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), struct_119784, 'unpack')
    # Calling unpack(args, kwargs) (line 113)
    unpack_call_result_119793 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), unpack_119785, *[str_119786, read_call_result_119791], **kwargs_119792)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___119794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), unpack_call_result_119793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_119795 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), getitem___119794, int_119783)
    
    # Processing the call keyword arguments (line 113)
    kwargs_119796 = {}
    # Getting the type of 'np' (line 113)
    np_119781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'np', False)
    # Obtaining the member 'int32' of a type (line 113)
    int32_119782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), np_119781, 'int32')
    # Calling int32(args, kwargs) (line 113)
    int32_call_result_119797 = invoke(stypy.reporting.localization.Localization(__file__, 113, 11), int32_119782, *[subscript_call_result_119795], **kwargs_119796)
    
    # Assigning a type to the variable 'stypy_return_type' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type', int32_call_result_119797)
    
    # ################# End of '_read_int32(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_int32' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_119798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119798)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_int32'
    return stypy_return_type_119798

# Assigning a type to the variable '_read_int32' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), '_read_int32', _read_int32)

@norecursion
def _read_int64(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_int64'
    module_type_store = module_type_store.open_function_context('_read_int64', 116, 0, False)
    
    # Passed parameters checking function
    _read_int64.stypy_localization = localization
    _read_int64.stypy_type_of_self = None
    _read_int64.stypy_type_store = module_type_store
    _read_int64.stypy_function_name = '_read_int64'
    _read_int64.stypy_param_names_list = ['f']
    _read_int64.stypy_varargs_param_name = None
    _read_int64.stypy_kwargs_param_name = None
    _read_int64.stypy_call_defaults = defaults
    _read_int64.stypy_call_varargs = varargs
    _read_int64.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_int64', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_int64', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_int64(...)' code ##################

    str_119799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 4), 'str', 'Read a signed 64-bit integer')
    
    # Call to int64(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Obtaining the type of the subscript
    int_119802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 51), 'int')
    
    # Call to unpack(...): (line 118)
    # Processing the call arguments (line 118)
    str_119805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 34), 'str', '>q')
    
    # Call to read(...): (line 118)
    # Processing the call arguments (line 118)
    int_119808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 47), 'int')
    # Processing the call keyword arguments (line 118)
    kwargs_119809 = {}
    # Getting the type of 'f' (line 118)
    f_119806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'f', False)
    # Obtaining the member 'read' of a type (line 118)
    read_119807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 40), f_119806, 'read')
    # Calling read(args, kwargs) (line 118)
    read_call_result_119810 = invoke(stypy.reporting.localization.Localization(__file__, 118, 40), read_119807, *[int_119808], **kwargs_119809)
    
    # Processing the call keyword arguments (line 118)
    kwargs_119811 = {}
    # Getting the type of 'struct' (line 118)
    struct_119803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 118)
    unpack_119804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 20), struct_119803, 'unpack')
    # Calling unpack(args, kwargs) (line 118)
    unpack_call_result_119812 = invoke(stypy.reporting.localization.Localization(__file__, 118, 20), unpack_119804, *[str_119805, read_call_result_119810], **kwargs_119811)
    
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___119813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 20), unpack_call_result_119812, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_119814 = invoke(stypy.reporting.localization.Localization(__file__, 118, 20), getitem___119813, int_119802)
    
    # Processing the call keyword arguments (line 118)
    kwargs_119815 = {}
    # Getting the type of 'np' (line 118)
    np_119800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'np', False)
    # Obtaining the member 'int64' of a type (line 118)
    int64_119801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 11), np_119800, 'int64')
    # Calling int64(args, kwargs) (line 118)
    int64_call_result_119816 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), int64_119801, *[subscript_call_result_119814], **kwargs_119815)
    
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', int64_call_result_119816)
    
    # ################# End of '_read_int64(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_int64' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_119817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119817)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_int64'
    return stypy_return_type_119817

# Assigning a type to the variable '_read_int64' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), '_read_int64', _read_int64)

@norecursion
def _read_uint16(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_uint16'
    module_type_store = module_type_store.open_function_context('_read_uint16', 121, 0, False)
    
    # Passed parameters checking function
    _read_uint16.stypy_localization = localization
    _read_uint16.stypy_type_of_self = None
    _read_uint16.stypy_type_store = module_type_store
    _read_uint16.stypy_function_name = '_read_uint16'
    _read_uint16.stypy_param_names_list = ['f']
    _read_uint16.stypy_varargs_param_name = None
    _read_uint16.stypy_kwargs_param_name = None
    _read_uint16.stypy_call_defaults = defaults
    _read_uint16.stypy_call_varargs = varargs
    _read_uint16.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_uint16', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_uint16', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_uint16(...)' code ##################

    str_119818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 4), 'str', 'Read an unsigned 16-bit integer')
    
    # Call to uint16(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining the type of the subscript
    int_119821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 57), 'int')
    
    # Call to unpack(...): (line 123)
    # Processing the call arguments (line 123)
    str_119824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 35), 'str', '>H')
    
    # Obtaining the type of the subscript
    int_119825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 51), 'int')
    int_119826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 53), 'int')
    slice_119827 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 123, 41), int_119825, int_119826, None)
    
    # Call to read(...): (line 123)
    # Processing the call arguments (line 123)
    int_119830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 48), 'int')
    # Processing the call keyword arguments (line 123)
    kwargs_119831 = {}
    # Getting the type of 'f' (line 123)
    f_119828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 41), 'f', False)
    # Obtaining the member 'read' of a type (line 123)
    read_119829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 41), f_119828, 'read')
    # Calling read(args, kwargs) (line 123)
    read_call_result_119832 = invoke(stypy.reporting.localization.Localization(__file__, 123, 41), read_119829, *[int_119830], **kwargs_119831)
    
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___119833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 41), read_call_result_119832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_119834 = invoke(stypy.reporting.localization.Localization(__file__, 123, 41), getitem___119833, slice_119827)
    
    # Processing the call keyword arguments (line 123)
    kwargs_119835 = {}
    # Getting the type of 'struct' (line 123)
    struct_119822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 21), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 123)
    unpack_119823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 21), struct_119822, 'unpack')
    # Calling unpack(args, kwargs) (line 123)
    unpack_call_result_119836 = invoke(stypy.reporting.localization.Localization(__file__, 123, 21), unpack_119823, *[str_119824, subscript_call_result_119834], **kwargs_119835)
    
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___119837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 21), unpack_call_result_119836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_119838 = invoke(stypy.reporting.localization.Localization(__file__, 123, 21), getitem___119837, int_119821)
    
    # Processing the call keyword arguments (line 123)
    kwargs_119839 = {}
    # Getting the type of 'np' (line 123)
    np_119819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'np', False)
    # Obtaining the member 'uint16' of a type (line 123)
    uint16_119820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), np_119819, 'uint16')
    # Calling uint16(args, kwargs) (line 123)
    uint16_call_result_119840 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), uint16_119820, *[subscript_call_result_119838], **kwargs_119839)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', uint16_call_result_119840)
    
    # ################# End of '_read_uint16(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_uint16' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_119841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119841)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_uint16'
    return stypy_return_type_119841

# Assigning a type to the variable '_read_uint16' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), '_read_uint16', _read_uint16)

@norecursion
def _read_uint32(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_uint32'
    module_type_store = module_type_store.open_function_context('_read_uint32', 126, 0, False)
    
    # Passed parameters checking function
    _read_uint32.stypy_localization = localization
    _read_uint32.stypy_type_of_self = None
    _read_uint32.stypy_type_store = module_type_store
    _read_uint32.stypy_function_name = '_read_uint32'
    _read_uint32.stypy_param_names_list = ['f']
    _read_uint32.stypy_varargs_param_name = None
    _read_uint32.stypy_kwargs_param_name = None
    _read_uint32.stypy_call_defaults = defaults
    _read_uint32.stypy_call_varargs = varargs
    _read_uint32.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_uint32', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_uint32', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_uint32(...)' code ##################

    str_119842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'str', 'Read an unsigned 32-bit integer')
    
    # Call to uint32(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Obtaining the type of the subscript
    int_119845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 52), 'int')
    
    # Call to unpack(...): (line 128)
    # Processing the call arguments (line 128)
    str_119848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 35), 'str', '>I')
    
    # Call to read(...): (line 128)
    # Processing the call arguments (line 128)
    int_119851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 48), 'int')
    # Processing the call keyword arguments (line 128)
    kwargs_119852 = {}
    # Getting the type of 'f' (line 128)
    f_119849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'f', False)
    # Obtaining the member 'read' of a type (line 128)
    read_119850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 41), f_119849, 'read')
    # Calling read(args, kwargs) (line 128)
    read_call_result_119853 = invoke(stypy.reporting.localization.Localization(__file__, 128, 41), read_119850, *[int_119851], **kwargs_119852)
    
    # Processing the call keyword arguments (line 128)
    kwargs_119854 = {}
    # Getting the type of 'struct' (line 128)
    struct_119846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 128)
    unpack_119847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 21), struct_119846, 'unpack')
    # Calling unpack(args, kwargs) (line 128)
    unpack_call_result_119855 = invoke(stypy.reporting.localization.Localization(__file__, 128, 21), unpack_119847, *[str_119848, read_call_result_119853], **kwargs_119854)
    
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___119856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 21), unpack_call_result_119855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_119857 = invoke(stypy.reporting.localization.Localization(__file__, 128, 21), getitem___119856, int_119845)
    
    # Processing the call keyword arguments (line 128)
    kwargs_119858 = {}
    # Getting the type of 'np' (line 128)
    np_119843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'np', False)
    # Obtaining the member 'uint32' of a type (line 128)
    uint32_119844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 11), np_119843, 'uint32')
    # Calling uint32(args, kwargs) (line 128)
    uint32_call_result_119859 = invoke(stypy.reporting.localization.Localization(__file__, 128, 11), uint32_119844, *[subscript_call_result_119857], **kwargs_119858)
    
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', uint32_call_result_119859)
    
    # ################# End of '_read_uint32(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_uint32' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_119860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119860)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_uint32'
    return stypy_return_type_119860

# Assigning a type to the variable '_read_uint32' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), '_read_uint32', _read_uint32)

@norecursion
def _read_uint64(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_uint64'
    module_type_store = module_type_store.open_function_context('_read_uint64', 131, 0, False)
    
    # Passed parameters checking function
    _read_uint64.stypy_localization = localization
    _read_uint64.stypy_type_of_self = None
    _read_uint64.stypy_type_store = module_type_store
    _read_uint64.stypy_function_name = '_read_uint64'
    _read_uint64.stypy_param_names_list = ['f']
    _read_uint64.stypy_varargs_param_name = None
    _read_uint64.stypy_kwargs_param_name = None
    _read_uint64.stypy_call_defaults = defaults
    _read_uint64.stypy_call_varargs = varargs
    _read_uint64.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_uint64', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_uint64', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_uint64(...)' code ##################

    str_119861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'str', 'Read an unsigned 64-bit integer')
    
    # Call to uint64(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Obtaining the type of the subscript
    int_119864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 52), 'int')
    
    # Call to unpack(...): (line 133)
    # Processing the call arguments (line 133)
    str_119867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'str', '>Q')
    
    # Call to read(...): (line 133)
    # Processing the call arguments (line 133)
    int_119870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 48), 'int')
    # Processing the call keyword arguments (line 133)
    kwargs_119871 = {}
    # Getting the type of 'f' (line 133)
    f_119868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'f', False)
    # Obtaining the member 'read' of a type (line 133)
    read_119869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 41), f_119868, 'read')
    # Calling read(args, kwargs) (line 133)
    read_call_result_119872 = invoke(stypy.reporting.localization.Localization(__file__, 133, 41), read_119869, *[int_119870], **kwargs_119871)
    
    # Processing the call keyword arguments (line 133)
    kwargs_119873 = {}
    # Getting the type of 'struct' (line 133)
    struct_119865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 133)
    unpack_119866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 21), struct_119865, 'unpack')
    # Calling unpack(args, kwargs) (line 133)
    unpack_call_result_119874 = invoke(stypy.reporting.localization.Localization(__file__, 133, 21), unpack_119866, *[str_119867, read_call_result_119872], **kwargs_119873)
    
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___119875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 21), unpack_call_result_119874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_119876 = invoke(stypy.reporting.localization.Localization(__file__, 133, 21), getitem___119875, int_119864)
    
    # Processing the call keyword arguments (line 133)
    kwargs_119877 = {}
    # Getting the type of 'np' (line 133)
    np_119862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'np', False)
    # Obtaining the member 'uint64' of a type (line 133)
    uint64_119863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 11), np_119862, 'uint64')
    # Calling uint64(args, kwargs) (line 133)
    uint64_call_result_119878 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), uint64_119863, *[subscript_call_result_119876], **kwargs_119877)
    
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', uint64_call_result_119878)
    
    # ################# End of '_read_uint64(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_uint64' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_119879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119879)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_uint64'
    return stypy_return_type_119879

# Assigning a type to the variable '_read_uint64' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), '_read_uint64', _read_uint64)

@norecursion
def _read_float32(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_float32'
    module_type_store = module_type_store.open_function_context('_read_float32', 136, 0, False)
    
    # Passed parameters checking function
    _read_float32.stypy_localization = localization
    _read_float32.stypy_type_of_self = None
    _read_float32.stypy_type_store = module_type_store
    _read_float32.stypy_function_name = '_read_float32'
    _read_float32.stypy_param_names_list = ['f']
    _read_float32.stypy_varargs_param_name = None
    _read_float32.stypy_kwargs_param_name = None
    _read_float32.stypy_call_defaults = defaults
    _read_float32.stypy_call_varargs = varargs
    _read_float32.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_float32', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_float32', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_float32(...)' code ##################

    str_119880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 4), 'str', 'Read a 32-bit float')
    
    # Call to float32(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining the type of the subscript
    int_119883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 53), 'int')
    
    # Call to unpack(...): (line 138)
    # Processing the call arguments (line 138)
    str_119886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'str', '>f')
    
    # Call to read(...): (line 138)
    # Processing the call arguments (line 138)
    int_119889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 49), 'int')
    # Processing the call keyword arguments (line 138)
    kwargs_119890 = {}
    # Getting the type of 'f' (line 138)
    f_119887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'f', False)
    # Obtaining the member 'read' of a type (line 138)
    read_119888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), f_119887, 'read')
    # Calling read(args, kwargs) (line 138)
    read_call_result_119891 = invoke(stypy.reporting.localization.Localization(__file__, 138, 42), read_119888, *[int_119889], **kwargs_119890)
    
    # Processing the call keyword arguments (line 138)
    kwargs_119892 = {}
    # Getting the type of 'struct' (line 138)
    struct_119884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 138)
    unpack_119885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 22), struct_119884, 'unpack')
    # Calling unpack(args, kwargs) (line 138)
    unpack_call_result_119893 = invoke(stypy.reporting.localization.Localization(__file__, 138, 22), unpack_119885, *[str_119886, read_call_result_119891], **kwargs_119892)
    
    # Obtaining the member '__getitem__' of a type (line 138)
    getitem___119894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 22), unpack_call_result_119893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 138)
    subscript_call_result_119895 = invoke(stypy.reporting.localization.Localization(__file__, 138, 22), getitem___119894, int_119883)
    
    # Processing the call keyword arguments (line 138)
    kwargs_119896 = {}
    # Getting the type of 'np' (line 138)
    np_119881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'np', False)
    # Obtaining the member 'float32' of a type (line 138)
    float32_119882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), np_119881, 'float32')
    # Calling float32(args, kwargs) (line 138)
    float32_call_result_119897 = invoke(stypy.reporting.localization.Localization(__file__, 138, 11), float32_119882, *[subscript_call_result_119895], **kwargs_119896)
    
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', float32_call_result_119897)
    
    # ################# End of '_read_float32(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_float32' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_119898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119898)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_float32'
    return stypy_return_type_119898

# Assigning a type to the variable '_read_float32' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), '_read_float32', _read_float32)

@norecursion
def _read_float64(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_float64'
    module_type_store = module_type_store.open_function_context('_read_float64', 141, 0, False)
    
    # Passed parameters checking function
    _read_float64.stypy_localization = localization
    _read_float64.stypy_type_of_self = None
    _read_float64.stypy_type_store = module_type_store
    _read_float64.stypy_function_name = '_read_float64'
    _read_float64.stypy_param_names_list = ['f']
    _read_float64.stypy_varargs_param_name = None
    _read_float64.stypy_kwargs_param_name = None
    _read_float64.stypy_call_defaults = defaults
    _read_float64.stypy_call_varargs = varargs
    _read_float64.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_float64', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_float64', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_float64(...)' code ##################

    str_119899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'str', 'Read a 64-bit float')
    
    # Call to float64(...): (line 143)
    # Processing the call arguments (line 143)
    
    # Obtaining the type of the subscript
    int_119902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 53), 'int')
    
    # Call to unpack(...): (line 143)
    # Processing the call arguments (line 143)
    str_119905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 36), 'str', '>d')
    
    # Call to read(...): (line 143)
    # Processing the call arguments (line 143)
    int_119908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 49), 'int')
    # Processing the call keyword arguments (line 143)
    kwargs_119909 = {}
    # Getting the type of 'f' (line 143)
    f_119906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'f', False)
    # Obtaining the member 'read' of a type (line 143)
    read_119907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 42), f_119906, 'read')
    # Calling read(args, kwargs) (line 143)
    read_call_result_119910 = invoke(stypy.reporting.localization.Localization(__file__, 143, 42), read_119907, *[int_119908], **kwargs_119909)
    
    # Processing the call keyword arguments (line 143)
    kwargs_119911 = {}
    # Getting the type of 'struct' (line 143)
    struct_119903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 143)
    unpack_119904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 22), struct_119903, 'unpack')
    # Calling unpack(args, kwargs) (line 143)
    unpack_call_result_119912 = invoke(stypy.reporting.localization.Localization(__file__, 143, 22), unpack_119904, *[str_119905, read_call_result_119910], **kwargs_119911)
    
    # Obtaining the member '__getitem__' of a type (line 143)
    getitem___119913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 22), unpack_call_result_119912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 143)
    subscript_call_result_119914 = invoke(stypy.reporting.localization.Localization(__file__, 143, 22), getitem___119913, int_119902)
    
    # Processing the call keyword arguments (line 143)
    kwargs_119915 = {}
    # Getting the type of 'np' (line 143)
    np_119900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'np', False)
    # Obtaining the member 'float64' of a type (line 143)
    float64_119901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 11), np_119900, 'float64')
    # Calling float64(args, kwargs) (line 143)
    float64_call_result_119916 = invoke(stypy.reporting.localization.Localization(__file__, 143, 11), float64_119901, *[subscript_call_result_119914], **kwargs_119915)
    
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', float64_call_result_119916)
    
    # ################# End of '_read_float64(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_float64' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_119917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119917)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_float64'
    return stypy_return_type_119917

# Assigning a type to the variable '_read_float64' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), '_read_float64', _read_float64)
# Declaration of the 'Pointer' class

class Pointer(object, ):
    str_119918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 4), 'str', 'Class used to define pointers')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Pointer.__init__', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 150):
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'index' (line 150)
        index_119919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'index')
        # Getting the type of 'self' (line 150)
        self_119920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'index' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_119920, 'index', index_119919)
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Pointer' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'Pointer', Pointer)
# Declaration of the 'ObjectPointer' class
# Getting the type of 'Pointer' (line 154)
Pointer_119921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'Pointer')

class ObjectPointer(Pointer_119921, ):
    str_119922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'str', 'Class used to define object pointers')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 154, 0, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ObjectPointer.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ObjectPointer' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'ObjectPointer', ObjectPointer)

@norecursion
def _read_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_string'
    module_type_store = module_type_store.open_function_context('_read_string', 159, 0, False)
    
    # Passed parameters checking function
    _read_string.stypy_localization = localization
    _read_string.stypy_type_of_self = None
    _read_string.stypy_type_store = module_type_store
    _read_string.stypy_function_name = '_read_string'
    _read_string.stypy_param_names_list = ['f']
    _read_string.stypy_varargs_param_name = None
    _read_string.stypy_kwargs_param_name = None
    _read_string.stypy_call_defaults = defaults
    _read_string.stypy_call_varargs = varargs
    _read_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_string', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_string', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_string(...)' code ##################

    str_119923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 4), 'str', 'Read a string')
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to _read_long(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'f' (line 161)
    f_119925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'f', False)
    # Processing the call keyword arguments (line 161)
    kwargs_119926 = {}
    # Getting the type of '_read_long' (line 161)
    _read_long_119924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 161)
    _read_long_call_result_119927 = invoke(stypy.reporting.localization.Localization(__file__, 161, 13), _read_long_119924, *[f_119925], **kwargs_119926)
    
    # Assigning a type to the variable 'length' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'length', _read_long_call_result_119927)
    
    
    # Getting the type of 'length' (line 162)
    length_119928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'length')
    int_119929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 16), 'int')
    # Applying the binary operator '>' (line 162)
    result_gt_119930 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 7), '>', length_119928, int_119929)
    
    # Testing the type of an if condition (line 162)
    if_condition_119931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), result_gt_119930)
    # Assigning a type to the variable 'if_condition_119931' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_119931', if_condition_119931)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to _read_bytes(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'f' (line 163)
    f_119933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'f', False)
    # Getting the type of 'length' (line 163)
    length_119934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 31), 'length', False)
    # Processing the call keyword arguments (line 163)
    kwargs_119935 = {}
    # Getting the type of '_read_bytes' (line 163)
    _read_bytes_119932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 163)
    _read_bytes_call_result_119936 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), _read_bytes_119932, *[f_119933, length_119934], **kwargs_119935)
    
    # Assigning a type to the variable 'chars' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'chars', _read_bytes_call_result_119936)
    
    # Call to _align_32(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'f' (line 164)
    f_119938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'f', False)
    # Processing the call keyword arguments (line 164)
    kwargs_119939 = {}
    # Getting the type of '_align_32' (line 164)
    _align_32_119937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), '_align_32', False)
    # Calling _align_32(args, kwargs) (line 164)
    _align_32_call_result_119940 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), _align_32_119937, *[f_119938], **kwargs_119939)
    
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Name (line 165):
    
    # Call to asstr(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'chars' (line 165)
    chars_119942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'chars', False)
    # Processing the call keyword arguments (line 165)
    kwargs_119943 = {}
    # Getting the type of 'asstr' (line 165)
    asstr_119941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'asstr', False)
    # Calling asstr(args, kwargs) (line 165)
    asstr_call_result_119944 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), asstr_119941, *[chars_119942], **kwargs_119943)
    
    # Assigning a type to the variable 'chars' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'chars', asstr_call_result_119944)
    # SSA branch for the else part of an if statement (line 162)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 167):
    
    # Assigning a Str to a Name (line 167):
    str_119945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 16), 'str', '')
    # Assigning a type to the variable 'chars' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'chars', str_119945)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'chars' (line 168)
    chars_119946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'chars')
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type', chars_119946)
    
    # ################# End of '_read_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_string' in the type store
    # Getting the type of 'stypy_return_type' (line 159)
    stypy_return_type_119947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119947)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_string'
    return stypy_return_type_119947

# Assigning a type to the variable '_read_string' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), '_read_string', _read_string)

@norecursion
def _read_string_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_string_data'
    module_type_store = module_type_store.open_function_context('_read_string_data', 171, 0, False)
    
    # Passed parameters checking function
    _read_string_data.stypy_localization = localization
    _read_string_data.stypy_type_of_self = None
    _read_string_data.stypy_type_store = module_type_store
    _read_string_data.stypy_function_name = '_read_string_data'
    _read_string_data.stypy_param_names_list = ['f']
    _read_string_data.stypy_varargs_param_name = None
    _read_string_data.stypy_kwargs_param_name = None
    _read_string_data.stypy_call_defaults = defaults
    _read_string_data.stypy_call_varargs = varargs
    _read_string_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_string_data', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_string_data', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_string_data(...)' code ##################

    str_119948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 4), 'str', 'Read a data string (length is specified twice)')
    
    # Assigning a Call to a Name (line 173):
    
    # Assigning a Call to a Name (line 173):
    
    # Call to _read_long(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'f' (line 173)
    f_119950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'f', False)
    # Processing the call keyword arguments (line 173)
    kwargs_119951 = {}
    # Getting the type of '_read_long' (line 173)
    _read_long_119949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 173)
    _read_long_call_result_119952 = invoke(stypy.reporting.localization.Localization(__file__, 173, 13), _read_long_119949, *[f_119950], **kwargs_119951)
    
    # Assigning a type to the variable 'length' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'length', _read_long_call_result_119952)
    
    
    # Getting the type of 'length' (line 174)
    length_119953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'length')
    int_119954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 16), 'int')
    # Applying the binary operator '>' (line 174)
    result_gt_119955 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), '>', length_119953, int_119954)
    
    # Testing the type of an if condition (line 174)
    if_condition_119956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_gt_119955)
    # Assigning a type to the variable 'if_condition_119956' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_119956', if_condition_119956)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to _read_long(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'f' (line 175)
    f_119958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'f', False)
    # Processing the call keyword arguments (line 175)
    kwargs_119959 = {}
    # Getting the type of '_read_long' (line 175)
    _read_long_119957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 175)
    _read_long_call_result_119960 = invoke(stypy.reporting.localization.Localization(__file__, 175, 17), _read_long_119957, *[f_119958], **kwargs_119959)
    
    # Assigning a type to the variable 'length' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'length', _read_long_call_result_119960)
    
    # Assigning a Call to a Name (line 176):
    
    # Assigning a Call to a Name (line 176):
    
    # Call to _read_bytes(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'f' (line 176)
    f_119962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 34), 'f', False)
    # Getting the type of 'length' (line 176)
    length_119963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 37), 'length', False)
    # Processing the call keyword arguments (line 176)
    kwargs_119964 = {}
    # Getting the type of '_read_bytes' (line 176)
    _read_bytes_119961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 22), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 176)
    _read_bytes_call_result_119965 = invoke(stypy.reporting.localization.Localization(__file__, 176, 22), _read_bytes_119961, *[f_119962, length_119963], **kwargs_119964)
    
    # Assigning a type to the variable 'string_data' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'string_data', _read_bytes_call_result_119965)
    
    # Call to _align_32(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'f' (line 177)
    f_119967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'f', False)
    # Processing the call keyword arguments (line 177)
    kwargs_119968 = {}
    # Getting the type of '_align_32' (line 177)
    _align_32_119966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), '_align_32', False)
    # Calling _align_32(args, kwargs) (line 177)
    _align_32_call_result_119969 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), _align_32_119966, *[f_119967], **kwargs_119968)
    
    # SSA branch for the else part of an if statement (line 174)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 179):
    
    # Assigning a Str to a Name (line 179):
    str_119970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'str', '')
    # Assigning a type to the variable 'string_data' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'string_data', str_119970)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'string_data' (line 180)
    string_data_119971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'string_data')
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type', string_data_119971)
    
    # ################# End of '_read_string_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_string_data' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_119972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119972)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_string_data'
    return stypy_return_type_119972

# Assigning a type to the variable '_read_string_data' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), '_read_string_data', _read_string_data)

@norecursion
def _read_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_data'
    module_type_store = module_type_store.open_function_context('_read_data', 183, 0, False)
    
    # Passed parameters checking function
    _read_data.stypy_localization = localization
    _read_data.stypy_type_of_self = None
    _read_data.stypy_type_store = module_type_store
    _read_data.stypy_function_name = '_read_data'
    _read_data.stypy_param_names_list = ['f', 'dtype']
    _read_data.stypy_varargs_param_name = None
    _read_data.stypy_kwargs_param_name = None
    _read_data.stypy_call_defaults = defaults
    _read_data.stypy_call_varargs = varargs
    _read_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_data', ['f', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_data', localization, ['f', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_data(...)' code ##################

    str_119973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'str', 'Read a variable with a specified data type')
    
    
    # Getting the type of 'dtype' (line 185)
    dtype_119974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'dtype')
    int_119975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 16), 'int')
    # Applying the binary operator '==' (line 185)
    result_eq_119976 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 7), '==', dtype_119974, int_119975)
    
    # Testing the type of an if condition (line 185)
    if_condition_119977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 4), result_eq_119976)
    # Assigning a type to the variable 'if_condition_119977' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'if_condition_119977', if_condition_119977)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to _read_int32(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'f' (line 186)
    f_119979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 'f', False)
    # Processing the call keyword arguments (line 186)
    kwargs_119980 = {}
    # Getting the type of '_read_int32' (line 186)
    _read_int32_119978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), '_read_int32', False)
    # Calling _read_int32(args, kwargs) (line 186)
    _read_int32_call_result_119981 = invoke(stypy.reporting.localization.Localization(__file__, 186, 11), _read_int32_119978, *[f_119979], **kwargs_119980)
    
    int_119982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 29), 'int')
    # Applying the binary operator '!=' (line 186)
    result_ne_119983 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 11), '!=', _read_int32_call_result_119981, int_119982)
    
    # Testing the type of an if condition (line 186)
    if_condition_119984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 8), result_ne_119983)
    # Assigning a type to the variable 'if_condition_119984' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'if_condition_119984', if_condition_119984)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 187)
    # Processing the call arguments (line 187)
    str_119986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 28), 'str', 'Error occurred while reading byte variable')
    # Processing the call keyword arguments (line 187)
    kwargs_119987 = {}
    # Getting the type of 'Exception' (line 187)
    Exception_119985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'Exception', False)
    # Calling Exception(args, kwargs) (line 187)
    Exception_call_result_119988 = invoke(stypy.reporting.localization.Localization(__file__, 187, 18), Exception_119985, *[str_119986], **kwargs_119987)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 12), Exception_call_result_119988, 'raise parameter', BaseException)
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _read_byte(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'f' (line 188)
    f_119990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 26), 'f', False)
    # Processing the call keyword arguments (line 188)
    kwargs_119991 = {}
    # Getting the type of '_read_byte' (line 188)
    _read_byte_119989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), '_read_byte', False)
    # Calling _read_byte(args, kwargs) (line 188)
    _read_byte_call_result_119992 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), _read_byte_119989, *[f_119990], **kwargs_119991)
    
    # Assigning a type to the variable 'stypy_return_type' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', _read_byte_call_result_119992)
    # SSA branch for the else part of an if statement (line 185)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 189)
    dtype_119993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 9), 'dtype')
    int_119994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 18), 'int')
    # Applying the binary operator '==' (line 189)
    result_eq_119995 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 9), '==', dtype_119993, int_119994)
    
    # Testing the type of an if condition (line 189)
    if_condition_119996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 9), result_eq_119995)
    # Assigning a type to the variable 'if_condition_119996' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 9), 'if_condition_119996', if_condition_119996)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_int16(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'f' (line 190)
    f_119998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'f', False)
    # Processing the call keyword arguments (line 190)
    kwargs_119999 = {}
    # Getting the type of '_read_int16' (line 190)
    _read_int16_119997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), '_read_int16', False)
    # Calling _read_int16(args, kwargs) (line 190)
    _read_int16_call_result_120000 = invoke(stypy.reporting.localization.Localization(__file__, 190, 15), _read_int16_119997, *[f_119998], **kwargs_119999)
    
    # Assigning a type to the variable 'stypy_return_type' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'stypy_return_type', _read_int16_call_result_120000)
    # SSA branch for the else part of an if statement (line 189)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 191)
    dtype_120001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 9), 'dtype')
    int_120002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 18), 'int')
    # Applying the binary operator '==' (line 191)
    result_eq_120003 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 9), '==', dtype_120001, int_120002)
    
    # Testing the type of an if condition (line 191)
    if_condition_120004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 9), result_eq_120003)
    # Assigning a type to the variable 'if_condition_120004' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 9), 'if_condition_120004', if_condition_120004)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_int32(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'f' (line 192)
    f_120006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'f', False)
    # Processing the call keyword arguments (line 192)
    kwargs_120007 = {}
    # Getting the type of '_read_int32' (line 192)
    _read_int32_120005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), '_read_int32', False)
    # Calling _read_int32(args, kwargs) (line 192)
    _read_int32_call_result_120008 = invoke(stypy.reporting.localization.Localization(__file__, 192, 15), _read_int32_120005, *[f_120006], **kwargs_120007)
    
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'stypy_return_type', _read_int32_call_result_120008)
    # SSA branch for the else part of an if statement (line 191)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 193)
    dtype_120009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 9), 'dtype')
    int_120010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 18), 'int')
    # Applying the binary operator '==' (line 193)
    result_eq_120011 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 9), '==', dtype_120009, int_120010)
    
    # Testing the type of an if condition (line 193)
    if_condition_120012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 9), result_eq_120011)
    # Assigning a type to the variable 'if_condition_120012' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 9), 'if_condition_120012', if_condition_120012)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_float32(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'f' (line 194)
    f_120014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 29), 'f', False)
    # Processing the call keyword arguments (line 194)
    kwargs_120015 = {}
    # Getting the type of '_read_float32' (line 194)
    _read_float32_120013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), '_read_float32', False)
    # Calling _read_float32(args, kwargs) (line 194)
    _read_float32_call_result_120016 = invoke(stypy.reporting.localization.Localization(__file__, 194, 15), _read_float32_120013, *[f_120014], **kwargs_120015)
    
    # Assigning a type to the variable 'stypy_return_type' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', _read_float32_call_result_120016)
    # SSA branch for the else part of an if statement (line 193)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 195)
    dtype_120017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 9), 'dtype')
    int_120018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'int')
    # Applying the binary operator '==' (line 195)
    result_eq_120019 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 9), '==', dtype_120017, int_120018)
    
    # Testing the type of an if condition (line 195)
    if_condition_120020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 9), result_eq_120019)
    # Assigning a type to the variable 'if_condition_120020' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 9), 'if_condition_120020', if_condition_120020)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_float64(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'f' (line 196)
    f_120022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 29), 'f', False)
    # Processing the call keyword arguments (line 196)
    kwargs_120023 = {}
    # Getting the type of '_read_float64' (line 196)
    _read_float64_120021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), '_read_float64', False)
    # Calling _read_float64(args, kwargs) (line 196)
    _read_float64_call_result_120024 = invoke(stypy.reporting.localization.Localization(__file__, 196, 15), _read_float64_120021, *[f_120022], **kwargs_120023)
    
    # Assigning a type to the variable 'stypy_return_type' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stypy_return_type', _read_float64_call_result_120024)
    # SSA branch for the else part of an if statement (line 195)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 197)
    dtype_120025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'dtype')
    int_120026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'int')
    # Applying the binary operator '==' (line 197)
    result_eq_120027 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 9), '==', dtype_120025, int_120026)
    
    # Testing the type of an if condition (line 197)
    if_condition_120028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 9), result_eq_120027)
    # Assigning a type to the variable 'if_condition_120028' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'if_condition_120028', if_condition_120028)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to _read_float32(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'f' (line 198)
    f_120030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), 'f', False)
    # Processing the call keyword arguments (line 198)
    kwargs_120031 = {}
    # Getting the type of '_read_float32' (line 198)
    _read_float32_120029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), '_read_float32', False)
    # Calling _read_float32(args, kwargs) (line 198)
    _read_float32_call_result_120032 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), _read_float32_120029, *[f_120030], **kwargs_120031)
    
    # Assigning a type to the variable 'real' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'real', _read_float32_call_result_120032)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to _read_float32(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'f' (line 199)
    f_120034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'f', False)
    # Processing the call keyword arguments (line 199)
    kwargs_120035 = {}
    # Getting the type of '_read_float32' (line 199)
    _read_float32_120033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), '_read_float32', False)
    # Calling _read_float32(args, kwargs) (line 199)
    _read_float32_call_result_120036 = invoke(stypy.reporting.localization.Localization(__file__, 199, 15), _read_float32_120033, *[f_120034], **kwargs_120035)
    
    # Assigning a type to the variable 'imag' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'imag', _read_float32_call_result_120036)
    
    # Call to complex64(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'real' (line 200)
    real_120039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'real', False)
    # Getting the type of 'imag' (line 200)
    imag_120040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'imag', False)
    complex_120041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 42), 'complex')
    # Applying the binary operator '*' (line 200)
    result_mul_120042 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 35), '*', imag_120040, complex_120041)
    
    # Applying the binary operator '+' (line 200)
    result_add_120043 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 28), '+', real_120039, result_mul_120042)
    
    # Processing the call keyword arguments (line 200)
    kwargs_120044 = {}
    # Getting the type of 'np' (line 200)
    np_120037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'np', False)
    # Obtaining the member 'complex64' of a type (line 200)
    complex64_120038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), np_120037, 'complex64')
    # Calling complex64(args, kwargs) (line 200)
    complex64_call_result_120045 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), complex64_120038, *[result_add_120043], **kwargs_120044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'stypy_return_type', complex64_call_result_120045)
    # SSA branch for the else part of an if statement (line 197)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 201)
    dtype_120046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 9), 'dtype')
    int_120047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 18), 'int')
    # Applying the binary operator '==' (line 201)
    result_eq_120048 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 9), '==', dtype_120046, int_120047)
    
    # Testing the type of an if condition (line 201)
    if_condition_120049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 9), result_eq_120048)
    # Assigning a type to the variable 'if_condition_120049' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 9), 'if_condition_120049', if_condition_120049)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_string_data(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'f' (line 202)
    f_120051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 33), 'f', False)
    # Processing the call keyword arguments (line 202)
    kwargs_120052 = {}
    # Getting the type of '_read_string_data' (line 202)
    _read_string_data_120050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), '_read_string_data', False)
    # Calling _read_string_data(args, kwargs) (line 202)
    _read_string_data_call_result_120053 = invoke(stypy.reporting.localization.Localization(__file__, 202, 15), _read_string_data_120050, *[f_120051], **kwargs_120052)
    
    # Assigning a type to the variable 'stypy_return_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stypy_return_type', _read_string_data_call_result_120053)
    # SSA branch for the else part of an if statement (line 201)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 203)
    dtype_120054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 9), 'dtype')
    int_120055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 18), 'int')
    # Applying the binary operator '==' (line 203)
    result_eq_120056 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 9), '==', dtype_120054, int_120055)
    
    # Testing the type of an if condition (line 203)
    if_condition_120057 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 9), result_eq_120056)
    # Assigning a type to the variable 'if_condition_120057' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 9), 'if_condition_120057', if_condition_120057)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 204)
    # Processing the call arguments (line 204)
    str_120059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 24), 'str', 'Should not be here - please report this')
    # Processing the call keyword arguments (line 204)
    kwargs_120060 = {}
    # Getting the type of 'Exception' (line 204)
    Exception_120058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 204)
    Exception_call_result_120061 = invoke(stypy.reporting.localization.Localization(__file__, 204, 14), Exception_120058, *[str_120059], **kwargs_120060)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 204, 8), Exception_call_result_120061, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 203)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 205)
    dtype_120062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 9), 'dtype')
    int_120063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 18), 'int')
    # Applying the binary operator '==' (line 205)
    result_eq_120064 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 9), '==', dtype_120062, int_120063)
    
    # Testing the type of an if condition (line 205)
    if_condition_120065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 9), result_eq_120064)
    # Assigning a type to the variable 'if_condition_120065' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 9), 'if_condition_120065', if_condition_120065)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to _read_float64(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'f' (line 206)
    f_120067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'f', False)
    # Processing the call keyword arguments (line 206)
    kwargs_120068 = {}
    # Getting the type of '_read_float64' (line 206)
    _read_float64_120066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), '_read_float64', False)
    # Calling _read_float64(args, kwargs) (line 206)
    _read_float64_call_result_120069 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), _read_float64_120066, *[f_120067], **kwargs_120068)
    
    # Assigning a type to the variable 'real' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'real', _read_float64_call_result_120069)
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to _read_float64(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'f' (line 207)
    f_120071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'f', False)
    # Processing the call keyword arguments (line 207)
    kwargs_120072 = {}
    # Getting the type of '_read_float64' (line 207)
    _read_float64_120070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), '_read_float64', False)
    # Calling _read_float64(args, kwargs) (line 207)
    _read_float64_call_result_120073 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), _read_float64_120070, *[f_120071], **kwargs_120072)
    
    # Assigning a type to the variable 'imag' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'imag', _read_float64_call_result_120073)
    
    # Call to complex128(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'real' (line 208)
    real_120076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'real', False)
    # Getting the type of 'imag' (line 208)
    imag_120077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 'imag', False)
    complex_120078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 43), 'complex')
    # Applying the binary operator '*' (line 208)
    result_mul_120079 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 36), '*', imag_120077, complex_120078)
    
    # Applying the binary operator '+' (line 208)
    result_add_120080 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 29), '+', real_120076, result_mul_120079)
    
    # Processing the call keyword arguments (line 208)
    kwargs_120081 = {}
    # Getting the type of 'np' (line 208)
    np_120074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'np', False)
    # Obtaining the member 'complex128' of a type (line 208)
    complex128_120075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), np_120074, 'complex128')
    # Calling complex128(args, kwargs) (line 208)
    complex128_call_result_120082 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), complex128_120075, *[result_add_120080], **kwargs_120081)
    
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', complex128_call_result_120082)
    # SSA branch for the else part of an if statement (line 205)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 209)
    dtype_120083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'dtype')
    int_120084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'int')
    # Applying the binary operator '==' (line 209)
    result_eq_120085 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 9), '==', dtype_120083, int_120084)
    
    # Testing the type of an if condition (line 209)
    if_condition_120086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 9), result_eq_120085)
    # Assigning a type to the variable 'if_condition_120086' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'if_condition_120086', if_condition_120086)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Pointer(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Call to _read_int32(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'f' (line 210)
    f_120089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 35), 'f', False)
    # Processing the call keyword arguments (line 210)
    kwargs_120090 = {}
    # Getting the type of '_read_int32' (line 210)
    _read_int32_120088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), '_read_int32', False)
    # Calling _read_int32(args, kwargs) (line 210)
    _read_int32_call_result_120091 = invoke(stypy.reporting.localization.Localization(__file__, 210, 23), _read_int32_120088, *[f_120089], **kwargs_120090)
    
    # Processing the call keyword arguments (line 210)
    kwargs_120092 = {}
    # Getting the type of 'Pointer' (line 210)
    Pointer_120087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'Pointer', False)
    # Calling Pointer(args, kwargs) (line 210)
    Pointer_call_result_120093 = invoke(stypy.reporting.localization.Localization(__file__, 210, 15), Pointer_120087, *[_read_int32_call_result_120091], **kwargs_120092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'stypy_return_type', Pointer_call_result_120093)
    # SSA branch for the else part of an if statement (line 209)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 211)
    dtype_120094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 9), 'dtype')
    int_120095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 18), 'int')
    # Applying the binary operator '==' (line 211)
    result_eq_120096 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 9), '==', dtype_120094, int_120095)
    
    # Testing the type of an if condition (line 211)
    if_condition_120097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 9), result_eq_120096)
    # Assigning a type to the variable 'if_condition_120097' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 9), 'if_condition_120097', if_condition_120097)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ObjectPointer(...): (line 212)
    # Processing the call arguments (line 212)
    
    # Call to _read_int32(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'f' (line 212)
    f_120100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'f', False)
    # Processing the call keyword arguments (line 212)
    kwargs_120101 = {}
    # Getting the type of '_read_int32' (line 212)
    _read_int32_120099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), '_read_int32', False)
    # Calling _read_int32(args, kwargs) (line 212)
    _read_int32_call_result_120102 = invoke(stypy.reporting.localization.Localization(__file__, 212, 29), _read_int32_120099, *[f_120100], **kwargs_120101)
    
    # Processing the call keyword arguments (line 212)
    kwargs_120103 = {}
    # Getting the type of 'ObjectPointer' (line 212)
    ObjectPointer_120098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'ObjectPointer', False)
    # Calling ObjectPointer(args, kwargs) (line 212)
    ObjectPointer_call_result_120104 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), ObjectPointer_120098, *[_read_int32_call_result_120102], **kwargs_120103)
    
    # Assigning a type to the variable 'stypy_return_type' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'stypy_return_type', ObjectPointer_call_result_120104)
    # SSA branch for the else part of an if statement (line 211)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 213)
    dtype_120105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 9), 'dtype')
    int_120106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 18), 'int')
    # Applying the binary operator '==' (line 213)
    result_eq_120107 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 9), '==', dtype_120105, int_120106)
    
    # Testing the type of an if condition (line 213)
    if_condition_120108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 9), result_eq_120107)
    # Assigning a type to the variable 'if_condition_120108' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 9), 'if_condition_120108', if_condition_120108)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_uint16(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'f' (line 214)
    f_120110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'f', False)
    # Processing the call keyword arguments (line 214)
    kwargs_120111 = {}
    # Getting the type of '_read_uint16' (line 214)
    _read_uint16_120109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), '_read_uint16', False)
    # Calling _read_uint16(args, kwargs) (line 214)
    _read_uint16_call_result_120112 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), _read_uint16_120109, *[f_120110], **kwargs_120111)
    
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', _read_uint16_call_result_120112)
    # SSA branch for the else part of an if statement (line 213)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 215)
    dtype_120113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 9), 'dtype')
    int_120114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 18), 'int')
    # Applying the binary operator '==' (line 215)
    result_eq_120115 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 9), '==', dtype_120113, int_120114)
    
    # Testing the type of an if condition (line 215)
    if_condition_120116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 9), result_eq_120115)
    # Assigning a type to the variable 'if_condition_120116' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 9), 'if_condition_120116', if_condition_120116)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_uint32(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'f' (line 216)
    f_120118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'f', False)
    # Processing the call keyword arguments (line 216)
    kwargs_120119 = {}
    # Getting the type of '_read_uint32' (line 216)
    _read_uint32_120117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), '_read_uint32', False)
    # Calling _read_uint32(args, kwargs) (line 216)
    _read_uint32_call_result_120120 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), _read_uint32_120117, *[f_120118], **kwargs_120119)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', _read_uint32_call_result_120120)
    # SSA branch for the else part of an if statement (line 215)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 217)
    dtype_120121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 9), 'dtype')
    int_120122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 18), 'int')
    # Applying the binary operator '==' (line 217)
    result_eq_120123 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 9), '==', dtype_120121, int_120122)
    
    # Testing the type of an if condition (line 217)
    if_condition_120124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 9), result_eq_120123)
    # Assigning a type to the variable 'if_condition_120124' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 9), 'if_condition_120124', if_condition_120124)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_int64(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'f' (line 218)
    f_120126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'f', False)
    # Processing the call keyword arguments (line 218)
    kwargs_120127 = {}
    # Getting the type of '_read_int64' (line 218)
    _read_int64_120125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), '_read_int64', False)
    # Calling _read_int64(args, kwargs) (line 218)
    _read_int64_call_result_120128 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), _read_int64_120125, *[f_120126], **kwargs_120127)
    
    # Assigning a type to the variable 'stypy_return_type' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'stypy_return_type', _read_int64_call_result_120128)
    # SSA branch for the else part of an if statement (line 217)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 219)
    dtype_120129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'dtype')
    int_120130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'int')
    # Applying the binary operator '==' (line 219)
    result_eq_120131 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 9), '==', dtype_120129, int_120130)
    
    # Testing the type of an if condition (line 219)
    if_condition_120132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 9), result_eq_120131)
    # Assigning a type to the variable 'if_condition_120132' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 9), 'if_condition_120132', if_condition_120132)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _read_uint64(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'f' (line 220)
    f_120134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'f', False)
    # Processing the call keyword arguments (line 220)
    kwargs_120135 = {}
    # Getting the type of '_read_uint64' (line 220)
    _read_uint64_120133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), '_read_uint64', False)
    # Calling _read_uint64(args, kwargs) (line 220)
    _read_uint64_call_result_120136 = invoke(stypy.reporting.localization.Localization(__file__, 220, 15), _read_uint64_120133, *[f_120134], **kwargs_120135)
    
    # Assigning a type to the variable 'stypy_return_type' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stypy_return_type', _read_uint64_call_result_120136)
    # SSA branch for the else part of an if statement (line 219)
    module_type_store.open_ssa_branch('else')
    
    # Call to Exception(...): (line 222)
    # Processing the call arguments (line 222)
    str_120138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 24), 'str', 'Unknown IDL type: %i - please report this')
    # Getting the type of 'dtype' (line 222)
    dtype_120139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 70), 'dtype', False)
    # Applying the binary operator '%' (line 222)
    result_mod_120140 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 24), '%', str_120138, dtype_120139)
    
    # Processing the call keyword arguments (line 222)
    kwargs_120141 = {}
    # Getting the type of 'Exception' (line 222)
    Exception_120137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 222)
    Exception_call_result_120142 = invoke(stypy.reporting.localization.Localization(__file__, 222, 14), Exception_120137, *[result_mod_120140], **kwargs_120141)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 222, 8), Exception_call_result_120142, 'raise parameter', BaseException)
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_read_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_data' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_120143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120143)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_data'
    return stypy_return_type_120143

# Assigning a type to the variable '_read_data' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), '_read_data', _read_data)

@norecursion
def _read_structure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_structure'
    module_type_store = module_type_store.open_function_context('_read_structure', 225, 0, False)
    
    # Passed parameters checking function
    _read_structure.stypy_localization = localization
    _read_structure.stypy_type_of_self = None
    _read_structure.stypy_type_store = module_type_store
    _read_structure.stypy_function_name = '_read_structure'
    _read_structure.stypy_param_names_list = ['f', 'array_desc', 'struct_desc']
    _read_structure.stypy_varargs_param_name = None
    _read_structure.stypy_kwargs_param_name = None
    _read_structure.stypy_call_defaults = defaults
    _read_structure.stypy_call_varargs = varargs
    _read_structure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_structure', ['f', 'array_desc', 'struct_desc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_structure', localization, ['f', 'array_desc', 'struct_desc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_structure(...)' code ##################

    str_120144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, (-1)), 'str', '\n    Read a structure, with the array and structure descriptors given as\n    `array_desc` and `structure_desc` respectively.\n    ')
    
    # Assigning a Subscript to a Name (line 231):
    
    # Assigning a Subscript to a Name (line 231):
    
    # Obtaining the type of the subscript
    str_120145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 23), 'str', 'nelements')
    # Getting the type of 'array_desc' (line 231)
    array_desc_120146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'array_desc')
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___120147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), array_desc_120146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_120148 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___120147, str_120145)
    
    # Assigning a type to the variable 'nrows' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'nrows', subscript_call_result_120148)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    str_120149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 26), 'str', 'tagtable')
    # Getting the type of 'struct_desc' (line 232)
    struct_desc_120150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 14), 'struct_desc')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___120151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 14), struct_desc_120150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_120152 = invoke(stypy.reporting.localization.Localization(__file__, 232, 14), getitem___120151, str_120149)
    
    # Assigning a type to the variable 'columns' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'columns', subscript_call_result_120152)
    
    # Assigning a List to a Name (line 234):
    
    # Assigning a List to a Name (line 234):
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_120153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    
    # Assigning a type to the variable 'dtype' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'dtype', list_120153)
    
    # Getting the type of 'columns' (line 235)
    columns_120154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'columns')
    # Testing the type of a for loop iterable (line 235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 4), columns_120154)
    # Getting the type of the for loop variable (line 235)
    for_loop_var_120155 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 4), columns_120154)
    # Assigning a type to the variable 'col' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'col', for_loop_var_120155)
    # SSA begins for a for statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    str_120156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 15), 'str', 'structure')
    # Getting the type of 'col' (line 236)
    col_120157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'col')
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___120158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 11), col_120157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_120159 = invoke(stypy.reporting.localization.Localization(__file__, 236, 11), getitem___120158, str_120156)
    
    
    # Obtaining the type of the subscript
    str_120160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 35), 'str', 'array')
    # Getting the type of 'col' (line 236)
    col_120161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 31), 'col')
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___120162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 31), col_120161, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_120163 = invoke(stypy.reporting.localization.Localization(__file__, 236, 31), getitem___120162, str_120160)
    
    # Applying the binary operator 'or' (line 236)
    result_or_keyword_120164 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), 'or', subscript_call_result_120159, subscript_call_result_120163)
    
    # Testing the type of an if condition (line 236)
    if_condition_120165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), result_or_keyword_120164)
    # Assigning a type to the variable 'if_condition_120165' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_120165', if_condition_120165)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 237)
    # Processing the call arguments (line 237)
    
    # Obtaining an instance of the builtin type 'tuple' (line 237)
    tuple_120168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 237)
    # Adding element type (line 237)
    
    # Obtaining an instance of the builtin type 'tuple' (line 237)
    tuple_120169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 237)
    # Adding element type (line 237)
    
    # Call to lower(...): (line 237)
    # Processing the call keyword arguments (line 237)
    kwargs_120175 = {}
    
    # Obtaining the type of the subscript
    str_120170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 31), 'str', 'name')
    # Getting the type of 'col' (line 237)
    col_120171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___120172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), col_120171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_120173 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), getitem___120172, str_120170)
    
    # Obtaining the member 'lower' of a type (line 237)
    lower_120174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), subscript_call_result_120173, 'lower')
    # Calling lower(args, kwargs) (line 237)
    lower_call_result_120176 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), lower_120174, *[], **kwargs_120175)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 27), tuple_120169, lower_call_result_120176)
    # Adding element type (line 237)
    
    # Obtaining the type of the subscript
    str_120177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 52), 'str', 'name')
    # Getting the type of 'col' (line 237)
    col_120178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 48), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___120179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 48), col_120178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_120180 = invoke(stypy.reporting.localization.Localization(__file__, 237, 48), getitem___120179, str_120177)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 27), tuple_120169, subscript_call_result_120180)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 26), tuple_120168, tuple_120169)
    # Adding element type (line 237)
    # Getting the type of 'np' (line 237)
    np_120181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 62), 'np', False)
    # Obtaining the member 'object_' of a type (line 237)
    object__120182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 62), np_120181, 'object_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 26), tuple_120168, object__120182)
    
    # Processing the call keyword arguments (line 237)
    kwargs_120183 = {}
    # Getting the type of 'dtype' (line 237)
    dtype_120166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'dtype', False)
    # Obtaining the member 'append' of a type (line 237)
    append_120167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), dtype_120166, 'append')
    # Calling append(args, kwargs) (line 237)
    append_call_result_120184 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), append_120167, *[tuple_120168], **kwargs_120183)
    
    # SSA branch for the else part of an if statement (line 236)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'str', 'typecode')
    # Getting the type of 'col' (line 239)
    col_120186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'col')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___120187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 15), col_120186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_120188 = invoke(stypy.reporting.localization.Localization(__file__, 239, 15), getitem___120187, str_120185)
    
    # Getting the type of 'DTYPE_DICT' (line 239)
    DTYPE_DICT_120189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 'DTYPE_DICT')
    # Applying the binary operator 'in' (line 239)
    result_contains_120190 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 15), 'in', subscript_call_result_120188, DTYPE_DICT_120189)
    
    # Testing the type of an if condition (line 239)
    if_condition_120191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 12), result_contains_120190)
    # Assigning a type to the variable 'if_condition_120191' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'if_condition_120191', if_condition_120191)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 240)
    # Processing the call arguments (line 240)
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_120194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_120195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    
    # Call to lower(...): (line 240)
    # Processing the call keyword arguments (line 240)
    kwargs_120201 = {}
    
    # Obtaining the type of the subscript
    str_120196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 35), 'str', 'name')
    # Getting the type of 'col' (line 240)
    col_120197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___120198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 31), col_120197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_120199 = invoke(stypy.reporting.localization.Localization(__file__, 240, 31), getitem___120198, str_120196)
    
    # Obtaining the member 'lower' of a type (line 240)
    lower_120200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 31), subscript_call_result_120199, 'lower')
    # Calling lower(args, kwargs) (line 240)
    lower_call_result_120202 = invoke(stypy.reporting.localization.Localization(__file__, 240, 31), lower_120200, *[], **kwargs_120201)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 31), tuple_120195, lower_call_result_120202)
    # Adding element type (line 240)
    
    # Obtaining the type of the subscript
    str_120203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 56), 'str', 'name')
    # Getting the type of 'col' (line 240)
    col_120204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 52), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___120205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 52), col_120204, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_120206 = invoke(stypy.reporting.localization.Localization(__file__, 240, 52), getitem___120205, str_120203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 31), tuple_120195, subscript_call_result_120206)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 30), tuple_120194, tuple_120195)
    # Adding element type (line 240)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 51), 'str', 'typecode')
    # Getting the type of 'col' (line 241)
    col_120208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 47), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___120209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 47), col_120208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_120210 = invoke(stypy.reporting.localization.Localization(__file__, 241, 47), getitem___120209, str_120207)
    
    # Getting the type of 'DTYPE_DICT' (line 241)
    DTYPE_DICT_120211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 36), 'DTYPE_DICT', False)
    # Obtaining the member '__getitem__' of a type (line 241)
    getitem___120212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 36), DTYPE_DICT_120211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 241)
    subscript_call_result_120213 = invoke(stypy.reporting.localization.Localization(__file__, 241, 36), getitem___120212, subscript_call_result_120210)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 30), tuple_120194, subscript_call_result_120213)
    
    # Processing the call keyword arguments (line 240)
    kwargs_120214 = {}
    # Getting the type of 'dtype' (line 240)
    dtype_120192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'dtype', False)
    # Obtaining the member 'append' of a type (line 240)
    append_120193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 16), dtype_120192, 'append')
    # Calling append(args, kwargs) (line 240)
    append_call_result_120215 = invoke(stypy.reporting.localization.Localization(__file__, 240, 16), append_120193, *[tuple_120194], **kwargs_120214)
    
    # SSA branch for the else part of an if statement (line 239)
    module_type_store.open_ssa_branch('else')
    
    # Call to Exception(...): (line 243)
    # Processing the call arguments (line 243)
    str_120217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 32), 'str', 'Variable type %i not implemented')
    
    # Obtaining the type of the subscript
    str_120218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 64), 'str', 'typecode')
    # Getting the type of 'col' (line 244)
    col_120219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 60), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___120220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 60), col_120219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_120221 = invoke(stypy.reporting.localization.Localization(__file__, 244, 60), getitem___120220, str_120218)
    
    # Applying the binary operator '%' (line 243)
    result_mod_120222 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 32), '%', str_120217, subscript_call_result_120221)
    
    # Processing the call keyword arguments (line 243)
    kwargs_120223 = {}
    # Getting the type of 'Exception' (line 243)
    Exception_120216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'Exception', False)
    # Calling Exception(args, kwargs) (line 243)
    Exception_call_result_120224 = invoke(stypy.reporting.localization.Localization(__file__, 243, 22), Exception_120216, *[result_mod_120222], **kwargs_120223)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 243, 16), Exception_call_result_120224, 'raise parameter', BaseException)
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to recarray(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_120227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    # Getting the type of 'nrows' (line 246)
    nrows_120228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'nrows', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 29), tuple_120227, nrows_120228)
    
    # Processing the call keyword arguments (line 246)
    # Getting the type of 'dtype' (line 246)
    dtype_120229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 45), 'dtype', False)
    keyword_120230 = dtype_120229
    kwargs_120231 = {'dtype': keyword_120230}
    # Getting the type of 'np' (line 246)
    np_120225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'np', False)
    # Obtaining the member 'recarray' of a type (line 246)
    recarray_120226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), np_120225, 'recarray')
    # Calling recarray(args, kwargs) (line 246)
    recarray_call_result_120232 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), recarray_120226, *[tuple_120227], **kwargs_120231)
    
    # Assigning a type to the variable 'structure' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'structure', recarray_call_result_120232)
    
    
    # Call to range(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'nrows' (line 248)
    nrows_120234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'nrows', False)
    # Processing the call keyword arguments (line 248)
    kwargs_120235 = {}
    # Getting the type of 'range' (line 248)
    range_120233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'range', False)
    # Calling range(args, kwargs) (line 248)
    range_call_result_120236 = invoke(stypy.reporting.localization.Localization(__file__, 248, 13), range_120233, *[nrows_120234], **kwargs_120235)
    
    # Testing the type of a for loop iterable (line 248)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 248, 4), range_call_result_120236)
    # Getting the type of the for loop variable (line 248)
    for_loop_var_120237 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 248, 4), range_call_result_120236)
    # Assigning a type to the variable 'i' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'i', for_loop_var_120237)
    # SSA begins for a for statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'columns' (line 249)
    columns_120238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'columns')
    # Testing the type of a for loop iterable (line 249)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 249, 8), columns_120238)
    # Getting the type of the for loop variable (line 249)
    for_loop_var_120239 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 249, 8), columns_120238)
    # Assigning a type to the variable 'col' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'col', for_loop_var_120239)
    # SSA begins for a for statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 250):
    
    # Assigning a Subscript to a Name (line 250):
    
    # Obtaining the type of the subscript
    str_120240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 24), 'str', 'typecode')
    # Getting the type of 'col' (line 250)
    col_120241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'col')
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___120242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 20), col_120241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 250)
    subscript_call_result_120243 = invoke(stypy.reporting.localization.Localization(__file__, 250, 20), getitem___120242, str_120240)
    
    # Assigning a type to the variable 'dtype' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'dtype', subscript_call_result_120243)
    
    
    # Obtaining the type of the subscript
    str_120244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 19), 'str', 'structure')
    # Getting the type of 'col' (line 251)
    col_120245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'col')
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___120246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), col_120245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_120247 = invoke(stypy.reporting.localization.Localization(__file__, 251, 15), getitem___120246, str_120244)
    
    # Testing the type of an if condition (line 251)
    if_condition_120248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 12), subscript_call_result_120247)
    # Assigning a type to the variable 'if_condition_120248' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'if_condition_120248', if_condition_120248)
    # SSA begins for if statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 252):
    
    # Assigning a Call to a Subscript (line 252):
    
    # Call to _read_structure(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'f' (line 252)
    f_120250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 60), 'f', False)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 66), 'str', 'name')
    # Getting the type of 'col' (line 253)
    col_120252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 62), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___120253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 62), col_120252, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_120254 = invoke(stypy.reporting.localization.Localization(__file__, 253, 62), getitem___120253, str_120251)
    
    
    # Obtaining the type of the subscript
    str_120255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 50), 'str', 'arrtable')
    # Getting the type of 'struct_desc' (line 253)
    struct_desc_120256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 38), 'struct_desc', False)
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___120257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 38), struct_desc_120256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_120258 = invoke(stypy.reporting.localization.Localization(__file__, 253, 38), getitem___120257, str_120255)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___120259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 38), subscript_call_result_120258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_120260 = invoke(stypy.reporting.localization.Localization(__file__, 253, 38), getitem___120259, subscript_call_result_120254)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 69), 'str', 'name')
    # Getting the type of 'col' (line 254)
    col_120262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 65), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___120263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 65), col_120262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_120264 = invoke(stypy.reporting.localization.Localization(__file__, 254, 65), getitem___120263, str_120261)
    
    
    # Obtaining the type of the subscript
    str_120265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 50), 'str', 'structtable')
    # Getting the type of 'struct_desc' (line 254)
    struct_desc_120266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 38), 'struct_desc', False)
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___120267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 38), struct_desc_120266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_120268 = invoke(stypy.reporting.localization.Localization(__file__, 254, 38), getitem___120267, str_120265)
    
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___120269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 38), subscript_call_result_120268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_120270 = invoke(stypy.reporting.localization.Localization(__file__, 254, 38), getitem___120269, subscript_call_result_120264)
    
    # Processing the call keyword arguments (line 252)
    kwargs_120271 = {}
    # Getting the type of '_read_structure' (line 252)
    _read_structure_120249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 44), '_read_structure', False)
    # Calling _read_structure(args, kwargs) (line 252)
    _read_structure_call_result_120272 = invoke(stypy.reporting.localization.Localization(__file__, 252, 44), _read_structure_120249, *[f_120250, subscript_call_result_120260, subscript_call_result_120270], **kwargs_120271)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 30), 'str', 'name')
    # Getting the type of 'col' (line 252)
    col_120274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'col')
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___120275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 26), col_120274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_120276 = invoke(stypy.reporting.localization.Localization(__file__, 252, 26), getitem___120275, str_120273)
    
    # Getting the type of 'structure' (line 252)
    structure_120277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'structure')
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___120278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), structure_120277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_120279 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), getitem___120278, subscript_call_result_120276)
    
    # Getting the type of 'i' (line 252)
    i_120280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 39), 'i')
    # Storing an element on a container (line 252)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 16), subscript_call_result_120279, (i_120280, _read_structure_call_result_120272))
    # SSA branch for the else part of an if statement (line 251)
    module_type_store.open_ssa_branch('else')
    
    
    # Obtaining the type of the subscript
    str_120281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 21), 'str', 'array')
    # Getting the type of 'col' (line 255)
    col_120282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'col')
    # Obtaining the member '__getitem__' of a type (line 255)
    getitem___120283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 17), col_120282, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 255)
    subscript_call_result_120284 = invoke(stypy.reporting.localization.Localization(__file__, 255, 17), getitem___120283, str_120281)
    
    # Testing the type of an if condition (line 255)
    if_condition_120285 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 17), subscript_call_result_120284)
    # Assigning a type to the variable 'if_condition_120285' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'if_condition_120285', if_condition_120285)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 256):
    
    # Assigning a Call to a Subscript (line 256):
    
    # Call to _read_array(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'f' (line 256)
    f_120287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 56), 'f', False)
    # Getting the type of 'dtype' (line 256)
    dtype_120288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 59), 'dtype', False)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 66), 'str', 'name')
    # Getting the type of 'col' (line 257)
    col_120290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 62), 'col', False)
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___120291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 62), col_120290, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_120292 = invoke(stypy.reporting.localization.Localization(__file__, 257, 62), getitem___120291, str_120289)
    
    
    # Obtaining the type of the subscript
    str_120293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 50), 'str', 'arrtable')
    # Getting the type of 'struct_desc' (line 257)
    struct_desc_120294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'struct_desc', False)
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___120295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 38), struct_desc_120294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_120296 = invoke(stypy.reporting.localization.Localization(__file__, 257, 38), getitem___120295, str_120293)
    
    # Obtaining the member '__getitem__' of a type (line 257)
    getitem___120297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 38), subscript_call_result_120296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 257)
    subscript_call_result_120298 = invoke(stypy.reporting.localization.Localization(__file__, 257, 38), getitem___120297, subscript_call_result_120292)
    
    # Processing the call keyword arguments (line 256)
    kwargs_120299 = {}
    # Getting the type of '_read_array' (line 256)
    _read_array_120286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 44), '_read_array', False)
    # Calling _read_array(args, kwargs) (line 256)
    _read_array_call_result_120300 = invoke(stypy.reporting.localization.Localization(__file__, 256, 44), _read_array_120286, *[f_120287, dtype_120288, subscript_call_result_120298], **kwargs_120299)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'str', 'name')
    # Getting the type of 'col' (line 256)
    col_120302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'col')
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___120303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 26), col_120302, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_120304 = invoke(stypy.reporting.localization.Localization(__file__, 256, 26), getitem___120303, str_120301)
    
    # Getting the type of 'structure' (line 256)
    structure_120305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'structure')
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___120306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 16), structure_120305, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_120307 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), getitem___120306, subscript_call_result_120304)
    
    # Getting the type of 'i' (line 256)
    i_120308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 39), 'i')
    # Storing an element on a container (line 256)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 16), subscript_call_result_120307, (i_120308, _read_array_call_result_120300))
    # SSA branch for the else part of an if statement (line 255)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Subscript (line 259):
    
    # Assigning a Call to a Subscript (line 259):
    
    # Call to _read_data(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'f' (line 259)
    f_120310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 55), 'f', False)
    # Getting the type of 'dtype' (line 259)
    dtype_120311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 58), 'dtype', False)
    # Processing the call keyword arguments (line 259)
    kwargs_120312 = {}
    # Getting the type of '_read_data' (line 259)
    _read_data_120309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 44), '_read_data', False)
    # Calling _read_data(args, kwargs) (line 259)
    _read_data_call_result_120313 = invoke(stypy.reporting.localization.Localization(__file__, 259, 44), _read_data_120309, *[f_120310, dtype_120311], **kwargs_120312)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 30), 'str', 'name')
    # Getting the type of 'col' (line 259)
    col_120315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'col')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___120316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 26), col_120315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_120317 = invoke(stypy.reporting.localization.Localization(__file__, 259, 26), getitem___120316, str_120314)
    
    # Getting the type of 'structure' (line 259)
    structure_120318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'structure')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___120319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 16), structure_120318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_120320 = invoke(stypy.reporting.localization.Localization(__file__, 259, 16), getitem___120319, subscript_call_result_120317)
    
    # Getting the type of 'i' (line 259)
    i_120321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 39), 'i')
    # Storing an element on a container (line 259)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 16), subscript_call_result_120320, (i_120321, _read_data_call_result_120313))
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 251)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_120322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'str', 'ndims')
    # Getting the type of 'array_desc' (line 262)
    array_desc_120323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 7), 'array_desc')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___120324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 7), array_desc_120323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_120325 = invoke(stypy.reporting.localization.Localization(__file__, 262, 7), getitem___120324, str_120322)
    
    int_120326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 29), 'int')
    # Applying the binary operator '>' (line 262)
    result_gt_120327 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 7), '>', subscript_call_result_120325, int_120326)
    
    # Testing the type of an if condition (line 262)
    if_condition_120328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 4), result_gt_120327)
    # Assigning a type to the variable 'if_condition_120328' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'if_condition_120328', if_condition_120328)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 263):
    
    # Assigning a Subscript to a Name (line 263):
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 263)
    # Processing the call arguments (line 263)
    
    # Obtaining the type of the subscript
    str_120330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 50), 'str', 'ndims')
    # Getting the type of 'array_desc' (line 263)
    array_desc_120331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 39), 'array_desc', False)
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___120332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 39), array_desc_120331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_120333 = invoke(stypy.reporting.localization.Localization(__file__, 263, 39), getitem___120332, str_120330)
    
    # Processing the call keyword arguments (line 263)
    kwargs_120334 = {}
    # Getting the type of 'int' (line 263)
    int_120329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 35), 'int', False)
    # Calling int(args, kwargs) (line 263)
    int_call_result_120335 = invoke(stypy.reporting.localization.Localization(__file__, 263, 35), int_120329, *[subscript_call_result_120333], **kwargs_120334)
    
    slice_120336 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 15), None, int_call_result_120335, None)
    
    # Obtaining the type of the subscript
    str_120337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 26), 'str', 'dims')
    # Getting the type of 'array_desc' (line 263)
    array_desc_120338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'array_desc')
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___120339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), array_desc_120338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_120340 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), getitem___120339, str_120337)
    
    # Obtaining the member '__getitem__' of a type (line 263)
    getitem___120341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), subscript_call_result_120340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 263)
    subscript_call_result_120342 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), getitem___120341, slice_120336)
    
    # Assigning a type to the variable 'dims' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'dims', subscript_call_result_120342)
    
    # Call to reverse(...): (line 264)
    # Processing the call keyword arguments (line 264)
    kwargs_120345 = {}
    # Getting the type of 'dims' (line 264)
    dims_120343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'dims', False)
    # Obtaining the member 'reverse' of a type (line 264)
    reverse_120344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), dims_120343, 'reverse')
    # Calling reverse(args, kwargs) (line 264)
    reverse_call_result_120346 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), reverse_120344, *[], **kwargs_120345)
    
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to reshape(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'dims' (line 265)
    dims_120349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 38), 'dims', False)
    # Processing the call keyword arguments (line 265)
    kwargs_120350 = {}
    # Getting the type of 'structure' (line 265)
    structure_120347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'structure', False)
    # Obtaining the member 'reshape' of a type (line 265)
    reshape_120348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 20), structure_120347, 'reshape')
    # Calling reshape(args, kwargs) (line 265)
    reshape_call_result_120351 = invoke(stypy.reporting.localization.Localization(__file__, 265, 20), reshape_120348, *[dims_120349], **kwargs_120350)
    
    # Assigning a type to the variable 'structure' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'structure', reshape_call_result_120351)
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'structure' (line 267)
    structure_120352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'structure')
    # Assigning a type to the variable 'stypy_return_type' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type', structure_120352)
    
    # ################# End of '_read_structure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_structure' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_120353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120353)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_structure'
    return stypy_return_type_120353

# Assigning a type to the variable '_read_structure' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), '_read_structure', _read_structure)

@norecursion
def _read_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_array'
    module_type_store = module_type_store.open_function_context('_read_array', 270, 0, False)
    
    # Passed parameters checking function
    _read_array.stypy_localization = localization
    _read_array.stypy_type_of_self = None
    _read_array.stypy_type_store = module_type_store
    _read_array.stypy_function_name = '_read_array'
    _read_array.stypy_param_names_list = ['f', 'typecode', 'array_desc']
    _read_array.stypy_varargs_param_name = None
    _read_array.stypy_kwargs_param_name = None
    _read_array.stypy_call_defaults = defaults
    _read_array.stypy_call_varargs = varargs
    _read_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_array', ['f', 'typecode', 'array_desc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_array', localization, ['f', 'typecode', 'array_desc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_array(...)' code ##################

    str_120354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, (-1)), 'str', '\n    Read an array of type `typecode`, with the array descriptor given as\n    `array_desc`.\n    ')
    
    
    # Getting the type of 'typecode' (line 276)
    typecode_120355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 7), 'typecode')
    
    # Obtaining an instance of the builtin type 'list' (line 276)
    list_120356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 276)
    # Adding element type (line 276)
    int_120357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120357)
    # Adding element type (line 276)
    int_120358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120358)
    # Adding element type (line 276)
    int_120359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120359)
    # Adding element type (line 276)
    int_120360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120360)
    # Adding element type (line 276)
    int_120361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120361)
    # Adding element type (line 276)
    int_120362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120362)
    # Adding element type (line 276)
    int_120363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120363)
    # Adding element type (line 276)
    int_120364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120364)
    # Adding element type (line 276)
    int_120365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 19), list_120356, int_120365)
    
    # Applying the binary operator 'in' (line 276)
    result_contains_120366 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 7), 'in', typecode_120355, list_120356)
    
    # Testing the type of an if condition (line 276)
    if_condition_120367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 4), result_contains_120366)
    # Assigning a type to the variable 'if_condition_120367' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'if_condition_120367', if_condition_120367)
    # SSA begins for if statement (line 276)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'typecode' (line 278)
    typecode_120368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'typecode')
    int_120369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 23), 'int')
    # Applying the binary operator '==' (line 278)
    result_eq_120370 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 11), '==', typecode_120368, int_120369)
    
    # Testing the type of an if condition (line 278)
    if_condition_120371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 8), result_eq_120370)
    # Assigning a type to the variable 'if_condition_120371' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'if_condition_120371', if_condition_120371)
    # SSA begins for if statement (line 278)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to _read_int32(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'f' (line 279)
    f_120373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 33), 'f', False)
    # Processing the call keyword arguments (line 279)
    kwargs_120374 = {}
    # Getting the type of '_read_int32' (line 279)
    _read_int32_120372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 21), '_read_int32', False)
    # Calling _read_int32(args, kwargs) (line 279)
    _read_int32_call_result_120375 = invoke(stypy.reporting.localization.Localization(__file__, 279, 21), _read_int32_120372, *[f_120373], **kwargs_120374)
    
    # Assigning a type to the variable 'nbytes' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'nbytes', _read_int32_call_result_120375)
    
    
    # Getting the type of 'nbytes' (line 280)
    nbytes_120376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'nbytes')
    
    # Obtaining the type of the subscript
    str_120377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 36), 'str', 'nbytes')
    # Getting the type of 'array_desc' (line 280)
    array_desc_120378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'array_desc')
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___120379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 25), array_desc_120378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_120380 = invoke(stypy.reporting.localization.Localization(__file__, 280, 25), getitem___120379, str_120377)
    
    # Applying the binary operator '!=' (line 280)
    result_ne_120381 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), '!=', nbytes_120376, subscript_call_result_120380)
    
    # Testing the type of an if condition (line 280)
    if_condition_120382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), result_ne_120381)
    # Assigning a type to the variable 'if_condition_120382' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_120382', if_condition_120382)
    # SSA begins for if statement (line 280)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 281)
    # Processing the call arguments (line 281)
    str_120385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 30), 'str', 'Not able to verify number of bytes from header')
    # Processing the call keyword arguments (line 281)
    kwargs_120386 = {}
    # Getting the type of 'warnings' (line 281)
    warnings_120383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 281)
    warn_120384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), warnings_120383, 'warn')
    # Calling warn(args, kwargs) (line 281)
    warn_call_result_120387 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), warn_120384, *[str_120385], **kwargs_120386)
    
    # SSA join for if statement (line 280)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 278)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 284):
    
    # Assigning a Call to a Name (line 284):
    
    # Call to fromstring(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Call to read(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Obtaining the type of the subscript
    str_120392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 48), 'str', 'nbytes')
    # Getting the type of 'array_desc' (line 284)
    array_desc_120393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 37), 'array_desc', False)
    # Obtaining the member '__getitem__' of a type (line 284)
    getitem___120394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 37), array_desc_120393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 284)
    subscript_call_result_120395 = invoke(stypy.reporting.localization.Localization(__file__, 284, 37), getitem___120394, str_120392)
    
    # Processing the call keyword arguments (line 284)
    kwargs_120396 = {}
    # Getting the type of 'f' (line 284)
    f_120390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 30), 'f', False)
    # Obtaining the member 'read' of a type (line 284)
    read_120391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 30), f_120390, 'read')
    # Calling read(args, kwargs) (line 284)
    read_call_result_120397 = invoke(stypy.reporting.localization.Localization(__file__, 284, 30), read_120391, *[subscript_call_result_120395], **kwargs_120396)
    
    # Processing the call keyword arguments (line 284)
    
    # Obtaining the type of the subscript
    # Getting the type of 'typecode' (line 285)
    typecode_120398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 49), 'typecode', False)
    # Getting the type of 'DTYPE_DICT' (line 285)
    DTYPE_DICT_120399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 38), 'DTYPE_DICT', False)
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___120400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 38), DTYPE_DICT_120399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 285)
    subscript_call_result_120401 = invoke(stypy.reporting.localization.Localization(__file__, 285, 38), getitem___120400, typecode_120398)
    
    keyword_120402 = subscript_call_result_120401
    kwargs_120403 = {'dtype': keyword_120402}
    # Getting the type of 'np' (line 284)
    np_120388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'np', False)
    # Obtaining the member 'fromstring' of a type (line 284)
    fromstring_120389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), np_120388, 'fromstring')
    # Calling fromstring(args, kwargs) (line 284)
    fromstring_call_result_120404 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), fromstring_120389, *[read_call_result_120397], **kwargs_120403)
    
    # Assigning a type to the variable 'array' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'array', fromstring_call_result_120404)
    # SSA branch for the else part of an if statement (line 276)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'typecode' (line 287)
    typecode_120405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 9), 'typecode')
    
    # Obtaining an instance of the builtin type 'list' (line 287)
    list_120406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 287)
    # Adding element type (line 287)
    int_120407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 21), list_120406, int_120407)
    # Adding element type (line 287)
    int_120408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 21), list_120406, int_120408)
    
    # Applying the binary operator 'in' (line 287)
    result_contains_120409 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 9), 'in', typecode_120405, list_120406)
    
    # Testing the type of an if condition (line 287)
    if_condition_120410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 9), result_contains_120409)
    # Assigning a type to the variable 'if_condition_120410' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 9), 'if_condition_120410', if_condition_120410)
    # SSA begins for if statement (line 287)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 291):
    
    # Assigning a Subscript to a Name (line 291):
    
    # Obtaining the type of the subscript
    int_120411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 60), 'int')
    int_120412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 63), 'int')
    slice_120413 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 291, 16), int_120411, None, int_120412)
    
    # Call to fromstring(...): (line 291)
    # Processing the call arguments (line 291)
    
    # Call to read(...): (line 291)
    # Processing the call arguments (line 291)
    
    # Obtaining the type of the subscript
    str_120418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 48), 'str', 'nbytes')
    # Getting the type of 'array_desc' (line 291)
    array_desc_120419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 37), 'array_desc', False)
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___120420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 37), array_desc_120419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_120421 = invoke(stypy.reporting.localization.Localization(__file__, 291, 37), getitem___120420, str_120418)
    
    int_120422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 58), 'int')
    # Applying the binary operator '*' (line 291)
    result_mul_120423 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 37), '*', subscript_call_result_120421, int_120422)
    
    # Processing the call keyword arguments (line 291)
    kwargs_120424 = {}
    # Getting the type of 'f' (line 291)
    f_120416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 30), 'f', False)
    # Obtaining the member 'read' of a type (line 291)
    read_120417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 30), f_120416, 'read')
    # Calling read(args, kwargs) (line 291)
    read_call_result_120425 = invoke(stypy.reporting.localization.Localization(__file__, 291, 30), read_120417, *[result_mul_120423], **kwargs_120424)
    
    # Processing the call keyword arguments (line 291)
    
    # Obtaining the type of the subscript
    # Getting the type of 'typecode' (line 292)
    typecode_120426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 49), 'typecode', False)
    # Getting the type of 'DTYPE_DICT' (line 292)
    DTYPE_DICT_120427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 38), 'DTYPE_DICT', False)
    # Obtaining the member '__getitem__' of a type (line 292)
    getitem___120428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 38), DTYPE_DICT_120427, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 292)
    subscript_call_result_120429 = invoke(stypy.reporting.localization.Localization(__file__, 292, 38), getitem___120428, typecode_120426)
    
    keyword_120430 = subscript_call_result_120429
    kwargs_120431 = {'dtype': keyword_120430}
    # Getting the type of 'np' (line 291)
    np_120414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'np', False)
    # Obtaining the member 'fromstring' of a type (line 291)
    fromstring_120415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), np_120414, 'fromstring')
    # Calling fromstring(args, kwargs) (line 291)
    fromstring_call_result_120432 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), fromstring_120415, *[read_call_result_120425], **kwargs_120431)
    
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___120433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), fromstring_call_result_120432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_120434 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), getitem___120433, slice_120413)
    
    # Assigning a type to the variable 'array' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'array', subscript_call_result_120434)
    # SSA branch for the else part of an if statement (line 287)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 297):
    
    # Assigning a List to a Name (line 297):
    
    # Obtaining an instance of the builtin type 'list' (line 297)
    list_120435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 297)
    
    # Assigning a type to the variable 'array' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'array', list_120435)
    
    
    # Call to range(...): (line 298)
    # Processing the call arguments (line 298)
    
    # Obtaining the type of the subscript
    str_120437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 34), 'str', 'nelements')
    # Getting the type of 'array_desc' (line 298)
    array_desc_120438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 'array_desc', False)
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___120439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 23), array_desc_120438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_120440 = invoke(stypy.reporting.localization.Localization(__file__, 298, 23), getitem___120439, str_120437)
    
    # Processing the call keyword arguments (line 298)
    kwargs_120441 = {}
    # Getting the type of 'range' (line 298)
    range_120436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'range', False)
    # Calling range(args, kwargs) (line 298)
    range_call_result_120442 = invoke(stypy.reporting.localization.Localization(__file__, 298, 17), range_120436, *[subscript_call_result_120440], **kwargs_120441)
    
    # Testing the type of a for loop iterable (line 298)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 298, 8), range_call_result_120442)
    # Getting the type of the for loop variable (line 298)
    for_loop_var_120443 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 298, 8), range_call_result_120442)
    # Assigning a type to the variable 'i' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'i', for_loop_var_120443)
    # SSA begins for a for statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 299):
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'typecode' (line 299)
    typecode_120444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'typecode')
    # Assigning a type to the variable 'dtype' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'dtype', typecode_120444)
    
    # Assigning a Call to a Name (line 300):
    
    # Assigning a Call to a Name (line 300):
    
    # Call to _read_data(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'f' (line 300)
    f_120446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 30), 'f', False)
    # Getting the type of 'dtype' (line 300)
    dtype_120447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'dtype', False)
    # Processing the call keyword arguments (line 300)
    kwargs_120448 = {}
    # Getting the type of '_read_data' (line 300)
    _read_data_120445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), '_read_data', False)
    # Calling _read_data(args, kwargs) (line 300)
    _read_data_call_result_120449 = invoke(stypy.reporting.localization.Localization(__file__, 300, 19), _read_data_120445, *[f_120446, dtype_120447], **kwargs_120448)
    
    # Assigning a type to the variable 'data' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'data', _read_data_call_result_120449)
    
    # Call to append(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'data' (line 301)
    data_120452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'data', False)
    # Processing the call keyword arguments (line 301)
    kwargs_120453 = {}
    # Getting the type of 'array' (line 301)
    array_120450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'array', False)
    # Obtaining the member 'append' of a type (line 301)
    append_120451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), array_120450, 'append')
    # Calling append(args, kwargs) (line 301)
    append_call_result_120454 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), append_120451, *[data_120452], **kwargs_120453)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 303):
    
    # Assigning a Call to a Name (line 303):
    
    # Call to array(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'array' (line 303)
    array_120457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 25), 'array', False)
    # Processing the call keyword arguments (line 303)
    # Getting the type of 'np' (line 303)
    np_120458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 38), 'np', False)
    # Obtaining the member 'object_' of a type (line 303)
    object__120459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 38), np_120458, 'object_')
    keyword_120460 = object__120459
    kwargs_120461 = {'dtype': keyword_120460}
    # Getting the type of 'np' (line 303)
    np_120455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 303)
    array_120456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), np_120455, 'array')
    # Calling array(args, kwargs) (line 303)
    array_call_result_120462 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), array_120456, *[array_120457], **kwargs_120461)
    
    # Assigning a type to the variable 'array' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'array', array_call_result_120462)
    # SSA join for if statement (line 287)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 276)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_120463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 18), 'str', 'ndims')
    # Getting the type of 'array_desc' (line 306)
    array_desc_120464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 7), 'array_desc')
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___120465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 7), array_desc_120464, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_120466 = invoke(stypy.reporting.localization.Localization(__file__, 306, 7), getitem___120465, str_120463)
    
    int_120467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 29), 'int')
    # Applying the binary operator '>' (line 306)
    result_gt_120468 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 7), '>', subscript_call_result_120466, int_120467)
    
    # Testing the type of an if condition (line 306)
    if_condition_120469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 4), result_gt_120468)
    # Assigning a type to the variable 'if_condition_120469' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'if_condition_120469', if_condition_120469)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 307):
    
    # Assigning a Subscript to a Name (line 307):
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 307)
    # Processing the call arguments (line 307)
    
    # Obtaining the type of the subscript
    str_120471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 50), 'str', 'ndims')
    # Getting the type of 'array_desc' (line 307)
    array_desc_120472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 39), 'array_desc', False)
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___120473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 39), array_desc_120472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_120474 = invoke(stypy.reporting.localization.Localization(__file__, 307, 39), getitem___120473, str_120471)
    
    # Processing the call keyword arguments (line 307)
    kwargs_120475 = {}
    # Getting the type of 'int' (line 307)
    int_120470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 35), 'int', False)
    # Calling int(args, kwargs) (line 307)
    int_call_result_120476 = invoke(stypy.reporting.localization.Localization(__file__, 307, 35), int_120470, *[subscript_call_result_120474], **kwargs_120475)
    
    slice_120477 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 307, 15), None, int_call_result_120476, None)
    
    # Obtaining the type of the subscript
    str_120478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 26), 'str', 'dims')
    # Getting the type of 'array_desc' (line 307)
    array_desc_120479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'array_desc')
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___120480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 15), array_desc_120479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_120481 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), getitem___120480, str_120478)
    
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___120482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 15), subscript_call_result_120481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_120483 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), getitem___120482, slice_120477)
    
    # Assigning a type to the variable 'dims' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'dims', subscript_call_result_120483)
    
    # Call to reverse(...): (line 308)
    # Processing the call keyword arguments (line 308)
    kwargs_120486 = {}
    # Getting the type of 'dims' (line 308)
    dims_120484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'dims', False)
    # Obtaining the member 'reverse' of a type (line 308)
    reverse_120485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), dims_120484, 'reverse')
    # Calling reverse(args, kwargs) (line 308)
    reverse_call_result_120487 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), reverse_120485, *[], **kwargs_120486)
    
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to reshape(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'dims' (line 309)
    dims_120490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 30), 'dims', False)
    # Processing the call keyword arguments (line 309)
    kwargs_120491 = {}
    # Getting the type of 'array' (line 309)
    array_120488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'array', False)
    # Obtaining the member 'reshape' of a type (line 309)
    reshape_120489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), array_120488, 'reshape')
    # Calling reshape(args, kwargs) (line 309)
    reshape_call_result_120492 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), reshape_120489, *[dims_120490], **kwargs_120491)
    
    # Assigning a type to the variable 'array' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'array', reshape_call_result_120492)
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _align_32(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'f' (line 312)
    f_120494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 14), 'f', False)
    # Processing the call keyword arguments (line 312)
    kwargs_120495 = {}
    # Getting the type of '_align_32' (line 312)
    _align_32_120493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), '_align_32', False)
    # Calling _align_32(args, kwargs) (line 312)
    _align_32_call_result_120496 = invoke(stypy.reporting.localization.Localization(__file__, 312, 4), _align_32_120493, *[f_120494], **kwargs_120495)
    
    # Getting the type of 'array' (line 314)
    array_120497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'array')
    # Assigning a type to the variable 'stypy_return_type' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type', array_120497)
    
    # ################# End of '_read_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_array' in the type store
    # Getting the type of 'stypy_return_type' (line 270)
    stypy_return_type_120498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120498)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_array'
    return stypy_return_type_120498

# Assigning a type to the variable '_read_array' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), '_read_array', _read_array)

@norecursion
def _read_record(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_record'
    module_type_store = module_type_store.open_function_context('_read_record', 317, 0, False)
    
    # Passed parameters checking function
    _read_record.stypy_localization = localization
    _read_record.stypy_type_of_self = None
    _read_record.stypy_type_store = module_type_store
    _read_record.stypy_function_name = '_read_record'
    _read_record.stypy_param_names_list = ['f']
    _read_record.stypy_varargs_param_name = None
    _read_record.stypy_kwargs_param_name = None
    _read_record.stypy_call_defaults = defaults
    _read_record.stypy_call_varargs = varargs
    _read_record.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_record', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_record', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_record(...)' code ##################

    str_120499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 4), 'str', 'Function to read in a full record')
    
    # Assigning a Dict to a Name (line 320):
    
    # Assigning a Dict to a Name (line 320):
    
    # Obtaining an instance of the builtin type 'dict' (line 320)
    dict_120500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 320)
    # Adding element type (key, value) (line 320)
    str_120501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 14), 'str', 'rectype')
    
    # Call to _read_long(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'f' (line 320)
    f_120503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 36), 'f', False)
    # Processing the call keyword arguments (line 320)
    kwargs_120504 = {}
    # Getting the type of '_read_long' (line 320)
    _read_long_120502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 320)
    _read_long_call_result_120505 = invoke(stypy.reporting.localization.Localization(__file__, 320, 25), _read_long_120502, *[f_120503], **kwargs_120504)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 13), dict_120500, (str_120501, _read_long_call_result_120505))
    
    # Assigning a type to the variable 'record' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'record', dict_120500)
    
    # Assigning a Call to a Name (line 322):
    
    # Assigning a Call to a Name (line 322):
    
    # Call to _read_uint32(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'f' (line 322)
    f_120507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 27), 'f', False)
    # Processing the call keyword arguments (line 322)
    kwargs_120508 = {}
    # Getting the type of '_read_uint32' (line 322)
    _read_uint32_120506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 14), '_read_uint32', False)
    # Calling _read_uint32(args, kwargs) (line 322)
    _read_uint32_call_result_120509 = invoke(stypy.reporting.localization.Localization(__file__, 322, 14), _read_uint32_120506, *[f_120507], **kwargs_120508)
    
    # Assigning a type to the variable 'nextrec' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'nextrec', _read_uint32_call_result_120509)
    
    # Getting the type of 'nextrec' (line 323)
    nextrec_120510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'nextrec')
    
    # Call to _read_uint32(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'f' (line 323)
    f_120512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 28), 'f', False)
    # Processing the call keyword arguments (line 323)
    kwargs_120513 = {}
    # Getting the type of '_read_uint32' (line 323)
    _read_uint32_120511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), '_read_uint32', False)
    # Calling _read_uint32(args, kwargs) (line 323)
    _read_uint32_call_result_120514 = invoke(stypy.reporting.localization.Localization(__file__, 323, 15), _read_uint32_120511, *[f_120512], **kwargs_120513)
    
    int_120515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 33), 'int')
    int_120516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 36), 'int')
    # Applying the binary operator '**' (line 323)
    result_pow_120517 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 33), '**', int_120515, int_120516)
    
    # Applying the binary operator '*' (line 323)
    result_mul_120518 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 15), '*', _read_uint32_call_result_120514, result_pow_120517)
    
    # Applying the binary operator '+=' (line 323)
    result_iadd_120519 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 4), '+=', nextrec_120510, result_mul_120518)
    # Assigning a type to the variable 'nextrec' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'nextrec', result_iadd_120519)
    
    
    # Call to _skip_bytes(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'f' (line 325)
    f_120521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'f', False)
    int_120522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 19), 'int')
    # Processing the call keyword arguments (line 325)
    kwargs_120523 = {}
    # Getting the type of '_skip_bytes' (line 325)
    _skip_bytes_120520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), '_skip_bytes', False)
    # Calling _skip_bytes(args, kwargs) (line 325)
    _skip_bytes_call_result_120524 = invoke(stypy.reporting.localization.Localization(__file__, 325, 4), _skip_bytes_120520, *[f_120521, int_120522], **kwargs_120523)
    
    
    
    
    
    # Obtaining the type of the subscript
    str_120525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 18), 'str', 'rectype')
    # Getting the type of 'record' (line 327)
    record_120526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'record')
    # Obtaining the member '__getitem__' of a type (line 327)
    getitem___120527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 11), record_120526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 327)
    subscript_call_result_120528 = invoke(stypy.reporting.localization.Localization(__file__, 327, 11), getitem___120527, str_120525)
    
    # Getting the type of 'RECTYPE_DICT' (line 327)
    RECTYPE_DICT_120529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'RECTYPE_DICT')
    # Applying the binary operator 'in' (line 327)
    result_contains_120530 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 11), 'in', subscript_call_result_120528, RECTYPE_DICT_120529)
    
    # Applying the 'not' unary operator (line 327)
    result_not__120531 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 7), 'not', result_contains_120530)
    
    # Testing the type of an if condition (line 327)
    if_condition_120532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 4), result_not__120531)
    # Assigning a type to the variable 'if_condition_120532' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'if_condition_120532', if_condition_120532)
    # SSA begins for if statement (line 327)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 328)
    # Processing the call arguments (line 328)
    str_120534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 24), 'str', 'Unknown RECTYPE: %i')
    
    # Obtaining the type of the subscript
    str_120535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 55), 'str', 'rectype')
    # Getting the type of 'record' (line 328)
    record_120536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 48), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___120537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 48), record_120536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_120538 = invoke(stypy.reporting.localization.Localization(__file__, 328, 48), getitem___120537, str_120535)
    
    # Applying the binary operator '%' (line 328)
    result_mod_120539 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 24), '%', str_120534, subscript_call_result_120538)
    
    # Processing the call keyword arguments (line 328)
    kwargs_120540 = {}
    # Getting the type of 'Exception' (line 328)
    Exception_120533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 328)
    Exception_call_result_120541 = invoke(stypy.reporting.localization.Localization(__file__, 328, 14), Exception_120533, *[result_mod_120539], **kwargs_120540)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 328, 8), Exception_call_result_120541, 'raise parameter', BaseException)
    # SSA join for if statement (line 327)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 330):
    
    # Assigning a Subscript to a Subscript (line 330):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_120542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 44), 'str', 'rectype')
    # Getting the type of 'record' (line 330)
    record_120543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 37), 'record')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___120544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 37), record_120543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_120545 = invoke(stypy.reporting.localization.Localization(__file__, 330, 37), getitem___120544, str_120542)
    
    # Getting the type of 'RECTYPE_DICT' (line 330)
    RECTYPE_DICT_120546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'RECTYPE_DICT')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___120547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 24), RECTYPE_DICT_120546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_120548 = invoke(stypy.reporting.localization.Localization(__file__, 330, 24), getitem___120547, subscript_call_result_120545)
    
    # Getting the type of 'record' (line 330)
    record_120549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'record')
    str_120550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 11), 'str', 'rectype')
    # Storing an element on a container (line 330)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 4), record_120549, (str_120550, subscript_call_result_120548))
    
    
    
    # Obtaining the type of the subscript
    str_120551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 14), 'str', 'rectype')
    # Getting the type of 'record' (line 332)
    record_120552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 7), 'record')
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___120553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 7), record_120552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_120554 = invoke(stypy.reporting.localization.Localization(__file__, 332, 7), getitem___120553, str_120551)
    
    
    # Obtaining an instance of the builtin type 'list' (line 332)
    list_120555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 332)
    # Adding element type (line 332)
    str_120556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 29), 'str', 'VARIABLE')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 28), list_120555, str_120556)
    # Adding element type (line 332)
    str_120557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 41), 'str', 'HEAP_DATA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 28), list_120555, str_120557)
    
    # Applying the binary operator 'in' (line 332)
    result_contains_120558 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 7), 'in', subscript_call_result_120554, list_120555)
    
    # Testing the type of an if condition (line 332)
    if_condition_120559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 4), result_contains_120558)
    # Assigning a type to the variable 'if_condition_120559' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'if_condition_120559', if_condition_120559)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    str_120560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 18), 'str', 'rectype')
    # Getting the type of 'record' (line 334)
    record_120561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'record')
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___120562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 11), record_120561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_120563 = invoke(stypy.reporting.localization.Localization(__file__, 334, 11), getitem___120562, str_120560)
    
    str_120564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 32), 'str', 'VARIABLE')
    # Applying the binary operator '==' (line 334)
    result_eq_120565 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), '==', subscript_call_result_120563, str_120564)
    
    # Testing the type of an if condition (line 334)
    if_condition_120566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_eq_120565)
    # Assigning a type to the variable 'if_condition_120566' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_120566', if_condition_120566)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 335):
    
    # Assigning a Call to a Subscript (line 335):
    
    # Call to _read_string(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'f' (line 335)
    f_120568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 45), 'f', False)
    # Processing the call keyword arguments (line 335)
    kwargs_120569 = {}
    # Getting the type of '_read_string' (line 335)
    _read_string_120567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 32), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 335)
    _read_string_call_result_120570 = invoke(stypy.reporting.localization.Localization(__file__, 335, 32), _read_string_120567, *[f_120568], **kwargs_120569)
    
    # Getting the type of 'record' (line 335)
    record_120571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'record')
    str_120572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 19), 'str', 'varname')
    # Storing an element on a container (line 335)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 12), record_120571, (str_120572, _read_string_call_result_120570))
    # SSA branch for the else part of an if statement (line 334)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Subscript (line 337):
    
    # Assigning a Call to a Subscript (line 337):
    
    # Call to _read_long(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'f' (line 337)
    f_120574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 46), 'f', False)
    # Processing the call keyword arguments (line 337)
    kwargs_120575 = {}
    # Getting the type of '_read_long' (line 337)
    _read_long_120573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 35), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 337)
    _read_long_call_result_120576 = invoke(stypy.reporting.localization.Localization(__file__, 337, 35), _read_long_120573, *[f_120574], **kwargs_120575)
    
    # Getting the type of 'record' (line 337)
    record_120577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'record')
    str_120578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 19), 'str', 'heap_index')
    # Storing an element on a container (line 337)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 12), record_120577, (str_120578, _read_long_call_result_120576))
    
    # Call to _skip_bytes(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'f' (line 338)
    f_120580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 24), 'f', False)
    int_120581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 27), 'int')
    # Processing the call keyword arguments (line 338)
    kwargs_120582 = {}
    # Getting the type of '_skip_bytes' (line 338)
    _skip_bytes_120579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), '_skip_bytes', False)
    # Calling _skip_bytes(args, kwargs) (line 338)
    _skip_bytes_call_result_120583 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), _skip_bytes_120579, *[f_120580, int_120581], **kwargs_120582)
    
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 340):
    
    # Assigning a Call to a Name (line 340):
    
    # Call to _read_typedesc(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'f' (line 340)
    f_120585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 37), 'f', False)
    # Processing the call keyword arguments (line 340)
    kwargs_120586 = {}
    # Getting the type of '_read_typedesc' (line 340)
    _read_typedesc_120584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 22), '_read_typedesc', False)
    # Calling _read_typedesc(args, kwargs) (line 340)
    _read_typedesc_call_result_120587 = invoke(stypy.reporting.localization.Localization(__file__, 340, 22), _read_typedesc_120584, *[f_120585], **kwargs_120586)
    
    # Assigning a type to the variable 'rectypedesc' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'rectypedesc', _read_typedesc_call_result_120587)
    
    
    
    # Obtaining the type of the subscript
    str_120588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'str', 'typecode')
    # Getting the type of 'rectypedesc' (line 342)
    rectypedesc_120589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 11), 'rectypedesc')
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___120590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 11), rectypedesc_120589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_120591 = invoke(stypy.reporting.localization.Localization(__file__, 342, 11), getitem___120590, str_120588)
    
    int_120592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 38), 'int')
    # Applying the binary operator '==' (line 342)
    result_eq_120593 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), '==', subscript_call_result_120591, int_120592)
    
    # Testing the type of an if condition (line 342)
    if_condition_120594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 8), result_eq_120593)
    # Assigning a type to the variable 'if_condition_120594' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'if_condition_120594', if_condition_120594)
    # SSA begins for if statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'nextrec' (line 344)
    nextrec_120595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'nextrec')
    
    # Call to tell(...): (line 344)
    # Processing the call keyword arguments (line 344)
    kwargs_120598 = {}
    # Getting the type of 'f' (line 344)
    f_120596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'f', False)
    # Obtaining the member 'tell' of a type (line 344)
    tell_120597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 26), f_120596, 'tell')
    # Calling tell(args, kwargs) (line 344)
    tell_call_result_120599 = invoke(stypy.reporting.localization.Localization(__file__, 344, 26), tell_120597, *[], **kwargs_120598)
    
    # Applying the binary operator '==' (line 344)
    result_eq_120600 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 15), '==', nextrec_120595, tell_call_result_120599)
    
    # Testing the type of an if condition (line 344)
    if_condition_120601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 12), result_eq_120600)
    # Assigning a type to the variable 'if_condition_120601' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'if_condition_120601', if_condition_120601)
    # SSA begins for if statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 345):
    
    # Assigning a Name to a Subscript (line 345):
    # Getting the type of 'None' (line 345)
    None_120602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 33), 'None')
    # Getting the type of 'record' (line 345)
    record_120603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 16), 'record')
    str_120604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 23), 'str', 'data')
    # Storing an element on a container (line 345)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 16), record_120603, (str_120604, None_120602))
    # SSA branch for the else part of an if statement (line 344)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 347)
    # Processing the call arguments (line 347)
    str_120606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 33), 'str', 'Unexpected type code: 0')
    # Processing the call keyword arguments (line 347)
    kwargs_120607 = {}
    # Getting the type of 'ValueError' (line 347)
    ValueError_120605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 347)
    ValueError_call_result_120608 = invoke(stypy.reporting.localization.Localization(__file__, 347, 22), ValueError_120605, *[str_120606], **kwargs_120607)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 347, 16), ValueError_call_result_120608, 'raise parameter', BaseException)
    # SSA join for if statement (line 344)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 342)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 351):
    
    # Assigning a Call to a Name (line 351):
    
    # Call to _read_long(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'f' (line 351)
    f_120610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 34), 'f', False)
    # Processing the call keyword arguments (line 351)
    kwargs_120611 = {}
    # Getting the type of '_read_long' (line 351)
    _read_long_120609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 351)
    _read_long_call_result_120612 = invoke(stypy.reporting.localization.Localization(__file__, 351, 23), _read_long_120609, *[f_120610], **kwargs_120611)
    
    # Assigning a type to the variable 'varstart' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'varstart', _read_long_call_result_120612)
    
    
    # Getting the type of 'varstart' (line 352)
    varstart_120613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'varstart')
    int_120614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 27), 'int')
    # Applying the binary operator '!=' (line 352)
    result_ne_120615 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 15), '!=', varstart_120613, int_120614)
    
    # Testing the type of an if condition (line 352)
    if_condition_120616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 12), result_ne_120615)
    # Assigning a type to the variable 'if_condition_120616' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'if_condition_120616', if_condition_120616)
    # SSA begins for if statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 353)
    # Processing the call arguments (line 353)
    str_120618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 32), 'str', 'VARSTART is not 7')
    # Processing the call keyword arguments (line 353)
    kwargs_120619 = {}
    # Getting the type of 'Exception' (line 353)
    Exception_120617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 22), 'Exception', False)
    # Calling Exception(args, kwargs) (line 353)
    Exception_call_result_120620 = invoke(stypy.reporting.localization.Localization(__file__, 353, 22), Exception_120617, *[str_120618], **kwargs_120619)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 353, 16), Exception_call_result_120620, 'raise parameter', BaseException)
    # SSA join for if statement (line 352)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_120621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 27), 'str', 'structure')
    # Getting the type of 'rectypedesc' (line 355)
    rectypedesc_120622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 15), 'rectypedesc')
    # Obtaining the member '__getitem__' of a type (line 355)
    getitem___120623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 15), rectypedesc_120622, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 355)
    subscript_call_result_120624 = invoke(stypy.reporting.localization.Localization(__file__, 355, 15), getitem___120623, str_120621)
    
    # Testing the type of an if condition (line 355)
    if_condition_120625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 12), subscript_call_result_120624)
    # Assigning a type to the variable 'if_condition_120625' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'if_condition_120625', if_condition_120625)
    # SSA begins for if statement (line 355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 356):
    
    # Assigning a Call to a Subscript (line 356):
    
    # Call to _read_structure(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'f' (line 356)
    f_120627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 49), 'f', False)
    
    # Obtaining the type of the subscript
    str_120628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 64), 'str', 'array_desc')
    # Getting the type of 'rectypedesc' (line 356)
    rectypedesc_120629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 52), 'rectypedesc', False)
    # Obtaining the member '__getitem__' of a type (line 356)
    getitem___120630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 52), rectypedesc_120629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 356)
    subscript_call_result_120631 = invoke(stypy.reporting.localization.Localization(__file__, 356, 52), getitem___120630, str_120628)
    
    
    # Obtaining the type of the subscript
    str_120632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 64), 'str', 'struct_desc')
    # Getting the type of 'rectypedesc' (line 357)
    rectypedesc_120633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 52), 'rectypedesc', False)
    # Obtaining the member '__getitem__' of a type (line 357)
    getitem___120634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 52), rectypedesc_120633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 357)
    subscript_call_result_120635 = invoke(stypy.reporting.localization.Localization(__file__, 357, 52), getitem___120634, str_120632)
    
    # Processing the call keyword arguments (line 356)
    kwargs_120636 = {}
    # Getting the type of '_read_structure' (line 356)
    _read_structure_120626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), '_read_structure', False)
    # Calling _read_structure(args, kwargs) (line 356)
    _read_structure_call_result_120637 = invoke(stypy.reporting.localization.Localization(__file__, 356, 33), _read_structure_120626, *[f_120627, subscript_call_result_120631, subscript_call_result_120635], **kwargs_120636)
    
    # Getting the type of 'record' (line 356)
    record_120638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'record')
    str_120639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 23), 'str', 'data')
    # Storing an element on a container (line 356)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 16), record_120638, (str_120639, _read_structure_call_result_120637))
    # SSA branch for the else part of an if statement (line 355)
    module_type_store.open_ssa_branch('else')
    
    
    # Obtaining the type of the subscript
    str_120640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 29), 'str', 'array')
    # Getting the type of 'rectypedesc' (line 358)
    rectypedesc_120641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'rectypedesc')
    # Obtaining the member '__getitem__' of a type (line 358)
    getitem___120642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 17), rectypedesc_120641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 358)
    subscript_call_result_120643 = invoke(stypy.reporting.localization.Localization(__file__, 358, 17), getitem___120642, str_120640)
    
    # Testing the type of an if condition (line 358)
    if_condition_120644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 17), subscript_call_result_120643)
    # Assigning a type to the variable 'if_condition_120644' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'if_condition_120644', if_condition_120644)
    # SSA begins for if statement (line 358)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 359):
    
    # Assigning a Call to a Subscript (line 359):
    
    # Call to _read_array(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'f' (line 359)
    f_120646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 45), 'f', False)
    
    # Obtaining the type of the subscript
    str_120647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 60), 'str', 'typecode')
    # Getting the type of 'rectypedesc' (line 359)
    rectypedesc_120648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 48), 'rectypedesc', False)
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___120649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 48), rectypedesc_120648, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_120650 = invoke(stypy.reporting.localization.Localization(__file__, 359, 48), getitem___120649, str_120647)
    
    
    # Obtaining the type of the subscript
    str_120651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 60), 'str', 'array_desc')
    # Getting the type of 'rectypedesc' (line 360)
    rectypedesc_120652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 48), 'rectypedesc', False)
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___120653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 48), rectypedesc_120652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_120654 = invoke(stypy.reporting.localization.Localization(__file__, 360, 48), getitem___120653, str_120651)
    
    # Processing the call keyword arguments (line 359)
    kwargs_120655 = {}
    # Getting the type of '_read_array' (line 359)
    _read_array_120645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), '_read_array', False)
    # Calling _read_array(args, kwargs) (line 359)
    _read_array_call_result_120656 = invoke(stypy.reporting.localization.Localization(__file__, 359, 33), _read_array_120645, *[f_120646, subscript_call_result_120650, subscript_call_result_120654], **kwargs_120655)
    
    # Getting the type of 'record' (line 359)
    record_120657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'record')
    str_120658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 23), 'str', 'data')
    # Storing an element on a container (line 359)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 16), record_120657, (str_120658, _read_array_call_result_120656))
    # SSA branch for the else part of an if statement (line 358)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 362):
    
    # Assigning a Subscript to a Name (line 362):
    
    # Obtaining the type of the subscript
    str_120659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 36), 'str', 'typecode')
    # Getting the type of 'rectypedesc' (line 362)
    rectypedesc_120660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'rectypedesc')
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___120661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 24), rectypedesc_120660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_120662 = invoke(stypy.reporting.localization.Localization(__file__, 362, 24), getitem___120661, str_120659)
    
    # Assigning a type to the variable 'dtype' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'dtype', subscript_call_result_120662)
    
    # Assigning a Call to a Subscript (line 363):
    
    # Assigning a Call to a Subscript (line 363):
    
    # Call to _read_data(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'f' (line 363)
    f_120664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 44), 'f', False)
    # Getting the type of 'dtype' (line 363)
    dtype_120665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 47), 'dtype', False)
    # Processing the call keyword arguments (line 363)
    kwargs_120666 = {}
    # Getting the type of '_read_data' (line 363)
    _read_data_120663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 33), '_read_data', False)
    # Calling _read_data(args, kwargs) (line 363)
    _read_data_call_result_120667 = invoke(stypy.reporting.localization.Localization(__file__, 363, 33), _read_data_120663, *[f_120664, dtype_120665], **kwargs_120666)
    
    # Getting the type of 'record' (line 363)
    record_120668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'record')
    str_120669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 23), 'str', 'data')
    # Storing an element on a container (line 363)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 16), record_120668, (str_120669, _read_data_call_result_120667))
    # SSA join for if statement (line 358)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 355)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 342)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 332)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 365)
    record_120671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 365)
    getitem___120672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 9), record_120671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 365)
    subscript_call_result_120673 = invoke(stypy.reporting.localization.Localization(__file__, 365, 9), getitem___120672, str_120670)
    
    str_120674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 30), 'str', 'TIMESTAMP')
    # Applying the binary operator '==' (line 365)
    result_eq_120675 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 9), '==', subscript_call_result_120673, str_120674)
    
    # Testing the type of an if condition (line 365)
    if_condition_120676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 9), result_eq_120675)
    # Assigning a type to the variable 'if_condition_120676' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 9), 'if_condition_120676', if_condition_120676)
    # SSA begins for if statement (line 365)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _skip_bytes(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'f' (line 367)
    f_120678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'f', False)
    int_120679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 23), 'int')
    int_120680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 25), 'int')
    # Applying the binary operator '*' (line 367)
    result_mul_120681 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 23), '*', int_120679, int_120680)
    
    # Processing the call keyword arguments (line 367)
    kwargs_120682 = {}
    # Getting the type of '_skip_bytes' (line 367)
    _skip_bytes_120677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), '_skip_bytes', False)
    # Calling _skip_bytes(args, kwargs) (line 367)
    _skip_bytes_call_result_120683 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), _skip_bytes_120677, *[f_120678, result_mul_120681], **kwargs_120682)
    
    
    # Assigning a Call to a Subscript (line 368):
    
    # Assigning a Call to a Subscript (line 368):
    
    # Call to _read_string(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'f' (line 368)
    f_120685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 38), 'f', False)
    # Processing the call keyword arguments (line 368)
    kwargs_120686 = {}
    # Getting the type of '_read_string' (line 368)
    _read_string_120684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 368)
    _read_string_call_result_120687 = invoke(stypy.reporting.localization.Localization(__file__, 368, 25), _read_string_120684, *[f_120685], **kwargs_120686)
    
    # Getting the type of 'record' (line 368)
    record_120688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'record')
    str_120689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 15), 'str', 'date')
    # Storing an element on a container (line 368)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 8), record_120688, (str_120689, _read_string_call_result_120687))
    
    # Assigning a Call to a Subscript (line 369):
    
    # Assigning a Call to a Subscript (line 369):
    
    # Call to _read_string(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'f' (line 369)
    f_120691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 38), 'f', False)
    # Processing the call keyword arguments (line 369)
    kwargs_120692 = {}
    # Getting the type of '_read_string' (line 369)
    _read_string_120690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 25), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 369)
    _read_string_call_result_120693 = invoke(stypy.reporting.localization.Localization(__file__, 369, 25), _read_string_120690, *[f_120691], **kwargs_120692)
    
    # Getting the type of 'record' (line 369)
    record_120694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'record')
    str_120695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 15), 'str', 'user')
    # Storing an element on a container (line 369)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 8), record_120694, (str_120695, _read_string_call_result_120693))
    
    # Assigning a Call to a Subscript (line 370):
    
    # Assigning a Call to a Subscript (line 370):
    
    # Call to _read_string(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'f' (line 370)
    f_120697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 38), 'f', False)
    # Processing the call keyword arguments (line 370)
    kwargs_120698 = {}
    # Getting the type of '_read_string' (line 370)
    _read_string_120696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 25), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 370)
    _read_string_call_result_120699 = invoke(stypy.reporting.localization.Localization(__file__, 370, 25), _read_string_120696, *[f_120697], **kwargs_120698)
    
    # Getting the type of 'record' (line 370)
    record_120700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'record')
    str_120701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 15), 'str', 'host')
    # Storing an element on a container (line 370)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 8), record_120700, (str_120701, _read_string_call_result_120699))
    # SSA branch for the else part of an if statement (line 365)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 372)
    record_120703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___120704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 9), record_120703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 372)
    subscript_call_result_120705 = invoke(stypy.reporting.localization.Localization(__file__, 372, 9), getitem___120704, str_120702)
    
    str_120706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 30), 'str', 'VERSION')
    # Applying the binary operator '==' (line 372)
    result_eq_120707 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 9), '==', subscript_call_result_120705, str_120706)
    
    # Testing the type of an if condition (line 372)
    if_condition_120708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 9), result_eq_120707)
    # Assigning a type to the variable 'if_condition_120708' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 9), 'if_condition_120708', if_condition_120708)
    # SSA begins for if statement (line 372)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 374):
    
    # Assigning a Call to a Subscript (line 374):
    
    # Call to _read_long(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'f' (line 374)
    f_120710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 38), 'f', False)
    # Processing the call keyword arguments (line 374)
    kwargs_120711 = {}
    # Getting the type of '_read_long' (line 374)
    _read_long_120709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 27), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 374)
    _read_long_call_result_120712 = invoke(stypy.reporting.localization.Localization(__file__, 374, 27), _read_long_120709, *[f_120710], **kwargs_120711)
    
    # Getting the type of 'record' (line 374)
    record_120713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'record')
    str_120714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 15), 'str', 'format')
    # Storing an element on a container (line 374)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), record_120713, (str_120714, _read_long_call_result_120712))
    
    # Assigning a Call to a Subscript (line 375):
    
    # Assigning a Call to a Subscript (line 375):
    
    # Call to _read_string(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'f' (line 375)
    f_120716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 38), 'f', False)
    # Processing the call keyword arguments (line 375)
    kwargs_120717 = {}
    # Getting the type of '_read_string' (line 375)
    _read_string_120715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 25), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 375)
    _read_string_call_result_120718 = invoke(stypy.reporting.localization.Localization(__file__, 375, 25), _read_string_120715, *[f_120716], **kwargs_120717)
    
    # Getting the type of 'record' (line 375)
    record_120719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'record')
    str_120720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 15), 'str', 'arch')
    # Storing an element on a container (line 375)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 8), record_120719, (str_120720, _read_string_call_result_120718))
    
    # Assigning a Call to a Subscript (line 376):
    
    # Assigning a Call to a Subscript (line 376):
    
    # Call to _read_string(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'f' (line 376)
    f_120722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 36), 'f', False)
    # Processing the call keyword arguments (line 376)
    kwargs_120723 = {}
    # Getting the type of '_read_string' (line 376)
    _read_string_120721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 23), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 376)
    _read_string_call_result_120724 = invoke(stypy.reporting.localization.Localization(__file__, 376, 23), _read_string_120721, *[f_120722], **kwargs_120723)
    
    # Getting the type of 'record' (line 376)
    record_120725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'record')
    str_120726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 15), 'str', 'os')
    # Storing an element on a container (line 376)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 8), record_120725, (str_120726, _read_string_call_result_120724))
    
    # Assigning a Call to a Subscript (line 377):
    
    # Assigning a Call to a Subscript (line 377):
    
    # Call to _read_string(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'f' (line 377)
    f_120728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 41), 'f', False)
    # Processing the call keyword arguments (line 377)
    kwargs_120729 = {}
    # Getting the type of '_read_string' (line 377)
    _read_string_120727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 28), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 377)
    _read_string_call_result_120730 = invoke(stypy.reporting.localization.Localization(__file__, 377, 28), _read_string_120727, *[f_120728], **kwargs_120729)
    
    # Getting the type of 'record' (line 377)
    record_120731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'record')
    str_120732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 15), 'str', 'release')
    # Storing an element on a container (line 377)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 8), record_120731, (str_120732, _read_string_call_result_120730))
    # SSA branch for the else part of an if statement (line 372)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 379)
    record_120734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 379)
    getitem___120735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 9), record_120734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 379)
    subscript_call_result_120736 = invoke(stypy.reporting.localization.Localization(__file__, 379, 9), getitem___120735, str_120733)
    
    str_120737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 30), 'str', 'IDENTIFICATON')
    # Applying the binary operator '==' (line 379)
    result_eq_120738 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 9), '==', subscript_call_result_120736, str_120737)
    
    # Testing the type of an if condition (line 379)
    if_condition_120739 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 9), result_eq_120738)
    # Assigning a type to the variable 'if_condition_120739' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 9), 'if_condition_120739', if_condition_120739)
    # SSA begins for if statement (line 379)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 381):
    
    # Assigning a Call to a Subscript (line 381):
    
    # Call to _read_string(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'f' (line 381)
    f_120741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 40), 'f', False)
    # Processing the call keyword arguments (line 381)
    kwargs_120742 = {}
    # Getting the type of '_read_string' (line 381)
    _read_string_120740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 27), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 381)
    _read_string_call_result_120743 = invoke(stypy.reporting.localization.Localization(__file__, 381, 27), _read_string_120740, *[f_120741], **kwargs_120742)
    
    # Getting the type of 'record' (line 381)
    record_120744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'record')
    str_120745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 15), 'str', 'author')
    # Storing an element on a container (line 381)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 8), record_120744, (str_120745, _read_string_call_result_120743))
    
    # Assigning a Call to a Subscript (line 382):
    
    # Assigning a Call to a Subscript (line 382):
    
    # Call to _read_string(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'f' (line 382)
    f_120747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 39), 'f', False)
    # Processing the call keyword arguments (line 382)
    kwargs_120748 = {}
    # Getting the type of '_read_string' (line 382)
    _read_string_120746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 26), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 382)
    _read_string_call_result_120749 = invoke(stypy.reporting.localization.Localization(__file__, 382, 26), _read_string_120746, *[f_120747], **kwargs_120748)
    
    # Getting the type of 'record' (line 382)
    record_120750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'record')
    str_120751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 15), 'str', 'title')
    # Storing an element on a container (line 382)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 8), record_120750, (str_120751, _read_string_call_result_120749))
    
    # Assigning a Call to a Subscript (line 383):
    
    # Assigning a Call to a Subscript (line 383):
    
    # Call to _read_string(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'f' (line 383)
    f_120753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 40), 'f', False)
    # Processing the call keyword arguments (line 383)
    kwargs_120754 = {}
    # Getting the type of '_read_string' (line 383)
    _read_string_120752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 27), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 383)
    _read_string_call_result_120755 = invoke(stypy.reporting.localization.Localization(__file__, 383, 27), _read_string_120752, *[f_120753], **kwargs_120754)
    
    # Getting the type of 'record' (line 383)
    record_120756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'record')
    str_120757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 15), 'str', 'idcode')
    # Storing an element on a container (line 383)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), record_120756, (str_120757, _read_string_call_result_120755))
    # SSA branch for the else part of an if statement (line 379)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 385)
    record_120759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 385)
    getitem___120760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 9), record_120759, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 385)
    subscript_call_result_120761 = invoke(stypy.reporting.localization.Localization(__file__, 385, 9), getitem___120760, str_120758)
    
    str_120762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 30), 'str', 'NOTICE')
    # Applying the binary operator '==' (line 385)
    result_eq_120763 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 9), '==', subscript_call_result_120761, str_120762)
    
    # Testing the type of an if condition (line 385)
    if_condition_120764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 9), result_eq_120763)
    # Assigning a type to the variable 'if_condition_120764' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 9), 'if_condition_120764', if_condition_120764)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 387):
    
    # Assigning a Call to a Subscript (line 387):
    
    # Call to _read_string(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'f' (line 387)
    f_120766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 40), 'f', False)
    # Processing the call keyword arguments (line 387)
    kwargs_120767 = {}
    # Getting the type of '_read_string' (line 387)
    _read_string_120765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 27), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 387)
    _read_string_call_result_120768 = invoke(stypy.reporting.localization.Localization(__file__, 387, 27), _read_string_120765, *[f_120766], **kwargs_120767)
    
    # Getting the type of 'record' (line 387)
    record_120769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'record')
    str_120770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 15), 'str', 'notice')
    # Storing an element on a container (line 387)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 8), record_120769, (str_120770, _read_string_call_result_120768))
    # SSA branch for the else part of an if statement (line 385)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 389)
    record_120772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___120773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 9), record_120772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_120774 = invoke(stypy.reporting.localization.Localization(__file__, 389, 9), getitem___120773, str_120771)
    
    str_120775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 30), 'str', 'DESCRIPTION')
    # Applying the binary operator '==' (line 389)
    result_eq_120776 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 9), '==', subscript_call_result_120774, str_120775)
    
    # Testing the type of an if condition (line 389)
    if_condition_120777 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 9), result_eq_120776)
    # Assigning a type to the variable 'if_condition_120777' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 9), 'if_condition_120777', if_condition_120777)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 391):
    
    # Assigning a Call to a Subscript (line 391):
    
    # Call to _read_string_data(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'f' (line 391)
    f_120779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 50), 'f', False)
    # Processing the call keyword arguments (line 391)
    kwargs_120780 = {}
    # Getting the type of '_read_string_data' (line 391)
    _read_string_data_120778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 32), '_read_string_data', False)
    # Calling _read_string_data(args, kwargs) (line 391)
    _read_string_data_call_result_120781 = invoke(stypy.reporting.localization.Localization(__file__, 391, 32), _read_string_data_120778, *[f_120779], **kwargs_120780)
    
    # Getting the type of 'record' (line 391)
    record_120782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'record')
    str_120783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 15), 'str', 'description')
    # Storing an element on a container (line 391)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 8), record_120782, (str_120783, _read_string_data_call_result_120781))
    # SSA branch for the else part of an if statement (line 389)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 393)
    record_120785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___120786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 9), record_120785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_120787 = invoke(stypy.reporting.localization.Localization(__file__, 393, 9), getitem___120786, str_120784)
    
    str_120788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 30), 'str', 'HEAP_HEADER')
    # Applying the binary operator '==' (line 393)
    result_eq_120789 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 9), '==', subscript_call_result_120787, str_120788)
    
    # Testing the type of an if condition (line 393)
    if_condition_120790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 9), result_eq_120789)
    # Assigning a type to the variable 'if_condition_120790' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 9), 'if_condition_120790', if_condition_120790)
    # SSA begins for if statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 395):
    
    # Assigning a Call to a Subscript (line 395):
    
    # Call to _read_long(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'f' (line 395)
    f_120792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 39), 'f', False)
    # Processing the call keyword arguments (line 395)
    kwargs_120793 = {}
    # Getting the type of '_read_long' (line 395)
    _read_long_120791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 28), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 395)
    _read_long_call_result_120794 = invoke(stypy.reporting.localization.Localization(__file__, 395, 28), _read_long_120791, *[f_120792], **kwargs_120793)
    
    # Getting the type of 'record' (line 395)
    record_120795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'record')
    str_120796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 15), 'str', 'nvalues')
    # Storing an element on a container (line 395)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 8), record_120795, (str_120796, _read_long_call_result_120794))
    
    # Assigning a List to a Subscript (line 396):
    
    # Assigning a List to a Subscript (line 396):
    
    # Obtaining an instance of the builtin type 'list' (line 396)
    list_120797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 396)
    
    # Getting the type of 'record' (line 396)
    record_120798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'record')
    str_120799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 15), 'str', 'indices')
    # Storing an element on a container (line 396)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 8), record_120798, (str_120799, list_120797))
    
    
    # Call to range(...): (line 397)
    # Processing the call arguments (line 397)
    
    # Obtaining the type of the subscript
    str_120801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 30), 'str', 'nvalues')
    # Getting the type of 'record' (line 397)
    record_120802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 23), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 397)
    getitem___120803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 23), record_120802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 397)
    subscript_call_result_120804 = invoke(stypy.reporting.localization.Localization(__file__, 397, 23), getitem___120803, str_120801)
    
    # Processing the call keyword arguments (line 397)
    kwargs_120805 = {}
    # Getting the type of 'range' (line 397)
    range_120800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 17), 'range', False)
    # Calling range(args, kwargs) (line 397)
    range_call_result_120806 = invoke(stypy.reporting.localization.Localization(__file__, 397, 17), range_120800, *[subscript_call_result_120804], **kwargs_120805)
    
    # Testing the type of a for loop iterable (line 397)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 397, 8), range_call_result_120806)
    # Getting the type of the for loop variable (line 397)
    for_loop_var_120807 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 397, 8), range_call_result_120806)
    # Assigning a type to the variable 'i' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'i', for_loop_var_120807)
    # SSA begins for a for statement (line 397)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 398)
    # Processing the call arguments (line 398)
    
    # Call to _read_long(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'f' (line 398)
    f_120814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 48), 'f', False)
    # Processing the call keyword arguments (line 398)
    kwargs_120815 = {}
    # Getting the type of '_read_long' (line 398)
    _read_long_120813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 37), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 398)
    _read_long_call_result_120816 = invoke(stypy.reporting.localization.Localization(__file__, 398, 37), _read_long_120813, *[f_120814], **kwargs_120815)
    
    # Processing the call keyword arguments (line 398)
    kwargs_120817 = {}
    
    # Obtaining the type of the subscript
    str_120808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 19), 'str', 'indices')
    # Getting the type of 'record' (line 398)
    record_120809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___120810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), record_120809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_120811 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), getitem___120810, str_120808)
    
    # Obtaining the member 'append' of a type (line 398)
    append_120812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), subscript_call_result_120811, 'append')
    # Calling append(args, kwargs) (line 398)
    append_call_result_120818 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), append_120812, *[_read_long_call_result_120816], **kwargs_120817)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 393)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 400)
    record_120820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___120821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 9), record_120820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_120822 = invoke(stypy.reporting.localization.Localization(__file__, 400, 9), getitem___120821, str_120819)
    
    str_120823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 30), 'str', 'COMMONBLOCK')
    # Applying the binary operator '==' (line 400)
    result_eq_120824 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 9), '==', subscript_call_result_120822, str_120823)
    
    # Testing the type of an if condition (line 400)
    if_condition_120825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 9), result_eq_120824)
    # Assigning a type to the variable 'if_condition_120825' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 9), 'if_condition_120825', if_condition_120825)
    # SSA begins for if statement (line 400)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 402):
    
    # Assigning a Call to a Subscript (line 402):
    
    # Call to _read_long(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'f' (line 402)
    f_120827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 37), 'f', False)
    # Processing the call keyword arguments (line 402)
    kwargs_120828 = {}
    # Getting the type of '_read_long' (line 402)
    _read_long_120826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 26), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 402)
    _read_long_call_result_120829 = invoke(stypy.reporting.localization.Localization(__file__, 402, 26), _read_long_120826, *[f_120827], **kwargs_120828)
    
    # Getting the type of 'record' (line 402)
    record_120830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'record')
    str_120831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 15), 'str', 'nvars')
    # Storing an element on a container (line 402)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 8), record_120830, (str_120831, _read_long_call_result_120829))
    
    # Assigning a Call to a Subscript (line 403):
    
    # Assigning a Call to a Subscript (line 403):
    
    # Call to _read_string(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'f' (line 403)
    f_120833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 38), 'f', False)
    # Processing the call keyword arguments (line 403)
    kwargs_120834 = {}
    # Getting the type of '_read_string' (line 403)
    _read_string_120832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 25), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 403)
    _read_string_call_result_120835 = invoke(stypy.reporting.localization.Localization(__file__, 403, 25), _read_string_120832, *[f_120833], **kwargs_120834)
    
    # Getting the type of 'record' (line 403)
    record_120836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'record')
    str_120837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 15), 'str', 'name')
    # Storing an element on a container (line 403)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 8), record_120836, (str_120837, _read_string_call_result_120835))
    
    # Assigning a List to a Subscript (line 404):
    
    # Assigning a List to a Subscript (line 404):
    
    # Obtaining an instance of the builtin type 'list' (line 404)
    list_120838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 404)
    
    # Getting the type of 'record' (line 404)
    record_120839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'record')
    str_120840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 15), 'str', 'varnames')
    # Storing an element on a container (line 404)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 8), record_120839, (str_120840, list_120838))
    
    
    # Call to range(...): (line 405)
    # Processing the call arguments (line 405)
    
    # Obtaining the type of the subscript
    str_120842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 30), 'str', 'nvars')
    # Getting the type of 'record' (line 405)
    record_120843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 23), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___120844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 23), record_120843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_120845 = invoke(stypy.reporting.localization.Localization(__file__, 405, 23), getitem___120844, str_120842)
    
    # Processing the call keyword arguments (line 405)
    kwargs_120846 = {}
    # Getting the type of 'range' (line 405)
    range_120841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 17), 'range', False)
    # Calling range(args, kwargs) (line 405)
    range_call_result_120847 = invoke(stypy.reporting.localization.Localization(__file__, 405, 17), range_120841, *[subscript_call_result_120845], **kwargs_120846)
    
    # Testing the type of a for loop iterable (line 405)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 405, 8), range_call_result_120847)
    # Getting the type of the for loop variable (line 405)
    for_loop_var_120848 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 405, 8), range_call_result_120847)
    # Assigning a type to the variable 'i' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'i', for_loop_var_120848)
    # SSA begins for a for statement (line 405)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 406)
    # Processing the call arguments (line 406)
    
    # Call to _read_string(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'f' (line 406)
    f_120855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 51), 'f', False)
    # Processing the call keyword arguments (line 406)
    kwargs_120856 = {}
    # Getting the type of '_read_string' (line 406)
    _read_string_120854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 38), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 406)
    _read_string_call_result_120857 = invoke(stypy.reporting.localization.Localization(__file__, 406, 38), _read_string_120854, *[f_120855], **kwargs_120856)
    
    # Processing the call keyword arguments (line 406)
    kwargs_120858 = {}
    
    # Obtaining the type of the subscript
    str_120849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 19), 'str', 'varnames')
    # Getting the type of 'record' (line 406)
    record_120850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___120851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), record_120850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_120852 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), getitem___120851, str_120849)
    
    # Obtaining the member 'append' of a type (line 406)
    append_120853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), subscript_call_result_120852, 'append')
    # Calling append(args, kwargs) (line 406)
    append_call_result_120859 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), append_120853, *[_read_string_call_result_120857], **kwargs_120858)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 400)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 408)
    record_120861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 408)
    getitem___120862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 9), record_120861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 408)
    subscript_call_result_120863 = invoke(stypy.reporting.localization.Localization(__file__, 408, 9), getitem___120862, str_120860)
    
    str_120864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 30), 'str', 'END_MARKER')
    # Applying the binary operator '==' (line 408)
    result_eq_120865 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 9), '==', subscript_call_result_120863, str_120864)
    
    # Testing the type of an if condition (line 408)
    if_condition_120866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 9), result_eq_120865)
    # Assigning a type to the variable 'if_condition_120866' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 9), 'if_condition_120866', if_condition_120866)
    # SSA begins for if statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 410):
    
    # Assigning a Name to a Subscript (line 410):
    # Getting the type of 'True' (line 410)
    True_120867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 24), 'True')
    # Getting the type of 'record' (line 410)
    record_120868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'record')
    str_120869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 15), 'str', 'end')
    # Storing an element on a container (line 410)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 8), record_120868, (str_120869, True_120867))
    # SSA branch for the else part of an if statement (line 408)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 412)
    record_120871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___120872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 9), record_120871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_120873 = invoke(stypy.reporting.localization.Localization(__file__, 412, 9), getitem___120872, str_120870)
    
    str_120874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 30), 'str', 'UNKNOWN')
    # Applying the binary operator '==' (line 412)
    result_eq_120875 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 9), '==', subscript_call_result_120873, str_120874)
    
    # Testing the type of an if condition (line 412)
    if_condition_120876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 412, 9), result_eq_120875)
    # Assigning a type to the variable 'if_condition_120876' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 9), 'if_condition_120876', if_condition_120876)
    # SSA begins for if statement (line 412)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 414)
    # Processing the call arguments (line 414)
    str_120879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 22), 'str', 'Skipping UNKNOWN record')
    # Processing the call keyword arguments (line 414)
    kwargs_120880 = {}
    # Getting the type of 'warnings' (line 414)
    warnings_120877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 414)
    warn_120878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), warnings_120877, 'warn')
    # Calling warn(args, kwargs) (line 414)
    warn_call_result_120881 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), warn_120878, *[str_120879], **kwargs_120880)
    
    # SSA branch for the else part of an if statement (line 412)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_120882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 16), 'str', 'rectype')
    # Getting the type of 'record' (line 416)
    record_120883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 9), 'record')
    # Obtaining the member '__getitem__' of a type (line 416)
    getitem___120884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 9), record_120883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 416)
    subscript_call_result_120885 = invoke(stypy.reporting.localization.Localization(__file__, 416, 9), getitem___120884, str_120882)
    
    str_120886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 30), 'str', 'SYSTEM_VARIABLE')
    # Applying the binary operator '==' (line 416)
    result_eq_120887 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 9), '==', subscript_call_result_120885, str_120886)
    
    # Testing the type of an if condition (line 416)
    if_condition_120888 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 416, 9), result_eq_120887)
    # Assigning a type to the variable 'if_condition_120888' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 9), 'if_condition_120888', if_condition_120888)
    # SSA begins for if statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 418)
    # Processing the call arguments (line 418)
    str_120891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 22), 'str', 'Skipping SYSTEM_VARIABLE record')
    # Processing the call keyword arguments (line 418)
    kwargs_120892 = {}
    # Getting the type of 'warnings' (line 418)
    warnings_120889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 418)
    warn_120890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), warnings_120889, 'warn')
    # Calling warn(args, kwargs) (line 418)
    warn_call_result_120893 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), warn_120890, *[str_120891], **kwargs_120892)
    
    # SSA branch for the else part of an if statement (line 416)
    module_type_store.open_ssa_branch('else')
    
    # Call to Exception(...): (line 422)
    # Processing the call arguments (line 422)
    str_120895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 24), 'str', "record['rectype']=%s not implemented")
    
    # Obtaining the type of the subscript
    str_120896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 67), 'str', 'rectype')
    # Getting the type of 'record' (line 423)
    record_120897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 60), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___120898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 60), record_120897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_120899 = invoke(stypy.reporting.localization.Localization(__file__, 423, 60), getitem___120898, str_120896)
    
    # Applying the binary operator '%' (line 422)
    result_mod_120900 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 24), '%', str_120895, subscript_call_result_120899)
    
    # Processing the call keyword arguments (line 422)
    kwargs_120901 = {}
    # Getting the type of 'Exception' (line 422)
    Exception_120894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 422)
    Exception_call_result_120902 = invoke(stypy.reporting.localization.Localization(__file__, 422, 14), Exception_120894, *[result_mod_120900], **kwargs_120901)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 422, 8), Exception_call_result_120902, 'raise parameter', BaseException)
    # SSA join for if statement (line 416)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 412)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 408)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 400)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 393)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 379)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 372)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 365)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to seek(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'nextrec' (line 425)
    nextrec_120905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), 'nextrec', False)
    # Processing the call keyword arguments (line 425)
    kwargs_120906 = {}
    # Getting the type of 'f' (line 425)
    f_120903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'f', False)
    # Obtaining the member 'seek' of a type (line 425)
    seek_120904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 4), f_120903, 'seek')
    # Calling seek(args, kwargs) (line 425)
    seek_call_result_120907 = invoke(stypy.reporting.localization.Localization(__file__, 425, 4), seek_120904, *[nextrec_120905], **kwargs_120906)
    
    # Getting the type of 'record' (line 427)
    record_120908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 11), 'record')
    # Assigning a type to the variable 'stypy_return_type' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'stypy_return_type', record_120908)
    
    # ################# End of '_read_record(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_record' in the type store
    # Getting the type of 'stypy_return_type' (line 317)
    stypy_return_type_120909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120909)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_record'
    return stypy_return_type_120909

# Assigning a type to the variable '_read_record' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), '_read_record', _read_record)

@norecursion
def _read_typedesc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_typedesc'
    module_type_store = module_type_store.open_function_context('_read_typedesc', 430, 0, False)
    
    # Passed parameters checking function
    _read_typedesc.stypy_localization = localization
    _read_typedesc.stypy_type_of_self = None
    _read_typedesc.stypy_type_store = module_type_store
    _read_typedesc.stypy_function_name = '_read_typedesc'
    _read_typedesc.stypy_param_names_list = ['f']
    _read_typedesc.stypy_varargs_param_name = None
    _read_typedesc.stypy_kwargs_param_name = None
    _read_typedesc.stypy_call_defaults = defaults
    _read_typedesc.stypy_call_varargs = varargs
    _read_typedesc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_typedesc', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_typedesc', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_typedesc(...)' code ##################

    str_120910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 4), 'str', 'Function to read in a type descriptor')
    
    # Assigning a Dict to a Name (line 433):
    
    # Assigning a Dict to a Name (line 433):
    
    # Obtaining an instance of the builtin type 'dict' (line 433)
    dict_120911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 433)
    # Adding element type (key, value) (line 433)
    str_120912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 16), 'str', 'typecode')
    
    # Call to _read_long(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'f' (line 433)
    f_120914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 39), 'f', False)
    # Processing the call keyword arguments (line 433)
    kwargs_120915 = {}
    # Getting the type of '_read_long' (line 433)
    _read_long_120913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 433)
    _read_long_call_result_120916 = invoke(stypy.reporting.localization.Localization(__file__, 433, 28), _read_long_120913, *[f_120914], **kwargs_120915)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 15), dict_120911, (str_120912, _read_long_call_result_120916))
    # Adding element type (key, value) (line 433)
    str_120917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 43), 'str', 'varflags')
    
    # Call to _read_long(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'f' (line 433)
    f_120919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 66), 'f', False)
    # Processing the call keyword arguments (line 433)
    kwargs_120920 = {}
    # Getting the type of '_read_long' (line 433)
    _read_long_120918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 55), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 433)
    _read_long_call_result_120921 = invoke(stypy.reporting.localization.Localization(__file__, 433, 55), _read_long_120918, *[f_120919], **kwargs_120920)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 15), dict_120911, (str_120917, _read_long_call_result_120921))
    
    # Assigning a type to the variable 'typedesc' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'typedesc', dict_120911)
    
    
    
    # Obtaining the type of the subscript
    str_120922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 16), 'str', 'varflags')
    # Getting the type of 'typedesc' (line 435)
    typedesc_120923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 7), 'typedesc')
    # Obtaining the member '__getitem__' of a type (line 435)
    getitem___120924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 7), typedesc_120923, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 435)
    subscript_call_result_120925 = invoke(stypy.reporting.localization.Localization(__file__, 435, 7), getitem___120924, str_120922)
    
    int_120926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 30), 'int')
    # Applying the binary operator '&' (line 435)
    result_and__120927 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 7), '&', subscript_call_result_120925, int_120926)
    
    int_120928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 35), 'int')
    # Applying the binary operator '==' (line 435)
    result_eq_120929 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 7), '==', result_and__120927, int_120928)
    
    # Testing the type of an if condition (line 435)
    if_condition_120930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 4), result_eq_120929)
    # Assigning a type to the variable 'if_condition_120930' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'if_condition_120930', if_condition_120930)
    # SSA begins for if statement (line 435)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 436)
    # Processing the call arguments (line 436)
    str_120932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 24), 'str', 'System variables not implemented')
    # Processing the call keyword arguments (line 436)
    kwargs_120933 = {}
    # Getting the type of 'Exception' (line 436)
    Exception_120931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 436)
    Exception_call_result_120934 = invoke(stypy.reporting.localization.Localization(__file__, 436, 14), Exception_120931, *[str_120932], **kwargs_120933)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 436, 8), Exception_call_result_120934, 'raise parameter', BaseException)
    # SSA join for if statement (line 435)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Subscript (line 438):
    
    # Assigning a Compare to a Subscript (line 438):
    
    
    # Obtaining the type of the subscript
    str_120935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 33), 'str', 'varflags')
    # Getting the type of 'typedesc' (line 438)
    typedesc_120936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'typedesc')
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___120937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 24), typedesc_120936, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_120938 = invoke(stypy.reporting.localization.Localization(__file__, 438, 24), getitem___120937, str_120935)
    
    int_120939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 47), 'int')
    # Applying the binary operator '&' (line 438)
    result_and__120940 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 24), '&', subscript_call_result_120938, int_120939)
    
    int_120941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 52), 'int')
    # Applying the binary operator '==' (line 438)
    result_eq_120942 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 24), '==', result_and__120940, int_120941)
    
    # Getting the type of 'typedesc' (line 438)
    typedesc_120943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'typedesc')
    str_120944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 13), 'str', 'array')
    # Storing an element on a container (line 438)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 4), typedesc_120943, (str_120944, result_eq_120942))
    
    # Assigning a Compare to a Subscript (line 439):
    
    # Assigning a Compare to a Subscript (line 439):
    
    
    # Obtaining the type of the subscript
    str_120945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 37), 'str', 'varflags')
    # Getting the type of 'typedesc' (line 439)
    typedesc_120946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'typedesc')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___120947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 28), typedesc_120946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_120948 = invoke(stypy.reporting.localization.Localization(__file__, 439, 28), getitem___120947, str_120945)
    
    int_120949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 51), 'int')
    # Applying the binary operator '&' (line 439)
    result_and__120950 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 28), '&', subscript_call_result_120948, int_120949)
    
    int_120951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 57), 'int')
    # Applying the binary operator '==' (line 439)
    result_eq_120952 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 28), '==', result_and__120950, int_120951)
    
    # Getting the type of 'typedesc' (line 439)
    typedesc_120953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'typedesc')
    str_120954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 13), 'str', 'structure')
    # Storing an element on a container (line 439)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 4), typedesc_120953, (str_120954, result_eq_120952))
    
    
    # Obtaining the type of the subscript
    str_120955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 16), 'str', 'structure')
    # Getting the type of 'typedesc' (line 441)
    typedesc_120956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 7), 'typedesc')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___120957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 7), typedesc_120956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_120958 = invoke(stypy.reporting.localization.Localization(__file__, 441, 7), getitem___120957, str_120955)
    
    # Testing the type of an if condition (line 441)
    if_condition_120959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 4), subscript_call_result_120958)
    # Assigning a type to the variable 'if_condition_120959' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'if_condition_120959', if_condition_120959)
    # SSA begins for if statement (line 441)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 442):
    
    # Assigning a Call to a Subscript (line 442):
    
    # Call to _read_arraydesc(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'f' (line 442)
    f_120961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 49), 'f', False)
    # Processing the call keyword arguments (line 442)
    kwargs_120962 = {}
    # Getting the type of '_read_arraydesc' (line 442)
    _read_arraydesc_120960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 33), '_read_arraydesc', False)
    # Calling _read_arraydesc(args, kwargs) (line 442)
    _read_arraydesc_call_result_120963 = invoke(stypy.reporting.localization.Localization(__file__, 442, 33), _read_arraydesc_120960, *[f_120961], **kwargs_120962)
    
    # Getting the type of 'typedesc' (line 442)
    typedesc_120964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'typedesc')
    str_120965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 17), 'str', 'array_desc')
    # Storing an element on a container (line 442)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 8), typedesc_120964, (str_120965, _read_arraydesc_call_result_120963))
    
    # Assigning a Call to a Subscript (line 443):
    
    # Assigning a Call to a Subscript (line 443):
    
    # Call to _read_structdesc(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'f' (line 443)
    f_120967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 51), 'f', False)
    # Processing the call keyword arguments (line 443)
    kwargs_120968 = {}
    # Getting the type of '_read_structdesc' (line 443)
    _read_structdesc_120966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 34), '_read_structdesc', False)
    # Calling _read_structdesc(args, kwargs) (line 443)
    _read_structdesc_call_result_120969 = invoke(stypy.reporting.localization.Localization(__file__, 443, 34), _read_structdesc_120966, *[f_120967], **kwargs_120968)
    
    # Getting the type of 'typedesc' (line 443)
    typedesc_120970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'typedesc')
    str_120971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 17), 'str', 'struct_desc')
    # Storing an element on a container (line 443)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 8), typedesc_120970, (str_120971, _read_structdesc_call_result_120969))
    # SSA branch for the else part of an if statement (line 441)
    module_type_store.open_ssa_branch('else')
    
    
    # Obtaining the type of the subscript
    str_120972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 18), 'str', 'array')
    # Getting the type of 'typedesc' (line 444)
    typedesc_120973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 9), 'typedesc')
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___120974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 9), typedesc_120973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_120975 = invoke(stypy.reporting.localization.Localization(__file__, 444, 9), getitem___120974, str_120972)
    
    # Testing the type of an if condition (line 444)
    if_condition_120976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 9), subscript_call_result_120975)
    # Assigning a type to the variable 'if_condition_120976' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 9), 'if_condition_120976', if_condition_120976)
    # SSA begins for if statement (line 444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 445):
    
    # Assigning a Call to a Subscript (line 445):
    
    # Call to _read_arraydesc(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'f' (line 445)
    f_120978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 49), 'f', False)
    # Processing the call keyword arguments (line 445)
    kwargs_120979 = {}
    # Getting the type of '_read_arraydesc' (line 445)
    _read_arraydesc_120977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 33), '_read_arraydesc', False)
    # Calling _read_arraydesc(args, kwargs) (line 445)
    _read_arraydesc_call_result_120980 = invoke(stypy.reporting.localization.Localization(__file__, 445, 33), _read_arraydesc_120977, *[f_120978], **kwargs_120979)
    
    # Getting the type of 'typedesc' (line 445)
    typedesc_120981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'typedesc')
    str_120982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 17), 'str', 'array_desc')
    # Storing an element on a container (line 445)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 8), typedesc_120981, (str_120982, _read_arraydesc_call_result_120980))
    # SSA join for if statement (line 444)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 441)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'typedesc' (line 447)
    typedesc_120983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'typedesc')
    # Assigning a type to the variable 'stypy_return_type' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type', typedesc_120983)
    
    # ################# End of '_read_typedesc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_typedesc' in the type store
    # Getting the type of 'stypy_return_type' (line 430)
    stypy_return_type_120984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120984)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_typedesc'
    return stypy_return_type_120984

# Assigning a type to the variable '_read_typedesc' (line 430)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), '_read_typedesc', _read_typedesc)

@norecursion
def _read_arraydesc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_arraydesc'
    module_type_store = module_type_store.open_function_context('_read_arraydesc', 450, 0, False)
    
    # Passed parameters checking function
    _read_arraydesc.stypy_localization = localization
    _read_arraydesc.stypy_type_of_self = None
    _read_arraydesc.stypy_type_store = module_type_store
    _read_arraydesc.stypy_function_name = '_read_arraydesc'
    _read_arraydesc.stypy_param_names_list = ['f']
    _read_arraydesc.stypy_varargs_param_name = None
    _read_arraydesc.stypy_kwargs_param_name = None
    _read_arraydesc.stypy_call_defaults = defaults
    _read_arraydesc.stypy_call_varargs = varargs
    _read_arraydesc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_arraydesc', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_arraydesc', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_arraydesc(...)' code ##################

    str_120985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 4), 'str', 'Function to read in an array descriptor')
    
    # Assigning a Dict to a Name (line 453):
    
    # Assigning a Dict to a Name (line 453):
    
    # Obtaining an instance of the builtin type 'dict' (line 453)
    dict_120986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 453)
    # Adding element type (key, value) (line 453)
    str_120987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 17), 'str', 'arrstart')
    
    # Call to _read_long(...): (line 453)
    # Processing the call arguments (line 453)
    # Getting the type of 'f' (line 453)
    f_120989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 40), 'f', False)
    # Processing the call keyword arguments (line 453)
    kwargs_120990 = {}
    # Getting the type of '_read_long' (line 453)
    _read_long_120988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 29), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 453)
    _read_long_call_result_120991 = invoke(stypy.reporting.localization.Localization(__file__, 453, 29), _read_long_120988, *[f_120989], **kwargs_120990)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 16), dict_120986, (str_120987, _read_long_call_result_120991))
    
    # Assigning a type to the variable 'arraydesc' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'arraydesc', dict_120986)
    
    
    
    # Obtaining the type of the subscript
    str_120992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 17), 'str', 'arrstart')
    # Getting the type of 'arraydesc' (line 455)
    arraydesc_120993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 7), 'arraydesc')
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___120994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 7), arraydesc_120993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_120995 = invoke(stypy.reporting.localization.Localization(__file__, 455, 7), getitem___120994, str_120992)
    
    int_120996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 32), 'int')
    # Applying the binary operator '==' (line 455)
    result_eq_120997 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 7), '==', subscript_call_result_120995, int_120996)
    
    # Testing the type of an if condition (line 455)
    if_condition_120998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 4), result_eq_120997)
    # Assigning a type to the variable 'if_condition_120998' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'if_condition_120998', if_condition_120998)
    # SSA begins for if statement (line 455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _skip_bytes(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'f' (line 457)
    f_121000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 20), 'f', False)
    int_121001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 23), 'int')
    # Processing the call keyword arguments (line 457)
    kwargs_121002 = {}
    # Getting the type of '_skip_bytes' (line 457)
    _skip_bytes_120999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), '_skip_bytes', False)
    # Calling _skip_bytes(args, kwargs) (line 457)
    _skip_bytes_call_result_121003 = invoke(stypy.reporting.localization.Localization(__file__, 457, 8), _skip_bytes_120999, *[f_121000, int_121001], **kwargs_121002)
    
    
    # Assigning a Call to a Subscript (line 459):
    
    # Assigning a Call to a Subscript (line 459):
    
    # Call to _read_long(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'f' (line 459)
    f_121005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 41), 'f', False)
    # Processing the call keyword arguments (line 459)
    kwargs_121006 = {}
    # Getting the type of '_read_long' (line 459)
    _read_long_121004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 30), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 459)
    _read_long_call_result_121007 = invoke(stypy.reporting.localization.Localization(__file__, 459, 30), _read_long_121004, *[f_121005], **kwargs_121006)
    
    # Getting the type of 'arraydesc' (line 459)
    arraydesc_121008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'arraydesc')
    str_121009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 18), 'str', 'nbytes')
    # Storing an element on a container (line 459)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 8), arraydesc_121008, (str_121009, _read_long_call_result_121007))
    
    # Assigning a Call to a Subscript (line 460):
    
    # Assigning a Call to a Subscript (line 460):
    
    # Call to _read_long(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'f' (line 460)
    f_121011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 44), 'f', False)
    # Processing the call keyword arguments (line 460)
    kwargs_121012 = {}
    # Getting the type of '_read_long' (line 460)
    _read_long_121010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 33), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 460)
    _read_long_call_result_121013 = invoke(stypy.reporting.localization.Localization(__file__, 460, 33), _read_long_121010, *[f_121011], **kwargs_121012)
    
    # Getting the type of 'arraydesc' (line 460)
    arraydesc_121014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'arraydesc')
    str_121015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 18), 'str', 'nelements')
    # Storing an element on a container (line 460)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 8), arraydesc_121014, (str_121015, _read_long_call_result_121013))
    
    # Assigning a Call to a Subscript (line 461):
    
    # Assigning a Call to a Subscript (line 461):
    
    # Call to _read_long(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'f' (line 461)
    f_121017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 40), 'f', False)
    # Processing the call keyword arguments (line 461)
    kwargs_121018 = {}
    # Getting the type of '_read_long' (line 461)
    _read_long_121016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 29), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 461)
    _read_long_call_result_121019 = invoke(stypy.reporting.localization.Localization(__file__, 461, 29), _read_long_121016, *[f_121017], **kwargs_121018)
    
    # Getting the type of 'arraydesc' (line 461)
    arraydesc_121020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'arraydesc')
    str_121021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 18), 'str', 'ndims')
    # Storing an element on a container (line 461)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 8), arraydesc_121020, (str_121021, _read_long_call_result_121019))
    
    # Call to _skip_bytes(...): (line 463)
    # Processing the call arguments (line 463)
    # Getting the type of 'f' (line 463)
    f_121023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 20), 'f', False)
    int_121024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 23), 'int')
    # Processing the call keyword arguments (line 463)
    kwargs_121025 = {}
    # Getting the type of '_skip_bytes' (line 463)
    _skip_bytes_121022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), '_skip_bytes', False)
    # Calling _skip_bytes(args, kwargs) (line 463)
    _skip_bytes_call_result_121026 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), _skip_bytes_121022, *[f_121023, int_121024], **kwargs_121025)
    
    
    # Assigning a Call to a Subscript (line 465):
    
    # Assigning a Call to a Subscript (line 465):
    
    # Call to _read_long(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'f' (line 465)
    f_121028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 39), 'f', False)
    # Processing the call keyword arguments (line 465)
    kwargs_121029 = {}
    # Getting the type of '_read_long' (line 465)
    _read_long_121027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 28), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 465)
    _read_long_call_result_121030 = invoke(stypy.reporting.localization.Localization(__file__, 465, 28), _read_long_121027, *[f_121028], **kwargs_121029)
    
    # Getting the type of 'arraydesc' (line 465)
    arraydesc_121031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'arraydesc')
    str_121032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 18), 'str', 'nmax')
    # Storing an element on a container (line 465)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 8), arraydesc_121031, (str_121032, _read_long_call_result_121030))
    
    # Assigning a List to a Subscript (line 467):
    
    # Assigning a List to a Subscript (line 467):
    
    # Obtaining an instance of the builtin type 'list' (line 467)
    list_121033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 467)
    
    # Getting the type of 'arraydesc' (line 467)
    arraydesc_121034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'arraydesc')
    str_121035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 18), 'str', 'dims')
    # Storing an element on a container (line 467)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 8), arraydesc_121034, (str_121035, list_121033))
    
    
    # Call to range(...): (line 468)
    # Processing the call arguments (line 468)
    
    # Obtaining the type of the subscript
    str_121037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 33), 'str', 'nmax')
    # Getting the type of 'arraydesc' (line 468)
    arraydesc_121038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 23), 'arraydesc', False)
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___121039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 23), arraydesc_121038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_121040 = invoke(stypy.reporting.localization.Localization(__file__, 468, 23), getitem___121039, str_121037)
    
    # Processing the call keyword arguments (line 468)
    kwargs_121041 = {}
    # Getting the type of 'range' (line 468)
    range_121036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 17), 'range', False)
    # Calling range(args, kwargs) (line 468)
    range_call_result_121042 = invoke(stypy.reporting.localization.Localization(__file__, 468, 17), range_121036, *[subscript_call_result_121040], **kwargs_121041)
    
    # Testing the type of a for loop iterable (line 468)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 468, 8), range_call_result_121042)
    # Getting the type of the for loop variable (line 468)
    for_loop_var_121043 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 468, 8), range_call_result_121042)
    # Assigning a type to the variable 'd' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'd', for_loop_var_121043)
    # SSA begins for a for statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 469)
    # Processing the call arguments (line 469)
    
    # Call to _read_long(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'f' (line 469)
    f_121050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 48), 'f', False)
    # Processing the call keyword arguments (line 469)
    kwargs_121051 = {}
    # Getting the type of '_read_long' (line 469)
    _read_long_121049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 37), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 469)
    _read_long_call_result_121052 = invoke(stypy.reporting.localization.Localization(__file__, 469, 37), _read_long_121049, *[f_121050], **kwargs_121051)
    
    # Processing the call keyword arguments (line 469)
    kwargs_121053 = {}
    
    # Obtaining the type of the subscript
    str_121044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 22), 'str', 'dims')
    # Getting the type of 'arraydesc' (line 469)
    arraydesc_121045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'arraydesc', False)
    # Obtaining the member '__getitem__' of a type (line 469)
    getitem___121046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 12), arraydesc_121045, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 469)
    subscript_call_result_121047 = invoke(stypy.reporting.localization.Localization(__file__, 469, 12), getitem___121046, str_121044)
    
    # Obtaining the member 'append' of a type (line 469)
    append_121048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 12), subscript_call_result_121047, 'append')
    # Calling append(args, kwargs) (line 469)
    append_call_result_121054 = invoke(stypy.reporting.localization.Localization(__file__, 469, 12), append_121048, *[_read_long_call_result_121052], **kwargs_121053)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 455)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    str_121055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 19), 'str', 'arrstart')
    # Getting the type of 'arraydesc' (line 471)
    arraydesc_121056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 9), 'arraydesc')
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___121057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 9), arraydesc_121056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_121058 = invoke(stypy.reporting.localization.Localization(__file__, 471, 9), getitem___121057, str_121055)
    
    int_121059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 34), 'int')
    # Applying the binary operator '==' (line 471)
    result_eq_121060 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 9), '==', subscript_call_result_121058, int_121059)
    
    # Testing the type of an if condition (line 471)
    if_condition_121061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 9), result_eq_121060)
    # Assigning a type to the variable 'if_condition_121061' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 9), 'if_condition_121061', if_condition_121061)
    # SSA begins for if statement (line 471)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 473)
    # Processing the call arguments (line 473)
    str_121064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 22), 'str', 'Using experimental 64-bit array read')
    # Processing the call keyword arguments (line 473)
    kwargs_121065 = {}
    # Getting the type of 'warnings' (line 473)
    warnings_121062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 473)
    warn_121063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), warnings_121062, 'warn')
    # Calling warn(args, kwargs) (line 473)
    warn_call_result_121066 = invoke(stypy.reporting.localization.Localization(__file__, 473, 8), warn_121063, *[str_121064], **kwargs_121065)
    
    
    # Call to _skip_bytes(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'f' (line 475)
    f_121068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'f', False)
    int_121069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 23), 'int')
    # Processing the call keyword arguments (line 475)
    kwargs_121070 = {}
    # Getting the type of '_skip_bytes' (line 475)
    _skip_bytes_121067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), '_skip_bytes', False)
    # Calling _skip_bytes(args, kwargs) (line 475)
    _skip_bytes_call_result_121071 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), _skip_bytes_121067, *[f_121068, int_121069], **kwargs_121070)
    
    
    # Assigning a Call to a Subscript (line 477):
    
    # Assigning a Call to a Subscript (line 477):
    
    # Call to _read_uint64(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'f' (line 477)
    f_121073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 43), 'f', False)
    # Processing the call keyword arguments (line 477)
    kwargs_121074 = {}
    # Getting the type of '_read_uint64' (line 477)
    _read_uint64_121072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 30), '_read_uint64', False)
    # Calling _read_uint64(args, kwargs) (line 477)
    _read_uint64_call_result_121075 = invoke(stypy.reporting.localization.Localization(__file__, 477, 30), _read_uint64_121072, *[f_121073], **kwargs_121074)
    
    # Getting the type of 'arraydesc' (line 477)
    arraydesc_121076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'arraydesc')
    str_121077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 18), 'str', 'nbytes')
    # Storing an element on a container (line 477)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 8), arraydesc_121076, (str_121077, _read_uint64_call_result_121075))
    
    # Assigning a Call to a Subscript (line 478):
    
    # Assigning a Call to a Subscript (line 478):
    
    # Call to _read_uint64(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'f' (line 478)
    f_121079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 46), 'f', False)
    # Processing the call keyword arguments (line 478)
    kwargs_121080 = {}
    # Getting the type of '_read_uint64' (line 478)
    _read_uint64_121078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 33), '_read_uint64', False)
    # Calling _read_uint64(args, kwargs) (line 478)
    _read_uint64_call_result_121081 = invoke(stypy.reporting.localization.Localization(__file__, 478, 33), _read_uint64_121078, *[f_121079], **kwargs_121080)
    
    # Getting the type of 'arraydesc' (line 478)
    arraydesc_121082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'arraydesc')
    str_121083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 18), 'str', 'nelements')
    # Storing an element on a container (line 478)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 8), arraydesc_121082, (str_121083, _read_uint64_call_result_121081))
    
    # Assigning a Call to a Subscript (line 479):
    
    # Assigning a Call to a Subscript (line 479):
    
    # Call to _read_long(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'f' (line 479)
    f_121085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 40), 'f', False)
    # Processing the call keyword arguments (line 479)
    kwargs_121086 = {}
    # Getting the type of '_read_long' (line 479)
    _read_long_121084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 29), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 479)
    _read_long_call_result_121087 = invoke(stypy.reporting.localization.Localization(__file__, 479, 29), _read_long_121084, *[f_121085], **kwargs_121086)
    
    # Getting the type of 'arraydesc' (line 479)
    arraydesc_121088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'arraydesc')
    str_121089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 18), 'str', 'ndims')
    # Storing an element on a container (line 479)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 8), arraydesc_121088, (str_121089, _read_long_call_result_121087))
    
    # Call to _skip_bytes(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'f' (line 481)
    f_121091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 20), 'f', False)
    int_121092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 23), 'int')
    # Processing the call keyword arguments (line 481)
    kwargs_121093 = {}
    # Getting the type of '_skip_bytes' (line 481)
    _skip_bytes_121090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), '_skip_bytes', False)
    # Calling _skip_bytes(args, kwargs) (line 481)
    _skip_bytes_call_result_121094 = invoke(stypy.reporting.localization.Localization(__file__, 481, 8), _skip_bytes_121090, *[f_121091, int_121092], **kwargs_121093)
    
    
    # Assigning a Num to a Subscript (line 483):
    
    # Assigning a Num to a Subscript (line 483):
    int_121095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 28), 'int')
    # Getting the type of 'arraydesc' (line 483)
    arraydesc_121096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'arraydesc')
    str_121097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 18), 'str', 'nmax')
    # Storing an element on a container (line 483)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 8), arraydesc_121096, (str_121097, int_121095))
    
    # Assigning a List to a Subscript (line 485):
    
    # Assigning a List to a Subscript (line 485):
    
    # Obtaining an instance of the builtin type 'list' (line 485)
    list_121098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 485)
    
    # Getting the type of 'arraydesc' (line 485)
    arraydesc_121099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'arraydesc')
    str_121100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 18), 'str', 'dims')
    # Storing an element on a container (line 485)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 8), arraydesc_121099, (str_121100, list_121098))
    
    
    # Call to range(...): (line 486)
    # Processing the call arguments (line 486)
    
    # Obtaining the type of the subscript
    str_121102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 33), 'str', 'nmax')
    # Getting the type of 'arraydesc' (line 486)
    arraydesc_121103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 23), 'arraydesc', False)
    # Obtaining the member '__getitem__' of a type (line 486)
    getitem___121104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 23), arraydesc_121103, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 486)
    subscript_call_result_121105 = invoke(stypy.reporting.localization.Localization(__file__, 486, 23), getitem___121104, str_121102)
    
    # Processing the call keyword arguments (line 486)
    kwargs_121106 = {}
    # Getting the type of 'range' (line 486)
    range_121101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 17), 'range', False)
    # Calling range(args, kwargs) (line 486)
    range_call_result_121107 = invoke(stypy.reporting.localization.Localization(__file__, 486, 17), range_121101, *[subscript_call_result_121105], **kwargs_121106)
    
    # Testing the type of a for loop iterable (line 486)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 486, 8), range_call_result_121107)
    # Getting the type of the for loop variable (line 486)
    for_loop_var_121108 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 486, 8), range_call_result_121107)
    # Assigning a type to the variable 'd' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'd', for_loop_var_121108)
    # SSA begins for a for statement (line 486)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 487):
    
    # Assigning a Call to a Name (line 487):
    
    # Call to _read_long(...): (line 487)
    # Processing the call arguments (line 487)
    # Getting the type of 'f' (line 487)
    f_121110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 27), 'f', False)
    # Processing the call keyword arguments (line 487)
    kwargs_121111 = {}
    # Getting the type of '_read_long' (line 487)
    _read_long_121109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 487)
    _read_long_call_result_121112 = invoke(stypy.reporting.localization.Localization(__file__, 487, 16), _read_long_121109, *[f_121110], **kwargs_121111)
    
    # Assigning a type to the variable 'v' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'v', _read_long_call_result_121112)
    
    
    # Getting the type of 'v' (line 488)
    v_121113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 15), 'v')
    int_121114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 20), 'int')
    # Applying the binary operator '!=' (line 488)
    result_ne_121115 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 15), '!=', v_121113, int_121114)
    
    # Testing the type of an if condition (line 488)
    if_condition_121116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 12), result_ne_121115)
    # Assigning a type to the variable 'if_condition_121116' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'if_condition_121116', if_condition_121116)
    # SSA begins for if statement (line 488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 489)
    # Processing the call arguments (line 489)
    str_121118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 32), 'str', 'Expected a zero in ARRAY_DESC')
    # Processing the call keyword arguments (line 489)
    kwargs_121119 = {}
    # Getting the type of 'Exception' (line 489)
    Exception_121117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 22), 'Exception', False)
    # Calling Exception(args, kwargs) (line 489)
    Exception_call_result_121120 = invoke(stypy.reporting.localization.Localization(__file__, 489, 22), Exception_121117, *[str_121118], **kwargs_121119)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 489, 16), Exception_call_result_121120, 'raise parameter', BaseException)
    # SSA join for if statement (line 488)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 490)
    # Processing the call arguments (line 490)
    
    # Call to _read_long(...): (line 490)
    # Processing the call arguments (line 490)
    # Getting the type of 'f' (line 490)
    f_121127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 48), 'f', False)
    # Processing the call keyword arguments (line 490)
    kwargs_121128 = {}
    # Getting the type of '_read_long' (line 490)
    _read_long_121126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 37), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 490)
    _read_long_call_result_121129 = invoke(stypy.reporting.localization.Localization(__file__, 490, 37), _read_long_121126, *[f_121127], **kwargs_121128)
    
    # Processing the call keyword arguments (line 490)
    kwargs_121130 = {}
    
    # Obtaining the type of the subscript
    str_121121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 22), 'str', 'dims')
    # Getting the type of 'arraydesc' (line 490)
    arraydesc_121122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'arraydesc', False)
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___121123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), arraydesc_121122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_121124 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), getitem___121123, str_121121)
    
    # Obtaining the member 'append' of a type (line 490)
    append_121125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), subscript_call_result_121124, 'append')
    # Calling append(args, kwargs) (line 490)
    append_call_result_121131 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), append_121125, *[_read_long_call_result_121129], **kwargs_121130)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 471)
    module_type_store.open_ssa_branch('else')
    
    # Call to Exception(...): (line 494)
    # Processing the call arguments (line 494)
    str_121133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 24), 'str', 'Unknown ARRSTART: %i')
    
    # Obtaining the type of the subscript
    str_121134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 59), 'str', 'arrstart')
    # Getting the type of 'arraydesc' (line 494)
    arraydesc_121135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 49), 'arraydesc', False)
    # Obtaining the member '__getitem__' of a type (line 494)
    getitem___121136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 49), arraydesc_121135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 494)
    subscript_call_result_121137 = invoke(stypy.reporting.localization.Localization(__file__, 494, 49), getitem___121136, str_121134)
    
    # Applying the binary operator '%' (line 494)
    result_mod_121138 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 24), '%', str_121133, subscript_call_result_121137)
    
    # Processing the call keyword arguments (line 494)
    kwargs_121139 = {}
    # Getting the type of 'Exception' (line 494)
    Exception_121132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 494)
    Exception_call_result_121140 = invoke(stypy.reporting.localization.Localization(__file__, 494, 14), Exception_121132, *[result_mod_121138], **kwargs_121139)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 494, 8), Exception_call_result_121140, 'raise parameter', BaseException)
    # SSA join for if statement (line 471)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 455)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'arraydesc' (line 496)
    arraydesc_121141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 11), 'arraydesc')
    # Assigning a type to the variable 'stypy_return_type' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'stypy_return_type', arraydesc_121141)
    
    # ################# End of '_read_arraydesc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_arraydesc' in the type store
    # Getting the type of 'stypy_return_type' (line 450)
    stypy_return_type_121142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121142)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_arraydesc'
    return stypy_return_type_121142

# Assigning a type to the variable '_read_arraydesc' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), '_read_arraydesc', _read_arraydesc)

@norecursion
def _read_structdesc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_structdesc'
    module_type_store = module_type_store.open_function_context('_read_structdesc', 499, 0, False)
    
    # Passed parameters checking function
    _read_structdesc.stypy_localization = localization
    _read_structdesc.stypy_type_of_self = None
    _read_structdesc.stypy_type_store = module_type_store
    _read_structdesc.stypy_function_name = '_read_structdesc'
    _read_structdesc.stypy_param_names_list = ['f']
    _read_structdesc.stypy_varargs_param_name = None
    _read_structdesc.stypy_kwargs_param_name = None
    _read_structdesc.stypy_call_defaults = defaults
    _read_structdesc.stypy_call_varargs = varargs
    _read_structdesc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_structdesc', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_structdesc', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_structdesc(...)' code ##################

    str_121143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 4), 'str', 'Function to read in a structure descriptor')
    
    # Assigning a Dict to a Name (line 502):
    
    # Assigning a Dict to a Name (line 502):
    
    # Obtaining an instance of the builtin type 'dict' (line 502)
    dict_121144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 502)
    
    # Assigning a type to the variable 'structdesc' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'structdesc', dict_121144)
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to _read_long(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'f' (line 504)
    f_121146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 29), 'f', False)
    # Processing the call keyword arguments (line 504)
    kwargs_121147 = {}
    # Getting the type of '_read_long' (line 504)
    _read_long_121145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 18), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 504)
    _read_long_call_result_121148 = invoke(stypy.reporting.localization.Localization(__file__, 504, 18), _read_long_121145, *[f_121146], **kwargs_121147)
    
    # Assigning a type to the variable 'structstart' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'structstart', _read_long_call_result_121148)
    
    
    # Getting the type of 'structstart' (line 505)
    structstart_121149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 7), 'structstart')
    int_121150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 22), 'int')
    # Applying the binary operator '!=' (line 505)
    result_ne_121151 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), '!=', structstart_121149, int_121150)
    
    # Testing the type of an if condition (line 505)
    if_condition_121152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 4), result_ne_121151)
    # Assigning a type to the variable 'if_condition_121152' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'if_condition_121152', if_condition_121152)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 506)
    # Processing the call arguments (line 506)
    str_121154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 24), 'str', 'STRUCTSTART should be 9')
    # Processing the call keyword arguments (line 506)
    kwargs_121155 = {}
    # Getting the type of 'Exception' (line 506)
    Exception_121153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 506)
    Exception_call_result_121156 = invoke(stypy.reporting.localization.Localization(__file__, 506, 14), Exception_121153, *[str_121154], **kwargs_121155)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 506, 8), Exception_call_result_121156, 'raise parameter', BaseException)
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 508):
    
    # Assigning a Call to a Subscript (line 508):
    
    # Call to _read_string(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'f' (line 508)
    f_121158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 38), 'f', False)
    # Processing the call keyword arguments (line 508)
    kwargs_121159 = {}
    # Getting the type of '_read_string' (line 508)
    _read_string_121157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 25), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 508)
    _read_string_call_result_121160 = invoke(stypy.reporting.localization.Localization(__file__, 508, 25), _read_string_121157, *[f_121158], **kwargs_121159)
    
    # Getting the type of 'structdesc' (line 508)
    structdesc_121161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'structdesc')
    str_121162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 15), 'str', 'name')
    # Storing an element on a container (line 508)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 4), structdesc_121161, (str_121162, _read_string_call_result_121160))
    
    # Assigning a Call to a Name (line 509):
    
    # Assigning a Call to a Name (line 509):
    
    # Call to _read_long(...): (line 509)
    # Processing the call arguments (line 509)
    # Getting the type of 'f' (line 509)
    f_121164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 24), 'f', False)
    # Processing the call keyword arguments (line 509)
    kwargs_121165 = {}
    # Getting the type of '_read_long' (line 509)
    _read_long_121163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 13), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 509)
    _read_long_call_result_121166 = invoke(stypy.reporting.localization.Localization(__file__, 509, 13), _read_long_121163, *[f_121164], **kwargs_121165)
    
    # Assigning a type to the variable 'predef' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'predef', _read_long_call_result_121166)
    
    # Assigning a Call to a Subscript (line 510):
    
    # Assigning a Call to a Subscript (line 510):
    
    # Call to _read_long(...): (line 510)
    # Processing the call arguments (line 510)
    # Getting the type of 'f' (line 510)
    f_121168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'f', False)
    # Processing the call keyword arguments (line 510)
    kwargs_121169 = {}
    # Getting the type of '_read_long' (line 510)
    _read_long_121167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 510)
    _read_long_call_result_121170 = invoke(stypy.reporting.localization.Localization(__file__, 510, 26), _read_long_121167, *[f_121168], **kwargs_121169)
    
    # Getting the type of 'structdesc' (line 510)
    structdesc_121171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'structdesc')
    str_121172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 15), 'str', 'ntags')
    # Storing an element on a container (line 510)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 4), structdesc_121171, (str_121172, _read_long_call_result_121170))
    
    # Assigning a Call to a Subscript (line 511):
    
    # Assigning a Call to a Subscript (line 511):
    
    # Call to _read_long(...): (line 511)
    # Processing the call arguments (line 511)
    # Getting the type of 'f' (line 511)
    f_121174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 38), 'f', False)
    # Processing the call keyword arguments (line 511)
    kwargs_121175 = {}
    # Getting the type of '_read_long' (line 511)
    _read_long_121173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 27), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 511)
    _read_long_call_result_121176 = invoke(stypy.reporting.localization.Localization(__file__, 511, 27), _read_long_121173, *[f_121174], **kwargs_121175)
    
    # Getting the type of 'structdesc' (line 511)
    structdesc_121177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'structdesc')
    str_121178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 15), 'str', 'nbytes')
    # Storing an element on a container (line 511)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 4), structdesc_121177, (str_121178, _read_long_call_result_121176))
    
    # Assigning a BinOp to a Subscript (line 513):
    
    # Assigning a BinOp to a Subscript (line 513):
    # Getting the type of 'predef' (line 513)
    predef_121179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 27), 'predef')
    int_121180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 36), 'int')
    # Applying the binary operator '&' (line 513)
    result_and__121181 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 27), '&', predef_121179, int_121180)
    
    # Getting the type of 'structdesc' (line 513)
    structdesc_121182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'structdesc')
    str_121183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 15), 'str', 'predef')
    # Storing an element on a container (line 513)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 4), structdesc_121182, (str_121183, result_and__121181))
    
    # Assigning a BinOp to a Subscript (line 514):
    
    # Assigning a BinOp to a Subscript (line 514):
    # Getting the type of 'predef' (line 514)
    predef_121184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 29), 'predef')
    int_121185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 38), 'int')
    # Applying the binary operator '&' (line 514)
    result_and__121186 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 29), '&', predef_121184, int_121185)
    
    # Getting the type of 'structdesc' (line 514)
    structdesc_121187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'structdesc')
    str_121188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 15), 'str', 'inherits')
    # Storing an element on a container (line 514)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 4), structdesc_121187, (str_121188, result_and__121186))
    
    # Assigning a BinOp to a Subscript (line 515):
    
    # Assigning a BinOp to a Subscript (line 515):
    # Getting the type of 'predef' (line 515)
    predef_121189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 29), 'predef')
    int_121190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 38), 'int')
    # Applying the binary operator '&' (line 515)
    result_and__121191 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 29), '&', predef_121189, int_121190)
    
    # Getting the type of 'structdesc' (line 515)
    structdesc_121192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'structdesc')
    str_121193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 15), 'str', 'is_super')
    # Storing an element on a container (line 515)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 4), structdesc_121192, (str_121193, result_and__121191))
    
    
    
    # Obtaining the type of the subscript
    str_121194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 22), 'str', 'predef')
    # Getting the type of 'structdesc' (line 517)
    structdesc_121195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___121196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 11), structdesc_121195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_121197 = invoke(stypy.reporting.localization.Localization(__file__, 517, 11), getitem___121196, str_121194)
    
    # Applying the 'not' unary operator (line 517)
    result_not__121198 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), 'not', subscript_call_result_121197)
    
    # Testing the type of an if condition (line 517)
    if_condition_121199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), result_not__121198)
    # Assigning a type to the variable 'if_condition_121199' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_121199', if_condition_121199)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 519):
    
    # Assigning a List to a Subscript (line 519):
    
    # Obtaining an instance of the builtin type 'list' (line 519)
    list_121200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 519)
    
    # Getting the type of 'structdesc' (line 519)
    structdesc_121201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'structdesc')
    str_121202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'str', 'tagtable')
    # Storing an element on a container (line 519)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 8), structdesc_121201, (str_121202, list_121200))
    
    
    # Call to range(...): (line 520)
    # Processing the call arguments (line 520)
    
    # Obtaining the type of the subscript
    str_121204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 34), 'str', 'ntags')
    # Getting the type of 'structdesc' (line 520)
    structdesc_121205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 23), 'structdesc', False)
    # Obtaining the member '__getitem__' of a type (line 520)
    getitem___121206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 23), structdesc_121205, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 520)
    subscript_call_result_121207 = invoke(stypy.reporting.localization.Localization(__file__, 520, 23), getitem___121206, str_121204)
    
    # Processing the call keyword arguments (line 520)
    kwargs_121208 = {}
    # Getting the type of 'range' (line 520)
    range_121203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'range', False)
    # Calling range(args, kwargs) (line 520)
    range_call_result_121209 = invoke(stypy.reporting.localization.Localization(__file__, 520, 17), range_121203, *[subscript_call_result_121207], **kwargs_121208)
    
    # Testing the type of a for loop iterable (line 520)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 520, 8), range_call_result_121209)
    # Getting the type of the for loop variable (line 520)
    for_loop_var_121210 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 520, 8), range_call_result_121209)
    # Assigning a type to the variable 't' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 't', for_loop_var_121210)
    # SSA begins for a for statement (line 520)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 521)
    # Processing the call arguments (line 521)
    
    # Call to _read_tagdesc(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 'f' (line 521)
    f_121217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 56), 'f', False)
    # Processing the call keyword arguments (line 521)
    kwargs_121218 = {}
    # Getting the type of '_read_tagdesc' (line 521)
    _read_tagdesc_121216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 42), '_read_tagdesc', False)
    # Calling _read_tagdesc(args, kwargs) (line 521)
    _read_tagdesc_call_result_121219 = invoke(stypy.reporting.localization.Localization(__file__, 521, 42), _read_tagdesc_121216, *[f_121217], **kwargs_121218)
    
    # Processing the call keyword arguments (line 521)
    kwargs_121220 = {}
    
    # Obtaining the type of the subscript
    str_121211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 23), 'str', 'tagtable')
    # Getting the type of 'structdesc' (line 521)
    structdesc_121212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'structdesc', False)
    # Obtaining the member '__getitem__' of a type (line 521)
    getitem___121213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 12), structdesc_121212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 521)
    subscript_call_result_121214 = invoke(stypy.reporting.localization.Localization(__file__, 521, 12), getitem___121213, str_121211)
    
    # Obtaining the member 'append' of a type (line 521)
    append_121215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 12), subscript_call_result_121214, 'append')
    # Calling append(args, kwargs) (line 521)
    append_call_result_121221 = invoke(stypy.reporting.localization.Localization(__file__, 521, 12), append_121215, *[_read_tagdesc_call_result_121219], **kwargs_121220)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_121222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 30), 'str', 'tagtable')
    # Getting the type of 'structdesc' (line 523)
    structdesc_121223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 523)
    getitem___121224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 19), structdesc_121223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 523)
    subscript_call_result_121225 = invoke(stypy.reporting.localization.Localization(__file__, 523, 19), getitem___121224, str_121222)
    
    # Testing the type of a for loop iterable (line 523)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 523, 8), subscript_call_result_121225)
    # Getting the type of the for loop variable (line 523)
    for_loop_var_121226 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 523, 8), subscript_call_result_121225)
    # Assigning a type to the variable 'tag' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'tag', for_loop_var_121226)
    # SSA begins for a for statement (line 523)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 524):
    
    # Assigning a Call to a Subscript (line 524):
    
    # Call to _read_string(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'f' (line 524)
    f_121228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 39), 'f', False)
    # Processing the call keyword arguments (line 524)
    kwargs_121229 = {}
    # Getting the type of '_read_string' (line 524)
    _read_string_121227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 26), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 524)
    _read_string_call_result_121230 = invoke(stypy.reporting.localization.Localization(__file__, 524, 26), _read_string_121227, *[f_121228], **kwargs_121229)
    
    # Getting the type of 'tag' (line 524)
    tag_121231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'tag')
    str_121232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 16), 'str', 'name')
    # Storing an element on a container (line 524)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 12), tag_121231, (str_121232, _read_string_call_result_121230))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Subscript (line 526):
    
    # Assigning a Dict to a Subscript (line 526):
    
    # Obtaining an instance of the builtin type 'dict' (line 526)
    dict_121233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 526)
    
    # Getting the type of 'structdesc' (line 526)
    structdesc_121234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'structdesc')
    str_121235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 19), 'str', 'arrtable')
    # Storing an element on a container (line 526)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 8), structdesc_121234, (str_121235, dict_121233))
    
    
    # Obtaining the type of the subscript
    str_121236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 30), 'str', 'tagtable')
    # Getting the type of 'structdesc' (line 527)
    structdesc_121237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 19), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 527)
    getitem___121238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 19), structdesc_121237, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 527)
    subscript_call_result_121239 = invoke(stypy.reporting.localization.Localization(__file__, 527, 19), getitem___121238, str_121236)
    
    # Testing the type of a for loop iterable (line 527)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 527, 8), subscript_call_result_121239)
    # Getting the type of the for loop variable (line 527)
    for_loop_var_121240 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 527, 8), subscript_call_result_121239)
    # Assigning a type to the variable 'tag' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'tag', for_loop_var_121240)
    # SSA begins for a for statement (line 527)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining the type of the subscript
    str_121241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 19), 'str', 'array')
    # Getting the type of 'tag' (line 528)
    tag_121242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'tag')
    # Obtaining the member '__getitem__' of a type (line 528)
    getitem___121243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 15), tag_121242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 528)
    subscript_call_result_121244 = invoke(stypy.reporting.localization.Localization(__file__, 528, 15), getitem___121243, str_121241)
    
    # Testing the type of an if condition (line 528)
    if_condition_121245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 12), subscript_call_result_121244)
    # Assigning a type to the variable 'if_condition_121245' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'if_condition_121245', if_condition_121245)
    # SSA begins for if statement (line 528)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 529):
    
    # Assigning a Call to a Subscript (line 529):
    
    # Call to _read_arraydesc(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'f' (line 529)
    f_121247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 70), 'f', False)
    # Processing the call keyword arguments (line 529)
    kwargs_121248 = {}
    # Getting the type of '_read_arraydesc' (line 529)
    _read_arraydesc_121246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 54), '_read_arraydesc', False)
    # Calling _read_arraydesc(args, kwargs) (line 529)
    _read_arraydesc_call_result_121249 = invoke(stypy.reporting.localization.Localization(__file__, 529, 54), _read_arraydesc_121246, *[f_121247], **kwargs_121248)
    
    
    # Obtaining the type of the subscript
    str_121250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 27), 'str', 'arrtable')
    # Getting the type of 'structdesc' (line 529)
    structdesc_121251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 16), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___121252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 16), structdesc_121251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 529)
    subscript_call_result_121253 = invoke(stypy.reporting.localization.Localization(__file__, 529, 16), getitem___121252, str_121250)
    
    
    # Obtaining the type of the subscript
    str_121254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 43), 'str', 'name')
    # Getting the type of 'tag' (line 529)
    tag_121255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 39), 'tag')
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___121256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 39), tag_121255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 529)
    subscript_call_result_121257 = invoke(stypy.reporting.localization.Localization(__file__, 529, 39), getitem___121256, str_121254)
    
    # Storing an element on a container (line 529)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 16), subscript_call_result_121253, (subscript_call_result_121257, _read_arraydesc_call_result_121249))
    # SSA join for if statement (line 528)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Subscript (line 531):
    
    # Assigning a Dict to a Subscript (line 531):
    
    # Obtaining an instance of the builtin type 'dict' (line 531)
    dict_121258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 36), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 531)
    
    # Getting the type of 'structdesc' (line 531)
    structdesc_121259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'structdesc')
    str_121260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 19), 'str', 'structtable')
    # Storing an element on a container (line 531)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 8), structdesc_121259, (str_121260, dict_121258))
    
    
    # Obtaining the type of the subscript
    str_121261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 30), 'str', 'tagtable')
    # Getting the type of 'structdesc' (line 532)
    structdesc_121262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 19), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 532)
    getitem___121263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 19), structdesc_121262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 532)
    subscript_call_result_121264 = invoke(stypy.reporting.localization.Localization(__file__, 532, 19), getitem___121263, str_121261)
    
    # Testing the type of a for loop iterable (line 532)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 532, 8), subscript_call_result_121264)
    # Getting the type of the for loop variable (line 532)
    for_loop_var_121265 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 532, 8), subscript_call_result_121264)
    # Assigning a type to the variable 'tag' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'tag', for_loop_var_121265)
    # SSA begins for a for statement (line 532)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining the type of the subscript
    str_121266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 19), 'str', 'structure')
    # Getting the type of 'tag' (line 533)
    tag_121267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'tag')
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___121268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 15), tag_121267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_121269 = invoke(stypy.reporting.localization.Localization(__file__, 533, 15), getitem___121268, str_121266)
    
    # Testing the type of an if condition (line 533)
    if_condition_121270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 533, 12), subscript_call_result_121269)
    # Assigning a type to the variable 'if_condition_121270' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'if_condition_121270', if_condition_121270)
    # SSA begins for if statement (line 533)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 534):
    
    # Assigning a Call to a Subscript (line 534):
    
    # Call to _read_structdesc(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'f' (line 534)
    f_121272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 74), 'f', False)
    # Processing the call keyword arguments (line 534)
    kwargs_121273 = {}
    # Getting the type of '_read_structdesc' (line 534)
    _read_structdesc_121271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 57), '_read_structdesc', False)
    # Calling _read_structdesc(args, kwargs) (line 534)
    _read_structdesc_call_result_121274 = invoke(stypy.reporting.localization.Localization(__file__, 534, 57), _read_structdesc_121271, *[f_121272], **kwargs_121273)
    
    
    # Obtaining the type of the subscript
    str_121275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 27), 'str', 'structtable')
    # Getting the type of 'structdesc' (line 534)
    structdesc_121276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___121277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 16), structdesc_121276, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_121278 = invoke(stypy.reporting.localization.Localization(__file__, 534, 16), getitem___121277, str_121275)
    
    
    # Obtaining the type of the subscript
    str_121279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 46), 'str', 'name')
    # Getting the type of 'tag' (line 534)
    tag_121280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 42), 'tag')
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___121281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 42), tag_121280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_121282 = invoke(stypy.reporting.localization.Localization(__file__, 534, 42), getitem___121281, str_121279)
    
    # Storing an element on a container (line 534)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 16), subscript_call_result_121278, (subscript_call_result_121282, _read_structdesc_call_result_121274))
    # SSA join for if statement (line 533)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    str_121283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 22), 'str', 'inherits')
    # Getting the type of 'structdesc' (line 536)
    structdesc_121284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 536)
    getitem___121285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 11), structdesc_121284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 536)
    subscript_call_result_121286 = invoke(stypy.reporting.localization.Localization(__file__, 536, 11), getitem___121285, str_121283)
    
    
    # Obtaining the type of the subscript
    str_121287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 48), 'str', 'is_super')
    # Getting the type of 'structdesc' (line 536)
    structdesc_121288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 37), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 536)
    getitem___121289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 37), structdesc_121288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 536)
    subscript_call_result_121290 = invoke(stypy.reporting.localization.Localization(__file__, 536, 37), getitem___121289, str_121287)
    
    # Applying the binary operator 'or' (line 536)
    result_or_keyword_121291 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), 'or', subscript_call_result_121286, subscript_call_result_121290)
    
    # Testing the type of an if condition (line 536)
    if_condition_121292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), result_or_keyword_121291)
    # Assigning a type to the variable 'if_condition_121292' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_121292', if_condition_121292)
    # SSA begins for if statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 537):
    
    # Assigning a Call to a Subscript (line 537):
    
    # Call to _read_string(...): (line 537)
    # Processing the call arguments (line 537)
    # Getting the type of 'f' (line 537)
    f_121294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 51), 'f', False)
    # Processing the call keyword arguments (line 537)
    kwargs_121295 = {}
    # Getting the type of '_read_string' (line 537)
    _read_string_121293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 38), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 537)
    _read_string_call_result_121296 = invoke(stypy.reporting.localization.Localization(__file__, 537, 38), _read_string_121293, *[f_121294], **kwargs_121295)
    
    # Getting the type of 'structdesc' (line 537)
    structdesc_121297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'structdesc')
    str_121298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 23), 'str', 'classname')
    # Storing an element on a container (line 537)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 12), structdesc_121297, (str_121298, _read_string_call_result_121296))
    
    # Assigning a Call to a Subscript (line 538):
    
    # Assigning a Call to a Subscript (line 538):
    
    # Call to _read_long(...): (line 538)
    # Processing the call arguments (line 538)
    # Getting the type of 'f' (line 538)
    f_121300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 51), 'f', False)
    # Processing the call keyword arguments (line 538)
    kwargs_121301 = {}
    # Getting the type of '_read_long' (line 538)
    _read_long_121299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 40), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 538)
    _read_long_call_result_121302 = invoke(stypy.reporting.localization.Localization(__file__, 538, 40), _read_long_121299, *[f_121300], **kwargs_121301)
    
    # Getting the type of 'structdesc' (line 538)
    structdesc_121303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'structdesc')
    str_121304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 23), 'str', 'nsupclasses')
    # Storing an element on a container (line 538)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 12), structdesc_121303, (str_121304, _read_long_call_result_121302))
    
    # Assigning a List to a Subscript (line 539):
    
    # Assigning a List to a Subscript (line 539):
    
    # Obtaining an instance of the builtin type 'list' (line 539)
    list_121305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 539)
    
    # Getting the type of 'structdesc' (line 539)
    structdesc_121306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'structdesc')
    str_121307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 23), 'str', 'supclassnames')
    # Storing an element on a container (line 539)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 12), structdesc_121306, (str_121307, list_121305))
    
    
    # Call to range(...): (line 540)
    # Processing the call arguments (line 540)
    
    # Obtaining the type of the subscript
    str_121309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 38), 'str', 'nsupclasses')
    # Getting the type of 'structdesc' (line 540)
    structdesc_121310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'structdesc', False)
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___121311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 27), structdesc_121310, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 540)
    subscript_call_result_121312 = invoke(stypy.reporting.localization.Localization(__file__, 540, 27), getitem___121311, str_121309)
    
    # Processing the call keyword arguments (line 540)
    kwargs_121313 = {}
    # Getting the type of 'range' (line 540)
    range_121308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 21), 'range', False)
    # Calling range(args, kwargs) (line 540)
    range_call_result_121314 = invoke(stypy.reporting.localization.Localization(__file__, 540, 21), range_121308, *[subscript_call_result_121312], **kwargs_121313)
    
    # Testing the type of a for loop iterable (line 540)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 540, 12), range_call_result_121314)
    # Getting the type of the for loop variable (line 540)
    for_loop_var_121315 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 540, 12), range_call_result_121314)
    # Assigning a type to the variable 's' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 's', for_loop_var_121315)
    # SSA begins for a for statement (line 540)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 541)
    # Processing the call arguments (line 541)
    
    # Call to _read_string(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'f' (line 541)
    f_121322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 64), 'f', False)
    # Processing the call keyword arguments (line 541)
    kwargs_121323 = {}
    # Getting the type of '_read_string' (line 541)
    _read_string_121321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 51), '_read_string', False)
    # Calling _read_string(args, kwargs) (line 541)
    _read_string_call_result_121324 = invoke(stypy.reporting.localization.Localization(__file__, 541, 51), _read_string_121321, *[f_121322], **kwargs_121323)
    
    # Processing the call keyword arguments (line 541)
    kwargs_121325 = {}
    
    # Obtaining the type of the subscript
    str_121316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 27), 'str', 'supclassnames')
    # Getting the type of 'structdesc' (line 541)
    structdesc_121317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'structdesc', False)
    # Obtaining the member '__getitem__' of a type (line 541)
    getitem___121318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 16), structdesc_121317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 541)
    subscript_call_result_121319 = invoke(stypy.reporting.localization.Localization(__file__, 541, 16), getitem___121318, str_121316)
    
    # Obtaining the member 'append' of a type (line 541)
    append_121320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 16), subscript_call_result_121319, 'append')
    # Calling append(args, kwargs) (line 541)
    append_call_result_121326 = invoke(stypy.reporting.localization.Localization(__file__, 541, 16), append_121320, *[_read_string_call_result_121324], **kwargs_121325)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Subscript (line 542):
    
    # Assigning a List to a Subscript (line 542):
    
    # Obtaining an instance of the builtin type 'list' (line 542)
    list_121327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 542)
    
    # Getting the type of 'structdesc' (line 542)
    structdesc_121328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'structdesc')
    str_121329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 23), 'str', 'supclasstable')
    # Storing an element on a container (line 542)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 12), structdesc_121328, (str_121329, list_121327))
    
    
    # Call to range(...): (line 543)
    # Processing the call arguments (line 543)
    
    # Obtaining the type of the subscript
    str_121331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 38), 'str', 'nsupclasses')
    # Getting the type of 'structdesc' (line 543)
    structdesc_121332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 27), 'structdesc', False)
    # Obtaining the member '__getitem__' of a type (line 543)
    getitem___121333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 27), structdesc_121332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 543)
    subscript_call_result_121334 = invoke(stypy.reporting.localization.Localization(__file__, 543, 27), getitem___121333, str_121331)
    
    # Processing the call keyword arguments (line 543)
    kwargs_121335 = {}
    # Getting the type of 'range' (line 543)
    range_121330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 21), 'range', False)
    # Calling range(args, kwargs) (line 543)
    range_call_result_121336 = invoke(stypy.reporting.localization.Localization(__file__, 543, 21), range_121330, *[subscript_call_result_121334], **kwargs_121335)
    
    # Testing the type of a for loop iterable (line 543)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 543, 12), range_call_result_121336)
    # Getting the type of the for loop variable (line 543)
    for_loop_var_121337 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 543, 12), range_call_result_121336)
    # Assigning a type to the variable 's' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 's', for_loop_var_121337)
    # SSA begins for a for statement (line 543)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 544)
    # Processing the call arguments (line 544)
    
    # Call to _read_structdesc(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'f' (line 544)
    f_121344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 68), 'f', False)
    # Processing the call keyword arguments (line 544)
    kwargs_121345 = {}
    # Getting the type of '_read_structdesc' (line 544)
    _read_structdesc_121343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 51), '_read_structdesc', False)
    # Calling _read_structdesc(args, kwargs) (line 544)
    _read_structdesc_call_result_121346 = invoke(stypy.reporting.localization.Localization(__file__, 544, 51), _read_structdesc_121343, *[f_121344], **kwargs_121345)
    
    # Processing the call keyword arguments (line 544)
    kwargs_121347 = {}
    
    # Obtaining the type of the subscript
    str_121338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 27), 'str', 'supclasstable')
    # Getting the type of 'structdesc' (line 544)
    structdesc_121339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'structdesc', False)
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___121340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 16), structdesc_121339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 544)
    subscript_call_result_121341 = invoke(stypy.reporting.localization.Localization(__file__, 544, 16), getitem___121340, str_121338)
    
    # Obtaining the member 'append' of a type (line 544)
    append_121342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 16), subscript_call_result_121341, 'append')
    # Calling append(args, kwargs) (line 544)
    append_call_result_121348 = invoke(stypy.reporting.localization.Localization(__file__, 544, 16), append_121342, *[_read_structdesc_call_result_121346], **kwargs_121347)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 546):
    
    # Assigning a Name to a Subscript (line 546):
    # Getting the type of 'structdesc' (line 546)
    structdesc_121349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 42), 'structdesc')
    # Getting the type of 'STRUCT_DICT' (line 546)
    STRUCT_DICT_121350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'STRUCT_DICT')
    
    # Obtaining the type of the subscript
    str_121351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 31), 'str', 'name')
    # Getting the type of 'structdesc' (line 546)
    structdesc_121352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 20), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___121353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 20), structdesc_121352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 546)
    subscript_call_result_121354 = invoke(stypy.reporting.localization.Localization(__file__, 546, 20), getitem___121353, str_121351)
    
    # Storing an element on a container (line 546)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 8), STRUCT_DICT_121350, (subscript_call_result_121354, structdesc_121349))
    # SSA branch for the else part of an if statement (line 517)
    module_type_store.open_ssa_branch('else')
    
    
    
    
    # Obtaining the type of the subscript
    str_121355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 26), 'str', 'name')
    # Getting the type of 'structdesc' (line 550)
    structdesc_121356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 550)
    getitem___121357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 15), structdesc_121356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 550)
    subscript_call_result_121358 = invoke(stypy.reporting.localization.Localization(__file__, 550, 15), getitem___121357, str_121355)
    
    # Getting the type of 'STRUCT_DICT' (line 550)
    STRUCT_DICT_121359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 37), 'STRUCT_DICT')
    # Applying the binary operator 'in' (line 550)
    result_contains_121360 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 15), 'in', subscript_call_result_121358, STRUCT_DICT_121359)
    
    # Applying the 'not' unary operator (line 550)
    result_not__121361 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 11), 'not', result_contains_121360)
    
    # Testing the type of an if condition (line 550)
    if_condition_121362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 550, 8), result_not__121361)
    # Assigning a type to the variable 'if_condition_121362' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'if_condition_121362', if_condition_121362)
    # SSA begins for if statement (line 550)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 551)
    # Processing the call arguments (line 551)
    str_121364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 28), 'str', "PREDEF=1 but can't find definition")
    # Processing the call keyword arguments (line 551)
    kwargs_121365 = {}
    # Getting the type of 'Exception' (line 551)
    Exception_121363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 18), 'Exception', False)
    # Calling Exception(args, kwargs) (line 551)
    Exception_call_result_121366 = invoke(stypy.reporting.localization.Localization(__file__, 551, 18), Exception_121363, *[str_121364], **kwargs_121365)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 551, 12), Exception_call_result_121366, 'raise parameter', BaseException)
    # SSA join for if statement (line 550)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 553):
    
    # Assigning a Subscript to a Name (line 553):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    str_121367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 44), 'str', 'name')
    # Getting the type of 'structdesc' (line 553)
    structdesc_121368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 33), 'structdesc')
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___121369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 33), structdesc_121368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 553)
    subscript_call_result_121370 = invoke(stypy.reporting.localization.Localization(__file__, 553, 33), getitem___121369, str_121367)
    
    # Getting the type of 'STRUCT_DICT' (line 553)
    STRUCT_DICT_121371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 21), 'STRUCT_DICT')
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___121372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 21), STRUCT_DICT_121371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 553)
    subscript_call_result_121373 = invoke(stypy.reporting.localization.Localization(__file__, 553, 21), getitem___121372, subscript_call_result_121370)
    
    # Assigning a type to the variable 'structdesc' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'structdesc', subscript_call_result_121373)
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'structdesc' (line 555)
    structdesc_121374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'structdesc')
    # Assigning a type to the variable 'stypy_return_type' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type', structdesc_121374)
    
    # ################# End of '_read_structdesc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_structdesc' in the type store
    # Getting the type of 'stypy_return_type' (line 499)
    stypy_return_type_121375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121375)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_structdesc'
    return stypy_return_type_121375

# Assigning a type to the variable '_read_structdesc' (line 499)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 0), '_read_structdesc', _read_structdesc)

@norecursion
def _read_tagdesc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_tagdesc'
    module_type_store = module_type_store.open_function_context('_read_tagdesc', 558, 0, False)
    
    # Passed parameters checking function
    _read_tagdesc.stypy_localization = localization
    _read_tagdesc.stypy_type_of_self = None
    _read_tagdesc.stypy_type_store = module_type_store
    _read_tagdesc.stypy_function_name = '_read_tagdesc'
    _read_tagdesc.stypy_param_names_list = ['f']
    _read_tagdesc.stypy_varargs_param_name = None
    _read_tagdesc.stypy_kwargs_param_name = None
    _read_tagdesc.stypy_call_defaults = defaults
    _read_tagdesc.stypy_call_varargs = varargs
    _read_tagdesc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_tagdesc', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_tagdesc', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_tagdesc(...)' code ##################

    str_121376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 4), 'str', 'Function to read in a tag descriptor')
    
    # Assigning a Dict to a Name (line 561):
    
    # Assigning a Dict to a Name (line 561):
    
    # Obtaining an instance of the builtin type 'dict' (line 561)
    dict_121377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 561)
    # Adding element type (key, value) (line 561)
    str_121378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 15), 'str', 'offset')
    
    # Call to _read_long(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'f' (line 561)
    f_121380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 36), 'f', False)
    # Processing the call keyword arguments (line 561)
    kwargs_121381 = {}
    # Getting the type of '_read_long' (line 561)
    _read_long_121379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 25), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 561)
    _read_long_call_result_121382 = invoke(stypy.reporting.localization.Localization(__file__, 561, 25), _read_long_121379, *[f_121380], **kwargs_121381)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 14), dict_121377, (str_121378, _read_long_call_result_121382))
    
    # Assigning a type to the variable 'tagdesc' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'tagdesc', dict_121377)
    
    
    
    # Obtaining the type of the subscript
    str_121383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 15), 'str', 'offset')
    # Getting the type of 'tagdesc' (line 563)
    tagdesc_121384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 7), 'tagdesc')
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___121385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 7), tagdesc_121384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_121386 = invoke(stypy.reporting.localization.Localization(__file__, 563, 7), getitem___121385, str_121383)
    
    int_121387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 28), 'int')
    # Applying the binary operator '==' (line 563)
    result_eq_121388 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 7), '==', subscript_call_result_121386, int_121387)
    
    # Testing the type of an if condition (line 563)
    if_condition_121389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 563, 4), result_eq_121388)
    # Assigning a type to the variable 'if_condition_121389' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'if_condition_121389', if_condition_121389)
    # SSA begins for if statement (line 563)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 564):
    
    # Assigning a Call to a Subscript (line 564):
    
    # Call to _read_uint64(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'f' (line 564)
    f_121391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 41), 'f', False)
    # Processing the call keyword arguments (line 564)
    kwargs_121392 = {}
    # Getting the type of '_read_uint64' (line 564)
    _read_uint64_121390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 28), '_read_uint64', False)
    # Calling _read_uint64(args, kwargs) (line 564)
    _read_uint64_call_result_121393 = invoke(stypy.reporting.localization.Localization(__file__, 564, 28), _read_uint64_121390, *[f_121391], **kwargs_121392)
    
    # Getting the type of 'tagdesc' (line 564)
    tagdesc_121394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'tagdesc')
    str_121395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 16), 'str', 'offset')
    # Storing an element on a container (line 564)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 564, 8), tagdesc_121394, (str_121395, _read_uint64_call_result_121393))
    # SSA join for if statement (line 563)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 566):
    
    # Assigning a Call to a Subscript (line 566):
    
    # Call to _read_long(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'f' (line 566)
    f_121397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 37), 'f', False)
    # Processing the call keyword arguments (line 566)
    kwargs_121398 = {}
    # Getting the type of '_read_long' (line 566)
    _read_long_121396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 26), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 566)
    _read_long_call_result_121399 = invoke(stypy.reporting.localization.Localization(__file__, 566, 26), _read_long_121396, *[f_121397], **kwargs_121398)
    
    # Getting the type of 'tagdesc' (line 566)
    tagdesc_121400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'tagdesc')
    str_121401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 12), 'str', 'typecode')
    # Storing an element on a container (line 566)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 4), tagdesc_121400, (str_121401, _read_long_call_result_121399))
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to _read_long(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'f' (line 567)
    f_121403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 26), 'f', False)
    # Processing the call keyword arguments (line 567)
    kwargs_121404 = {}
    # Getting the type of '_read_long' (line 567)
    _read_long_121402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 567)
    _read_long_call_result_121405 = invoke(stypy.reporting.localization.Localization(__file__, 567, 15), _read_long_121402, *[f_121403], **kwargs_121404)
    
    # Assigning a type to the variable 'tagflags' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'tagflags', _read_long_call_result_121405)
    
    # Assigning a Compare to a Subscript (line 569):
    
    # Assigning a Compare to a Subscript (line 569):
    
    # Getting the type of 'tagflags' (line 569)
    tagflags_121406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 'tagflags')
    int_121407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 34), 'int')
    # Applying the binary operator '&' (line 569)
    result_and__121408 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 23), '&', tagflags_121406, int_121407)
    
    int_121409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 39), 'int')
    # Applying the binary operator '==' (line 569)
    result_eq_121410 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 23), '==', result_and__121408, int_121409)
    
    # Getting the type of 'tagdesc' (line 569)
    tagdesc_121411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tagdesc')
    str_121412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 12), 'str', 'array')
    # Storing an element on a container (line 569)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 4), tagdesc_121411, (str_121412, result_eq_121410))
    
    # Assigning a Compare to a Subscript (line 570):
    
    # Assigning a Compare to a Subscript (line 570):
    
    # Getting the type of 'tagflags' (line 570)
    tagflags_121413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 27), 'tagflags')
    int_121414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 38), 'int')
    # Applying the binary operator '&' (line 570)
    result_and__121415 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 27), '&', tagflags_121413, int_121414)
    
    int_121416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 44), 'int')
    # Applying the binary operator '==' (line 570)
    result_eq_121417 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 27), '==', result_and__121415, int_121416)
    
    # Getting the type of 'tagdesc' (line 570)
    tagdesc_121418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'tagdesc')
    str_121419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 12), 'str', 'structure')
    # Storing an element on a container (line 570)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 4), tagdesc_121418, (str_121419, result_eq_121417))
    
    # Assigning a Compare to a Subscript (line 571):
    
    # Assigning a Compare to a Subscript (line 571):
    
    
    # Obtaining the type of the subscript
    str_121420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 32), 'str', 'typecode')
    # Getting the type of 'tagdesc' (line 571)
    tagdesc_121421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 24), 'tagdesc')
    # Obtaining the member '__getitem__' of a type (line 571)
    getitem___121422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 24), tagdesc_121421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 571)
    subscript_call_result_121423 = invoke(stypy.reporting.localization.Localization(__file__, 571, 24), getitem___121422, str_121420)
    
    # Getting the type of 'DTYPE_DICT' (line 571)
    DTYPE_DICT_121424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 47), 'DTYPE_DICT')
    # Applying the binary operator 'in' (line 571)
    result_contains_121425 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 24), 'in', subscript_call_result_121423, DTYPE_DICT_121424)
    
    # Getting the type of 'tagdesc' (line 571)
    tagdesc_121426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'tagdesc')
    str_121427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 12), 'str', 'scalar')
    # Storing an element on a container (line 571)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 4), tagdesc_121426, (str_121427, result_contains_121425))
    # Getting the type of 'tagdesc' (line 574)
    tagdesc_121428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), 'tagdesc')
    # Assigning a type to the variable 'stypy_return_type' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'stypy_return_type', tagdesc_121428)
    
    # ################# End of '_read_tagdesc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_tagdesc' in the type store
    # Getting the type of 'stypy_return_type' (line 558)
    stypy_return_type_121429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121429)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_tagdesc'
    return stypy_return_type_121429

# Assigning a type to the variable '_read_tagdesc' (line 558)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 0), '_read_tagdesc', _read_tagdesc)

@norecursion
def _replace_heap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_replace_heap'
    module_type_store = module_type_store.open_function_context('_replace_heap', 577, 0, False)
    
    # Passed parameters checking function
    _replace_heap.stypy_localization = localization
    _replace_heap.stypy_type_of_self = None
    _replace_heap.stypy_type_store = module_type_store
    _replace_heap.stypy_function_name = '_replace_heap'
    _replace_heap.stypy_param_names_list = ['variable', 'heap']
    _replace_heap.stypy_varargs_param_name = None
    _replace_heap.stypy_kwargs_param_name = None
    _replace_heap.stypy_call_defaults = defaults
    _replace_heap.stypy_call_varargs = varargs
    _replace_heap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_replace_heap', ['variable', 'heap'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_replace_heap', localization, ['variable', 'heap'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_replace_heap(...)' code ##################

    
    
    # Call to isinstance(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'variable' (line 579)
    variable_121431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 18), 'variable', False)
    # Getting the type of 'Pointer' (line 579)
    Pointer_121432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 28), 'Pointer', False)
    # Processing the call keyword arguments (line 579)
    kwargs_121433 = {}
    # Getting the type of 'isinstance' (line 579)
    isinstance_121430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 579)
    isinstance_call_result_121434 = invoke(stypy.reporting.localization.Localization(__file__, 579, 7), isinstance_121430, *[variable_121431, Pointer_121432], **kwargs_121433)
    
    # Testing the type of an if condition (line 579)
    if_condition_121435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 4), isinstance_call_result_121434)
    # Assigning a type to the variable 'if_condition_121435' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'if_condition_121435', if_condition_121435)
    # SSA begins for if statement (line 579)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isinstance(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'variable' (line 581)
    variable_121437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'variable', False)
    # Getting the type of 'Pointer' (line 581)
    Pointer_121438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 35), 'Pointer', False)
    # Processing the call keyword arguments (line 581)
    kwargs_121439 = {}
    # Getting the type of 'isinstance' (line 581)
    isinstance_121436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 14), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 581)
    isinstance_call_result_121440 = invoke(stypy.reporting.localization.Localization(__file__, 581, 14), isinstance_121436, *[variable_121437, Pointer_121438], **kwargs_121439)
    
    # Testing the type of an if condition (line 581)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 8), isinstance_call_result_121440)
    # SSA begins for while statement (line 581)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # Getting the type of 'variable' (line 583)
    variable_121441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'variable')
    # Obtaining the member 'index' of a type (line 583)
    index_121442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 15), variable_121441, 'index')
    int_121443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 33), 'int')
    # Applying the binary operator '==' (line 583)
    result_eq_121444 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 15), '==', index_121442, int_121443)
    
    # Testing the type of an if condition (line 583)
    if_condition_121445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 12), result_eq_121444)
    # Assigning a type to the variable 'if_condition_121445' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'if_condition_121445', if_condition_121445)
    # SSA begins for if statement (line 583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 584):
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'None' (line 584)
    None_121446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 27), 'None')
    # Assigning a type to the variable 'variable' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'variable', None_121446)
    # SSA branch for the else part of an if statement (line 583)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'variable' (line 586)
    variable_121447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 19), 'variable')
    # Obtaining the member 'index' of a type (line 586)
    index_121448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 19), variable_121447, 'index')
    # Getting the type of 'heap' (line 586)
    heap_121449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), 'heap')
    # Applying the binary operator 'in' (line 586)
    result_contains_121450 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 19), 'in', index_121448, heap_121449)
    
    # Testing the type of an if condition (line 586)
    if_condition_121451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 16), result_contains_121450)
    # Assigning a type to the variable 'if_condition_121451' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'if_condition_121451', if_condition_121451)
    # SSA begins for if statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 587):
    
    # Assigning a Subscript to a Name (line 587):
    
    # Obtaining the type of the subscript
    # Getting the type of 'variable' (line 587)
    variable_121452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 36), 'variable')
    # Obtaining the member 'index' of a type (line 587)
    index_121453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 36), variable_121452, 'index')
    # Getting the type of 'heap' (line 587)
    heap_121454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 31), 'heap')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___121455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 31), heap_121454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_121456 = invoke(stypy.reporting.localization.Localization(__file__, 587, 31), getitem___121455, index_121453)
    
    # Assigning a type to the variable 'variable' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 20), 'variable', subscript_call_result_121456)
    # SSA branch for the else part of an if statement (line 586)
    module_type_store.open_ssa_branch('else')
    
    # Call to warn(...): (line 589)
    # Processing the call arguments (line 589)
    str_121459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 34), 'str', 'Variable referenced by pointer not found in heap: variable will be set to None')
    # Processing the call keyword arguments (line 589)
    kwargs_121460 = {}
    # Getting the type of 'warnings' (line 589)
    warnings_121457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 20), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 589)
    warn_121458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 20), warnings_121457, 'warn')
    # Calling warn(args, kwargs) (line 589)
    warn_call_result_121461 = invoke(stypy.reporting.localization.Localization(__file__, 589, 20), warn_121458, *[str_121459], **kwargs_121460)
    
    
    # Assigning a Name to a Name (line 591):
    
    # Assigning a Name to a Name (line 591):
    # Getting the type of 'None' (line 591)
    None_121462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 31), 'None')
    # Assigning a type to the variable 'variable' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 20), 'variable', None_121462)
    # SSA join for if statement (line 586)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 583)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 581)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 593):
    
    # Assigning a Subscript to a Name (line 593):
    
    # Obtaining the type of the subscript
    int_121463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
    
    # Call to _replace_heap(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'variable' (line 593)
    variable_121465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 37), 'variable', False)
    # Getting the type of 'heap' (line 593)
    heap_121466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 47), 'heap', False)
    # Processing the call keyword arguments (line 593)
    kwargs_121467 = {}
    # Getting the type of '_replace_heap' (line 593)
    _replace_heap_121464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 593)
    _replace_heap_call_result_121468 = invoke(stypy.reporting.localization.Localization(__file__, 593, 23), _replace_heap_121464, *[variable_121465, heap_121466], **kwargs_121467)
    
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___121469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), _replace_heap_call_result_121468, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_121470 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___121469, int_121463)
    
    # Assigning a type to the variable 'tuple_var_assignment_119602' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_119602', subscript_call_result_121470)
    
    # Assigning a Subscript to a Name (line 593):
    
    # Obtaining the type of the subscript
    int_121471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
    
    # Call to _replace_heap(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'variable' (line 593)
    variable_121473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 37), 'variable', False)
    # Getting the type of 'heap' (line 593)
    heap_121474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 47), 'heap', False)
    # Processing the call keyword arguments (line 593)
    kwargs_121475 = {}
    # Getting the type of '_replace_heap' (line 593)
    _replace_heap_121472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 593)
    _replace_heap_call_result_121476 = invoke(stypy.reporting.localization.Localization(__file__, 593, 23), _replace_heap_121472, *[variable_121473, heap_121474], **kwargs_121475)
    
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___121477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), _replace_heap_call_result_121476, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_121478 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___121477, int_121471)
    
    # Assigning a type to the variable 'tuple_var_assignment_119603' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_119603', subscript_call_result_121478)
    
    # Assigning a Name to a Name (line 593):
    # Getting the type of 'tuple_var_assignment_119602' (line 593)
    tuple_var_assignment_119602_121479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_119602')
    # Assigning a type to the variable 'replace' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'replace', tuple_var_assignment_119602_121479)
    
    # Assigning a Name to a Name (line 593):
    # Getting the type of 'tuple_var_assignment_119603' (line 593)
    tuple_var_assignment_119603_121480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_119603')
    # Assigning a type to the variable 'new' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 17), 'new', tuple_var_assignment_119603_121480)
    
    # Getting the type of 'replace' (line 595)
    replace_121481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 11), 'replace')
    # Testing the type of an if condition (line 595)
    if_condition_121482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 8), replace_121481)
    # Assigning a type to the variable 'if_condition_121482' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'if_condition_121482', if_condition_121482)
    # SSA begins for if statement (line 595)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 596):
    
    # Assigning a Name to a Name (line 596):
    # Getting the type of 'new' (line 596)
    new_121483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 23), 'new')
    # Assigning a type to the variable 'variable' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'variable', new_121483)
    # SSA join for if statement (line 595)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 598)
    tuple_121484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 598)
    # Adding element type (line 598)
    # Getting the type of 'True' (line 598)
    True_121485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 15), tuple_121484, True_121485)
    # Adding element type (line 598)
    # Getting the type of 'variable' (line 598)
    variable_121486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 21), 'variable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 15), tuple_121484, variable_121486)
    
    # Assigning a type to the variable 'stypy_return_type' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'stypy_return_type', tuple_121484)
    # SSA branch for the else part of an if statement (line 579)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 600)
    # Processing the call arguments (line 600)
    # Getting the type of 'variable' (line 600)
    variable_121488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 20), 'variable', False)
    # Getting the type of 'np' (line 600)
    np_121489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 30), 'np', False)
    # Obtaining the member 'core' of a type (line 600)
    core_121490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 30), np_121489, 'core')
    # Obtaining the member 'records' of a type (line 600)
    records_121491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 30), core_121490, 'records')
    # Obtaining the member 'recarray' of a type (line 600)
    recarray_121492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 30), records_121491, 'recarray')
    # Processing the call keyword arguments (line 600)
    kwargs_121493 = {}
    # Getting the type of 'isinstance' (line 600)
    isinstance_121487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 600)
    isinstance_call_result_121494 = invoke(stypy.reporting.localization.Localization(__file__, 600, 9), isinstance_121487, *[variable_121488, recarray_121492], **kwargs_121493)
    
    # Testing the type of an if condition (line 600)
    if_condition_121495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 9), isinstance_call_result_121494)
    # Assigning a type to the variable 'if_condition_121495' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 9), 'if_condition_121495', if_condition_121495)
    # SSA begins for if statement (line 600)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'variable' (line 603)
    variable_121497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 36), 'variable', False)
    # Processing the call keyword arguments (line 603)
    kwargs_121498 = {}
    # Getting the type of 'enumerate' (line 603)
    enumerate_121496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 26), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 603)
    enumerate_call_result_121499 = invoke(stypy.reporting.localization.Localization(__file__, 603, 26), enumerate_121496, *[variable_121497], **kwargs_121498)
    
    # Testing the type of a for loop iterable (line 603)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 603, 8), enumerate_call_result_121499)
    # Getting the type of the for loop variable (line 603)
    for_loop_var_121500 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 603, 8), enumerate_call_result_121499)
    # Assigning a type to the variable 'ir' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'ir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 8), for_loop_var_121500))
    # Assigning a type to the variable 'record' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'record', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 8), for_loop_var_121500))
    # SSA begins for a for statement (line 603)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 605):
    
    # Assigning a Subscript to a Name (line 605):
    
    # Obtaining the type of the subscript
    int_121501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 12), 'int')
    
    # Call to _replace_heap(...): (line 605)
    # Processing the call arguments (line 605)
    # Getting the type of 'record' (line 605)
    record_121503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 41), 'record', False)
    # Getting the type of 'heap' (line 605)
    heap_121504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 49), 'heap', False)
    # Processing the call keyword arguments (line 605)
    kwargs_121505 = {}
    # Getting the type of '_replace_heap' (line 605)
    _replace_heap_121502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 27), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 605)
    _replace_heap_call_result_121506 = invoke(stypy.reporting.localization.Localization(__file__, 605, 27), _replace_heap_121502, *[record_121503, heap_121504], **kwargs_121505)
    
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___121507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 12), _replace_heap_call_result_121506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_121508 = invoke(stypy.reporting.localization.Localization(__file__, 605, 12), getitem___121507, int_121501)
    
    # Assigning a type to the variable 'tuple_var_assignment_119604' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'tuple_var_assignment_119604', subscript_call_result_121508)
    
    # Assigning a Subscript to a Name (line 605):
    
    # Obtaining the type of the subscript
    int_121509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 12), 'int')
    
    # Call to _replace_heap(...): (line 605)
    # Processing the call arguments (line 605)
    # Getting the type of 'record' (line 605)
    record_121511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 41), 'record', False)
    # Getting the type of 'heap' (line 605)
    heap_121512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 49), 'heap', False)
    # Processing the call keyword arguments (line 605)
    kwargs_121513 = {}
    # Getting the type of '_replace_heap' (line 605)
    _replace_heap_121510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 27), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 605)
    _replace_heap_call_result_121514 = invoke(stypy.reporting.localization.Localization(__file__, 605, 27), _replace_heap_121510, *[record_121511, heap_121512], **kwargs_121513)
    
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___121515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 12), _replace_heap_call_result_121514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_121516 = invoke(stypy.reporting.localization.Localization(__file__, 605, 12), getitem___121515, int_121509)
    
    # Assigning a type to the variable 'tuple_var_assignment_119605' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'tuple_var_assignment_119605', subscript_call_result_121516)
    
    # Assigning a Name to a Name (line 605):
    # Getting the type of 'tuple_var_assignment_119604' (line 605)
    tuple_var_assignment_119604_121517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'tuple_var_assignment_119604')
    # Assigning a type to the variable 'replace' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'replace', tuple_var_assignment_119604_121517)
    
    # Assigning a Name to a Name (line 605):
    # Getting the type of 'tuple_var_assignment_119605' (line 605)
    tuple_var_assignment_119605_121518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'tuple_var_assignment_119605')
    # Assigning a type to the variable 'new' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 21), 'new', tuple_var_assignment_119605_121518)
    
    # Getting the type of 'replace' (line 607)
    replace_121519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 15), 'replace')
    # Testing the type of an if condition (line 607)
    if_condition_121520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 12), replace_121519)
    # Assigning a type to the variable 'if_condition_121520' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'if_condition_121520', if_condition_121520)
    # SSA begins for if statement (line 607)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 608):
    
    # Assigning a Name to a Subscript (line 608):
    # Getting the type of 'new' (line 608)
    new_121521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 31), 'new')
    # Getting the type of 'variable' (line 608)
    variable_121522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 16), 'variable')
    # Getting the type of 'ir' (line 608)
    ir_121523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 25), 'ir')
    # Storing an element on a container (line 608)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 16), variable_121522, (ir_121523, new_121521))
    # SSA join for if statement (line 607)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 610)
    tuple_121524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 610)
    # Adding element type (line 610)
    # Getting the type of 'False' (line 610)
    False_121525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 15), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 15), tuple_121524, False_121525)
    # Adding element type (line 610)
    # Getting the type of 'variable' (line 610)
    variable_121526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 22), 'variable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 15), tuple_121524, variable_121526)
    
    # Assigning a type to the variable 'stypy_return_type' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'stypy_return_type', tuple_121524)
    # SSA branch for the else part of an if statement (line 600)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 612)
    # Processing the call arguments (line 612)
    # Getting the type of 'variable' (line 612)
    variable_121528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 20), 'variable', False)
    # Getting the type of 'np' (line 612)
    np_121529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 30), 'np', False)
    # Obtaining the member 'core' of a type (line 612)
    core_121530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 30), np_121529, 'core')
    # Obtaining the member 'records' of a type (line 612)
    records_121531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 30), core_121530, 'records')
    # Obtaining the member 'record' of a type (line 612)
    record_121532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 30), records_121531, 'record')
    # Processing the call keyword arguments (line 612)
    kwargs_121533 = {}
    # Getting the type of 'isinstance' (line 612)
    isinstance_121527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 612)
    isinstance_call_result_121534 = invoke(stypy.reporting.localization.Localization(__file__, 612, 9), isinstance_121527, *[variable_121528, record_121532], **kwargs_121533)
    
    # Testing the type of an if condition (line 612)
    if_condition_121535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 9), isinstance_call_result_121534)
    # Assigning a type to the variable 'if_condition_121535' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 9), 'if_condition_121535', if_condition_121535)
    # SSA begins for if statement (line 612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'variable' (line 615)
    variable_121537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 35), 'variable', False)
    # Processing the call keyword arguments (line 615)
    kwargs_121538 = {}
    # Getting the type of 'enumerate' (line 615)
    enumerate_121536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 25), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 615)
    enumerate_call_result_121539 = invoke(stypy.reporting.localization.Localization(__file__, 615, 25), enumerate_121536, *[variable_121537], **kwargs_121538)
    
    # Testing the type of a for loop iterable (line 615)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 615, 8), enumerate_call_result_121539)
    # Getting the type of the for loop variable (line 615)
    for_loop_var_121540 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 615, 8), enumerate_call_result_121539)
    # Assigning a type to the variable 'iv' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'iv', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 8), for_loop_var_121540))
    # Assigning a type to the variable 'value' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 8), for_loop_var_121540))
    # SSA begins for a for statement (line 615)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 617):
    
    # Assigning a Subscript to a Name (line 617):
    
    # Obtaining the type of the subscript
    int_121541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 12), 'int')
    
    # Call to _replace_heap(...): (line 617)
    # Processing the call arguments (line 617)
    # Getting the type of 'value' (line 617)
    value_121543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 41), 'value', False)
    # Getting the type of 'heap' (line 617)
    heap_121544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 48), 'heap', False)
    # Processing the call keyword arguments (line 617)
    kwargs_121545 = {}
    # Getting the type of '_replace_heap' (line 617)
    _replace_heap_121542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 27), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 617)
    _replace_heap_call_result_121546 = invoke(stypy.reporting.localization.Localization(__file__, 617, 27), _replace_heap_121542, *[value_121543, heap_121544], **kwargs_121545)
    
    # Obtaining the member '__getitem__' of a type (line 617)
    getitem___121547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 12), _replace_heap_call_result_121546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 617)
    subscript_call_result_121548 = invoke(stypy.reporting.localization.Localization(__file__, 617, 12), getitem___121547, int_121541)
    
    # Assigning a type to the variable 'tuple_var_assignment_119606' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'tuple_var_assignment_119606', subscript_call_result_121548)
    
    # Assigning a Subscript to a Name (line 617):
    
    # Obtaining the type of the subscript
    int_121549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 12), 'int')
    
    # Call to _replace_heap(...): (line 617)
    # Processing the call arguments (line 617)
    # Getting the type of 'value' (line 617)
    value_121551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 41), 'value', False)
    # Getting the type of 'heap' (line 617)
    heap_121552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 48), 'heap', False)
    # Processing the call keyword arguments (line 617)
    kwargs_121553 = {}
    # Getting the type of '_replace_heap' (line 617)
    _replace_heap_121550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 27), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 617)
    _replace_heap_call_result_121554 = invoke(stypy.reporting.localization.Localization(__file__, 617, 27), _replace_heap_121550, *[value_121551, heap_121552], **kwargs_121553)
    
    # Obtaining the member '__getitem__' of a type (line 617)
    getitem___121555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 12), _replace_heap_call_result_121554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 617)
    subscript_call_result_121556 = invoke(stypy.reporting.localization.Localization(__file__, 617, 12), getitem___121555, int_121549)
    
    # Assigning a type to the variable 'tuple_var_assignment_119607' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'tuple_var_assignment_119607', subscript_call_result_121556)
    
    # Assigning a Name to a Name (line 617):
    # Getting the type of 'tuple_var_assignment_119606' (line 617)
    tuple_var_assignment_119606_121557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'tuple_var_assignment_119606')
    # Assigning a type to the variable 'replace' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'replace', tuple_var_assignment_119606_121557)
    
    # Assigning a Name to a Name (line 617):
    # Getting the type of 'tuple_var_assignment_119607' (line 617)
    tuple_var_assignment_119607_121558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'tuple_var_assignment_119607')
    # Assigning a type to the variable 'new' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 21), 'new', tuple_var_assignment_119607_121558)
    
    # Getting the type of 'replace' (line 619)
    replace_121559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 15), 'replace')
    # Testing the type of an if condition (line 619)
    if_condition_121560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 12), replace_121559)
    # Assigning a type to the variable 'if_condition_121560' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'if_condition_121560', if_condition_121560)
    # SSA begins for if statement (line 619)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 620):
    
    # Assigning a Name to a Subscript (line 620):
    # Getting the type of 'new' (line 620)
    new_121561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 31), 'new')
    # Getting the type of 'variable' (line 620)
    variable_121562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 16), 'variable')
    # Getting the type of 'iv' (line 620)
    iv_121563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 25), 'iv')
    # Storing an element on a container (line 620)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 16), variable_121562, (iv_121563, new_121561))
    # SSA join for if statement (line 619)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 622)
    tuple_121564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 622)
    # Adding element type (line 622)
    # Getting the type of 'False' (line 622)
    False_121565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 15), tuple_121564, False_121565)
    # Adding element type (line 622)
    # Getting the type of 'variable' (line 622)
    variable_121566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 22), 'variable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 15), tuple_121564, variable_121566)
    
    # Assigning a type to the variable 'stypy_return_type' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'stypy_return_type', tuple_121564)
    # SSA branch for the else part of an if statement (line 612)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 624)
    # Processing the call arguments (line 624)
    # Getting the type of 'variable' (line 624)
    variable_121568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 20), 'variable', False)
    # Getting the type of 'np' (line 624)
    np_121569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 30), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 624)
    ndarray_121570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 30), np_121569, 'ndarray')
    # Processing the call keyword arguments (line 624)
    kwargs_121571 = {}
    # Getting the type of 'isinstance' (line 624)
    isinstance_121567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 624)
    isinstance_call_result_121572 = invoke(stypy.reporting.localization.Localization(__file__, 624, 9), isinstance_121567, *[variable_121568, ndarray_121570], **kwargs_121571)
    
    # Testing the type of an if condition (line 624)
    if_condition_121573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 624, 9), isinstance_call_result_121572)
    # Assigning a type to the variable 'if_condition_121573' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 9), 'if_condition_121573', if_condition_121573)
    # SSA begins for if statement (line 624)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'variable' (line 627)
    variable_121574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), 'variable')
    # Obtaining the member 'dtype' of a type (line 627)
    dtype_121575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 11), variable_121574, 'dtype')
    # Obtaining the member 'type' of a type (line 627)
    type_121576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 11), dtype_121575, 'type')
    # Getting the type of 'np' (line 627)
    np_121577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 34), 'np')
    # Obtaining the member 'object_' of a type (line 627)
    object__121578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 34), np_121577, 'object_')
    # Applying the binary operator 'is' (line 627)
    result_is__121579 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 11), 'is', type_121576, object__121578)
    
    # Testing the type of an if condition (line 627)
    if_condition_121580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 627, 8), result_is__121579)
    # Assigning a type to the variable 'if_condition_121580' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'if_condition_121580', if_condition_121580)
    # SSA begins for if statement (line 627)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 629)
    # Processing the call arguments (line 629)
    # Getting the type of 'variable' (line 629)
    variable_121582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 28), 'variable', False)
    # Obtaining the member 'size' of a type (line 629)
    size_121583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 28), variable_121582, 'size')
    # Processing the call keyword arguments (line 629)
    kwargs_121584 = {}
    # Getting the type of 'range' (line 629)
    range_121581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 22), 'range', False)
    # Calling range(args, kwargs) (line 629)
    range_call_result_121585 = invoke(stypy.reporting.localization.Localization(__file__, 629, 22), range_121581, *[size_121583], **kwargs_121584)
    
    # Testing the type of a for loop iterable (line 629)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 629, 12), range_call_result_121585)
    # Getting the type of the for loop variable (line 629)
    for_loop_var_121586 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 629, 12), range_call_result_121585)
    # Assigning a type to the variable 'iv' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 12), 'iv', for_loop_var_121586)
    # SSA begins for a for statement (line 629)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 631):
    
    # Assigning a Subscript to a Name (line 631):
    
    # Obtaining the type of the subscript
    int_121587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 16), 'int')
    
    # Call to _replace_heap(...): (line 631)
    # Processing the call arguments (line 631)
    
    # Call to item(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'iv' (line 631)
    iv_121591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 59), 'iv', False)
    # Processing the call keyword arguments (line 631)
    kwargs_121592 = {}
    # Getting the type of 'variable' (line 631)
    variable_121589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 45), 'variable', False)
    # Obtaining the member 'item' of a type (line 631)
    item_121590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 45), variable_121589, 'item')
    # Calling item(args, kwargs) (line 631)
    item_call_result_121593 = invoke(stypy.reporting.localization.Localization(__file__, 631, 45), item_121590, *[iv_121591], **kwargs_121592)
    
    # Getting the type of 'heap' (line 631)
    heap_121594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 64), 'heap', False)
    # Processing the call keyword arguments (line 631)
    kwargs_121595 = {}
    # Getting the type of '_replace_heap' (line 631)
    _replace_heap_121588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 31), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 631)
    _replace_heap_call_result_121596 = invoke(stypy.reporting.localization.Localization(__file__, 631, 31), _replace_heap_121588, *[item_call_result_121593, heap_121594], **kwargs_121595)
    
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___121597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 16), _replace_heap_call_result_121596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 631)
    subscript_call_result_121598 = invoke(stypy.reporting.localization.Localization(__file__, 631, 16), getitem___121597, int_121587)
    
    # Assigning a type to the variable 'tuple_var_assignment_119608' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'tuple_var_assignment_119608', subscript_call_result_121598)
    
    # Assigning a Subscript to a Name (line 631):
    
    # Obtaining the type of the subscript
    int_121599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 16), 'int')
    
    # Call to _replace_heap(...): (line 631)
    # Processing the call arguments (line 631)
    
    # Call to item(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'iv' (line 631)
    iv_121603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 59), 'iv', False)
    # Processing the call keyword arguments (line 631)
    kwargs_121604 = {}
    # Getting the type of 'variable' (line 631)
    variable_121601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 45), 'variable', False)
    # Obtaining the member 'item' of a type (line 631)
    item_121602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 45), variable_121601, 'item')
    # Calling item(args, kwargs) (line 631)
    item_call_result_121605 = invoke(stypy.reporting.localization.Localization(__file__, 631, 45), item_121602, *[iv_121603], **kwargs_121604)
    
    # Getting the type of 'heap' (line 631)
    heap_121606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 64), 'heap', False)
    # Processing the call keyword arguments (line 631)
    kwargs_121607 = {}
    # Getting the type of '_replace_heap' (line 631)
    _replace_heap_121600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 31), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 631)
    _replace_heap_call_result_121608 = invoke(stypy.reporting.localization.Localization(__file__, 631, 31), _replace_heap_121600, *[item_call_result_121605, heap_121606], **kwargs_121607)
    
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___121609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 16), _replace_heap_call_result_121608, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 631)
    subscript_call_result_121610 = invoke(stypy.reporting.localization.Localization(__file__, 631, 16), getitem___121609, int_121599)
    
    # Assigning a type to the variable 'tuple_var_assignment_119609' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'tuple_var_assignment_119609', subscript_call_result_121610)
    
    # Assigning a Name to a Name (line 631):
    # Getting the type of 'tuple_var_assignment_119608' (line 631)
    tuple_var_assignment_119608_121611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'tuple_var_assignment_119608')
    # Assigning a type to the variable 'replace' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'replace', tuple_var_assignment_119608_121611)
    
    # Assigning a Name to a Name (line 631):
    # Getting the type of 'tuple_var_assignment_119609' (line 631)
    tuple_var_assignment_119609_121612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 16), 'tuple_var_assignment_119609')
    # Assigning a type to the variable 'new' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 25), 'new', tuple_var_assignment_119609_121612)
    
    # Getting the type of 'replace' (line 633)
    replace_121613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 19), 'replace')
    # Testing the type of an if condition (line 633)
    if_condition_121614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 633, 16), replace_121613)
    # Assigning a type to the variable 'if_condition_121614' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'if_condition_121614', if_condition_121614)
    # SSA begins for if statement (line 633)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to itemset(...): (line 634)
    # Processing the call arguments (line 634)
    # Getting the type of 'iv' (line 634)
    iv_121617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 37), 'iv', False)
    # Getting the type of 'new' (line 634)
    new_121618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 41), 'new', False)
    # Processing the call keyword arguments (line 634)
    kwargs_121619 = {}
    # Getting the type of 'variable' (line 634)
    variable_121615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 20), 'variable', False)
    # Obtaining the member 'itemset' of a type (line 634)
    itemset_121616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 20), variable_121615, 'itemset')
    # Calling itemset(args, kwargs) (line 634)
    itemset_call_result_121620 = invoke(stypy.reporting.localization.Localization(__file__, 634, 20), itemset_121616, *[iv_121617, new_121618], **kwargs_121619)
    
    # SSA join for if statement (line 633)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 627)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 636)
    tuple_121621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 636)
    # Adding element type (line 636)
    # Getting the type of 'False' (line 636)
    False_121622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 15), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 15), tuple_121621, False_121622)
    # Adding element type (line 636)
    # Getting the type of 'variable' (line 636)
    variable_121623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 22), 'variable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 15), tuple_121621, variable_121623)
    
    # Assigning a type to the variable 'stypy_return_type' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'stypy_return_type', tuple_121621)
    # SSA branch for the else part of an if statement (line 624)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 640)
    tuple_121624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 640)
    # Adding element type (line 640)
    # Getting the type of 'False' (line 640)
    False_121625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 15), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 15), tuple_121624, False_121625)
    # Adding element type (line 640)
    # Getting the type of 'variable' (line 640)
    variable_121626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 22), 'variable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 15), tuple_121624, variable_121626)
    
    # Assigning a type to the variable 'stypy_return_type' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'stypy_return_type', tuple_121624)
    # SSA join for if statement (line 624)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 612)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 600)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 579)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_replace_heap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_replace_heap' in the type store
    # Getting the type of 'stypy_return_type' (line 577)
    stypy_return_type_121627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121627)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_replace_heap'
    return stypy_return_type_121627

# Assigning a type to the variable '_replace_heap' (line 577)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 0), '_replace_heap', _replace_heap)
# Declaration of the 'AttrDict' class
# Getting the type of 'dict' (line 643)
dict_121628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'dict')

class AttrDict(dict_121628, ):
    str_121629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, (-1)), 'str', "\n    A case-insensitive dictionary with access via item, attribute, and call\n    notations:\n\n        >>> d = AttrDict()\n        >>> d['Variable'] = 123\n        >>> d['Variable']\n        123\n        >>> d.Variable\n        123\n        >>> d.variable\n        123\n        >>> d('VARIABLE')\n        123\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'dict' (line 660)
        dict_121630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 660)
        
        defaults = [dict_121630]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 660, 4, False)
        # Assigning a type to the variable 'self' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AttrDict.__init__', ['init'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['init'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 661)
        # Processing the call arguments (line 661)
        # Getting the type of 'self' (line 661)
        self_121633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 22), 'self', False)
        # Getting the type of 'init' (line 661)
        init_121634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 28), 'init', False)
        # Processing the call keyword arguments (line 661)
        kwargs_121635 = {}
        # Getting the type of 'dict' (line 661)
        dict_121631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'dict', False)
        # Obtaining the member '__init__' of a type (line 661)
        init___121632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 8), dict_121631, '__init__')
        # Calling __init__(args, kwargs) (line 661)
        init___call_result_121636 = invoke(stypy.reporting.localization.Localization(__file__, 661, 8), init___121632, *[self_121633, init_121634], **kwargs_121635)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 663, 4, False)
        # Assigning a type to the variable 'self' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AttrDict.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_function_name', 'AttrDict.__getitem__')
        AttrDict.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        AttrDict.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AttrDict.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AttrDict.__getitem__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Call to __getitem__(...): (line 664)
        # Processing the call arguments (line 664)
        
        # Call to lower(...): (line 664)
        # Processing the call keyword arguments (line 664)
        kwargs_121645 = {}
        # Getting the type of 'name' (line 664)
        name_121643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 49), 'name', False)
        # Obtaining the member 'lower' of a type (line 664)
        lower_121644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 49), name_121643, 'lower')
        # Calling lower(args, kwargs) (line 664)
        lower_call_result_121646 = invoke(stypy.reporting.localization.Localization(__file__, 664, 49), lower_121644, *[], **kwargs_121645)
        
        # Processing the call keyword arguments (line 664)
        kwargs_121647 = {}
        
        # Call to super(...): (line 664)
        # Processing the call arguments (line 664)
        # Getting the type of 'AttrDict' (line 664)
        AttrDict_121638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 21), 'AttrDict', False)
        # Getting the type of 'self' (line 664)
        self_121639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 31), 'self', False)
        # Processing the call keyword arguments (line 664)
        kwargs_121640 = {}
        # Getting the type of 'super' (line 664)
        super_121637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 'super', False)
        # Calling super(args, kwargs) (line 664)
        super_call_result_121641 = invoke(stypy.reporting.localization.Localization(__file__, 664, 15), super_121637, *[AttrDict_121638, self_121639], **kwargs_121640)
        
        # Obtaining the member '__getitem__' of a type (line 664)
        getitem___121642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 15), super_call_result_121641, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 664)
        getitem___call_result_121648 = invoke(stypy.reporting.localization.Localization(__file__, 664, 15), getitem___121642, *[lower_call_result_121646], **kwargs_121647)
        
        # Assigning a type to the variable 'stypy_return_type' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'stypy_return_type', getitem___call_result_121648)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 663)
        stypy_return_type_121649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_121649


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 666, 4, False)
        # Assigning a type to the variable 'self' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AttrDict.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_function_name', 'AttrDict.__setitem__')
        AttrDict.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['key', 'value'])
        AttrDict.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AttrDict.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AttrDict.__setitem__', ['key', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['key', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        
        # Call to __setitem__(...): (line 667)
        # Processing the call arguments (line 667)
        
        # Call to lower(...): (line 667)
        # Processing the call keyword arguments (line 667)
        kwargs_121658 = {}
        # Getting the type of 'key' (line 667)
        key_121656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 49), 'key', False)
        # Obtaining the member 'lower' of a type (line 667)
        lower_121657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 49), key_121656, 'lower')
        # Calling lower(args, kwargs) (line 667)
        lower_call_result_121659 = invoke(stypy.reporting.localization.Localization(__file__, 667, 49), lower_121657, *[], **kwargs_121658)
        
        # Getting the type of 'value' (line 667)
        value_121660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 62), 'value', False)
        # Processing the call keyword arguments (line 667)
        kwargs_121661 = {}
        
        # Call to super(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'AttrDict' (line 667)
        AttrDict_121651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 21), 'AttrDict', False)
        # Getting the type of 'self' (line 667)
        self_121652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 31), 'self', False)
        # Processing the call keyword arguments (line 667)
        kwargs_121653 = {}
        # Getting the type of 'super' (line 667)
        super_121650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 15), 'super', False)
        # Calling super(args, kwargs) (line 667)
        super_call_result_121654 = invoke(stypy.reporting.localization.Localization(__file__, 667, 15), super_121650, *[AttrDict_121651, self_121652], **kwargs_121653)
        
        # Obtaining the member '__setitem__' of a type (line 667)
        setitem___121655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 15), super_call_result_121654, '__setitem__')
        # Calling __setitem__(args, kwargs) (line 667)
        setitem___call_result_121662 = invoke(stypy.reporting.localization.Localization(__file__, 667, 15), setitem___121655, *[lower_call_result_121659, value_121660], **kwargs_121661)
        
        # Assigning a type to the variable 'stypy_return_type' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'stypy_return_type', setitem___call_result_121662)
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 666)
        stypy_return_type_121663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121663)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_121663

    
    # Assigning a Name to a Name (line 669):
    
    # Assigning a Name to a Name (line 670):
    
    # Assigning a Name to a Name (line 671):

# Assigning a type to the variable 'AttrDict' (line 643)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 0), 'AttrDict', AttrDict)

# Assigning a Name to a Name (line 669):
# Getting the type of 'AttrDict'
AttrDict_121664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AttrDict')
# Obtaining the member '__getitem__' of a type
getitem___121665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AttrDict_121664, '__getitem__')
# Getting the type of 'AttrDict'
AttrDict_121666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AttrDict')
# Setting the type of the member '__getattr__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AttrDict_121666, '__getattr__', getitem___121665)

# Assigning a Name to a Name (line 670):
# Getting the type of 'AttrDict'
AttrDict_121667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AttrDict')
# Obtaining the member '__setitem__' of a type
setitem___121668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AttrDict_121667, '__setitem__')
# Getting the type of 'AttrDict'
AttrDict_121669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AttrDict')
# Setting the type of the member '__setattr__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AttrDict_121669, '__setattr__', setitem___121668)

# Assigning a Name to a Name (line 671):
# Getting the type of 'AttrDict'
AttrDict_121670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AttrDict')
# Obtaining the member '__getitem__' of a type
getitem___121671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AttrDict_121670, '__getitem__')
# Getting the type of 'AttrDict'
AttrDict_121672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'AttrDict')
# Setting the type of the member '__call__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), AttrDict_121672, '__call__', getitem___121671)

@norecursion
def readsav(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 674)
    None_121673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 29), 'None')
    # Getting the type of 'False' (line 674)
    False_121674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 47), 'False')
    # Getting the type of 'None' (line 675)
    None_121675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 35), 'None')
    # Getting the type of 'False' (line 675)
    False_121676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 49), 'False')
    defaults = [None_121673, False_121674, None_121675, False_121676]
    # Create a new context for function 'readsav'
    module_type_store = module_type_store.open_function_context('readsav', 674, 0, False)
    
    # Passed parameters checking function
    readsav.stypy_localization = localization
    readsav.stypy_type_of_self = None
    readsav.stypy_type_store = module_type_store
    readsav.stypy_function_name = 'readsav'
    readsav.stypy_param_names_list = ['file_name', 'idict', 'python_dict', 'uncompressed_file_name', 'verbose']
    readsav.stypy_varargs_param_name = None
    readsav.stypy_kwargs_param_name = None
    readsav.stypy_call_defaults = defaults
    readsav.stypy_call_varargs = varargs
    readsav.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'readsav', ['file_name', 'idict', 'python_dict', 'uncompressed_file_name', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'readsav', localization, ['file_name', 'idict', 'python_dict', 'uncompressed_file_name', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'readsav(...)' code ##################

    str_121677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, (-1)), 'str', '\n    Read an IDL .sav file.\n\n    Parameters\n    ----------\n    file_name : str\n        Name of the IDL save file.\n    idict : dict, optional\n        Dictionary in which to insert .sav file variables.\n    python_dict : bool, optional\n        By default, the object return is not a Python dictionary, but a\n        case-insensitive dictionary with item, attribute, and call access\n        to variables. To get a standard Python dictionary, set this option\n        to True.\n    uncompressed_file_name : str, optional\n        This option only has an effect for .sav files written with the\n        /compress option. If a file name is specified, compressed .sav\n        files are uncompressed to this file. Otherwise, readsav will use\n        the `tempfile` module to determine a temporary filename\n        automatically, and will remove the temporary file upon successfully\n        reading it in.\n    verbose : bool, optional\n        Whether to print out information about the save file, including\n        the records read, and available variables.\n\n    Returns\n    -------\n    idl_dict : AttrDict or dict\n        If `python_dict` is set to False (default), this function returns a\n        case-insensitive dictionary with item, attribute, and call access\n        to variables. If `python_dict` is set to True, this function\n        returns a Python dictionary with all variable names in lowercase.\n        If `idict` was specified, then variables are written to the\n        dictionary specified, and the updated dictionary is returned.\n\n    ')
    
    # Assigning a List to a Name (line 714):
    
    # Assigning a List to a Name (line 714):
    
    # Obtaining an instance of the builtin type 'list' (line 714)
    list_121678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 714)
    
    # Assigning a type to the variable 'records' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'records', list_121678)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'python_dict' (line 715)
    python_dict_121679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 7), 'python_dict')
    # Getting the type of 'idict' (line 715)
    idict_121680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 22), 'idict')
    # Applying the binary operator 'or' (line 715)
    result_or_keyword_121681 = python_operator(stypy.reporting.localization.Localization(__file__, 715, 7), 'or', python_dict_121679, idict_121680)
    
    # Testing the type of an if condition (line 715)
    if_condition_121682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 715, 4), result_or_keyword_121681)
    # Assigning a type to the variable 'if_condition_121682' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'if_condition_121682', if_condition_121682)
    # SSA begins for if statement (line 715)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 716):
    
    # Assigning a Dict to a Name (line 716):
    
    # Obtaining an instance of the builtin type 'dict' (line 716)
    dict_121683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 20), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 716)
    
    # Assigning a type to the variable 'variables' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'variables', dict_121683)
    # SSA branch for the else part of an if statement (line 715)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 718):
    
    # Assigning a Call to a Name (line 718):
    
    # Call to AttrDict(...): (line 718)
    # Processing the call keyword arguments (line 718)
    kwargs_121685 = {}
    # Getting the type of 'AttrDict' (line 718)
    AttrDict_121684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 20), 'AttrDict', False)
    # Calling AttrDict(args, kwargs) (line 718)
    AttrDict_call_result_121686 = invoke(stypy.reporting.localization.Localization(__file__, 718, 20), AttrDict_121684, *[], **kwargs_121685)
    
    # Assigning a type to the variable 'variables' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'variables', AttrDict_call_result_121686)
    # SSA join for if statement (line 715)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 721):
    
    # Assigning a Call to a Name (line 721):
    
    # Call to open(...): (line 721)
    # Processing the call arguments (line 721)
    # Getting the type of 'file_name' (line 721)
    file_name_121688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 13), 'file_name', False)
    str_121689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 24), 'str', 'rb')
    # Processing the call keyword arguments (line 721)
    kwargs_121690 = {}
    # Getting the type of 'open' (line 721)
    open_121687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'open', False)
    # Calling open(args, kwargs) (line 721)
    open_call_result_121691 = invoke(stypy.reporting.localization.Localization(__file__, 721, 8), open_121687, *[file_name_121688, str_121689], **kwargs_121690)
    
    # Assigning a type to the variable 'f' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'f', open_call_result_121691)
    
    # Assigning a Call to a Name (line 724):
    
    # Assigning a Call to a Name (line 724):
    
    # Call to _read_bytes(...): (line 724)
    # Processing the call arguments (line 724)
    # Getting the type of 'f' (line 724)
    f_121693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 28), 'f', False)
    int_121694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 31), 'int')
    # Processing the call keyword arguments (line 724)
    kwargs_121695 = {}
    # Getting the type of '_read_bytes' (line 724)
    _read_bytes_121692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 16), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 724)
    _read_bytes_call_result_121696 = invoke(stypy.reporting.localization.Localization(__file__, 724, 16), _read_bytes_121692, *[f_121693, int_121694], **kwargs_121695)
    
    # Assigning a type to the variable 'signature' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'signature', _read_bytes_call_result_121696)
    
    
    # Getting the type of 'signature' (line 725)
    signature_121697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 7), 'signature')
    str_121698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 20), 'str', 'SR')
    # Applying the binary operator '!=' (line 725)
    result_ne_121699 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 7), '!=', signature_121697, str_121698)
    
    # Testing the type of an if condition (line 725)
    if_condition_121700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 725, 4), result_ne_121699)
    # Assigning a type to the variable 'if_condition_121700' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'if_condition_121700', if_condition_121700)
    # SSA begins for if statement (line 725)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Exception(...): (line 726)
    # Processing the call arguments (line 726)
    str_121702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 24), 'str', 'Invalid SIGNATURE: %s')
    # Getting the type of 'signature' (line 726)
    signature_121703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 50), 'signature', False)
    # Applying the binary operator '%' (line 726)
    result_mod_121704 = python_operator(stypy.reporting.localization.Localization(__file__, 726, 24), '%', str_121702, signature_121703)
    
    # Processing the call keyword arguments (line 726)
    kwargs_121705 = {}
    # Getting the type of 'Exception' (line 726)
    Exception_121701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 726)
    Exception_call_result_121706 = invoke(stypy.reporting.localization.Localization(__file__, 726, 14), Exception_121701, *[result_mod_121704], **kwargs_121705)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 726, 8), Exception_call_result_121706, 'raise parameter', BaseException)
    # SSA join for if statement (line 725)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 730):
    
    # Assigning a Call to a Name (line 730):
    
    # Call to _read_bytes(...): (line 730)
    # Processing the call arguments (line 730)
    # Getting the type of 'f' (line 730)
    f_121708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 25), 'f', False)
    int_121709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 28), 'int')
    # Processing the call keyword arguments (line 730)
    kwargs_121710 = {}
    # Getting the type of '_read_bytes' (line 730)
    _read_bytes_121707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 13), '_read_bytes', False)
    # Calling _read_bytes(args, kwargs) (line 730)
    _read_bytes_call_result_121711 = invoke(stypy.reporting.localization.Localization(__file__, 730, 13), _read_bytes_121707, *[f_121708, int_121709], **kwargs_121710)
    
    # Assigning a type to the variable 'recfmt' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'recfmt', _read_bytes_call_result_121711)
    
    
    # Getting the type of 'recfmt' (line 732)
    recfmt_121712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 7), 'recfmt')
    str_121713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 17), 'str', '\x00\x04')
    # Applying the binary operator '==' (line 732)
    result_eq_121714 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 7), '==', recfmt_121712, str_121713)
    
    # Testing the type of an if condition (line 732)
    if_condition_121715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 732, 4), result_eq_121714)
    # Assigning a type to the variable 'if_condition_121715' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'if_condition_121715', if_condition_121715)
    # SSA begins for if statement (line 732)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 732)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'recfmt' (line 735)
    recfmt_121716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 9), 'recfmt')
    str_121717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 19), 'str', '\x00\x06')
    # Applying the binary operator '==' (line 735)
    result_eq_121718 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 9), '==', recfmt_121716, str_121717)
    
    # Testing the type of an if condition (line 735)
    if_condition_121719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 735, 9), result_eq_121718)
    # Assigning a type to the variable 'if_condition_121719' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 9), 'if_condition_121719', if_condition_121719)
    # SSA begins for if statement (line 735)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'verbose' (line 737)
    verbose_121720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 11), 'verbose')
    # Testing the type of an if condition (line 737)
    if_condition_121721 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 737, 8), verbose_121720)
    # Assigning a type to the variable 'if_condition_121721' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'if_condition_121721', if_condition_121721)
    # SSA begins for if statement (line 737)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 738)
    # Processing the call arguments (line 738)
    str_121723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 18), 'str', 'IDL Save file is compressed')
    # Processing the call keyword arguments (line 738)
    kwargs_121724 = {}
    # Getting the type of 'print' (line 738)
    print_121722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'print', False)
    # Calling print(args, kwargs) (line 738)
    print_call_result_121725 = invoke(stypy.reporting.localization.Localization(__file__, 738, 12), print_121722, *[str_121723], **kwargs_121724)
    
    # SSA join for if statement (line 737)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'uncompressed_file_name' (line 740)
    uncompressed_file_name_121726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 11), 'uncompressed_file_name')
    # Testing the type of an if condition (line 740)
    if_condition_121727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 740, 8), uncompressed_file_name_121726)
    # Assigning a type to the variable 'if_condition_121727' (line 740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), 'if_condition_121727', if_condition_121727)
    # SSA begins for if statement (line 740)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 741):
    
    # Assigning a Call to a Name (line 741):
    
    # Call to open(...): (line 741)
    # Processing the call arguments (line 741)
    # Getting the type of 'uncompressed_file_name' (line 741)
    uncompressed_file_name_121729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 24), 'uncompressed_file_name', False)
    str_121730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 48), 'str', 'w+b')
    # Processing the call keyword arguments (line 741)
    kwargs_121731 = {}
    # Getting the type of 'open' (line 741)
    open_121728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 19), 'open', False)
    # Calling open(args, kwargs) (line 741)
    open_call_result_121732 = invoke(stypy.reporting.localization.Localization(__file__, 741, 19), open_121728, *[uncompressed_file_name_121729, str_121730], **kwargs_121731)
    
    # Assigning a type to the variable 'fout' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'fout', open_call_result_121732)
    # SSA branch for the else part of an if statement (line 740)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 743):
    
    # Assigning a Call to a Name (line 743):
    
    # Call to NamedTemporaryFile(...): (line 743)
    # Processing the call keyword arguments (line 743)
    str_121735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 54), 'str', '.sav')
    keyword_121736 = str_121735
    kwargs_121737 = {'suffix': keyword_121736}
    # Getting the type of 'tempfile' (line 743)
    tempfile_121733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 19), 'tempfile', False)
    # Obtaining the member 'NamedTemporaryFile' of a type (line 743)
    NamedTemporaryFile_121734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 19), tempfile_121733, 'NamedTemporaryFile')
    # Calling NamedTemporaryFile(args, kwargs) (line 743)
    NamedTemporaryFile_call_result_121738 = invoke(stypy.reporting.localization.Localization(__file__, 743, 19), NamedTemporaryFile_121734, *[], **kwargs_121737)
    
    # Assigning a type to the variable 'fout' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 12), 'fout', NamedTemporaryFile_call_result_121738)
    # SSA join for if statement (line 740)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'verbose' (line 745)
    verbose_121739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 11), 'verbose')
    # Testing the type of an if condition (line 745)
    if_condition_121740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 745, 8), verbose_121739)
    # Assigning a type to the variable 'if_condition_121740' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'if_condition_121740', if_condition_121740)
    # SSA begins for if statement (line 745)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 746)
    # Processing the call arguments (line 746)
    str_121742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 18), 'str', ' -> expanding to %s')
    # Getting the type of 'fout' (line 746)
    fout_121743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 42), 'fout', False)
    # Obtaining the member 'name' of a type (line 746)
    name_121744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 42), fout_121743, 'name')
    # Applying the binary operator '%' (line 746)
    result_mod_121745 = python_operator(stypy.reporting.localization.Localization(__file__, 746, 18), '%', str_121742, name_121744)
    
    # Processing the call keyword arguments (line 746)
    kwargs_121746 = {}
    # Getting the type of 'print' (line 746)
    print_121741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 12), 'print', False)
    # Calling print(args, kwargs) (line 746)
    print_call_result_121747 = invoke(stypy.reporting.localization.Localization(__file__, 746, 12), print_121741, *[result_mod_121745], **kwargs_121746)
    
    # SSA join for if statement (line 745)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 749)
    # Processing the call arguments (line 749)
    str_121750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 19), 'str', 'SR\x00\x04')
    # Processing the call keyword arguments (line 749)
    kwargs_121751 = {}
    # Getting the type of 'fout' (line 749)
    fout_121748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'fout', False)
    # Obtaining the member 'write' of a type (line 749)
    write_121749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 8), fout_121748, 'write')
    # Calling write(args, kwargs) (line 749)
    write_call_result_121752 = invoke(stypy.reporting.localization.Localization(__file__, 749, 8), write_121749, *[str_121750], **kwargs_121751)
    
    
    # Getting the type of 'True' (line 752)
    True_121753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 14), 'True')
    # Testing the type of an if condition (line 752)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 752, 8), True_121753)
    # SSA begins for while statement (line 752)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 755):
    
    # Assigning a Call to a Name (line 755):
    
    # Call to _read_long(...): (line 755)
    # Processing the call arguments (line 755)
    # Getting the type of 'f' (line 755)
    f_121755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 33), 'f', False)
    # Processing the call keyword arguments (line 755)
    kwargs_121756 = {}
    # Getting the type of '_read_long' (line 755)
    _read_long_121754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 22), '_read_long', False)
    # Calling _read_long(args, kwargs) (line 755)
    _read_long_call_result_121757 = invoke(stypy.reporting.localization.Localization(__file__, 755, 22), _read_long_121754, *[f_121755], **kwargs_121756)
    
    # Assigning a type to the variable 'rectype' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'rectype', _read_long_call_result_121757)
    
    # Call to write(...): (line 756)
    # Processing the call arguments (line 756)
    
    # Call to pack(...): (line 756)
    # Processing the call arguments (line 756)
    str_121762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 35), 'str', '>l')
    
    # Call to int(...): (line 756)
    # Processing the call arguments (line 756)
    # Getting the type of 'rectype' (line 756)
    rectype_121764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 45), 'rectype', False)
    # Processing the call keyword arguments (line 756)
    kwargs_121765 = {}
    # Getting the type of 'int' (line 756)
    int_121763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 41), 'int', False)
    # Calling int(args, kwargs) (line 756)
    int_call_result_121766 = invoke(stypy.reporting.localization.Localization(__file__, 756, 41), int_121763, *[rectype_121764], **kwargs_121765)
    
    # Processing the call keyword arguments (line 756)
    kwargs_121767 = {}
    # Getting the type of 'struct' (line 756)
    struct_121760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 23), 'struct', False)
    # Obtaining the member 'pack' of a type (line 756)
    pack_121761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 23), struct_121760, 'pack')
    # Calling pack(args, kwargs) (line 756)
    pack_call_result_121768 = invoke(stypy.reporting.localization.Localization(__file__, 756, 23), pack_121761, *[str_121762, int_call_result_121766], **kwargs_121767)
    
    # Processing the call keyword arguments (line 756)
    kwargs_121769 = {}
    # Getting the type of 'fout' (line 756)
    fout_121758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'fout', False)
    # Obtaining the member 'write' of a type (line 756)
    write_121759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 12), fout_121758, 'write')
    # Calling write(args, kwargs) (line 756)
    write_call_result_121770 = invoke(stypy.reporting.localization.Localization(__file__, 756, 12), write_121759, *[pack_call_result_121768], **kwargs_121769)
    
    
    # Assigning a Call to a Name (line 759):
    
    # Assigning a Call to a Name (line 759):
    
    # Call to _read_uint32(...): (line 759)
    # Processing the call arguments (line 759)
    # Getting the type of 'f' (line 759)
    f_121772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 35), 'f', False)
    # Processing the call keyword arguments (line 759)
    kwargs_121773 = {}
    # Getting the type of '_read_uint32' (line 759)
    _read_uint32_121771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 22), '_read_uint32', False)
    # Calling _read_uint32(args, kwargs) (line 759)
    _read_uint32_call_result_121774 = invoke(stypy.reporting.localization.Localization(__file__, 759, 22), _read_uint32_121771, *[f_121772], **kwargs_121773)
    
    # Assigning a type to the variable 'nextrec' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'nextrec', _read_uint32_call_result_121774)
    
    # Getting the type of 'nextrec' (line 760)
    nextrec_121775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'nextrec')
    
    # Call to _read_uint32(...): (line 760)
    # Processing the call arguments (line 760)
    # Getting the type of 'f' (line 760)
    f_121777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 36), 'f', False)
    # Processing the call keyword arguments (line 760)
    kwargs_121778 = {}
    # Getting the type of '_read_uint32' (line 760)
    _read_uint32_121776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 23), '_read_uint32', False)
    # Calling _read_uint32(args, kwargs) (line 760)
    _read_uint32_call_result_121779 = invoke(stypy.reporting.localization.Localization(__file__, 760, 23), _read_uint32_121776, *[f_121777], **kwargs_121778)
    
    int_121780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 41), 'int')
    int_121781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 44), 'int')
    # Applying the binary operator '**' (line 760)
    result_pow_121782 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 41), '**', int_121780, int_121781)
    
    # Applying the binary operator '*' (line 760)
    result_mul_121783 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 23), '*', _read_uint32_call_result_121779, result_pow_121782)
    
    # Applying the binary operator '+=' (line 760)
    result_iadd_121784 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 12), '+=', nextrec_121775, result_mul_121783)
    # Assigning a type to the variable 'nextrec' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'nextrec', result_iadd_121784)
    
    
    # Assigning a Call to a Name (line 763):
    
    # Assigning a Call to a Name (line 763):
    
    # Call to read(...): (line 763)
    # Processing the call arguments (line 763)
    int_121787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 29), 'int')
    # Processing the call keyword arguments (line 763)
    kwargs_121788 = {}
    # Getting the type of 'f' (line 763)
    f_121785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 22), 'f', False)
    # Obtaining the member 'read' of a type (line 763)
    read_121786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 22), f_121785, 'read')
    # Calling read(args, kwargs) (line 763)
    read_call_result_121789 = invoke(stypy.reporting.localization.Localization(__file__, 763, 22), read_121786, *[int_121787], **kwargs_121788)
    
    # Assigning a type to the variable 'unknown' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'unknown', read_call_result_121789)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'rectype' (line 766)
    rectype_121790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 28), 'rectype')
    # Getting the type of 'RECTYPE_DICT' (line 766)
    RECTYPE_DICT_121791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 15), 'RECTYPE_DICT')
    # Obtaining the member '__getitem__' of a type (line 766)
    getitem___121792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 15), RECTYPE_DICT_121791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 766)
    subscript_call_result_121793 = invoke(stypy.reporting.localization.Localization(__file__, 766, 15), getitem___121792, rectype_121790)
    
    str_121794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 40), 'str', 'END_MARKER')
    # Applying the binary operator '==' (line 766)
    result_eq_121795 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 15), '==', subscript_call_result_121793, str_121794)
    
    # Testing the type of an if condition (line 766)
    if_condition_121796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 12), result_eq_121795)
    # Assigning a type to the variable 'if_condition_121796' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 12), 'if_condition_121796', if_condition_121796)
    # SSA begins for if statement (line 766)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to write(...): (line 767)
    # Processing the call arguments (line 767)
    
    # Call to pack(...): (line 767)
    # Processing the call arguments (line 767)
    str_121801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 39), 'str', '>I')
    
    # Call to int(...): (line 767)
    # Processing the call arguments (line 767)
    # Getting the type of 'nextrec' (line 767)
    nextrec_121803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 49), 'nextrec', False)
    # Processing the call keyword arguments (line 767)
    kwargs_121804 = {}
    # Getting the type of 'int' (line 767)
    int_121802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 45), 'int', False)
    # Calling int(args, kwargs) (line 767)
    int_call_result_121805 = invoke(stypy.reporting.localization.Localization(__file__, 767, 45), int_121802, *[nextrec_121803], **kwargs_121804)
    
    int_121806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 60), 'int')
    int_121807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 63), 'int')
    # Applying the binary operator '**' (line 767)
    result_pow_121808 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 60), '**', int_121806, int_121807)
    
    # Applying the binary operator '%' (line 767)
    result_mod_121809 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 45), '%', int_call_result_121805, result_pow_121808)
    
    # Processing the call keyword arguments (line 767)
    kwargs_121810 = {}
    # Getting the type of 'struct' (line 767)
    struct_121799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 27), 'struct', False)
    # Obtaining the member 'pack' of a type (line 767)
    pack_121800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 27), struct_121799, 'pack')
    # Calling pack(args, kwargs) (line 767)
    pack_call_result_121811 = invoke(stypy.reporting.localization.Localization(__file__, 767, 27), pack_121800, *[str_121801, result_mod_121809], **kwargs_121810)
    
    # Processing the call keyword arguments (line 767)
    kwargs_121812 = {}
    # Getting the type of 'fout' (line 767)
    fout_121797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'fout', False)
    # Obtaining the member 'write' of a type (line 767)
    write_121798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 16), fout_121797, 'write')
    # Calling write(args, kwargs) (line 767)
    write_call_result_121813 = invoke(stypy.reporting.localization.Localization(__file__, 767, 16), write_121798, *[pack_call_result_121811], **kwargs_121812)
    
    
    # Call to write(...): (line 768)
    # Processing the call arguments (line 768)
    
    # Call to pack(...): (line 768)
    # Processing the call arguments (line 768)
    str_121818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 39), 'str', '>I')
    
    # Call to int(...): (line 768)
    # Processing the call arguments (line 768)
    # Getting the type of 'nextrec' (line 768)
    nextrec_121820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 50), 'nextrec', False)
    # Getting the type of 'nextrec' (line 768)
    nextrec_121821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 61), 'nextrec', False)
    int_121822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 71), 'int')
    int_121823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 74), 'int')
    # Applying the binary operator '**' (line 768)
    result_pow_121824 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 71), '**', int_121822, int_121823)
    
    # Applying the binary operator '%' (line 768)
    result_mod_121825 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 61), '%', nextrec_121821, result_pow_121824)
    
    # Applying the binary operator '-' (line 768)
    result_sub_121826 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 50), '-', nextrec_121820, result_mod_121825)
    
    int_121827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 81), 'int')
    int_121828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 84), 'int')
    # Applying the binary operator '**' (line 768)
    result_pow_121829 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 81), '**', int_121827, int_121828)
    
    # Applying the binary operator 'div' (line 768)
    result_div_121830 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 49), 'div', result_sub_121826, result_pow_121829)
    
    # Processing the call keyword arguments (line 768)
    kwargs_121831 = {}
    # Getting the type of 'int' (line 768)
    int_121819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 45), 'int', False)
    # Calling int(args, kwargs) (line 768)
    int_call_result_121832 = invoke(stypy.reporting.localization.Localization(__file__, 768, 45), int_121819, *[result_div_121830], **kwargs_121831)
    
    # Processing the call keyword arguments (line 768)
    kwargs_121833 = {}
    # Getting the type of 'struct' (line 768)
    struct_121816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 27), 'struct', False)
    # Obtaining the member 'pack' of a type (line 768)
    pack_121817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 27), struct_121816, 'pack')
    # Calling pack(args, kwargs) (line 768)
    pack_call_result_121834 = invoke(stypy.reporting.localization.Localization(__file__, 768, 27), pack_121817, *[str_121818, int_call_result_121832], **kwargs_121833)
    
    # Processing the call keyword arguments (line 768)
    kwargs_121835 = {}
    # Getting the type of 'fout' (line 768)
    fout_121814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 16), 'fout', False)
    # Obtaining the member 'write' of a type (line 768)
    write_121815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 16), fout_121814, 'write')
    # Calling write(args, kwargs) (line 768)
    write_call_result_121836 = invoke(stypy.reporting.localization.Localization(__file__, 768, 16), write_121815, *[pack_call_result_121834], **kwargs_121835)
    
    
    # Call to write(...): (line 769)
    # Processing the call arguments (line 769)
    # Getting the type of 'unknown' (line 769)
    unknown_121839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 27), 'unknown', False)
    # Processing the call keyword arguments (line 769)
    kwargs_121840 = {}
    # Getting the type of 'fout' (line 769)
    fout_121837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 16), 'fout', False)
    # Obtaining the member 'write' of a type (line 769)
    write_121838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 16), fout_121837, 'write')
    # Calling write(args, kwargs) (line 769)
    write_call_result_121841 = invoke(stypy.reporting.localization.Localization(__file__, 769, 16), write_121838, *[unknown_121839], **kwargs_121840)
    
    # SSA join for if statement (line 766)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 773):
    
    # Assigning a Call to a Name (line 773):
    
    # Call to tell(...): (line 773)
    # Processing the call keyword arguments (line 773)
    kwargs_121844 = {}
    # Getting the type of 'f' (line 773)
    f_121842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 18), 'f', False)
    # Obtaining the member 'tell' of a type (line 773)
    tell_121843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 18), f_121842, 'tell')
    # Calling tell(args, kwargs) (line 773)
    tell_call_result_121845 = invoke(stypy.reporting.localization.Localization(__file__, 773, 18), tell_121843, *[], **kwargs_121844)
    
    # Assigning a type to the variable 'pos' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 12), 'pos', tell_call_result_121845)
    
    # Assigning a Call to a Name (line 776):
    
    # Assigning a Call to a Name (line 776):
    
    # Call to decompress(...): (line 776)
    # Processing the call arguments (line 776)
    
    # Call to read(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'nextrec' (line 776)
    nextrec_121850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 48), 'nextrec', False)
    # Getting the type of 'pos' (line 776)
    pos_121851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 56), 'pos', False)
    # Applying the binary operator '-' (line 776)
    result_sub_121852 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 48), '-', nextrec_121850, pos_121851)
    
    # Processing the call keyword arguments (line 776)
    kwargs_121853 = {}
    # Getting the type of 'f' (line 776)
    f_121848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 41), 'f', False)
    # Obtaining the member 'read' of a type (line 776)
    read_121849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 41), f_121848, 'read')
    # Calling read(args, kwargs) (line 776)
    read_call_result_121854 = invoke(stypy.reporting.localization.Localization(__file__, 776, 41), read_121849, *[result_sub_121852], **kwargs_121853)
    
    # Processing the call keyword arguments (line 776)
    kwargs_121855 = {}
    # Getting the type of 'zlib' (line 776)
    zlib_121846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 25), 'zlib', False)
    # Obtaining the member 'decompress' of a type (line 776)
    decompress_121847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 25), zlib_121846, 'decompress')
    # Calling decompress(args, kwargs) (line 776)
    decompress_call_result_121856 = invoke(stypy.reporting.localization.Localization(__file__, 776, 25), decompress_121847, *[read_call_result_121854], **kwargs_121855)
    
    # Assigning a type to the variable 'rec_string' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 12), 'rec_string', decompress_call_result_121856)
    
    # Assigning a BinOp to a Name (line 779):
    
    # Assigning a BinOp to a Name (line 779):
    
    # Call to tell(...): (line 779)
    # Processing the call keyword arguments (line 779)
    kwargs_121859 = {}
    # Getting the type of 'fout' (line 779)
    fout_121857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 22), 'fout', False)
    # Obtaining the member 'tell' of a type (line 779)
    tell_121858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 22), fout_121857, 'tell')
    # Calling tell(args, kwargs) (line 779)
    tell_call_result_121860 = invoke(stypy.reporting.localization.Localization(__file__, 779, 22), tell_121858, *[], **kwargs_121859)
    
    
    # Call to len(...): (line 779)
    # Processing the call arguments (line 779)
    # Getting the type of 'rec_string' (line 779)
    rec_string_121862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 40), 'rec_string', False)
    # Processing the call keyword arguments (line 779)
    kwargs_121863 = {}
    # Getting the type of 'len' (line 779)
    len_121861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 36), 'len', False)
    # Calling len(args, kwargs) (line 779)
    len_call_result_121864 = invoke(stypy.reporting.localization.Localization(__file__, 779, 36), len_121861, *[rec_string_121862], **kwargs_121863)
    
    # Applying the binary operator '+' (line 779)
    result_add_121865 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 22), '+', tell_call_result_121860, len_call_result_121864)
    
    int_121866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 54), 'int')
    # Applying the binary operator '+' (line 779)
    result_add_121867 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 52), '+', result_add_121865, int_121866)
    
    # Assigning a type to the variable 'nextrec' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 12), 'nextrec', result_add_121867)
    
    # Call to write(...): (line 782)
    # Processing the call arguments (line 782)
    
    # Call to pack(...): (line 782)
    # Processing the call arguments (line 782)
    str_121872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 35), 'str', '>I')
    
    # Call to int(...): (line 782)
    # Processing the call arguments (line 782)
    # Getting the type of 'nextrec' (line 782)
    nextrec_121874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 45), 'nextrec', False)
    int_121875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 55), 'int')
    int_121876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 58), 'int')
    # Applying the binary operator '**' (line 782)
    result_pow_121877 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 55), '**', int_121875, int_121876)
    
    # Applying the binary operator '%' (line 782)
    result_mod_121878 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 45), '%', nextrec_121874, result_pow_121877)
    
    # Processing the call keyword arguments (line 782)
    kwargs_121879 = {}
    # Getting the type of 'int' (line 782)
    int_121873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 41), 'int', False)
    # Calling int(args, kwargs) (line 782)
    int_call_result_121880 = invoke(stypy.reporting.localization.Localization(__file__, 782, 41), int_121873, *[result_mod_121878], **kwargs_121879)
    
    # Processing the call keyword arguments (line 782)
    kwargs_121881 = {}
    # Getting the type of 'struct' (line 782)
    struct_121870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 23), 'struct', False)
    # Obtaining the member 'pack' of a type (line 782)
    pack_121871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 23), struct_121870, 'pack')
    # Calling pack(args, kwargs) (line 782)
    pack_call_result_121882 = invoke(stypy.reporting.localization.Localization(__file__, 782, 23), pack_121871, *[str_121872, int_call_result_121880], **kwargs_121881)
    
    # Processing the call keyword arguments (line 782)
    kwargs_121883 = {}
    # Getting the type of 'fout' (line 782)
    fout_121868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 12), 'fout', False)
    # Obtaining the member 'write' of a type (line 782)
    write_121869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 12), fout_121868, 'write')
    # Calling write(args, kwargs) (line 782)
    write_call_result_121884 = invoke(stypy.reporting.localization.Localization(__file__, 782, 12), write_121869, *[pack_call_result_121882], **kwargs_121883)
    
    
    # Call to write(...): (line 783)
    # Processing the call arguments (line 783)
    
    # Call to pack(...): (line 783)
    # Processing the call arguments (line 783)
    str_121889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 35), 'str', '>I')
    
    # Call to int(...): (line 783)
    # Processing the call arguments (line 783)
    # Getting the type of 'nextrec' (line 783)
    nextrec_121891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 46), 'nextrec', False)
    # Getting the type of 'nextrec' (line 783)
    nextrec_121892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 57), 'nextrec', False)
    int_121893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 67), 'int')
    int_121894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 70), 'int')
    # Applying the binary operator '**' (line 783)
    result_pow_121895 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 67), '**', int_121893, int_121894)
    
    # Applying the binary operator '%' (line 783)
    result_mod_121896 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 57), '%', nextrec_121892, result_pow_121895)
    
    # Applying the binary operator '-' (line 783)
    result_sub_121897 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 46), '-', nextrec_121891, result_mod_121896)
    
    int_121898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 77), 'int')
    int_121899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 80), 'int')
    # Applying the binary operator '**' (line 783)
    result_pow_121900 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 77), '**', int_121898, int_121899)
    
    # Applying the binary operator 'div' (line 783)
    result_div_121901 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 45), 'div', result_sub_121897, result_pow_121900)
    
    # Processing the call keyword arguments (line 783)
    kwargs_121902 = {}
    # Getting the type of 'int' (line 783)
    int_121890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 41), 'int', False)
    # Calling int(args, kwargs) (line 783)
    int_call_result_121903 = invoke(stypy.reporting.localization.Localization(__file__, 783, 41), int_121890, *[result_div_121901], **kwargs_121902)
    
    # Processing the call keyword arguments (line 783)
    kwargs_121904 = {}
    # Getting the type of 'struct' (line 783)
    struct_121887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 23), 'struct', False)
    # Obtaining the member 'pack' of a type (line 783)
    pack_121888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 23), struct_121887, 'pack')
    # Calling pack(args, kwargs) (line 783)
    pack_call_result_121905 = invoke(stypy.reporting.localization.Localization(__file__, 783, 23), pack_121888, *[str_121889, int_call_result_121903], **kwargs_121904)
    
    # Processing the call keyword arguments (line 783)
    kwargs_121906 = {}
    # Getting the type of 'fout' (line 783)
    fout_121885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'fout', False)
    # Obtaining the member 'write' of a type (line 783)
    write_121886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), fout_121885, 'write')
    # Calling write(args, kwargs) (line 783)
    write_call_result_121907 = invoke(stypy.reporting.localization.Localization(__file__, 783, 12), write_121886, *[pack_call_result_121905], **kwargs_121906)
    
    
    # Call to write(...): (line 784)
    # Processing the call arguments (line 784)
    # Getting the type of 'unknown' (line 784)
    unknown_121910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 23), 'unknown', False)
    # Processing the call keyword arguments (line 784)
    kwargs_121911 = {}
    # Getting the type of 'fout' (line 784)
    fout_121908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'fout', False)
    # Obtaining the member 'write' of a type (line 784)
    write_121909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 12), fout_121908, 'write')
    # Calling write(args, kwargs) (line 784)
    write_call_result_121912 = invoke(stypy.reporting.localization.Localization(__file__, 784, 12), write_121909, *[unknown_121910], **kwargs_121911)
    
    
    # Call to write(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 'rec_string' (line 785)
    rec_string_121915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 23), 'rec_string', False)
    # Processing the call keyword arguments (line 785)
    kwargs_121916 = {}
    # Getting the type of 'fout' (line 785)
    fout_121913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'fout', False)
    # Obtaining the member 'write' of a type (line 785)
    write_121914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 12), fout_121913, 'write')
    # Calling write(args, kwargs) (line 785)
    write_call_result_121917 = invoke(stypy.reporting.localization.Localization(__file__, 785, 12), write_121914, *[rec_string_121915], **kwargs_121916)
    
    # SSA join for while statement (line 752)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 788)
    # Processing the call keyword arguments (line 788)
    kwargs_121920 = {}
    # Getting the type of 'f' (line 788)
    f_121918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'f', False)
    # Obtaining the member 'close' of a type (line 788)
    close_121919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 8), f_121918, 'close')
    # Calling close(args, kwargs) (line 788)
    close_call_result_121921 = invoke(stypy.reporting.localization.Localization(__file__, 788, 8), close_121919, *[], **kwargs_121920)
    
    
    # Assigning a Name to a Name (line 791):
    
    # Assigning a Name to a Name (line 791):
    # Getting the type of 'fout' (line 791)
    fout_121922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'fout')
    # Assigning a type to the variable 'f' (line 791)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 8), 'f', fout_121922)
    
    # Call to seek(...): (line 792)
    # Processing the call arguments (line 792)
    int_121925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 15), 'int')
    # Processing the call keyword arguments (line 792)
    kwargs_121926 = {}
    # Getting the type of 'f' (line 792)
    f_121923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'f', False)
    # Obtaining the member 'seek' of a type (line 792)
    seek_121924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), f_121923, 'seek')
    # Calling seek(args, kwargs) (line 792)
    seek_call_result_121927 = invoke(stypy.reporting.localization.Localization(__file__, 792, 8), seek_121924, *[int_121925], **kwargs_121926)
    
    # SSA branch for the else part of an if statement (line 735)
    module_type_store.open_ssa_branch('else')
    
    # Call to Exception(...): (line 795)
    # Processing the call arguments (line 795)
    str_121929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 24), 'str', 'Invalid RECFMT: %s')
    # Getting the type of 'recfmt' (line 795)
    recfmt_121930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 47), 'recfmt', False)
    # Applying the binary operator '%' (line 795)
    result_mod_121931 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 24), '%', str_121929, recfmt_121930)
    
    # Processing the call keyword arguments (line 795)
    kwargs_121932 = {}
    # Getting the type of 'Exception' (line 795)
    Exception_121928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 14), 'Exception', False)
    # Calling Exception(args, kwargs) (line 795)
    Exception_call_result_121933 = invoke(stypy.reporting.localization.Localization(__file__, 795, 14), Exception_121928, *[result_mod_121931], **kwargs_121932)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 795, 8), Exception_call_result_121933, 'raise parameter', BaseException)
    # SSA join for if statement (line 735)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 732)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'True' (line 798)
    True_121934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 10), 'True')
    # Testing the type of an if condition (line 798)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 798, 4), True_121934)
    # SSA begins for while statement (line 798)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 799):
    
    # Assigning a Call to a Name (line 799):
    
    # Call to _read_record(...): (line 799)
    # Processing the call arguments (line 799)
    # Getting the type of 'f' (line 799)
    f_121936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 25), 'f', False)
    # Processing the call keyword arguments (line 799)
    kwargs_121937 = {}
    # Getting the type of '_read_record' (line 799)
    _read_record_121935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 12), '_read_record', False)
    # Calling _read_record(args, kwargs) (line 799)
    _read_record_call_result_121938 = invoke(stypy.reporting.localization.Localization(__file__, 799, 12), _read_record_121935, *[f_121936], **kwargs_121937)
    
    # Assigning a type to the variable 'r' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'r', _read_record_call_result_121938)
    
    # Call to append(...): (line 800)
    # Processing the call arguments (line 800)
    # Getting the type of 'r' (line 800)
    r_121941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 23), 'r', False)
    # Processing the call keyword arguments (line 800)
    kwargs_121942 = {}
    # Getting the type of 'records' (line 800)
    records_121939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'records', False)
    # Obtaining the member 'append' of a type (line 800)
    append_121940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 8), records_121939, 'append')
    # Calling append(args, kwargs) (line 800)
    append_call_result_121943 = invoke(stypy.reporting.localization.Localization(__file__, 800, 8), append_121940, *[r_121941], **kwargs_121942)
    
    
    
    str_121944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 11), 'str', 'end')
    # Getting the type of 'r' (line 801)
    r_121945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 20), 'r')
    # Applying the binary operator 'in' (line 801)
    result_contains_121946 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 11), 'in', str_121944, r_121945)
    
    # Testing the type of an if condition (line 801)
    if_condition_121947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 801, 8), result_contains_121946)
    # Assigning a type to the variable 'if_condition_121947' (line 801)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'if_condition_121947', if_condition_121947)
    # SSA begins for if statement (line 801)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_121948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 17), 'str', 'end')
    # Getting the type of 'r' (line 802)
    r_121949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 15), 'r')
    # Obtaining the member '__getitem__' of a type (line 802)
    getitem___121950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 15), r_121949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 802)
    subscript_call_result_121951 = invoke(stypy.reporting.localization.Localization(__file__, 802, 15), getitem___121950, str_121948)
    
    # Testing the type of an if condition (line 802)
    if_condition_121952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 802, 12), subscript_call_result_121951)
    # Assigning a type to the variable 'if_condition_121952' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 12), 'if_condition_121952', if_condition_121952)
    # SSA begins for if statement (line 802)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 802)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 801)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 798)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 806)
    # Processing the call keyword arguments (line 806)
    kwargs_121955 = {}
    # Getting the type of 'f' (line 806)
    f_121953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 806)
    close_121954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 4), f_121953, 'close')
    # Calling close(args, kwargs) (line 806)
    close_call_result_121956 = invoke(stypy.reporting.localization.Localization(__file__, 806, 4), close_121954, *[], **kwargs_121955)
    
    
    # Assigning a Dict to a Name (line 809):
    
    # Assigning a Dict to a Name (line 809):
    
    # Obtaining an instance of the builtin type 'dict' (line 809)
    dict_121957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 809)
    
    # Assigning a type to the variable 'heap' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'heap', dict_121957)
    
    # Getting the type of 'records' (line 810)
    records_121958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 13), 'records')
    # Testing the type of a for loop iterable (line 810)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 810, 4), records_121958)
    # Getting the type of the for loop variable (line 810)
    for_loop_var_121959 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 810, 4), records_121958)
    # Assigning a type to the variable 'r' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'r', for_loop_var_121959)
    # SSA begins for a for statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_121960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 13), 'str', 'rectype')
    # Getting the type of 'r' (line 811)
    r_121961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 11), 'r')
    # Obtaining the member '__getitem__' of a type (line 811)
    getitem___121962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 11), r_121961, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 811)
    subscript_call_result_121963 = invoke(stypy.reporting.localization.Localization(__file__, 811, 11), getitem___121962, str_121960)
    
    str_121964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 27), 'str', 'HEAP_DATA')
    # Applying the binary operator '==' (line 811)
    result_eq_121965 = python_operator(stypy.reporting.localization.Localization(__file__, 811, 11), '==', subscript_call_result_121963, str_121964)
    
    # Testing the type of an if condition (line 811)
    if_condition_121966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 811, 8), result_eq_121965)
    # Assigning a type to the variable 'if_condition_121966' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'if_condition_121966', if_condition_121966)
    # SSA begins for if statement (line 811)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 812):
    
    # Assigning a Subscript to a Subscript (line 812):
    
    # Obtaining the type of the subscript
    str_121967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 38), 'str', 'data')
    # Getting the type of 'r' (line 812)
    r_121968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 36), 'r')
    # Obtaining the member '__getitem__' of a type (line 812)
    getitem___121969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 36), r_121968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 812)
    subscript_call_result_121970 = invoke(stypy.reporting.localization.Localization(__file__, 812, 36), getitem___121969, str_121967)
    
    # Getting the type of 'heap' (line 812)
    heap_121971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'heap')
    
    # Obtaining the type of the subscript
    str_121972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 19), 'str', 'heap_index')
    # Getting the type of 'r' (line 812)
    r_121973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 17), 'r')
    # Obtaining the member '__getitem__' of a type (line 812)
    getitem___121974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 17), r_121973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 812)
    subscript_call_result_121975 = invoke(stypy.reporting.localization.Localization(__file__, 812, 17), getitem___121974, str_121972)
    
    # Storing an element on a container (line 812)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 812, 12), heap_121971, (subscript_call_result_121975, subscript_call_result_121970))
    # SSA join for if statement (line 811)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'records' (line 815)
    records_121976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 13), 'records')
    # Testing the type of a for loop iterable (line 815)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 815, 4), records_121976)
    # Getting the type of the for loop variable (line 815)
    for_loop_var_121977 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 815, 4), records_121976)
    # Assigning a type to the variable 'r' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'r', for_loop_var_121977)
    # SSA begins for a for statement (line 815)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_121978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 13), 'str', 'rectype')
    # Getting the type of 'r' (line 816)
    r_121979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 11), 'r')
    # Obtaining the member '__getitem__' of a type (line 816)
    getitem___121980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 11), r_121979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 816)
    subscript_call_result_121981 = invoke(stypy.reporting.localization.Localization(__file__, 816, 11), getitem___121980, str_121978)
    
    str_121982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 27), 'str', 'VARIABLE')
    # Applying the binary operator '==' (line 816)
    result_eq_121983 = python_operator(stypy.reporting.localization.Localization(__file__, 816, 11), '==', subscript_call_result_121981, str_121982)
    
    # Testing the type of an if condition (line 816)
    if_condition_121984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 816, 8), result_eq_121983)
    # Assigning a type to the variable 'if_condition_121984' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'if_condition_121984', if_condition_121984)
    # SSA begins for if statement (line 816)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 817):
    
    # Assigning a Subscript to a Name (line 817):
    
    # Obtaining the type of the subscript
    int_121985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 12), 'int')
    
    # Call to _replace_heap(...): (line 817)
    # Processing the call arguments (line 817)
    
    # Obtaining the type of the subscript
    str_121987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 43), 'str', 'data')
    # Getting the type of 'r' (line 817)
    r_121988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 41), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 817)
    getitem___121989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 41), r_121988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 817)
    subscript_call_result_121990 = invoke(stypy.reporting.localization.Localization(__file__, 817, 41), getitem___121989, str_121987)
    
    # Getting the type of 'heap' (line 817)
    heap_121991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 52), 'heap', False)
    # Processing the call keyword arguments (line 817)
    kwargs_121992 = {}
    # Getting the type of '_replace_heap' (line 817)
    _replace_heap_121986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 27), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 817)
    _replace_heap_call_result_121993 = invoke(stypy.reporting.localization.Localization(__file__, 817, 27), _replace_heap_121986, *[subscript_call_result_121990, heap_121991], **kwargs_121992)
    
    # Obtaining the member '__getitem__' of a type (line 817)
    getitem___121994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 12), _replace_heap_call_result_121993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 817)
    subscript_call_result_121995 = invoke(stypy.reporting.localization.Localization(__file__, 817, 12), getitem___121994, int_121985)
    
    # Assigning a type to the variable 'tuple_var_assignment_119610' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'tuple_var_assignment_119610', subscript_call_result_121995)
    
    # Assigning a Subscript to a Name (line 817):
    
    # Obtaining the type of the subscript
    int_121996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 12), 'int')
    
    # Call to _replace_heap(...): (line 817)
    # Processing the call arguments (line 817)
    
    # Obtaining the type of the subscript
    str_121998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 43), 'str', 'data')
    # Getting the type of 'r' (line 817)
    r_121999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 41), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 817)
    getitem___122000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 41), r_121999, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 817)
    subscript_call_result_122001 = invoke(stypy.reporting.localization.Localization(__file__, 817, 41), getitem___122000, str_121998)
    
    # Getting the type of 'heap' (line 817)
    heap_122002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 52), 'heap', False)
    # Processing the call keyword arguments (line 817)
    kwargs_122003 = {}
    # Getting the type of '_replace_heap' (line 817)
    _replace_heap_121997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 27), '_replace_heap', False)
    # Calling _replace_heap(args, kwargs) (line 817)
    _replace_heap_call_result_122004 = invoke(stypy.reporting.localization.Localization(__file__, 817, 27), _replace_heap_121997, *[subscript_call_result_122001, heap_122002], **kwargs_122003)
    
    # Obtaining the member '__getitem__' of a type (line 817)
    getitem___122005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 12), _replace_heap_call_result_122004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 817)
    subscript_call_result_122006 = invoke(stypy.reporting.localization.Localization(__file__, 817, 12), getitem___122005, int_121996)
    
    # Assigning a type to the variable 'tuple_var_assignment_119611' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'tuple_var_assignment_119611', subscript_call_result_122006)
    
    # Assigning a Name to a Name (line 817):
    # Getting the type of 'tuple_var_assignment_119610' (line 817)
    tuple_var_assignment_119610_122007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'tuple_var_assignment_119610')
    # Assigning a type to the variable 'replace' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'replace', tuple_var_assignment_119610_122007)
    
    # Assigning a Name to a Name (line 817):
    # Getting the type of 'tuple_var_assignment_119611' (line 817)
    tuple_var_assignment_119611_122008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'tuple_var_assignment_119611')
    # Assigning a type to the variable 'new' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 21), 'new', tuple_var_assignment_119611_122008)
    
    # Getting the type of 'replace' (line 818)
    replace_122009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 15), 'replace')
    # Testing the type of an if condition (line 818)
    if_condition_122010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 818, 12), replace_122009)
    # Assigning a type to the variable 'if_condition_122010' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'if_condition_122010', if_condition_122010)
    # SSA begins for if statement (line 818)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 819):
    
    # Assigning a Name to a Subscript (line 819):
    # Getting the type of 'new' (line 819)
    new_122011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 28), 'new')
    # Getting the type of 'r' (line 819)
    r_122012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 16), 'r')
    str_122013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 18), 'str', 'data')
    # Storing an element on a container (line 819)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 819, 16), r_122012, (str_122013, new_122011))
    # SSA join for if statement (line 818)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 820):
    
    # Assigning a Subscript to a Subscript (line 820):
    
    # Obtaining the type of the subscript
    str_122014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 48), 'str', 'data')
    # Getting the type of 'r' (line 820)
    r_122015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 46), 'r')
    # Obtaining the member '__getitem__' of a type (line 820)
    getitem___122016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 46), r_122015, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 820)
    subscript_call_result_122017 = invoke(stypy.reporting.localization.Localization(__file__, 820, 46), getitem___122016, str_122014)
    
    # Getting the type of 'variables' (line 820)
    variables_122018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'variables')
    
    # Call to lower(...): (line 820)
    # Processing the call keyword arguments (line 820)
    kwargs_122024 = {}
    
    # Obtaining the type of the subscript
    str_122019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 24), 'str', 'varname')
    # Getting the type of 'r' (line 820)
    r_122020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 22), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 820)
    getitem___122021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 22), r_122020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 820)
    subscript_call_result_122022 = invoke(stypy.reporting.localization.Localization(__file__, 820, 22), getitem___122021, str_122019)
    
    # Obtaining the member 'lower' of a type (line 820)
    lower_122023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 22), subscript_call_result_122022, 'lower')
    # Calling lower(args, kwargs) (line 820)
    lower_call_result_122025 = invoke(stypy.reporting.localization.Localization(__file__, 820, 22), lower_122023, *[], **kwargs_122024)
    
    # Storing an element on a container (line 820)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 820, 12), variables_122018, (lower_call_result_122025, subscript_call_result_122017))
    # SSA join for if statement (line 816)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'verbose' (line 822)
    verbose_122026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 7), 'verbose')
    # Testing the type of an if condition (line 822)
    if_condition_122027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 822, 4), verbose_122026)
    # Assigning a type to the variable 'if_condition_122027' (line 822)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 4), 'if_condition_122027', if_condition_122027)
    # SSA begins for if statement (line 822)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'records' (line 825)
    records_122028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 22), 'records')
    # Testing the type of a for loop iterable (line 825)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 825, 8), records_122028)
    # Getting the type of the for loop variable (line 825)
    for_loop_var_122029 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 825, 8), records_122028)
    # Assigning a type to the variable 'record' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 8), 'record', for_loop_var_122029)
    # SSA begins for a for statement (line 825)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_122030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 22), 'str', 'rectype')
    # Getting the type of 'record' (line 826)
    record_122031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 15), 'record')
    # Obtaining the member '__getitem__' of a type (line 826)
    getitem___122032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 15), record_122031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 826)
    subscript_call_result_122033 = invoke(stypy.reporting.localization.Localization(__file__, 826, 15), getitem___122032, str_122030)
    
    str_122034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 36), 'str', 'TIMESTAMP')
    # Applying the binary operator '==' (line 826)
    result_eq_122035 = python_operator(stypy.reporting.localization.Localization(__file__, 826, 15), '==', subscript_call_result_122033, str_122034)
    
    # Testing the type of an if condition (line 826)
    if_condition_122036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 826, 12), result_eq_122035)
    # Assigning a type to the variable 'if_condition_122036' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 12), 'if_condition_122036', if_condition_122036)
    # SSA begins for if statement (line 826)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 827)
    # Processing the call arguments (line 827)
    str_122038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 22), 'str', '-')
    int_122039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 26), 'int')
    # Applying the binary operator '*' (line 827)
    result_mul_122040 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 22), '*', str_122038, int_122039)
    
    # Processing the call keyword arguments (line 827)
    kwargs_122041 = {}
    # Getting the type of 'print' (line 827)
    print_122037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 16), 'print', False)
    # Calling print(args, kwargs) (line 827)
    print_call_result_122042 = invoke(stypy.reporting.localization.Localization(__file__, 827, 16), print_122037, *[result_mul_122040], **kwargs_122041)
    
    
    # Call to print(...): (line 828)
    # Processing the call arguments (line 828)
    str_122044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 22), 'str', 'Date: %s')
    
    # Obtaining the type of the subscript
    str_122045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 42), 'str', 'date')
    # Getting the type of 'record' (line 828)
    record_122046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 35), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 828)
    getitem___122047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 35), record_122046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 828)
    subscript_call_result_122048 = invoke(stypy.reporting.localization.Localization(__file__, 828, 35), getitem___122047, str_122045)
    
    # Applying the binary operator '%' (line 828)
    result_mod_122049 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 22), '%', str_122044, subscript_call_result_122048)
    
    # Processing the call keyword arguments (line 828)
    kwargs_122050 = {}
    # Getting the type of 'print' (line 828)
    print_122043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 16), 'print', False)
    # Calling print(args, kwargs) (line 828)
    print_call_result_122051 = invoke(stypy.reporting.localization.Localization(__file__, 828, 16), print_122043, *[result_mod_122049], **kwargs_122050)
    
    
    # Call to print(...): (line 829)
    # Processing the call arguments (line 829)
    str_122053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 22), 'str', 'User: %s')
    
    # Obtaining the type of the subscript
    str_122054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 42), 'str', 'user')
    # Getting the type of 'record' (line 829)
    record_122055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 35), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 829)
    getitem___122056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 35), record_122055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 829)
    subscript_call_result_122057 = invoke(stypy.reporting.localization.Localization(__file__, 829, 35), getitem___122056, str_122054)
    
    # Applying the binary operator '%' (line 829)
    result_mod_122058 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 22), '%', str_122053, subscript_call_result_122057)
    
    # Processing the call keyword arguments (line 829)
    kwargs_122059 = {}
    # Getting the type of 'print' (line 829)
    print_122052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 16), 'print', False)
    # Calling print(args, kwargs) (line 829)
    print_call_result_122060 = invoke(stypy.reporting.localization.Localization(__file__, 829, 16), print_122052, *[result_mod_122058], **kwargs_122059)
    
    
    # Call to print(...): (line 830)
    # Processing the call arguments (line 830)
    str_122062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 22), 'str', 'Host: %s')
    
    # Obtaining the type of the subscript
    str_122063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 42), 'str', 'host')
    # Getting the type of 'record' (line 830)
    record_122064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 35), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 830)
    getitem___122065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 35), record_122064, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 830)
    subscript_call_result_122066 = invoke(stypy.reporting.localization.Localization(__file__, 830, 35), getitem___122065, str_122063)
    
    # Applying the binary operator '%' (line 830)
    result_mod_122067 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 22), '%', str_122062, subscript_call_result_122066)
    
    # Processing the call keyword arguments (line 830)
    kwargs_122068 = {}
    # Getting the type of 'print' (line 830)
    print_122061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 16), 'print', False)
    # Calling print(args, kwargs) (line 830)
    print_call_result_122069 = invoke(stypy.reporting.localization.Localization(__file__, 830, 16), print_122061, *[result_mod_122067], **kwargs_122068)
    
    # SSA join for if statement (line 826)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'records' (line 834)
    records_122070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 22), 'records')
    # Testing the type of a for loop iterable (line 834)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 834, 8), records_122070)
    # Getting the type of the for loop variable (line 834)
    for_loop_var_122071 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 834, 8), records_122070)
    # Assigning a type to the variable 'record' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'record', for_loop_var_122071)
    # SSA begins for a for statement (line 834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_122072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 22), 'str', 'rectype')
    # Getting the type of 'record' (line 835)
    record_122073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 15), 'record')
    # Obtaining the member '__getitem__' of a type (line 835)
    getitem___122074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 15), record_122073, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 835)
    subscript_call_result_122075 = invoke(stypy.reporting.localization.Localization(__file__, 835, 15), getitem___122074, str_122072)
    
    str_122076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 36), 'str', 'VERSION')
    # Applying the binary operator '==' (line 835)
    result_eq_122077 = python_operator(stypy.reporting.localization.Localization(__file__, 835, 15), '==', subscript_call_result_122075, str_122076)
    
    # Testing the type of an if condition (line 835)
    if_condition_122078 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 835, 12), result_eq_122077)
    # Assigning a type to the variable 'if_condition_122078' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 12), 'if_condition_122078', if_condition_122078)
    # SSA begins for if statement (line 835)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 836)
    # Processing the call arguments (line 836)
    str_122080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 22), 'str', '-')
    int_122081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 26), 'int')
    # Applying the binary operator '*' (line 836)
    result_mul_122082 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 22), '*', str_122080, int_122081)
    
    # Processing the call keyword arguments (line 836)
    kwargs_122083 = {}
    # Getting the type of 'print' (line 836)
    print_122079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 16), 'print', False)
    # Calling print(args, kwargs) (line 836)
    print_call_result_122084 = invoke(stypy.reporting.localization.Localization(__file__, 836, 16), print_122079, *[result_mul_122082], **kwargs_122083)
    
    
    # Call to print(...): (line 837)
    # Processing the call arguments (line 837)
    str_122086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 22), 'str', 'Format: %s')
    
    # Obtaining the type of the subscript
    str_122087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 44), 'str', 'format')
    # Getting the type of 'record' (line 837)
    record_122088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 37), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 837)
    getitem___122089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 37), record_122088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 837)
    subscript_call_result_122090 = invoke(stypy.reporting.localization.Localization(__file__, 837, 37), getitem___122089, str_122087)
    
    # Applying the binary operator '%' (line 837)
    result_mod_122091 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 22), '%', str_122086, subscript_call_result_122090)
    
    # Processing the call keyword arguments (line 837)
    kwargs_122092 = {}
    # Getting the type of 'print' (line 837)
    print_122085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 16), 'print', False)
    # Calling print(args, kwargs) (line 837)
    print_call_result_122093 = invoke(stypy.reporting.localization.Localization(__file__, 837, 16), print_122085, *[result_mod_122091], **kwargs_122092)
    
    
    # Call to print(...): (line 838)
    # Processing the call arguments (line 838)
    str_122095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 22), 'str', 'Architecture: %s')
    
    # Obtaining the type of the subscript
    str_122096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 50), 'str', 'arch')
    # Getting the type of 'record' (line 838)
    record_122097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 43), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 838)
    getitem___122098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 43), record_122097, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 838)
    subscript_call_result_122099 = invoke(stypy.reporting.localization.Localization(__file__, 838, 43), getitem___122098, str_122096)
    
    # Applying the binary operator '%' (line 838)
    result_mod_122100 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 22), '%', str_122095, subscript_call_result_122099)
    
    # Processing the call keyword arguments (line 838)
    kwargs_122101 = {}
    # Getting the type of 'print' (line 838)
    print_122094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 16), 'print', False)
    # Calling print(args, kwargs) (line 838)
    print_call_result_122102 = invoke(stypy.reporting.localization.Localization(__file__, 838, 16), print_122094, *[result_mod_122100], **kwargs_122101)
    
    
    # Call to print(...): (line 839)
    # Processing the call arguments (line 839)
    str_122104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 22), 'str', 'Operating System: %s')
    
    # Obtaining the type of the subscript
    str_122105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 54), 'str', 'os')
    # Getting the type of 'record' (line 839)
    record_122106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 47), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___122107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 47), record_122106, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_122108 = invoke(stypy.reporting.localization.Localization(__file__, 839, 47), getitem___122107, str_122105)
    
    # Applying the binary operator '%' (line 839)
    result_mod_122109 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 22), '%', str_122104, subscript_call_result_122108)
    
    # Processing the call keyword arguments (line 839)
    kwargs_122110 = {}
    # Getting the type of 'print' (line 839)
    print_122103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 16), 'print', False)
    # Calling print(args, kwargs) (line 839)
    print_call_result_122111 = invoke(stypy.reporting.localization.Localization(__file__, 839, 16), print_122103, *[result_mod_122109], **kwargs_122110)
    
    
    # Call to print(...): (line 840)
    # Processing the call arguments (line 840)
    str_122113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 22), 'str', 'IDL Version: %s')
    
    # Obtaining the type of the subscript
    str_122114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 49), 'str', 'release')
    # Getting the type of 'record' (line 840)
    record_122115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 42), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 840)
    getitem___122116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 42), record_122115, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 840)
    subscript_call_result_122117 = invoke(stypy.reporting.localization.Localization(__file__, 840, 42), getitem___122116, str_122114)
    
    # Applying the binary operator '%' (line 840)
    result_mod_122118 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 22), '%', str_122113, subscript_call_result_122117)
    
    # Processing the call keyword arguments (line 840)
    kwargs_122119 = {}
    # Getting the type of 'print' (line 840)
    print_122112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'print', False)
    # Calling print(args, kwargs) (line 840)
    print_call_result_122120 = invoke(stypy.reporting.localization.Localization(__file__, 840, 16), print_122112, *[result_mod_122118], **kwargs_122119)
    
    # SSA join for if statement (line 835)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'records' (line 844)
    records_122121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 22), 'records')
    # Testing the type of a for loop iterable (line 844)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 844, 8), records_122121)
    # Getting the type of the for loop variable (line 844)
    for_loop_var_122122 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 844, 8), records_122121)
    # Assigning a type to the variable 'record' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'record', for_loop_var_122122)
    # SSA begins for a for statement (line 844)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_122123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 22), 'str', 'rectype')
    # Getting the type of 'record' (line 845)
    record_122124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 15), 'record')
    # Obtaining the member '__getitem__' of a type (line 845)
    getitem___122125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 15), record_122124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 845)
    subscript_call_result_122126 = invoke(stypy.reporting.localization.Localization(__file__, 845, 15), getitem___122125, str_122123)
    
    str_122127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 36), 'str', 'IDENTIFICATON')
    # Applying the binary operator '==' (line 845)
    result_eq_122128 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 15), '==', subscript_call_result_122126, str_122127)
    
    # Testing the type of an if condition (line 845)
    if_condition_122129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 845, 12), result_eq_122128)
    # Assigning a type to the variable 'if_condition_122129' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'if_condition_122129', if_condition_122129)
    # SSA begins for if statement (line 845)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 846)
    # Processing the call arguments (line 846)
    str_122131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 22), 'str', '-')
    int_122132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 26), 'int')
    # Applying the binary operator '*' (line 846)
    result_mul_122133 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 22), '*', str_122131, int_122132)
    
    # Processing the call keyword arguments (line 846)
    kwargs_122134 = {}
    # Getting the type of 'print' (line 846)
    print_122130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 16), 'print', False)
    # Calling print(args, kwargs) (line 846)
    print_call_result_122135 = invoke(stypy.reporting.localization.Localization(__file__, 846, 16), print_122130, *[result_mul_122133], **kwargs_122134)
    
    
    # Call to print(...): (line 847)
    # Processing the call arguments (line 847)
    str_122137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 22), 'str', 'Author: %s')
    
    # Obtaining the type of the subscript
    str_122138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 44), 'str', 'author')
    # Getting the type of 'record' (line 847)
    record_122139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 37), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 847)
    getitem___122140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 37), record_122139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 847)
    subscript_call_result_122141 = invoke(stypy.reporting.localization.Localization(__file__, 847, 37), getitem___122140, str_122138)
    
    # Applying the binary operator '%' (line 847)
    result_mod_122142 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 22), '%', str_122137, subscript_call_result_122141)
    
    # Processing the call keyword arguments (line 847)
    kwargs_122143 = {}
    # Getting the type of 'print' (line 847)
    print_122136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 16), 'print', False)
    # Calling print(args, kwargs) (line 847)
    print_call_result_122144 = invoke(stypy.reporting.localization.Localization(__file__, 847, 16), print_122136, *[result_mod_122142], **kwargs_122143)
    
    
    # Call to print(...): (line 848)
    # Processing the call arguments (line 848)
    str_122146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 22), 'str', 'Title: %s')
    
    # Obtaining the type of the subscript
    str_122147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 43), 'str', 'title')
    # Getting the type of 'record' (line 848)
    record_122148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 36), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 848)
    getitem___122149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 36), record_122148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 848)
    subscript_call_result_122150 = invoke(stypy.reporting.localization.Localization(__file__, 848, 36), getitem___122149, str_122147)
    
    # Applying the binary operator '%' (line 848)
    result_mod_122151 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 22), '%', str_122146, subscript_call_result_122150)
    
    # Processing the call keyword arguments (line 848)
    kwargs_122152 = {}
    # Getting the type of 'print' (line 848)
    print_122145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 16), 'print', False)
    # Calling print(args, kwargs) (line 848)
    print_call_result_122153 = invoke(stypy.reporting.localization.Localization(__file__, 848, 16), print_122145, *[result_mod_122151], **kwargs_122152)
    
    
    # Call to print(...): (line 849)
    # Processing the call arguments (line 849)
    str_122155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 22), 'str', 'ID Code: %s')
    
    # Obtaining the type of the subscript
    str_122156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 45), 'str', 'idcode')
    # Getting the type of 'record' (line 849)
    record_122157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 38), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 849)
    getitem___122158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 38), record_122157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 849)
    subscript_call_result_122159 = invoke(stypy.reporting.localization.Localization(__file__, 849, 38), getitem___122158, str_122156)
    
    # Applying the binary operator '%' (line 849)
    result_mod_122160 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 22), '%', str_122155, subscript_call_result_122159)
    
    # Processing the call keyword arguments (line 849)
    kwargs_122161 = {}
    # Getting the type of 'print' (line 849)
    print_122154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 16), 'print', False)
    # Calling print(args, kwargs) (line 849)
    print_call_result_122162 = invoke(stypy.reporting.localization.Localization(__file__, 849, 16), print_122154, *[result_mod_122160], **kwargs_122161)
    
    # SSA join for if statement (line 845)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'records' (line 853)
    records_122163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 22), 'records')
    # Testing the type of a for loop iterable (line 853)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 853, 8), records_122163)
    # Getting the type of the for loop variable (line 853)
    for_loop_var_122164 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 853, 8), records_122163)
    # Assigning a type to the variable 'record' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'record', for_loop_var_122164)
    # SSA begins for a for statement (line 853)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    str_122165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 22), 'str', 'rectype')
    # Getting the type of 'record' (line 854)
    record_122166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 15), 'record')
    # Obtaining the member '__getitem__' of a type (line 854)
    getitem___122167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 15), record_122166, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 854)
    subscript_call_result_122168 = invoke(stypy.reporting.localization.Localization(__file__, 854, 15), getitem___122167, str_122165)
    
    str_122169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 36), 'str', 'DESCRIPTION')
    # Applying the binary operator '==' (line 854)
    result_eq_122170 = python_operator(stypy.reporting.localization.Localization(__file__, 854, 15), '==', subscript_call_result_122168, str_122169)
    
    # Testing the type of an if condition (line 854)
    if_condition_122171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 854, 12), result_eq_122170)
    # Assigning a type to the variable 'if_condition_122171' (line 854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'if_condition_122171', if_condition_122171)
    # SSA begins for if statement (line 854)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 855)
    # Processing the call arguments (line 855)
    str_122173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 22), 'str', '-')
    int_122174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 26), 'int')
    # Applying the binary operator '*' (line 855)
    result_mul_122175 = python_operator(stypy.reporting.localization.Localization(__file__, 855, 22), '*', str_122173, int_122174)
    
    # Processing the call keyword arguments (line 855)
    kwargs_122176 = {}
    # Getting the type of 'print' (line 855)
    print_122172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 16), 'print', False)
    # Calling print(args, kwargs) (line 855)
    print_call_result_122177 = invoke(stypy.reporting.localization.Localization(__file__, 855, 16), print_122172, *[result_mul_122175], **kwargs_122176)
    
    
    # Call to print(...): (line 856)
    # Processing the call arguments (line 856)
    str_122179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 22), 'str', 'Description: %s')
    
    # Obtaining the type of the subscript
    str_122180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 49), 'str', 'description')
    # Getting the type of 'record' (line 856)
    record_122181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 42), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 856)
    getitem___122182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 42), record_122181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 856)
    subscript_call_result_122183 = invoke(stypy.reporting.localization.Localization(__file__, 856, 42), getitem___122182, str_122180)
    
    # Applying the binary operator '%' (line 856)
    result_mod_122184 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 22), '%', str_122179, subscript_call_result_122183)
    
    # Processing the call keyword arguments (line 856)
    kwargs_122185 = {}
    # Getting the type of 'print' (line 856)
    print_122178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 16), 'print', False)
    # Calling print(args, kwargs) (line 856)
    print_call_result_122186 = invoke(stypy.reporting.localization.Localization(__file__, 856, 16), print_122178, *[result_mod_122184], **kwargs_122185)
    
    # SSA join for if statement (line 854)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 859)
    # Processing the call arguments (line 859)
    str_122188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 14), 'str', '-')
    int_122189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 18), 'int')
    # Applying the binary operator '*' (line 859)
    result_mul_122190 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 14), '*', str_122188, int_122189)
    
    # Processing the call keyword arguments (line 859)
    kwargs_122191 = {}
    # Getting the type of 'print' (line 859)
    print_122187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'print', False)
    # Calling print(args, kwargs) (line 859)
    print_call_result_122192 = invoke(stypy.reporting.localization.Localization(__file__, 859, 8), print_122187, *[result_mul_122190], **kwargs_122191)
    
    
    # Call to print(...): (line 860)
    # Processing the call arguments (line 860)
    str_122194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 14), 'str', 'Successfully read %i records of which:')
    
    # Call to len(...): (line 861)
    # Processing the call arguments (line 861)
    # Getting the type of 'records' (line 861)
    records_122196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 49), 'records', False)
    # Processing the call keyword arguments (line 861)
    kwargs_122197 = {}
    # Getting the type of 'len' (line 861)
    len_122195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 45), 'len', False)
    # Calling len(args, kwargs) (line 861)
    len_call_result_122198 = invoke(stypy.reporting.localization.Localization(__file__, 861, 45), len_122195, *[records_122196], **kwargs_122197)
    
    # Applying the binary operator '%' (line 860)
    result_mod_122199 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 14), '%', str_122194, len_call_result_122198)
    
    # Processing the call keyword arguments (line 860)
    kwargs_122200 = {}
    # Getting the type of 'print' (line 860)
    print_122193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'print', False)
    # Calling print(args, kwargs) (line 860)
    print_call_result_122201 = invoke(stypy.reporting.localization.Localization(__file__, 860, 8), print_122193, *[result_mod_122199], **kwargs_122200)
    
    
    # Assigning a ListComp to a Name (line 864):
    
    # Assigning a ListComp to a Name (line 864):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'records' (line 864)
    records_122206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 42), 'records')
    comprehension_122207 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 864, 20), records_122206)
    # Assigning a type to the variable 'r' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 20), 'r', comprehension_122207)
    
    # Obtaining the type of the subscript
    str_122202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 22), 'str', 'rectype')
    # Getting the type of 'r' (line 864)
    r_122203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 20), 'r')
    # Obtaining the member '__getitem__' of a type (line 864)
    getitem___122204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 20), r_122203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 864)
    subscript_call_result_122205 = invoke(stypy.reporting.localization.Localization(__file__, 864, 20), getitem___122204, str_122202)
    
    list_122208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 864, 20), list_122208, subscript_call_result_122205)
    # Assigning a type to the variable 'rectypes' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'rectypes', list_122208)
    
    
    # Call to set(...): (line 866)
    # Processing the call arguments (line 866)
    # Getting the type of 'rectypes' (line 866)
    rectypes_122210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 22), 'rectypes', False)
    # Processing the call keyword arguments (line 866)
    kwargs_122211 = {}
    # Getting the type of 'set' (line 866)
    set_122209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 18), 'set', False)
    # Calling set(args, kwargs) (line 866)
    set_call_result_122212 = invoke(stypy.reporting.localization.Localization(__file__, 866, 18), set_122209, *[rectypes_122210], **kwargs_122211)
    
    # Testing the type of a for loop iterable (line 866)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 866, 8), set_call_result_122212)
    # Getting the type of the for loop variable (line 866)
    for_loop_var_122213 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 866, 8), set_call_result_122212)
    # Assigning a type to the variable 'rt' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 8), 'rt', for_loop_var_122213)
    # SSA begins for a for statement (line 866)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'rt' (line 867)
    rt_122214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 15), 'rt')
    str_122215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 21), 'str', 'END_MARKER')
    # Applying the binary operator '!=' (line 867)
    result_ne_122216 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 15), '!=', rt_122214, str_122215)
    
    # Testing the type of an if condition (line 867)
    if_condition_122217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 867, 12), result_ne_122216)
    # Assigning a type to the variable 'if_condition_122217' (line 867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 12), 'if_condition_122217', if_condition_122217)
    # SSA begins for if statement (line 867)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 868)
    # Processing the call arguments (line 868)
    str_122219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 22), 'str', ' - %i are of type %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 868)
    tuple_122220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 868)
    # Adding element type (line 868)
    
    # Call to count(...): (line 868)
    # Processing the call arguments (line 868)
    # Getting the type of 'rt' (line 868)
    rt_122223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 63), 'rt', False)
    # Processing the call keyword arguments (line 868)
    kwargs_122224 = {}
    # Getting the type of 'rectypes' (line 868)
    rectypes_122221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 48), 'rectypes', False)
    # Obtaining the member 'count' of a type (line 868)
    count_122222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 48), rectypes_122221, 'count')
    # Calling count(args, kwargs) (line 868)
    count_call_result_122225 = invoke(stypy.reporting.localization.Localization(__file__, 868, 48), count_122222, *[rt_122223], **kwargs_122224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 868, 48), tuple_122220, count_call_result_122225)
    # Adding element type (line 868)
    # Getting the type of 'rt' (line 868)
    rt_122226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 68), 'rt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 868, 48), tuple_122220, rt_122226)
    
    # Applying the binary operator '%' (line 868)
    result_mod_122227 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 22), '%', str_122219, tuple_122220)
    
    # Processing the call keyword arguments (line 868)
    kwargs_122228 = {}
    # Getting the type of 'print' (line 868)
    print_122218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 16), 'print', False)
    # Calling print(args, kwargs) (line 868)
    print_call_result_122229 = invoke(stypy.reporting.localization.Localization(__file__, 868, 16), print_122218, *[result_mod_122227], **kwargs_122228)
    
    # SSA join for if statement (line 867)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 869)
    # Processing the call arguments (line 869)
    str_122231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 14), 'str', '-')
    int_122232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 18), 'int')
    # Applying the binary operator '*' (line 869)
    result_mul_122233 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 14), '*', str_122231, int_122232)
    
    # Processing the call keyword arguments (line 869)
    kwargs_122234 = {}
    # Getting the type of 'print' (line 869)
    print_122230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'print', False)
    # Calling print(args, kwargs) (line 869)
    print_call_result_122235 = invoke(stypy.reporting.localization.Localization(__file__, 869, 8), print_122230, *[result_mul_122233], **kwargs_122234)
    
    
    
    str_122236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 11), 'str', 'VARIABLE')
    # Getting the type of 'rectypes' (line 871)
    rectypes_122237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 25), 'rectypes')
    # Applying the binary operator 'in' (line 871)
    result_contains_122238 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 11), 'in', str_122236, rectypes_122237)
    
    # Testing the type of an if condition (line 871)
    if_condition_122239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 8), result_contains_122238)
    # Assigning a type to the variable 'if_condition_122239' (line 871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'if_condition_122239', if_condition_122239)
    # SSA begins for if statement (line 871)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 872)
    # Processing the call arguments (line 872)
    str_122241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 18), 'str', 'Available variables:')
    # Processing the call keyword arguments (line 872)
    kwargs_122242 = {}
    # Getting the type of 'print' (line 872)
    print_122240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'print', False)
    # Calling print(args, kwargs) (line 872)
    print_call_result_122243 = invoke(stypy.reporting.localization.Localization(__file__, 872, 12), print_122240, *[str_122241], **kwargs_122242)
    
    
    # Getting the type of 'variables' (line 873)
    variables_122244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 23), 'variables')
    # Testing the type of a for loop iterable (line 873)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 873, 12), variables_122244)
    # Getting the type of the for loop variable (line 873)
    for_loop_var_122245 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 873, 12), variables_122244)
    # Assigning a type to the variable 'var' (line 873)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 12), 'var', for_loop_var_122245)
    # SSA begins for a for statement (line 873)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to print(...): (line 874)
    # Processing the call arguments (line 874)
    str_122247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 22), 'str', ' - %s [%s]')
    
    # Obtaining an instance of the builtin type 'tuple' (line 874)
    tuple_122248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 874)
    # Adding element type (line 874)
    # Getting the type of 'var' (line 874)
    var_122249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 38), 'var', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 38), tuple_122248, var_122249)
    # Adding element type (line 874)
    
    # Call to type(...): (line 874)
    # Processing the call arguments (line 874)
    
    # Obtaining the type of the subscript
    # Getting the type of 'var' (line 874)
    var_122251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 58), 'var', False)
    # Getting the type of 'variables' (line 874)
    variables_122252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 48), 'variables', False)
    # Obtaining the member '__getitem__' of a type (line 874)
    getitem___122253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 48), variables_122252, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 874)
    subscript_call_result_122254 = invoke(stypy.reporting.localization.Localization(__file__, 874, 48), getitem___122253, var_122251)
    
    # Processing the call keyword arguments (line 874)
    kwargs_122255 = {}
    # Getting the type of 'type' (line 874)
    type_122250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 43), 'type', False)
    # Calling type(args, kwargs) (line 874)
    type_call_result_122256 = invoke(stypy.reporting.localization.Localization(__file__, 874, 43), type_122250, *[subscript_call_result_122254], **kwargs_122255)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 38), tuple_122248, type_call_result_122256)
    
    # Applying the binary operator '%' (line 874)
    result_mod_122257 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 22), '%', str_122247, tuple_122248)
    
    # Processing the call keyword arguments (line 874)
    kwargs_122258 = {}
    # Getting the type of 'print' (line 874)
    print_122246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 16), 'print', False)
    # Calling print(args, kwargs) (line 874)
    print_call_result_122259 = invoke(stypy.reporting.localization.Localization(__file__, 874, 16), print_122246, *[result_mod_122257], **kwargs_122258)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 875)
    # Processing the call arguments (line 875)
    str_122261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 18), 'str', '-')
    int_122262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 22), 'int')
    # Applying the binary operator '*' (line 875)
    result_mul_122263 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 18), '*', str_122261, int_122262)
    
    # Processing the call keyword arguments (line 875)
    kwargs_122264 = {}
    # Getting the type of 'print' (line 875)
    print_122260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'print', False)
    # Calling print(args, kwargs) (line 875)
    print_call_result_122265 = invoke(stypy.reporting.localization.Localization(__file__, 875, 12), print_122260, *[result_mul_122263], **kwargs_122264)
    
    # SSA join for if statement (line 871)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 822)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'idict' (line 877)
    idict_122266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 7), 'idict')
    # Testing the type of an if condition (line 877)
    if_condition_122267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 877, 4), idict_122266)
    # Assigning a type to the variable 'if_condition_122267' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 4), 'if_condition_122267', if_condition_122267)
    # SSA begins for if statement (line 877)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'variables' (line 878)
    variables_122268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 19), 'variables')
    # Testing the type of a for loop iterable (line 878)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 878, 8), variables_122268)
    # Getting the type of the for loop variable (line 878)
    for_loop_var_122269 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 878, 8), variables_122268)
    # Assigning a type to the variable 'var' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 8), 'var', for_loop_var_122269)
    # SSA begins for a for statement (line 878)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 879):
    
    # Assigning a Subscript to a Subscript (line 879):
    
    # Obtaining the type of the subscript
    # Getting the type of 'var' (line 879)
    var_122270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 35), 'var')
    # Getting the type of 'variables' (line 879)
    variables_122271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 25), 'variables')
    # Obtaining the member '__getitem__' of a type (line 879)
    getitem___122272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 25), variables_122271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 879)
    subscript_call_result_122273 = invoke(stypy.reporting.localization.Localization(__file__, 879, 25), getitem___122272, var_122270)
    
    # Getting the type of 'idict' (line 879)
    idict_122274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 12), 'idict')
    # Getting the type of 'var' (line 879)
    var_122275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 18), 'var')
    # Storing an element on a container (line 879)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 879, 12), idict_122274, (var_122275, subscript_call_result_122273))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'idict' (line 880)
    idict_122276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 15), 'idict')
    # Assigning a type to the variable 'stypy_return_type' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 8), 'stypy_return_type', idict_122276)
    # SSA branch for the else part of an if statement (line 877)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'variables' (line 882)
    variables_122277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 15), 'variables')
    # Assigning a type to the variable 'stypy_return_type' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'stypy_return_type', variables_122277)
    # SSA join for if statement (line 877)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'readsav(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'readsav' in the type store
    # Getting the type of 'stypy_return_type' (line 674)
    stypy_return_type_122278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'readsav'
    return stypy_return_type_122278

# Assigning a type to the variable 'readsav' (line 674)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 0), 'readsav', readsav)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
