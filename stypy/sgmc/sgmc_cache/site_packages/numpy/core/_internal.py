
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A place for code to be called from core C-code.
3: 
4: Some things are more easily handled Python.
5: 
6: '''
7: from __future__ import division, absolute_import, print_function
8: 
9: import re
10: import sys
11: 
12: from numpy.compat import asbytes, basestring
13: from .multiarray import dtype, array, ndarray
14: import ctypes
15: from .numerictypes import object_
16: 
17: if (sys.byteorder == 'little'):
18:     _nbo = asbytes('<')
19: else:
20:     _nbo = asbytes('>')
21: 
22: def _makenames_list(adict, align):
23:     allfields = []
24:     fnames = list(adict.keys())
25:     for fname in fnames:
26:         obj = adict[fname]
27:         n = len(obj)
28:         if not isinstance(obj, tuple) or n not in [2, 3]:
29:             raise ValueError("entry not a 2- or 3- tuple")
30:         if (n > 2) and (obj[2] == fname):
31:             continue
32:         num = int(obj[1])
33:         if (num < 0):
34:             raise ValueError("invalid offset.")
35:         format = dtype(obj[0], align=align)
36:         if (format.itemsize == 0):
37:             raise ValueError("all itemsizes must be fixed.")
38:         if (n > 2):
39:             title = obj[2]
40:         else:
41:             title = None
42:         allfields.append((fname, format, num, title))
43:     # sort by offsets
44:     allfields.sort(key=lambda x: x[2])
45:     names = [x[0] for x in allfields]
46:     formats = [x[1] for x in allfields]
47:     offsets = [x[2] for x in allfields]
48:     titles = [x[3] for x in allfields]
49: 
50:     return names, formats, offsets, titles
51: 
52: # Called in PyArray_DescrConverter function when
53: #  a dictionary without "names" and "formats"
54: #  fields is used as a data-type descriptor.
55: def _usefields(adict, align):
56:     try:
57:         names = adict[-1]
58:     except KeyError:
59:         names = None
60:     if names is None:
61:         names, formats, offsets, titles = _makenames_list(adict, align)
62:     else:
63:         formats = []
64:         offsets = []
65:         titles = []
66:         for name in names:
67:             res = adict[name]
68:             formats.append(res[0])
69:             offsets.append(res[1])
70:             if (len(res) > 2):
71:                 titles.append(res[2])
72:             else:
73:                 titles.append(None)
74: 
75:     return dtype({"names": names,
76:                   "formats": formats,
77:                   "offsets": offsets,
78:                   "titles": titles}, align)
79: 
80: 
81: # construct an array_protocol descriptor list
82: #  from the fields attribute of a descriptor
83: # This calls itself recursively but should eventually hit
84: #  a descriptor that has no fields and then return
85: #  a simple typestring
86: 
87: def _array_descr(descriptor):
88:     fields = descriptor.fields
89:     if fields is None:
90:         subdtype = descriptor.subdtype
91:         if subdtype is None:
92:             if descriptor.metadata is None:
93:                 return descriptor.str
94:             else:
95:                 new = descriptor.metadata.copy()
96:                 if new:
97:                     return (descriptor.str, new)
98:                 else:
99:                     return descriptor.str
100:         else:
101:             return (_array_descr(subdtype[0]), subdtype[1])
102: 
103:     names = descriptor.names
104:     ordered_fields = [fields[x] + (x,) for x in names]
105:     result = []
106:     offset = 0
107:     for field in ordered_fields:
108:         if field[1] > offset:
109:             num = field[1] - offset
110:             result.append(('', '|V%d' % num))
111:             offset += num
112:         if len(field) > 3:
113:             name = (field[2], field[3])
114:         else:
115:             name = field[2]
116:         if field[0].subdtype:
117:             tup = (name, _array_descr(field[0].subdtype[0]),
118:                    field[0].subdtype[1])
119:         else:
120:             tup = (name, _array_descr(field[0]))
121:         offset += field[0].itemsize
122:         result.append(tup)
123: 
124:     if descriptor.itemsize > offset:
125:         num = descriptor.itemsize - offset
126:         result.append(('', '|V%d' % num))
127: 
128:     return result
129: 
130: # Build a new array from the information in a pickle.
131: # Note that the name numpy.core._internal._reconstruct is embedded in
132: # pickles of ndarrays made with NumPy before release 1.0
133: # so don't remove the name here, or you'll
134: # break backward compatibilty.
135: def _reconstruct(subtype, shape, dtype):
136:     return ndarray.__new__(subtype, shape, dtype)
137: 
138: 
139: # format_re was originally from numarray by J. Todd Miller
140: 
141: format_re = re.compile(asbytes(
142:                            r'(?P<order1>[<>|=]?)'
143:                            r'(?P<repeats> *[(]?[ ,0-9L]*[)]? *)'
144:                            r'(?P<order2>[<>|=]?)'
145:                            r'(?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)'))
146: sep_re = re.compile(asbytes(r'\s*,\s*'))
147: space_re = re.compile(asbytes(r'\s+$'))
148: 
149: # astr is a string (perhaps comma separated)
150: 
151: _convorder = {asbytes('='): _nbo}
152: 
153: def _commastring(astr):
154:     startindex = 0
155:     result = []
156:     while startindex < len(astr):
157:         mo = format_re.match(astr, pos=startindex)
158:         try:
159:             (order1, repeats, order2, dtype) = mo.groups()
160:         except (TypeError, AttributeError):
161:             raise ValueError('format number %d of "%s" is not recognized' %
162:                                             (len(result)+1, astr))
163:         startindex = mo.end()
164:         # Separator or ending padding
165:         if startindex < len(astr):
166:             if space_re.match(astr, pos=startindex):
167:                 startindex = len(astr)
168:             else:
169:                 mo = sep_re.match(astr, pos=startindex)
170:                 if not mo:
171:                     raise ValueError(
172:                         'format number %d of "%s" is not recognized' %
173:                         (len(result)+1, astr))
174:                 startindex = mo.end()
175: 
176:         if order2 == asbytes(''):
177:             order = order1
178:         elif order1 == asbytes(''):
179:             order = order2
180:         else:
181:             order1 = _convorder.get(order1, order1)
182:             order2 = _convorder.get(order2, order2)
183:             if (order1 != order2):
184:                 raise ValueError(
185:                     'inconsistent byte-order specification %s and %s' %
186:                     (order1, order2))
187:             order = order1
188: 
189:         if order in [asbytes('|'), asbytes('='), _nbo]:
190:             order = asbytes('')
191:         dtype = order + dtype
192:         if (repeats == asbytes('')):
193:             newitem = dtype
194:         else:
195:             newitem = (dtype, eval(repeats))
196:         result.append(newitem)
197: 
198:     return result
199: 
200: def _getintp_ctype():
201:     val = _getintp_ctype.cache
202:     if val is not None:
203:         return val
204:     char = dtype('p').char
205:     if (char == 'i'):
206:         val = ctypes.c_int
207:     elif char == 'l':
208:         val = ctypes.c_long
209:     elif char == 'q':
210:         val = ctypes.c_longlong
211:     else:
212:         val = ctypes.c_long
213:     _getintp_ctype.cache = val
214:     return val
215: _getintp_ctype.cache = None
216: 
217: # Used for .ctypes attribute of ndarray
218: 
219: class _missing_ctypes(object):
220:     def cast(self, num, obj):
221:         return num
222: 
223:     def c_void_p(self, num):
224:         return num
225: 
226: class _ctypes(object):
227:     def __init__(self, array, ptr=None):
228:         try:
229:             self._ctypes = ctypes
230:         except ImportError:
231:             self._ctypes = _missing_ctypes()
232:         self._arr = array
233:         self._data = ptr
234:         if self._arr.ndim == 0:
235:             self._zerod = True
236:         else:
237:             self._zerod = False
238: 
239:     def data_as(self, obj):
240:         return self._ctypes.cast(self._data, obj)
241: 
242:     def shape_as(self, obj):
243:         if self._zerod:
244:             return None
245:         return (obj*self._arr.ndim)(*self._arr.shape)
246: 
247:     def strides_as(self, obj):
248:         if self._zerod:
249:             return None
250:         return (obj*self._arr.ndim)(*self._arr.strides)
251: 
252:     def get_data(self):
253:         return self._data
254: 
255:     def get_shape(self):
256:         if self._zerod:
257:             return None
258:         return (_getintp_ctype()*self._arr.ndim)(*self._arr.shape)
259: 
260:     def get_strides(self):
261:         if self._zerod:
262:             return None
263:         return (_getintp_ctype()*self._arr.ndim)(*self._arr.strides)
264: 
265:     def get_as_parameter(self):
266:         return self._ctypes.c_void_p(self._data)
267: 
268:     data = property(get_data, None, doc="c-types data")
269:     shape = property(get_shape, None, doc="c-types shape")
270:     strides = property(get_strides, None, doc="c-types strides")
271:     _as_parameter_ = property(get_as_parameter, None, doc="_as parameter_")
272: 
273: 
274: # Given a datatype and an order object
275: #  return a new names tuple
276: #  with the order indicated
277: def _newnames(datatype, order):
278:     oldnames = datatype.names
279:     nameslist = list(oldnames)
280:     if isinstance(order, str):
281:         order = [order]
282:     if isinstance(order, (list, tuple)):
283:         for name in order:
284:             try:
285:                 nameslist.remove(name)
286:             except ValueError:
287:                 raise ValueError("unknown field name: %s" % (name,))
288:         return tuple(list(order) + nameslist)
289:     raise ValueError("unsupported order value: %s" % (order,))
290: 
291: def _copy_fields(ary):
292:     '''Return copy of structured array with padding between fields removed.
293: 
294:     Parameters
295:     ----------
296:     ary : ndarray
297:        Structured array from which to remove padding bytes
298: 
299:     Returns
300:     -------
301:     ary_copy : ndarray
302:        Copy of ary with padding bytes removed
303:     '''
304:     dt = ary.dtype
305:     copy_dtype = {'names': dt.names,
306:                   'formats': [dt.fields[name][0] for name in dt.names]}
307:     return array(ary, dtype=copy_dtype, copy=True)
308: 
309: def _getfield_is_safe(oldtype, newtype, offset):
310:     ''' Checks safety of getfield for object arrays.
311: 
312:     As in _view_is_safe, we need to check that memory containing objects is not
313:     reinterpreted as a non-object datatype and vice versa.
314: 
315:     Parameters
316:     ----------
317:     oldtype : data-type
318:         Data type of the original ndarray.
319:     newtype : data-type
320:         Data type of the field being accessed by ndarray.getfield
321:     offset : int
322:         Offset of the field being accessed by ndarray.getfield
323: 
324:     Raises
325:     ------
326:     TypeError
327:         If the field access is invalid
328: 
329:     '''
330:     if newtype.hasobject or oldtype.hasobject:
331:         if offset == 0 and newtype == oldtype:
332:             return
333:         if oldtype.names:
334:             for name in oldtype.names:
335:                 if (oldtype.fields[name][1] == offset and
336:                         oldtype.fields[name][0] == newtype):
337:                     return
338:         raise TypeError("Cannot get/set field of an object array")
339:     return
340: 
341: def _view_is_safe(oldtype, newtype):
342:     ''' Checks safety of a view involving object arrays, for example when
343:     doing::
344: 
345:         np.zeros(10, dtype=oldtype).view(newtype)
346: 
347:     Parameters
348:     ----------
349:     oldtype : data-type
350:         Data type of original ndarray
351:     newtype : data-type
352:         Data type of the view
353: 
354:     Raises
355:     ------
356:     TypeError
357:         If the new type is incompatible with the old type.
358: 
359:     '''
360: 
361:     # if the types are equivalent, there is no problem.
362:     # for example: dtype((np.record, 'i4,i4')) == dtype((np.void, 'i4,i4'))
363:     if oldtype == newtype:
364:         return
365: 
366:     if newtype.hasobject or oldtype.hasobject:
367:         raise TypeError("Cannot change data-type for object array.")
368:     return
369: 
370: # Given a string containing a PEP 3118 format specifier,
371: # construct a Numpy dtype
372: 
373: _pep3118_native_map = {
374:     '?': '?',
375:     'b': 'b',
376:     'B': 'B',
377:     'h': 'h',
378:     'H': 'H',
379:     'i': 'i',
380:     'I': 'I',
381:     'l': 'l',
382:     'L': 'L',
383:     'q': 'q',
384:     'Q': 'Q',
385:     'e': 'e',
386:     'f': 'f',
387:     'd': 'd',
388:     'g': 'g',
389:     'Zf': 'F',
390:     'Zd': 'D',
391:     'Zg': 'G',
392:     's': 'S',
393:     'w': 'U',
394:     'O': 'O',
395:     'x': 'V',  # padding
396: }
397: _pep3118_native_typechars = ''.join(_pep3118_native_map.keys())
398: 
399: _pep3118_standard_map = {
400:     '?': '?',
401:     'b': 'b',
402:     'B': 'B',
403:     'h': 'i2',
404:     'H': 'u2',
405:     'i': 'i4',
406:     'I': 'u4',
407:     'l': 'i4',
408:     'L': 'u4',
409:     'q': 'i8',
410:     'Q': 'u8',
411:     'e': 'f2',
412:     'f': 'f',
413:     'd': 'd',
414:     'Zf': 'F',
415:     'Zd': 'D',
416:     's': 'S',
417:     'w': 'U',
418:     'O': 'O',
419:     'x': 'V',  # padding
420: }
421: _pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())
422: 
423: def _dtype_from_pep3118(spec, byteorder='@', is_subdtype=False):
424:     fields = {}
425:     offset = 0
426:     explicit_name = False
427:     this_explicit_name = False
428:     common_alignment = 1
429:     is_padding = False
430: 
431:     dummy_name_index = [0]
432: 
433:     def next_dummy_name():
434:         dummy_name_index[0] += 1
435: 
436:     def get_dummy_name():
437:         while True:
438:             name = 'f%d' % dummy_name_index[0]
439:             if name not in fields:
440:                 return name
441:             next_dummy_name()
442: 
443:     # Parse spec
444:     while spec:
445:         value = None
446: 
447:         # End of structure, bail out to upper level
448:         if spec[0] == '}':
449:             spec = spec[1:]
450:             break
451: 
452:         # Sub-arrays (1)
453:         shape = None
454:         if spec[0] == '(':
455:             j = spec.index(')')
456:             shape = tuple(map(int, spec[1:j].split(',')))
457:             spec = spec[j+1:]
458: 
459:         # Byte order
460:         if spec[0] in ('@', '=', '<', '>', '^', '!'):
461:             byteorder = spec[0]
462:             if byteorder == '!':
463:                 byteorder = '>'
464:             spec = spec[1:]
465: 
466:         # Byte order characters also control native vs. standard type sizes
467:         if byteorder in ('@', '^'):
468:             type_map = _pep3118_native_map
469:             type_map_chars = _pep3118_native_typechars
470:         else:
471:             type_map = _pep3118_standard_map
472:             type_map_chars = _pep3118_standard_typechars
473: 
474:         # Item sizes
475:         itemsize = 1
476:         if spec[0].isdigit():
477:             j = 1
478:             for j in range(1, len(spec)):
479:                 if not spec[j].isdigit():
480:                     break
481:             itemsize = int(spec[:j])
482:             spec = spec[j:]
483: 
484:         # Data types
485:         is_padding = False
486: 
487:         if spec[:2] == 'T{':
488:             value, spec, align, next_byteorder = _dtype_from_pep3118(
489:                 spec[2:], byteorder=byteorder, is_subdtype=True)
490:         elif spec[0] in type_map_chars:
491:             next_byteorder = byteorder
492:             if spec[0] == 'Z':
493:                 j = 2
494:             else:
495:                 j = 1
496:             typechar = spec[:j]
497:             spec = spec[j:]
498:             is_padding = (typechar == 'x')
499:             dtypechar = type_map[typechar]
500:             if dtypechar in 'USV':
501:                 dtypechar += '%d' % itemsize
502:                 itemsize = 1
503:             numpy_byteorder = {'@': '=', '^': '='}.get(byteorder, byteorder)
504:             value = dtype(numpy_byteorder + dtypechar)
505:             align = value.alignment
506:         else:
507:             raise ValueError("Unknown PEP 3118 data type specifier %r" % spec)
508: 
509:         #
510:         # Native alignment may require padding
511:         #
512:         # Here we assume that the presence of a '@' character implicitly implies
513:         # that the start of the array is *already* aligned.
514:         #
515:         extra_offset = 0
516:         if byteorder == '@':
517:             start_padding = (-offset) % align
518:             intra_padding = (-value.itemsize) % align
519: 
520:             offset += start_padding
521: 
522:             if intra_padding != 0:
523:                 if itemsize > 1 or (shape is not None and _prod(shape) > 1):
524:                     # Inject internal padding to the end of the sub-item
525:                     value = _add_trailing_padding(value, intra_padding)
526:                 else:
527:                     # We can postpone the injection of internal padding,
528:                     # as the item appears at most once
529:                     extra_offset += intra_padding
530: 
531:             # Update common alignment
532:             common_alignment = (align*common_alignment
533:                                 / _gcd(align, common_alignment))
534: 
535:         # Convert itemsize to sub-array
536:         if itemsize != 1:
537:             value = dtype((value, (itemsize,)))
538: 
539:         # Sub-arrays (2)
540:         if shape is not None:
541:             value = dtype((value, shape))
542: 
543:         # Field name
544:         this_explicit_name = False
545:         if spec and spec.startswith(':'):
546:             i = spec[1:].index(':') + 1
547:             name = spec[1:i]
548:             spec = spec[i+1:]
549:             explicit_name = True
550:             this_explicit_name = True
551:         else:
552:             name = get_dummy_name()
553: 
554:         if not is_padding or this_explicit_name:
555:             if name in fields:
556:                 raise RuntimeError("Duplicate field name '%s' in PEP3118 format"
557:                                    % name)
558:             fields[name] = (value, offset)
559:             if not this_explicit_name:
560:                 next_dummy_name()
561: 
562:         byteorder = next_byteorder
563: 
564:         offset += value.itemsize
565:         offset += extra_offset
566: 
567:     # Check if this was a simple 1-item type
568:     if (len(fields) == 1 and not explicit_name and
569:             fields['f0'][1] == 0 and not is_subdtype):
570:         ret = fields['f0'][0]
571:     else:
572:         ret = dtype(fields)
573: 
574:     # Trailing padding must be explicitly added
575:     padding = offset - ret.itemsize
576:     if byteorder == '@':
577:         padding += (-offset) % common_alignment
578:     if is_padding and not this_explicit_name:
579:         ret = _add_trailing_padding(ret, padding)
580: 
581:     # Finished
582:     if is_subdtype:
583:         return ret, spec, common_alignment, byteorder
584:     else:
585:         return ret
586: 
587: def _add_trailing_padding(value, padding):
588:     '''Inject the specified number of padding bytes at the end of a dtype'''
589:     if value.fields is None:
590:         vfields = {'f0': (value, 0)}
591:     else:
592:         vfields = dict(value.fields)
593: 
594:     if (value.names and value.names[-1] == '' and
595:            value[''].char == 'V'):
596:         # A trailing padding field is already present
597:         vfields[''] = ('V%d' % (vfields[''][0].itemsize + padding),
598:                        vfields[''][1])
599:         value = dtype(vfields)
600:     else:
601:         # Get a free name for the padding field
602:         j = 0
603:         while True:
604:             name = 'pad%d' % j
605:             if name not in vfields:
606:                 vfields[name] = ('V%d' % padding, value.itemsize)
607:                 break
608:             j += 1
609: 
610:         value = dtype(vfields)
611:         if '' not in vfields:
612:             # Strip out the name of the padding field
613:             names = list(value.names)
614:             names[-1] = ''
615:             value.names = tuple(names)
616:     return value
617: 
618: def _prod(a):
619:     p = 1
620:     for x in a:
621:         p *= x
622:     return p
623: 
624: def _gcd(a, b):
625:     '''Calculate the greatest common divisor of a and b'''
626:     while b:
627:         a, b = b, a % b
628:     return a
629: 
630: # Exception used in shares_memory()
631: class TooHardError(RuntimeError):
632:     pass
633: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_18770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nA place for code to be called from core C-code.\n\nSome things are more easily handled Python.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import re' statement (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.compat import asbytes, basestring' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_18771 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat')

if (type(import_18771) is not StypyTypeError):

    if (import_18771 != 'pyd_module'):
        __import__(import_18771)
        sys_modules_18772 = sys.modules[import_18771]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', sys_modules_18772.module_type_store, module_type_store, ['asbytes', 'basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_18772, sys_modules_18772.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'basestring'], [asbytes, basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', import_18771)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.core.multiarray import dtype, array, ndarray' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_18773 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.multiarray')

if (type(import_18773) is not StypyTypeError):

    if (import_18773 != 'pyd_module'):
        __import__(import_18773)
        sys_modules_18774 = sys.modules[import_18773]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.multiarray', sys_modules_18774.module_type_store, module_type_store, ['dtype', 'array', 'ndarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_18774, sys_modules_18774.module_type_store, module_type_store)
    else:
        from numpy.core.multiarray import dtype, array, ndarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.multiarray', None, module_type_store, ['dtype', 'array', 'ndarray'], [dtype, array, ndarray])

else:
    # Assigning a type to the variable 'numpy.core.multiarray' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core.multiarray', import_18773)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import ctypes' statement (line 14)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'ctypes', ctypes, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.core.numerictypes import object_' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_18775 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.numerictypes')

if (type(import_18775) is not StypyTypeError):

    if (import_18775 != 'pyd_module'):
        __import__(import_18775)
        sys_modules_18776 = sys.modules[import_18775]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.numerictypes', sys_modules_18776.module_type_store, module_type_store, ['object_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_18776, sys_modules_18776.module_type_store, module_type_store)
    else:
        from numpy.core.numerictypes import object_

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.numerictypes', None, module_type_store, ['object_'], [object_])

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core.numerictypes', import_18775)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')



# Getting the type of 'sys' (line 17)
sys_18777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'sys')
# Obtaining the member 'byteorder' of a type (line 17)
byteorder_18778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), sys_18777, 'byteorder')
str_18779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'str', 'little')
# Applying the binary operator '==' (line 17)
result_eq_18780 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 4), '==', byteorder_18778, str_18779)

# Testing the type of an if condition (line 17)
if_condition_18781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 0), result_eq_18780)
# Assigning a type to the variable 'if_condition_18781' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'if_condition_18781', if_condition_18781)
# SSA begins for if statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 18):

# Assigning a Call to a Name (line 18):

# Call to asbytes(...): (line 18)
# Processing the call arguments (line 18)
str_18783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'str', '<')
# Processing the call keyword arguments (line 18)
kwargs_18784 = {}
# Getting the type of 'asbytes' (line 18)
asbytes_18782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 18)
asbytes_call_result_18785 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), asbytes_18782, *[str_18783], **kwargs_18784)

# Assigning a type to the variable '_nbo' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), '_nbo', asbytes_call_result_18785)
# SSA branch for the else part of an if statement (line 17)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 20):

# Assigning a Call to a Name (line 20):

# Call to asbytes(...): (line 20)
# Processing the call arguments (line 20)
str_18787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'str', '>')
# Processing the call keyword arguments (line 20)
kwargs_18788 = {}
# Getting the type of 'asbytes' (line 20)
asbytes_18786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 20)
asbytes_call_result_18789 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), asbytes_18786, *[str_18787], **kwargs_18788)

# Assigning a type to the variable '_nbo' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), '_nbo', asbytes_call_result_18789)
# SSA join for if statement (line 17)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _makenames_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_makenames_list'
    module_type_store = module_type_store.open_function_context('_makenames_list', 22, 0, False)
    
    # Passed parameters checking function
    _makenames_list.stypy_localization = localization
    _makenames_list.stypy_type_of_self = None
    _makenames_list.stypy_type_store = module_type_store
    _makenames_list.stypy_function_name = '_makenames_list'
    _makenames_list.stypy_param_names_list = ['adict', 'align']
    _makenames_list.stypy_varargs_param_name = None
    _makenames_list.stypy_kwargs_param_name = None
    _makenames_list.stypy_call_defaults = defaults
    _makenames_list.stypy_call_varargs = varargs
    _makenames_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_makenames_list', ['adict', 'align'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_makenames_list', localization, ['adict', 'align'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_makenames_list(...)' code ##################

    
    # Assigning a List to a Name (line 23):
    
    # Assigning a List to a Name (line 23):
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_18790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    
    # Assigning a type to the variable 'allfields' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'allfields', list_18790)
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to list(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to keys(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_18794 = {}
    # Getting the type of 'adict' (line 24)
    adict_18792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'adict', False)
    # Obtaining the member 'keys' of a type (line 24)
    keys_18793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 18), adict_18792, 'keys')
    # Calling keys(args, kwargs) (line 24)
    keys_call_result_18795 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), keys_18793, *[], **kwargs_18794)
    
    # Processing the call keyword arguments (line 24)
    kwargs_18796 = {}
    # Getting the type of 'list' (line 24)
    list_18791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'list', False)
    # Calling list(args, kwargs) (line 24)
    list_call_result_18797 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), list_18791, *[keys_call_result_18795], **kwargs_18796)
    
    # Assigning a type to the variable 'fnames' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'fnames', list_call_result_18797)
    
    # Getting the type of 'fnames' (line 25)
    fnames_18798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'fnames')
    # Testing the type of a for loop iterable (line 25)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 4), fnames_18798)
    # Getting the type of the for loop variable (line 25)
    for_loop_var_18799 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 4), fnames_18798)
    # Assigning a type to the variable 'fname' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'fname', for_loop_var_18799)
    # SSA begins for a for statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 26):
    
    # Assigning a Subscript to a Name (line 26):
    
    # Obtaining the type of the subscript
    # Getting the type of 'fname' (line 26)
    fname_18800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'fname')
    # Getting the type of 'adict' (line 26)
    adict_18801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'adict')
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___18802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), adict_18801, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 26)
    subscript_call_result_18803 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), getitem___18802, fname_18800)
    
    # Assigning a type to the variable 'obj' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'obj', subscript_call_result_18803)
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Call to len(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'obj' (line 27)
    obj_18805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'obj', False)
    # Processing the call keyword arguments (line 27)
    kwargs_18806 = {}
    # Getting the type of 'len' (line 27)
    len_18804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'len', False)
    # Calling len(args, kwargs) (line 27)
    len_call_result_18807 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), len_18804, *[obj_18805], **kwargs_18806)
    
    # Assigning a type to the variable 'n' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'n', len_call_result_18807)
    
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'obj' (line 28)
    obj_18809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'obj', False)
    # Getting the type of 'tuple' (line 28)
    tuple_18810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'tuple', False)
    # Processing the call keyword arguments (line 28)
    kwargs_18811 = {}
    # Getting the type of 'isinstance' (line 28)
    isinstance_18808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 28)
    isinstance_call_result_18812 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), isinstance_18808, *[obj_18809, tuple_18810], **kwargs_18811)
    
    # Applying the 'not' unary operator (line 28)
    result_not__18813 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), 'not', isinstance_call_result_18812)
    
    
    # Getting the type of 'n' (line 28)
    n_18814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 41), 'n')
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_18815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    int_18816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 50), list_18815, int_18816)
    # Adding element type (line 28)
    int_18817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 50), list_18815, int_18817)
    
    # Applying the binary operator 'notin' (line 28)
    result_contains_18818 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 41), 'notin', n_18814, list_18815)
    
    # Applying the binary operator 'or' (line 28)
    result_or_keyword_18819 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), 'or', result_not__18813, result_contains_18818)
    
    # Testing the type of an if condition (line 28)
    if_condition_18820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 8), result_or_keyword_18819)
    # Assigning a type to the variable 'if_condition_18820' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'if_condition_18820', if_condition_18820)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 29)
    # Processing the call arguments (line 29)
    str_18822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'str', 'entry not a 2- or 3- tuple')
    # Processing the call keyword arguments (line 29)
    kwargs_18823 = {}
    # Getting the type of 'ValueError' (line 29)
    ValueError_18821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 29)
    ValueError_call_result_18824 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), ValueError_18821, *[str_18822], **kwargs_18823)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 29, 12), ValueError_call_result_18824, 'raise parameter', BaseException)
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 30)
    n_18825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'n')
    int_18826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'int')
    # Applying the binary operator '>' (line 30)
    result_gt_18827 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '>', n_18825, int_18826)
    
    
    
    # Obtaining the type of the subscript
    int_18828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
    # Getting the type of 'obj' (line 30)
    obj_18829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'obj')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___18830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 24), obj_18829, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_18831 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), getitem___18830, int_18828)
    
    # Getting the type of 'fname' (line 30)
    fname_18832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'fname')
    # Applying the binary operator '==' (line 30)
    result_eq_18833 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 24), '==', subscript_call_result_18831, fname_18832)
    
    # Applying the binary operator 'and' (line 30)
    result_and_keyword_18834 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), 'and', result_gt_18827, result_eq_18833)
    
    # Testing the type of an if condition (line 30)
    if_condition_18835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), result_and_keyword_18834)
    # Assigning a type to the variable 'if_condition_18835' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_18835', if_condition_18835)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to int(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Obtaining the type of the subscript
    int_18837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'int')
    # Getting the type of 'obj' (line 32)
    obj_18838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), 'obj', False)
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___18839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 18), obj_18838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_18840 = invoke(stypy.reporting.localization.Localization(__file__, 32, 18), getitem___18839, int_18837)
    
    # Processing the call keyword arguments (line 32)
    kwargs_18841 = {}
    # Getting the type of 'int' (line 32)
    int_18836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'int', False)
    # Calling int(args, kwargs) (line 32)
    int_call_result_18842 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), int_18836, *[subscript_call_result_18840], **kwargs_18841)
    
    # Assigning a type to the variable 'num' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'num', int_call_result_18842)
    
    
    # Getting the type of 'num' (line 33)
    num_18843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'num')
    int_18844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'int')
    # Applying the binary operator '<' (line 33)
    result_lt_18845 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 12), '<', num_18843, int_18844)
    
    # Testing the type of an if condition (line 33)
    if_condition_18846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 8), result_lt_18845)
    # Assigning a type to the variable 'if_condition_18846' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'if_condition_18846', if_condition_18846)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 34)
    # Processing the call arguments (line 34)
    str_18848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'str', 'invalid offset.')
    # Processing the call keyword arguments (line 34)
    kwargs_18849 = {}
    # Getting the type of 'ValueError' (line 34)
    ValueError_18847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 34)
    ValueError_call_result_18850 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), ValueError_18847, *[str_18848], **kwargs_18849)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 34, 12), ValueError_call_result_18850, 'raise parameter', BaseException)
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to dtype(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining the type of the subscript
    int_18852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 27), 'int')
    # Getting the type of 'obj' (line 35)
    obj_18853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'obj', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___18854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), obj_18853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_18855 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), getitem___18854, int_18852)
    
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'align' (line 35)
    align_18856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'align', False)
    keyword_18857 = align_18856
    kwargs_18858 = {'align': keyword_18857}
    # Getting the type of 'dtype' (line 35)
    dtype_18851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'dtype', False)
    # Calling dtype(args, kwargs) (line 35)
    dtype_call_result_18859 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), dtype_18851, *[subscript_call_result_18855], **kwargs_18858)
    
    # Assigning a type to the variable 'format' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'format', dtype_call_result_18859)
    
    
    # Getting the type of 'format' (line 36)
    format_18860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'format')
    # Obtaining the member 'itemsize' of a type (line 36)
    itemsize_18861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), format_18860, 'itemsize')
    int_18862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'int')
    # Applying the binary operator '==' (line 36)
    result_eq_18863 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '==', itemsize_18861, int_18862)
    
    # Testing the type of an if condition (line 36)
    if_condition_18864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), result_eq_18863)
    # Assigning a type to the variable 'if_condition_18864' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_18864', if_condition_18864)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 37)
    # Processing the call arguments (line 37)
    str_18866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'str', 'all itemsizes must be fixed.')
    # Processing the call keyword arguments (line 37)
    kwargs_18867 = {}
    # Getting the type of 'ValueError' (line 37)
    ValueError_18865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 37)
    ValueError_call_result_18868 = invoke(stypy.reporting.localization.Localization(__file__, 37, 18), ValueError_18865, *[str_18866], **kwargs_18867)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 37, 12), ValueError_call_result_18868, 'raise parameter', BaseException)
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 38)
    n_18869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'n')
    int_18870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'int')
    # Applying the binary operator '>' (line 38)
    result_gt_18871 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 12), '>', n_18869, int_18870)
    
    # Testing the type of an if condition (line 38)
    if_condition_18872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), result_gt_18871)
    # Assigning a type to the variable 'if_condition_18872' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_18872', if_condition_18872)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 39):
    
    # Assigning a Subscript to a Name (line 39):
    
    # Obtaining the type of the subscript
    int_18873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
    # Getting the type of 'obj' (line 39)
    obj_18874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'obj')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___18875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), obj_18874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_18876 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), getitem___18875, int_18873)
    
    # Assigning a type to the variable 'title' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'title', subscript_call_result_18876)
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 41):
    
    # Assigning a Name to a Name (line 41):
    # Getting the type of 'None' (line 41)
    None_18877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'None')
    # Assigning a type to the variable 'title' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'title', None_18877)
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_18880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'fname' (line 42)
    fname_18881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'fname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), tuple_18880, fname_18881)
    # Adding element type (line 42)
    # Getting the type of 'format' (line 42)
    format_18882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'format', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), tuple_18880, format_18882)
    # Adding element type (line 42)
    # Getting the type of 'num' (line 42)
    num_18883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'num', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), tuple_18880, num_18883)
    # Adding element type (line 42)
    # Getting the type of 'title' (line 42)
    title_18884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'title', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 26), tuple_18880, title_18884)
    
    # Processing the call keyword arguments (line 42)
    kwargs_18885 = {}
    # Getting the type of 'allfields' (line 42)
    allfields_18878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'allfields', False)
    # Obtaining the member 'append' of a type (line 42)
    append_18879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), allfields_18878, 'append')
    # Calling append(args, kwargs) (line 42)
    append_call_result_18886 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), append_18879, *[tuple_18880], **kwargs_18885)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort(...): (line 44)
    # Processing the call keyword arguments (line 44)

    @norecursion
    def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_7'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 44, 23, True)
        # Passed parameters checking function
        _stypy_temp_lambda_7.stypy_localization = localization
        _stypy_temp_lambda_7.stypy_type_of_self = None
        _stypy_temp_lambda_7.stypy_type_store = module_type_store
        _stypy_temp_lambda_7.stypy_function_name = '_stypy_temp_lambda_7'
        _stypy_temp_lambda_7.stypy_param_names_list = ['x']
        _stypy_temp_lambda_7.stypy_varargs_param_name = None
        _stypy_temp_lambda_7.stypy_kwargs_param_name = None
        _stypy_temp_lambda_7.stypy_call_defaults = defaults
        _stypy_temp_lambda_7.stypy_call_varargs = varargs
        _stypy_temp_lambda_7.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_7', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_7', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Obtaining the type of the subscript
        int_18889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'int')
        # Getting the type of 'x' (line 44)
        x_18890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___18891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 33), x_18890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_18892 = invoke(stypy.reporting.localization.Localization(__file__, 44, 33), getitem___18891, int_18889)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'stypy_return_type', subscript_call_result_18892)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_7' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_18893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_7'
        return stypy_return_type_18893

    # Assigning a type to the variable '_stypy_temp_lambda_7' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
    # Getting the type of '_stypy_temp_lambda_7' (line 44)
    _stypy_temp_lambda_7_18894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), '_stypy_temp_lambda_7')
    keyword_18895 = _stypy_temp_lambda_7_18894
    kwargs_18896 = {'key': keyword_18895}
    # Getting the type of 'allfields' (line 44)
    allfields_18887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'allfields', False)
    # Obtaining the member 'sort' of a type (line 44)
    sort_18888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), allfields_18887, 'sort')
    # Calling sort(args, kwargs) (line 44)
    sort_call_result_18897 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), sort_18888, *[], **kwargs_18896)
    
    
    # Assigning a ListComp to a Name (line 45):
    
    # Assigning a ListComp to a Name (line 45):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'allfields' (line 45)
    allfields_18902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'allfields')
    comprehension_18903 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 13), allfields_18902)
    # Assigning a type to the variable 'x' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'x', comprehension_18903)
    
    # Obtaining the type of the subscript
    int_18898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'int')
    # Getting the type of 'x' (line 45)
    x_18899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___18900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), x_18899, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_18901 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), getitem___18900, int_18898)
    
    list_18904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 13), list_18904, subscript_call_result_18901)
    # Assigning a type to the variable 'names' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'names', list_18904)
    
    # Assigning a ListComp to a Name (line 46):
    
    # Assigning a ListComp to a Name (line 46):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'allfields' (line 46)
    allfields_18909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'allfields')
    comprehension_18910 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), allfields_18909)
    # Assigning a type to the variable 'x' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'x', comprehension_18910)
    
    # Obtaining the type of the subscript
    int_18905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'int')
    # Getting the type of 'x' (line 46)
    x_18906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___18907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), x_18906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_18908 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), getitem___18907, int_18905)
    
    list_18911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 15), list_18911, subscript_call_result_18908)
    # Assigning a type to the variable 'formats' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'formats', list_18911)
    
    # Assigning a ListComp to a Name (line 47):
    
    # Assigning a ListComp to a Name (line 47):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'allfields' (line 47)
    allfields_18916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'allfields')
    comprehension_18917 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 15), allfields_18916)
    # Assigning a type to the variable 'x' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'x', comprehension_18917)
    
    # Obtaining the type of the subscript
    int_18912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 17), 'int')
    # Getting the type of 'x' (line 47)
    x_18913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___18914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), x_18913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_18915 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), getitem___18914, int_18912)
    
    list_18918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 15), list_18918, subscript_call_result_18915)
    # Assigning a type to the variable 'offsets' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'offsets', list_18918)
    
    # Assigning a ListComp to a Name (line 48):
    
    # Assigning a ListComp to a Name (line 48):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'allfields' (line 48)
    allfields_18923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'allfields')
    comprehension_18924 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 14), allfields_18923)
    # Assigning a type to the variable 'x' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'x', comprehension_18924)
    
    # Obtaining the type of the subscript
    int_18919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'int')
    # Getting the type of 'x' (line 48)
    x_18920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'x')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___18921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), x_18920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_18922 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), getitem___18921, int_18919)
    
    list_18925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 14), list_18925, subscript_call_result_18922)
    # Assigning a type to the variable 'titles' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'titles', list_18925)
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_18926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    # Getting the type of 'names' (line 50)
    names_18927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'names')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_18926, names_18927)
    # Adding element type (line 50)
    # Getting the type of 'formats' (line 50)
    formats_18928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'formats')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_18926, formats_18928)
    # Adding element type (line 50)
    # Getting the type of 'offsets' (line 50)
    offsets_18929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'offsets')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_18926, offsets_18929)
    # Adding element type (line 50)
    # Getting the type of 'titles' (line 50)
    titles_18930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'titles')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_18926, titles_18930)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', tuple_18926)
    
    # ################# End of '_makenames_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_makenames_list' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_18931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_makenames_list'
    return stypy_return_type_18931

# Assigning a type to the variable '_makenames_list' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_makenames_list', _makenames_list)

@norecursion
def _usefields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_usefields'
    module_type_store = module_type_store.open_function_context('_usefields', 55, 0, False)
    
    # Passed parameters checking function
    _usefields.stypy_localization = localization
    _usefields.stypy_type_of_self = None
    _usefields.stypy_type_store = module_type_store
    _usefields.stypy_function_name = '_usefields'
    _usefields.stypy_param_names_list = ['adict', 'align']
    _usefields.stypy_varargs_param_name = None
    _usefields.stypy_kwargs_param_name = None
    _usefields.stypy_call_defaults = defaults
    _usefields.stypy_call_varargs = varargs
    _usefields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_usefields', ['adict', 'align'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_usefields', localization, ['adict', 'align'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_usefields(...)' code ##################

    
    
    # SSA begins for try-except statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 57):
    
    # Assigning a Subscript to a Name (line 57):
    
    # Obtaining the type of the subscript
    int_18932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
    # Getting the type of 'adict' (line 57)
    adict_18933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'adict')
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___18934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), adict_18933, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_18935 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), getitem___18934, int_18932)
    
    # Assigning a type to the variable 'names' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'names', subscript_call_result_18935)
    # SSA branch for the except part of a try statement (line 56)
    # SSA branch for the except 'KeyError' branch of a try statement (line 56)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 59):
    
    # Assigning a Name to a Name (line 59):
    # Getting the type of 'None' (line 59)
    None_18936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'None')
    # Assigning a type to the variable 'names' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'names', None_18936)
    # SSA join for try-except statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 60)
    # Getting the type of 'names' (line 60)
    names_18937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'names')
    # Getting the type of 'None' (line 60)
    None_18938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'None')
    
    (may_be_18939, more_types_in_union_18940) = may_be_none(names_18937, None_18938)

    if may_be_18939:

        if more_types_in_union_18940:
            # Runtime conditional SSA (line 60)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Tuple (line 61):
        
        # Assigning a Call to a Name:
        
        # Call to _makenames_list(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'adict' (line 61)
        adict_18942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 58), 'adict', False)
        # Getting the type of 'align' (line 61)
        align_18943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 65), 'align', False)
        # Processing the call keyword arguments (line 61)
        kwargs_18944 = {}
        # Getting the type of '_makenames_list' (line 61)
        _makenames_list_18941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), '_makenames_list', False)
        # Calling _makenames_list(args, kwargs) (line 61)
        _makenames_list_call_result_18945 = invoke(stypy.reporting.localization.Localization(__file__, 61, 42), _makenames_list_18941, *[adict_18942, align_18943], **kwargs_18944)
        
        # Assigning a type to the variable 'call_assignment_18753' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18753', _makenames_list_call_result_18945)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_18948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        # Processing the call keyword arguments
        kwargs_18949 = {}
        # Getting the type of 'call_assignment_18753' (line 61)
        call_assignment_18753_18946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18753', False)
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___18947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), call_assignment_18753_18946, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_18950 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___18947, *[int_18948], **kwargs_18949)
        
        # Assigning a type to the variable 'call_assignment_18754' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18754', getitem___call_result_18950)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'call_assignment_18754' (line 61)
        call_assignment_18754_18951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18754')
        # Assigning a type to the variable 'names' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'names', call_assignment_18754_18951)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_18954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        # Processing the call keyword arguments
        kwargs_18955 = {}
        # Getting the type of 'call_assignment_18753' (line 61)
        call_assignment_18753_18952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18753', False)
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___18953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), call_assignment_18753_18952, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_18956 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___18953, *[int_18954], **kwargs_18955)
        
        # Assigning a type to the variable 'call_assignment_18755' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18755', getitem___call_result_18956)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'call_assignment_18755' (line 61)
        call_assignment_18755_18957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18755')
        # Assigning a type to the variable 'formats' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'formats', call_assignment_18755_18957)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_18960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        # Processing the call keyword arguments
        kwargs_18961 = {}
        # Getting the type of 'call_assignment_18753' (line 61)
        call_assignment_18753_18958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18753', False)
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___18959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), call_assignment_18753_18958, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_18962 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___18959, *[int_18960], **kwargs_18961)
        
        # Assigning a type to the variable 'call_assignment_18756' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18756', getitem___call_result_18962)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'call_assignment_18756' (line 61)
        call_assignment_18756_18963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18756')
        # Assigning a type to the variable 'offsets' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'offsets', call_assignment_18756_18963)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_18966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        # Processing the call keyword arguments
        kwargs_18967 = {}
        # Getting the type of 'call_assignment_18753' (line 61)
        call_assignment_18753_18964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18753', False)
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___18965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), call_assignment_18753_18964, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_18968 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___18965, *[int_18966], **kwargs_18967)
        
        # Assigning a type to the variable 'call_assignment_18757' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18757', getitem___call_result_18968)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'call_assignment_18757' (line 61)
        call_assignment_18757_18969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'call_assignment_18757')
        # Assigning a type to the variable 'titles' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'titles', call_assignment_18757_18969)

        if more_types_in_union_18940:
            # Runtime conditional SSA for else branch (line 60)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_18939) or more_types_in_union_18940):
        
        # Assigning a List to a Name (line 63):
        
        # Assigning a List to a Name (line 63):
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_18970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        
        # Assigning a type to the variable 'formats' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'formats', list_18970)
        
        # Assigning a List to a Name (line 64):
        
        # Assigning a List to a Name (line 64):
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_18971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        
        # Assigning a type to the variable 'offsets' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'offsets', list_18971)
        
        # Assigning a List to a Name (line 65):
        
        # Assigning a List to a Name (line 65):
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_18972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        
        # Assigning a type to the variable 'titles' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'titles', list_18972)
        
        # Getting the type of 'names' (line 66)
        names_18973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'names')
        # Testing the type of a for loop iterable (line 66)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 8), names_18973)
        # Getting the type of the for loop variable (line 66)
        for_loop_var_18974 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 8), names_18973)
        # Assigning a type to the variable 'name' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'name', for_loop_var_18974)
        # SSA begins for a for statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 67):
        
        # Assigning a Subscript to a Name (line 67):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 67)
        name_18975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'name')
        # Getting the type of 'adict' (line 67)
        adict_18976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'adict')
        # Obtaining the member '__getitem__' of a type (line 67)
        getitem___18977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), adict_18976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 67)
        subscript_call_result_18978 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), getitem___18977, name_18975)
        
        # Assigning a type to the variable 'res' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'res', subscript_call_result_18978)
        
        # Call to append(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining the type of the subscript
        int_18981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'int')
        # Getting the type of 'res' (line 68)
        res_18982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___18983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 27), res_18982, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_18984 = invoke(stypy.reporting.localization.Localization(__file__, 68, 27), getitem___18983, int_18981)
        
        # Processing the call keyword arguments (line 68)
        kwargs_18985 = {}
        # Getting the type of 'formats' (line 68)
        formats_18979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'formats', False)
        # Obtaining the member 'append' of a type (line 68)
        append_18980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), formats_18979, 'append')
        # Calling append(args, kwargs) (line 68)
        append_call_result_18986 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), append_18980, *[subscript_call_result_18984], **kwargs_18985)
        
        
        # Call to append(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining the type of the subscript
        int_18989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'int')
        # Getting the type of 'res' (line 69)
        res_18990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___18991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 27), res_18990, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_18992 = invoke(stypy.reporting.localization.Localization(__file__, 69, 27), getitem___18991, int_18989)
        
        # Processing the call keyword arguments (line 69)
        kwargs_18993 = {}
        # Getting the type of 'offsets' (line 69)
        offsets_18987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'offsets', False)
        # Obtaining the member 'append' of a type (line 69)
        append_18988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), offsets_18987, 'append')
        # Calling append(args, kwargs) (line 69)
        append_call_result_18994 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), append_18988, *[subscript_call_result_18992], **kwargs_18993)
        
        
        
        
        # Call to len(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'res' (line 70)
        res_18996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'res', False)
        # Processing the call keyword arguments (line 70)
        kwargs_18997 = {}
        # Getting the type of 'len' (line 70)
        len_18995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'len', False)
        # Calling len(args, kwargs) (line 70)
        len_call_result_18998 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), len_18995, *[res_18996], **kwargs_18997)
        
        int_18999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 27), 'int')
        # Applying the binary operator '>' (line 70)
        result_gt_19000 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 16), '>', len_call_result_18998, int_18999)
        
        # Testing the type of an if condition (line 70)
        if_condition_19001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), result_gt_19000)
        # Assigning a type to the variable 'if_condition_19001' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_19001', if_condition_19001)
        # SSA begins for if statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining the type of the subscript
        int_19004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'int')
        # Getting the type of 'res' (line 71)
        res_19005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___19006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 30), res_19005, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_19007 = invoke(stypy.reporting.localization.Localization(__file__, 71, 30), getitem___19006, int_19004)
        
        # Processing the call keyword arguments (line 71)
        kwargs_19008 = {}
        # Getting the type of 'titles' (line 71)
        titles_19002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'titles', False)
        # Obtaining the member 'append' of a type (line 71)
        append_19003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), titles_19002, 'append')
        # Calling append(args, kwargs) (line 71)
        append_call_result_19009 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), append_19003, *[subscript_call_result_19007], **kwargs_19008)
        
        # SSA branch for the else part of an if statement (line 70)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'None' (line 73)
        None_19012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'None', False)
        # Processing the call keyword arguments (line 73)
        kwargs_19013 = {}
        # Getting the type of 'titles' (line 73)
        titles_19010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'titles', False)
        # Obtaining the member 'append' of a type (line 73)
        append_19011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), titles_19010, 'append')
        # Calling append(args, kwargs) (line 73)
        append_call_result_19014 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), append_19011, *[None_19012], **kwargs_19013)
        
        # SSA join for if statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_18939 and more_types_in_union_18940):
            # SSA join for if statement (line 60)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to dtype(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining an instance of the builtin type 'dict' (line 75)
    dict_19016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 75)
    # Adding element type (key, value) (line 75)
    str_19017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'str', 'names')
    # Getting the type of 'names' (line 75)
    names_19018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'names', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_19016, (str_19017, names_19018))
    # Adding element type (key, value) (line 75)
    str_19019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'str', 'formats')
    # Getting the type of 'formats' (line 76)
    formats_19020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'formats', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_19016, (str_19019, formats_19020))
    # Adding element type (key, value) (line 75)
    str_19021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'str', 'offsets')
    # Getting the type of 'offsets' (line 77)
    offsets_19022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'offsets', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_19016, (str_19021, offsets_19022))
    # Adding element type (key, value) (line 75)
    str_19023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 18), 'str', 'titles')
    # Getting the type of 'titles' (line 78)
    titles_19024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'titles', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 17), dict_19016, (str_19023, titles_19024))
    
    # Getting the type of 'align' (line 78)
    align_19025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'align', False)
    # Processing the call keyword arguments (line 75)
    kwargs_19026 = {}
    # Getting the type of 'dtype' (line 75)
    dtype_19015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'dtype', False)
    # Calling dtype(args, kwargs) (line 75)
    dtype_call_result_19027 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), dtype_19015, *[dict_19016, align_19025], **kwargs_19026)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', dtype_call_result_19027)
    
    # ################# End of '_usefields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_usefields' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_19028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19028)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_usefields'
    return stypy_return_type_19028

# Assigning a type to the variable '_usefields' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '_usefields', _usefields)

@norecursion
def _array_descr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_array_descr'
    module_type_store = module_type_store.open_function_context('_array_descr', 87, 0, False)
    
    # Passed parameters checking function
    _array_descr.stypy_localization = localization
    _array_descr.stypy_type_of_self = None
    _array_descr.stypy_type_store = module_type_store
    _array_descr.stypy_function_name = '_array_descr'
    _array_descr.stypy_param_names_list = ['descriptor']
    _array_descr.stypy_varargs_param_name = None
    _array_descr.stypy_kwargs_param_name = None
    _array_descr.stypy_call_defaults = defaults
    _array_descr.stypy_call_varargs = varargs
    _array_descr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_array_descr', ['descriptor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_array_descr', localization, ['descriptor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_array_descr(...)' code ##################

    
    # Assigning a Attribute to a Name (line 88):
    
    # Assigning a Attribute to a Name (line 88):
    # Getting the type of 'descriptor' (line 88)
    descriptor_19029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'descriptor')
    # Obtaining the member 'fields' of a type (line 88)
    fields_19030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 13), descriptor_19029, 'fields')
    # Assigning a type to the variable 'fields' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'fields', fields_19030)
    
    # Type idiom detected: calculating its left and rigth part (line 89)
    # Getting the type of 'fields' (line 89)
    fields_19031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'fields')
    # Getting the type of 'None' (line 89)
    None_19032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'None')
    
    (may_be_19033, more_types_in_union_19034) = may_be_none(fields_19031, None_19032)

    if may_be_19033:

        if more_types_in_union_19034:
            # Runtime conditional SSA (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 90):
        
        # Assigning a Attribute to a Name (line 90):
        # Getting the type of 'descriptor' (line 90)
        descriptor_19035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'descriptor')
        # Obtaining the member 'subdtype' of a type (line 90)
        subdtype_19036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), descriptor_19035, 'subdtype')
        # Assigning a type to the variable 'subdtype' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'subdtype', subdtype_19036)
        
        # Type idiom detected: calculating its left and rigth part (line 91)
        # Getting the type of 'subdtype' (line 91)
        subdtype_19037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'subdtype')
        # Getting the type of 'None' (line 91)
        None_19038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'None')
        
        (may_be_19039, more_types_in_union_19040) = may_be_none(subdtype_19037, None_19038)

        if may_be_19039:

            if more_types_in_union_19040:
                # Runtime conditional SSA (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 92)
            # Getting the type of 'descriptor' (line 92)
            descriptor_19041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'descriptor')
            # Obtaining the member 'metadata' of a type (line 92)
            metadata_19042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), descriptor_19041, 'metadata')
            # Getting the type of 'None' (line 92)
            None_19043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'None')
            
            (may_be_19044, more_types_in_union_19045) = may_be_none(metadata_19042, None_19043)

            if may_be_19044:

                if more_types_in_union_19045:
                    # Runtime conditional SSA (line 92)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'descriptor' (line 93)
                descriptor_19046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'descriptor')
                # Obtaining the member 'str' of a type (line 93)
                str_19047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), descriptor_19046, 'str')
                # Assigning a type to the variable 'stypy_return_type' (line 93)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'stypy_return_type', str_19047)

                if more_types_in_union_19045:
                    # Runtime conditional SSA for else branch (line 92)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_19044) or more_types_in_union_19045):
                
                # Assigning a Call to a Name (line 95):
                
                # Assigning a Call to a Name (line 95):
                
                # Call to copy(...): (line 95)
                # Processing the call keyword arguments (line 95)
                kwargs_19051 = {}
                # Getting the type of 'descriptor' (line 95)
                descriptor_19048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'descriptor', False)
                # Obtaining the member 'metadata' of a type (line 95)
                metadata_19049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 22), descriptor_19048, 'metadata')
                # Obtaining the member 'copy' of a type (line 95)
                copy_19050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 22), metadata_19049, 'copy')
                # Calling copy(args, kwargs) (line 95)
                copy_call_result_19052 = invoke(stypy.reporting.localization.Localization(__file__, 95, 22), copy_19050, *[], **kwargs_19051)
                
                # Assigning a type to the variable 'new' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'new', copy_call_result_19052)
                
                # Getting the type of 'new' (line 96)
                new_19053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'new')
                # Testing the type of an if condition (line 96)
                if_condition_19054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 16), new_19053)
                # Assigning a type to the variable 'if_condition_19054' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'if_condition_19054', if_condition_19054)
                # SSA begins for if statement (line 96)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 97)
                tuple_19055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 97)
                # Adding element type (line 97)
                # Getting the type of 'descriptor' (line 97)
                descriptor_19056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'descriptor')
                # Obtaining the member 'str' of a type (line 97)
                str_19057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), descriptor_19056, 'str')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), tuple_19055, str_19057)
                # Adding element type (line 97)
                # Getting the type of 'new' (line 97)
                new_19058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 44), 'new')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), tuple_19055, new_19058)
                
                # Assigning a type to the variable 'stypy_return_type' (line 97)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'stypy_return_type', tuple_19055)
                # SSA branch for the else part of an if statement (line 96)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'descriptor' (line 99)
                descriptor_19059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'descriptor')
                # Obtaining the member 'str' of a type (line 99)
                str_19060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), descriptor_19059, 'str')
                # Assigning a type to the variable 'stypy_return_type' (line 99)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'stypy_return_type', str_19060)
                # SSA join for if statement (line 96)
                module_type_store = module_type_store.join_ssa_context()
                

                if (may_be_19044 and more_types_in_union_19045):
                    # SSA join for if statement (line 92)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_19040:
                # Runtime conditional SSA for else branch (line 91)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_19039) or more_types_in_union_19040):
            
            # Obtaining an instance of the builtin type 'tuple' (line 101)
            tuple_19061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 101)
            # Adding element type (line 101)
            
            # Call to _array_descr(...): (line 101)
            # Processing the call arguments (line 101)
            
            # Obtaining the type of the subscript
            int_19063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'int')
            # Getting the type of 'subdtype' (line 101)
            subdtype_19064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'subdtype', False)
            # Obtaining the member '__getitem__' of a type (line 101)
            getitem___19065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 33), subdtype_19064, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 101)
            subscript_call_result_19066 = invoke(stypy.reporting.localization.Localization(__file__, 101, 33), getitem___19065, int_19063)
            
            # Processing the call keyword arguments (line 101)
            kwargs_19067 = {}
            # Getting the type of '_array_descr' (line 101)
            _array_descr_19062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), '_array_descr', False)
            # Calling _array_descr(args, kwargs) (line 101)
            _array_descr_call_result_19068 = invoke(stypy.reporting.localization.Localization(__file__, 101, 20), _array_descr_19062, *[subscript_call_result_19066], **kwargs_19067)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), tuple_19061, _array_descr_call_result_19068)
            # Adding element type (line 101)
            
            # Obtaining the type of the subscript
            int_19069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 56), 'int')
            # Getting the type of 'subdtype' (line 101)
            subdtype_19070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 47), 'subdtype')
            # Obtaining the member '__getitem__' of a type (line 101)
            getitem___19071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 47), subdtype_19070, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 101)
            subscript_call_result_19072 = invoke(stypy.reporting.localization.Localization(__file__, 101, 47), getitem___19071, int_19069)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 20), tuple_19061, subscript_call_result_19072)
            
            # Assigning a type to the variable 'stypy_return_type' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', tuple_19061)

            if (may_be_19039 and more_types_in_union_19040):
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_19034:
            # SSA join for if statement (line 89)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 103):
    
    # Assigning a Attribute to a Name (line 103):
    # Getting the type of 'descriptor' (line 103)
    descriptor_19073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'descriptor')
    # Obtaining the member 'names' of a type (line 103)
    names_19074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), descriptor_19073, 'names')
    # Assigning a type to the variable 'names' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'names', names_19074)
    
    # Assigning a ListComp to a Name (line 104):
    
    # Assigning a ListComp to a Name (line 104):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'names' (line 104)
    names_19082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'names')
    comprehension_19083 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 22), names_19082)
    # Assigning a type to the variable 'x' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'x', comprehension_19083)
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 104)
    x_19075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'x')
    # Getting the type of 'fields' (line 104)
    fields_19076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'fields')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___19077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 22), fields_19076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_19078 = invoke(stypy.reporting.localization.Localization(__file__, 104, 22), getitem___19077, x_19075)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_19079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    # Getting the type of 'x' (line 104)
    x_19080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 35), tuple_19079, x_19080)
    
    # Applying the binary operator '+' (line 104)
    result_add_19081 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), '+', subscript_call_result_19078, tuple_19079)
    
    list_19084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 22), list_19084, result_add_19081)
    # Assigning a type to the variable 'ordered_fields' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'ordered_fields', list_19084)
    
    # Assigning a List to a Name (line 105):
    
    # Assigning a List to a Name (line 105):
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_19085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    
    # Assigning a type to the variable 'result' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'result', list_19085)
    
    # Assigning a Num to a Name (line 106):
    
    # Assigning a Num to a Name (line 106):
    int_19086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 13), 'int')
    # Assigning a type to the variable 'offset' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'offset', int_19086)
    
    # Getting the type of 'ordered_fields' (line 107)
    ordered_fields_19087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'ordered_fields')
    # Testing the type of a for loop iterable (line 107)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 107, 4), ordered_fields_19087)
    # Getting the type of the for loop variable (line 107)
    for_loop_var_19088 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 107, 4), ordered_fields_19087)
    # Assigning a type to the variable 'field' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'field', for_loop_var_19088)
    # SSA begins for a for statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_19089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 17), 'int')
    # Getting the type of 'field' (line 108)
    field_19090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'field')
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___19091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), field_19090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_19092 = invoke(stypy.reporting.localization.Localization(__file__, 108, 11), getitem___19091, int_19089)
    
    # Getting the type of 'offset' (line 108)
    offset_19093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'offset')
    # Applying the binary operator '>' (line 108)
    result_gt_19094 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), '>', subscript_call_result_19092, offset_19093)
    
    # Testing the type of an if condition (line 108)
    if_condition_19095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 8), result_gt_19094)
    # Assigning a type to the variable 'if_condition_19095' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'if_condition_19095', if_condition_19095)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 109):
    
    # Assigning a BinOp to a Name (line 109):
    
    # Obtaining the type of the subscript
    int_19096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 24), 'int')
    # Getting the type of 'field' (line 109)
    field_19097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'field')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___19098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 18), field_19097, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_19099 = invoke(stypy.reporting.localization.Localization(__file__, 109, 18), getitem___19098, int_19096)
    
    # Getting the type of 'offset' (line 109)
    offset_19100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 29), 'offset')
    # Applying the binary operator '-' (line 109)
    result_sub_19101 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 18), '-', subscript_call_result_19099, offset_19100)
    
    # Assigning a type to the variable 'num' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'num', result_sub_19101)
    
    # Call to append(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining an instance of the builtin type 'tuple' (line 110)
    tuple_19104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 110)
    # Adding element type (line 110)
    str_19105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 27), tuple_19104, str_19105)
    # Adding element type (line 110)
    str_19106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'str', '|V%d')
    # Getting the type of 'num' (line 110)
    num_19107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 40), 'num', False)
    # Applying the binary operator '%' (line 110)
    result_mod_19108 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 31), '%', str_19106, num_19107)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 27), tuple_19104, result_mod_19108)
    
    # Processing the call keyword arguments (line 110)
    kwargs_19109 = {}
    # Getting the type of 'result' (line 110)
    result_19102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'result', False)
    # Obtaining the member 'append' of a type (line 110)
    append_19103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), result_19102, 'append')
    # Calling append(args, kwargs) (line 110)
    append_call_result_19110 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), append_19103, *[tuple_19104], **kwargs_19109)
    
    
    # Getting the type of 'offset' (line 111)
    offset_19111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'offset')
    # Getting the type of 'num' (line 111)
    num_19112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'num')
    # Applying the binary operator '+=' (line 111)
    result_iadd_19113 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 12), '+=', offset_19111, num_19112)
    # Assigning a type to the variable 'offset' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'offset', result_iadd_19113)
    
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'field' (line 112)
    field_19115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'field', False)
    # Processing the call keyword arguments (line 112)
    kwargs_19116 = {}
    # Getting the type of 'len' (line 112)
    len_19114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'len', False)
    # Calling len(args, kwargs) (line 112)
    len_call_result_19117 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), len_19114, *[field_19115], **kwargs_19116)
    
    int_19118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'int')
    # Applying the binary operator '>' (line 112)
    result_gt_19119 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), '>', len_call_result_19117, int_19118)
    
    # Testing the type of an if condition (line 112)
    if_condition_19120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_gt_19119)
    # Assigning a type to the variable 'if_condition_19120' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_19120', if_condition_19120)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 113):
    
    # Assigning a Tuple to a Name (line 113):
    
    # Obtaining an instance of the builtin type 'tuple' (line 113)
    tuple_19121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 113)
    # Adding element type (line 113)
    
    # Obtaining the type of the subscript
    int_19122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 26), 'int')
    # Getting the type of 'field' (line 113)
    field_19123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'field')
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___19124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), field_19123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_19125 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), getitem___19124, int_19122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), tuple_19121, subscript_call_result_19125)
    # Adding element type (line 113)
    
    # Obtaining the type of the subscript
    int_19126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 36), 'int')
    # Getting the type of 'field' (line 113)
    field_19127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'field')
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___19128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 30), field_19127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_19129 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), getitem___19128, int_19126)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), tuple_19121, subscript_call_result_19129)
    
    # Assigning a type to the variable 'name' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'name', tuple_19121)
    # SSA branch for the else part of an if statement (line 112)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 115):
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_19130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'int')
    # Getting the type of 'field' (line 115)
    field_19131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'field')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___19132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 19), field_19131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_19133 = invoke(stypy.reporting.localization.Localization(__file__, 115, 19), getitem___19132, int_19130)
    
    # Assigning a type to the variable 'name' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'name', subscript_call_result_19133)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    int_19134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 17), 'int')
    # Getting the type of 'field' (line 116)
    field_19135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'field')
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___19136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), field_19135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_19137 = invoke(stypy.reporting.localization.Localization(__file__, 116, 11), getitem___19136, int_19134)
    
    # Obtaining the member 'subdtype' of a type (line 116)
    subdtype_19138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), subscript_call_result_19137, 'subdtype')
    # Testing the type of an if condition (line 116)
    if_condition_19139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), subdtype_19138)
    # Assigning a type to the variable 'if_condition_19139' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_19139', if_condition_19139)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 117):
    
    # Assigning a Tuple to a Name (line 117):
    
    # Obtaining an instance of the builtin type 'tuple' (line 117)
    tuple_19140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 117)
    # Adding element type (line 117)
    # Getting the type of 'name' (line 117)
    name_19141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 19), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 19), tuple_19140, name_19141)
    # Adding element type (line 117)
    
    # Call to _array_descr(...): (line 117)
    # Processing the call arguments (line 117)
    
    # Obtaining the type of the subscript
    int_19143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 56), 'int')
    
    # Obtaining the type of the subscript
    int_19144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'int')
    # Getting the type of 'field' (line 117)
    field_19145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'field', False)
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___19146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 38), field_19145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_19147 = invoke(stypy.reporting.localization.Localization(__file__, 117, 38), getitem___19146, int_19144)
    
    # Obtaining the member 'subdtype' of a type (line 117)
    subdtype_19148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 38), subscript_call_result_19147, 'subdtype')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___19149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 38), subdtype_19148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_19150 = invoke(stypy.reporting.localization.Localization(__file__, 117, 38), getitem___19149, int_19143)
    
    # Processing the call keyword arguments (line 117)
    kwargs_19151 = {}
    # Getting the type of '_array_descr' (line 117)
    _array_descr_19142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), '_array_descr', False)
    # Calling _array_descr(args, kwargs) (line 117)
    _array_descr_call_result_19152 = invoke(stypy.reporting.localization.Localization(__file__, 117, 25), _array_descr_19142, *[subscript_call_result_19150], **kwargs_19151)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 19), tuple_19140, _array_descr_call_result_19152)
    # Adding element type (line 117)
    
    # Obtaining the type of the subscript
    int_19153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'int')
    
    # Obtaining the type of the subscript
    int_19154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'int')
    # Getting the type of 'field' (line 118)
    field_19155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'field')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___19156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), field_19155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_19157 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), getitem___19156, int_19154)
    
    # Obtaining the member 'subdtype' of a type (line 118)
    subdtype_19158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), subscript_call_result_19157, 'subdtype')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___19159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), subdtype_19158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_19160 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), getitem___19159, int_19153)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 19), tuple_19140, subscript_call_result_19160)
    
    # Assigning a type to the variable 'tup' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'tup', tuple_19140)
    # SSA branch for the else part of an if statement (line 116)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 120):
    
    # Assigning a Tuple to a Name (line 120):
    
    # Obtaining an instance of the builtin type 'tuple' (line 120)
    tuple_19161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 120)
    # Adding element type (line 120)
    # Getting the type of 'name' (line 120)
    name_19162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), tuple_19161, name_19162)
    # Adding element type (line 120)
    
    # Call to _array_descr(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Obtaining the type of the subscript
    int_19164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 44), 'int')
    # Getting the type of 'field' (line 120)
    field_19165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 38), 'field', False)
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___19166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 38), field_19165, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_19167 = invoke(stypy.reporting.localization.Localization(__file__, 120, 38), getitem___19166, int_19164)
    
    # Processing the call keyword arguments (line 120)
    kwargs_19168 = {}
    # Getting the type of '_array_descr' (line 120)
    _array_descr_19163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), '_array_descr', False)
    # Calling _array_descr(args, kwargs) (line 120)
    _array_descr_call_result_19169 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), _array_descr_19163, *[subscript_call_result_19167], **kwargs_19168)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 19), tuple_19161, _array_descr_call_result_19169)
    
    # Assigning a type to the variable 'tup' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tup', tuple_19161)
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'offset' (line 121)
    offset_19170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'offset')
    
    # Obtaining the type of the subscript
    int_19171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'int')
    # Getting the type of 'field' (line 121)
    field_19172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'field')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___19173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 18), field_19172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_19174 = invoke(stypy.reporting.localization.Localization(__file__, 121, 18), getitem___19173, int_19171)
    
    # Obtaining the member 'itemsize' of a type (line 121)
    itemsize_19175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 18), subscript_call_result_19174, 'itemsize')
    # Applying the binary operator '+=' (line 121)
    result_iadd_19176 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 8), '+=', offset_19170, itemsize_19175)
    # Assigning a type to the variable 'offset' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'offset', result_iadd_19176)
    
    
    # Call to append(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'tup' (line 122)
    tup_19179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'tup', False)
    # Processing the call keyword arguments (line 122)
    kwargs_19180 = {}
    # Getting the type of 'result' (line 122)
    result_19177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 122)
    append_19178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), result_19177, 'append')
    # Calling append(args, kwargs) (line 122)
    append_call_result_19181 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), append_19178, *[tup_19179], **kwargs_19180)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'descriptor' (line 124)
    descriptor_19182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'descriptor')
    # Obtaining the member 'itemsize' of a type (line 124)
    itemsize_19183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 7), descriptor_19182, 'itemsize')
    # Getting the type of 'offset' (line 124)
    offset_19184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'offset')
    # Applying the binary operator '>' (line 124)
    result_gt_19185 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), '>', itemsize_19183, offset_19184)
    
    # Testing the type of an if condition (line 124)
    if_condition_19186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_gt_19185)
    # Assigning a type to the variable 'if_condition_19186' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_19186', if_condition_19186)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 125):
    
    # Assigning a BinOp to a Name (line 125):
    # Getting the type of 'descriptor' (line 125)
    descriptor_19187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'descriptor')
    # Obtaining the member 'itemsize' of a type (line 125)
    itemsize_19188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 14), descriptor_19187, 'itemsize')
    # Getting the type of 'offset' (line 125)
    offset_19189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'offset')
    # Applying the binary operator '-' (line 125)
    result_sub_19190 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 14), '-', itemsize_19188, offset_19189)
    
    # Assigning a type to the variable 'num' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'num', result_sub_19190)
    
    # Call to append(...): (line 126)
    # Processing the call arguments (line 126)
    
    # Obtaining an instance of the builtin type 'tuple' (line 126)
    tuple_19193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 126)
    # Adding element type (line 126)
    str_19194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 23), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 23), tuple_19193, str_19194)
    # Adding element type (line 126)
    str_19195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 27), 'str', '|V%d')
    # Getting the type of 'num' (line 126)
    num_19196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'num', False)
    # Applying the binary operator '%' (line 126)
    result_mod_19197 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 27), '%', str_19195, num_19196)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 23), tuple_19193, result_mod_19197)
    
    # Processing the call keyword arguments (line 126)
    kwargs_19198 = {}
    # Getting the type of 'result' (line 126)
    result_19191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 126)
    append_19192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), result_19191, 'append')
    # Calling append(args, kwargs) (line 126)
    append_call_result_19199 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), append_19192, *[tuple_19193], **kwargs_19198)
    
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 128)
    result_19200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', result_19200)
    
    # ################# End of '_array_descr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_array_descr' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_19201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19201)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_array_descr'
    return stypy_return_type_19201

# Assigning a type to the variable '_array_descr' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), '_array_descr', _array_descr)

@norecursion
def _reconstruct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_reconstruct'
    module_type_store = module_type_store.open_function_context('_reconstruct', 135, 0, False)
    
    # Passed parameters checking function
    _reconstruct.stypy_localization = localization
    _reconstruct.stypy_type_of_self = None
    _reconstruct.stypy_type_store = module_type_store
    _reconstruct.stypy_function_name = '_reconstruct'
    _reconstruct.stypy_param_names_list = ['subtype', 'shape', 'dtype']
    _reconstruct.stypy_varargs_param_name = None
    _reconstruct.stypy_kwargs_param_name = None
    _reconstruct.stypy_call_defaults = defaults
    _reconstruct.stypy_call_varargs = varargs
    _reconstruct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_reconstruct', ['subtype', 'shape', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_reconstruct', localization, ['subtype', 'shape', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_reconstruct(...)' code ##################

    
    # Call to __new__(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'subtype' (line 136)
    subtype_19204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'subtype', False)
    # Getting the type of 'shape' (line 136)
    shape_19205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 36), 'shape', False)
    # Getting the type of 'dtype' (line 136)
    dtype_19206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 43), 'dtype', False)
    # Processing the call keyword arguments (line 136)
    kwargs_19207 = {}
    # Getting the type of 'ndarray' (line 136)
    ndarray_19202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'ndarray', False)
    # Obtaining the member '__new__' of a type (line 136)
    new___19203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 11), ndarray_19202, '__new__')
    # Calling __new__(args, kwargs) (line 136)
    new___call_result_19208 = invoke(stypy.reporting.localization.Localization(__file__, 136, 11), new___19203, *[subtype_19204, shape_19205, dtype_19206], **kwargs_19207)
    
    # Assigning a type to the variable 'stypy_return_type' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type', new___call_result_19208)
    
    # ################# End of '_reconstruct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_reconstruct' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_19209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19209)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_reconstruct'
    return stypy_return_type_19209

# Assigning a type to the variable '_reconstruct' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), '_reconstruct', _reconstruct)

# Assigning a Call to a Name (line 141):

# Assigning a Call to a Name (line 141):

# Call to compile(...): (line 141)
# Processing the call arguments (line 141)

# Call to asbytes(...): (line 141)
# Processing the call arguments (line 141)
str_19213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'str', '(?P<order1>[<>|=]?)(?P<repeats> *[(]?[ ,0-9L]*[)]? *)(?P<order2>[<>|=]?)(?P<dtype>[A-Za-z0-9.?]*(?:\\[[a-zA-Z0-9,.]+\\])?)')
# Processing the call keyword arguments (line 141)
kwargs_19214 = {}
# Getting the type of 'asbytes' (line 141)
asbytes_19212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 141)
asbytes_call_result_19215 = invoke(stypy.reporting.localization.Localization(__file__, 141, 23), asbytes_19212, *[str_19213], **kwargs_19214)

# Processing the call keyword arguments (line 141)
kwargs_19216 = {}
# Getting the type of 're' (line 141)
re_19210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 're', False)
# Obtaining the member 'compile' of a type (line 141)
compile_19211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), re_19210, 'compile')
# Calling compile(args, kwargs) (line 141)
compile_call_result_19217 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), compile_19211, *[asbytes_call_result_19215], **kwargs_19216)

# Assigning a type to the variable 'format_re' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'format_re', compile_call_result_19217)

# Assigning a Call to a Name (line 146):

# Assigning a Call to a Name (line 146):

# Call to compile(...): (line 146)
# Processing the call arguments (line 146)

# Call to asbytes(...): (line 146)
# Processing the call arguments (line 146)
str_19221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'str', '\\s*,\\s*')
# Processing the call keyword arguments (line 146)
kwargs_19222 = {}
# Getting the type of 'asbytes' (line 146)
asbytes_19220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 146)
asbytes_call_result_19223 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), asbytes_19220, *[str_19221], **kwargs_19222)

# Processing the call keyword arguments (line 146)
kwargs_19224 = {}
# Getting the type of 're' (line 146)
re_19218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 9), 're', False)
# Obtaining the member 'compile' of a type (line 146)
compile_19219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 9), re_19218, 'compile')
# Calling compile(args, kwargs) (line 146)
compile_call_result_19225 = invoke(stypy.reporting.localization.Localization(__file__, 146, 9), compile_19219, *[asbytes_call_result_19223], **kwargs_19224)

# Assigning a type to the variable 'sep_re' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'sep_re', compile_call_result_19225)

# Assigning a Call to a Name (line 147):

# Assigning a Call to a Name (line 147):

# Call to compile(...): (line 147)
# Processing the call arguments (line 147)

# Call to asbytes(...): (line 147)
# Processing the call arguments (line 147)
str_19229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'str', '\\s+$')
# Processing the call keyword arguments (line 147)
kwargs_19230 = {}
# Getting the type of 'asbytes' (line 147)
asbytes_19228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 147)
asbytes_call_result_19231 = invoke(stypy.reporting.localization.Localization(__file__, 147, 22), asbytes_19228, *[str_19229], **kwargs_19230)

# Processing the call keyword arguments (line 147)
kwargs_19232 = {}
# Getting the type of 're' (line 147)
re_19226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 're', False)
# Obtaining the member 'compile' of a type (line 147)
compile_19227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 11), re_19226, 'compile')
# Calling compile(args, kwargs) (line 147)
compile_call_result_19233 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), compile_19227, *[asbytes_call_result_19231], **kwargs_19232)

# Assigning a type to the variable 'space_re' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'space_re', compile_call_result_19233)

# Assigning a Dict to a Name (line 151):

# Assigning a Dict to a Name (line 151):

# Obtaining an instance of the builtin type 'dict' (line 151)
dict_19234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 151)
# Adding element type (key, value) (line 151)

# Call to asbytes(...): (line 151)
# Processing the call arguments (line 151)
str_19236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 22), 'str', '=')
# Processing the call keyword arguments (line 151)
kwargs_19237 = {}
# Getting the type of 'asbytes' (line 151)
asbytes_19235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 151)
asbytes_call_result_19238 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), asbytes_19235, *[str_19236], **kwargs_19237)

# Getting the type of '_nbo' (line 151)
_nbo_19239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), '_nbo')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 13), dict_19234, (asbytes_call_result_19238, _nbo_19239))

# Assigning a type to the variable '_convorder' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), '_convorder', dict_19234)

@norecursion
def _commastring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_commastring'
    module_type_store = module_type_store.open_function_context('_commastring', 153, 0, False)
    
    # Passed parameters checking function
    _commastring.stypy_localization = localization
    _commastring.stypy_type_of_self = None
    _commastring.stypy_type_store = module_type_store
    _commastring.stypy_function_name = '_commastring'
    _commastring.stypy_param_names_list = ['astr']
    _commastring.stypy_varargs_param_name = None
    _commastring.stypy_kwargs_param_name = None
    _commastring.stypy_call_defaults = defaults
    _commastring.stypy_call_varargs = varargs
    _commastring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_commastring', ['astr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_commastring', localization, ['astr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_commastring(...)' code ##################

    
    # Assigning a Num to a Name (line 154):
    
    # Assigning a Num to a Name (line 154):
    int_19240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'int')
    # Assigning a type to the variable 'startindex' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'startindex', int_19240)
    
    # Assigning a List to a Name (line 155):
    
    # Assigning a List to a Name (line 155):
    
    # Obtaining an instance of the builtin type 'list' (line 155)
    list_19241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 155)
    
    # Assigning a type to the variable 'result' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'result', list_19241)
    
    
    # Getting the type of 'startindex' (line 156)
    startindex_19242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 10), 'startindex')
    
    # Call to len(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'astr' (line 156)
    astr_19244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'astr', False)
    # Processing the call keyword arguments (line 156)
    kwargs_19245 = {}
    # Getting the type of 'len' (line 156)
    len_19243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'len', False)
    # Calling len(args, kwargs) (line 156)
    len_call_result_19246 = invoke(stypy.reporting.localization.Localization(__file__, 156, 23), len_19243, *[astr_19244], **kwargs_19245)
    
    # Applying the binary operator '<' (line 156)
    result_lt_19247 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 10), '<', startindex_19242, len_call_result_19246)
    
    # Testing the type of an if condition (line 156)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 4), result_lt_19247)
    # SSA begins for while statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to match(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'astr' (line 157)
    astr_19250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'astr', False)
    # Processing the call keyword arguments (line 157)
    # Getting the type of 'startindex' (line 157)
    startindex_19251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'startindex', False)
    keyword_19252 = startindex_19251
    kwargs_19253 = {'pos': keyword_19252}
    # Getting the type of 'format_re' (line 157)
    format_re_19248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'format_re', False)
    # Obtaining the member 'match' of a type (line 157)
    match_19249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 13), format_re_19248, 'match')
    # Calling match(args, kwargs) (line 157)
    match_call_result_19254 = invoke(stypy.reporting.localization.Localization(__file__, 157, 13), match_19249, *[astr_19250], **kwargs_19253)
    
    # Assigning a type to the variable 'mo' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'mo', match_call_result_19254)
    
    
    # SSA begins for try-except statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 159):
    
    # Assigning a Call to a Name:
    
    # Call to groups(...): (line 159)
    # Processing the call keyword arguments (line 159)
    kwargs_19257 = {}
    # Getting the type of 'mo' (line 159)
    mo_19255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 47), 'mo', False)
    # Obtaining the member 'groups' of a type (line 159)
    groups_19256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 47), mo_19255, 'groups')
    # Calling groups(args, kwargs) (line 159)
    groups_call_result_19258 = invoke(stypy.reporting.localization.Localization(__file__, 159, 47), groups_19256, *[], **kwargs_19257)
    
    # Assigning a type to the variable 'call_assignment_18758' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18758', groups_call_result_19258)
    
    # Assigning a Call to a Name (line 159):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_19261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
    # Processing the call keyword arguments
    kwargs_19262 = {}
    # Getting the type of 'call_assignment_18758' (line 159)
    call_assignment_18758_19259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18758', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___19260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), call_assignment_18758_19259, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_19263 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___19260, *[int_19261], **kwargs_19262)
    
    # Assigning a type to the variable 'call_assignment_18759' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18759', getitem___call_result_19263)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'call_assignment_18759' (line 159)
    call_assignment_18759_19264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18759')
    # Assigning a type to the variable 'order1' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'order1', call_assignment_18759_19264)
    
    # Assigning a Call to a Name (line 159):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_19267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
    # Processing the call keyword arguments
    kwargs_19268 = {}
    # Getting the type of 'call_assignment_18758' (line 159)
    call_assignment_18758_19265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18758', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___19266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), call_assignment_18758_19265, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_19269 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___19266, *[int_19267], **kwargs_19268)
    
    # Assigning a type to the variable 'call_assignment_18760' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18760', getitem___call_result_19269)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'call_assignment_18760' (line 159)
    call_assignment_18760_19270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18760')
    # Assigning a type to the variable 'repeats' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'repeats', call_assignment_18760_19270)
    
    # Assigning a Call to a Name (line 159):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_19273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
    # Processing the call keyword arguments
    kwargs_19274 = {}
    # Getting the type of 'call_assignment_18758' (line 159)
    call_assignment_18758_19271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18758', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___19272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), call_assignment_18758_19271, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_19275 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___19272, *[int_19273], **kwargs_19274)
    
    # Assigning a type to the variable 'call_assignment_18761' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18761', getitem___call_result_19275)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'call_assignment_18761' (line 159)
    call_assignment_18761_19276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18761')
    # Assigning a type to the variable 'order2' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'order2', call_assignment_18761_19276)
    
    # Assigning a Call to a Name (line 159):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_19279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
    # Processing the call keyword arguments
    kwargs_19280 = {}
    # Getting the type of 'call_assignment_18758' (line 159)
    call_assignment_18758_19277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18758', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___19278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), call_assignment_18758_19277, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_19281 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___19278, *[int_19279], **kwargs_19280)
    
    # Assigning a type to the variable 'call_assignment_18762' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18762', getitem___call_result_19281)
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'call_assignment_18762' (line 159)
    call_assignment_18762_19282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'call_assignment_18762')
    # Assigning a type to the variable 'dtype' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 38), 'dtype', call_assignment_18762_19282)
    # SSA branch for the except part of a try statement (line 158)
    # SSA branch for the except 'Tuple' branch of a try statement (line 158)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 161)
    # Processing the call arguments (line 161)
    str_19284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 29), 'str', 'format number %d of "%s" is not recognized')
    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_19285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    
    # Call to len(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'result' (line 162)
    result_19287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 49), 'result', False)
    # Processing the call keyword arguments (line 162)
    kwargs_19288 = {}
    # Getting the type of 'len' (line 162)
    len_19286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 45), 'len', False)
    # Calling len(args, kwargs) (line 162)
    len_call_result_19289 = invoke(stypy.reporting.localization.Localization(__file__, 162, 45), len_19286, *[result_19287], **kwargs_19288)
    
    int_19290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 57), 'int')
    # Applying the binary operator '+' (line 162)
    result_add_19291 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 45), '+', len_call_result_19289, int_19290)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 45), tuple_19285, result_add_19291)
    # Adding element type (line 162)
    # Getting the type of 'astr' (line 162)
    astr_19292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 60), 'astr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 45), tuple_19285, astr_19292)
    
    # Applying the binary operator '%' (line 161)
    result_mod_19293 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 29), '%', str_19284, tuple_19285)
    
    # Processing the call keyword arguments (line 161)
    kwargs_19294 = {}
    # Getting the type of 'ValueError' (line 161)
    ValueError_19283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 161)
    ValueError_call_result_19295 = invoke(stypy.reporting.localization.Localization(__file__, 161, 18), ValueError_19283, *[result_mod_19293], **kwargs_19294)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 161, 12), ValueError_call_result_19295, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to end(...): (line 163)
    # Processing the call keyword arguments (line 163)
    kwargs_19298 = {}
    # Getting the type of 'mo' (line 163)
    mo_19296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'mo', False)
    # Obtaining the member 'end' of a type (line 163)
    end_19297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), mo_19296, 'end')
    # Calling end(args, kwargs) (line 163)
    end_call_result_19299 = invoke(stypy.reporting.localization.Localization(__file__, 163, 21), end_19297, *[], **kwargs_19298)
    
    # Assigning a type to the variable 'startindex' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'startindex', end_call_result_19299)
    
    
    # Getting the type of 'startindex' (line 165)
    startindex_19300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'startindex')
    
    # Call to len(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'astr' (line 165)
    astr_19302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'astr', False)
    # Processing the call keyword arguments (line 165)
    kwargs_19303 = {}
    # Getting the type of 'len' (line 165)
    len_19301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'len', False)
    # Calling len(args, kwargs) (line 165)
    len_call_result_19304 = invoke(stypy.reporting.localization.Localization(__file__, 165, 24), len_19301, *[astr_19302], **kwargs_19303)
    
    # Applying the binary operator '<' (line 165)
    result_lt_19305 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 11), '<', startindex_19300, len_call_result_19304)
    
    # Testing the type of an if condition (line 165)
    if_condition_19306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), result_lt_19305)
    # Assigning a type to the variable 'if_condition_19306' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_19306', if_condition_19306)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to match(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'astr' (line 166)
    astr_19309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'astr', False)
    # Processing the call keyword arguments (line 166)
    # Getting the type of 'startindex' (line 166)
    startindex_19310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 40), 'startindex', False)
    keyword_19311 = startindex_19310
    kwargs_19312 = {'pos': keyword_19311}
    # Getting the type of 'space_re' (line 166)
    space_re_19307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'space_re', False)
    # Obtaining the member 'match' of a type (line 166)
    match_19308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 15), space_re_19307, 'match')
    # Calling match(args, kwargs) (line 166)
    match_call_result_19313 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), match_19308, *[astr_19309], **kwargs_19312)
    
    # Testing the type of an if condition (line 166)
    if_condition_19314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 12), match_call_result_19313)
    # Assigning a type to the variable 'if_condition_19314' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'if_condition_19314', if_condition_19314)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to len(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'astr' (line 167)
    astr_19316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'astr', False)
    # Processing the call keyword arguments (line 167)
    kwargs_19317 = {}
    # Getting the type of 'len' (line 167)
    len_19315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'len', False)
    # Calling len(args, kwargs) (line 167)
    len_call_result_19318 = invoke(stypy.reporting.localization.Localization(__file__, 167, 29), len_19315, *[astr_19316], **kwargs_19317)
    
    # Assigning a type to the variable 'startindex' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'startindex', len_call_result_19318)
    # SSA branch for the else part of an if statement (line 166)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to match(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'astr' (line 169)
    astr_19321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'astr', False)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'startindex' (line 169)
    startindex_19322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 44), 'startindex', False)
    keyword_19323 = startindex_19322
    kwargs_19324 = {'pos': keyword_19323}
    # Getting the type of 'sep_re' (line 169)
    sep_re_19319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'sep_re', False)
    # Obtaining the member 'match' of a type (line 169)
    match_19320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 21), sep_re_19319, 'match')
    # Calling match(args, kwargs) (line 169)
    match_call_result_19325 = invoke(stypy.reporting.localization.Localization(__file__, 169, 21), match_19320, *[astr_19321], **kwargs_19324)
    
    # Assigning a type to the variable 'mo' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'mo', match_call_result_19325)
    
    
    # Getting the type of 'mo' (line 170)
    mo_19326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'mo')
    # Applying the 'not' unary operator (line 170)
    result_not__19327 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 19), 'not', mo_19326)
    
    # Testing the type of an if condition (line 170)
    if_condition_19328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 16), result_not__19327)
    # Assigning a type to the variable 'if_condition_19328' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'if_condition_19328', if_condition_19328)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 171)
    # Processing the call arguments (line 171)
    str_19330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'str', 'format number %d of "%s" is not recognized')
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_19331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    # Adding element type (line 173)
    
    # Call to len(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'result' (line 173)
    result_19333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'result', False)
    # Processing the call keyword arguments (line 173)
    kwargs_19334 = {}
    # Getting the type of 'len' (line 173)
    len_19332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 'len', False)
    # Calling len(args, kwargs) (line 173)
    len_call_result_19335 = invoke(stypy.reporting.localization.Localization(__file__, 173, 25), len_19332, *[result_19333], **kwargs_19334)
    
    int_19336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 37), 'int')
    # Applying the binary operator '+' (line 173)
    result_add_19337 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 25), '+', len_call_result_19335, int_19336)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 25), tuple_19331, result_add_19337)
    # Adding element type (line 173)
    # Getting the type of 'astr' (line 173)
    astr_19338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 40), 'astr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 25), tuple_19331, astr_19338)
    
    # Applying the binary operator '%' (line 172)
    result_mod_19339 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 24), '%', str_19330, tuple_19331)
    
    # Processing the call keyword arguments (line 171)
    kwargs_19340 = {}
    # Getting the type of 'ValueError' (line 171)
    ValueError_19329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 171)
    ValueError_call_result_19341 = invoke(stypy.reporting.localization.Localization(__file__, 171, 26), ValueError_19329, *[result_mod_19339], **kwargs_19340)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 171, 20), ValueError_call_result_19341, 'raise parameter', BaseException)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 174):
    
    # Assigning a Call to a Name (line 174):
    
    # Call to end(...): (line 174)
    # Processing the call keyword arguments (line 174)
    kwargs_19344 = {}
    # Getting the type of 'mo' (line 174)
    mo_19342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'mo', False)
    # Obtaining the member 'end' of a type (line 174)
    end_19343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 29), mo_19342, 'end')
    # Calling end(args, kwargs) (line 174)
    end_call_result_19345 = invoke(stypy.reporting.localization.Localization(__file__, 174, 29), end_19343, *[], **kwargs_19344)
    
    # Assigning a type to the variable 'startindex' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'startindex', end_call_result_19345)
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'order2' (line 176)
    order2_19346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'order2')
    
    # Call to asbytes(...): (line 176)
    # Processing the call arguments (line 176)
    str_19348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 29), 'str', '')
    # Processing the call keyword arguments (line 176)
    kwargs_19349 = {}
    # Getting the type of 'asbytes' (line 176)
    asbytes_19347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 176)
    asbytes_call_result_19350 = invoke(stypy.reporting.localization.Localization(__file__, 176, 21), asbytes_19347, *[str_19348], **kwargs_19349)
    
    # Applying the binary operator '==' (line 176)
    result_eq_19351 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 11), '==', order2_19346, asbytes_call_result_19350)
    
    # Testing the type of an if condition (line 176)
    if_condition_19352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 8), result_eq_19351)
    # Assigning a type to the variable 'if_condition_19352' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'if_condition_19352', if_condition_19352)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 177):
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'order1' (line 177)
    order1_19353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'order1')
    # Assigning a type to the variable 'order' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'order', order1_19353)
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'order1' (line 178)
    order1_19354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'order1')
    
    # Call to asbytes(...): (line 178)
    # Processing the call arguments (line 178)
    str_19356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'str', '')
    # Processing the call keyword arguments (line 178)
    kwargs_19357 = {}
    # Getting the type of 'asbytes' (line 178)
    asbytes_19355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 178)
    asbytes_call_result_19358 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), asbytes_19355, *[str_19356], **kwargs_19357)
    
    # Applying the binary operator '==' (line 178)
    result_eq_19359 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 13), '==', order1_19354, asbytes_call_result_19358)
    
    # Testing the type of an if condition (line 178)
    if_condition_19360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 13), result_eq_19359)
    # Assigning a type to the variable 'if_condition_19360' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'if_condition_19360', if_condition_19360)
    # SSA begins for if statement (line 178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 179):
    
    # Assigning a Name to a Name (line 179):
    # Getting the type of 'order2' (line 179)
    order2_19361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'order2')
    # Assigning a type to the variable 'order' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'order', order2_19361)
    # SSA branch for the else part of an if statement (line 178)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 181):
    
    # Assigning a Call to a Name (line 181):
    
    # Call to get(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'order1' (line 181)
    order1_19364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'order1', False)
    # Getting the type of 'order1' (line 181)
    order1_19365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 44), 'order1', False)
    # Processing the call keyword arguments (line 181)
    kwargs_19366 = {}
    # Getting the type of '_convorder' (line 181)
    _convorder_19362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), '_convorder', False)
    # Obtaining the member 'get' of a type (line 181)
    get_19363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 21), _convorder_19362, 'get')
    # Calling get(args, kwargs) (line 181)
    get_call_result_19367 = invoke(stypy.reporting.localization.Localization(__file__, 181, 21), get_19363, *[order1_19364, order1_19365], **kwargs_19366)
    
    # Assigning a type to the variable 'order1' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'order1', get_call_result_19367)
    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to get(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'order2' (line 182)
    order2_19370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 36), 'order2', False)
    # Getting the type of 'order2' (line 182)
    order2_19371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'order2', False)
    # Processing the call keyword arguments (line 182)
    kwargs_19372 = {}
    # Getting the type of '_convorder' (line 182)
    _convorder_19368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 21), '_convorder', False)
    # Obtaining the member 'get' of a type (line 182)
    get_19369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 21), _convorder_19368, 'get')
    # Calling get(args, kwargs) (line 182)
    get_call_result_19373 = invoke(stypy.reporting.localization.Localization(__file__, 182, 21), get_19369, *[order2_19370, order2_19371], **kwargs_19372)
    
    # Assigning a type to the variable 'order2' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'order2', get_call_result_19373)
    
    
    # Getting the type of 'order1' (line 183)
    order1_19374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'order1')
    # Getting the type of 'order2' (line 183)
    order2_19375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'order2')
    # Applying the binary operator '!=' (line 183)
    result_ne_19376 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 16), '!=', order1_19374, order2_19375)
    
    # Testing the type of an if condition (line 183)
    if_condition_19377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 12), result_ne_19376)
    # Assigning a type to the variable 'if_condition_19377' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'if_condition_19377', if_condition_19377)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 184)
    # Processing the call arguments (line 184)
    str_19379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'str', 'inconsistent byte-order specification %s and %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 186)
    tuple_19380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 186)
    # Adding element type (line 186)
    # Getting the type of 'order1' (line 186)
    order1_19381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 21), 'order1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), tuple_19380, order1_19381)
    # Adding element type (line 186)
    # Getting the type of 'order2' (line 186)
    order2_19382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'order2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 21), tuple_19380, order2_19382)
    
    # Applying the binary operator '%' (line 185)
    result_mod_19383 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 20), '%', str_19379, tuple_19380)
    
    # Processing the call keyword arguments (line 184)
    kwargs_19384 = {}
    # Getting the type of 'ValueError' (line 184)
    ValueError_19378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 184)
    ValueError_call_result_19385 = invoke(stypy.reporting.localization.Localization(__file__, 184, 22), ValueError_19378, *[result_mod_19383], **kwargs_19384)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 184, 16), ValueError_call_result_19385, 'raise parameter', BaseException)
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 187):
    
    # Assigning a Name to a Name (line 187):
    # Getting the type of 'order1' (line 187)
    order1_19386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'order1')
    # Assigning a type to the variable 'order' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'order', order1_19386)
    # SSA join for if statement (line 178)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'order' (line 189)
    order_19387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'order')
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_19388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    # Adding element type (line 189)
    
    # Call to asbytes(...): (line 189)
    # Processing the call arguments (line 189)
    str_19390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 29), 'str', '|')
    # Processing the call keyword arguments (line 189)
    kwargs_19391 = {}
    # Getting the type of 'asbytes' (line 189)
    asbytes_19389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 189)
    asbytes_call_result_19392 = invoke(stypy.reporting.localization.Localization(__file__, 189, 21), asbytes_19389, *[str_19390], **kwargs_19391)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 20), list_19388, asbytes_call_result_19392)
    # Adding element type (line 189)
    
    # Call to asbytes(...): (line 189)
    # Processing the call arguments (line 189)
    str_19394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 43), 'str', '=')
    # Processing the call keyword arguments (line 189)
    kwargs_19395 = {}
    # Getting the type of 'asbytes' (line 189)
    asbytes_19393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 189)
    asbytes_call_result_19396 = invoke(stypy.reporting.localization.Localization(__file__, 189, 35), asbytes_19393, *[str_19394], **kwargs_19395)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 20), list_19388, asbytes_call_result_19396)
    # Adding element type (line 189)
    # Getting the type of '_nbo' (line 189)
    _nbo_19397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 49), '_nbo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 20), list_19388, _nbo_19397)
    
    # Applying the binary operator 'in' (line 189)
    result_contains_19398 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), 'in', order_19387, list_19388)
    
    # Testing the type of an if condition (line 189)
    if_condition_19399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 8), result_contains_19398)
    # Assigning a type to the variable 'if_condition_19399' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'if_condition_19399', if_condition_19399)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to asbytes(...): (line 190)
    # Processing the call arguments (line 190)
    str_19401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 28), 'str', '')
    # Processing the call keyword arguments (line 190)
    kwargs_19402 = {}
    # Getting the type of 'asbytes' (line 190)
    asbytes_19400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 190)
    asbytes_call_result_19403 = invoke(stypy.reporting.localization.Localization(__file__, 190, 20), asbytes_19400, *[str_19401], **kwargs_19402)
    
    # Assigning a type to the variable 'order' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'order', asbytes_call_result_19403)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 191):
    
    # Assigning a BinOp to a Name (line 191):
    # Getting the type of 'order' (line 191)
    order_19404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'order')
    # Getting the type of 'dtype' (line 191)
    dtype_19405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'dtype')
    # Applying the binary operator '+' (line 191)
    result_add_19406 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '+', order_19404, dtype_19405)
    
    # Assigning a type to the variable 'dtype' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'dtype', result_add_19406)
    
    
    # Getting the type of 'repeats' (line 192)
    repeats_19407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'repeats')
    
    # Call to asbytes(...): (line 192)
    # Processing the call arguments (line 192)
    str_19409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 31), 'str', '')
    # Processing the call keyword arguments (line 192)
    kwargs_19410 = {}
    # Getting the type of 'asbytes' (line 192)
    asbytes_19408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'asbytes', False)
    # Calling asbytes(args, kwargs) (line 192)
    asbytes_call_result_19411 = invoke(stypy.reporting.localization.Localization(__file__, 192, 23), asbytes_19408, *[str_19409], **kwargs_19410)
    
    # Applying the binary operator '==' (line 192)
    result_eq_19412 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 12), '==', repeats_19407, asbytes_call_result_19411)
    
    # Testing the type of an if condition (line 192)
    if_condition_19413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), result_eq_19412)
    # Assigning a type to the variable 'if_condition_19413' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_19413', if_condition_19413)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 193):
    
    # Assigning a Name to a Name (line 193):
    # Getting the type of 'dtype' (line 193)
    dtype_19414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'dtype')
    # Assigning a type to the variable 'newitem' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'newitem', dtype_19414)
    # SSA branch for the else part of an if statement (line 192)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 195):
    
    # Assigning a Tuple to a Name (line 195):
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_19415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    # Getting the type of 'dtype' (line 195)
    dtype_19416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 23), tuple_19415, dtype_19416)
    # Adding element type (line 195)
    
    # Call to eval(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'repeats' (line 195)
    repeats_19418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'repeats', False)
    # Processing the call keyword arguments (line 195)
    kwargs_19419 = {}
    # Getting the type of 'eval' (line 195)
    eval_19417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'eval', False)
    # Calling eval(args, kwargs) (line 195)
    eval_call_result_19420 = invoke(stypy.reporting.localization.Localization(__file__, 195, 30), eval_19417, *[repeats_19418], **kwargs_19419)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 23), tuple_19415, eval_call_result_19420)
    
    # Assigning a type to the variable 'newitem' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'newitem', tuple_19415)
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'newitem' (line 196)
    newitem_19423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 22), 'newitem', False)
    # Processing the call keyword arguments (line 196)
    kwargs_19424 = {}
    # Getting the type of 'result' (line 196)
    result_19421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 196)
    append_19422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), result_19421, 'append')
    # Calling append(args, kwargs) (line 196)
    append_call_result_19425 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), append_19422, *[newitem_19423], **kwargs_19424)
    
    # SSA join for while statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 198)
    result_19426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type', result_19426)
    
    # ################# End of '_commastring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_commastring' in the type store
    # Getting the type of 'stypy_return_type' (line 153)
    stypy_return_type_19427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_commastring'
    return stypy_return_type_19427

# Assigning a type to the variable '_commastring' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), '_commastring', _commastring)

@norecursion
def _getintp_ctype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_getintp_ctype'
    module_type_store = module_type_store.open_function_context('_getintp_ctype', 200, 0, False)
    
    # Passed parameters checking function
    _getintp_ctype.stypy_localization = localization
    _getintp_ctype.stypy_type_of_self = None
    _getintp_ctype.stypy_type_store = module_type_store
    _getintp_ctype.stypy_function_name = '_getintp_ctype'
    _getintp_ctype.stypy_param_names_list = []
    _getintp_ctype.stypy_varargs_param_name = None
    _getintp_ctype.stypy_kwargs_param_name = None
    _getintp_ctype.stypy_call_defaults = defaults
    _getintp_ctype.stypy_call_varargs = varargs
    _getintp_ctype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getintp_ctype', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getintp_ctype', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getintp_ctype(...)' code ##################

    
    # Assigning a Attribute to a Name (line 201):
    
    # Assigning a Attribute to a Name (line 201):
    # Getting the type of '_getintp_ctype' (line 201)
    _getintp_ctype_19428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 10), '_getintp_ctype')
    # Obtaining the member 'cache' of a type (line 201)
    cache_19429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 10), _getintp_ctype_19428, 'cache')
    # Assigning a type to the variable 'val' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'val', cache_19429)
    
    # Type idiom detected: calculating its left and rigth part (line 202)
    # Getting the type of 'val' (line 202)
    val_19430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'val')
    # Getting the type of 'None' (line 202)
    None_19431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'None')
    
    (may_be_19432, more_types_in_union_19433) = may_not_be_none(val_19430, None_19431)

    if may_be_19432:

        if more_types_in_union_19433:
            # Runtime conditional SSA (line 202)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'val' (line 203)
        val_19434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'val')
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', val_19434)

        if more_types_in_union_19433:
            # SSA join for if statement (line 202)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 204):
    
    # Assigning a Attribute to a Name (line 204):
    
    # Call to dtype(...): (line 204)
    # Processing the call arguments (line 204)
    str_19436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 17), 'str', 'p')
    # Processing the call keyword arguments (line 204)
    kwargs_19437 = {}
    # Getting the type of 'dtype' (line 204)
    dtype_19435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'dtype', False)
    # Calling dtype(args, kwargs) (line 204)
    dtype_call_result_19438 = invoke(stypy.reporting.localization.Localization(__file__, 204, 11), dtype_19435, *[str_19436], **kwargs_19437)
    
    # Obtaining the member 'char' of a type (line 204)
    char_19439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), dtype_call_result_19438, 'char')
    # Assigning a type to the variable 'char' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'char', char_19439)
    
    
    # Getting the type of 'char' (line 205)
    char_19440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'char')
    str_19441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 16), 'str', 'i')
    # Applying the binary operator '==' (line 205)
    result_eq_19442 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 8), '==', char_19440, str_19441)
    
    # Testing the type of an if condition (line 205)
    if_condition_19443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 4), result_eq_19442)
    # Assigning a type to the variable 'if_condition_19443' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'if_condition_19443', if_condition_19443)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 206):
    
    # Assigning a Attribute to a Name (line 206):
    # Getting the type of 'ctypes' (line 206)
    ctypes_19444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 14), 'ctypes')
    # Obtaining the member 'c_int' of a type (line 206)
    c_int_19445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 14), ctypes_19444, 'c_int')
    # Assigning a type to the variable 'val' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'val', c_int_19445)
    # SSA branch for the else part of an if statement (line 205)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'char' (line 207)
    char_19446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 9), 'char')
    str_19447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 17), 'str', 'l')
    # Applying the binary operator '==' (line 207)
    result_eq_19448 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 9), '==', char_19446, str_19447)
    
    # Testing the type of an if condition (line 207)
    if_condition_19449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 9), result_eq_19448)
    # Assigning a type to the variable 'if_condition_19449' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 9), 'if_condition_19449', if_condition_19449)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 208):
    
    # Assigning a Attribute to a Name (line 208):
    # Getting the type of 'ctypes' (line 208)
    ctypes_19450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 14), 'ctypes')
    # Obtaining the member 'c_long' of a type (line 208)
    c_long_19451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 14), ctypes_19450, 'c_long')
    # Assigning a type to the variable 'val' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'val', c_long_19451)
    # SSA branch for the else part of an if statement (line 207)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'char' (line 209)
    char_19452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'char')
    str_19453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 17), 'str', 'q')
    # Applying the binary operator '==' (line 209)
    result_eq_19454 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 9), '==', char_19452, str_19453)
    
    # Testing the type of an if condition (line 209)
    if_condition_19455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 9), result_eq_19454)
    # Assigning a type to the variable 'if_condition_19455' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'if_condition_19455', if_condition_19455)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 210):
    
    # Assigning a Attribute to a Name (line 210):
    # Getting the type of 'ctypes' (line 210)
    ctypes_19456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'ctypes')
    # Obtaining the member 'c_longlong' of a type (line 210)
    c_longlong_19457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 14), ctypes_19456, 'c_longlong')
    # Assigning a type to the variable 'val' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'val', c_longlong_19457)
    # SSA branch for the else part of an if statement (line 209)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 212):
    
    # Assigning a Attribute to a Name (line 212):
    # Getting the type of 'ctypes' (line 212)
    ctypes_19458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'ctypes')
    # Obtaining the member 'c_long' of a type (line 212)
    c_long_19459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 14), ctypes_19458, 'c_long')
    # Assigning a type to the variable 'val' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'val', c_long_19459)
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Attribute (line 213):
    
    # Assigning a Name to a Attribute (line 213):
    # Getting the type of 'val' (line 213)
    val_19460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'val')
    # Getting the type of '_getintp_ctype' (line 213)
    _getintp_ctype_19461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), '_getintp_ctype')
    # Setting the type of the member 'cache' of a type (line 213)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 4), _getintp_ctype_19461, 'cache', val_19460)
    # Getting the type of 'val' (line 214)
    val_19462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'val')
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', val_19462)
    
    # ################# End of '_getintp_ctype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getintp_ctype' in the type store
    # Getting the type of 'stypy_return_type' (line 200)
    stypy_return_type_19463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19463)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getintp_ctype'
    return stypy_return_type_19463

# Assigning a type to the variable '_getintp_ctype' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), '_getintp_ctype', _getintp_ctype)

# Assigning a Name to a Attribute (line 215):

# Assigning a Name to a Attribute (line 215):
# Getting the type of 'None' (line 215)
None_19464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'None')
# Getting the type of '_getintp_ctype' (line 215)
_getintp_ctype_19465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), '_getintp_ctype')
# Setting the type of the member 'cache' of a type (line 215)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 0), _getintp_ctype_19465, 'cache', None_19464)
# Declaration of the '_missing_ctypes' class

class _missing_ctypes(object, ):

    @norecursion
    def cast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cast'
        module_type_store = module_type_store.open_function_context('cast', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _missing_ctypes.cast.__dict__.__setitem__('stypy_localization', localization)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_type_store', module_type_store)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_function_name', '_missing_ctypes.cast')
        _missing_ctypes.cast.__dict__.__setitem__('stypy_param_names_list', ['num', 'obj'])
        _missing_ctypes.cast.__dict__.__setitem__('stypy_varargs_param_name', None)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_call_defaults', defaults)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_call_varargs', varargs)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _missing_ctypes.cast.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_missing_ctypes.cast', ['num', 'obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cast', localization, ['num', 'obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cast(...)' code ##################

        # Getting the type of 'num' (line 221)
        num_19466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'num')
        # Assigning a type to the variable 'stypy_return_type' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'stypy_return_type', num_19466)
        
        # ################# End of 'cast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cast' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_19467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cast'
        return stypy_return_type_19467


    @norecursion
    def c_void_p(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'c_void_p'
        module_type_store = module_type_store.open_function_context('c_void_p', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_localization', localization)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_type_store', module_type_store)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_function_name', '_missing_ctypes.c_void_p')
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_param_names_list', ['num'])
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_varargs_param_name', None)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_call_defaults', defaults)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_call_varargs', varargs)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _missing_ctypes.c_void_p.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_missing_ctypes.c_void_p', ['num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'c_void_p', localization, ['num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'c_void_p(...)' code ##################

        # Getting the type of 'num' (line 224)
        num_19468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'num')
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', num_19468)
        
        # ################# End of 'c_void_p(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'c_void_p' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_19469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'c_void_p'
        return stypy_return_type_19469


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 219, 0, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_missing_ctypes.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_missing_ctypes' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), '_missing_ctypes', _missing_ctypes)
# Declaration of the '_ctypes' class

class _ctypes(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 227)
        None_19470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'None')
        defaults = [None_19470]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 227, 4, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.__init__', ['array', 'ptr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['array', 'ptr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Attribute (line 229):
        
        # Assigning a Name to a Attribute (line 229):
        # Getting the type of 'ctypes' (line 229)
        ctypes_19471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'ctypes')
        # Getting the type of 'self' (line 229)
        self_19472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self')
        # Setting the type of the member '_ctypes' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), self_19472, '_ctypes', ctypes_19471)
        # SSA branch for the except part of a try statement (line 228)
        # SSA branch for the except 'ImportError' branch of a try statement (line 228)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Attribute (line 231):
        
        # Assigning a Call to a Attribute (line 231):
        
        # Call to _missing_ctypes(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_19474 = {}
        # Getting the type of '_missing_ctypes' (line 231)
        _missing_ctypes_19473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), '_missing_ctypes', False)
        # Calling _missing_ctypes(args, kwargs) (line 231)
        _missing_ctypes_call_result_19475 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), _missing_ctypes_19473, *[], **kwargs_19474)
        
        # Getting the type of 'self' (line 231)
        self_19476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self')
        # Setting the type of the member '_ctypes' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_19476, '_ctypes', _missing_ctypes_call_result_19475)
        # SSA join for try-except statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 232):
        
        # Assigning a Name to a Attribute (line 232):
        # Getting the type of 'array' (line 232)
        array_19477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'array')
        # Getting the type of 'self' (line 232)
        self_19478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self')
        # Setting the type of the member '_arr' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_19478, '_arr', array_19477)
        
        # Assigning a Name to a Attribute (line 233):
        
        # Assigning a Name to a Attribute (line 233):
        # Getting the type of 'ptr' (line 233)
        ptr_19479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'ptr')
        # Getting the type of 'self' (line 233)
        self_19480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self')
        # Setting the type of the member '_data' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_19480, '_data', ptr_19479)
        
        
        # Getting the type of 'self' (line 234)
        self_19481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'self')
        # Obtaining the member '_arr' of a type (line 234)
        _arr_19482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 11), self_19481, '_arr')
        # Obtaining the member 'ndim' of a type (line 234)
        ndim_19483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 11), _arr_19482, 'ndim')
        int_19484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 29), 'int')
        # Applying the binary operator '==' (line 234)
        result_eq_19485 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 11), '==', ndim_19483, int_19484)
        
        # Testing the type of an if condition (line 234)
        if_condition_19486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 8), result_eq_19485)
        # Assigning a type to the variable 'if_condition_19486' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'if_condition_19486', if_condition_19486)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 235):
        
        # Assigning a Name to a Attribute (line 235):
        # Getting the type of 'True' (line 235)
        True_19487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'True')
        # Getting the type of 'self' (line 235)
        self_19488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self')
        # Setting the type of the member '_zerod' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_19488, '_zerod', True_19487)
        # SSA branch for the else part of an if statement (line 234)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 237):
        
        # Assigning a Name to a Attribute (line 237):
        # Getting the type of 'False' (line 237)
        False_19489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'False')
        # Getting the type of 'self' (line 237)
        self_19490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self')
        # Setting the type of the member '_zerod' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_19490, '_zerod', False_19489)
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def data_as(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'data_as'
        module_type_store = module_type_store.open_function_context('data_as', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.data_as.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.data_as.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.data_as.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.data_as.__dict__.__setitem__('stypy_function_name', '_ctypes.data_as')
        _ctypes.data_as.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        _ctypes.data_as.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.data_as.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.data_as.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.data_as.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.data_as.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.data_as.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.data_as', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'data_as', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'data_as(...)' code ##################

        
        # Call to cast(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_19494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 33), 'self', False)
        # Obtaining the member '_data' of a type (line 240)
        _data_19495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 33), self_19494, '_data')
        # Getting the type of 'obj' (line 240)
        obj_19496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'obj', False)
        # Processing the call keyword arguments (line 240)
        kwargs_19497 = {}
        # Getting the type of 'self' (line 240)
        self_19491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'self', False)
        # Obtaining the member '_ctypes' of a type (line 240)
        _ctypes_19492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), self_19491, '_ctypes')
        # Obtaining the member 'cast' of a type (line 240)
        cast_19493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), _ctypes_19492, 'cast')
        # Calling cast(args, kwargs) (line 240)
        cast_call_result_19498 = invoke(stypy.reporting.localization.Localization(__file__, 240, 15), cast_19493, *[_data_19495, obj_19496], **kwargs_19497)
        
        # Assigning a type to the variable 'stypy_return_type' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'stypy_return_type', cast_call_result_19498)
        
        # ################# End of 'data_as(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'data_as' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_19499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19499)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'data_as'
        return stypy_return_type_19499


    @norecursion
    def shape_as(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shape_as'
        module_type_store = module_type_store.open_function_context('shape_as', 242, 4, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.shape_as.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.shape_as.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.shape_as.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.shape_as.__dict__.__setitem__('stypy_function_name', '_ctypes.shape_as')
        _ctypes.shape_as.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        _ctypes.shape_as.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.shape_as.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.shape_as.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.shape_as.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.shape_as.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.shape_as.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.shape_as', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shape_as', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shape_as(...)' code ##################

        
        # Getting the type of 'self' (line 243)
        self_19500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'self')
        # Obtaining the member '_zerod' of a type (line 243)
        _zerod_19501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 11), self_19500, '_zerod')
        # Testing the type of an if condition (line 243)
        if_condition_19502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), _zerod_19501)
        # Assigning a type to the variable 'if_condition_19502' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_19502', if_condition_19502)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 244)
        None_19503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'stypy_return_type', None_19503)
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 245)
        # Getting the type of 'self' (line 245)
        self_19509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 37), 'self', False)
        # Obtaining the member '_arr' of a type (line 245)
        _arr_19510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 37), self_19509, '_arr')
        # Obtaining the member 'shape' of a type (line 245)
        shape_19511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 37), _arr_19510, 'shape')
        # Processing the call keyword arguments (line 245)
        kwargs_19512 = {}
        # Getting the type of 'obj' (line 245)
        obj_19504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'obj', False)
        # Getting the type of 'self' (line 245)
        self_19505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'self', False)
        # Obtaining the member '_arr' of a type (line 245)
        _arr_19506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), self_19505, '_arr')
        # Obtaining the member 'ndim' of a type (line 245)
        ndim_19507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), _arr_19506, 'ndim')
        # Applying the binary operator '*' (line 245)
        result_mul_19508 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 16), '*', obj_19504, ndim_19507)
        
        # Calling (args, kwargs) (line 245)
        _call_result_19513 = invoke(stypy.reporting.localization.Localization(__file__, 245, 16), result_mul_19508, *[shape_19511], **kwargs_19512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', _call_result_19513)
        
        # ################# End of 'shape_as(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shape_as' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_19514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shape_as'
        return stypy_return_type_19514


    @norecursion
    def strides_as(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'strides_as'
        module_type_store = module_type_store.open_function_context('strides_as', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.strides_as.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.strides_as.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.strides_as.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.strides_as.__dict__.__setitem__('stypy_function_name', '_ctypes.strides_as')
        _ctypes.strides_as.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        _ctypes.strides_as.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.strides_as.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.strides_as.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.strides_as.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.strides_as.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.strides_as.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.strides_as', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'strides_as', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'strides_as(...)' code ##################

        
        # Getting the type of 'self' (line 248)
        self_19515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'self')
        # Obtaining the member '_zerod' of a type (line 248)
        _zerod_19516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 11), self_19515, '_zerod')
        # Testing the type of an if condition (line 248)
        if_condition_19517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 8), _zerod_19516)
        # Assigning a type to the variable 'if_condition_19517' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'if_condition_19517', if_condition_19517)
        # SSA begins for if statement (line 248)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 249)
        None_19518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type', None_19518)
        # SSA join for if statement (line 248)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 250)
        # Getting the type of 'self' (line 250)
        self_19524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 37), 'self', False)
        # Obtaining the member '_arr' of a type (line 250)
        _arr_19525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 37), self_19524, '_arr')
        # Obtaining the member 'strides' of a type (line 250)
        strides_19526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 37), _arr_19525, 'strides')
        # Processing the call keyword arguments (line 250)
        kwargs_19527 = {}
        # Getting the type of 'obj' (line 250)
        obj_19519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'obj', False)
        # Getting the type of 'self' (line 250)
        self_19520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'self', False)
        # Obtaining the member '_arr' of a type (line 250)
        _arr_19521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 20), self_19520, '_arr')
        # Obtaining the member 'ndim' of a type (line 250)
        ndim_19522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 20), _arr_19521, 'ndim')
        # Applying the binary operator '*' (line 250)
        result_mul_19523 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 16), '*', obj_19519, ndim_19522)
        
        # Calling (args, kwargs) (line 250)
        _call_result_19528 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), result_mul_19523, *[strides_19526], **kwargs_19527)
        
        # Assigning a type to the variable 'stypy_return_type' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type', _call_result_19528)
        
        # ################# End of 'strides_as(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'strides_as' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_19529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19529)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'strides_as'
        return stypy_return_type_19529


    @norecursion
    def get_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_data'
        module_type_store = module_type_store.open_function_context('get_data', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_data.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_data.__dict__.__setitem__('stypy_function_name', '_ctypes.get_data')
        _ctypes.get_data.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_data(...)' code ##################

        # Getting the type of 'self' (line 253)
        self_19530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'self')
        # Obtaining the member '_data' of a type (line 253)
        _data_19531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 15), self_19530, '_data')
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type', _data_19531)
        
        # ################# End of 'get_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_data' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_19532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_data'
        return stypy_return_type_19532


    @norecursion
    def get_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_shape'
        module_type_store = module_type_store.open_function_context('get_shape', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_shape.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_shape.__dict__.__setitem__('stypy_function_name', '_ctypes.get_shape')
        _ctypes.get_shape.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_shape', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 256)
        self_19533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'self')
        # Obtaining the member '_zerod' of a type (line 256)
        _zerod_19534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), self_19533, '_zerod')
        # Testing the type of an if condition (line 256)
        if_condition_19535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 8), _zerod_19534)
        # Assigning a type to the variable 'if_condition_19535' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'if_condition_19535', if_condition_19535)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 257)
        None_19536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'stypy_return_type', None_19536)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 258)
        # Getting the type of 'self' (line 258)
        self_19544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 50), 'self', False)
        # Obtaining the member '_arr' of a type (line 258)
        _arr_19545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 50), self_19544, '_arr')
        # Obtaining the member 'shape' of a type (line 258)
        shape_19546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 50), _arr_19545, 'shape')
        # Processing the call keyword arguments (line 258)
        kwargs_19547 = {}
        
        # Call to _getintp_ctype(...): (line 258)
        # Processing the call keyword arguments (line 258)
        kwargs_19538 = {}
        # Getting the type of '_getintp_ctype' (line 258)
        _getintp_ctype_19537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), '_getintp_ctype', False)
        # Calling _getintp_ctype(args, kwargs) (line 258)
        _getintp_ctype_call_result_19539 = invoke(stypy.reporting.localization.Localization(__file__, 258, 16), _getintp_ctype_19537, *[], **kwargs_19538)
        
        # Getting the type of 'self' (line 258)
        self_19540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 'self', False)
        # Obtaining the member '_arr' of a type (line 258)
        _arr_19541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 33), self_19540, '_arr')
        # Obtaining the member 'ndim' of a type (line 258)
        ndim_19542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 33), _arr_19541, 'ndim')
        # Applying the binary operator '*' (line 258)
        result_mul_19543 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 16), '*', _getintp_ctype_call_result_19539, ndim_19542)
        
        # Calling (args, kwargs) (line 258)
        _call_result_19548 = invoke(stypy.reporting.localization.Localization(__file__, 258, 16), result_mul_19543, *[shape_19546], **kwargs_19547)
        
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', _call_result_19548)
        
        # ################# End of 'get_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_19549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19549)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_shape'
        return stypy_return_type_19549


    @norecursion
    def get_strides(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_strides'
        module_type_store = module_type_store.open_function_context('get_strides', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_strides.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_strides.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_strides.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_strides.__dict__.__setitem__('stypy_function_name', '_ctypes.get_strides')
        _ctypes.get_strides.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_strides.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_strides.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_strides.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_strides.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_strides.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_strides.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_strides', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_strides', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_strides(...)' code ##################

        
        # Getting the type of 'self' (line 261)
        self_19550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'self')
        # Obtaining the member '_zerod' of a type (line 261)
        _zerod_19551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 11), self_19550, '_zerod')
        # Testing the type of an if condition (line 261)
        if_condition_19552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), _zerod_19551)
        # Assigning a type to the variable 'if_condition_19552' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_19552', if_condition_19552)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 262)
        None_19553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'stypy_return_type', None_19553)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to (...): (line 263)
        # Getting the type of 'self' (line 263)
        self_19561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 50), 'self', False)
        # Obtaining the member '_arr' of a type (line 263)
        _arr_19562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 50), self_19561, '_arr')
        # Obtaining the member 'strides' of a type (line 263)
        strides_19563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 50), _arr_19562, 'strides')
        # Processing the call keyword arguments (line 263)
        kwargs_19564 = {}
        
        # Call to _getintp_ctype(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_19555 = {}
        # Getting the type of '_getintp_ctype' (line 263)
        _getintp_ctype_19554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), '_getintp_ctype', False)
        # Calling _getintp_ctype(args, kwargs) (line 263)
        _getintp_ctype_call_result_19556 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), _getintp_ctype_19554, *[], **kwargs_19555)
        
        # Getting the type of 'self' (line 263)
        self_19557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'self', False)
        # Obtaining the member '_arr' of a type (line 263)
        _arr_19558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 33), self_19557, '_arr')
        # Obtaining the member 'ndim' of a type (line 263)
        ndim_19559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 33), _arr_19558, 'ndim')
        # Applying the binary operator '*' (line 263)
        result_mul_19560 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 16), '*', _getintp_ctype_call_result_19556, ndim_19559)
        
        # Calling (args, kwargs) (line 263)
        _call_result_19565 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), result_mul_19560, *[strides_19563], **kwargs_19564)
        
        # Assigning a type to the variable 'stypy_return_type' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'stypy_return_type', _call_result_19565)
        
        # ################# End of 'get_strides(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_strides' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_19566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19566)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_strides'
        return stypy_return_type_19566


    @norecursion
    def get_as_parameter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_as_parameter'
        module_type_store = module_type_store.open_function_context('get_as_parameter', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_function_name', '_ctypes.get_as_parameter')
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_as_parameter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_as_parameter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_as_parameter(...)' code ##################

        
        # Call to c_void_p(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'self' (line 266)
        self_19570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'self', False)
        # Obtaining the member '_data' of a type (line 266)
        _data_19571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 37), self_19570, '_data')
        # Processing the call keyword arguments (line 266)
        kwargs_19572 = {}
        # Getting the type of 'self' (line 266)
        self_19567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'self', False)
        # Obtaining the member '_ctypes' of a type (line 266)
        _ctypes_19568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 15), self_19567, '_ctypes')
        # Obtaining the member 'c_void_p' of a type (line 266)
        c_void_p_19569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 15), _ctypes_19568, 'c_void_p')
        # Calling c_void_p(args, kwargs) (line 266)
        c_void_p_call_result_19573 = invoke(stypy.reporting.localization.Localization(__file__, 266, 15), c_void_p_19569, *[_data_19571], **kwargs_19572)
        
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type', c_void_p_call_result_19573)
        
        # ################# End of 'get_as_parameter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_as_parameter' in the type store
        # Getting the type of 'stypy_return_type' (line 265)
        stypy_return_type_19574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19574)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_as_parameter'
        return stypy_return_type_19574

    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 271):

# Assigning a type to the variable '_ctypes' (line 226)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), '_ctypes', _ctypes)

# Assigning a Call to a Name (line 268):

# Call to property(...): (line 268)
# Processing the call arguments (line 268)
# Getting the type of '_ctypes'
_ctypes_19576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_data' of a type
get_data_19577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19576, 'get_data')
# Getting the type of 'None' (line 268)
None_19578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'None', False)
# Processing the call keyword arguments (line 268)
str_19579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'str', 'c-types data')
keyword_19580 = str_19579
kwargs_19581 = {'doc': keyword_19580}
# Getting the type of 'property' (line 268)
property_19575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'property', False)
# Calling property(args, kwargs) (line 268)
property_call_result_19582 = invoke(stypy.reporting.localization.Localization(__file__, 268, 11), property_19575, *[get_data_19577, None_19578], **kwargs_19581)

# Getting the type of '_ctypes'
_ctypes_19583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'data' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19583, 'data', property_call_result_19582)

# Assigning a Call to a Name (line 269):

# Call to property(...): (line 269)
# Processing the call arguments (line 269)
# Getting the type of '_ctypes'
_ctypes_19585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_shape' of a type
get_shape_19586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19585, 'get_shape')
# Getting the type of 'None' (line 269)
None_19587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'None', False)
# Processing the call keyword arguments (line 269)
str_19588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 42), 'str', 'c-types shape')
keyword_19589 = str_19588
kwargs_19590 = {'doc': keyword_19589}
# Getting the type of 'property' (line 269)
property_19584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'property', False)
# Calling property(args, kwargs) (line 269)
property_call_result_19591 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), property_19584, *[get_shape_19586, None_19587], **kwargs_19590)

# Getting the type of '_ctypes'
_ctypes_19592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'shape' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19592, 'shape', property_call_result_19591)

# Assigning a Call to a Name (line 270):

# Call to property(...): (line 270)
# Processing the call arguments (line 270)
# Getting the type of '_ctypes'
_ctypes_19594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_strides' of a type
get_strides_19595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19594, 'get_strides')
# Getting the type of 'None' (line 270)
None_19596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 36), 'None', False)
# Processing the call keyword arguments (line 270)
str_19597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 46), 'str', 'c-types strides')
keyword_19598 = str_19597
kwargs_19599 = {'doc': keyword_19598}
# Getting the type of 'property' (line 270)
property_19593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 14), 'property', False)
# Calling property(args, kwargs) (line 270)
property_call_result_19600 = invoke(stypy.reporting.localization.Localization(__file__, 270, 14), property_19593, *[get_strides_19595, None_19596], **kwargs_19599)

# Getting the type of '_ctypes'
_ctypes_19601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'strides' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19601, 'strides', property_call_result_19600)

# Assigning a Call to a Name (line 271):

# Call to property(...): (line 271)
# Processing the call arguments (line 271)
# Getting the type of '_ctypes'
_ctypes_19603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_as_parameter' of a type
get_as_parameter_19604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19603, 'get_as_parameter')
# Getting the type of 'None' (line 271)
None_19605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 48), 'None', False)
# Processing the call keyword arguments (line 271)
str_19606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 58), 'str', '_as parameter_')
keyword_19607 = str_19606
kwargs_19608 = {'doc': keyword_19607}
# Getting the type of 'property' (line 271)
property_19602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'property', False)
# Calling property(args, kwargs) (line 271)
property_call_result_19609 = invoke(stypy.reporting.localization.Localization(__file__, 271, 21), property_19602, *[get_as_parameter_19604, None_19605], **kwargs_19608)

# Getting the type of '_ctypes'
_ctypes_19610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member '_as_parameter_' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_19610, '_as_parameter_', property_call_result_19609)

@norecursion
def _newnames(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_newnames'
    module_type_store = module_type_store.open_function_context('_newnames', 277, 0, False)
    
    # Passed parameters checking function
    _newnames.stypy_localization = localization
    _newnames.stypy_type_of_self = None
    _newnames.stypy_type_store = module_type_store
    _newnames.stypy_function_name = '_newnames'
    _newnames.stypy_param_names_list = ['datatype', 'order']
    _newnames.stypy_varargs_param_name = None
    _newnames.stypy_kwargs_param_name = None
    _newnames.stypy_call_defaults = defaults
    _newnames.stypy_call_varargs = varargs
    _newnames.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_newnames', ['datatype', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_newnames', localization, ['datatype', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_newnames(...)' code ##################

    
    # Assigning a Attribute to a Name (line 278):
    
    # Assigning a Attribute to a Name (line 278):
    # Getting the type of 'datatype' (line 278)
    datatype_19611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'datatype')
    # Obtaining the member 'names' of a type (line 278)
    names_19612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 15), datatype_19611, 'names')
    # Assigning a type to the variable 'oldnames' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'oldnames', names_19612)
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to list(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'oldnames' (line 279)
    oldnames_19614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 21), 'oldnames', False)
    # Processing the call keyword arguments (line 279)
    kwargs_19615 = {}
    # Getting the type of 'list' (line 279)
    list_19613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'list', False)
    # Calling list(args, kwargs) (line 279)
    list_call_result_19616 = invoke(stypy.reporting.localization.Localization(__file__, 279, 16), list_19613, *[oldnames_19614], **kwargs_19615)
    
    # Assigning a type to the variable 'nameslist' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'nameslist', list_call_result_19616)
    
    # Type idiom detected: calculating its left and rigth part (line 280)
    # Getting the type of 'str' (line 280)
    str_19617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'str')
    # Getting the type of 'order' (line 280)
    order_19618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 18), 'order')
    
    (may_be_19619, more_types_in_union_19620) = may_be_subtype(str_19617, order_19618)

    if may_be_19619:

        if more_types_in_union_19620:
            # Runtime conditional SSA (line 280)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'order' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'order', remove_not_subtype_from_union(order_19618, str))
        
        # Assigning a List to a Name (line 281):
        
        # Assigning a List to a Name (line 281):
        
        # Obtaining an instance of the builtin type 'list' (line 281)
        list_19621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 281)
        # Adding element type (line 281)
        # Getting the type of 'order' (line 281)
        order_19622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'order')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 16), list_19621, order_19622)
        
        # Assigning a type to the variable 'order' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'order', list_19621)

        if more_types_in_union_19620:
            # SSA join for if statement (line 280)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isinstance(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'order' (line 282)
    order_19624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'order', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 282)
    tuple_19625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 282)
    # Adding element type (line 282)
    # Getting the type of 'list' (line 282)
    list_19626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 26), tuple_19625, list_19626)
    # Adding element type (line 282)
    # Getting the type of 'tuple' (line 282)
    tuple_19627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 26), tuple_19625, tuple_19627)
    
    # Processing the call keyword arguments (line 282)
    kwargs_19628 = {}
    # Getting the type of 'isinstance' (line 282)
    isinstance_19623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 282)
    isinstance_call_result_19629 = invoke(stypy.reporting.localization.Localization(__file__, 282, 7), isinstance_19623, *[order_19624, tuple_19625], **kwargs_19628)
    
    # Testing the type of an if condition (line 282)
    if_condition_19630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 4), isinstance_call_result_19629)
    # Assigning a type to the variable 'if_condition_19630' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'if_condition_19630', if_condition_19630)
    # SSA begins for if statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'order' (line 283)
    order_19631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'order')
    # Testing the type of a for loop iterable (line 283)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 8), order_19631)
    # Getting the type of the for loop variable (line 283)
    for_loop_var_19632 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 8), order_19631)
    # Assigning a type to the variable 'name' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'name', for_loop_var_19632)
    # SSA begins for a for statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to remove(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'name' (line 285)
    name_19635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'name', False)
    # Processing the call keyword arguments (line 285)
    kwargs_19636 = {}
    # Getting the type of 'nameslist' (line 285)
    nameslist_19633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'nameslist', False)
    # Obtaining the member 'remove' of a type (line 285)
    remove_19634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 16), nameslist_19633, 'remove')
    # Calling remove(args, kwargs) (line 285)
    remove_call_result_19637 = invoke(stypy.reporting.localization.Localization(__file__, 285, 16), remove_19634, *[name_19635], **kwargs_19636)
    
    # SSA branch for the except part of a try statement (line 284)
    # SSA branch for the except 'ValueError' branch of a try statement (line 284)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 287)
    # Processing the call arguments (line 287)
    str_19639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 33), 'str', 'unknown field name: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 287)
    tuple_19640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 287)
    # Adding element type (line 287)
    # Getting the type of 'name' (line 287)
    name_19641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 61), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 61), tuple_19640, name_19641)
    
    # Applying the binary operator '%' (line 287)
    result_mod_19642 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 33), '%', str_19639, tuple_19640)
    
    # Processing the call keyword arguments (line 287)
    kwargs_19643 = {}
    # Getting the type of 'ValueError' (line 287)
    ValueError_19638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 287)
    ValueError_call_result_19644 = invoke(stypy.reporting.localization.Localization(__file__, 287, 22), ValueError_19638, *[result_mod_19642], **kwargs_19643)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 287, 16), ValueError_call_result_19644, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 284)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to tuple(...): (line 288)
    # Processing the call arguments (line 288)
    
    # Call to list(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'order' (line 288)
    order_19647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 26), 'order', False)
    # Processing the call keyword arguments (line 288)
    kwargs_19648 = {}
    # Getting the type of 'list' (line 288)
    list_19646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 'list', False)
    # Calling list(args, kwargs) (line 288)
    list_call_result_19649 = invoke(stypy.reporting.localization.Localization(__file__, 288, 21), list_19646, *[order_19647], **kwargs_19648)
    
    # Getting the type of 'nameslist' (line 288)
    nameslist_19650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 35), 'nameslist', False)
    # Applying the binary operator '+' (line 288)
    result_add_19651 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 21), '+', list_call_result_19649, nameslist_19650)
    
    # Processing the call keyword arguments (line 288)
    kwargs_19652 = {}
    # Getting the type of 'tuple' (line 288)
    tuple_19645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 288)
    tuple_call_result_19653 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), tuple_19645, *[result_add_19651], **kwargs_19652)
    
    # Assigning a type to the variable 'stypy_return_type' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', tuple_call_result_19653)
    # SSA join for if statement (line 282)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 289)
    # Processing the call arguments (line 289)
    str_19655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 21), 'str', 'unsupported order value: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 289)
    tuple_19656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 289)
    # Adding element type (line 289)
    # Getting the type of 'order' (line 289)
    order_19657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 54), 'order', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 54), tuple_19656, order_19657)
    
    # Applying the binary operator '%' (line 289)
    result_mod_19658 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 21), '%', str_19655, tuple_19656)
    
    # Processing the call keyword arguments (line 289)
    kwargs_19659 = {}
    # Getting the type of 'ValueError' (line 289)
    ValueError_19654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 289)
    ValueError_call_result_19660 = invoke(stypy.reporting.localization.Localization(__file__, 289, 10), ValueError_19654, *[result_mod_19658], **kwargs_19659)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 289, 4), ValueError_call_result_19660, 'raise parameter', BaseException)
    
    # ################# End of '_newnames(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_newnames' in the type store
    # Getting the type of 'stypy_return_type' (line 277)
    stypy_return_type_19661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19661)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_newnames'
    return stypy_return_type_19661

# Assigning a type to the variable '_newnames' (line 277)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 0), '_newnames', _newnames)

@norecursion
def _copy_fields(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_copy_fields'
    module_type_store = module_type_store.open_function_context('_copy_fields', 291, 0, False)
    
    # Passed parameters checking function
    _copy_fields.stypy_localization = localization
    _copy_fields.stypy_type_of_self = None
    _copy_fields.stypy_type_store = module_type_store
    _copy_fields.stypy_function_name = '_copy_fields'
    _copy_fields.stypy_param_names_list = ['ary']
    _copy_fields.stypy_varargs_param_name = None
    _copy_fields.stypy_kwargs_param_name = None
    _copy_fields.stypy_call_defaults = defaults
    _copy_fields.stypy_call_varargs = varargs
    _copy_fields.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_copy_fields', ['ary'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_copy_fields', localization, ['ary'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_copy_fields(...)' code ##################

    str_19662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, (-1)), 'str', 'Return copy of structured array with padding between fields removed.\n\n    Parameters\n    ----------\n    ary : ndarray\n       Structured array from which to remove padding bytes\n\n    Returns\n    -------\n    ary_copy : ndarray\n       Copy of ary with padding bytes removed\n    ')
    
    # Assigning a Attribute to a Name (line 304):
    
    # Assigning a Attribute to a Name (line 304):
    # Getting the type of 'ary' (line 304)
    ary_19663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 9), 'ary')
    # Obtaining the member 'dtype' of a type (line 304)
    dtype_19664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 9), ary_19663, 'dtype')
    # Assigning a type to the variable 'dt' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'dt', dtype_19664)
    
    # Assigning a Dict to a Name (line 305):
    
    # Assigning a Dict to a Name (line 305):
    
    # Obtaining an instance of the builtin type 'dict' (line 305)
    dict_19665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 305)
    # Adding element type (key, value) (line 305)
    str_19666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 18), 'str', 'names')
    # Getting the type of 'dt' (line 305)
    dt_19667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 27), 'dt')
    # Obtaining the member 'names' of a type (line 305)
    names_19668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 27), dt_19667, 'names')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 17), dict_19665, (str_19666, names_19668))
    # Adding element type (key, value) (line 305)
    str_19669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 18), 'str', 'formats')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'dt' (line 306)
    dt_19678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 61), 'dt')
    # Obtaining the member 'names' of a type (line 306)
    names_19679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 61), dt_19678, 'names')
    comprehension_19680 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 30), names_19679)
    # Assigning a type to the variable 'name' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'name', comprehension_19680)
    
    # Obtaining the type of the subscript
    int_19670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 46), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 306)
    name_19671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 40), 'name')
    # Getting the type of 'dt' (line 306)
    dt_19672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'dt')
    # Obtaining the member 'fields' of a type (line 306)
    fields_19673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 30), dt_19672, 'fields')
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___19674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 30), fields_19673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_19675 = invoke(stypy.reporting.localization.Localization(__file__, 306, 30), getitem___19674, name_19671)
    
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___19676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 30), subscript_call_result_19675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_19677 = invoke(stypy.reporting.localization.Localization(__file__, 306, 30), getitem___19676, int_19670)
    
    list_19681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 30), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 30), list_19681, subscript_call_result_19677)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 17), dict_19665, (str_19669, list_19681))
    
    # Assigning a type to the variable 'copy_dtype' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'copy_dtype', dict_19665)
    
    # Call to array(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'ary' (line 307)
    ary_19683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'ary', False)
    # Processing the call keyword arguments (line 307)
    # Getting the type of 'copy_dtype' (line 307)
    copy_dtype_19684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 28), 'copy_dtype', False)
    keyword_19685 = copy_dtype_19684
    # Getting the type of 'True' (line 307)
    True_19686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 45), 'True', False)
    keyword_19687 = True_19686
    kwargs_19688 = {'dtype': keyword_19685, 'copy': keyword_19687}
    # Getting the type of 'array' (line 307)
    array_19682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'array', False)
    # Calling array(args, kwargs) (line 307)
    array_call_result_19689 = invoke(stypy.reporting.localization.Localization(__file__, 307, 11), array_19682, *[ary_19683], **kwargs_19688)
    
    # Assigning a type to the variable 'stypy_return_type' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'stypy_return_type', array_call_result_19689)
    
    # ################# End of '_copy_fields(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_copy_fields' in the type store
    # Getting the type of 'stypy_return_type' (line 291)
    stypy_return_type_19690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19690)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_copy_fields'
    return stypy_return_type_19690

# Assigning a type to the variable '_copy_fields' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), '_copy_fields', _copy_fields)

@norecursion
def _getfield_is_safe(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_getfield_is_safe'
    module_type_store = module_type_store.open_function_context('_getfield_is_safe', 309, 0, False)
    
    # Passed parameters checking function
    _getfield_is_safe.stypy_localization = localization
    _getfield_is_safe.stypy_type_of_self = None
    _getfield_is_safe.stypy_type_store = module_type_store
    _getfield_is_safe.stypy_function_name = '_getfield_is_safe'
    _getfield_is_safe.stypy_param_names_list = ['oldtype', 'newtype', 'offset']
    _getfield_is_safe.stypy_varargs_param_name = None
    _getfield_is_safe.stypy_kwargs_param_name = None
    _getfield_is_safe.stypy_call_defaults = defaults
    _getfield_is_safe.stypy_call_varargs = varargs
    _getfield_is_safe.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getfield_is_safe', ['oldtype', 'newtype', 'offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getfield_is_safe', localization, ['oldtype', 'newtype', 'offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getfield_is_safe(...)' code ##################

    str_19691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', ' Checks safety of getfield for object arrays.\n\n    As in _view_is_safe, we need to check that memory containing objects is not\n    reinterpreted as a non-object datatype and vice versa.\n\n    Parameters\n    ----------\n    oldtype : data-type\n        Data type of the original ndarray.\n    newtype : data-type\n        Data type of the field being accessed by ndarray.getfield\n    offset : int\n        Offset of the field being accessed by ndarray.getfield\n\n    Raises\n    ------\n    TypeError\n        If the field access is invalid\n\n    ')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'newtype' (line 330)
    newtype_19692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 7), 'newtype')
    # Obtaining the member 'hasobject' of a type (line 330)
    hasobject_19693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 7), newtype_19692, 'hasobject')
    # Getting the type of 'oldtype' (line 330)
    oldtype_19694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 28), 'oldtype')
    # Obtaining the member 'hasobject' of a type (line 330)
    hasobject_19695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 28), oldtype_19694, 'hasobject')
    # Applying the binary operator 'or' (line 330)
    result_or_keyword_19696 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 7), 'or', hasobject_19693, hasobject_19695)
    
    # Testing the type of an if condition (line 330)
    if_condition_19697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 4), result_or_keyword_19696)
    # Assigning a type to the variable 'if_condition_19697' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'if_condition_19697', if_condition_19697)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'offset' (line 331)
    offset_19698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'offset')
    int_19699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 21), 'int')
    # Applying the binary operator '==' (line 331)
    result_eq_19700 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), '==', offset_19698, int_19699)
    
    
    # Getting the type of 'newtype' (line 331)
    newtype_19701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'newtype')
    # Getting the type of 'oldtype' (line 331)
    oldtype_19702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 38), 'oldtype')
    # Applying the binary operator '==' (line 331)
    result_eq_19703 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 27), '==', newtype_19701, oldtype_19702)
    
    # Applying the binary operator 'and' (line 331)
    result_and_keyword_19704 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), 'and', result_eq_19700, result_eq_19703)
    
    # Testing the type of an if condition (line 331)
    if_condition_19705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), result_and_keyword_19704)
    # Assigning a type to the variable 'if_condition_19705' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_19705', if_condition_19705)
    # SSA begins for if statement (line 331)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 331)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'oldtype' (line 333)
    oldtype_19706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'oldtype')
    # Obtaining the member 'names' of a type (line 333)
    names_19707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 11), oldtype_19706, 'names')
    # Testing the type of an if condition (line 333)
    if_condition_19708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), names_19707)
    # Assigning a type to the variable 'if_condition_19708' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_19708', if_condition_19708)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'oldtype' (line 334)
    oldtype_19709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'oldtype')
    # Obtaining the member 'names' of a type (line 334)
    names_19710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 24), oldtype_19709, 'names')
    # Testing the type of a for loop iterable (line 334)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 334, 12), names_19710)
    # Getting the type of the for loop variable (line 334)
    for_loop_var_19711 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 334, 12), names_19710)
    # Assigning a type to the variable 'name' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'name', for_loop_var_19711)
    # SSA begins for a for statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_19712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 41), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 335)
    name_19713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 35), 'name')
    # Getting the type of 'oldtype' (line 335)
    oldtype_19714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'oldtype')
    # Obtaining the member 'fields' of a type (line 335)
    fields_19715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 20), oldtype_19714, 'fields')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___19716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 20), fields_19715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_19717 = invoke(stypy.reporting.localization.Localization(__file__, 335, 20), getitem___19716, name_19713)
    
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___19718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 20), subscript_call_result_19717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_19719 = invoke(stypy.reporting.localization.Localization(__file__, 335, 20), getitem___19718, int_19712)
    
    # Getting the type of 'offset' (line 335)
    offset_19720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 47), 'offset')
    # Applying the binary operator '==' (line 335)
    result_eq_19721 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 20), '==', subscript_call_result_19719, offset_19720)
    
    
    
    # Obtaining the type of the subscript
    int_19722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 45), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 336)
    name_19723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 39), 'name')
    # Getting the type of 'oldtype' (line 336)
    oldtype_19724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'oldtype')
    # Obtaining the member 'fields' of a type (line 336)
    fields_19725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 24), oldtype_19724, 'fields')
    # Obtaining the member '__getitem__' of a type (line 336)
    getitem___19726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 24), fields_19725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 336)
    subscript_call_result_19727 = invoke(stypy.reporting.localization.Localization(__file__, 336, 24), getitem___19726, name_19723)
    
    # Obtaining the member '__getitem__' of a type (line 336)
    getitem___19728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 24), subscript_call_result_19727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 336)
    subscript_call_result_19729 = invoke(stypy.reporting.localization.Localization(__file__, 336, 24), getitem___19728, int_19722)
    
    # Getting the type of 'newtype' (line 336)
    newtype_19730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 51), 'newtype')
    # Applying the binary operator '==' (line 336)
    result_eq_19731 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 24), '==', subscript_call_result_19729, newtype_19730)
    
    # Applying the binary operator 'and' (line 335)
    result_and_keyword_19732 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 20), 'and', result_eq_19721, result_eq_19731)
    
    # Testing the type of an if condition (line 335)
    if_condition_19733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 16), result_and_keyword_19732)
    # Assigning a type to the variable 'if_condition_19733' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'if_condition_19733', if_condition_19733)
    # SSA begins for if statement (line 335)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 335)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to TypeError(...): (line 338)
    # Processing the call arguments (line 338)
    str_19735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 24), 'str', 'Cannot get/set field of an object array')
    # Processing the call keyword arguments (line 338)
    kwargs_19736 = {}
    # Getting the type of 'TypeError' (line 338)
    TypeError_19734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 338)
    TypeError_call_result_19737 = invoke(stypy.reporting.localization.Localization(__file__, 338, 14), TypeError_19734, *[str_19735], **kwargs_19736)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 338, 8), TypeError_call_result_19737, 'raise parameter', BaseException)
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of '_getfield_is_safe(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getfield_is_safe' in the type store
    # Getting the type of 'stypy_return_type' (line 309)
    stypy_return_type_19738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19738)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getfield_is_safe'
    return stypy_return_type_19738

# Assigning a type to the variable '_getfield_is_safe' (line 309)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 0), '_getfield_is_safe', _getfield_is_safe)

@norecursion
def _view_is_safe(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_view_is_safe'
    module_type_store = module_type_store.open_function_context('_view_is_safe', 341, 0, False)
    
    # Passed parameters checking function
    _view_is_safe.stypy_localization = localization
    _view_is_safe.stypy_type_of_self = None
    _view_is_safe.stypy_type_store = module_type_store
    _view_is_safe.stypy_function_name = '_view_is_safe'
    _view_is_safe.stypy_param_names_list = ['oldtype', 'newtype']
    _view_is_safe.stypy_varargs_param_name = None
    _view_is_safe.stypy_kwargs_param_name = None
    _view_is_safe.stypy_call_defaults = defaults
    _view_is_safe.stypy_call_varargs = varargs
    _view_is_safe.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_view_is_safe', ['oldtype', 'newtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_view_is_safe', localization, ['oldtype', 'newtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_view_is_safe(...)' code ##################

    str_19739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, (-1)), 'str', ' Checks safety of a view involving object arrays, for example when\n    doing::\n\n        np.zeros(10, dtype=oldtype).view(newtype)\n\n    Parameters\n    ----------\n    oldtype : data-type\n        Data type of original ndarray\n    newtype : data-type\n        Data type of the view\n\n    Raises\n    ------\n    TypeError\n        If the new type is incompatible with the old type.\n\n    ')
    
    
    # Getting the type of 'oldtype' (line 363)
    oldtype_19740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 7), 'oldtype')
    # Getting the type of 'newtype' (line 363)
    newtype_19741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'newtype')
    # Applying the binary operator '==' (line 363)
    result_eq_19742 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 7), '==', oldtype_19740, newtype_19741)
    
    # Testing the type of an if condition (line 363)
    if_condition_19743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 4), result_eq_19742)
    # Assigning a type to the variable 'if_condition_19743' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'if_condition_19743', if_condition_19743)
    # SSA begins for if statement (line 363)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 363)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'newtype' (line 366)
    newtype_19744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 7), 'newtype')
    # Obtaining the member 'hasobject' of a type (line 366)
    hasobject_19745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 7), newtype_19744, 'hasobject')
    # Getting the type of 'oldtype' (line 366)
    oldtype_19746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 28), 'oldtype')
    # Obtaining the member 'hasobject' of a type (line 366)
    hasobject_19747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 28), oldtype_19746, 'hasobject')
    # Applying the binary operator 'or' (line 366)
    result_or_keyword_19748 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 7), 'or', hasobject_19745, hasobject_19747)
    
    # Testing the type of an if condition (line 366)
    if_condition_19749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 4), result_or_keyword_19748)
    # Assigning a type to the variable 'if_condition_19749' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'if_condition_19749', if_condition_19749)
    # SSA begins for if statement (line 366)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 367)
    # Processing the call arguments (line 367)
    str_19751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 24), 'str', 'Cannot change data-type for object array.')
    # Processing the call keyword arguments (line 367)
    kwargs_19752 = {}
    # Getting the type of 'TypeError' (line 367)
    TypeError_19750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 367)
    TypeError_call_result_19753 = invoke(stypy.reporting.localization.Localization(__file__, 367, 14), TypeError_19750, *[str_19751], **kwargs_19752)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 367, 8), TypeError_call_result_19753, 'raise parameter', BaseException)
    # SSA join for if statement (line 366)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of '_view_is_safe(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_view_is_safe' in the type store
    # Getting the type of 'stypy_return_type' (line 341)
    stypy_return_type_19754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19754)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_view_is_safe'
    return stypy_return_type_19754

# Assigning a type to the variable '_view_is_safe' (line 341)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), '_view_is_safe', _view_is_safe)

# Assigning a Dict to a Name (line 373):

# Assigning a Dict to a Name (line 373):

# Obtaining an instance of the builtin type 'dict' (line 373)
dict_19755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 373)
# Adding element type (key, value) (line 373)
str_19756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 4), 'str', '?')
str_19757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 9), 'str', '?')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19756, str_19757))
# Adding element type (key, value) (line 373)
str_19758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'str', 'b')
str_19759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 9), 'str', 'b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19758, str_19759))
# Adding element type (key, value) (line 373)
str_19760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 4), 'str', 'B')
str_19761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 9), 'str', 'B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19760, str_19761))
# Adding element type (key, value) (line 373)
str_19762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 4), 'str', 'h')
str_19763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 9), 'str', 'h')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19762, str_19763))
# Adding element type (key, value) (line 373)
str_19764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 4), 'str', 'H')
str_19765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 9), 'str', 'H')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19764, str_19765))
# Adding element type (key, value) (line 373)
str_19766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 4), 'str', 'i')
str_19767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 9), 'str', 'i')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19766, str_19767))
# Adding element type (key, value) (line 373)
str_19768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 4), 'str', 'I')
str_19769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 9), 'str', 'I')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19768, str_19769))
# Adding element type (key, value) (line 373)
str_19770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 4), 'str', 'l')
str_19771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 9), 'str', 'l')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19770, str_19771))
# Adding element type (key, value) (line 373)
str_19772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 4), 'str', 'L')
str_19773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 9), 'str', 'L')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19772, str_19773))
# Adding element type (key, value) (line 373)
str_19774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 4), 'str', 'q')
str_19775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 9), 'str', 'q')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19774, str_19775))
# Adding element type (key, value) (line 373)
str_19776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 4), 'str', 'Q')
str_19777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 9), 'str', 'Q')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19776, str_19777))
# Adding element type (key, value) (line 373)
str_19778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 4), 'str', 'e')
str_19779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 9), 'str', 'e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19778, str_19779))
# Adding element type (key, value) (line 373)
str_19780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'str', 'f')
str_19781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 9), 'str', 'f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19780, str_19781))
# Adding element type (key, value) (line 373)
str_19782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 4), 'str', 'd')
str_19783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 9), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19782, str_19783))
# Adding element type (key, value) (line 373)
str_19784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 4), 'str', 'g')
str_19785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 9), 'str', 'g')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19784, str_19785))
# Adding element type (key, value) (line 373)
str_19786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 4), 'str', 'Zf')
str_19787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 10), 'str', 'F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19786, str_19787))
# Adding element type (key, value) (line 373)
str_19788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 4), 'str', 'Zd')
str_19789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 10), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19788, str_19789))
# Adding element type (key, value) (line 373)
str_19790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 4), 'str', 'Zg')
str_19791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 10), 'str', 'G')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19790, str_19791))
# Adding element type (key, value) (line 373)
str_19792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 4), 'str', 's')
str_19793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 9), 'str', 'S')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19792, str_19793))
# Adding element type (key, value) (line 373)
str_19794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 4), 'str', 'w')
str_19795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 9), 'str', 'U')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19794, str_19795))
# Adding element type (key, value) (line 373)
str_19796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 4), 'str', 'O')
str_19797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 9), 'str', 'O')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19796, str_19797))
# Adding element type (key, value) (line 373)
str_19798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 4), 'str', 'x')
str_19799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 9), 'str', 'V')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 22), dict_19755, (str_19798, str_19799))

# Assigning a type to the variable '_pep3118_native_map' (line 373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), '_pep3118_native_map', dict_19755)

# Assigning a Call to a Name (line 397):

# Assigning a Call to a Name (line 397):

# Call to join(...): (line 397)
# Processing the call arguments (line 397)

# Call to keys(...): (line 397)
# Processing the call keyword arguments (line 397)
kwargs_19804 = {}
# Getting the type of '_pep3118_native_map' (line 397)
_pep3118_native_map_19802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 36), '_pep3118_native_map', False)
# Obtaining the member 'keys' of a type (line 397)
keys_19803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 36), _pep3118_native_map_19802, 'keys')
# Calling keys(args, kwargs) (line 397)
keys_call_result_19805 = invoke(stypy.reporting.localization.Localization(__file__, 397, 36), keys_19803, *[], **kwargs_19804)

# Processing the call keyword arguments (line 397)
kwargs_19806 = {}
str_19800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 28), 'str', '')
# Obtaining the member 'join' of a type (line 397)
join_19801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 28), str_19800, 'join')
# Calling join(args, kwargs) (line 397)
join_call_result_19807 = invoke(stypy.reporting.localization.Localization(__file__, 397, 28), join_19801, *[keys_call_result_19805], **kwargs_19806)

# Assigning a type to the variable '_pep3118_native_typechars' (line 397)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 0), '_pep3118_native_typechars', join_call_result_19807)

# Assigning a Dict to a Name (line 399):

# Assigning a Dict to a Name (line 399):

# Obtaining an instance of the builtin type 'dict' (line 399)
dict_19808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 24), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 399)
# Adding element type (key, value) (line 399)
str_19809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 4), 'str', '?')
str_19810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 9), 'str', '?')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19809, str_19810))
# Adding element type (key, value) (line 399)
str_19811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 4), 'str', 'b')
str_19812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 9), 'str', 'b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19811, str_19812))
# Adding element type (key, value) (line 399)
str_19813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 4), 'str', 'B')
str_19814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 9), 'str', 'B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19813, str_19814))
# Adding element type (key, value) (line 399)
str_19815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 4), 'str', 'h')
str_19816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 9), 'str', 'i2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19815, str_19816))
# Adding element type (key, value) (line 399)
str_19817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'str', 'H')
str_19818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 9), 'str', 'u2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19817, str_19818))
# Adding element type (key, value) (line 399)
str_19819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 4), 'str', 'i')
str_19820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 9), 'str', 'i4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19819, str_19820))
# Adding element type (key, value) (line 399)
str_19821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'str', 'I')
str_19822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 9), 'str', 'u4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19821, str_19822))
# Adding element type (key, value) (line 399)
str_19823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 4), 'str', 'l')
str_19824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 9), 'str', 'i4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19823, str_19824))
# Adding element type (key, value) (line 399)
str_19825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 4), 'str', 'L')
str_19826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 9), 'str', 'u4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19825, str_19826))
# Adding element type (key, value) (line 399)
str_19827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 4), 'str', 'q')
str_19828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 9), 'str', 'i8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19827, str_19828))
# Adding element type (key, value) (line 399)
str_19829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 4), 'str', 'Q')
str_19830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 9), 'str', 'u8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19829, str_19830))
# Adding element type (key, value) (line 399)
str_19831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 4), 'str', 'e')
str_19832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 9), 'str', 'f2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19831, str_19832))
# Adding element type (key, value) (line 399)
str_19833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 4), 'str', 'f')
str_19834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 9), 'str', 'f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19833, str_19834))
# Adding element type (key, value) (line 399)
str_19835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 4), 'str', 'd')
str_19836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 9), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19835, str_19836))
# Adding element type (key, value) (line 399)
str_19837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 4), 'str', 'Zf')
str_19838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 10), 'str', 'F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19837, str_19838))
# Adding element type (key, value) (line 399)
str_19839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 4), 'str', 'Zd')
str_19840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 10), 'str', 'D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19839, str_19840))
# Adding element type (key, value) (line 399)
str_19841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 4), 'str', 's')
str_19842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 9), 'str', 'S')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19841, str_19842))
# Adding element type (key, value) (line 399)
str_19843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 4), 'str', 'w')
str_19844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 9), 'str', 'U')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19843, str_19844))
# Adding element type (key, value) (line 399)
str_19845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 4), 'str', 'O')
str_19846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 9), 'str', 'O')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19845, str_19846))
# Adding element type (key, value) (line 399)
str_19847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 4), 'str', 'x')
str_19848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 9), 'str', 'V')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 24), dict_19808, (str_19847, str_19848))

# Assigning a type to the variable '_pep3118_standard_map' (line 399)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), '_pep3118_standard_map', dict_19808)

# Assigning a Call to a Name (line 421):

# Assigning a Call to a Name (line 421):

# Call to join(...): (line 421)
# Processing the call arguments (line 421)

# Call to keys(...): (line 421)
# Processing the call keyword arguments (line 421)
kwargs_19853 = {}
# Getting the type of '_pep3118_standard_map' (line 421)
_pep3118_standard_map_19851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 38), '_pep3118_standard_map', False)
# Obtaining the member 'keys' of a type (line 421)
keys_19852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 38), _pep3118_standard_map_19851, 'keys')
# Calling keys(args, kwargs) (line 421)
keys_call_result_19854 = invoke(stypy.reporting.localization.Localization(__file__, 421, 38), keys_19852, *[], **kwargs_19853)

# Processing the call keyword arguments (line 421)
kwargs_19855 = {}
str_19849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 30), 'str', '')
# Obtaining the member 'join' of a type (line 421)
join_19850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 30), str_19849, 'join')
# Calling join(args, kwargs) (line 421)
join_call_result_19856 = invoke(stypy.reporting.localization.Localization(__file__, 421, 30), join_19850, *[keys_call_result_19854], **kwargs_19855)

# Assigning a type to the variable '_pep3118_standard_typechars' (line 421)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 0), '_pep3118_standard_typechars', join_call_result_19856)

@norecursion
def _dtype_from_pep3118(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_19857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 40), 'str', '@')
    # Getting the type of 'False' (line 423)
    False_19858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 57), 'False')
    defaults = [str_19857, False_19858]
    # Create a new context for function '_dtype_from_pep3118'
    module_type_store = module_type_store.open_function_context('_dtype_from_pep3118', 423, 0, False)
    
    # Passed parameters checking function
    _dtype_from_pep3118.stypy_localization = localization
    _dtype_from_pep3118.stypy_type_of_self = None
    _dtype_from_pep3118.stypy_type_store = module_type_store
    _dtype_from_pep3118.stypy_function_name = '_dtype_from_pep3118'
    _dtype_from_pep3118.stypy_param_names_list = ['spec', 'byteorder', 'is_subdtype']
    _dtype_from_pep3118.stypy_varargs_param_name = None
    _dtype_from_pep3118.stypy_kwargs_param_name = None
    _dtype_from_pep3118.stypy_call_defaults = defaults
    _dtype_from_pep3118.stypy_call_varargs = varargs
    _dtype_from_pep3118.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dtype_from_pep3118', ['spec', 'byteorder', 'is_subdtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dtype_from_pep3118', localization, ['spec', 'byteorder', 'is_subdtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dtype_from_pep3118(...)' code ##################

    
    # Assigning a Dict to a Name (line 424):
    
    # Assigning a Dict to a Name (line 424):
    
    # Obtaining an instance of the builtin type 'dict' (line 424)
    dict_19859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 424)
    
    # Assigning a type to the variable 'fields' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'fields', dict_19859)
    
    # Assigning a Num to a Name (line 425):
    
    # Assigning a Num to a Name (line 425):
    int_19860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 13), 'int')
    # Assigning a type to the variable 'offset' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'offset', int_19860)
    
    # Assigning a Name to a Name (line 426):
    
    # Assigning a Name to a Name (line 426):
    # Getting the type of 'False' (line 426)
    False_19861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 20), 'False')
    # Assigning a type to the variable 'explicit_name' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'explicit_name', False_19861)
    
    # Assigning a Name to a Name (line 427):
    
    # Assigning a Name to a Name (line 427):
    # Getting the type of 'False' (line 427)
    False_19862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 25), 'False')
    # Assigning a type to the variable 'this_explicit_name' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'this_explicit_name', False_19862)
    
    # Assigning a Num to a Name (line 428):
    
    # Assigning a Num to a Name (line 428):
    int_19863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 23), 'int')
    # Assigning a type to the variable 'common_alignment' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'common_alignment', int_19863)
    
    # Assigning a Name to a Name (line 429):
    
    # Assigning a Name to a Name (line 429):
    # Getting the type of 'False' (line 429)
    False_19864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 17), 'False')
    # Assigning a type to the variable 'is_padding' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'is_padding', False_19864)
    
    # Assigning a List to a Name (line 431):
    
    # Assigning a List to a Name (line 431):
    
    # Obtaining an instance of the builtin type 'list' (line 431)
    list_19865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 431)
    # Adding element type (line 431)
    int_19866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 23), list_19865, int_19866)
    
    # Assigning a type to the variable 'dummy_name_index' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'dummy_name_index', list_19865)

    @norecursion
    def next_dummy_name(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'next_dummy_name'
        module_type_store = module_type_store.open_function_context('next_dummy_name', 433, 4, False)
        
        # Passed parameters checking function
        next_dummy_name.stypy_localization = localization
        next_dummy_name.stypy_type_of_self = None
        next_dummy_name.stypy_type_store = module_type_store
        next_dummy_name.stypy_function_name = 'next_dummy_name'
        next_dummy_name.stypy_param_names_list = []
        next_dummy_name.stypy_varargs_param_name = None
        next_dummy_name.stypy_kwargs_param_name = None
        next_dummy_name.stypy_call_defaults = defaults
        next_dummy_name.stypy_call_varargs = varargs
        next_dummy_name.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'next_dummy_name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'next_dummy_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'next_dummy_name(...)' code ##################

        
        # Getting the type of 'dummy_name_index' (line 434)
        dummy_name_index_19867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'dummy_name_index')
        
        # Obtaining the type of the subscript
        int_19868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 25), 'int')
        # Getting the type of 'dummy_name_index' (line 434)
        dummy_name_index_19869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'dummy_name_index')
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___19870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), dummy_name_index_19869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_19871 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), getitem___19870, int_19868)
        
        int_19872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 31), 'int')
        # Applying the binary operator '+=' (line 434)
        result_iadd_19873 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 8), '+=', subscript_call_result_19871, int_19872)
        # Getting the type of 'dummy_name_index' (line 434)
        dummy_name_index_19874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'dummy_name_index')
        int_19875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 25), 'int')
        # Storing an element on a container (line 434)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 8), dummy_name_index_19874, (int_19875, result_iadd_19873))
        
        
        # ################# End of 'next_dummy_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'next_dummy_name' in the type store
        # Getting the type of 'stypy_return_type' (line 433)
        stypy_return_type_19876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'next_dummy_name'
        return stypy_return_type_19876

    # Assigning a type to the variable 'next_dummy_name' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'next_dummy_name', next_dummy_name)

    @norecursion
    def get_dummy_name(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_dummy_name'
        module_type_store = module_type_store.open_function_context('get_dummy_name', 436, 4, False)
        
        # Passed parameters checking function
        get_dummy_name.stypy_localization = localization
        get_dummy_name.stypy_type_of_self = None
        get_dummy_name.stypy_type_store = module_type_store
        get_dummy_name.stypy_function_name = 'get_dummy_name'
        get_dummy_name.stypy_param_names_list = []
        get_dummy_name.stypy_varargs_param_name = None
        get_dummy_name.stypy_kwargs_param_name = None
        get_dummy_name.stypy_call_defaults = defaults
        get_dummy_name.stypy_call_varargs = varargs
        get_dummy_name.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_dummy_name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_dummy_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_dummy_name(...)' code ##################

        
        # Getting the type of 'True' (line 437)
        True_19877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 14), 'True')
        # Testing the type of an if condition (line 437)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), True_19877)
        # SSA begins for while statement (line 437)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 438):
        
        # Assigning a BinOp to a Name (line 438):
        str_19878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 19), 'str', 'f%d')
        
        # Obtaining the type of the subscript
        int_19879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 44), 'int')
        # Getting the type of 'dummy_name_index' (line 438)
        dummy_name_index_19880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'dummy_name_index')
        # Obtaining the member '__getitem__' of a type (line 438)
        getitem___19881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 27), dummy_name_index_19880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 438)
        subscript_call_result_19882 = invoke(stypy.reporting.localization.Localization(__file__, 438, 27), getitem___19881, int_19879)
        
        # Applying the binary operator '%' (line 438)
        result_mod_19883 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 19), '%', str_19878, subscript_call_result_19882)
        
        # Assigning a type to the variable 'name' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'name', result_mod_19883)
        
        
        # Getting the type of 'name' (line 439)
        name_19884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 15), 'name')
        # Getting the type of 'fields' (line 439)
        fields_19885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 27), 'fields')
        # Applying the binary operator 'notin' (line 439)
        result_contains_19886 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 15), 'notin', name_19884, fields_19885)
        
        # Testing the type of an if condition (line 439)
        if_condition_19887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 12), result_contains_19886)
        # Assigning a type to the variable 'if_condition_19887' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'if_condition_19887', if_condition_19887)
        # SSA begins for if statement (line 439)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'name' (line 440)
        name_19888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 16), 'stypy_return_type', name_19888)
        # SSA join for if statement (line 439)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to next_dummy_name(...): (line 441)
        # Processing the call keyword arguments (line 441)
        kwargs_19890 = {}
        # Getting the type of 'next_dummy_name' (line 441)
        next_dummy_name_19889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'next_dummy_name', False)
        # Calling next_dummy_name(args, kwargs) (line 441)
        next_dummy_name_call_result_19891 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), next_dummy_name_19889, *[], **kwargs_19890)
        
        # SSA join for while statement (line 437)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_dummy_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_dummy_name' in the type store
        # Getting the type of 'stypy_return_type' (line 436)
        stypy_return_type_19892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19892)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_dummy_name'
        return stypy_return_type_19892

    # Assigning a type to the variable 'get_dummy_name' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'get_dummy_name', get_dummy_name)
    
    # Getting the type of 'spec' (line 444)
    spec_19893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 10), 'spec')
    # Testing the type of an if condition (line 444)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 4), spec_19893)
    # SSA begins for while statement (line 444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Name to a Name (line 445):
    
    # Assigning a Name to a Name (line 445):
    # Getting the type of 'None' (line 445)
    None_19894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'None')
    # Assigning a type to the variable 'value' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'value', None_19894)
    
    
    
    # Obtaining the type of the subscript
    int_19895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 16), 'int')
    # Getting the type of 'spec' (line 448)
    spec_19896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'spec')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___19897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 11), spec_19896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_19898 = invoke(stypy.reporting.localization.Localization(__file__, 448, 11), getitem___19897, int_19895)
    
    str_19899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 22), 'str', '}')
    # Applying the binary operator '==' (line 448)
    result_eq_19900 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 11), '==', subscript_call_result_19898, str_19899)
    
    # Testing the type of an if condition (line 448)
    if_condition_19901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 8), result_eq_19900)
    # Assigning a type to the variable 'if_condition_19901' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'if_condition_19901', if_condition_19901)
    # SSA begins for if statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 449):
    
    # Assigning a Subscript to a Name (line 449):
    
    # Obtaining the type of the subscript
    int_19902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 24), 'int')
    slice_19903 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 19), int_19902, None, None)
    # Getting the type of 'spec' (line 449)
    spec_19904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 19), 'spec')
    # Obtaining the member '__getitem__' of a type (line 449)
    getitem___19905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 19), spec_19904, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 449)
    subscript_call_result_19906 = invoke(stypy.reporting.localization.Localization(__file__, 449, 19), getitem___19905, slice_19903)
    
    # Assigning a type to the variable 'spec' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'spec', subscript_call_result_19906)
    # SSA join for if statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 453):
    
    # Assigning a Name to a Name (line 453):
    # Getting the type of 'None' (line 453)
    None_19907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'None')
    # Assigning a type to the variable 'shape' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'shape', None_19907)
    
    
    
    # Obtaining the type of the subscript
    int_19908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 16), 'int')
    # Getting the type of 'spec' (line 454)
    spec_19909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'spec')
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___19910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 11), spec_19909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_19911 = invoke(stypy.reporting.localization.Localization(__file__, 454, 11), getitem___19910, int_19908)
    
    str_19912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 22), 'str', '(')
    # Applying the binary operator '==' (line 454)
    result_eq_19913 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 11), '==', subscript_call_result_19911, str_19912)
    
    # Testing the type of an if condition (line 454)
    if_condition_19914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 8), result_eq_19913)
    # Assigning a type to the variable 'if_condition_19914' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'if_condition_19914', if_condition_19914)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 455):
    
    # Assigning a Call to a Name (line 455):
    
    # Call to index(...): (line 455)
    # Processing the call arguments (line 455)
    str_19917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 27), 'str', ')')
    # Processing the call keyword arguments (line 455)
    kwargs_19918 = {}
    # Getting the type of 'spec' (line 455)
    spec_19915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'spec', False)
    # Obtaining the member 'index' of a type (line 455)
    index_19916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 16), spec_19915, 'index')
    # Calling index(args, kwargs) (line 455)
    index_call_result_19919 = invoke(stypy.reporting.localization.Localization(__file__, 455, 16), index_19916, *[str_19917], **kwargs_19918)
    
    # Assigning a type to the variable 'j' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'j', index_call_result_19919)
    
    # Assigning a Call to a Name (line 456):
    
    # Assigning a Call to a Name (line 456):
    
    # Call to tuple(...): (line 456)
    # Processing the call arguments (line 456)
    
    # Call to map(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'int' (line 456)
    int_19922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'int', False)
    
    # Call to split(...): (line 456)
    # Processing the call arguments (line 456)
    str_19930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 51), 'str', ',')
    # Processing the call keyword arguments (line 456)
    kwargs_19931 = {}
    
    # Obtaining the type of the subscript
    int_19923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 40), 'int')
    # Getting the type of 'j' (line 456)
    j_19924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'j', False)
    slice_19925 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 35), int_19923, j_19924, None)
    # Getting the type of 'spec' (line 456)
    spec_19926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'spec', False)
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___19927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 35), spec_19926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_19928 = invoke(stypy.reporting.localization.Localization(__file__, 456, 35), getitem___19927, slice_19925)
    
    # Obtaining the member 'split' of a type (line 456)
    split_19929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 35), subscript_call_result_19928, 'split')
    # Calling split(args, kwargs) (line 456)
    split_call_result_19932 = invoke(stypy.reporting.localization.Localization(__file__, 456, 35), split_19929, *[str_19930], **kwargs_19931)
    
    # Processing the call keyword arguments (line 456)
    kwargs_19933 = {}
    # Getting the type of 'map' (line 456)
    map_19921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 26), 'map', False)
    # Calling map(args, kwargs) (line 456)
    map_call_result_19934 = invoke(stypy.reporting.localization.Localization(__file__, 456, 26), map_19921, *[int_19922, split_call_result_19932], **kwargs_19933)
    
    # Processing the call keyword arguments (line 456)
    kwargs_19935 = {}
    # Getting the type of 'tuple' (line 456)
    tuple_19920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 456)
    tuple_call_result_19936 = invoke(stypy.reporting.localization.Localization(__file__, 456, 20), tuple_19920, *[map_call_result_19934], **kwargs_19935)
    
    # Assigning a type to the variable 'shape' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'shape', tuple_call_result_19936)
    
    # Assigning a Subscript to a Name (line 457):
    
    # Assigning a Subscript to a Name (line 457):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 457)
    j_19937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 24), 'j')
    int_19938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 26), 'int')
    # Applying the binary operator '+' (line 457)
    result_add_19939 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 24), '+', j_19937, int_19938)
    
    slice_19940 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 457, 19), result_add_19939, None, None)
    # Getting the type of 'spec' (line 457)
    spec_19941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'spec')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___19942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 19), spec_19941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_19943 = invoke(stypy.reporting.localization.Localization(__file__, 457, 19), getitem___19942, slice_19940)
    
    # Assigning a type to the variable 'spec' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'spec', subscript_call_result_19943)
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_19944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 16), 'int')
    # Getting the type of 'spec' (line 460)
    spec_19945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'spec')
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___19946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 11), spec_19945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_19947 = invoke(stypy.reporting.localization.Localization(__file__, 460, 11), getitem___19946, int_19944)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 460)
    tuple_19948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 460)
    # Adding element type (line 460)
    str_19949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'str', '@')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_19948, str_19949)
    # Adding element type (line 460)
    str_19950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 28), 'str', '=')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_19948, str_19950)
    # Adding element type (line 460)
    str_19951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 33), 'str', '<')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_19948, str_19951)
    # Adding element type (line 460)
    str_19952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 38), 'str', '>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_19948, str_19952)
    # Adding element type (line 460)
    str_19953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 43), 'str', '^')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_19948, str_19953)
    # Adding element type (line 460)
    str_19954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 48), 'str', '!')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_19948, str_19954)
    
    # Applying the binary operator 'in' (line 460)
    result_contains_19955 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 11), 'in', subscript_call_result_19947, tuple_19948)
    
    # Testing the type of an if condition (line 460)
    if_condition_19956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), result_contains_19955)
    # Assigning a type to the variable 'if_condition_19956' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'if_condition_19956', if_condition_19956)
    # SSA begins for if statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 461):
    
    # Assigning a Subscript to a Name (line 461):
    
    # Obtaining the type of the subscript
    int_19957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 29), 'int')
    # Getting the type of 'spec' (line 461)
    spec_19958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'spec')
    # Obtaining the member '__getitem__' of a type (line 461)
    getitem___19959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 24), spec_19958, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 461)
    subscript_call_result_19960 = invoke(stypy.reporting.localization.Localization(__file__, 461, 24), getitem___19959, int_19957)
    
    # Assigning a type to the variable 'byteorder' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'byteorder', subscript_call_result_19960)
    
    
    # Getting the type of 'byteorder' (line 462)
    byteorder_19961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'byteorder')
    str_19962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 28), 'str', '!')
    # Applying the binary operator '==' (line 462)
    result_eq_19963 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 15), '==', byteorder_19961, str_19962)
    
    # Testing the type of an if condition (line 462)
    if_condition_19964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 12), result_eq_19963)
    # Assigning a type to the variable 'if_condition_19964' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'if_condition_19964', if_condition_19964)
    # SSA begins for if statement (line 462)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 463):
    
    # Assigning a Str to a Name (line 463):
    str_19965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 28), 'str', '>')
    # Assigning a type to the variable 'byteorder' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'byteorder', str_19965)
    # SSA join for if statement (line 462)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 464):
    
    # Assigning a Subscript to a Name (line 464):
    
    # Obtaining the type of the subscript
    int_19966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 24), 'int')
    slice_19967 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 464, 19), int_19966, None, None)
    # Getting the type of 'spec' (line 464)
    spec_19968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 19), 'spec')
    # Obtaining the member '__getitem__' of a type (line 464)
    getitem___19969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 19), spec_19968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 464)
    subscript_call_result_19970 = invoke(stypy.reporting.localization.Localization(__file__, 464, 19), getitem___19969, slice_19967)
    
    # Assigning a type to the variable 'spec' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'spec', subscript_call_result_19970)
    # SSA join for if statement (line 460)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'byteorder' (line 467)
    byteorder_19971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'byteorder')
    
    # Obtaining an instance of the builtin type 'tuple' (line 467)
    tuple_19972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 467)
    # Adding element type (line 467)
    str_19973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 25), 'str', '@')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 25), tuple_19972, str_19973)
    # Adding element type (line 467)
    str_19974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 30), 'str', '^')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 25), tuple_19972, str_19974)
    
    # Applying the binary operator 'in' (line 467)
    result_contains_19975 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), 'in', byteorder_19971, tuple_19972)
    
    # Testing the type of an if condition (line 467)
    if_condition_19976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 8), result_contains_19975)
    # Assigning a type to the variable 'if_condition_19976' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'if_condition_19976', if_condition_19976)
    # SSA begins for if statement (line 467)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 468):
    
    # Assigning a Name to a Name (line 468):
    # Getting the type of '_pep3118_native_map' (line 468)
    _pep3118_native_map_19977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 23), '_pep3118_native_map')
    # Assigning a type to the variable 'type_map' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'type_map', _pep3118_native_map_19977)
    
    # Assigning a Name to a Name (line 469):
    
    # Assigning a Name to a Name (line 469):
    # Getting the type of '_pep3118_native_typechars' (line 469)
    _pep3118_native_typechars_19978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 29), '_pep3118_native_typechars')
    # Assigning a type to the variable 'type_map_chars' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'type_map_chars', _pep3118_native_typechars_19978)
    # SSA branch for the else part of an if statement (line 467)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 471):
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of '_pep3118_standard_map' (line 471)
    _pep3118_standard_map_19979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 23), '_pep3118_standard_map')
    # Assigning a type to the variable 'type_map' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'type_map', _pep3118_standard_map_19979)
    
    # Assigning a Name to a Name (line 472):
    
    # Assigning a Name to a Name (line 472):
    # Getting the type of '_pep3118_standard_typechars' (line 472)
    _pep3118_standard_typechars_19980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 29), '_pep3118_standard_typechars')
    # Assigning a type to the variable 'type_map_chars' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'type_map_chars', _pep3118_standard_typechars_19980)
    # SSA join for if statement (line 467)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 475):
    
    # Assigning a Num to a Name (line 475):
    int_19981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 19), 'int')
    # Assigning a type to the variable 'itemsize' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'itemsize', int_19981)
    
    
    # Call to isdigit(...): (line 476)
    # Processing the call keyword arguments (line 476)
    kwargs_19987 = {}
    
    # Obtaining the type of the subscript
    int_19982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 16), 'int')
    # Getting the type of 'spec' (line 476)
    spec_19983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 11), 'spec', False)
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___19984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 11), spec_19983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_19985 = invoke(stypy.reporting.localization.Localization(__file__, 476, 11), getitem___19984, int_19982)
    
    # Obtaining the member 'isdigit' of a type (line 476)
    isdigit_19986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 11), subscript_call_result_19985, 'isdigit')
    # Calling isdigit(args, kwargs) (line 476)
    isdigit_call_result_19988 = invoke(stypy.reporting.localization.Localization(__file__, 476, 11), isdigit_19986, *[], **kwargs_19987)
    
    # Testing the type of an if condition (line 476)
    if_condition_19989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 8), isdigit_call_result_19988)
    # Assigning a type to the variable 'if_condition_19989' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'if_condition_19989', if_condition_19989)
    # SSA begins for if statement (line 476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 477):
    
    # Assigning a Num to a Name (line 477):
    int_19990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 16), 'int')
    # Assigning a type to the variable 'j' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'j', int_19990)
    
    
    # Call to range(...): (line 478)
    # Processing the call arguments (line 478)
    int_19992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 27), 'int')
    
    # Call to len(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'spec' (line 478)
    spec_19994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 34), 'spec', False)
    # Processing the call keyword arguments (line 478)
    kwargs_19995 = {}
    # Getting the type of 'len' (line 478)
    len_19993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 30), 'len', False)
    # Calling len(args, kwargs) (line 478)
    len_call_result_19996 = invoke(stypy.reporting.localization.Localization(__file__, 478, 30), len_19993, *[spec_19994], **kwargs_19995)
    
    # Processing the call keyword arguments (line 478)
    kwargs_19997 = {}
    # Getting the type of 'range' (line 478)
    range_19991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'range', False)
    # Calling range(args, kwargs) (line 478)
    range_call_result_19998 = invoke(stypy.reporting.localization.Localization(__file__, 478, 21), range_19991, *[int_19992, len_call_result_19996], **kwargs_19997)
    
    # Testing the type of a for loop iterable (line 478)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 478, 12), range_call_result_19998)
    # Getting the type of the for loop variable (line 478)
    for_loop_var_19999 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 478, 12), range_call_result_19998)
    # Assigning a type to the variable 'j' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'j', for_loop_var_19999)
    # SSA begins for a for statement (line 478)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to isdigit(...): (line 479)
    # Processing the call keyword arguments (line 479)
    kwargs_20005 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 479)
    j_20000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'j', False)
    # Getting the type of 'spec' (line 479)
    spec_20001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 'spec', False)
    # Obtaining the member '__getitem__' of a type (line 479)
    getitem___20002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), spec_20001, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 479)
    subscript_call_result_20003 = invoke(stypy.reporting.localization.Localization(__file__, 479, 23), getitem___20002, j_20000)
    
    # Obtaining the member 'isdigit' of a type (line 479)
    isdigit_20004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), subscript_call_result_20003, 'isdigit')
    # Calling isdigit(args, kwargs) (line 479)
    isdigit_call_result_20006 = invoke(stypy.reporting.localization.Localization(__file__, 479, 23), isdigit_20004, *[], **kwargs_20005)
    
    # Applying the 'not' unary operator (line 479)
    result_not__20007 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 19), 'not', isdigit_call_result_20006)
    
    # Testing the type of an if condition (line 479)
    if_condition_20008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 16), result_not__20007)
    # Assigning a type to the variable 'if_condition_20008' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'if_condition_20008', if_condition_20008)
    # SSA begins for if statement (line 479)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 479)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 481):
    
    # Assigning a Call to a Name (line 481):
    
    # Call to int(...): (line 481)
    # Processing the call arguments (line 481)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 481)
    j_20010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 33), 'j', False)
    slice_20011 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 481, 27), None, j_20010, None)
    # Getting the type of 'spec' (line 481)
    spec_20012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 27), 'spec', False)
    # Obtaining the member '__getitem__' of a type (line 481)
    getitem___20013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 27), spec_20012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 481)
    subscript_call_result_20014 = invoke(stypy.reporting.localization.Localization(__file__, 481, 27), getitem___20013, slice_20011)
    
    # Processing the call keyword arguments (line 481)
    kwargs_20015 = {}
    # Getting the type of 'int' (line 481)
    int_20009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 23), 'int', False)
    # Calling int(args, kwargs) (line 481)
    int_call_result_20016 = invoke(stypy.reporting.localization.Localization(__file__, 481, 23), int_20009, *[subscript_call_result_20014], **kwargs_20015)
    
    # Assigning a type to the variable 'itemsize' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'itemsize', int_call_result_20016)
    
    # Assigning a Subscript to a Name (line 482):
    
    # Assigning a Subscript to a Name (line 482):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 482)
    j_20017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 24), 'j')
    slice_20018 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 482, 19), j_20017, None, None)
    # Getting the type of 'spec' (line 482)
    spec_20019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'spec')
    # Obtaining the member '__getitem__' of a type (line 482)
    getitem___20020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 19), spec_20019, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 482)
    subscript_call_result_20021 = invoke(stypy.reporting.localization.Localization(__file__, 482, 19), getitem___20020, slice_20018)
    
    # Assigning a type to the variable 'spec' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'spec', subscript_call_result_20021)
    # SSA join for if statement (line 476)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 485):
    
    # Assigning a Name to a Name (line 485):
    # Getting the type of 'False' (line 485)
    False_20022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 21), 'False')
    # Assigning a type to the variable 'is_padding' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'is_padding', False_20022)
    
    
    
    # Obtaining the type of the subscript
    int_20023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 17), 'int')
    slice_20024 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 487, 11), None, int_20023, None)
    # Getting the type of 'spec' (line 487)
    spec_20025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 11), 'spec')
    # Obtaining the member '__getitem__' of a type (line 487)
    getitem___20026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 11), spec_20025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 487)
    subscript_call_result_20027 = invoke(stypy.reporting.localization.Localization(__file__, 487, 11), getitem___20026, slice_20024)
    
    str_20028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 23), 'str', 'T{')
    # Applying the binary operator '==' (line 487)
    result_eq_20029 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 11), '==', subscript_call_result_20027, str_20028)
    
    # Testing the type of an if condition (line 487)
    if_condition_20030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 8), result_eq_20029)
    # Assigning a type to the variable 'if_condition_20030' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'if_condition_20030', if_condition_20030)
    # SSA begins for if statement (line 487)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 488):
    
    # Assigning a Call to a Name:
    
    # Call to _dtype_from_pep3118(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Obtaining the type of the subscript
    int_20032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 21), 'int')
    slice_20033 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 489, 16), int_20032, None, None)
    # Getting the type of 'spec' (line 489)
    spec_20034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 16), 'spec', False)
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___20035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 16), spec_20034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 489)
    subscript_call_result_20036 = invoke(stypy.reporting.localization.Localization(__file__, 489, 16), getitem___20035, slice_20033)
    
    # Processing the call keyword arguments (line 488)
    # Getting the type of 'byteorder' (line 489)
    byteorder_20037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 36), 'byteorder', False)
    keyword_20038 = byteorder_20037
    # Getting the type of 'True' (line 489)
    True_20039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 59), 'True', False)
    keyword_20040 = True_20039
    kwargs_20041 = {'is_subdtype': keyword_20040, 'byteorder': keyword_20038}
    # Getting the type of '_dtype_from_pep3118' (line 488)
    _dtype_from_pep3118_20031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 49), '_dtype_from_pep3118', False)
    # Calling _dtype_from_pep3118(args, kwargs) (line 488)
    _dtype_from_pep3118_call_result_20042 = invoke(stypy.reporting.localization.Localization(__file__, 488, 49), _dtype_from_pep3118_20031, *[subscript_call_result_20036], **kwargs_20041)
    
    # Assigning a type to the variable 'call_assignment_18763' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18763', _dtype_from_pep3118_call_result_20042)
    
    # Assigning a Call to a Name (line 488):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_20045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 12), 'int')
    # Processing the call keyword arguments
    kwargs_20046 = {}
    # Getting the type of 'call_assignment_18763' (line 488)
    call_assignment_18763_20043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18763', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___20044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), call_assignment_18763_20043, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_20047 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20044, *[int_20045], **kwargs_20046)
    
    # Assigning a type to the variable 'call_assignment_18764' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18764', getitem___call_result_20047)
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'call_assignment_18764' (line 488)
    call_assignment_18764_20048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18764')
    # Assigning a type to the variable 'value' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'value', call_assignment_18764_20048)
    
    # Assigning a Call to a Name (line 488):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_20051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 12), 'int')
    # Processing the call keyword arguments
    kwargs_20052 = {}
    # Getting the type of 'call_assignment_18763' (line 488)
    call_assignment_18763_20049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18763', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___20050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), call_assignment_18763_20049, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_20053 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20050, *[int_20051], **kwargs_20052)
    
    # Assigning a type to the variable 'call_assignment_18765' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18765', getitem___call_result_20053)
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'call_assignment_18765' (line 488)
    call_assignment_18765_20054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18765')
    # Assigning a type to the variable 'spec' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 'spec', call_assignment_18765_20054)
    
    # Assigning a Call to a Name (line 488):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_20057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 12), 'int')
    # Processing the call keyword arguments
    kwargs_20058 = {}
    # Getting the type of 'call_assignment_18763' (line 488)
    call_assignment_18763_20055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18763', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___20056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), call_assignment_18763_20055, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_20059 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20056, *[int_20057], **kwargs_20058)
    
    # Assigning a type to the variable 'call_assignment_18766' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18766', getitem___call_result_20059)
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'call_assignment_18766' (line 488)
    call_assignment_18766_20060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18766')
    # Assigning a type to the variable 'align' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 25), 'align', call_assignment_18766_20060)
    
    # Assigning a Call to a Name (line 488):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_20063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 12), 'int')
    # Processing the call keyword arguments
    kwargs_20064 = {}
    # Getting the type of 'call_assignment_18763' (line 488)
    call_assignment_18763_20061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18763', False)
    # Obtaining the member '__getitem__' of a type (line 488)
    getitem___20062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), call_assignment_18763_20061, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_20065 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20062, *[int_20063], **kwargs_20064)
    
    # Assigning a type to the variable 'call_assignment_18767' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18767', getitem___call_result_20065)
    
    # Assigning a Name to a Name (line 488):
    # Getting the type of 'call_assignment_18767' (line 488)
    call_assignment_18767_20066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'call_assignment_18767')
    # Assigning a type to the variable 'next_byteorder' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 32), 'next_byteorder', call_assignment_18767_20066)
    # SSA branch for the else part of an if statement (line 487)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_20067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 18), 'int')
    # Getting the type of 'spec' (line 490)
    spec_20068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 13), 'spec')
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___20069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 13), spec_20068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 490)
    subscript_call_result_20070 = invoke(stypy.reporting.localization.Localization(__file__, 490, 13), getitem___20069, int_20067)
    
    # Getting the type of 'type_map_chars' (line 490)
    type_map_chars_20071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 24), 'type_map_chars')
    # Applying the binary operator 'in' (line 490)
    result_contains_20072 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 13), 'in', subscript_call_result_20070, type_map_chars_20071)
    
    # Testing the type of an if condition (line 490)
    if_condition_20073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 13), result_contains_20072)
    # Assigning a type to the variable 'if_condition_20073' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 13), 'if_condition_20073', if_condition_20073)
    # SSA begins for if statement (line 490)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 491):
    
    # Assigning a Name to a Name (line 491):
    # Getting the type of 'byteorder' (line 491)
    byteorder_20074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 29), 'byteorder')
    # Assigning a type to the variable 'next_byteorder' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'next_byteorder', byteorder_20074)
    
    
    
    # Obtaining the type of the subscript
    int_20075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 20), 'int')
    # Getting the type of 'spec' (line 492)
    spec_20076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 15), 'spec')
    # Obtaining the member '__getitem__' of a type (line 492)
    getitem___20077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 15), spec_20076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 492)
    subscript_call_result_20078 = invoke(stypy.reporting.localization.Localization(__file__, 492, 15), getitem___20077, int_20075)
    
    str_20079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 26), 'str', 'Z')
    # Applying the binary operator '==' (line 492)
    result_eq_20080 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 15), '==', subscript_call_result_20078, str_20079)
    
    # Testing the type of an if condition (line 492)
    if_condition_20081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 492, 12), result_eq_20080)
    # Assigning a type to the variable 'if_condition_20081' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'if_condition_20081', if_condition_20081)
    # SSA begins for if statement (line 492)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 493):
    
    # Assigning a Num to a Name (line 493):
    int_20082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 20), 'int')
    # Assigning a type to the variable 'j' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 16), 'j', int_20082)
    # SSA branch for the else part of an if statement (line 492)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 495):
    
    # Assigning a Num to a Name (line 495):
    int_20083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 20), 'int')
    # Assigning a type to the variable 'j' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'j', int_20083)
    # SSA join for if statement (line 492)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 496):
    
    # Assigning a Subscript to a Name (line 496):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 496)
    j_20084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 29), 'j')
    slice_20085 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 23), None, j_20084, None)
    # Getting the type of 'spec' (line 496)
    spec_20086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 23), 'spec')
    # Obtaining the member '__getitem__' of a type (line 496)
    getitem___20087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 23), spec_20086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 496)
    subscript_call_result_20088 = invoke(stypy.reporting.localization.Localization(__file__, 496, 23), getitem___20087, slice_20085)
    
    # Assigning a type to the variable 'typechar' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'typechar', subscript_call_result_20088)
    
    # Assigning a Subscript to a Name (line 497):
    
    # Assigning a Subscript to a Name (line 497):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 497)
    j_20089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 24), 'j')
    slice_20090 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 497, 19), j_20089, None, None)
    # Getting the type of 'spec' (line 497)
    spec_20091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'spec')
    # Obtaining the member '__getitem__' of a type (line 497)
    getitem___20092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 19), spec_20091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 497)
    subscript_call_result_20093 = invoke(stypy.reporting.localization.Localization(__file__, 497, 19), getitem___20092, slice_20090)
    
    # Assigning a type to the variable 'spec' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'spec', subscript_call_result_20093)
    
    # Assigning a Compare to a Name (line 498):
    
    # Assigning a Compare to a Name (line 498):
    
    # Getting the type of 'typechar' (line 498)
    typechar_20094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 26), 'typechar')
    str_20095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 38), 'str', 'x')
    # Applying the binary operator '==' (line 498)
    result_eq_20096 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 26), '==', typechar_20094, str_20095)
    
    # Assigning a type to the variable 'is_padding' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'is_padding', result_eq_20096)
    
    # Assigning a Subscript to a Name (line 499):
    
    # Assigning a Subscript to a Name (line 499):
    
    # Obtaining the type of the subscript
    # Getting the type of 'typechar' (line 499)
    typechar_20097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 33), 'typechar')
    # Getting the type of 'type_map' (line 499)
    type_map_20098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'type_map')
    # Obtaining the member '__getitem__' of a type (line 499)
    getitem___20099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 24), type_map_20098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 499)
    subscript_call_result_20100 = invoke(stypy.reporting.localization.Localization(__file__, 499, 24), getitem___20099, typechar_20097)
    
    # Assigning a type to the variable 'dtypechar' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'dtypechar', subscript_call_result_20100)
    
    
    # Getting the type of 'dtypechar' (line 500)
    dtypechar_20101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'dtypechar')
    str_20102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 28), 'str', 'USV')
    # Applying the binary operator 'in' (line 500)
    result_contains_20103 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 15), 'in', dtypechar_20101, str_20102)
    
    # Testing the type of an if condition (line 500)
    if_condition_20104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 500, 12), result_contains_20103)
    # Assigning a type to the variable 'if_condition_20104' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'if_condition_20104', if_condition_20104)
    # SSA begins for if statement (line 500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'dtypechar' (line 501)
    dtypechar_20105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'dtypechar')
    str_20106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 29), 'str', '%d')
    # Getting the type of 'itemsize' (line 501)
    itemsize_20107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 36), 'itemsize')
    # Applying the binary operator '%' (line 501)
    result_mod_20108 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 29), '%', str_20106, itemsize_20107)
    
    # Applying the binary operator '+=' (line 501)
    result_iadd_20109 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 16), '+=', dtypechar_20105, result_mod_20108)
    # Assigning a type to the variable 'dtypechar' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'dtypechar', result_iadd_20109)
    
    
    # Assigning a Num to a Name (line 502):
    
    # Assigning a Num to a Name (line 502):
    int_20110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 27), 'int')
    # Assigning a type to the variable 'itemsize' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'itemsize', int_20110)
    # SSA join for if statement (line 500)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 503):
    
    # Assigning a Call to a Name (line 503):
    
    # Call to get(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'byteorder' (line 503)
    byteorder_20117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 55), 'byteorder', False)
    # Getting the type of 'byteorder' (line 503)
    byteorder_20118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 66), 'byteorder', False)
    # Processing the call keyword arguments (line 503)
    kwargs_20119 = {}
    
    # Obtaining an instance of the builtin type 'dict' (line 503)
    dict_20111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 30), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 503)
    # Adding element type (key, value) (line 503)
    str_20112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 31), 'str', '@')
    str_20113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 36), 'str', '=')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 30), dict_20111, (str_20112, str_20113))
    # Adding element type (key, value) (line 503)
    str_20114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 41), 'str', '^')
    str_20115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 46), 'str', '=')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 30), dict_20111, (str_20114, str_20115))
    
    # Obtaining the member 'get' of a type (line 503)
    get_20116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 30), dict_20111, 'get')
    # Calling get(args, kwargs) (line 503)
    get_call_result_20120 = invoke(stypy.reporting.localization.Localization(__file__, 503, 30), get_20116, *[byteorder_20117, byteorder_20118], **kwargs_20119)
    
    # Assigning a type to the variable 'numpy_byteorder' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'numpy_byteorder', get_call_result_20120)
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to dtype(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'numpy_byteorder' (line 504)
    numpy_byteorder_20122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 26), 'numpy_byteorder', False)
    # Getting the type of 'dtypechar' (line 504)
    dtypechar_20123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 44), 'dtypechar', False)
    # Applying the binary operator '+' (line 504)
    result_add_20124 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 26), '+', numpy_byteorder_20122, dtypechar_20123)
    
    # Processing the call keyword arguments (line 504)
    kwargs_20125 = {}
    # Getting the type of 'dtype' (line 504)
    dtype_20121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 20), 'dtype', False)
    # Calling dtype(args, kwargs) (line 504)
    dtype_call_result_20126 = invoke(stypy.reporting.localization.Localization(__file__, 504, 20), dtype_20121, *[result_add_20124], **kwargs_20125)
    
    # Assigning a type to the variable 'value' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'value', dtype_call_result_20126)
    
    # Assigning a Attribute to a Name (line 505):
    
    # Assigning a Attribute to a Name (line 505):
    # Getting the type of 'value' (line 505)
    value_20127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 20), 'value')
    # Obtaining the member 'alignment' of a type (line 505)
    alignment_20128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 20), value_20127, 'alignment')
    # Assigning a type to the variable 'align' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'align', alignment_20128)
    # SSA branch for the else part of an if statement (line 490)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 507)
    # Processing the call arguments (line 507)
    str_20130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 29), 'str', 'Unknown PEP 3118 data type specifier %r')
    # Getting the type of 'spec' (line 507)
    spec_20131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 73), 'spec', False)
    # Applying the binary operator '%' (line 507)
    result_mod_20132 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 29), '%', str_20130, spec_20131)
    
    # Processing the call keyword arguments (line 507)
    kwargs_20133 = {}
    # Getting the type of 'ValueError' (line 507)
    ValueError_20129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 507)
    ValueError_call_result_20134 = invoke(stypy.reporting.localization.Localization(__file__, 507, 18), ValueError_20129, *[result_mod_20132], **kwargs_20133)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 507, 12), ValueError_call_result_20134, 'raise parameter', BaseException)
    # SSA join for if statement (line 490)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 487)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 515):
    
    # Assigning a Num to a Name (line 515):
    int_20135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 23), 'int')
    # Assigning a type to the variable 'extra_offset' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'extra_offset', int_20135)
    
    
    # Getting the type of 'byteorder' (line 516)
    byteorder_20136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'byteorder')
    str_20137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 24), 'str', '@')
    # Applying the binary operator '==' (line 516)
    result_eq_20138 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 11), '==', byteorder_20136, str_20137)
    
    # Testing the type of an if condition (line 516)
    if_condition_20139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 516, 8), result_eq_20138)
    # Assigning a type to the variable 'if_condition_20139' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'if_condition_20139', if_condition_20139)
    # SSA begins for if statement (line 516)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 517):
    
    # Assigning a BinOp to a Name (line 517):
    
    # Getting the type of 'offset' (line 517)
    offset_20140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 30), 'offset')
    # Applying the 'usub' unary operator (line 517)
    result___neg___20141 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 29), 'usub', offset_20140)
    
    # Getting the type of 'align' (line 517)
    align_20142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 40), 'align')
    # Applying the binary operator '%' (line 517)
    result_mod_20143 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 28), '%', result___neg___20141, align_20142)
    
    # Assigning a type to the variable 'start_padding' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'start_padding', result_mod_20143)
    
    # Assigning a BinOp to a Name (line 518):
    
    # Assigning a BinOp to a Name (line 518):
    
    # Getting the type of 'value' (line 518)
    value_20144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 30), 'value')
    # Obtaining the member 'itemsize' of a type (line 518)
    itemsize_20145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 30), value_20144, 'itemsize')
    # Applying the 'usub' unary operator (line 518)
    result___neg___20146 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 29), 'usub', itemsize_20145)
    
    # Getting the type of 'align' (line 518)
    align_20147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 48), 'align')
    # Applying the binary operator '%' (line 518)
    result_mod_20148 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 28), '%', result___neg___20146, align_20147)
    
    # Assigning a type to the variable 'intra_padding' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'intra_padding', result_mod_20148)
    
    # Getting the type of 'offset' (line 520)
    offset_20149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'offset')
    # Getting the type of 'start_padding' (line 520)
    start_padding_20150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 22), 'start_padding')
    # Applying the binary operator '+=' (line 520)
    result_iadd_20151 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 12), '+=', offset_20149, start_padding_20150)
    # Assigning a type to the variable 'offset' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'offset', result_iadd_20151)
    
    
    
    # Getting the type of 'intra_padding' (line 522)
    intra_padding_20152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'intra_padding')
    int_20153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 32), 'int')
    # Applying the binary operator '!=' (line 522)
    result_ne_20154 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 15), '!=', intra_padding_20152, int_20153)
    
    # Testing the type of an if condition (line 522)
    if_condition_20155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 12), result_ne_20154)
    # Assigning a type to the variable 'if_condition_20155' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'if_condition_20155', if_condition_20155)
    # SSA begins for if statement (line 522)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'itemsize' (line 523)
    itemsize_20156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'itemsize')
    int_20157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 30), 'int')
    # Applying the binary operator '>' (line 523)
    result_gt_20158 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 19), '>', itemsize_20156, int_20157)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 523)
    shape_20159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 36), 'shape')
    # Getting the type of 'None' (line 523)
    None_20160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 49), 'None')
    # Applying the binary operator 'isnot' (line 523)
    result_is_not_20161 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 36), 'isnot', shape_20159, None_20160)
    
    
    
    # Call to _prod(...): (line 523)
    # Processing the call arguments (line 523)
    # Getting the type of 'shape' (line 523)
    shape_20163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 64), 'shape', False)
    # Processing the call keyword arguments (line 523)
    kwargs_20164 = {}
    # Getting the type of '_prod' (line 523)
    _prod_20162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 58), '_prod', False)
    # Calling _prod(args, kwargs) (line 523)
    _prod_call_result_20165 = invoke(stypy.reporting.localization.Localization(__file__, 523, 58), _prod_20162, *[shape_20163], **kwargs_20164)
    
    int_20166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 73), 'int')
    # Applying the binary operator '>' (line 523)
    result_gt_20167 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 58), '>', _prod_call_result_20165, int_20166)
    
    # Applying the binary operator 'and' (line 523)
    result_and_keyword_20168 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 36), 'and', result_is_not_20161, result_gt_20167)
    
    # Applying the binary operator 'or' (line 523)
    result_or_keyword_20169 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 19), 'or', result_gt_20158, result_and_keyword_20168)
    
    # Testing the type of an if condition (line 523)
    if_condition_20170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 523, 16), result_or_keyword_20169)
    # Assigning a type to the variable 'if_condition_20170' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'if_condition_20170', if_condition_20170)
    # SSA begins for if statement (line 523)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 525):
    
    # Assigning a Call to a Name (line 525):
    
    # Call to _add_trailing_padding(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'value' (line 525)
    value_20172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 50), 'value', False)
    # Getting the type of 'intra_padding' (line 525)
    intra_padding_20173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 57), 'intra_padding', False)
    # Processing the call keyword arguments (line 525)
    kwargs_20174 = {}
    # Getting the type of '_add_trailing_padding' (line 525)
    _add_trailing_padding_20171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 28), '_add_trailing_padding', False)
    # Calling _add_trailing_padding(args, kwargs) (line 525)
    _add_trailing_padding_call_result_20175 = invoke(stypy.reporting.localization.Localization(__file__, 525, 28), _add_trailing_padding_20171, *[value_20172, intra_padding_20173], **kwargs_20174)
    
    # Assigning a type to the variable 'value' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 20), 'value', _add_trailing_padding_call_result_20175)
    # SSA branch for the else part of an if statement (line 523)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'extra_offset' (line 529)
    extra_offset_20176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 20), 'extra_offset')
    # Getting the type of 'intra_padding' (line 529)
    intra_padding_20177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 36), 'intra_padding')
    # Applying the binary operator '+=' (line 529)
    result_iadd_20178 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 20), '+=', extra_offset_20176, intra_padding_20177)
    # Assigning a type to the variable 'extra_offset' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 20), 'extra_offset', result_iadd_20178)
    
    # SSA join for if statement (line 523)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 522)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 532):
    
    # Assigning a BinOp to a Name (line 532):
    # Getting the type of 'align' (line 532)
    align_20179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 32), 'align')
    # Getting the type of 'common_alignment' (line 532)
    common_alignment_20180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 38), 'common_alignment')
    # Applying the binary operator '*' (line 532)
    result_mul_20181 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 32), '*', align_20179, common_alignment_20180)
    
    
    # Call to _gcd(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'align' (line 533)
    align_20183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 39), 'align', False)
    # Getting the type of 'common_alignment' (line 533)
    common_alignment_20184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 46), 'common_alignment', False)
    # Processing the call keyword arguments (line 533)
    kwargs_20185 = {}
    # Getting the type of '_gcd' (line 533)
    _gcd_20182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 34), '_gcd', False)
    # Calling _gcd(args, kwargs) (line 533)
    _gcd_call_result_20186 = invoke(stypy.reporting.localization.Localization(__file__, 533, 34), _gcd_20182, *[align_20183, common_alignment_20184], **kwargs_20185)
    
    # Applying the binary operator 'div' (line 533)
    result_div_20187 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 32), 'div', result_mul_20181, _gcd_call_result_20186)
    
    # Assigning a type to the variable 'common_alignment' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'common_alignment', result_div_20187)
    # SSA join for if statement (line 516)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'itemsize' (line 536)
    itemsize_20188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'itemsize')
    int_20189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 23), 'int')
    # Applying the binary operator '!=' (line 536)
    result_ne_20190 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), '!=', itemsize_20188, int_20189)
    
    # Testing the type of an if condition (line 536)
    if_condition_20191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), result_ne_20190)
    # Assigning a type to the variable 'if_condition_20191' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_20191', if_condition_20191)
    # SSA begins for if statement (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 537):
    
    # Assigning a Call to a Name (line 537):
    
    # Call to dtype(...): (line 537)
    # Processing the call arguments (line 537)
    
    # Obtaining an instance of the builtin type 'tuple' (line 537)
    tuple_20193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 537)
    # Adding element type (line 537)
    # Getting the type of 'value' (line 537)
    value_20194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 27), 'value', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 27), tuple_20193, value_20194)
    # Adding element type (line 537)
    
    # Obtaining an instance of the builtin type 'tuple' (line 537)
    tuple_20195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 537)
    # Adding element type (line 537)
    # Getting the type of 'itemsize' (line 537)
    itemsize_20196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 35), 'itemsize', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 35), tuple_20195, itemsize_20196)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 27), tuple_20193, tuple_20195)
    
    # Processing the call keyword arguments (line 537)
    kwargs_20197 = {}
    # Getting the type of 'dtype' (line 537)
    dtype_20192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 20), 'dtype', False)
    # Calling dtype(args, kwargs) (line 537)
    dtype_call_result_20198 = invoke(stypy.reporting.localization.Localization(__file__, 537, 20), dtype_20192, *[tuple_20193], **kwargs_20197)
    
    # Assigning a type to the variable 'value' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'value', dtype_call_result_20198)
    # SSA join for if statement (line 536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 540)
    # Getting the type of 'shape' (line 540)
    shape_20199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'shape')
    # Getting the type of 'None' (line 540)
    None_20200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 24), 'None')
    
    (may_be_20201, more_types_in_union_20202) = may_not_be_none(shape_20199, None_20200)

    if may_be_20201:

        if more_types_in_union_20202:
            # Runtime conditional SSA (line 540)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 541):
        
        # Assigning a Call to a Name (line 541):
        
        # Call to dtype(...): (line 541)
        # Processing the call arguments (line 541)
        
        # Obtaining an instance of the builtin type 'tuple' (line 541)
        tuple_20204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 541)
        # Adding element type (line 541)
        # Getting the type of 'value' (line 541)
        value_20205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 27), 'value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 27), tuple_20204, value_20205)
        # Adding element type (line 541)
        # Getting the type of 'shape' (line 541)
        shape_20206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 34), 'shape', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 27), tuple_20204, shape_20206)
        
        # Processing the call keyword arguments (line 541)
        kwargs_20207 = {}
        # Getting the type of 'dtype' (line 541)
        dtype_20203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 20), 'dtype', False)
        # Calling dtype(args, kwargs) (line 541)
        dtype_call_result_20208 = invoke(stypy.reporting.localization.Localization(__file__, 541, 20), dtype_20203, *[tuple_20204], **kwargs_20207)
        
        # Assigning a type to the variable 'value' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'value', dtype_call_result_20208)

        if more_types_in_union_20202:
            # SSA join for if statement (line 540)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 544):
    
    # Assigning a Name to a Name (line 544):
    # Getting the type of 'False' (line 544)
    False_20209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 29), 'False')
    # Assigning a type to the variable 'this_explicit_name' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'this_explicit_name', False_20209)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'spec' (line 545)
    spec_20210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 11), 'spec')
    
    # Call to startswith(...): (line 545)
    # Processing the call arguments (line 545)
    str_20213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 36), 'str', ':')
    # Processing the call keyword arguments (line 545)
    kwargs_20214 = {}
    # Getting the type of 'spec' (line 545)
    spec_20211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 20), 'spec', False)
    # Obtaining the member 'startswith' of a type (line 545)
    startswith_20212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 20), spec_20211, 'startswith')
    # Calling startswith(args, kwargs) (line 545)
    startswith_call_result_20215 = invoke(stypy.reporting.localization.Localization(__file__, 545, 20), startswith_20212, *[str_20213], **kwargs_20214)
    
    # Applying the binary operator 'and' (line 545)
    result_and_keyword_20216 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 11), 'and', spec_20210, startswith_call_result_20215)
    
    # Testing the type of an if condition (line 545)
    if_condition_20217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 545, 8), result_and_keyword_20216)
    # Assigning a type to the variable 'if_condition_20217' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'if_condition_20217', if_condition_20217)
    # SSA begins for if statement (line 545)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 546):
    
    # Assigning a BinOp to a Name (line 546):
    
    # Call to index(...): (line 546)
    # Processing the call arguments (line 546)
    str_20224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 31), 'str', ':')
    # Processing the call keyword arguments (line 546)
    kwargs_20225 = {}
    
    # Obtaining the type of the subscript
    int_20218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 21), 'int')
    slice_20219 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 546, 16), int_20218, None, None)
    # Getting the type of 'spec' (line 546)
    spec_20220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'spec', False)
    # Obtaining the member '__getitem__' of a type (line 546)
    getitem___20221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), spec_20220, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 546)
    subscript_call_result_20222 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), getitem___20221, slice_20219)
    
    # Obtaining the member 'index' of a type (line 546)
    index_20223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), subscript_call_result_20222, 'index')
    # Calling index(args, kwargs) (line 546)
    index_call_result_20226 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), index_20223, *[str_20224], **kwargs_20225)
    
    int_20227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 38), 'int')
    # Applying the binary operator '+' (line 546)
    result_add_20228 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 16), '+', index_call_result_20226, int_20227)
    
    # Assigning a type to the variable 'i' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'i', result_add_20228)
    
    # Assigning a Subscript to a Name (line 547):
    
    # Assigning a Subscript to a Name (line 547):
    
    # Obtaining the type of the subscript
    int_20229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 24), 'int')
    # Getting the type of 'i' (line 547)
    i_20230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 26), 'i')
    slice_20231 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 547, 19), int_20229, i_20230, None)
    # Getting the type of 'spec' (line 547)
    spec_20232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 19), 'spec')
    # Obtaining the member '__getitem__' of a type (line 547)
    getitem___20233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 19), spec_20232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 547)
    subscript_call_result_20234 = invoke(stypy.reporting.localization.Localization(__file__, 547, 19), getitem___20233, slice_20231)
    
    # Assigning a type to the variable 'name' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'name', subscript_call_result_20234)
    
    # Assigning a Subscript to a Name (line 548):
    
    # Assigning a Subscript to a Name (line 548):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 548)
    i_20235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 24), 'i')
    int_20236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 26), 'int')
    # Applying the binary operator '+' (line 548)
    result_add_20237 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 24), '+', i_20235, int_20236)
    
    slice_20238 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 548, 19), result_add_20237, None, None)
    # Getting the type of 'spec' (line 548)
    spec_20239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'spec')
    # Obtaining the member '__getitem__' of a type (line 548)
    getitem___20240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 19), spec_20239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 548)
    subscript_call_result_20241 = invoke(stypy.reporting.localization.Localization(__file__, 548, 19), getitem___20240, slice_20238)
    
    # Assigning a type to the variable 'spec' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'spec', subscript_call_result_20241)
    
    # Assigning a Name to a Name (line 549):
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'True' (line 549)
    True_20242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 28), 'True')
    # Assigning a type to the variable 'explicit_name' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'explicit_name', True_20242)
    
    # Assigning a Name to a Name (line 550):
    
    # Assigning a Name to a Name (line 550):
    # Getting the type of 'True' (line 550)
    True_20243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 33), 'True')
    # Assigning a type to the variable 'this_explicit_name' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'this_explicit_name', True_20243)
    # SSA branch for the else part of an if statement (line 545)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 552):
    
    # Assigning a Call to a Name (line 552):
    
    # Call to get_dummy_name(...): (line 552)
    # Processing the call keyword arguments (line 552)
    kwargs_20245 = {}
    # Getting the type of 'get_dummy_name' (line 552)
    get_dummy_name_20244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 19), 'get_dummy_name', False)
    # Calling get_dummy_name(args, kwargs) (line 552)
    get_dummy_name_call_result_20246 = invoke(stypy.reporting.localization.Localization(__file__, 552, 19), get_dummy_name_20244, *[], **kwargs_20245)
    
    # Assigning a type to the variable 'name' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'name', get_dummy_name_call_result_20246)
    # SSA join for if statement (line 545)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'is_padding' (line 554)
    is_padding_20247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), 'is_padding')
    # Applying the 'not' unary operator (line 554)
    result_not__20248 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), 'not', is_padding_20247)
    
    # Getting the type of 'this_explicit_name' (line 554)
    this_explicit_name_20249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 29), 'this_explicit_name')
    # Applying the binary operator 'or' (line 554)
    result_or_keyword_20250 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), 'or', result_not__20248, this_explicit_name_20249)
    
    # Testing the type of an if condition (line 554)
    if_condition_20251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 8), result_or_keyword_20250)
    # Assigning a type to the variable 'if_condition_20251' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'if_condition_20251', if_condition_20251)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'name' (line 555)
    name_20252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'name')
    # Getting the type of 'fields' (line 555)
    fields_20253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 23), 'fields')
    # Applying the binary operator 'in' (line 555)
    result_contains_20254 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 15), 'in', name_20252, fields_20253)
    
    # Testing the type of an if condition (line 555)
    if_condition_20255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 12), result_contains_20254)
    # Assigning a type to the variable 'if_condition_20255' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'if_condition_20255', if_condition_20255)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 556)
    # Processing the call arguments (line 556)
    str_20257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 35), 'str', "Duplicate field name '%s' in PEP3118 format")
    # Getting the type of 'name' (line 557)
    name_20258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 37), 'name', False)
    # Applying the binary operator '%' (line 556)
    result_mod_20259 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 35), '%', str_20257, name_20258)
    
    # Processing the call keyword arguments (line 556)
    kwargs_20260 = {}
    # Getting the type of 'RuntimeError' (line 556)
    RuntimeError_20256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 22), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 556)
    RuntimeError_call_result_20261 = invoke(stypy.reporting.localization.Localization(__file__, 556, 22), RuntimeError_20256, *[result_mod_20259], **kwargs_20260)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 556, 16), RuntimeError_call_result_20261, 'raise parameter', BaseException)
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Subscript (line 558):
    
    # Assigning a Tuple to a Subscript (line 558):
    
    # Obtaining an instance of the builtin type 'tuple' (line 558)
    tuple_20262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 558)
    # Adding element type (line 558)
    # Getting the type of 'value' (line 558)
    value_20263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 28), 'value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 28), tuple_20262, value_20263)
    # Adding element type (line 558)
    # Getting the type of 'offset' (line 558)
    offset_20264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 35), 'offset')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 28), tuple_20262, offset_20264)
    
    # Getting the type of 'fields' (line 558)
    fields_20265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'fields')
    # Getting the type of 'name' (line 558)
    name_20266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 19), 'name')
    # Storing an element on a container (line 558)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), fields_20265, (name_20266, tuple_20262))
    
    
    # Getting the type of 'this_explicit_name' (line 559)
    this_explicit_name_20267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 19), 'this_explicit_name')
    # Applying the 'not' unary operator (line 559)
    result_not__20268 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 15), 'not', this_explicit_name_20267)
    
    # Testing the type of an if condition (line 559)
    if_condition_20269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 12), result_not__20268)
    # Assigning a type to the variable 'if_condition_20269' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'if_condition_20269', if_condition_20269)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to next_dummy_name(...): (line 560)
    # Processing the call keyword arguments (line 560)
    kwargs_20271 = {}
    # Getting the type of 'next_dummy_name' (line 560)
    next_dummy_name_20270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'next_dummy_name', False)
    # Calling next_dummy_name(args, kwargs) (line 560)
    next_dummy_name_call_result_20272 = invoke(stypy.reporting.localization.Localization(__file__, 560, 16), next_dummy_name_20270, *[], **kwargs_20271)
    
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 562):
    
    # Assigning a Name to a Name (line 562):
    # Getting the type of 'next_byteorder' (line 562)
    next_byteorder_20273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'next_byteorder')
    # Assigning a type to the variable 'byteorder' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'byteorder', next_byteorder_20273)
    
    # Getting the type of 'offset' (line 564)
    offset_20274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'offset')
    # Getting the type of 'value' (line 564)
    value_20275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 18), 'value')
    # Obtaining the member 'itemsize' of a type (line 564)
    itemsize_20276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 18), value_20275, 'itemsize')
    # Applying the binary operator '+=' (line 564)
    result_iadd_20277 = python_operator(stypy.reporting.localization.Localization(__file__, 564, 8), '+=', offset_20274, itemsize_20276)
    # Assigning a type to the variable 'offset' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'offset', result_iadd_20277)
    
    
    # Getting the type of 'offset' (line 565)
    offset_20278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'offset')
    # Getting the type of 'extra_offset' (line 565)
    extra_offset_20279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 18), 'extra_offset')
    # Applying the binary operator '+=' (line 565)
    result_iadd_20280 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 8), '+=', offset_20278, extra_offset_20279)
    # Assigning a type to the variable 'offset' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'offset', result_iadd_20280)
    
    # SSA join for while statement (line 444)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 568)
    # Processing the call arguments (line 568)
    # Getting the type of 'fields' (line 568)
    fields_20282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'fields', False)
    # Processing the call keyword arguments (line 568)
    kwargs_20283 = {}
    # Getting the type of 'len' (line 568)
    len_20281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'len', False)
    # Calling len(args, kwargs) (line 568)
    len_call_result_20284 = invoke(stypy.reporting.localization.Localization(__file__, 568, 8), len_20281, *[fields_20282], **kwargs_20283)
    
    int_20285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 23), 'int')
    # Applying the binary operator '==' (line 568)
    result_eq_20286 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 8), '==', len_call_result_20284, int_20285)
    
    
    # Getting the type of 'explicit_name' (line 568)
    explicit_name_20287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 33), 'explicit_name')
    # Applying the 'not' unary operator (line 568)
    result_not__20288 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 29), 'not', explicit_name_20287)
    
    # Applying the binary operator 'and' (line 568)
    result_and_keyword_20289 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 8), 'and', result_eq_20286, result_not__20288)
    
    
    # Obtaining the type of the subscript
    int_20290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 25), 'int')
    
    # Obtaining the type of the subscript
    str_20291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 19), 'str', 'f0')
    # Getting the type of 'fields' (line 569)
    fields_20292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'fields')
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___20293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 12), fields_20292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_20294 = invoke(stypy.reporting.localization.Localization(__file__, 569, 12), getitem___20293, str_20291)
    
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___20295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 12), subscript_call_result_20294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_20296 = invoke(stypy.reporting.localization.Localization(__file__, 569, 12), getitem___20295, int_20290)
    
    int_20297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 31), 'int')
    # Applying the binary operator '==' (line 569)
    result_eq_20298 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 12), '==', subscript_call_result_20296, int_20297)
    
    # Applying the binary operator 'and' (line 568)
    result_and_keyword_20299 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 8), 'and', result_and_keyword_20289, result_eq_20298)
    
    # Getting the type of 'is_subdtype' (line 569)
    is_subdtype_20300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 41), 'is_subdtype')
    # Applying the 'not' unary operator (line 569)
    result_not__20301 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 37), 'not', is_subdtype_20300)
    
    # Applying the binary operator 'and' (line 568)
    result_and_keyword_20302 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 8), 'and', result_and_keyword_20299, result_not__20301)
    
    # Testing the type of an if condition (line 568)
    if_condition_20303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 4), result_and_keyword_20302)
    # Assigning a type to the variable 'if_condition_20303' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'if_condition_20303', if_condition_20303)
    # SSA begins for if statement (line 568)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 570):
    
    # Assigning a Subscript to a Name (line 570):
    
    # Obtaining the type of the subscript
    int_20304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 27), 'int')
    
    # Obtaining the type of the subscript
    str_20305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 21), 'str', 'f0')
    # Getting the type of 'fields' (line 570)
    fields_20306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 14), 'fields')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___20307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 14), fields_20306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_20308 = invoke(stypy.reporting.localization.Localization(__file__, 570, 14), getitem___20307, str_20305)
    
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___20309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 14), subscript_call_result_20308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_20310 = invoke(stypy.reporting.localization.Localization(__file__, 570, 14), getitem___20309, int_20304)
    
    # Assigning a type to the variable 'ret' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'ret', subscript_call_result_20310)
    # SSA branch for the else part of an if statement (line 568)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 572):
    
    # Assigning a Call to a Name (line 572):
    
    # Call to dtype(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'fields' (line 572)
    fields_20312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 20), 'fields', False)
    # Processing the call keyword arguments (line 572)
    kwargs_20313 = {}
    # Getting the type of 'dtype' (line 572)
    dtype_20311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 14), 'dtype', False)
    # Calling dtype(args, kwargs) (line 572)
    dtype_call_result_20314 = invoke(stypy.reporting.localization.Localization(__file__, 572, 14), dtype_20311, *[fields_20312], **kwargs_20313)
    
    # Assigning a type to the variable 'ret' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'ret', dtype_call_result_20314)
    # SSA join for if statement (line 568)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 575):
    
    # Assigning a BinOp to a Name (line 575):
    # Getting the type of 'offset' (line 575)
    offset_20315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 14), 'offset')
    # Getting the type of 'ret' (line 575)
    ret_20316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 'ret')
    # Obtaining the member 'itemsize' of a type (line 575)
    itemsize_20317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 23), ret_20316, 'itemsize')
    # Applying the binary operator '-' (line 575)
    result_sub_20318 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 14), '-', offset_20315, itemsize_20317)
    
    # Assigning a type to the variable 'padding' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'padding', result_sub_20318)
    
    
    # Getting the type of 'byteorder' (line 576)
    byteorder_20319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 7), 'byteorder')
    str_20320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 20), 'str', '@')
    # Applying the binary operator '==' (line 576)
    result_eq_20321 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 7), '==', byteorder_20319, str_20320)
    
    # Testing the type of an if condition (line 576)
    if_condition_20322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 4), result_eq_20321)
    # Assigning a type to the variable 'if_condition_20322' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'if_condition_20322', if_condition_20322)
    # SSA begins for if statement (line 576)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'padding' (line 577)
    padding_20323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'padding')
    
    # Getting the type of 'offset' (line 577)
    offset_20324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 21), 'offset')
    # Applying the 'usub' unary operator (line 577)
    result___neg___20325 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 20), 'usub', offset_20324)
    
    # Getting the type of 'common_alignment' (line 577)
    common_alignment_20326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'common_alignment')
    # Applying the binary operator '%' (line 577)
    result_mod_20327 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 19), '%', result___neg___20325, common_alignment_20326)
    
    # Applying the binary operator '+=' (line 577)
    result_iadd_20328 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 8), '+=', padding_20323, result_mod_20327)
    # Assigning a type to the variable 'padding' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'padding', result_iadd_20328)
    
    # SSA join for if statement (line 576)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'is_padding' (line 578)
    is_padding_20329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 7), 'is_padding')
    
    # Getting the type of 'this_explicit_name' (line 578)
    this_explicit_name_20330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 26), 'this_explicit_name')
    # Applying the 'not' unary operator (line 578)
    result_not__20331 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 22), 'not', this_explicit_name_20330)
    
    # Applying the binary operator 'and' (line 578)
    result_and_keyword_20332 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 7), 'and', is_padding_20329, result_not__20331)
    
    # Testing the type of an if condition (line 578)
    if_condition_20333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 578, 4), result_and_keyword_20332)
    # Assigning a type to the variable 'if_condition_20333' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'if_condition_20333', if_condition_20333)
    # SSA begins for if statement (line 578)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 579):
    
    # Assigning a Call to a Name (line 579):
    
    # Call to _add_trailing_padding(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'ret' (line 579)
    ret_20335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 36), 'ret', False)
    # Getting the type of 'padding' (line 579)
    padding_20336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 41), 'padding', False)
    # Processing the call keyword arguments (line 579)
    kwargs_20337 = {}
    # Getting the type of '_add_trailing_padding' (line 579)
    _add_trailing_padding_20334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 14), '_add_trailing_padding', False)
    # Calling _add_trailing_padding(args, kwargs) (line 579)
    _add_trailing_padding_call_result_20338 = invoke(stypy.reporting.localization.Localization(__file__, 579, 14), _add_trailing_padding_20334, *[ret_20335, padding_20336], **kwargs_20337)
    
    # Assigning a type to the variable 'ret' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'ret', _add_trailing_padding_call_result_20338)
    # SSA join for if statement (line 578)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'is_subdtype' (line 582)
    is_subdtype_20339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 7), 'is_subdtype')
    # Testing the type of an if condition (line 582)
    if_condition_20340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 582, 4), is_subdtype_20339)
    # Assigning a type to the variable 'if_condition_20340' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'if_condition_20340', if_condition_20340)
    # SSA begins for if statement (line 582)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 583)
    tuple_20341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 583)
    # Adding element type (line 583)
    # Getting the type of 'ret' (line 583)
    ret_20342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'ret')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 15), tuple_20341, ret_20342)
    # Adding element type (line 583)
    # Getting the type of 'spec' (line 583)
    spec_20343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'spec')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 15), tuple_20341, spec_20343)
    # Adding element type (line 583)
    # Getting the type of 'common_alignment' (line 583)
    common_alignment_20344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 26), 'common_alignment')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 15), tuple_20341, common_alignment_20344)
    # Adding element type (line 583)
    # Getting the type of 'byteorder' (line 583)
    byteorder_20345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 44), 'byteorder')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 15), tuple_20341, byteorder_20345)
    
    # Assigning a type to the variable 'stypy_return_type' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'stypy_return_type', tuple_20341)
    # SSA branch for the else part of an if statement (line 582)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'ret' (line 585)
    ret_20346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 15), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'stypy_return_type', ret_20346)
    # SSA join for if statement (line 582)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_dtype_from_pep3118(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dtype_from_pep3118' in the type store
    # Getting the type of 'stypy_return_type' (line 423)
    stypy_return_type_20347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20347)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dtype_from_pep3118'
    return stypy_return_type_20347

# Assigning a type to the variable '_dtype_from_pep3118' (line 423)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), '_dtype_from_pep3118', _dtype_from_pep3118)

@norecursion
def _add_trailing_padding(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_add_trailing_padding'
    module_type_store = module_type_store.open_function_context('_add_trailing_padding', 587, 0, False)
    
    # Passed parameters checking function
    _add_trailing_padding.stypy_localization = localization
    _add_trailing_padding.stypy_type_of_self = None
    _add_trailing_padding.stypy_type_store = module_type_store
    _add_trailing_padding.stypy_function_name = '_add_trailing_padding'
    _add_trailing_padding.stypy_param_names_list = ['value', 'padding']
    _add_trailing_padding.stypy_varargs_param_name = None
    _add_trailing_padding.stypy_kwargs_param_name = None
    _add_trailing_padding.stypy_call_defaults = defaults
    _add_trailing_padding.stypy_call_varargs = varargs
    _add_trailing_padding.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_add_trailing_padding', ['value', 'padding'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_add_trailing_padding', localization, ['value', 'padding'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_add_trailing_padding(...)' code ##################

    str_20348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 4), 'str', 'Inject the specified number of padding bytes at the end of a dtype')
    
    # Type idiom detected: calculating its left and rigth part (line 589)
    # Getting the type of 'value' (line 589)
    value_20349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 7), 'value')
    # Obtaining the member 'fields' of a type (line 589)
    fields_20350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 7), value_20349, 'fields')
    # Getting the type of 'None' (line 589)
    None_20351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 23), 'None')
    
    (may_be_20352, more_types_in_union_20353) = may_be_none(fields_20350, None_20351)

    if may_be_20352:

        if more_types_in_union_20353:
            # Runtime conditional SSA (line 589)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 590):
        
        # Assigning a Dict to a Name (line 590):
        
        # Obtaining an instance of the builtin type 'dict' (line 590)
        dict_20354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 590)
        # Adding element type (key, value) (line 590)
        str_20355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 19), 'str', 'f0')
        
        # Obtaining an instance of the builtin type 'tuple' (line 590)
        tuple_20356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 590)
        # Adding element type (line 590)
        # Getting the type of 'value' (line 590)
        value_20357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 26), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 26), tuple_20356, value_20357)
        # Adding element type (line 590)
        int_20358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 26), tuple_20356, int_20358)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 18), dict_20354, (str_20355, tuple_20356))
        
        # Assigning a type to the variable 'vfields' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'vfields', dict_20354)

        if more_types_in_union_20353:
            # Runtime conditional SSA for else branch (line 589)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_20352) or more_types_in_union_20353):
        
        # Assigning a Call to a Name (line 592):
        
        # Assigning a Call to a Name (line 592):
        
        # Call to dict(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'value' (line 592)
        value_20360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 23), 'value', False)
        # Obtaining the member 'fields' of a type (line 592)
        fields_20361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 23), value_20360, 'fields')
        # Processing the call keyword arguments (line 592)
        kwargs_20362 = {}
        # Getting the type of 'dict' (line 592)
        dict_20359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 18), 'dict', False)
        # Calling dict(args, kwargs) (line 592)
        dict_call_result_20363 = invoke(stypy.reporting.localization.Localization(__file__, 592, 18), dict_20359, *[fields_20361], **kwargs_20362)
        
        # Assigning a type to the variable 'vfields' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'vfields', dict_call_result_20363)

        if (may_be_20352 and more_types_in_union_20353):
            # SSA join for if statement (line 589)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'value' (line 594)
    value_20364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'value')
    # Obtaining the member 'names' of a type (line 594)
    names_20365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 8), value_20364, 'names')
    
    
    # Obtaining the type of the subscript
    int_20366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 36), 'int')
    # Getting the type of 'value' (line 594)
    value_20367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 24), 'value')
    # Obtaining the member 'names' of a type (line 594)
    names_20368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 24), value_20367, 'names')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___20369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 24), names_20368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_20370 = invoke(stypy.reporting.localization.Localization(__file__, 594, 24), getitem___20369, int_20366)
    
    str_20371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 43), 'str', '')
    # Applying the binary operator '==' (line 594)
    result_eq_20372 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 24), '==', subscript_call_result_20370, str_20371)
    
    # Applying the binary operator 'and' (line 594)
    result_and_keyword_20373 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 8), 'and', names_20365, result_eq_20372)
    
    
    # Obtaining the type of the subscript
    str_20374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 17), 'str', '')
    # Getting the type of 'value' (line 595)
    value_20375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 11), 'value')
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___20376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 11), value_20375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 595)
    subscript_call_result_20377 = invoke(stypy.reporting.localization.Localization(__file__, 595, 11), getitem___20376, str_20374)
    
    # Obtaining the member 'char' of a type (line 595)
    char_20378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 11), subscript_call_result_20377, 'char')
    str_20379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 29), 'str', 'V')
    # Applying the binary operator '==' (line 595)
    result_eq_20380 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 11), '==', char_20378, str_20379)
    
    # Applying the binary operator 'and' (line 594)
    result_and_keyword_20381 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 8), 'and', result_and_keyword_20373, result_eq_20380)
    
    # Testing the type of an if condition (line 594)
    if_condition_20382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 594, 4), result_and_keyword_20381)
    # Assigning a type to the variable 'if_condition_20382' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'if_condition_20382', if_condition_20382)
    # SSA begins for if statement (line 594)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Subscript (line 597):
    
    # Assigning a Tuple to a Subscript (line 597):
    
    # Obtaining an instance of the builtin type 'tuple' (line 597)
    tuple_20383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 597)
    # Adding element type (line 597)
    str_20384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 23), 'str', 'V%d')
    
    # Obtaining the type of the subscript
    int_20385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 44), 'int')
    
    # Obtaining the type of the subscript
    str_20386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 40), 'str', '')
    # Getting the type of 'vfields' (line 597)
    vfields_20387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 32), 'vfields')
    # Obtaining the member '__getitem__' of a type (line 597)
    getitem___20388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 32), vfields_20387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 597)
    subscript_call_result_20389 = invoke(stypy.reporting.localization.Localization(__file__, 597, 32), getitem___20388, str_20386)
    
    # Obtaining the member '__getitem__' of a type (line 597)
    getitem___20390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 32), subscript_call_result_20389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 597)
    subscript_call_result_20391 = invoke(stypy.reporting.localization.Localization(__file__, 597, 32), getitem___20390, int_20385)
    
    # Obtaining the member 'itemsize' of a type (line 597)
    itemsize_20392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 32), subscript_call_result_20391, 'itemsize')
    # Getting the type of 'padding' (line 597)
    padding_20393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 58), 'padding')
    # Applying the binary operator '+' (line 597)
    result_add_20394 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 32), '+', itemsize_20392, padding_20393)
    
    # Applying the binary operator '%' (line 597)
    result_mod_20395 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 23), '%', str_20384, result_add_20394)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 23), tuple_20383, result_mod_20395)
    # Adding element type (line 597)
    
    # Obtaining the type of the subscript
    int_20396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 35), 'int')
    
    # Obtaining the type of the subscript
    str_20397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 31), 'str', '')
    # Getting the type of 'vfields' (line 598)
    vfields_20398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 23), 'vfields')
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___20399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 23), vfields_20398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 598)
    subscript_call_result_20400 = invoke(stypy.reporting.localization.Localization(__file__, 598, 23), getitem___20399, str_20397)
    
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___20401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 23), subscript_call_result_20400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 598)
    subscript_call_result_20402 = invoke(stypy.reporting.localization.Localization(__file__, 598, 23), getitem___20401, int_20396)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 23), tuple_20383, subscript_call_result_20402)
    
    # Getting the type of 'vfields' (line 597)
    vfields_20403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'vfields')
    str_20404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 16), 'str', '')
    # Storing an element on a container (line 597)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 597, 8), vfields_20403, (str_20404, tuple_20383))
    
    # Assigning a Call to a Name (line 599):
    
    # Assigning a Call to a Name (line 599):
    
    # Call to dtype(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'vfields' (line 599)
    vfields_20406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 22), 'vfields', False)
    # Processing the call keyword arguments (line 599)
    kwargs_20407 = {}
    # Getting the type of 'dtype' (line 599)
    dtype_20405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 16), 'dtype', False)
    # Calling dtype(args, kwargs) (line 599)
    dtype_call_result_20408 = invoke(stypy.reporting.localization.Localization(__file__, 599, 16), dtype_20405, *[vfields_20406], **kwargs_20407)
    
    # Assigning a type to the variable 'value' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'value', dtype_call_result_20408)
    # SSA branch for the else part of an if statement (line 594)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 602):
    
    # Assigning a Num to a Name (line 602):
    int_20409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 12), 'int')
    # Assigning a type to the variable 'j' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'j', int_20409)
    
    # Getting the type of 'True' (line 603)
    True_20410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 14), 'True')
    # Testing the type of an if condition (line 603)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 8), True_20410)
    # SSA begins for while statement (line 603)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 604):
    
    # Assigning a BinOp to a Name (line 604):
    str_20411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 19), 'str', 'pad%d')
    # Getting the type of 'j' (line 604)
    j_20412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 29), 'j')
    # Applying the binary operator '%' (line 604)
    result_mod_20413 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 19), '%', str_20411, j_20412)
    
    # Assigning a type to the variable 'name' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'name', result_mod_20413)
    
    
    # Getting the type of 'name' (line 605)
    name_20414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'name')
    # Getting the type of 'vfields' (line 605)
    vfields_20415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 27), 'vfields')
    # Applying the binary operator 'notin' (line 605)
    result_contains_20416 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 15), 'notin', name_20414, vfields_20415)
    
    # Testing the type of an if condition (line 605)
    if_condition_20417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 12), result_contains_20416)
    # Assigning a type to the variable 'if_condition_20417' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'if_condition_20417', if_condition_20417)
    # SSA begins for if statement (line 605)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Subscript (line 606):
    
    # Assigning a Tuple to a Subscript (line 606):
    
    # Obtaining an instance of the builtin type 'tuple' (line 606)
    tuple_20418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 606)
    # Adding element type (line 606)
    str_20419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 33), 'str', 'V%d')
    # Getting the type of 'padding' (line 606)
    padding_20420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 41), 'padding')
    # Applying the binary operator '%' (line 606)
    result_mod_20421 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 33), '%', str_20419, padding_20420)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 33), tuple_20418, result_mod_20421)
    # Adding element type (line 606)
    # Getting the type of 'value' (line 606)
    value_20422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 50), 'value')
    # Obtaining the member 'itemsize' of a type (line 606)
    itemsize_20423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 50), value_20422, 'itemsize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 33), tuple_20418, itemsize_20423)
    
    # Getting the type of 'vfields' (line 606)
    vfields_20424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'vfields')
    # Getting the type of 'name' (line 606)
    name_20425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 24), 'name')
    # Storing an element on a container (line 606)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 606, 16), vfields_20424, (name_20425, tuple_20418))
    # SSA join for if statement (line 605)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'j' (line 608)
    j_20426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'j')
    int_20427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 17), 'int')
    # Applying the binary operator '+=' (line 608)
    result_iadd_20428 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 12), '+=', j_20426, int_20427)
    # Assigning a type to the variable 'j' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'j', result_iadd_20428)
    
    # SSA join for while statement (line 603)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 610):
    
    # Assigning a Call to a Name (line 610):
    
    # Call to dtype(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'vfields' (line 610)
    vfields_20430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 22), 'vfields', False)
    # Processing the call keyword arguments (line 610)
    kwargs_20431 = {}
    # Getting the type of 'dtype' (line 610)
    dtype_20429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 16), 'dtype', False)
    # Calling dtype(args, kwargs) (line 610)
    dtype_call_result_20432 = invoke(stypy.reporting.localization.Localization(__file__, 610, 16), dtype_20429, *[vfields_20430], **kwargs_20431)
    
    # Assigning a type to the variable 'value' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'value', dtype_call_result_20432)
    
    
    str_20433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 11), 'str', '')
    # Getting the type of 'vfields' (line 611)
    vfields_20434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 21), 'vfields')
    # Applying the binary operator 'notin' (line 611)
    result_contains_20435 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 11), 'notin', str_20433, vfields_20434)
    
    # Testing the type of an if condition (line 611)
    if_condition_20436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 8), result_contains_20435)
    # Assigning a type to the variable 'if_condition_20436' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'if_condition_20436', if_condition_20436)
    # SSA begins for if statement (line 611)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 613):
    
    # Assigning a Call to a Name (line 613):
    
    # Call to list(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'value' (line 613)
    value_20438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 25), 'value', False)
    # Obtaining the member 'names' of a type (line 613)
    names_20439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 25), value_20438, 'names')
    # Processing the call keyword arguments (line 613)
    kwargs_20440 = {}
    # Getting the type of 'list' (line 613)
    list_20437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'list', False)
    # Calling list(args, kwargs) (line 613)
    list_call_result_20441 = invoke(stypy.reporting.localization.Localization(__file__, 613, 20), list_20437, *[names_20439], **kwargs_20440)
    
    # Assigning a type to the variable 'names' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'names', list_call_result_20441)
    
    # Assigning a Str to a Subscript (line 614):
    
    # Assigning a Str to a Subscript (line 614):
    str_20442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 24), 'str', '')
    # Getting the type of 'names' (line 614)
    names_20443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'names')
    int_20444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 18), 'int')
    # Storing an element on a container (line 614)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 12), names_20443, (int_20444, str_20442))
    
    # Assigning a Call to a Attribute (line 615):
    
    # Assigning a Call to a Attribute (line 615):
    
    # Call to tuple(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'names' (line 615)
    names_20446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 32), 'names', False)
    # Processing the call keyword arguments (line 615)
    kwargs_20447 = {}
    # Getting the type of 'tuple' (line 615)
    tuple_20445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 26), 'tuple', False)
    # Calling tuple(args, kwargs) (line 615)
    tuple_call_result_20448 = invoke(stypy.reporting.localization.Localization(__file__, 615, 26), tuple_20445, *[names_20446], **kwargs_20447)
    
    # Getting the type of 'value' (line 615)
    value_20449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'value')
    # Setting the type of the member 'names' of a type (line 615)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 12), value_20449, 'names', tuple_call_result_20448)
    # SSA join for if statement (line 611)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 594)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'value' (line 616)
    value_20450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'value')
    # Assigning a type to the variable 'stypy_return_type' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'stypy_return_type', value_20450)
    
    # ################# End of '_add_trailing_padding(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_add_trailing_padding' in the type store
    # Getting the type of 'stypy_return_type' (line 587)
    stypy_return_type_20451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20451)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_add_trailing_padding'
    return stypy_return_type_20451

# Assigning a type to the variable '_add_trailing_padding' (line 587)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), '_add_trailing_padding', _add_trailing_padding)

@norecursion
def _prod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_prod'
    module_type_store = module_type_store.open_function_context('_prod', 618, 0, False)
    
    # Passed parameters checking function
    _prod.stypy_localization = localization
    _prod.stypy_type_of_self = None
    _prod.stypy_type_store = module_type_store
    _prod.stypy_function_name = '_prod'
    _prod.stypy_param_names_list = ['a']
    _prod.stypy_varargs_param_name = None
    _prod.stypy_kwargs_param_name = None
    _prod.stypy_call_defaults = defaults
    _prod.stypy_call_varargs = varargs
    _prod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prod', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prod', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prod(...)' code ##################

    
    # Assigning a Num to a Name (line 619):
    
    # Assigning a Num to a Name (line 619):
    int_20452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 8), 'int')
    # Assigning a type to the variable 'p' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'p', int_20452)
    
    # Getting the type of 'a' (line 620)
    a_20453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 13), 'a')
    # Testing the type of a for loop iterable (line 620)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 620, 4), a_20453)
    # Getting the type of the for loop variable (line 620)
    for_loop_var_20454 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 620, 4), a_20453)
    # Assigning a type to the variable 'x' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'x', for_loop_var_20454)
    # SSA begins for a for statement (line 620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'p' (line 621)
    p_20455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'p')
    # Getting the type of 'x' (line 621)
    x_20456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 13), 'x')
    # Applying the binary operator '*=' (line 621)
    result_imul_20457 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 8), '*=', p_20455, x_20456)
    # Assigning a type to the variable 'p' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'p', result_imul_20457)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'p' (line 622)
    p_20458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 11), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'stypy_return_type', p_20458)
    
    # ################# End of '_prod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prod' in the type store
    # Getting the type of 'stypy_return_type' (line 618)
    stypy_return_type_20459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20459)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prod'
    return stypy_return_type_20459

# Assigning a type to the variable '_prod' (line 618)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 0), '_prod', _prod)

@norecursion
def _gcd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_gcd'
    module_type_store = module_type_store.open_function_context('_gcd', 624, 0, False)
    
    # Passed parameters checking function
    _gcd.stypy_localization = localization
    _gcd.stypy_type_of_self = None
    _gcd.stypy_type_store = module_type_store
    _gcd.stypy_function_name = '_gcd'
    _gcd.stypy_param_names_list = ['a', 'b']
    _gcd.stypy_varargs_param_name = None
    _gcd.stypy_kwargs_param_name = None
    _gcd.stypy_call_defaults = defaults
    _gcd.stypy_call_varargs = varargs
    _gcd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_gcd', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_gcd', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_gcd(...)' code ##################

    str_20460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 4), 'str', 'Calculate the greatest common divisor of a and b')
    
    # Getting the type of 'b' (line 626)
    b_20461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 10), 'b')
    # Testing the type of an if condition (line 626)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 4), b_20461)
    # SSA begins for while statement (line 626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Tuple to a Tuple (line 627):
    
    # Assigning a Name to a Name (line 627):
    # Getting the type of 'b' (line 627)
    b_20462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 15), 'b')
    # Assigning a type to the variable 'tuple_assignment_18768' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'tuple_assignment_18768', b_20462)
    
    # Assigning a BinOp to a Name (line 627):
    # Getting the type of 'a' (line 627)
    a_20463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 18), 'a')
    # Getting the type of 'b' (line 627)
    b_20464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 22), 'b')
    # Applying the binary operator '%' (line 627)
    result_mod_20465 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 18), '%', a_20463, b_20464)
    
    # Assigning a type to the variable 'tuple_assignment_18769' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'tuple_assignment_18769', result_mod_20465)
    
    # Assigning a Name to a Name (line 627):
    # Getting the type of 'tuple_assignment_18768' (line 627)
    tuple_assignment_18768_20466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'tuple_assignment_18768')
    # Assigning a type to the variable 'a' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'a', tuple_assignment_18768_20466)
    
    # Assigning a Name to a Name (line 627):
    # Getting the type of 'tuple_assignment_18769' (line 627)
    tuple_assignment_18769_20467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'tuple_assignment_18769')
    # Assigning a type to the variable 'b' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), 'b', tuple_assignment_18769_20467)
    # SSA join for while statement (line 626)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 628)
    a_20468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'stypy_return_type', a_20468)
    
    # ################# End of '_gcd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_gcd' in the type store
    # Getting the type of 'stypy_return_type' (line 624)
    stypy_return_type_20469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20469)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_gcd'
    return stypy_return_type_20469

# Assigning a type to the variable '_gcd' (line 624)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 0), '_gcd', _gcd)
# Declaration of the 'TooHardError' class
# Getting the type of 'RuntimeError' (line 631)
RuntimeError_20470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 19), 'RuntimeError')

class TooHardError(RuntimeError_20470, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 631, 0, False)
        # Assigning a type to the variable 'self' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TooHardError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TooHardError' (line 631)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 0), 'TooHardError', TooHardError)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
