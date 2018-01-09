
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: NetCDF reader/writer module.
3: 
4: This module is used to read and create NetCDF files. NetCDF files are
5: accessed through the `netcdf_file` object. Data written to and from NetCDF
6: files are contained in `netcdf_variable` objects. Attributes are given
7: as member variables of the `netcdf_file` and `netcdf_variable` objects.
8: 
9: This module implements the Scientific.IO.NetCDF API to read and create
10: NetCDF files. The same API is also used in the PyNIO and pynetcdf
11: modules, allowing these modules to be used interchangeably when working
12: with NetCDF files.
13: 
14: Only NetCDF3 is supported here; for NetCDF4 see
15: `netCDF4-python <http://unidata.github.io/netcdf4-python/>`__,
16: which has a similar API.
17: 
18: '''
19: 
20: from __future__ import division, print_function, absolute_import
21: 
22: # TODO:
23: # * properly implement ``_FillValue``.
24: # * fix character variables.
25: # * implement PAGESIZE for Python 2.6?
26: 
27: # The Scientific.IO.NetCDF API allows attributes to be added directly to
28: # instances of ``netcdf_file`` and ``netcdf_variable``. To differentiate
29: # between user-set attributes and instance attributes, user-set attributes
30: # are automatically stored in the ``_attributes`` attribute by overloading
31: #``__setattr__``. This is the reason why the code sometimes uses
32: #``obj.__dict__['key'] = value``, instead of simply ``obj.key = value``;
33: # otherwise the key would be inserted into userspace attributes.
34: 
35: 
36: __all__ = ['netcdf_file']
37: 
38: 
39: import warnings
40: import weakref
41: from operator import mul
42: from collections import OrderedDict
43: 
44: import mmap as mm
45: 
46: import numpy as np
47: from numpy.compat import asbytes, asstr
48: from numpy import fromstring, dtype, empty, array, asarray
49: from numpy import little_endian as LITTLE_ENDIAN
50: from functools import reduce
51: 
52: from scipy._lib.six import integer_types, text_type, binary_type
53: 
54: ABSENT = b'\x00\x00\x00\x00\x00\x00\x00\x00'
55: ZERO = b'\x00\x00\x00\x00'
56: NC_BYTE = b'\x00\x00\x00\x01'
57: NC_CHAR = b'\x00\x00\x00\x02'
58: NC_SHORT = b'\x00\x00\x00\x03'
59: NC_INT = b'\x00\x00\x00\x04'
60: NC_FLOAT = b'\x00\x00\x00\x05'
61: NC_DOUBLE = b'\x00\x00\x00\x06'
62: NC_DIMENSION = b'\x00\x00\x00\n'
63: NC_VARIABLE = b'\x00\x00\x00\x0b'
64: NC_ATTRIBUTE = b'\x00\x00\x00\x0c'
65: 
66: 
67: TYPEMAP = {NC_BYTE: ('b', 1),
68:             NC_CHAR: ('c', 1),
69:             NC_SHORT: ('h', 2),
70:             NC_INT: ('i', 4),
71:             NC_FLOAT: ('f', 4),
72:             NC_DOUBLE: ('d', 8)}
73: 
74: REVERSE = {('b', 1): NC_BYTE,
75:             ('B', 1): NC_CHAR,
76:             ('c', 1): NC_CHAR,
77:             ('h', 2): NC_SHORT,
78:             ('i', 4): NC_INT,
79:             ('f', 4): NC_FLOAT,
80:             ('d', 8): NC_DOUBLE,
81: 
82:             # these come from asarray(1).dtype.char and asarray('foo').dtype.char,
83:             # used when getting the types from generic attributes.
84:             ('l', 4): NC_INT,
85:             ('S', 1): NC_CHAR}
86: 
87: 
88: class netcdf_file(object):
89:     '''
90:     A file object for NetCDF data.
91: 
92:     A `netcdf_file` object has two standard attributes: `dimensions` and
93:     `variables`. The values of both are dictionaries, mapping dimension
94:     names to their associated lengths and variable names to variables,
95:     respectively. Application programs should never modify these
96:     dictionaries.
97: 
98:     All other attributes correspond to global attributes defined in the
99:     NetCDF file. Global file attributes are created by assigning to an
100:     attribute of the `netcdf_file` object.
101: 
102:     Parameters
103:     ----------
104:     filename : string or file-like
105:         string -> filename
106:     mode : {'r', 'w', 'a'}, optional
107:         read-write-append mode, default is 'r'
108:     mmap : None or bool, optional
109:         Whether to mmap `filename` when reading.  Default is True
110:         when `filename` is a file name, False when `filename` is a
111:         file-like object. Note that when mmap is in use, data arrays
112:         returned refer directly to the mmapped data on disk, and the
113:         file cannot be closed as long as references to it exist.
114:     version : {1, 2}, optional
115:         version of netcdf to read / write, where 1 means *Classic
116:         format* and 2 means *64-bit offset format*.  Default is 1.  See
117:         `here <https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html#select_format>`__
118:         for more info.
119:     maskandscale : bool, optional
120:         Whether to automatically scale and/or mask data based on attributes.
121:         Default is False.
122: 
123:     Notes
124:     -----
125:     The major advantage of this module over other modules is that it doesn't
126:     require the code to be linked to the NetCDF libraries. This module is
127:     derived from `pupynere <https://bitbucket.org/robertodealmeida/pupynere/>`_.
128: 
129:     NetCDF files are a self-describing binary data format. The file contains
130:     metadata that describes the dimensions and variables in the file. More
131:     details about NetCDF files can be found `here
132:     <https://www.unidata.ucar.edu/software/netcdf/docs/user_guide.html>`__. There
133:     are three main sections to a NetCDF data structure:
134: 
135:     1. Dimensions
136:     2. Variables
137:     3. Attributes
138: 
139:     The dimensions section records the name and length of each dimension used
140:     by the variables. The variables would then indicate which dimensions it
141:     uses and any attributes such as data units, along with containing the data
142:     values for the variable. It is good practice to include a
143:     variable that is the same name as a dimension to provide the values for
144:     that axes. Lastly, the attributes section would contain additional
145:     information such as the name of the file creator or the instrument used to
146:     collect the data.
147: 
148:     When writing data to a NetCDF file, there is often the need to indicate the
149:     'record dimension'. A record dimension is the unbounded dimension for a
150:     variable. For example, a temperature variable may have dimensions of
151:     latitude, longitude and time. If one wants to add more temperature data to
152:     the NetCDF file as time progresses, then the temperature variable should
153:     have the time dimension flagged as the record dimension.
154: 
155:     In addition, the NetCDF file header contains the position of the data in
156:     the file, so access can be done in an efficient manner without loading
157:     unnecessary data into memory. It uses the ``mmap`` module to create
158:     Numpy arrays mapped to the data on disk, for the same purpose.
159: 
160:     Note that when `netcdf_file` is used to open a file with mmap=True
161:     (default for read-only), arrays returned by it refer to data
162:     directly on the disk. The file should not be closed, and cannot be cleanly
163:     closed when asked, if such arrays are alive. You may want to copy data arrays
164:     obtained from mmapped Netcdf file if they are to be processed after the file
165:     is closed, see the example below.
166: 
167:     Examples
168:     --------
169:     To create a NetCDF file:
170: 
171:     >>> from scipy.io import netcdf
172:     >>> f = netcdf.netcdf_file('simple.nc', 'w')
173:     >>> f.history = 'Created for a test'
174:     >>> f.createDimension('time', 10)
175:     >>> time = f.createVariable('time', 'i', ('time',))
176:     >>> time[:] = np.arange(10)
177:     >>> time.units = 'days since 2008-01-01'
178:     >>> f.close()
179: 
180:     Note the assignment of ``range(10)`` to ``time[:]``.  Exposing the slice
181:     of the time variable allows for the data to be set in the object, rather
182:     than letting ``range(10)`` overwrite the ``time`` variable.
183: 
184:     To read the NetCDF file we just created:
185: 
186:     >>> from scipy.io import netcdf
187:     >>> f = netcdf.netcdf_file('simple.nc', 'r')
188:     >>> print(f.history)
189:     b'Created for a test'
190:     >>> time = f.variables['time']
191:     >>> print(time.units)
192:     b'days since 2008-01-01'
193:     >>> print(time.shape)
194:     (10,)
195:     >>> print(time[-1])
196:     9
197: 
198:     NetCDF files, when opened read-only, return arrays that refer
199:     directly to memory-mapped data on disk:
200: 
201:     >>> data = time[:]
202:     >>> data.base.base
203:     <mmap.mmap object at 0x7fe753763180>
204: 
205:     If the data is to be processed after the file is closed, it needs
206:     to be copied to main memory:
207: 
208:     >>> data = time[:].copy()
209:     >>> f.close()
210:     >>> data.mean()
211:     4.5
212: 
213:     A NetCDF file can also be used as context manager:
214: 
215:     >>> from scipy.io import netcdf
216:     >>> with netcdf.netcdf_file('simple.nc', 'r') as f:
217:     ...     print(f.history)
218:     b'Created for a test'
219: 
220:     '''
221:     def __init__(self, filename, mode='r', mmap=None, version=1,
222:                  maskandscale=False):
223:         '''Initialize netcdf_file from fileobj (str or file-like).'''
224:         if mode not in 'rwa':
225:             raise ValueError("Mode must be either 'r', 'w' or 'a'.")
226: 
227:         if hasattr(filename, 'seek'):  # file-like
228:             self.fp = filename
229:             self.filename = 'None'
230:             if mmap is None:
231:                 mmap = False
232:             elif mmap and not hasattr(filename, 'fileno'):
233:                 raise ValueError('Cannot use file object for mmap')
234:         else:  # maybe it's a string
235:             self.filename = filename
236:             omode = 'r+' if mode == 'a' else mode
237:             self.fp = open(self.filename, '%sb' % omode)
238:             if mmap is None:
239:                 mmap = True
240: 
241:         if mode != 'r':
242:             # Cannot read write-only files
243:             mmap = False
244: 
245:         self.use_mmap = mmap
246:         self.mode = mode
247:         self.version_byte = version
248:         self.maskandscale = maskandscale
249: 
250:         self.dimensions = OrderedDict()
251:         self.variables = OrderedDict()
252: 
253:         self._dims = []
254:         self._recs = 0
255:         self._recsize = 0
256: 
257:         self._mm = None
258:         self._mm_buf = None
259:         if self.use_mmap:
260:             self._mm = mm.mmap(self.fp.fileno(), 0, access=mm.ACCESS_READ)
261:             self._mm_buf = np.frombuffer(self._mm, dtype=np.int8)
262: 
263:         self._attributes = OrderedDict()
264: 
265:         if mode in 'ra':
266:             self._read()
267: 
268:     def __setattr__(self, attr, value):
269:         # Store user defined attributes in a separate dict,
270:         # so we can save them to file later.
271:         try:
272:             self._attributes[attr] = value
273:         except AttributeError:
274:             pass
275:         self.__dict__[attr] = value
276: 
277:     def close(self):
278:         '''Closes the NetCDF file.'''
279:         if hasattr(self, 'fp') and not self.fp.closed:
280:             try:
281:                 self.flush()
282:             finally:
283:                 self.variables = OrderedDict()
284:                 if self._mm_buf is not None:
285:                     ref = weakref.ref(self._mm_buf)
286:                     self._mm_buf = None
287:                     if ref() is None:
288:                         # self._mm_buf is gc'd, and we can close the mmap
289:                         self._mm.close()
290:                     else:
291:                         # we cannot close self._mm, since self._mm_buf is
292:                         # alive and there may still be arrays referring to it
293:                         warnings.warn((
294:                             "Cannot close a netcdf_file opened with mmap=True, when "
295:                             "netcdf_variables or arrays referring to its data still exist. "
296:                             "All data arrays obtained from such files refer directly to "
297:                             "data on disk, and must be copied before the file can be cleanly "
298:                             "closed. (See netcdf_file docstring for more information on mmap.)"
299:                         ), category=RuntimeWarning)
300:                 self._mm = None
301:                 self.fp.close()
302:     __del__ = close
303: 
304:     def __enter__(self):
305:         return self
306: 
307:     def __exit__(self, type, value, traceback):
308:         self.close()
309: 
310:     def createDimension(self, name, length):
311:         '''
312:         Adds a dimension to the Dimension section of the NetCDF data structure.
313: 
314:         Note that this function merely adds a new dimension that the variables can
315:         reference.  The values for the dimension, if desired, should be added as
316:         a variable using `createVariable`, referring to this dimension.
317: 
318:         Parameters
319:         ----------
320:         name : str
321:             Name of the dimension (Eg, 'lat' or 'time').
322:         length : int
323:             Length of the dimension.
324: 
325:         See Also
326:         --------
327:         createVariable
328: 
329:         '''
330:         if length is None and self._dims:
331:             raise ValueError("Only first dimension may be unlimited!")
332: 
333:         self.dimensions[name] = length
334:         self._dims.append(name)
335: 
336:     def createVariable(self, name, type, dimensions):
337:         '''
338:         Create an empty variable for the `netcdf_file` object, specifying its data
339:         type and the dimensions it uses.
340: 
341:         Parameters
342:         ----------
343:         name : str
344:             Name of the new variable.
345:         type : dtype or str
346:             Data type of the variable.
347:         dimensions : sequence of str
348:             List of the dimension names used by the variable, in the desired order.
349: 
350:         Returns
351:         -------
352:         variable : netcdf_variable
353:             The newly created ``netcdf_variable`` object.
354:             This object has also been added to the `netcdf_file` object as well.
355: 
356:         See Also
357:         --------
358:         createDimension
359: 
360:         Notes
361:         -----
362:         Any dimensions to be used by the variable should already exist in the
363:         NetCDF data structure or should be created by `createDimension` prior to
364:         creating the NetCDF variable.
365: 
366:         '''
367:         shape = tuple([self.dimensions[dim] for dim in dimensions])
368:         shape_ = tuple([dim or 0 for dim in shape])  # replace None with 0 for numpy
369: 
370:         type = dtype(type)
371:         typecode, size = type.char, type.itemsize
372:         if (typecode, size) not in REVERSE:
373:             raise ValueError("NetCDF 3 does not support type %s" % type)
374: 
375:         data = empty(shape_, dtype=type.newbyteorder("B"))  # convert to big endian always for NetCDF 3
376:         self.variables[name] = netcdf_variable(
377:                 data, typecode, size, shape, dimensions,
378:                 maskandscale=self.maskandscale)
379:         return self.variables[name]
380: 
381:     def flush(self):
382:         '''
383:         Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.
384: 
385:         See Also
386:         --------
387:         sync : Identical function
388: 
389:         '''
390:         if hasattr(self, 'mode') and self.mode in 'wa':
391:             self._write()
392:     sync = flush
393: 
394:     def _write(self):
395:         self.fp.seek(0)
396:         self.fp.write(b'CDF')
397:         self.fp.write(array(self.version_byte, '>b').tostring())
398: 
399:         # Write headers and data.
400:         self._write_numrecs()
401:         self._write_dim_array()
402:         self._write_gatt_array()
403:         self._write_var_array()
404: 
405:     def _write_numrecs(self):
406:         # Get highest record count from all record variables.
407:         for var in self.variables.values():
408:             if var.isrec and len(var.data) > self._recs:
409:                 self.__dict__['_recs'] = len(var.data)
410:         self._pack_int(self._recs)
411: 
412:     def _write_dim_array(self):
413:         if self.dimensions:
414:             self.fp.write(NC_DIMENSION)
415:             self._pack_int(len(self.dimensions))
416:             for name in self._dims:
417:                 self._pack_string(name)
418:                 length = self.dimensions[name]
419:                 self._pack_int(length or 0)  # replace None with 0 for record dimension
420:         else:
421:             self.fp.write(ABSENT)
422: 
423:     def _write_gatt_array(self):
424:         self._write_att_array(self._attributes)
425: 
426:     def _write_att_array(self, attributes):
427:         if attributes:
428:             self.fp.write(NC_ATTRIBUTE)
429:             self._pack_int(len(attributes))
430:             for name, values in attributes.items():
431:                 self._pack_string(name)
432:                 self._write_values(values)
433:         else:
434:             self.fp.write(ABSENT)
435: 
436:     def _write_var_array(self):
437:         if self.variables:
438:             self.fp.write(NC_VARIABLE)
439:             self._pack_int(len(self.variables))
440: 
441:             # Sort variable names non-recs first, then recs.
442:             def sortkey(n):
443:                 v = self.variables[n]
444:                 if v.isrec:
445:                     return (-1,)
446:                 return v._shape
447:             variables = sorted(self.variables, key=sortkey, reverse=True)
448: 
449:             # Set the metadata for all variables.
450:             for name in variables:
451:                 self._write_var_metadata(name)
452:             # Now that we have the metadata, we know the vsize of
453:             # each record variable, so we can calculate recsize.
454:             self.__dict__['_recsize'] = sum([
455:                     var._vsize for var in self.variables.values()
456:                     if var.isrec])
457:             # Set the data for all variables.
458:             for name in variables:
459:                 self._write_var_data(name)
460:         else:
461:             self.fp.write(ABSENT)
462: 
463:     def _write_var_metadata(self, name):
464:         var = self.variables[name]
465: 
466:         self._pack_string(name)
467:         self._pack_int(len(var.dimensions))
468:         for dimname in var.dimensions:
469:             dimid = self._dims.index(dimname)
470:             self._pack_int(dimid)
471: 
472:         self._write_att_array(var._attributes)
473: 
474:         nc_type = REVERSE[var.typecode(), var.itemsize()]
475:         self.fp.write(asbytes(nc_type))
476: 
477:         if not var.isrec:
478:             vsize = var.data.size * var.data.itemsize
479:             vsize += -vsize % 4
480:         else:  # record variable
481:             try:
482:                 vsize = var.data[0].size * var.data.itemsize
483:             except IndexError:
484:                 vsize = 0
485:             rec_vars = len([v for v in self.variables.values()
486:                             if v.isrec])
487:             if rec_vars > 1:
488:                 vsize += -vsize % 4
489:         self.variables[name].__dict__['_vsize'] = vsize
490:         self._pack_int(vsize)
491: 
492:         # Pack a bogus begin, and set the real value later.
493:         self.variables[name].__dict__['_begin'] = self.fp.tell()
494:         self._pack_begin(0)
495: 
496:     def _write_var_data(self, name):
497:         var = self.variables[name]
498: 
499:         # Set begin in file header.
500:         the_beguine = self.fp.tell()
501:         self.fp.seek(var._begin)
502:         self._pack_begin(the_beguine)
503:         self.fp.seek(the_beguine)
504: 
505:         # Write data.
506:         if not var.isrec:
507:             self.fp.write(var.data.tostring())
508:             count = var.data.size * var.data.itemsize
509:             self.fp.write(b'0' * (var._vsize - count))
510:         else:  # record variable
511:             # Handle rec vars with shape[0] < nrecs.
512:             if self._recs > len(var.data):
513:                 shape = (self._recs,) + var.data.shape[1:]
514:                 # Resize in-place does not always work since
515:                 # the array might not be single-segment
516:                 try:
517:                     var.data.resize(shape)
518:                 except ValueError:
519:                     var.__dict__['data'] = np.resize(var.data, shape).astype(var.data.dtype)
520: 
521:             pos0 = pos = self.fp.tell()
522:             for rec in var.data:
523:                 # Apparently scalars cannot be converted to big endian. If we
524:                 # try to convert a ``=i4`` scalar to, say, '>i4' the dtype
525:                 # will remain as ``=i4``.
526:                 if not rec.shape and (rec.dtype.byteorder == '<' or
527:                         (rec.dtype.byteorder == '=' and LITTLE_ENDIAN)):
528:                     rec = rec.byteswap()
529:                 self.fp.write(rec.tostring())
530:                 # Padding
531:                 count = rec.size * rec.itemsize
532:                 self.fp.write(b'0' * (var._vsize - count))
533:                 pos += self._recsize
534:                 self.fp.seek(pos)
535:             self.fp.seek(pos0 + var._vsize)
536: 
537:     def _write_values(self, values):
538:         if hasattr(values, 'dtype'):
539:             nc_type = REVERSE[values.dtype.char, values.dtype.itemsize]
540:         else:
541:             types = [(t, NC_INT) for t in integer_types]
542:             types += [
543:                     (float, NC_FLOAT),
544:                     (str, NC_CHAR)
545:                     ]
546:             # bytes index into scalars in py3k.  Check for "string" types
547:             if isinstance(values, text_type) or isinstance(values, binary_type):
548:                 sample = values
549:             else:
550:                 try:
551:                     sample = values[0]  # subscriptable?
552:                 except TypeError:
553:                     sample = values     # scalar
554: 
555:             for class_, nc_type in types:
556:                 if isinstance(sample, class_):
557:                     break
558: 
559:         typecode, size = TYPEMAP[nc_type]
560:         dtype_ = '>%s' % typecode
561:         # asarray() dies with bytes and '>c' in py3k.  Change to 'S'
562:         dtype_ = 'S' if dtype_ == '>c' else dtype_
563: 
564:         values = asarray(values, dtype=dtype_)
565: 
566:         self.fp.write(asbytes(nc_type))
567: 
568:         if values.dtype.char == 'S':
569:             nelems = values.itemsize
570:         else:
571:             nelems = values.size
572:         self._pack_int(nelems)
573: 
574:         if not values.shape and (values.dtype.byteorder == '<' or
575:                 (values.dtype.byteorder == '=' and LITTLE_ENDIAN)):
576:             values = values.byteswap()
577:         self.fp.write(values.tostring())
578:         count = values.size * values.itemsize
579:         self.fp.write(b'0' * (-count % 4))  # pad
580: 
581:     def _read(self):
582:         # Check magic bytes and version
583:         magic = self.fp.read(3)
584:         if not magic == b'CDF':
585:             raise TypeError("Error: %s is not a valid NetCDF 3 file" %
586:                             self.filename)
587:         self.__dict__['version_byte'] = fromstring(self.fp.read(1), '>b')[0]
588: 
589:         # Read file headers and set data.
590:         self._read_numrecs()
591:         self._read_dim_array()
592:         self._read_gatt_array()
593:         self._read_var_array()
594: 
595:     def _read_numrecs(self):
596:         self.__dict__['_recs'] = self._unpack_int()
597: 
598:     def _read_dim_array(self):
599:         header = self.fp.read(4)
600:         if header not in [ZERO, NC_DIMENSION]:
601:             raise ValueError("Unexpected header.")
602:         count = self._unpack_int()
603: 
604:         for dim in range(count):
605:             name = asstr(self._unpack_string())
606:             length = self._unpack_int() or None  # None for record dimension
607:             self.dimensions[name] = length
608:             self._dims.append(name)  # preserve order
609: 
610:     def _read_gatt_array(self):
611:         for k, v in self._read_att_array().items():
612:             self.__setattr__(k, v)
613: 
614:     def _read_att_array(self):
615:         header = self.fp.read(4)
616:         if header not in [ZERO, NC_ATTRIBUTE]:
617:             raise ValueError("Unexpected header.")
618:         count = self._unpack_int()
619: 
620:         attributes = OrderedDict()
621:         for attr in range(count):
622:             name = asstr(self._unpack_string())
623:             attributes[name] = self._read_values()
624:         return attributes
625: 
626:     def _read_var_array(self):
627:         header = self.fp.read(4)
628:         if header not in [ZERO, NC_VARIABLE]:
629:             raise ValueError("Unexpected header.")
630: 
631:         begin = 0
632:         dtypes = {'names': [], 'formats': []}
633:         rec_vars = []
634:         count = self._unpack_int()
635:         for var in range(count):
636:             (name, dimensions, shape, attributes,
637:              typecode, size, dtype_, begin_, vsize) = self._read_var()
638:             # https://www.unidata.ucar.edu/software/netcdf/docs/user_guide.html
639:             # Note that vsize is the product of the dimension lengths
640:             # (omitting the record dimension) and the number of bytes
641:             # per value (determined from the type), increased to the
642:             # next multiple of 4, for each variable. If a record
643:             # variable, this is the amount of space per record. The
644:             # netCDF "record size" is calculated as the sum of the
645:             # vsize's of all the record variables.
646:             #
647:             # The vsize field is actually redundant, because its value
648:             # may be computed from other information in the header. The
649:             # 32-bit vsize field is not large enough to contain the size
650:             # of variables that require more than 2^32 - 4 bytes, so
651:             # 2^32 - 1 is used in the vsize field for such variables.
652:             if shape and shape[0] is None:  # record variable
653:                 rec_vars.append(name)
654:                 # The netCDF "record size" is calculated as the sum of
655:                 # the vsize's of all the record variables.
656:                 self.__dict__['_recsize'] += vsize
657:                 if begin == 0:
658:                     begin = begin_
659:                 dtypes['names'].append(name)
660:                 dtypes['formats'].append(str(shape[1:]) + dtype_)
661: 
662:                 # Handle padding with a virtual variable.
663:                 if typecode in 'bch':
664:                     actual_size = reduce(mul, (1,) + shape[1:]) * size
665:                     padding = -actual_size % 4
666:                     if padding:
667:                         dtypes['names'].append('_padding_%d' % var)
668:                         dtypes['formats'].append('(%d,)>b' % padding)
669: 
670:                 # Data will be set later.
671:                 data = None
672:             else:  # not a record variable
673:                 # Calculate size to avoid problems with vsize (above)
674:                 a_size = reduce(mul, shape, 1) * size
675:                 if self.use_mmap:
676:                     data = self._mm_buf[begin_:begin_+a_size].view(dtype=dtype_)
677:                     data.shape = shape
678:                 else:
679:                     pos = self.fp.tell()
680:                     self.fp.seek(begin_)
681:                     data = fromstring(self.fp.read(a_size), dtype=dtype_)
682:                     data.shape = shape
683:                     self.fp.seek(pos)
684: 
685:             # Add variable.
686:             self.variables[name] = netcdf_variable(
687:                     data, typecode, size, shape, dimensions, attributes,
688:                     maskandscale=self.maskandscale)
689: 
690:         if rec_vars:
691:             # Remove padding when only one record variable.
692:             if len(rec_vars) == 1:
693:                 dtypes['names'] = dtypes['names'][:1]
694:                 dtypes['formats'] = dtypes['formats'][:1]
695: 
696:             # Build rec array.
697:             if self.use_mmap:
698:                 rec_array = self._mm_buf[begin:begin+self._recs*self._recsize].view(dtype=dtypes)
699:                 rec_array.shape = (self._recs,)
700:             else:
701:                 pos = self.fp.tell()
702:                 self.fp.seek(begin)
703:                 rec_array = fromstring(self.fp.read(self._recs*self._recsize), dtype=dtypes)
704:                 rec_array.shape = (self._recs,)
705:                 self.fp.seek(pos)
706: 
707:             for var in rec_vars:
708:                 self.variables[var].__dict__['data'] = rec_array[var]
709: 
710:     def _read_var(self):
711:         name = asstr(self._unpack_string())
712:         dimensions = []
713:         shape = []
714:         dims = self._unpack_int()
715: 
716:         for i in range(dims):
717:             dimid = self._unpack_int()
718:             dimname = self._dims[dimid]
719:             dimensions.append(dimname)
720:             dim = self.dimensions[dimname]
721:             shape.append(dim)
722:         dimensions = tuple(dimensions)
723:         shape = tuple(shape)
724: 
725:         attributes = self._read_att_array()
726:         nc_type = self.fp.read(4)
727:         vsize = self._unpack_int()
728:         begin = [self._unpack_int, self._unpack_int64][self.version_byte-1]()
729: 
730:         typecode, size = TYPEMAP[nc_type]
731:         dtype_ = '>%s' % typecode
732: 
733:         return name, dimensions, shape, attributes, typecode, size, dtype_, begin, vsize
734: 
735:     def _read_values(self):
736:         nc_type = self.fp.read(4)
737:         n = self._unpack_int()
738: 
739:         typecode, size = TYPEMAP[nc_type]
740: 
741:         count = n*size
742:         values = self.fp.read(int(count))
743:         self.fp.read(-count % 4)  # read padding
744: 
745:         if typecode is not 'c':
746:             values = fromstring(values, dtype='>%s' % typecode)
747:             if values.shape == (1,):
748:                 values = values[0]
749:         else:
750:             values = values.rstrip(b'\x00')
751:         return values
752: 
753:     def _pack_begin(self, begin):
754:         if self.version_byte == 1:
755:             self._pack_int(begin)
756:         elif self.version_byte == 2:
757:             self._pack_int64(begin)
758: 
759:     def _pack_int(self, value):
760:         self.fp.write(array(value, '>i').tostring())
761:     _pack_int32 = _pack_int
762: 
763:     def _unpack_int(self):
764:         return int(fromstring(self.fp.read(4), '>i')[0])
765:     _unpack_int32 = _unpack_int
766: 
767:     def _pack_int64(self, value):
768:         self.fp.write(array(value, '>q').tostring())
769: 
770:     def _unpack_int64(self):
771:         return fromstring(self.fp.read(8), '>q')[0]
772: 
773:     def _pack_string(self, s):
774:         count = len(s)
775:         self._pack_int(count)
776:         self.fp.write(asbytes(s))
777:         self.fp.write(b'0' * (-count % 4))  # pad
778: 
779:     def _unpack_string(self):
780:         count = self._unpack_int()
781:         s = self.fp.read(count).rstrip(b'\x00')
782:         self.fp.read(-count % 4)  # read padding
783:         return s
784: 
785: 
786: class netcdf_variable(object):
787:     '''
788:     A data object for the `netcdf` module.
789: 
790:     `netcdf_variable` objects are constructed by calling the method
791:     `netcdf_file.createVariable` on the `netcdf_file` object. `netcdf_variable`
792:     objects behave much like array objects defined in numpy, except that their
793:     data resides in a file. Data is read by indexing and written by assigning
794:     to an indexed subset; the entire array can be accessed by the index ``[:]``
795:     or (for scalars) by using the methods `getValue` and `assignValue`.
796:     `netcdf_variable` objects also have attribute `shape` with the same meaning
797:     as for arrays, but the shape cannot be modified. There is another read-only
798:     attribute `dimensions`, whose value is the tuple of dimension names.
799: 
800:     All other attributes correspond to variable attributes defined in
801:     the NetCDF file. Variable attributes are created by assigning to an
802:     attribute of the `netcdf_variable` object.
803: 
804:     Parameters
805:     ----------
806:     data : array_like
807:         The data array that holds the values for the variable.
808:         Typically, this is initialized as empty, but with the proper shape.
809:     typecode : dtype character code
810:         Desired data-type for the data array.
811:     size : int
812:         Desired element size for the data array.
813:     shape : sequence of ints
814:         The shape of the array.  This should match the lengths of the
815:         variable's dimensions.
816:     dimensions : sequence of strings
817:         The names of the dimensions used by the variable.  Must be in the
818:         same order of the dimension lengths given by `shape`.
819:     attributes : dict, optional
820:         Attribute values (any type) keyed by string names.  These attributes
821:         become attributes for the netcdf_variable object.
822:     maskandscale : bool, optional
823:         Whether to automatically scale and/or mask data based on attributes.
824:         Default is False.
825: 
826: 
827:     Attributes
828:     ----------
829:     dimensions : list of str
830:         List of names of dimensions used by the variable object.
831:     isrec, shape
832:         Properties
833: 
834:     See also
835:     --------
836:     isrec, shape
837: 
838:     '''
839:     def __init__(self, data, typecode, size, shape, dimensions,
840:                  attributes=None,
841:                  maskandscale=False):
842:         self.data = data
843:         self._typecode = typecode
844:         self._size = size
845:         self._shape = shape
846:         self.dimensions = dimensions
847:         self.maskandscale = maskandscale
848: 
849:         self._attributes = attributes or OrderedDict()
850:         for k, v in self._attributes.items():
851:             self.__dict__[k] = v
852: 
853:     def __setattr__(self, attr, value):
854:         # Store user defined attributes in a separate dict,
855:         # so we can save them to file later.
856:         try:
857:             self._attributes[attr] = value
858:         except AttributeError:
859:             pass
860:         self.__dict__[attr] = value
861: 
862:     def isrec(self):
863:         '''Returns whether the variable has a record dimension or not.
864: 
865:         A record dimension is a dimension along which additional data could be
866:         easily appended in the netcdf data structure without much rewriting of
867:         the data file. This attribute is a read-only property of the
868:         `netcdf_variable`.
869: 
870:         '''
871:         return bool(self.data.shape) and not self._shape[0]
872:     isrec = property(isrec)
873: 
874:     def shape(self):
875:         '''Returns the shape tuple of the data variable.
876: 
877:         This is a read-only attribute and can not be modified in the
878:         same manner of other numpy arrays.
879:         '''
880:         return self.data.shape
881:     shape = property(shape)
882: 
883:     def getValue(self):
884:         '''
885:         Retrieve a scalar value from a `netcdf_variable` of length one.
886: 
887:         Raises
888:         ------
889:         ValueError
890:             If the netcdf variable is an array of length greater than one,
891:             this exception will be raised.
892: 
893:         '''
894:         return self.data.item()
895: 
896:     def assignValue(self, value):
897:         '''
898:         Assign a scalar value to a `netcdf_variable` of length one.
899: 
900:         Parameters
901:         ----------
902:         value : scalar
903:             Scalar value (of compatible type) to assign to a length-one netcdf
904:             variable. This value will be written to file.
905: 
906:         Raises
907:         ------
908:         ValueError
909:             If the input is not a scalar, or if the destination is not a length-one
910:             netcdf variable.
911: 
912:         '''
913:         if not self.data.flags.writeable:
914:             # Work-around for a bug in NumPy.  Calling itemset() on a read-only
915:             # memory-mapped array causes a seg. fault.
916:             # See NumPy ticket #1622, and SciPy ticket #1202.
917:             # This check for `writeable` can be removed when the oldest version
918:             # of numpy still supported by scipy contains the fix for #1622.
919:             raise RuntimeError("variable is not writeable")
920: 
921:         self.data.itemset(value)
922: 
923:     def typecode(self):
924:         '''
925:         Return the typecode of the variable.
926: 
927:         Returns
928:         -------
929:         typecode : char
930:             The character typecode of the variable (eg, 'i' for int).
931: 
932:         '''
933:         return self._typecode
934: 
935:     def itemsize(self):
936:         '''
937:         Return the itemsize of the variable.
938: 
939:         Returns
940:         -------
941:         itemsize : int
942:             The element size of the variable (eg, 8 for float64).
943: 
944:         '''
945:         return self._size
946: 
947:     def __getitem__(self, index):
948:         if not self.maskandscale:
949:             return self.data[index]
950: 
951:         data = self.data[index].copy()
952:         missing_value = self._get_missing_value()
953:         data = self._apply_missing_value(data, missing_value)
954:         scale_factor = self._attributes.get('scale_factor')
955:         add_offset = self._attributes.get('add_offset')
956:         if add_offset is not None or scale_factor is not None:
957:             data = data.astype(np.float64)
958:         if scale_factor is not None:
959:             data = data * scale_factor
960:         if add_offset is not None:
961:             data += add_offset
962: 
963:         return data
964: 
965:     def __setitem__(self, index, data):
966:         if self.maskandscale:
967:             missing_value = (
968:                     self._get_missing_value() or
969:                     getattr(data, 'fill_value', 999999))
970:             self._attributes.setdefault('missing_value', missing_value)
971:             self._attributes.setdefault('_FillValue', missing_value)
972:             data = ((data - self._attributes.get('add_offset', 0.0)) /
973:                     self._attributes.get('scale_factor', 1.0))
974:             data = np.ma.asarray(data).filled(missing_value)
975:             if self._typecode not in 'fd' and data.dtype.kind == 'f':
976:                 data = np.round(data)
977: 
978:         # Expand data for record vars?
979:         if self.isrec:
980:             if isinstance(index, tuple):
981:                 rec_index = index[0]
982:             else:
983:                 rec_index = index
984:             if isinstance(rec_index, slice):
985:                 recs = (rec_index.start or 0) + len(data)
986:             else:
987:                 recs = rec_index + 1
988:             if recs > len(self.data):
989:                 shape = (recs,) + self._shape[1:]
990:                 # Resize in-place does not always work since
991:                 # the array might not be single-segment
992:                 try:
993:                     self.data.resize(shape)
994:                 except ValueError:
995:                     self.__dict__['data'] = np.resize(self.data, shape).astype(self.data.dtype)
996:         self.data[index] = data
997: 
998:     def _get_missing_value(self):
999:         '''
1000:         Returns the value denoting "no data" for this variable.
1001: 
1002:         If this variable does not have a missing/fill value, returns None.
1003: 
1004:         If both _FillValue and missing_value are given, give precedence to
1005:         _FillValue. The netCDF standard gives special meaning to _FillValue;
1006:         missing_value is  just used for compatibility with old datasets.
1007:         '''
1008: 
1009:         if '_FillValue' in self._attributes:
1010:             missing_value = self._attributes['_FillValue']
1011:         elif 'missing_value' in self._attributes:
1012:             missing_value = self._attributes['missing_value']
1013:         else:
1014:             missing_value = None
1015: 
1016:         return missing_value
1017: 
1018:     @staticmethod
1019:     def _apply_missing_value(data, missing_value):
1020:         '''
1021:         Applies the given missing value to the data array.
1022: 
1023:         Returns a numpy.ma array, with any value equal to missing_value masked
1024:         out (unless missing_value is None, in which case the original array is
1025:         returned).
1026:         '''
1027: 
1028:         if missing_value is None:
1029:             newdata = data
1030:         else:
1031:             try:
1032:                 missing_value_isnan = np.isnan(missing_value)
1033:             except (TypeError, NotImplementedError):
1034:                 # some data types (e.g., characters) cannot be tested for NaN
1035:                 missing_value_isnan = False
1036: 
1037:             if missing_value_isnan:
1038:                 mymask = np.isnan(data)
1039:             else:
1040:                 mymask = (data == missing_value)
1041: 
1042:             newdata = np.ma.masked_where(mymask, data)
1043: 
1044:         return newdata
1045: 
1046: 
1047: NetCDFFile = netcdf_file
1048: NetCDFVariable = netcdf_variable
1049: 
1050: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_124561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\nNetCDF reader/writer module.\n\nThis module is used to read and create NetCDF files. NetCDF files are\naccessed through the `netcdf_file` object. Data written to and from NetCDF\nfiles are contained in `netcdf_variable` objects. Attributes are given\nas member variables of the `netcdf_file` and `netcdf_variable` objects.\n\nThis module implements the Scientific.IO.NetCDF API to read and create\nNetCDF files. The same API is also used in the PyNIO and pynetcdf\nmodules, allowing these modules to be used interchangeably when working\nwith NetCDF files.\n\nOnly NetCDF3 is supported here; for NetCDF4 see\n`netCDF4-python <http://unidata.github.io/netcdf4-python/>`__,\nwhich has a similar API.\n\n')

# Assigning a List to a Name (line 36):

# Assigning a List to a Name (line 36):
__all__ = ['netcdf_file']
module_type_store.set_exportable_members(['netcdf_file'])

# Obtaining an instance of the builtin type 'list' (line 36)
list_124562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
str_124563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'str', 'netcdf_file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 10), list_124562, str_124563)

# Assigning a type to the variable '__all__' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), '__all__', list_124562)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'import warnings' statement (line 39)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'import weakref' statement (line 40)
import weakref

import_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'weakref', weakref, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from operator import mul' statement (line 41)
try:
    from operator import mul

except:
    mul = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'operator', None, module_type_store, ['mul'], [mul])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'from collections import OrderedDict' statement (line 42)
try:
    from collections import OrderedDict

except:
    OrderedDict = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'collections', None, module_type_store, ['OrderedDict'], [OrderedDict])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'import mmap' statement (line 44)
import mmap as mm

import_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'mm', mm, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 0))

# 'import numpy' statement (line 46)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_124564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'numpy')

if (type(import_124564) is not StypyTypeError):

    if (import_124564 != 'pyd_module'):
        __import__(import_124564)
        sys_modules_124565 = sys.modules[import_124564]
        import_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'np', sys_modules_124565.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'numpy', import_124564)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 0))

# 'from numpy.compat import asbytes, asstr' statement (line 47)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_124566 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.compat')

if (type(import_124566) is not StypyTypeError):

    if (import_124566 != 'pyd_module'):
        __import__(import_124566)
        sys_modules_124567 = sys.modules[import_124566]
        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.compat', sys_modules_124567.module_type_store, module_type_store, ['asbytes', 'asstr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 47, 0), __file__, sys_modules_124567, sys_modules_124567.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes, asstr

        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.compat', None, module_type_store, ['asbytes', 'asstr'], [asbytes, asstr])

else:
    # Assigning a type to the variable 'numpy.compat' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.compat', import_124566)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'from numpy import fromstring, dtype, empty, array, asarray' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_124568 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy')

if (type(import_124568) is not StypyTypeError):

    if (import_124568 != 'pyd_module'):
        __import__(import_124568)
        sys_modules_124569 = sys.modules[import_124568]
        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy', sys_modules_124569.module_type_store, module_type_store, ['fromstring', 'dtype', 'empty', 'array', 'asarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 48, 0), __file__, sys_modules_124569, sys_modules_124569.module_type_store, module_type_store)
    else:
        from numpy import fromstring, dtype, empty, array, asarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy', None, module_type_store, ['fromstring', 'dtype', 'empty', 'array', 'asarray'], [fromstring, dtype, empty, array, asarray])

else:
    # Assigning a type to the variable 'numpy' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy', import_124568)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 49, 0))

# 'from numpy import LITTLE_ENDIAN' statement (line 49)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_124570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy')

if (type(import_124570) is not StypyTypeError):

    if (import_124570 != 'pyd_module'):
        __import__(import_124570)
        sys_modules_124571 = sys.modules[import_124570]
        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy', sys_modules_124571.module_type_store, module_type_store, ['little_endian'])
        nest_module(stypy.reporting.localization.Localization(__file__, 49, 0), __file__, sys_modules_124571, sys_modules_124571.module_type_store, module_type_store)
    else:
        from numpy import little_endian as LITTLE_ENDIAN

        import_from_module(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy', None, module_type_store, ['little_endian'], [LITTLE_ENDIAN])

else:
    # Assigning a type to the variable 'numpy' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'numpy', import_124570)

# Adding an alias
module_type_store.add_alias('LITTLE_ENDIAN', 'little_endian')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'from functools import reduce' statement (line 50)
try:
    from functools import reduce

except:
    reduce = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'functools', None, module_type_store, ['reduce'], [reduce])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 0))

# 'from scipy._lib.six import integer_types, text_type, binary_type' statement (line 52)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_124572 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy._lib.six')

if (type(import_124572) is not StypyTypeError):

    if (import_124572 != 'pyd_module'):
        __import__(import_124572)
        sys_modules_124573 = sys.modules[import_124572]
        import_from_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy._lib.six', sys_modules_124573.module_type_store, module_type_store, ['integer_types', 'text_type', 'binary_type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 52, 0), __file__, sys_modules_124573, sys_modules_124573.module_type_store, module_type_store)
    else:
        from scipy._lib.six import integer_types, text_type, binary_type

        import_from_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy._lib.six', None, module_type_store, ['integer_types', 'text_type', 'binary_type'], [integer_types, text_type, binary_type])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy._lib.six', import_124572)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')


# Assigning a Str to a Name (line 54):

# Assigning a Str to a Name (line 54):
str_124574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'str', '\x00\x00\x00\x00\x00\x00\x00\x00')
# Assigning a type to the variable 'ABSENT' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'ABSENT', str_124574)

# Assigning a Str to a Name (line 55):

# Assigning a Str to a Name (line 55):
str_124575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 7), 'str', '\x00\x00\x00\x00')
# Assigning a type to the variable 'ZERO' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'ZERO', str_124575)

# Assigning a Str to a Name (line 56):

# Assigning a Str to a Name (line 56):
str_124576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 10), 'str', '\x00\x00\x00\x01')
# Assigning a type to the variable 'NC_BYTE' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'NC_BYTE', str_124576)

# Assigning a Str to a Name (line 57):

# Assigning a Str to a Name (line 57):
str_124577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 10), 'str', '\x00\x00\x00\x02')
# Assigning a type to the variable 'NC_CHAR' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'NC_CHAR', str_124577)

# Assigning a Str to a Name (line 58):

# Assigning a Str to a Name (line 58):
str_124578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 11), 'str', '\x00\x00\x00\x03')
# Assigning a type to the variable 'NC_SHORT' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'NC_SHORT', str_124578)

# Assigning a Str to a Name (line 59):

# Assigning a Str to a Name (line 59):
str_124579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'str', '\x00\x00\x00\x04')
# Assigning a type to the variable 'NC_INT' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'NC_INT', str_124579)

# Assigning a Str to a Name (line 60):

# Assigning a Str to a Name (line 60):
str_124580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 11), 'str', '\x00\x00\x00\x05')
# Assigning a type to the variable 'NC_FLOAT' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'NC_FLOAT', str_124580)

# Assigning a Str to a Name (line 61):

# Assigning a Str to a Name (line 61):
str_124581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'str', '\x00\x00\x00\x06')
# Assigning a type to the variable 'NC_DOUBLE' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'NC_DOUBLE', str_124581)

# Assigning a Str to a Name (line 62):

# Assigning a Str to a Name (line 62):
str_124582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'str', '\x00\x00\x00\n')
# Assigning a type to the variable 'NC_DIMENSION' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'NC_DIMENSION', str_124582)

# Assigning a Str to a Name (line 63):

# Assigning a Str to a Name (line 63):
str_124583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'str', '\x00\x00\x00\x0b')
# Assigning a type to the variable 'NC_VARIABLE' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'NC_VARIABLE', str_124583)

# Assigning a Str to a Name (line 64):

# Assigning a Str to a Name (line 64):
str_124584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 15), 'str', '\x00\x00\x00\x0c')
# Assigning a type to the variable 'NC_ATTRIBUTE' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'NC_ATTRIBUTE', str_124584)

# Assigning a Dict to a Name (line 67):

# Assigning a Dict to a Name (line 67):

# Obtaining an instance of the builtin type 'dict' (line 67)
dict_124585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 67)
# Adding element type (key, value) (line 67)
# Getting the type of 'NC_BYTE' (line 67)
NC_BYTE_124586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'NC_BYTE')

# Obtaining an instance of the builtin type 'tuple' (line 67)
tuple_124587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 67)
# Adding element type (line 67)
str_124588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 21), tuple_124587, str_124588)
# Adding element type (line 67)
int_124589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 21), tuple_124587, int_124589)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_124585, (NC_BYTE_124586, tuple_124587))
# Adding element type (key, value) (line 67)
# Getting the type of 'NC_CHAR' (line 68)
NC_CHAR_124590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'NC_CHAR')

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_124591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
str_124592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), tuple_124591, str_124592)
# Adding element type (line 68)
int_124593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), tuple_124591, int_124593)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_124585, (NC_CHAR_124590, tuple_124591))
# Adding element type (key, value) (line 67)
# Getting the type of 'NC_SHORT' (line 69)
NC_SHORT_124594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'NC_SHORT')

# Obtaining an instance of the builtin type 'tuple' (line 69)
tuple_124595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 69)
# Adding element type (line 69)
str_124596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'str', 'h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 23), tuple_124595, str_124596)
# Adding element type (line 69)
int_124597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 23), tuple_124595, int_124597)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_124585, (NC_SHORT_124594, tuple_124595))
# Adding element type (key, value) (line 67)
# Getting the type of 'NC_INT' (line 70)
NC_INT_124598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'NC_INT')

# Obtaining an instance of the builtin type 'tuple' (line 70)
tuple_124599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 70)
# Adding element type (line 70)
str_124600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 21), tuple_124599, str_124600)
# Adding element type (line 70)
int_124601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 21), tuple_124599, int_124601)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_124585, (NC_INT_124598, tuple_124599))
# Adding element type (key, value) (line 67)
# Getting the type of 'NC_FLOAT' (line 71)
NC_FLOAT_124602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'NC_FLOAT')

# Obtaining an instance of the builtin type 'tuple' (line 71)
tuple_124603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 71)
# Adding element type (line 71)
str_124604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 23), tuple_124603, str_124604)
# Adding element type (line 71)
int_124605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 23), tuple_124603, int_124605)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_124585, (NC_FLOAT_124602, tuple_124603))
# Adding element type (key, value) (line 67)
# Getting the type of 'NC_DOUBLE' (line 72)
NC_DOUBLE_124606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'NC_DOUBLE')

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_124607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
str_124608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), tuple_124607, str_124608)
# Adding element type (line 72)
int_124609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), tuple_124607, int_124609)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 10), dict_124585, (NC_DOUBLE_124606, tuple_124607))

# Assigning a type to the variable 'TYPEMAP' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'TYPEMAP', dict_124585)

# Assigning a Dict to a Name (line 74):

# Assigning a Dict to a Name (line 74):

# Obtaining an instance of the builtin type 'dict' (line 74)
dict_124610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 74)
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_124611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
str_124612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), tuple_124611, str_124612)
# Adding element type (line 74)
int_124613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), tuple_124611, int_124613)

# Getting the type of 'NC_BYTE' (line 74)
NC_BYTE_124614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'NC_BYTE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124611, NC_BYTE_124614))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_124615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)
str_124616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'str', 'B')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 13), tuple_124615, str_124616)
# Adding element type (line 75)
int_124617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 13), tuple_124615, int_124617)

# Getting the type of 'NC_CHAR' (line 75)
NC_CHAR_124618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'NC_CHAR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124615, NC_CHAR_124618))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_124619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_124620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 13), tuple_124619, str_124620)
# Adding element type (line 76)
int_124621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 13), tuple_124619, int_124621)

# Getting the type of 'NC_CHAR' (line 76)
NC_CHAR_124622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'NC_CHAR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124619, NC_CHAR_124622))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_124623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
str_124624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'str', 'h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 13), tuple_124623, str_124624)
# Adding element type (line 77)
int_124625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 13), tuple_124623, int_124625)

# Getting the type of 'NC_SHORT' (line 77)
NC_SHORT_124626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'NC_SHORT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124623, NC_SHORT_124626))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_124627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
str_124628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 13), tuple_124627, str_124628)
# Adding element type (line 78)
int_124629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 13), tuple_124627, int_124629)

# Getting the type of 'NC_INT' (line 78)
NC_INT_124630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'NC_INT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124627, NC_INT_124630))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_124631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
str_124632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 13), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 13), tuple_124631, str_124632)
# Adding element type (line 79)
int_124633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 13), tuple_124631, int_124633)

# Getting the type of 'NC_FLOAT' (line 79)
NC_FLOAT_124634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'NC_FLOAT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124631, NC_FLOAT_124634))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_124635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
str_124636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), tuple_124635, str_124636)
# Adding element type (line 80)
int_124637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), tuple_124635, int_124637)

# Getting the type of 'NC_DOUBLE' (line 80)
NC_DOUBLE_124638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'NC_DOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124635, NC_DOUBLE_124638))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 84)
tuple_124639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 84)
# Adding element type (line 84)
str_124640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 13), 'str', 'l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 13), tuple_124639, str_124640)
# Adding element type (line 84)
int_124641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 13), tuple_124639, int_124641)

# Getting the type of 'NC_INT' (line 84)
NC_INT_124642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'NC_INT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124639, NC_INT_124642))
# Adding element type (key, value) (line 74)

# Obtaining an instance of the builtin type 'tuple' (line 85)
tuple_124643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 85)
# Adding element type (line 85)
str_124644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'str', 'S')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_124643, str_124644)
# Adding element type (line 85)
int_124645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_124643, int_124645)

# Getting the type of 'NC_CHAR' (line 85)
NC_CHAR_124646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'NC_CHAR')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 10), dict_124610, (tuple_124643, NC_CHAR_124646))

# Assigning a type to the variable 'REVERSE' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'REVERSE', dict_124610)
# Declaration of the 'netcdf_file' class

class netcdf_file(object, ):
    str_124647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'str', "\n    A file object for NetCDF data.\n\n    A `netcdf_file` object has two standard attributes: `dimensions` and\n    `variables`. The values of both are dictionaries, mapping dimension\n    names to their associated lengths and variable names to variables,\n    respectively. Application programs should never modify these\n    dictionaries.\n\n    All other attributes correspond to global attributes defined in the\n    NetCDF file. Global file attributes are created by assigning to an\n    attribute of the `netcdf_file` object.\n\n    Parameters\n    ----------\n    filename : string or file-like\n        string -> filename\n    mode : {'r', 'w', 'a'}, optional\n        read-write-append mode, default is 'r'\n    mmap : None or bool, optional\n        Whether to mmap `filename` when reading.  Default is True\n        when `filename` is a file name, False when `filename` is a\n        file-like object. Note that when mmap is in use, data arrays\n        returned refer directly to the mmapped data on disk, and the\n        file cannot be closed as long as references to it exist.\n    version : {1, 2}, optional\n        version of netcdf to read / write, where 1 means *Classic\n        format* and 2 means *64-bit offset format*.  Default is 1.  See\n        `here <https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_introduction.html#select_format>`__\n        for more info.\n    maskandscale : bool, optional\n        Whether to automatically scale and/or mask data based on attributes.\n        Default is False.\n\n    Notes\n    -----\n    The major advantage of this module over other modules is that it doesn't\n    require the code to be linked to the NetCDF libraries. This module is\n    derived from `pupynere <https://bitbucket.org/robertodealmeida/pupynere/>`_.\n\n    NetCDF files are a self-describing binary data format. The file contains\n    metadata that describes the dimensions and variables in the file. More\n    details about NetCDF files can be found `here\n    <https://www.unidata.ucar.edu/software/netcdf/docs/user_guide.html>`__. There\n    are three main sections to a NetCDF data structure:\n\n    1. Dimensions\n    2. Variables\n    3. Attributes\n\n    The dimensions section records the name and length of each dimension used\n    by the variables. The variables would then indicate which dimensions it\n    uses and any attributes such as data units, along with containing the data\n    values for the variable. It is good practice to include a\n    variable that is the same name as a dimension to provide the values for\n    that axes. Lastly, the attributes section would contain additional\n    information such as the name of the file creator or the instrument used to\n    collect the data.\n\n    When writing data to a NetCDF file, there is often the need to indicate the\n    'record dimension'. A record dimension is the unbounded dimension for a\n    variable. For example, a temperature variable may have dimensions of\n    latitude, longitude and time. If one wants to add more temperature data to\n    the NetCDF file as time progresses, then the temperature variable should\n    have the time dimension flagged as the record dimension.\n\n    In addition, the NetCDF file header contains the position of the data in\n    the file, so access can be done in an efficient manner without loading\n    unnecessary data into memory. It uses the ``mmap`` module to create\n    Numpy arrays mapped to the data on disk, for the same purpose.\n\n    Note that when `netcdf_file` is used to open a file with mmap=True\n    (default for read-only), arrays returned by it refer to data\n    directly on the disk. The file should not be closed, and cannot be cleanly\n    closed when asked, if such arrays are alive. You may want to copy data arrays\n    obtained from mmapped Netcdf file if they are to be processed after the file\n    is closed, see the example below.\n\n    Examples\n    --------\n    To create a NetCDF file:\n\n    >>> from scipy.io import netcdf\n    >>> f = netcdf.netcdf_file('simple.nc', 'w')\n    >>> f.history = 'Created for a test'\n    >>> f.createDimension('time', 10)\n    >>> time = f.createVariable('time', 'i', ('time',))\n    >>> time[:] = np.arange(10)\n    >>> time.units = 'days since 2008-01-01'\n    >>> f.close()\n\n    Note the assignment of ``range(10)`` to ``time[:]``.  Exposing the slice\n    of the time variable allows for the data to be set in the object, rather\n    than letting ``range(10)`` overwrite the ``time`` variable.\n\n    To read the NetCDF file we just created:\n\n    >>> from scipy.io import netcdf\n    >>> f = netcdf.netcdf_file('simple.nc', 'r')\n    >>> print(f.history)\n    b'Created for a test'\n    >>> time = f.variables['time']\n    >>> print(time.units)\n    b'days since 2008-01-01'\n    >>> print(time.shape)\n    (10,)\n    >>> print(time[-1])\n    9\n\n    NetCDF files, when opened read-only, return arrays that refer\n    directly to memory-mapped data on disk:\n\n    >>> data = time[:]\n    >>> data.base.base\n    <mmap.mmap object at 0x7fe753763180>\n\n    If the data is to be processed after the file is closed, it needs\n    to be copied to main memory:\n\n    >>> data = time[:].copy()\n    >>> f.close()\n    >>> data.mean()\n    4.5\n\n    A NetCDF file can also be used as context manager:\n\n    >>> from scipy.io import netcdf\n    >>> with netcdf.netcdf_file('simple.nc', 'r') as f:\n    ...     print(f.history)\n    b'Created for a test'\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_124648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 38), 'str', 'r')
        # Getting the type of 'None' (line 221)
        None_124649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 48), 'None')
        int_124650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 62), 'int')
        # Getting the type of 'False' (line 222)
        False_124651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 30), 'False')
        defaults = [str_124648, None_124649, int_124650, False_124651]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.__init__', ['filename', 'mode', 'mmap', 'version', 'maskandscale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename', 'mode', 'mmap', 'version', 'maskandscale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_124652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'str', 'Initialize netcdf_file from fileobj (str or file-like).')
        
        
        # Getting the type of 'mode' (line 224)
        mode_124653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'mode')
        str_124654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 23), 'str', 'rwa')
        # Applying the binary operator 'notin' (line 224)
        result_contains_124655 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 11), 'notin', mode_124653, str_124654)
        
        # Testing the type of an if condition (line 224)
        if_condition_124656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 8), result_contains_124655)
        # Assigning a type to the variable 'if_condition_124656' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'if_condition_124656', if_condition_124656)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 225)
        # Processing the call arguments (line 225)
        str_124658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 29), 'str', "Mode must be either 'r', 'w' or 'a'.")
        # Processing the call keyword arguments (line 225)
        kwargs_124659 = {}
        # Getting the type of 'ValueError' (line 225)
        ValueError_124657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 225)
        ValueError_call_result_124660 = invoke(stypy.reporting.localization.Localization(__file__, 225, 18), ValueError_124657, *[str_124658], **kwargs_124659)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 225, 12), ValueError_call_result_124660, 'raise parameter', BaseException)
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 227)
        str_124661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 29), 'str', 'seek')
        # Getting the type of 'filename' (line 227)
        filename_124662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'filename')
        
        (may_be_124663, more_types_in_union_124664) = may_provide_member(str_124661, filename_124662)

        if may_be_124663:

            if more_types_in_union_124664:
                # Runtime conditional SSA (line 227)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'filename' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'filename', remove_not_member_provider_from_union(filename_124662, 'seek'))
            
            # Assigning a Name to a Attribute (line 228):
            
            # Assigning a Name to a Attribute (line 228):
            # Getting the type of 'filename' (line 228)
            filename_124665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'filename')
            # Getting the type of 'self' (line 228)
            self_124666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'self')
            # Setting the type of the member 'fp' of a type (line 228)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), self_124666, 'fp', filename_124665)
            
            # Assigning a Str to a Attribute (line 229):
            
            # Assigning a Str to a Attribute (line 229):
            str_124667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 28), 'str', 'None')
            # Getting the type of 'self' (line 229)
            self_124668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self')
            # Setting the type of the member 'filename' of a type (line 229)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), self_124668, 'filename', str_124667)
            
            # Type idiom detected: calculating its left and rigth part (line 230)
            # Getting the type of 'mmap' (line 230)
            mmap_124669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'mmap')
            # Getting the type of 'None' (line 230)
            None_124670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'None')
            
            (may_be_124671, more_types_in_union_124672) = may_be_none(mmap_124669, None_124670)

            if may_be_124671:

                if more_types_in_union_124672:
                    # Runtime conditional SSA (line 230)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Name (line 231):
                
                # Assigning a Name to a Name (line 231):
                # Getting the type of 'False' (line 231)
                False_124673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'False')
                # Assigning a type to the variable 'mmap' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'mmap', False_124673)

                if more_types_in_union_124672:
                    # Runtime conditional SSA for else branch (line 230)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_124671) or more_types_in_union_124672):
                
                
                # Evaluating a boolean operation
                # Getting the type of 'mmap' (line 232)
                mmap_124674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'mmap')
                
                
                # Call to hasattr(...): (line 232)
                # Processing the call arguments (line 232)
                # Getting the type of 'filename' (line 232)
                filename_124676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 'filename', False)
                str_124677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 48), 'str', 'fileno')
                # Processing the call keyword arguments (line 232)
                kwargs_124678 = {}
                # Getting the type of 'hasattr' (line 232)
                hasattr_124675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'hasattr', False)
                # Calling hasattr(args, kwargs) (line 232)
                hasattr_call_result_124679 = invoke(stypy.reporting.localization.Localization(__file__, 232, 30), hasattr_124675, *[filename_124676, str_124677], **kwargs_124678)
                
                # Applying the 'not' unary operator (line 232)
                result_not__124680 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 26), 'not', hasattr_call_result_124679)
                
                # Applying the binary operator 'and' (line 232)
                result_and_keyword_124681 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 17), 'and', mmap_124674, result_not__124680)
                
                # Testing the type of an if condition (line 232)
                if_condition_124682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 17), result_and_keyword_124681)
                # Assigning a type to the variable 'if_condition_124682' (line 232)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'if_condition_124682', if_condition_124682)
                # SSA begins for if statement (line 232)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to ValueError(...): (line 233)
                # Processing the call arguments (line 233)
                str_124684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 33), 'str', 'Cannot use file object for mmap')
                # Processing the call keyword arguments (line 233)
                kwargs_124685 = {}
                # Getting the type of 'ValueError' (line 233)
                ValueError_124683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 233)
                ValueError_call_result_124686 = invoke(stypy.reporting.localization.Localization(__file__, 233, 22), ValueError_124683, *[str_124684], **kwargs_124685)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 233, 16), ValueError_call_result_124686, 'raise parameter', BaseException)
                # SSA join for if statement (line 232)
                module_type_store = module_type_store.join_ssa_context()
                

                if (may_be_124671 and more_types_in_union_124672):
                    # SSA join for if statement (line 230)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_124664:
                # Runtime conditional SSA for else branch (line 227)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_124663) or more_types_in_union_124664):
            # Assigning a type to the variable 'filename' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'filename', remove_member_provider_from_union(filename_124662, 'seek'))
            
            # Assigning a Name to a Attribute (line 235):
            
            # Assigning a Name to a Attribute (line 235):
            # Getting the type of 'filename' (line 235)
            filename_124687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'filename')
            # Getting the type of 'self' (line 235)
            self_124688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self')
            # Setting the type of the member 'filename' of a type (line 235)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_124688, 'filename', filename_124687)
            
            # Assigning a IfExp to a Name (line 236):
            
            # Assigning a IfExp to a Name (line 236):
            
            
            # Getting the type of 'mode' (line 236)
            mode_124689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'mode')
            str_124690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 36), 'str', 'a')
            # Applying the binary operator '==' (line 236)
            result_eq_124691 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 28), '==', mode_124689, str_124690)
            
            # Testing the type of an if expression (line 236)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 20), result_eq_124691)
            # SSA begins for if expression (line 236)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            str_124692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 20), 'str', 'r+')
            # SSA branch for the else part of an if expression (line 236)
            module_type_store.open_ssa_branch('if expression else')
            # Getting the type of 'mode' (line 236)
            mode_124693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 45), 'mode')
            # SSA join for if expression (line 236)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_124694 = union_type.UnionType.add(str_124692, mode_124693)
            
            # Assigning a type to the variable 'omode' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'omode', if_exp_124694)
            
            # Assigning a Call to a Attribute (line 237):
            
            # Assigning a Call to a Attribute (line 237):
            
            # Call to open(...): (line 237)
            # Processing the call arguments (line 237)
            # Getting the type of 'self' (line 237)
            self_124696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'self', False)
            # Obtaining the member 'filename' of a type (line 237)
            filename_124697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), self_124696, 'filename')
            str_124698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 42), 'str', '%sb')
            # Getting the type of 'omode' (line 237)
            omode_124699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 50), 'omode', False)
            # Applying the binary operator '%' (line 237)
            result_mod_124700 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 42), '%', str_124698, omode_124699)
            
            # Processing the call keyword arguments (line 237)
            kwargs_124701 = {}
            # Getting the type of 'open' (line 237)
            open_124695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'open', False)
            # Calling open(args, kwargs) (line 237)
            open_call_result_124702 = invoke(stypy.reporting.localization.Localization(__file__, 237, 22), open_124695, *[filename_124697, result_mod_124700], **kwargs_124701)
            
            # Getting the type of 'self' (line 237)
            self_124703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self')
            # Setting the type of the member 'fp' of a type (line 237)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_124703, 'fp', open_call_result_124702)
            
            # Type idiom detected: calculating its left and rigth part (line 238)
            # Getting the type of 'mmap' (line 238)
            mmap_124704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'mmap')
            # Getting the type of 'None' (line 238)
            None_124705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'None')
            
            (may_be_124706, more_types_in_union_124707) = may_be_none(mmap_124704, None_124705)

            if may_be_124706:

                if more_types_in_union_124707:
                    # Runtime conditional SSA (line 238)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Name (line 239):
                
                # Assigning a Name to a Name (line 239):
                # Getting the type of 'True' (line 239)
                True_124708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'True')
                # Assigning a type to the variable 'mmap' (line 239)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'mmap', True_124708)

                if more_types_in_union_124707:
                    # SSA join for if statement (line 238)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_124663 and more_types_in_union_124664):
                # SSA join for if statement (line 227)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'mode' (line 241)
        mode_124709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'mode')
        str_124710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 19), 'str', 'r')
        # Applying the binary operator '!=' (line 241)
        result_ne_124711 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 11), '!=', mode_124709, str_124710)
        
        # Testing the type of an if condition (line 241)
        if_condition_124712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), result_ne_124711)
        # Assigning a type to the variable 'if_condition_124712' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_124712', if_condition_124712)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 243):
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'False' (line 243)
        False_124713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'False')
        # Assigning a type to the variable 'mmap' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'mmap', False_124713)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 245):
        
        # Assigning a Name to a Attribute (line 245):
        # Getting the type of 'mmap' (line 245)
        mmap_124714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'mmap')
        # Getting the type of 'self' (line 245)
        self_124715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self')
        # Setting the type of the member 'use_mmap' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_124715, 'use_mmap', mmap_124714)
        
        # Assigning a Name to a Attribute (line 246):
        
        # Assigning a Name to a Attribute (line 246):
        # Getting the type of 'mode' (line 246)
        mode_124716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'mode')
        # Getting the type of 'self' (line 246)
        self_124717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self')
        # Setting the type of the member 'mode' of a type (line 246)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_124717, 'mode', mode_124716)
        
        # Assigning a Name to a Attribute (line 247):
        
        # Assigning a Name to a Attribute (line 247):
        # Getting the type of 'version' (line 247)
        version_124718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'version')
        # Getting the type of 'self' (line 247)
        self_124719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self')
        # Setting the type of the member 'version_byte' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_124719, 'version_byte', version_124718)
        
        # Assigning a Name to a Attribute (line 248):
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'maskandscale' (line 248)
        maskandscale_124720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'maskandscale')
        # Getting the type of 'self' (line 248)
        self_124721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'maskandscale' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_124721, 'maskandscale', maskandscale_124720)
        
        # Assigning a Call to a Attribute (line 250):
        
        # Assigning a Call to a Attribute (line 250):
        
        # Call to OrderedDict(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_124723 = {}
        # Getting the type of 'OrderedDict' (line 250)
        OrderedDict_124722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 26), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 250)
        OrderedDict_call_result_124724 = invoke(stypy.reporting.localization.Localization(__file__, 250, 26), OrderedDict_124722, *[], **kwargs_124723)
        
        # Getting the type of 'self' (line 250)
        self_124725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'dimensions' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_124725, 'dimensions', OrderedDict_call_result_124724)
        
        # Assigning a Call to a Attribute (line 251):
        
        # Assigning a Call to a Attribute (line 251):
        
        # Call to OrderedDict(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_124727 = {}
        # Getting the type of 'OrderedDict' (line 251)
        OrderedDict_124726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 25), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 251)
        OrderedDict_call_result_124728 = invoke(stypy.reporting.localization.Localization(__file__, 251, 25), OrderedDict_124726, *[], **kwargs_124727)
        
        # Getting the type of 'self' (line 251)
        self_124729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'variables' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_124729, 'variables', OrderedDict_call_result_124728)
        
        # Assigning a List to a Attribute (line 253):
        
        # Assigning a List to a Attribute (line 253):
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_124730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        
        # Getting the type of 'self' (line 253)
        self_124731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self')
        # Setting the type of the member '_dims' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_124731, '_dims', list_124730)
        
        # Assigning a Num to a Attribute (line 254):
        
        # Assigning a Num to a Attribute (line 254):
        int_124732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'int')
        # Getting the type of 'self' (line 254)
        self_124733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self')
        # Setting the type of the member '_recs' of a type (line 254)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_124733, '_recs', int_124732)
        
        # Assigning a Num to a Attribute (line 255):
        
        # Assigning a Num to a Attribute (line 255):
        int_124734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 24), 'int')
        # Getting the type of 'self' (line 255)
        self_124735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self')
        # Setting the type of the member '_recsize' of a type (line 255)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_124735, '_recsize', int_124734)
        
        # Assigning a Name to a Attribute (line 257):
        
        # Assigning a Name to a Attribute (line 257):
        # Getting the type of 'None' (line 257)
        None_124736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 'None')
        # Getting the type of 'self' (line 257)
        self_124737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self')
        # Setting the type of the member '_mm' of a type (line 257)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_124737, '_mm', None_124736)
        
        # Assigning a Name to a Attribute (line 258):
        
        # Assigning a Name to a Attribute (line 258):
        # Getting the type of 'None' (line 258)
        None_124738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 23), 'None')
        # Getting the type of 'self' (line 258)
        self_124739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self')
        # Setting the type of the member '_mm_buf' of a type (line 258)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_124739, '_mm_buf', None_124738)
        
        # Getting the type of 'self' (line 259)
        self_124740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'self')
        # Obtaining the member 'use_mmap' of a type (line 259)
        use_mmap_124741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), self_124740, 'use_mmap')
        # Testing the type of an if condition (line 259)
        if_condition_124742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 8), use_mmap_124741)
        # Assigning a type to the variable 'if_condition_124742' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'if_condition_124742', if_condition_124742)
        # SSA begins for if statement (line 259)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 260):
        
        # Assigning a Call to a Attribute (line 260):
        
        # Call to mmap(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Call to fileno(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_124748 = {}
        # Getting the type of 'self' (line 260)
        self_124745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 31), 'self', False)
        # Obtaining the member 'fp' of a type (line 260)
        fp_124746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 31), self_124745, 'fp')
        # Obtaining the member 'fileno' of a type (line 260)
        fileno_124747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 31), fp_124746, 'fileno')
        # Calling fileno(args, kwargs) (line 260)
        fileno_call_result_124749 = invoke(stypy.reporting.localization.Localization(__file__, 260, 31), fileno_124747, *[], **kwargs_124748)
        
        int_124750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 49), 'int')
        # Processing the call keyword arguments (line 260)
        # Getting the type of 'mm' (line 260)
        mm_124751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 59), 'mm', False)
        # Obtaining the member 'ACCESS_READ' of a type (line 260)
        ACCESS_READ_124752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 59), mm_124751, 'ACCESS_READ')
        keyword_124753 = ACCESS_READ_124752
        kwargs_124754 = {'access': keyword_124753}
        # Getting the type of 'mm' (line 260)
        mm_124743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'mm', False)
        # Obtaining the member 'mmap' of a type (line 260)
        mmap_124744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 23), mm_124743, 'mmap')
        # Calling mmap(args, kwargs) (line 260)
        mmap_call_result_124755 = invoke(stypy.reporting.localization.Localization(__file__, 260, 23), mmap_124744, *[fileno_call_result_124749, int_124750], **kwargs_124754)
        
        # Getting the type of 'self' (line 260)
        self_124756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self')
        # Setting the type of the member '_mm' of a type (line 260)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_124756, '_mm', mmap_call_result_124755)
        
        # Assigning a Call to a Attribute (line 261):
        
        # Assigning a Call to a Attribute (line 261):
        
        # Call to frombuffer(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'self' (line 261)
        self_124759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 41), 'self', False)
        # Obtaining the member '_mm' of a type (line 261)
        _mm_124760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 41), self_124759, '_mm')
        # Processing the call keyword arguments (line 261)
        # Getting the type of 'np' (line 261)
        np_124761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 57), 'np', False)
        # Obtaining the member 'int8' of a type (line 261)
        int8_124762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 57), np_124761, 'int8')
        keyword_124763 = int8_124762
        kwargs_124764 = {'dtype': keyword_124763}
        # Getting the type of 'np' (line 261)
        np_124757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 27), 'np', False)
        # Obtaining the member 'frombuffer' of a type (line 261)
        frombuffer_124758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 27), np_124757, 'frombuffer')
        # Calling frombuffer(args, kwargs) (line 261)
        frombuffer_call_result_124765 = invoke(stypy.reporting.localization.Localization(__file__, 261, 27), frombuffer_124758, *[_mm_124760], **kwargs_124764)
        
        # Getting the type of 'self' (line 261)
        self_124766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'self')
        # Setting the type of the member '_mm_buf' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), self_124766, '_mm_buf', frombuffer_call_result_124765)
        # SSA join for if statement (line 259)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 263):
        
        # Assigning a Call to a Attribute (line 263):
        
        # Call to OrderedDict(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_124768 = {}
        # Getting the type of 'OrderedDict' (line 263)
        OrderedDict_124767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 263)
        OrderedDict_call_result_124769 = invoke(stypy.reporting.localization.Localization(__file__, 263, 27), OrderedDict_124767, *[], **kwargs_124768)
        
        # Getting the type of 'self' (line 263)
        self_124770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self')
        # Setting the type of the member '_attributes' of a type (line 263)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_124770, '_attributes', OrderedDict_call_result_124769)
        
        
        # Getting the type of 'mode' (line 265)
        mode_124771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'mode')
        str_124772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 19), 'str', 'ra')
        # Applying the binary operator 'in' (line 265)
        result_contains_124773 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'in', mode_124771, str_124772)
        
        # Testing the type of an if condition (line 265)
        if_condition_124774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_contains_124773)
        # Assigning a type to the variable 'if_condition_124774' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_124774', if_condition_124774)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _read(...): (line 266)
        # Processing the call keyword arguments (line 266)
        kwargs_124777 = {}
        # Getting the type of 'self' (line 266)
        self_124775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self', False)
        # Obtaining the member '_read' of a type (line 266)
        _read_124776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_124775, '_read')
        # Calling _read(args, kwargs) (line 266)
        _read_call_result_124778 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), _read_124776, *[], **kwargs_124777)
        
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __setattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setattr__'
        module_type_store = module_type_store.open_function_context('__setattr__', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_function_name', 'netcdf_file.__setattr__')
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_param_names_list', ['attr', 'value'])
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file.__setattr__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.__setattr__', ['attr', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setattr__', localization, ['attr', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setattr__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Subscript (line 272):
        
        # Assigning a Name to a Subscript (line 272):
        # Getting the type of 'value' (line 272)
        value_124779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 37), 'value')
        # Getting the type of 'self' (line 272)
        self_124780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'self')
        # Obtaining the member '_attributes' of a type (line 272)
        _attributes_124781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), self_124780, '_attributes')
        # Getting the type of 'attr' (line 272)
        attr_124782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'attr')
        # Storing an element on a container (line 272)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 12), _attributes_124781, (attr_124782, value_124779))
        # SSA branch for the except part of a try statement (line 271)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 271)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 275):
        
        # Assigning a Name to a Subscript (line 275):
        # Getting the type of 'value' (line 275)
        value_124783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), 'value')
        # Getting the type of 'self' (line 275)
        self_124784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 275)
        dict___124785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_124784, '__dict__')
        # Getting the type of 'attr' (line 275)
        attr_124786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'attr')
        # Storing an element on a container (line 275)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 8), dict___124785, (attr_124786, value_124783))
        
        # ################# End of '__setattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_124787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setattr__'
        return stypy_return_type_124787


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file.close.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file.close.__dict__.__setitem__('stypy_function_name', 'netcdf_file.close')
        netcdf_file.close.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.close', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'close', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'close(...)' code ##################

        str_124788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'str', 'Closes the NetCDF file.')
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'self' (line 279)
        self_124790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'self', False)
        str_124791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 25), 'str', 'fp')
        # Processing the call keyword arguments (line 279)
        kwargs_124792 = {}
        # Getting the type of 'hasattr' (line 279)
        hasattr_124789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 279)
        hasattr_call_result_124793 = invoke(stypy.reporting.localization.Localization(__file__, 279, 11), hasattr_124789, *[self_124790, str_124791], **kwargs_124792)
        
        
        # Getting the type of 'self' (line 279)
        self_124794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 39), 'self')
        # Obtaining the member 'fp' of a type (line 279)
        fp_124795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 39), self_124794, 'fp')
        # Obtaining the member 'closed' of a type (line 279)
        closed_124796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 39), fp_124795, 'closed')
        # Applying the 'not' unary operator (line 279)
        result_not__124797 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 35), 'not', closed_124796)
        
        # Applying the binary operator 'and' (line 279)
        result_and_keyword_124798 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 11), 'and', hasattr_call_result_124793, result_not__124797)
        
        # Testing the type of an if condition (line 279)
        if_condition_124799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), result_and_keyword_124798)
        # Assigning a type to the variable 'if_condition_124799' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_124799', if_condition_124799)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Try-finally block (line 280)
        
        # Call to flush(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_124802 = {}
        # Getting the type of 'self' (line 281)
        self_124800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'self', False)
        # Obtaining the member 'flush' of a type (line 281)
        flush_124801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), self_124800, 'flush')
        # Calling flush(args, kwargs) (line 281)
        flush_call_result_124803 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), flush_124801, *[], **kwargs_124802)
        
        
        # finally branch of the try-finally block (line 280)
        
        # Assigning a Call to a Attribute (line 283):
        
        # Assigning a Call to a Attribute (line 283):
        
        # Call to OrderedDict(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_124805 = {}
        # Getting the type of 'OrderedDict' (line 283)
        OrderedDict_124804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 33), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 283)
        OrderedDict_call_result_124806 = invoke(stypy.reporting.localization.Localization(__file__, 283, 33), OrderedDict_124804, *[], **kwargs_124805)
        
        # Getting the type of 'self' (line 283)
        self_124807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'self')
        # Setting the type of the member 'variables' of a type (line 283)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 16), self_124807, 'variables', OrderedDict_call_result_124806)
        
        
        # Getting the type of 'self' (line 284)
        self_124808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'self')
        # Obtaining the member '_mm_buf' of a type (line 284)
        _mm_buf_124809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 19), self_124808, '_mm_buf')
        # Getting the type of 'None' (line 284)
        None_124810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 39), 'None')
        # Applying the binary operator 'isnot' (line 284)
        result_is_not_124811 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 19), 'isnot', _mm_buf_124809, None_124810)
        
        # Testing the type of an if condition (line 284)
        if_condition_124812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 16), result_is_not_124811)
        # Assigning a type to the variable 'if_condition_124812' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'if_condition_124812', if_condition_124812)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to ref(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'self' (line 285)
        self_124815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 38), 'self', False)
        # Obtaining the member '_mm_buf' of a type (line 285)
        _mm_buf_124816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 38), self_124815, '_mm_buf')
        # Processing the call keyword arguments (line 285)
        kwargs_124817 = {}
        # Getting the type of 'weakref' (line 285)
        weakref_124813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'weakref', False)
        # Obtaining the member 'ref' of a type (line 285)
        ref_124814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 26), weakref_124813, 'ref')
        # Calling ref(args, kwargs) (line 285)
        ref_call_result_124818 = invoke(stypy.reporting.localization.Localization(__file__, 285, 26), ref_124814, *[_mm_buf_124816], **kwargs_124817)
        
        # Assigning a type to the variable 'ref' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'ref', ref_call_result_124818)
        
        # Assigning a Name to a Attribute (line 286):
        
        # Assigning a Name to a Attribute (line 286):
        # Getting the type of 'None' (line 286)
        None_124819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 35), 'None')
        # Getting the type of 'self' (line 286)
        self_124820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'self')
        # Setting the type of the member '_mm_buf' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 20), self_124820, '_mm_buf', None_124819)
        
        # Type idiom detected: calculating its left and rigth part (line 287)
        
        # Call to ref(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_124822 = {}
        # Getting the type of 'ref' (line 287)
        ref_124821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'ref', False)
        # Calling ref(args, kwargs) (line 287)
        ref_call_result_124823 = invoke(stypy.reporting.localization.Localization(__file__, 287, 23), ref_124821, *[], **kwargs_124822)
        
        # Getting the type of 'None' (line 287)
        None_124824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'None')
        
        (may_be_124825, more_types_in_union_124826) = may_be_none(ref_call_result_124823, None_124824)

        if may_be_124825:

            if more_types_in_union_124826:
                # Runtime conditional SSA (line 287)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to close(...): (line 289)
            # Processing the call keyword arguments (line 289)
            kwargs_124830 = {}
            # Getting the type of 'self' (line 289)
            self_124827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 24), 'self', False)
            # Obtaining the member '_mm' of a type (line 289)
            _mm_124828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 24), self_124827, '_mm')
            # Obtaining the member 'close' of a type (line 289)
            close_124829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 24), _mm_124828, 'close')
            # Calling close(args, kwargs) (line 289)
            close_call_result_124831 = invoke(stypy.reporting.localization.Localization(__file__, 289, 24), close_124829, *[], **kwargs_124830)
            

            if more_types_in_union_124826:
                # Runtime conditional SSA for else branch (line 287)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_124825) or more_types_in_union_124826):
            
            # Call to warn(...): (line 293)
            # Processing the call arguments (line 293)
            str_124834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 28), 'str', 'Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist. All data arrays obtained from such files refer directly to data on disk, and must be copied before the file can be cleanly closed. (See netcdf_file docstring for more information on mmap.)')
            # Processing the call keyword arguments (line 293)
            # Getting the type of 'RuntimeWarning' (line 299)
            RuntimeWarning_124835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'RuntimeWarning', False)
            keyword_124836 = RuntimeWarning_124835
            kwargs_124837 = {'category': keyword_124836}
            # Getting the type of 'warnings' (line 293)
            warnings_124832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 293)
            warn_124833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 24), warnings_124832, 'warn')
            # Calling warn(args, kwargs) (line 293)
            warn_call_result_124838 = invoke(stypy.reporting.localization.Localization(__file__, 293, 24), warn_124833, *[str_124834], **kwargs_124837)
            

            if (may_be_124825 and more_types_in_union_124826):
                # SSA join for if statement (line 287)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 300):
        
        # Assigning a Name to a Attribute (line 300):
        # Getting the type of 'None' (line 300)
        None_124839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'None')
        # Getting the type of 'self' (line 300)
        self_124840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'self')
        # Setting the type of the member '_mm' of a type (line 300)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 16), self_124840, '_mm', None_124839)
        
        # Call to close(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_124844 = {}
        # Getting the type of 'self' (line 301)
        self_124841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'self', False)
        # Obtaining the member 'fp' of a type (line 301)
        fp_124842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 16), self_124841, 'fp')
        # Obtaining the member 'close' of a type (line 301)
        close_124843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 16), fp_124842, 'close')
        # Calling close(args, kwargs) (line 301)
        close_call_result_124845 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), close_124843, *[], **kwargs_124844)
        
        
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_124846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124846)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_124846

    
    # Assigning a Name to a Name (line 302):

    @norecursion
    def __enter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__enter__'
        module_type_store = module_type_store.open_function_context('__enter__', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file.__enter__.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_function_name', 'netcdf_file.__enter__')
        netcdf_file.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.__enter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__enter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__enter__(...)' code ##################

        # Getting the type of 'self' (line 305)
        self_124847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'stypy_return_type', self_124847)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_124848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_124848


    @norecursion
    def __exit__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exit__'
        module_type_store = module_type_store.open_function_context('__exit__', 307, 4, False)
        # Assigning a type to the variable 'self' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file.__exit__.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_function_name', 'netcdf_file.__exit__')
        netcdf_file.__exit__.__dict__.__setitem__('stypy_param_names_list', ['type', 'value', 'traceback'])
        netcdf_file.__exit__.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.__exit__', ['type', 'value', 'traceback'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exit__', localization, ['type', 'value', 'traceback'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exit__(...)' code ##################

        
        # Call to close(...): (line 308)
        # Processing the call keyword arguments (line 308)
        kwargs_124851 = {}
        # Getting the type of 'self' (line 308)
        self_124849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self', False)
        # Obtaining the member 'close' of a type (line 308)
        close_124850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_124849, 'close')
        # Calling close(args, kwargs) (line 308)
        close_call_result_124852 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), close_124850, *[], **kwargs_124851)
        
        
        # ################# End of '__exit__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exit__' in the type store
        # Getting the type of 'stypy_return_type' (line 307)
        stypy_return_type_124853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124853)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exit__'
        return stypy_return_type_124853


    @norecursion
    def createDimension(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'createDimension'
        module_type_store = module_type_store.open_function_context('createDimension', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file.createDimension.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_function_name', 'netcdf_file.createDimension')
        netcdf_file.createDimension.__dict__.__setitem__('stypy_param_names_list', ['name', 'length'])
        netcdf_file.createDimension.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file.createDimension.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.createDimension', ['name', 'length'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'createDimension', localization, ['name', 'length'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'createDimension(...)' code ##################

        str_124854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', "\n        Adds a dimension to the Dimension section of the NetCDF data structure.\n\n        Note that this function merely adds a new dimension that the variables can\n        reference.  The values for the dimension, if desired, should be added as\n        a variable using `createVariable`, referring to this dimension.\n\n        Parameters\n        ----------\n        name : str\n            Name of the dimension (Eg, 'lat' or 'time').\n        length : int\n            Length of the dimension.\n\n        See Also\n        --------\n        createVariable\n\n        ")
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'length' (line 330)
        length_124855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'length')
        # Getting the type of 'None' (line 330)
        None_124856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 21), 'None')
        # Applying the binary operator 'is' (line 330)
        result_is__124857 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 11), 'is', length_124855, None_124856)
        
        # Getting the type of 'self' (line 330)
        self_124858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 30), 'self')
        # Obtaining the member '_dims' of a type (line 330)
        _dims_124859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 30), self_124858, '_dims')
        # Applying the binary operator 'and' (line 330)
        result_and_keyword_124860 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 11), 'and', result_is__124857, _dims_124859)
        
        # Testing the type of an if condition (line 330)
        if_condition_124861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 8), result_and_keyword_124860)
        # Assigning a type to the variable 'if_condition_124861' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'if_condition_124861', if_condition_124861)
        # SSA begins for if statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 331)
        # Processing the call arguments (line 331)
        str_124863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 29), 'str', 'Only first dimension may be unlimited!')
        # Processing the call keyword arguments (line 331)
        kwargs_124864 = {}
        # Getting the type of 'ValueError' (line 331)
        ValueError_124862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 331)
        ValueError_call_result_124865 = invoke(stypy.reporting.localization.Localization(__file__, 331, 18), ValueError_124862, *[str_124863], **kwargs_124864)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 331, 12), ValueError_call_result_124865, 'raise parameter', BaseException)
        # SSA join for if statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 333):
        
        # Assigning a Name to a Subscript (line 333):
        # Getting the type of 'length' (line 333)
        length_124866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 32), 'length')
        # Getting the type of 'self' (line 333)
        self_124867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self')
        # Obtaining the member 'dimensions' of a type (line 333)
        dimensions_124868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_124867, 'dimensions')
        # Getting the type of 'name' (line 333)
        name_124869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'name')
        # Storing an element on a container (line 333)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), dimensions_124868, (name_124869, length_124866))
        
        # Call to append(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'name' (line 334)
        name_124873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 26), 'name', False)
        # Processing the call keyword arguments (line 334)
        kwargs_124874 = {}
        # Getting the type of 'self' (line 334)
        self_124870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'self', False)
        # Obtaining the member '_dims' of a type (line 334)
        _dims_124871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), self_124870, '_dims')
        # Obtaining the member 'append' of a type (line 334)
        append_124872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), _dims_124871, 'append')
        # Calling append(args, kwargs) (line 334)
        append_call_result_124875 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), append_124872, *[name_124873], **kwargs_124874)
        
        
        # ################# End of 'createDimension(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'createDimension' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_124876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'createDimension'
        return stypy_return_type_124876


    @norecursion
    def createVariable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'createVariable'
        module_type_store = module_type_store.open_function_context('createVariable', 336, 4, False)
        # Assigning a type to the variable 'self' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file.createVariable.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_function_name', 'netcdf_file.createVariable')
        netcdf_file.createVariable.__dict__.__setitem__('stypy_param_names_list', ['name', 'type', 'dimensions'])
        netcdf_file.createVariable.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file.createVariable.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.createVariable', ['name', 'type', 'dimensions'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'createVariable', localization, ['name', 'type', 'dimensions'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'createVariable(...)' code ##################

        str_124877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, (-1)), 'str', '\n        Create an empty variable for the `netcdf_file` object, specifying its data\n        type and the dimensions it uses.\n\n        Parameters\n        ----------\n        name : str\n            Name of the new variable.\n        type : dtype or str\n            Data type of the variable.\n        dimensions : sequence of str\n            List of the dimension names used by the variable, in the desired order.\n\n        Returns\n        -------\n        variable : netcdf_variable\n            The newly created ``netcdf_variable`` object.\n            This object has also been added to the `netcdf_file` object as well.\n\n        See Also\n        --------\n        createDimension\n\n        Notes\n        -----\n        Any dimensions to be used by the variable should already exist in the\n        NetCDF data structure or should be created by `createDimension` prior to\n        creating the NetCDF variable.\n\n        ')
        
        # Assigning a Call to a Name (line 367):
        
        # Assigning a Call to a Name (line 367):
        
        # Call to tuple(...): (line 367)
        # Processing the call arguments (line 367)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'dimensions' (line 367)
        dimensions_124884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 55), 'dimensions', False)
        comprehension_124885 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 23), dimensions_124884)
        # Assigning a type to the variable 'dim' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'dim', comprehension_124885)
        
        # Obtaining the type of the subscript
        # Getting the type of 'dim' (line 367)
        dim_124879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 39), 'dim', False)
        # Getting the type of 'self' (line 367)
        self_124880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'self', False)
        # Obtaining the member 'dimensions' of a type (line 367)
        dimensions_124881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 23), self_124880, 'dimensions')
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___124882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 23), dimensions_124881, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_124883 = invoke(stypy.reporting.localization.Localization(__file__, 367, 23), getitem___124882, dim_124879)
        
        list_124886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 23), list_124886, subscript_call_result_124883)
        # Processing the call keyword arguments (line 367)
        kwargs_124887 = {}
        # Getting the type of 'tuple' (line 367)
        tuple_124878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 367)
        tuple_call_result_124888 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), tuple_124878, *[list_124886], **kwargs_124887)
        
        # Assigning a type to the variable 'shape' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'shape', tuple_call_result_124888)
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to tuple(...): (line 368)
        # Processing the call arguments (line 368)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'shape' (line 368)
        shape_124893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 44), 'shape', False)
        comprehension_124894 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 24), shape_124893)
        # Assigning a type to the variable 'dim' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'dim', comprehension_124894)
        
        # Evaluating a boolean operation
        # Getting the type of 'dim' (line 368)
        dim_124890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'dim', False)
        int_124891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 31), 'int')
        # Applying the binary operator 'or' (line 368)
        result_or_keyword_124892 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 24), 'or', dim_124890, int_124891)
        
        list_124895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 24), list_124895, result_or_keyword_124892)
        # Processing the call keyword arguments (line 368)
        kwargs_124896 = {}
        # Getting the type of 'tuple' (line 368)
        tuple_124889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), 'tuple', False)
        # Calling tuple(args, kwargs) (line 368)
        tuple_call_result_124897 = invoke(stypy.reporting.localization.Localization(__file__, 368, 17), tuple_124889, *[list_124895], **kwargs_124896)
        
        # Assigning a type to the variable 'shape_' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'shape_', tuple_call_result_124897)
        
        # Assigning a Call to a Name (line 370):
        
        # Assigning a Call to a Name (line 370):
        
        # Call to dtype(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'type' (line 370)
        type_124899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 21), 'type', False)
        # Processing the call keyword arguments (line 370)
        kwargs_124900 = {}
        # Getting the type of 'dtype' (line 370)
        dtype_124898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'dtype', False)
        # Calling dtype(args, kwargs) (line 370)
        dtype_call_result_124901 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), dtype_124898, *[type_124899], **kwargs_124900)
        
        # Assigning a type to the variable 'type' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'type', dtype_call_result_124901)
        
        # Assigning a Tuple to a Tuple (line 371):
        
        # Assigning a Attribute to a Name (line 371):
        # Getting the type of 'type' (line 371)
        type_124902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 25), 'type')
        # Obtaining the member 'char' of a type (line 371)
        char_124903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 25), type_124902, 'char')
        # Assigning a type to the variable 'tuple_assignment_124544' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_assignment_124544', char_124903)
        
        # Assigning a Attribute to a Name (line 371):
        # Getting the type of 'type' (line 371)
        type_124904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 36), 'type')
        # Obtaining the member 'itemsize' of a type (line 371)
        itemsize_124905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 36), type_124904, 'itemsize')
        # Assigning a type to the variable 'tuple_assignment_124545' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_assignment_124545', itemsize_124905)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_assignment_124544' (line 371)
        tuple_assignment_124544_124906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_assignment_124544')
        # Assigning a type to the variable 'typecode' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'typecode', tuple_assignment_124544_124906)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_assignment_124545' (line 371)
        tuple_assignment_124545_124907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_assignment_124545')
        # Assigning a type to the variable 'size' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 18), 'size', tuple_assignment_124545_124907)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 372)
        tuple_124908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 372)
        # Adding element type (line 372)
        # Getting the type of 'typecode' (line 372)
        typecode_124909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'typecode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 12), tuple_124908, typecode_124909)
        # Adding element type (line 372)
        # Getting the type of 'size' (line 372)
        size_124910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 12), tuple_124908, size_124910)
        
        # Getting the type of 'REVERSE' (line 372)
        REVERSE_124911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 35), 'REVERSE')
        # Applying the binary operator 'notin' (line 372)
        result_contains_124912 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 11), 'notin', tuple_124908, REVERSE_124911)
        
        # Testing the type of an if condition (line 372)
        if_condition_124913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), result_contains_124912)
        # Assigning a type to the variable 'if_condition_124913' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_124913', if_condition_124913)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 373)
        # Processing the call arguments (line 373)
        str_124915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 29), 'str', 'NetCDF 3 does not support type %s')
        # Getting the type of 'type' (line 373)
        type_124916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 67), 'type', False)
        # Applying the binary operator '%' (line 373)
        result_mod_124917 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 29), '%', str_124915, type_124916)
        
        # Processing the call keyword arguments (line 373)
        kwargs_124918 = {}
        # Getting the type of 'ValueError' (line 373)
        ValueError_124914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 373)
        ValueError_call_result_124919 = invoke(stypy.reporting.localization.Localization(__file__, 373, 18), ValueError_124914, *[result_mod_124917], **kwargs_124918)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 373, 12), ValueError_call_result_124919, 'raise parameter', BaseException)
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 375):
        
        # Assigning a Call to a Name (line 375):
        
        # Call to empty(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'shape_' (line 375)
        shape__124921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 21), 'shape_', False)
        # Processing the call keyword arguments (line 375)
        
        # Call to newbyteorder(...): (line 375)
        # Processing the call arguments (line 375)
        str_124924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 53), 'str', 'B')
        # Processing the call keyword arguments (line 375)
        kwargs_124925 = {}
        # Getting the type of 'type' (line 375)
        type_124922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 35), 'type', False)
        # Obtaining the member 'newbyteorder' of a type (line 375)
        newbyteorder_124923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 35), type_124922, 'newbyteorder')
        # Calling newbyteorder(args, kwargs) (line 375)
        newbyteorder_call_result_124926 = invoke(stypy.reporting.localization.Localization(__file__, 375, 35), newbyteorder_124923, *[str_124924], **kwargs_124925)
        
        keyword_124927 = newbyteorder_call_result_124926
        kwargs_124928 = {'dtype': keyword_124927}
        # Getting the type of 'empty' (line 375)
        empty_124920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'empty', False)
        # Calling empty(args, kwargs) (line 375)
        empty_call_result_124929 = invoke(stypy.reporting.localization.Localization(__file__, 375, 15), empty_124920, *[shape__124921], **kwargs_124928)
        
        # Assigning a type to the variable 'data' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'data', empty_call_result_124929)
        
        # Assigning a Call to a Subscript (line 376):
        
        # Assigning a Call to a Subscript (line 376):
        
        # Call to netcdf_variable(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'data' (line 377)
        data_124931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'data', False)
        # Getting the type of 'typecode' (line 377)
        typecode_124932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 22), 'typecode', False)
        # Getting the type of 'size' (line 377)
        size_124933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 32), 'size', False)
        # Getting the type of 'shape' (line 377)
        shape_124934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 38), 'shape', False)
        # Getting the type of 'dimensions' (line 377)
        dimensions_124935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 45), 'dimensions', False)
        # Processing the call keyword arguments (line 376)
        # Getting the type of 'self' (line 378)
        self_124936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 29), 'self', False)
        # Obtaining the member 'maskandscale' of a type (line 378)
        maskandscale_124937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 29), self_124936, 'maskandscale')
        keyword_124938 = maskandscale_124937
        kwargs_124939 = {'maskandscale': keyword_124938}
        # Getting the type of 'netcdf_variable' (line 376)
        netcdf_variable_124930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 31), 'netcdf_variable', False)
        # Calling netcdf_variable(args, kwargs) (line 376)
        netcdf_variable_call_result_124940 = invoke(stypy.reporting.localization.Localization(__file__, 376, 31), netcdf_variable_124930, *[data_124931, typecode_124932, size_124933, shape_124934, dimensions_124935], **kwargs_124939)
        
        # Getting the type of 'self' (line 376)
        self_124941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'self')
        # Obtaining the member 'variables' of a type (line 376)
        variables_124942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), self_124941, 'variables')
        # Getting the type of 'name' (line 376)
        name_124943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 23), 'name')
        # Storing an element on a container (line 376)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 8), variables_124942, (name_124943, netcdf_variable_call_result_124940))
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 379)
        name_124944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 30), 'name')
        # Getting the type of 'self' (line 379)
        self_124945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'self')
        # Obtaining the member 'variables' of a type (line 379)
        variables_124946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), self_124945, 'variables')
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___124947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), variables_124946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 379)
        subscript_call_result_124948 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), getitem___124947, name_124944)
        
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', subscript_call_result_124948)
        
        # ################# End of 'createVariable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'createVariable' in the type store
        # Getting the type of 'stypy_return_type' (line 336)
        stypy_return_type_124949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124949)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'createVariable'
        return stypy_return_type_124949


    @norecursion
    def flush(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flush'
        module_type_store = module_type_store.open_function_context('flush', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file.flush.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file.flush.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file.flush.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file.flush.__dict__.__setitem__('stypy_function_name', 'netcdf_file.flush')
        netcdf_file.flush.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file.flush.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file.flush.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file.flush.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file.flush.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file.flush.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file.flush.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file.flush', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flush', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flush(...)' code ##################

        str_124950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, (-1)), 'str', '\n        Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.\n\n        See Also\n        --------\n        sync : Identical function\n\n        ')
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'self' (line 390)
        self_124952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'self', False)
        str_124953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'str', 'mode')
        # Processing the call keyword arguments (line 390)
        kwargs_124954 = {}
        # Getting the type of 'hasattr' (line 390)
        hasattr_124951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 390)
        hasattr_call_result_124955 = invoke(stypy.reporting.localization.Localization(__file__, 390, 11), hasattr_124951, *[self_124952, str_124953], **kwargs_124954)
        
        
        # Getting the type of 'self' (line 390)
        self_124956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 37), 'self')
        # Obtaining the member 'mode' of a type (line 390)
        mode_124957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 37), self_124956, 'mode')
        str_124958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 50), 'str', 'wa')
        # Applying the binary operator 'in' (line 390)
        result_contains_124959 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 37), 'in', mode_124957, str_124958)
        
        # Applying the binary operator 'and' (line 390)
        result_and_keyword_124960 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 11), 'and', hasattr_call_result_124955, result_contains_124959)
        
        # Testing the type of an if condition (line 390)
        if_condition_124961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 8), result_and_keyword_124960)
        # Assigning a type to the variable 'if_condition_124961' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'if_condition_124961', if_condition_124961)
        # SSA begins for if statement (line 390)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _write(...): (line 391)
        # Processing the call keyword arguments (line 391)
        kwargs_124964 = {}
        # Getting the type of 'self' (line 391)
        self_124962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'self', False)
        # Obtaining the member '_write' of a type (line 391)
        _write_124963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 12), self_124962, '_write')
        # Calling _write(args, kwargs) (line 391)
        _write_call_result_124965 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), _write_124963, *[], **kwargs_124964)
        
        # SSA join for if statement (line 390)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'flush(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flush' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_124966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_124966)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flush'
        return stypy_return_type_124966

    
    # Assigning a Name to a Name (line 392):

    @norecursion
    def _write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write'
        module_type_store = module_type_store.open_function_context('_write', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write')
        netcdf_file._write.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._write.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write(...)' code ##################

        
        # Call to seek(...): (line 395)
        # Processing the call arguments (line 395)
        int_124970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 21), 'int')
        # Processing the call keyword arguments (line 395)
        kwargs_124971 = {}
        # Getting the type of 'self' (line 395)
        self_124967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 395)
        fp_124968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), self_124967, 'fp')
        # Obtaining the member 'seek' of a type (line 395)
        seek_124969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), fp_124968, 'seek')
        # Calling seek(args, kwargs) (line 395)
        seek_call_result_124972 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), seek_124969, *[int_124970], **kwargs_124971)
        
        
        # Call to write(...): (line 396)
        # Processing the call arguments (line 396)
        str_124976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 22), 'str', 'CDF')
        # Processing the call keyword arguments (line 396)
        kwargs_124977 = {}
        # Getting the type of 'self' (line 396)
        self_124973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 396)
        fp_124974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_124973, 'fp')
        # Obtaining the member 'write' of a type (line 396)
        write_124975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), fp_124974, 'write')
        # Calling write(args, kwargs) (line 396)
        write_call_result_124978 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), write_124975, *[str_124976], **kwargs_124977)
        
        
        # Call to write(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Call to tostring(...): (line 397)
        # Processing the call keyword arguments (line 397)
        kwargs_124989 = {}
        
        # Call to array(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'self' (line 397)
        self_124983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 28), 'self', False)
        # Obtaining the member 'version_byte' of a type (line 397)
        version_byte_124984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 28), self_124983, 'version_byte')
        str_124985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 47), 'str', '>b')
        # Processing the call keyword arguments (line 397)
        kwargs_124986 = {}
        # Getting the type of 'array' (line 397)
        array_124982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'array', False)
        # Calling array(args, kwargs) (line 397)
        array_call_result_124987 = invoke(stypy.reporting.localization.Localization(__file__, 397, 22), array_124982, *[version_byte_124984, str_124985], **kwargs_124986)
        
        # Obtaining the member 'tostring' of a type (line 397)
        tostring_124988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 22), array_call_result_124987, 'tostring')
        # Calling tostring(args, kwargs) (line 397)
        tostring_call_result_124990 = invoke(stypy.reporting.localization.Localization(__file__, 397, 22), tostring_124988, *[], **kwargs_124989)
        
        # Processing the call keyword arguments (line 397)
        kwargs_124991 = {}
        # Getting the type of 'self' (line 397)
        self_124979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 397)
        fp_124980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_124979, 'fp')
        # Obtaining the member 'write' of a type (line 397)
        write_124981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), fp_124980, 'write')
        # Calling write(args, kwargs) (line 397)
        write_call_result_124992 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), write_124981, *[tostring_call_result_124990], **kwargs_124991)
        
        
        # Call to _write_numrecs(...): (line 400)
        # Processing the call keyword arguments (line 400)
        kwargs_124995 = {}
        # Getting the type of 'self' (line 400)
        self_124993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'self', False)
        # Obtaining the member '_write_numrecs' of a type (line 400)
        _write_numrecs_124994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), self_124993, '_write_numrecs')
        # Calling _write_numrecs(args, kwargs) (line 400)
        _write_numrecs_call_result_124996 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), _write_numrecs_124994, *[], **kwargs_124995)
        
        
        # Call to _write_dim_array(...): (line 401)
        # Processing the call keyword arguments (line 401)
        kwargs_124999 = {}
        # Getting the type of 'self' (line 401)
        self_124997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'self', False)
        # Obtaining the member '_write_dim_array' of a type (line 401)
        _write_dim_array_124998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), self_124997, '_write_dim_array')
        # Calling _write_dim_array(args, kwargs) (line 401)
        _write_dim_array_call_result_125000 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), _write_dim_array_124998, *[], **kwargs_124999)
        
        
        # Call to _write_gatt_array(...): (line 402)
        # Processing the call keyword arguments (line 402)
        kwargs_125003 = {}
        # Getting the type of 'self' (line 402)
        self_125001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self', False)
        # Obtaining the member '_write_gatt_array' of a type (line 402)
        _write_gatt_array_125002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_125001, '_write_gatt_array')
        # Calling _write_gatt_array(args, kwargs) (line 402)
        _write_gatt_array_call_result_125004 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), _write_gatt_array_125002, *[], **kwargs_125003)
        
        
        # Call to _write_var_array(...): (line 403)
        # Processing the call keyword arguments (line 403)
        kwargs_125007 = {}
        # Getting the type of 'self' (line 403)
        self_125005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self', False)
        # Obtaining the member '_write_var_array' of a type (line 403)
        _write_var_array_125006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_125005, '_write_var_array')
        # Calling _write_var_array(args, kwargs) (line 403)
        _write_var_array_call_result_125008 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), _write_var_array_125006, *[], **kwargs_125007)
        
        
        # ################# End of '_write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_125009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write'
        return stypy_return_type_125009


    @norecursion
    def _write_numrecs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_numrecs'
        module_type_store = module_type_store.open_function_context('_write_numrecs', 405, 4, False)
        # Assigning a type to the variable 'self' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_numrecs')
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_numrecs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_numrecs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_numrecs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_numrecs(...)' code ##################

        
        
        # Call to values(...): (line 407)
        # Processing the call keyword arguments (line 407)
        kwargs_125013 = {}
        # Getting the type of 'self' (line 407)
        self_125010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 19), 'self', False)
        # Obtaining the member 'variables' of a type (line 407)
        variables_125011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 19), self_125010, 'variables')
        # Obtaining the member 'values' of a type (line 407)
        values_125012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 19), variables_125011, 'values')
        # Calling values(args, kwargs) (line 407)
        values_call_result_125014 = invoke(stypy.reporting.localization.Localization(__file__, 407, 19), values_125012, *[], **kwargs_125013)
        
        # Testing the type of a for loop iterable (line 407)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 407, 8), values_call_result_125014)
        # Getting the type of the for loop variable (line 407)
        for_loop_var_125015 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 407, 8), values_call_result_125014)
        # Assigning a type to the variable 'var' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'var', for_loop_var_125015)
        # SSA begins for a for statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'var' (line 408)
        var_125016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'var')
        # Obtaining the member 'isrec' of a type (line 408)
        isrec_125017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), var_125016, 'isrec')
        
        
        # Call to len(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'var' (line 408)
        var_125019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 33), 'var', False)
        # Obtaining the member 'data' of a type (line 408)
        data_125020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 33), var_125019, 'data')
        # Processing the call keyword arguments (line 408)
        kwargs_125021 = {}
        # Getting the type of 'len' (line 408)
        len_125018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 29), 'len', False)
        # Calling len(args, kwargs) (line 408)
        len_call_result_125022 = invoke(stypy.reporting.localization.Localization(__file__, 408, 29), len_125018, *[data_125020], **kwargs_125021)
        
        # Getting the type of 'self' (line 408)
        self_125023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 45), 'self')
        # Obtaining the member '_recs' of a type (line 408)
        _recs_125024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 45), self_125023, '_recs')
        # Applying the binary operator '>' (line 408)
        result_gt_125025 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 29), '>', len_call_result_125022, _recs_125024)
        
        # Applying the binary operator 'and' (line 408)
        result_and_keyword_125026 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 15), 'and', isrec_125017, result_gt_125025)
        
        # Testing the type of an if condition (line 408)
        if_condition_125027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 12), result_and_keyword_125026)
        # Assigning a type to the variable 'if_condition_125027' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'if_condition_125027', if_condition_125027)
        # SSA begins for if statement (line 408)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 409):
        
        # Assigning a Call to a Subscript (line 409):
        
        # Call to len(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'var' (line 409)
        var_125029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 45), 'var', False)
        # Obtaining the member 'data' of a type (line 409)
        data_125030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 45), var_125029, 'data')
        # Processing the call keyword arguments (line 409)
        kwargs_125031 = {}
        # Getting the type of 'len' (line 409)
        len_125028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 41), 'len', False)
        # Calling len(args, kwargs) (line 409)
        len_call_result_125032 = invoke(stypy.reporting.localization.Localization(__file__, 409, 41), len_125028, *[data_125030], **kwargs_125031)
        
        # Getting the type of 'self' (line 409)
        self_125033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), 'self')
        # Obtaining the member '__dict__' of a type (line 409)
        dict___125034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 16), self_125033, '__dict__')
        str_125035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 30), 'str', '_recs')
        # Storing an element on a container (line 409)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 16), dict___125034, (str_125035, len_call_result_125032))
        # SSA join for if statement (line 408)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _pack_int(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'self' (line 410)
        self_125038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 23), 'self', False)
        # Obtaining the member '_recs' of a type (line 410)
        _recs_125039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 23), self_125038, '_recs')
        # Processing the call keyword arguments (line 410)
        kwargs_125040 = {}
        # Getting the type of 'self' (line 410)
        self_125036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 410)
        _pack_int_125037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_125036, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 410)
        _pack_int_call_result_125041 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), _pack_int_125037, *[_recs_125039], **kwargs_125040)
        
        
        # ################# End of '_write_numrecs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_numrecs' in the type store
        # Getting the type of 'stypy_return_type' (line 405)
        stypy_return_type_125042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_numrecs'
        return stypy_return_type_125042


    @norecursion
    def _write_dim_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_dim_array'
        module_type_store = module_type_store.open_function_context('_write_dim_array', 412, 4, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_dim_array')
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_dim_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_dim_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_dim_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_dim_array(...)' code ##################

        
        # Getting the type of 'self' (line 413)
        self_125043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'self')
        # Obtaining the member 'dimensions' of a type (line 413)
        dimensions_125044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 11), self_125043, 'dimensions')
        # Testing the type of an if condition (line 413)
        if_condition_125045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 8), dimensions_125044)
        # Assigning a type to the variable 'if_condition_125045' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'if_condition_125045', if_condition_125045)
        # SSA begins for if statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'NC_DIMENSION' (line 414)
        NC_DIMENSION_125049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'NC_DIMENSION', False)
        # Processing the call keyword arguments (line 414)
        kwargs_125050 = {}
        # Getting the type of 'self' (line 414)
        self_125046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 414)
        fp_125047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), self_125046, 'fp')
        # Obtaining the member 'write' of a type (line 414)
        write_125048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), fp_125047, 'write')
        # Calling write(args, kwargs) (line 414)
        write_call_result_125051 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), write_125048, *[NC_DIMENSION_125049], **kwargs_125050)
        
        
        # Call to _pack_int(...): (line 415)
        # Processing the call arguments (line 415)
        
        # Call to len(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'self' (line 415)
        self_125055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 31), 'self', False)
        # Obtaining the member 'dimensions' of a type (line 415)
        dimensions_125056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 31), self_125055, 'dimensions')
        # Processing the call keyword arguments (line 415)
        kwargs_125057 = {}
        # Getting the type of 'len' (line 415)
        len_125054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 27), 'len', False)
        # Calling len(args, kwargs) (line 415)
        len_call_result_125058 = invoke(stypy.reporting.localization.Localization(__file__, 415, 27), len_125054, *[dimensions_125056], **kwargs_125057)
        
        # Processing the call keyword arguments (line 415)
        kwargs_125059 = {}
        # Getting the type of 'self' (line 415)
        self_125052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 415)
        _pack_int_125053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 12), self_125052, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 415)
        _pack_int_call_result_125060 = invoke(stypy.reporting.localization.Localization(__file__, 415, 12), _pack_int_125053, *[len_call_result_125058], **kwargs_125059)
        
        
        # Getting the type of 'self' (line 416)
        self_125061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'self')
        # Obtaining the member '_dims' of a type (line 416)
        _dims_125062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 24), self_125061, '_dims')
        # Testing the type of a for loop iterable (line 416)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 416, 12), _dims_125062)
        # Getting the type of the for loop variable (line 416)
        for_loop_var_125063 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 416, 12), _dims_125062)
        # Assigning a type to the variable 'name' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'name', for_loop_var_125063)
        # SSA begins for a for statement (line 416)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _pack_string(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'name' (line 417)
        name_125066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 34), 'name', False)
        # Processing the call keyword arguments (line 417)
        kwargs_125067 = {}
        # Getting the type of 'self' (line 417)
        self_125064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 16), 'self', False)
        # Obtaining the member '_pack_string' of a type (line 417)
        _pack_string_125065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 16), self_125064, '_pack_string')
        # Calling _pack_string(args, kwargs) (line 417)
        _pack_string_call_result_125068 = invoke(stypy.reporting.localization.Localization(__file__, 417, 16), _pack_string_125065, *[name_125066], **kwargs_125067)
        
        
        # Assigning a Subscript to a Name (line 418):
        
        # Assigning a Subscript to a Name (line 418):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 418)
        name_125069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 41), 'name')
        # Getting the type of 'self' (line 418)
        self_125070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 25), 'self')
        # Obtaining the member 'dimensions' of a type (line 418)
        dimensions_125071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 25), self_125070, 'dimensions')
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___125072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 25), dimensions_125071, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 418)
        subscript_call_result_125073 = invoke(stypy.reporting.localization.Localization(__file__, 418, 25), getitem___125072, name_125069)
        
        # Assigning a type to the variable 'length' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'length', subscript_call_result_125073)
        
        # Call to _pack_int(...): (line 419)
        # Processing the call arguments (line 419)
        
        # Evaluating a boolean operation
        # Getting the type of 'length' (line 419)
        length_125076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 31), 'length', False)
        int_125077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 41), 'int')
        # Applying the binary operator 'or' (line 419)
        result_or_keyword_125078 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 31), 'or', length_125076, int_125077)
        
        # Processing the call keyword arguments (line 419)
        kwargs_125079 = {}
        # Getting the type of 'self' (line 419)
        self_125074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 419)
        _pack_int_125075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 16), self_125074, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 419)
        _pack_int_call_result_125080 = invoke(stypy.reporting.localization.Localization(__file__, 419, 16), _pack_int_125075, *[result_or_keyword_125078], **kwargs_125079)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 413)
        module_type_store.open_ssa_branch('else')
        
        # Call to write(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'ABSENT' (line 421)
        ABSENT_125084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 26), 'ABSENT', False)
        # Processing the call keyword arguments (line 421)
        kwargs_125085 = {}
        # Getting the type of 'self' (line 421)
        self_125081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 421)
        fp_125082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 12), self_125081, 'fp')
        # Obtaining the member 'write' of a type (line 421)
        write_125083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 12), fp_125082, 'write')
        # Calling write(args, kwargs) (line 421)
        write_call_result_125086 = invoke(stypy.reporting.localization.Localization(__file__, 421, 12), write_125083, *[ABSENT_125084], **kwargs_125085)
        
        # SSA join for if statement (line 413)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_write_dim_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_dim_array' in the type store
        # Getting the type of 'stypy_return_type' (line 412)
        stypy_return_type_125087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125087)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_dim_array'
        return stypy_return_type_125087


    @norecursion
    def _write_gatt_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_gatt_array'
        module_type_store = module_type_store.open_function_context('_write_gatt_array', 423, 4, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_gatt_array')
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_gatt_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_gatt_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_gatt_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_gatt_array(...)' code ##################

        
        # Call to _write_att_array(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'self' (line 424)
        self_125090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'self', False)
        # Obtaining the member '_attributes' of a type (line 424)
        _attributes_125091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 30), self_125090, '_attributes')
        # Processing the call keyword arguments (line 424)
        kwargs_125092 = {}
        # Getting the type of 'self' (line 424)
        self_125088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'self', False)
        # Obtaining the member '_write_att_array' of a type (line 424)
        _write_att_array_125089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), self_125088, '_write_att_array')
        # Calling _write_att_array(args, kwargs) (line 424)
        _write_att_array_call_result_125093 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), _write_att_array_125089, *[_attributes_125091], **kwargs_125092)
        
        
        # ################# End of '_write_gatt_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_gatt_array' in the type store
        # Getting the type of 'stypy_return_type' (line 423)
        stypy_return_type_125094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_gatt_array'
        return stypy_return_type_125094


    @norecursion
    def _write_att_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_att_array'
        module_type_store = module_type_store.open_function_context('_write_att_array', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_att_array')
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_param_names_list', ['attributes'])
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_att_array.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_att_array', ['attributes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_att_array', localization, ['attributes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_att_array(...)' code ##################

        
        # Getting the type of 'attributes' (line 427)
        attributes_125095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 11), 'attributes')
        # Testing the type of an if condition (line 427)
        if_condition_125096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 8), attributes_125095)
        # Assigning a type to the variable 'if_condition_125096' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'if_condition_125096', if_condition_125096)
        # SSA begins for if statement (line 427)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'NC_ATTRIBUTE' (line 428)
        NC_ATTRIBUTE_125100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'NC_ATTRIBUTE', False)
        # Processing the call keyword arguments (line 428)
        kwargs_125101 = {}
        # Getting the type of 'self' (line 428)
        self_125097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 428)
        fp_125098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 12), self_125097, 'fp')
        # Obtaining the member 'write' of a type (line 428)
        write_125099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 12), fp_125098, 'write')
        # Calling write(args, kwargs) (line 428)
        write_call_result_125102 = invoke(stypy.reporting.localization.Localization(__file__, 428, 12), write_125099, *[NC_ATTRIBUTE_125100], **kwargs_125101)
        
        
        # Call to _pack_int(...): (line 429)
        # Processing the call arguments (line 429)
        
        # Call to len(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'attributes' (line 429)
        attributes_125106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 31), 'attributes', False)
        # Processing the call keyword arguments (line 429)
        kwargs_125107 = {}
        # Getting the type of 'len' (line 429)
        len_125105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 27), 'len', False)
        # Calling len(args, kwargs) (line 429)
        len_call_result_125108 = invoke(stypy.reporting.localization.Localization(__file__, 429, 27), len_125105, *[attributes_125106], **kwargs_125107)
        
        # Processing the call keyword arguments (line 429)
        kwargs_125109 = {}
        # Getting the type of 'self' (line 429)
        self_125103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 429)
        _pack_int_125104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 12), self_125103, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 429)
        _pack_int_call_result_125110 = invoke(stypy.reporting.localization.Localization(__file__, 429, 12), _pack_int_125104, *[len_call_result_125108], **kwargs_125109)
        
        
        
        # Call to items(...): (line 430)
        # Processing the call keyword arguments (line 430)
        kwargs_125113 = {}
        # Getting the type of 'attributes' (line 430)
        attributes_125111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 32), 'attributes', False)
        # Obtaining the member 'items' of a type (line 430)
        items_125112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 32), attributes_125111, 'items')
        # Calling items(args, kwargs) (line 430)
        items_call_result_125114 = invoke(stypy.reporting.localization.Localization(__file__, 430, 32), items_125112, *[], **kwargs_125113)
        
        # Testing the type of a for loop iterable (line 430)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 430, 12), items_call_result_125114)
        # Getting the type of the for loop variable (line 430)
        for_loop_var_125115 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 430, 12), items_call_result_125114)
        # Assigning a type to the variable 'name' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 12), for_loop_var_125115))
        # Assigning a type to the variable 'values' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'values', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 12), for_loop_var_125115))
        # SSA begins for a for statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _pack_string(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'name' (line 431)
        name_125118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 34), 'name', False)
        # Processing the call keyword arguments (line 431)
        kwargs_125119 = {}
        # Getting the type of 'self' (line 431)
        self_125116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'self', False)
        # Obtaining the member '_pack_string' of a type (line 431)
        _pack_string_125117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 16), self_125116, '_pack_string')
        # Calling _pack_string(args, kwargs) (line 431)
        _pack_string_call_result_125120 = invoke(stypy.reporting.localization.Localization(__file__, 431, 16), _pack_string_125117, *[name_125118], **kwargs_125119)
        
        
        # Call to _write_values(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'values' (line 432)
        values_125123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 35), 'values', False)
        # Processing the call keyword arguments (line 432)
        kwargs_125124 = {}
        # Getting the type of 'self' (line 432)
        self_125121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'self', False)
        # Obtaining the member '_write_values' of a type (line 432)
        _write_values_125122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 16), self_125121, '_write_values')
        # Calling _write_values(args, kwargs) (line 432)
        _write_values_call_result_125125 = invoke(stypy.reporting.localization.Localization(__file__, 432, 16), _write_values_125122, *[values_125123], **kwargs_125124)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 427)
        module_type_store.open_ssa_branch('else')
        
        # Call to write(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'ABSENT' (line 434)
        ABSENT_125129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 26), 'ABSENT', False)
        # Processing the call keyword arguments (line 434)
        kwargs_125130 = {}
        # Getting the type of 'self' (line 434)
        self_125126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 434)
        fp_125127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), self_125126, 'fp')
        # Obtaining the member 'write' of a type (line 434)
        write_125128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), fp_125127, 'write')
        # Calling write(args, kwargs) (line 434)
        write_call_result_125131 = invoke(stypy.reporting.localization.Localization(__file__, 434, 12), write_125128, *[ABSENT_125129], **kwargs_125130)
        
        # SSA join for if statement (line 427)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_write_att_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_att_array' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_125132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_att_array'
        return stypy_return_type_125132


    @norecursion
    def _write_var_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_var_array'
        module_type_store = module_type_store.open_function_context('_write_var_array', 436, 4, False)
        # Assigning a type to the variable 'self' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_var_array')
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_var_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_var_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_var_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_var_array(...)' code ##################

        
        # Getting the type of 'self' (line 437)
        self_125133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'self')
        # Obtaining the member 'variables' of a type (line 437)
        variables_125134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 11), self_125133, 'variables')
        # Testing the type of an if condition (line 437)
        if_condition_125135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), variables_125134)
        # Assigning a type to the variable 'if_condition_125135' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'if_condition_125135', if_condition_125135)
        # SSA begins for if statement (line 437)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'NC_VARIABLE' (line 438)
        NC_VARIABLE_125139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 26), 'NC_VARIABLE', False)
        # Processing the call keyword arguments (line 438)
        kwargs_125140 = {}
        # Getting the type of 'self' (line 438)
        self_125136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 438)
        fp_125137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), self_125136, 'fp')
        # Obtaining the member 'write' of a type (line 438)
        write_125138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), fp_125137, 'write')
        # Calling write(args, kwargs) (line 438)
        write_call_result_125141 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), write_125138, *[NC_VARIABLE_125139], **kwargs_125140)
        
        
        # Call to _pack_int(...): (line 439)
        # Processing the call arguments (line 439)
        
        # Call to len(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'self' (line 439)
        self_125145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 31), 'self', False)
        # Obtaining the member 'variables' of a type (line 439)
        variables_125146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 31), self_125145, 'variables')
        # Processing the call keyword arguments (line 439)
        kwargs_125147 = {}
        # Getting the type of 'len' (line 439)
        len_125144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 27), 'len', False)
        # Calling len(args, kwargs) (line 439)
        len_call_result_125148 = invoke(stypy.reporting.localization.Localization(__file__, 439, 27), len_125144, *[variables_125146], **kwargs_125147)
        
        # Processing the call keyword arguments (line 439)
        kwargs_125149 = {}
        # Getting the type of 'self' (line 439)
        self_125142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 439)
        _pack_int_125143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 12), self_125142, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 439)
        _pack_int_call_result_125150 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), _pack_int_125143, *[len_call_result_125148], **kwargs_125149)
        

        @norecursion
        def sortkey(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'sortkey'
            module_type_store = module_type_store.open_function_context('sortkey', 442, 12, False)
            
            # Passed parameters checking function
            sortkey.stypy_localization = localization
            sortkey.stypy_type_of_self = None
            sortkey.stypy_type_store = module_type_store
            sortkey.stypy_function_name = 'sortkey'
            sortkey.stypy_param_names_list = ['n']
            sortkey.stypy_varargs_param_name = None
            sortkey.stypy_kwargs_param_name = None
            sortkey.stypy_call_defaults = defaults
            sortkey.stypy_call_varargs = varargs
            sortkey.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'sortkey', ['n'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'sortkey', localization, ['n'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'sortkey(...)' code ##################

            
            # Assigning a Subscript to a Name (line 443):
            
            # Assigning a Subscript to a Name (line 443):
            
            # Obtaining the type of the subscript
            # Getting the type of 'n' (line 443)
            n_125151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 35), 'n')
            # Getting the type of 'self' (line 443)
            self_125152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'self')
            # Obtaining the member 'variables' of a type (line 443)
            variables_125153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 20), self_125152, 'variables')
            # Obtaining the member '__getitem__' of a type (line 443)
            getitem___125154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 20), variables_125153, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 443)
            subscript_call_result_125155 = invoke(stypy.reporting.localization.Localization(__file__, 443, 20), getitem___125154, n_125151)
            
            # Assigning a type to the variable 'v' (line 443)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'v', subscript_call_result_125155)
            
            # Getting the type of 'v' (line 444)
            v_125156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'v')
            # Obtaining the member 'isrec' of a type (line 444)
            isrec_125157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 19), v_125156, 'isrec')
            # Testing the type of an if condition (line 444)
            if_condition_125158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 16), isrec_125157)
            # Assigning a type to the variable 'if_condition_125158' (line 444)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'if_condition_125158', if_condition_125158)
            # SSA begins for if statement (line 444)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 445)
            tuple_125159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 445)
            # Adding element type (line 445)
            int_125160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 28), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 28), tuple_125159, int_125160)
            
            # Assigning a type to the variable 'stypy_return_type' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'stypy_return_type', tuple_125159)
            # SSA join for if statement (line 444)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'v' (line 446)
            v_125161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 23), 'v')
            # Obtaining the member '_shape' of a type (line 446)
            _shape_125162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 23), v_125161, '_shape')
            # Assigning a type to the variable 'stypy_return_type' (line 446)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'stypy_return_type', _shape_125162)
            
            # ################# End of 'sortkey(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'sortkey' in the type store
            # Getting the type of 'stypy_return_type' (line 442)
            stypy_return_type_125163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_125163)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'sortkey'
            return stypy_return_type_125163

        # Assigning a type to the variable 'sortkey' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'sortkey', sortkey)
        
        # Assigning a Call to a Name (line 447):
        
        # Assigning a Call to a Name (line 447):
        
        # Call to sorted(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'self' (line 447)
        self_125165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 31), 'self', False)
        # Obtaining the member 'variables' of a type (line 447)
        variables_125166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 31), self_125165, 'variables')
        # Processing the call keyword arguments (line 447)
        # Getting the type of 'sortkey' (line 447)
        sortkey_125167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 51), 'sortkey', False)
        keyword_125168 = sortkey_125167
        # Getting the type of 'True' (line 447)
        True_125169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 68), 'True', False)
        keyword_125170 = True_125169
        kwargs_125171 = {'reverse': keyword_125170, 'key': keyword_125168}
        # Getting the type of 'sorted' (line 447)
        sorted_125164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'sorted', False)
        # Calling sorted(args, kwargs) (line 447)
        sorted_call_result_125172 = invoke(stypy.reporting.localization.Localization(__file__, 447, 24), sorted_125164, *[variables_125166], **kwargs_125171)
        
        # Assigning a type to the variable 'variables' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'variables', sorted_call_result_125172)
        
        # Getting the type of 'variables' (line 450)
        variables_125173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'variables')
        # Testing the type of a for loop iterable (line 450)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 450, 12), variables_125173)
        # Getting the type of the for loop variable (line 450)
        for_loop_var_125174 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 450, 12), variables_125173)
        # Assigning a type to the variable 'name' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'name', for_loop_var_125174)
        # SSA begins for a for statement (line 450)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _write_var_metadata(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'name' (line 451)
        name_125177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'name', False)
        # Processing the call keyword arguments (line 451)
        kwargs_125178 = {}
        # Getting the type of 'self' (line 451)
        self_125175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'self', False)
        # Obtaining the member '_write_var_metadata' of a type (line 451)
        _write_var_metadata_125176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 16), self_125175, '_write_var_metadata')
        # Calling _write_var_metadata(args, kwargs) (line 451)
        _write_var_metadata_call_result_125179 = invoke(stypy.reporting.localization.Localization(__file__, 451, 16), _write_var_metadata_125176, *[name_125177], **kwargs_125178)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 454):
        
        # Assigning a Call to a Subscript (line 454):
        
        # Call to sum(...): (line 454)
        # Processing the call arguments (line 454)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to values(...): (line 455)
        # Processing the call keyword arguments (line 455)
        kwargs_125188 = {}
        # Getting the type of 'self' (line 455)
        self_125185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 42), 'self', False)
        # Obtaining the member 'variables' of a type (line 455)
        variables_125186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 42), self_125185, 'variables')
        # Obtaining the member 'values' of a type (line 455)
        values_125187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 42), variables_125186, 'values')
        # Calling values(args, kwargs) (line 455)
        values_call_result_125189 = invoke(stypy.reporting.localization.Localization(__file__, 455, 42), values_125187, *[], **kwargs_125188)
        
        comprehension_125190 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 20), values_call_result_125189)
        # Assigning a type to the variable 'var' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'var', comprehension_125190)
        # Getting the type of 'var' (line 456)
        var_125183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'var', False)
        # Obtaining the member 'isrec' of a type (line 456)
        isrec_125184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 23), var_125183, 'isrec')
        # Getting the type of 'var' (line 455)
        var_125181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'var', False)
        # Obtaining the member '_vsize' of a type (line 455)
        _vsize_125182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 20), var_125181, '_vsize')
        list_125191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 20), list_125191, _vsize_125182)
        # Processing the call keyword arguments (line 454)
        kwargs_125192 = {}
        # Getting the type of 'sum' (line 454)
        sum_125180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 40), 'sum', False)
        # Calling sum(args, kwargs) (line 454)
        sum_call_result_125193 = invoke(stypy.reporting.localization.Localization(__file__, 454, 40), sum_125180, *[list_125191], **kwargs_125192)
        
        # Getting the type of 'self' (line 454)
        self_125194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'self')
        # Obtaining the member '__dict__' of a type (line 454)
        dict___125195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 12), self_125194, '__dict__')
        str_125196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 26), 'str', '_recsize')
        # Storing an element on a container (line 454)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 12), dict___125195, (str_125196, sum_call_result_125193))
        
        # Getting the type of 'variables' (line 458)
        variables_125197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'variables')
        # Testing the type of a for loop iterable (line 458)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 458, 12), variables_125197)
        # Getting the type of the for loop variable (line 458)
        for_loop_var_125198 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 458, 12), variables_125197)
        # Assigning a type to the variable 'name' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'name', for_loop_var_125198)
        # SSA begins for a for statement (line 458)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _write_var_data(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'name' (line 459)
        name_125201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 37), 'name', False)
        # Processing the call keyword arguments (line 459)
        kwargs_125202 = {}
        # Getting the type of 'self' (line 459)
        self_125199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'self', False)
        # Obtaining the member '_write_var_data' of a type (line 459)
        _write_var_data_125200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 16), self_125199, '_write_var_data')
        # Calling _write_var_data(args, kwargs) (line 459)
        _write_var_data_call_result_125203 = invoke(stypy.reporting.localization.Localization(__file__, 459, 16), _write_var_data_125200, *[name_125201], **kwargs_125202)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 437)
        module_type_store.open_ssa_branch('else')
        
        # Call to write(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'ABSENT' (line 461)
        ABSENT_125207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 26), 'ABSENT', False)
        # Processing the call keyword arguments (line 461)
        kwargs_125208 = {}
        # Getting the type of 'self' (line 461)
        self_125204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 461)
        fp_125205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), self_125204, 'fp')
        # Obtaining the member 'write' of a type (line 461)
        write_125206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), fp_125205, 'write')
        # Calling write(args, kwargs) (line 461)
        write_call_result_125209 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), write_125206, *[ABSENT_125207], **kwargs_125208)
        
        # SSA join for if statement (line 437)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_write_var_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_var_array' in the type store
        # Getting the type of 'stypy_return_type' (line 436)
        stypy_return_type_125210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125210)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_var_array'
        return stypy_return_type_125210


    @norecursion
    def _write_var_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_var_metadata'
        module_type_store = module_type_store.open_function_context('_write_var_metadata', 463, 4, False)
        # Assigning a type to the variable 'self' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_var_metadata')
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_param_names_list', ['name'])
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_var_metadata.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_var_metadata', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_var_metadata', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_var_metadata(...)' code ##################

        
        # Assigning a Subscript to a Name (line 464):
        
        # Assigning a Subscript to a Name (line 464):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 464)
        name_125211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 29), 'name')
        # Getting the type of 'self' (line 464)
        self_125212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 14), 'self')
        # Obtaining the member 'variables' of a type (line 464)
        variables_125213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 14), self_125212, 'variables')
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___125214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 14), variables_125213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 464)
        subscript_call_result_125215 = invoke(stypy.reporting.localization.Localization(__file__, 464, 14), getitem___125214, name_125211)
        
        # Assigning a type to the variable 'var' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'var', subscript_call_result_125215)
        
        # Call to _pack_string(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'name' (line 466)
        name_125218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 26), 'name', False)
        # Processing the call keyword arguments (line 466)
        kwargs_125219 = {}
        # Getting the type of 'self' (line 466)
        self_125216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'self', False)
        # Obtaining the member '_pack_string' of a type (line 466)
        _pack_string_125217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), self_125216, '_pack_string')
        # Calling _pack_string(args, kwargs) (line 466)
        _pack_string_call_result_125220 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), _pack_string_125217, *[name_125218], **kwargs_125219)
        
        
        # Call to _pack_int(...): (line 467)
        # Processing the call arguments (line 467)
        
        # Call to len(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'var' (line 467)
        var_125224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 27), 'var', False)
        # Obtaining the member 'dimensions' of a type (line 467)
        dimensions_125225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 27), var_125224, 'dimensions')
        # Processing the call keyword arguments (line 467)
        kwargs_125226 = {}
        # Getting the type of 'len' (line 467)
        len_125223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 23), 'len', False)
        # Calling len(args, kwargs) (line 467)
        len_call_result_125227 = invoke(stypy.reporting.localization.Localization(__file__, 467, 23), len_125223, *[dimensions_125225], **kwargs_125226)
        
        # Processing the call keyword arguments (line 467)
        kwargs_125228 = {}
        # Getting the type of 'self' (line 467)
        self_125221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 467)
        _pack_int_125222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), self_125221, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 467)
        _pack_int_call_result_125229 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), _pack_int_125222, *[len_call_result_125227], **kwargs_125228)
        
        
        # Getting the type of 'var' (line 468)
        var_125230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 23), 'var')
        # Obtaining the member 'dimensions' of a type (line 468)
        dimensions_125231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 23), var_125230, 'dimensions')
        # Testing the type of a for loop iterable (line 468)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 468, 8), dimensions_125231)
        # Getting the type of the for loop variable (line 468)
        for_loop_var_125232 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 468, 8), dimensions_125231)
        # Assigning a type to the variable 'dimname' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'dimname', for_loop_var_125232)
        # SSA begins for a for statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 469):
        
        # Assigning a Call to a Name (line 469):
        
        # Call to index(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'dimname' (line 469)
        dimname_125236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 37), 'dimname', False)
        # Processing the call keyword arguments (line 469)
        kwargs_125237 = {}
        # Getting the type of 'self' (line 469)
        self_125233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 20), 'self', False)
        # Obtaining the member '_dims' of a type (line 469)
        _dims_125234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 20), self_125233, '_dims')
        # Obtaining the member 'index' of a type (line 469)
        index_125235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 20), _dims_125234, 'index')
        # Calling index(args, kwargs) (line 469)
        index_call_result_125238 = invoke(stypy.reporting.localization.Localization(__file__, 469, 20), index_125235, *[dimname_125236], **kwargs_125237)
        
        # Assigning a type to the variable 'dimid' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'dimid', index_call_result_125238)
        
        # Call to _pack_int(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'dimid' (line 470)
        dimid_125241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 27), 'dimid', False)
        # Processing the call keyword arguments (line 470)
        kwargs_125242 = {}
        # Getting the type of 'self' (line 470)
        self_125239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 470)
        _pack_int_125240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), self_125239, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 470)
        _pack_int_call_result_125243 = invoke(stypy.reporting.localization.Localization(__file__, 470, 12), _pack_int_125240, *[dimid_125241], **kwargs_125242)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _write_att_array(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'var' (line 472)
        var_125246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 30), 'var', False)
        # Obtaining the member '_attributes' of a type (line 472)
        _attributes_125247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 30), var_125246, '_attributes')
        # Processing the call keyword arguments (line 472)
        kwargs_125248 = {}
        # Getting the type of 'self' (line 472)
        self_125244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'self', False)
        # Obtaining the member '_write_att_array' of a type (line 472)
        _write_att_array_125245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), self_125244, '_write_att_array')
        # Calling _write_att_array(args, kwargs) (line 472)
        _write_att_array_call_result_125249 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), _write_att_array_125245, *[_attributes_125247], **kwargs_125248)
        
        
        # Assigning a Subscript to a Name (line 474):
        
        # Assigning a Subscript to a Name (line 474):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 474)
        tuple_125250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 474)
        # Adding element type (line 474)
        
        # Call to typecode(...): (line 474)
        # Processing the call keyword arguments (line 474)
        kwargs_125253 = {}
        # Getting the type of 'var' (line 474)
        var_125251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 26), 'var', False)
        # Obtaining the member 'typecode' of a type (line 474)
        typecode_125252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 26), var_125251, 'typecode')
        # Calling typecode(args, kwargs) (line 474)
        typecode_call_result_125254 = invoke(stypy.reporting.localization.Localization(__file__, 474, 26), typecode_125252, *[], **kwargs_125253)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 26), tuple_125250, typecode_call_result_125254)
        # Adding element type (line 474)
        
        # Call to itemsize(...): (line 474)
        # Processing the call keyword arguments (line 474)
        kwargs_125257 = {}
        # Getting the type of 'var' (line 474)
        var_125255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 42), 'var', False)
        # Obtaining the member 'itemsize' of a type (line 474)
        itemsize_125256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 42), var_125255, 'itemsize')
        # Calling itemsize(args, kwargs) (line 474)
        itemsize_call_result_125258 = invoke(stypy.reporting.localization.Localization(__file__, 474, 42), itemsize_125256, *[], **kwargs_125257)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 26), tuple_125250, itemsize_call_result_125258)
        
        # Getting the type of 'REVERSE' (line 474)
        REVERSE_125259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'REVERSE')
        # Obtaining the member '__getitem__' of a type (line 474)
        getitem___125260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 18), REVERSE_125259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 474)
        subscript_call_result_125261 = invoke(stypy.reporting.localization.Localization(__file__, 474, 18), getitem___125260, tuple_125250)
        
        # Assigning a type to the variable 'nc_type' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'nc_type', subscript_call_result_125261)
        
        # Call to write(...): (line 475)
        # Processing the call arguments (line 475)
        
        # Call to asbytes(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'nc_type' (line 475)
        nc_type_125266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 30), 'nc_type', False)
        # Processing the call keyword arguments (line 475)
        kwargs_125267 = {}
        # Getting the type of 'asbytes' (line 475)
        asbytes_125265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 22), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 475)
        asbytes_call_result_125268 = invoke(stypy.reporting.localization.Localization(__file__, 475, 22), asbytes_125265, *[nc_type_125266], **kwargs_125267)
        
        # Processing the call keyword arguments (line 475)
        kwargs_125269 = {}
        # Getting the type of 'self' (line 475)
        self_125262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 475)
        fp_125263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), self_125262, 'fp')
        # Obtaining the member 'write' of a type (line 475)
        write_125264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), fp_125263, 'write')
        # Calling write(args, kwargs) (line 475)
        write_call_result_125270 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), write_125264, *[asbytes_call_result_125268], **kwargs_125269)
        
        
        
        # Getting the type of 'var' (line 477)
        var_125271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'var')
        # Obtaining the member 'isrec' of a type (line 477)
        isrec_125272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 15), var_125271, 'isrec')
        # Applying the 'not' unary operator (line 477)
        result_not__125273 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 11), 'not', isrec_125272)
        
        # Testing the type of an if condition (line 477)
        if_condition_125274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 8), result_not__125273)
        # Assigning a type to the variable 'if_condition_125274' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'if_condition_125274', if_condition_125274)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 478):
        
        # Assigning a BinOp to a Name (line 478):
        # Getting the type of 'var' (line 478)
        var_125275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'var')
        # Obtaining the member 'data' of a type (line 478)
        data_125276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), var_125275, 'data')
        # Obtaining the member 'size' of a type (line 478)
        size_125277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), data_125276, 'size')
        # Getting the type of 'var' (line 478)
        var_125278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 36), 'var')
        # Obtaining the member 'data' of a type (line 478)
        data_125279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 36), var_125278, 'data')
        # Obtaining the member 'itemsize' of a type (line 478)
        itemsize_125280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 36), data_125279, 'itemsize')
        # Applying the binary operator '*' (line 478)
        result_mul_125281 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 20), '*', size_125277, itemsize_125280)
        
        # Assigning a type to the variable 'vsize' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'vsize', result_mul_125281)
        
        # Getting the type of 'vsize' (line 479)
        vsize_125282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'vsize')
        
        # Getting the type of 'vsize' (line 479)
        vsize_125283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 22), 'vsize')
        # Applying the 'usub' unary operator (line 479)
        result___neg___125284 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 21), 'usub', vsize_125283)
        
        int_125285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 30), 'int')
        # Applying the binary operator '%' (line 479)
        result_mod_125286 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 21), '%', result___neg___125284, int_125285)
        
        # Applying the binary operator '+=' (line 479)
        result_iadd_125287 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 12), '+=', vsize_125282, result_mod_125286)
        # Assigning a type to the variable 'vsize' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'vsize', result_iadd_125287)
        
        # SSA branch for the else part of an if statement (line 477)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 481)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a BinOp to a Name (line 482):
        
        # Assigning a BinOp to a Name (line 482):
        
        # Obtaining the type of the subscript
        int_125288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 33), 'int')
        # Getting the type of 'var' (line 482)
        var_125289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 24), 'var')
        # Obtaining the member 'data' of a type (line 482)
        data_125290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 24), var_125289, 'data')
        # Obtaining the member '__getitem__' of a type (line 482)
        getitem___125291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 24), data_125290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 482)
        subscript_call_result_125292 = invoke(stypy.reporting.localization.Localization(__file__, 482, 24), getitem___125291, int_125288)
        
        # Obtaining the member 'size' of a type (line 482)
        size_125293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 24), subscript_call_result_125292, 'size')
        # Getting the type of 'var' (line 482)
        var_125294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 43), 'var')
        # Obtaining the member 'data' of a type (line 482)
        data_125295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 43), var_125294, 'data')
        # Obtaining the member 'itemsize' of a type (line 482)
        itemsize_125296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 43), data_125295, 'itemsize')
        # Applying the binary operator '*' (line 482)
        result_mul_125297 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 24), '*', size_125293, itemsize_125296)
        
        # Assigning a type to the variable 'vsize' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'vsize', result_mul_125297)
        # SSA branch for the except part of a try statement (line 481)
        # SSA branch for the except 'IndexError' branch of a try statement (line 481)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 484):
        
        # Assigning a Num to a Name (line 484):
        int_125298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 24), 'int')
        # Assigning a type to the variable 'vsize' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'vsize', int_125298)
        # SSA join for try-except statement (line 481)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 485):
        
        # Assigning a Call to a Name (line 485):
        
        # Call to len(...): (line 485)
        # Processing the call arguments (line 485)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to values(...): (line 485)
        # Processing the call keyword arguments (line 485)
        kwargs_125306 = {}
        # Getting the type of 'self' (line 485)
        self_125303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 39), 'self', False)
        # Obtaining the member 'variables' of a type (line 485)
        variables_125304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 39), self_125303, 'variables')
        # Obtaining the member 'values' of a type (line 485)
        values_125305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 39), variables_125304, 'values')
        # Calling values(args, kwargs) (line 485)
        values_call_result_125307 = invoke(stypy.reporting.localization.Localization(__file__, 485, 39), values_125305, *[], **kwargs_125306)
        
        comprehension_125308 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 28), values_call_result_125307)
        # Assigning a type to the variable 'v' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 28), 'v', comprehension_125308)
        # Getting the type of 'v' (line 486)
        v_125301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 31), 'v', False)
        # Obtaining the member 'isrec' of a type (line 486)
        isrec_125302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 31), v_125301, 'isrec')
        # Getting the type of 'v' (line 485)
        v_125300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 28), 'v', False)
        list_125309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 28), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 28), list_125309, v_125300)
        # Processing the call keyword arguments (line 485)
        kwargs_125310 = {}
        # Getting the type of 'len' (line 485)
        len_125299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 23), 'len', False)
        # Calling len(args, kwargs) (line 485)
        len_call_result_125311 = invoke(stypy.reporting.localization.Localization(__file__, 485, 23), len_125299, *[list_125309], **kwargs_125310)
        
        # Assigning a type to the variable 'rec_vars' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'rec_vars', len_call_result_125311)
        
        
        # Getting the type of 'rec_vars' (line 487)
        rec_vars_125312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'rec_vars')
        int_125313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 26), 'int')
        # Applying the binary operator '>' (line 487)
        result_gt_125314 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 15), '>', rec_vars_125312, int_125313)
        
        # Testing the type of an if condition (line 487)
        if_condition_125315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 12), result_gt_125314)
        # Assigning a type to the variable 'if_condition_125315' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'if_condition_125315', if_condition_125315)
        # SSA begins for if statement (line 487)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'vsize' (line 488)
        vsize_125316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'vsize')
        
        # Getting the type of 'vsize' (line 488)
        vsize_125317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 26), 'vsize')
        # Applying the 'usub' unary operator (line 488)
        result___neg___125318 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 25), 'usub', vsize_125317)
        
        int_125319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 34), 'int')
        # Applying the binary operator '%' (line 488)
        result_mod_125320 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 25), '%', result___neg___125318, int_125319)
        
        # Applying the binary operator '+=' (line 488)
        result_iadd_125321 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 16), '+=', vsize_125316, result_mod_125320)
        # Assigning a type to the variable 'vsize' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'vsize', result_iadd_125321)
        
        # SSA join for if statement (line 487)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 489):
        
        # Assigning a Name to a Subscript (line 489):
        # Getting the type of 'vsize' (line 489)
        vsize_125322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 50), 'vsize')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 489)
        name_125323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 23), 'name')
        # Getting the type of 'self' (line 489)
        self_125324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'self')
        # Obtaining the member 'variables' of a type (line 489)
        variables_125325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), self_125324, 'variables')
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___125326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), variables_125325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_125327 = invoke(stypy.reporting.localization.Localization(__file__, 489, 8), getitem___125326, name_125323)
        
        # Obtaining the member '__dict__' of a type (line 489)
        dict___125328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), subscript_call_result_125327, '__dict__')
        str_125329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 38), 'str', '_vsize')
        # Storing an element on a container (line 489)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 8), dict___125328, (str_125329, vsize_125322))
        
        # Call to _pack_int(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'vsize' (line 490)
        vsize_125332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'vsize', False)
        # Processing the call keyword arguments (line 490)
        kwargs_125333 = {}
        # Getting the type of 'self' (line 490)
        self_125330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 490)
        _pack_int_125331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), self_125330, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 490)
        _pack_int_call_result_125334 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), _pack_int_125331, *[vsize_125332], **kwargs_125333)
        
        
        # Assigning a Call to a Subscript (line 493):
        
        # Assigning a Call to a Subscript (line 493):
        
        # Call to tell(...): (line 493)
        # Processing the call keyword arguments (line 493)
        kwargs_125338 = {}
        # Getting the type of 'self' (line 493)
        self_125335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 50), 'self', False)
        # Obtaining the member 'fp' of a type (line 493)
        fp_125336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 50), self_125335, 'fp')
        # Obtaining the member 'tell' of a type (line 493)
        tell_125337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 50), fp_125336, 'tell')
        # Calling tell(args, kwargs) (line 493)
        tell_call_result_125339 = invoke(stypy.reporting.localization.Localization(__file__, 493, 50), tell_125337, *[], **kwargs_125338)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 493)
        name_125340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 'name')
        # Getting the type of 'self' (line 493)
        self_125341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'self')
        # Obtaining the member 'variables' of a type (line 493)
        variables_125342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), self_125341, 'variables')
        # Obtaining the member '__getitem__' of a type (line 493)
        getitem___125343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), variables_125342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 493)
        subscript_call_result_125344 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), getitem___125343, name_125340)
        
        # Obtaining the member '__dict__' of a type (line 493)
        dict___125345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), subscript_call_result_125344, '__dict__')
        str_125346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 38), 'str', '_begin')
        # Storing an element on a container (line 493)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 8), dict___125345, (str_125346, tell_call_result_125339))
        
        # Call to _pack_begin(...): (line 494)
        # Processing the call arguments (line 494)
        int_125349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 25), 'int')
        # Processing the call keyword arguments (line 494)
        kwargs_125350 = {}
        # Getting the type of 'self' (line 494)
        self_125347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'self', False)
        # Obtaining the member '_pack_begin' of a type (line 494)
        _pack_begin_125348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), self_125347, '_pack_begin')
        # Calling _pack_begin(args, kwargs) (line 494)
        _pack_begin_call_result_125351 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), _pack_begin_125348, *[int_125349], **kwargs_125350)
        
        
        # ################# End of '_write_var_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_var_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 463)
        stypy_return_type_125352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125352)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_var_metadata'
        return stypy_return_type_125352


    @norecursion
    def _write_var_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_var_data'
        module_type_store = module_type_store.open_function_context('_write_var_data', 496, 4, False)
        # Assigning a type to the variable 'self' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_var_data')
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_param_names_list', ['name'])
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_var_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_var_data', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_var_data', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_var_data(...)' code ##################

        
        # Assigning a Subscript to a Name (line 497):
        
        # Assigning a Subscript to a Name (line 497):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 497)
        name_125353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 29), 'name')
        # Getting the type of 'self' (line 497)
        self_125354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 14), 'self')
        # Obtaining the member 'variables' of a type (line 497)
        variables_125355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 14), self_125354, 'variables')
        # Obtaining the member '__getitem__' of a type (line 497)
        getitem___125356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 14), variables_125355, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 497)
        subscript_call_result_125357 = invoke(stypy.reporting.localization.Localization(__file__, 497, 14), getitem___125356, name_125353)
        
        # Assigning a type to the variable 'var' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'var', subscript_call_result_125357)
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to tell(...): (line 500)
        # Processing the call keyword arguments (line 500)
        kwargs_125361 = {}
        # Getting the type of 'self' (line 500)
        self_125358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 22), 'self', False)
        # Obtaining the member 'fp' of a type (line 500)
        fp_125359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 22), self_125358, 'fp')
        # Obtaining the member 'tell' of a type (line 500)
        tell_125360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 22), fp_125359, 'tell')
        # Calling tell(args, kwargs) (line 500)
        tell_call_result_125362 = invoke(stypy.reporting.localization.Localization(__file__, 500, 22), tell_125360, *[], **kwargs_125361)
        
        # Assigning a type to the variable 'the_beguine' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'the_beguine', tell_call_result_125362)
        
        # Call to seek(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'var' (line 501)
        var_125366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 21), 'var', False)
        # Obtaining the member '_begin' of a type (line 501)
        _begin_125367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 21), var_125366, '_begin')
        # Processing the call keyword arguments (line 501)
        kwargs_125368 = {}
        # Getting the type of 'self' (line 501)
        self_125363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 501)
        fp_125364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), self_125363, 'fp')
        # Obtaining the member 'seek' of a type (line 501)
        seek_125365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), fp_125364, 'seek')
        # Calling seek(args, kwargs) (line 501)
        seek_call_result_125369 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), seek_125365, *[_begin_125367], **kwargs_125368)
        
        
        # Call to _pack_begin(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'the_beguine' (line 502)
        the_beguine_125372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'the_beguine', False)
        # Processing the call keyword arguments (line 502)
        kwargs_125373 = {}
        # Getting the type of 'self' (line 502)
        self_125370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'self', False)
        # Obtaining the member '_pack_begin' of a type (line 502)
        _pack_begin_125371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), self_125370, '_pack_begin')
        # Calling _pack_begin(args, kwargs) (line 502)
        _pack_begin_call_result_125374 = invoke(stypy.reporting.localization.Localization(__file__, 502, 8), _pack_begin_125371, *[the_beguine_125372], **kwargs_125373)
        
        
        # Call to seek(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'the_beguine' (line 503)
        the_beguine_125378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 21), 'the_beguine', False)
        # Processing the call keyword arguments (line 503)
        kwargs_125379 = {}
        # Getting the type of 'self' (line 503)
        self_125375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 503)
        fp_125376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), self_125375, 'fp')
        # Obtaining the member 'seek' of a type (line 503)
        seek_125377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), fp_125376, 'seek')
        # Calling seek(args, kwargs) (line 503)
        seek_call_result_125380 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), seek_125377, *[the_beguine_125378], **kwargs_125379)
        
        
        
        # Getting the type of 'var' (line 506)
        var_125381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), 'var')
        # Obtaining the member 'isrec' of a type (line 506)
        isrec_125382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 15), var_125381, 'isrec')
        # Applying the 'not' unary operator (line 506)
        result_not__125383 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 11), 'not', isrec_125382)
        
        # Testing the type of an if condition (line 506)
        if_condition_125384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 506, 8), result_not__125383)
        # Assigning a type to the variable 'if_condition_125384' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'if_condition_125384', if_condition_125384)
        # SSA begins for if statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 507)
        # Processing the call arguments (line 507)
        
        # Call to tostring(...): (line 507)
        # Processing the call keyword arguments (line 507)
        kwargs_125391 = {}
        # Getting the type of 'var' (line 507)
        var_125388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 26), 'var', False)
        # Obtaining the member 'data' of a type (line 507)
        data_125389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 26), var_125388, 'data')
        # Obtaining the member 'tostring' of a type (line 507)
        tostring_125390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 26), data_125389, 'tostring')
        # Calling tostring(args, kwargs) (line 507)
        tostring_call_result_125392 = invoke(stypy.reporting.localization.Localization(__file__, 507, 26), tostring_125390, *[], **kwargs_125391)
        
        # Processing the call keyword arguments (line 507)
        kwargs_125393 = {}
        # Getting the type of 'self' (line 507)
        self_125385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 507)
        fp_125386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 12), self_125385, 'fp')
        # Obtaining the member 'write' of a type (line 507)
        write_125387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 12), fp_125386, 'write')
        # Calling write(args, kwargs) (line 507)
        write_call_result_125394 = invoke(stypy.reporting.localization.Localization(__file__, 507, 12), write_125387, *[tostring_call_result_125392], **kwargs_125393)
        
        
        # Assigning a BinOp to a Name (line 508):
        
        # Assigning a BinOp to a Name (line 508):
        # Getting the type of 'var' (line 508)
        var_125395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 20), 'var')
        # Obtaining the member 'data' of a type (line 508)
        data_125396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 20), var_125395, 'data')
        # Obtaining the member 'size' of a type (line 508)
        size_125397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 20), data_125396, 'size')
        # Getting the type of 'var' (line 508)
        var_125398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 36), 'var')
        # Obtaining the member 'data' of a type (line 508)
        data_125399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 36), var_125398, 'data')
        # Obtaining the member 'itemsize' of a type (line 508)
        itemsize_125400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 36), data_125399, 'itemsize')
        # Applying the binary operator '*' (line 508)
        result_mul_125401 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 20), '*', size_125397, itemsize_125400)
        
        # Assigning a type to the variable 'count' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'count', result_mul_125401)
        
        # Call to write(...): (line 509)
        # Processing the call arguments (line 509)
        str_125405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 26), 'str', '0')
        # Getting the type of 'var' (line 509)
        var_125406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 34), 'var', False)
        # Obtaining the member '_vsize' of a type (line 509)
        _vsize_125407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 34), var_125406, '_vsize')
        # Getting the type of 'count' (line 509)
        count_125408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 47), 'count', False)
        # Applying the binary operator '-' (line 509)
        result_sub_125409 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 34), '-', _vsize_125407, count_125408)
        
        # Applying the binary operator '*' (line 509)
        result_mul_125410 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 26), '*', str_125405, result_sub_125409)
        
        # Processing the call keyword arguments (line 509)
        kwargs_125411 = {}
        # Getting the type of 'self' (line 509)
        self_125402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 509)
        fp_125403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 12), self_125402, 'fp')
        # Obtaining the member 'write' of a type (line 509)
        write_125404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 12), fp_125403, 'write')
        # Calling write(args, kwargs) (line 509)
        write_call_result_125412 = invoke(stypy.reporting.localization.Localization(__file__, 509, 12), write_125404, *[result_mul_125410], **kwargs_125411)
        
        # SSA branch for the else part of an if statement (line 506)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 512)
        self_125413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'self')
        # Obtaining the member '_recs' of a type (line 512)
        _recs_125414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 15), self_125413, '_recs')
        
        # Call to len(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'var' (line 512)
        var_125416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 32), 'var', False)
        # Obtaining the member 'data' of a type (line 512)
        data_125417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 32), var_125416, 'data')
        # Processing the call keyword arguments (line 512)
        kwargs_125418 = {}
        # Getting the type of 'len' (line 512)
        len_125415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 28), 'len', False)
        # Calling len(args, kwargs) (line 512)
        len_call_result_125419 = invoke(stypy.reporting.localization.Localization(__file__, 512, 28), len_125415, *[data_125417], **kwargs_125418)
        
        # Applying the binary operator '>' (line 512)
        result_gt_125420 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 15), '>', _recs_125414, len_call_result_125419)
        
        # Testing the type of an if condition (line 512)
        if_condition_125421 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 12), result_gt_125420)
        # Assigning a type to the variable 'if_condition_125421' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'if_condition_125421', if_condition_125421)
        # SSA begins for if statement (line 512)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 513):
        
        # Assigning a BinOp to a Name (line 513):
        
        # Obtaining an instance of the builtin type 'tuple' (line 513)
        tuple_125422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 513)
        # Adding element type (line 513)
        # Getting the type of 'self' (line 513)
        self_125423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 25), 'self')
        # Obtaining the member '_recs' of a type (line 513)
        _recs_125424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 25), self_125423, '_recs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 25), tuple_125422, _recs_125424)
        
        
        # Obtaining the type of the subscript
        int_125425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 55), 'int')
        slice_125426 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 513, 40), int_125425, None, None)
        # Getting the type of 'var' (line 513)
        var_125427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 40), 'var')
        # Obtaining the member 'data' of a type (line 513)
        data_125428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 40), var_125427, 'data')
        # Obtaining the member 'shape' of a type (line 513)
        shape_125429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 40), data_125428, 'shape')
        # Obtaining the member '__getitem__' of a type (line 513)
        getitem___125430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 40), shape_125429, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 513)
        subscript_call_result_125431 = invoke(stypy.reporting.localization.Localization(__file__, 513, 40), getitem___125430, slice_125426)
        
        # Applying the binary operator '+' (line 513)
        result_add_125432 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 24), '+', tuple_125422, subscript_call_result_125431)
        
        # Assigning a type to the variable 'shape' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'shape', result_add_125432)
        
        
        # SSA begins for try-except statement (line 516)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to resize(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'shape' (line 517)
        shape_125436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 36), 'shape', False)
        # Processing the call keyword arguments (line 517)
        kwargs_125437 = {}
        # Getting the type of 'var' (line 517)
        var_125433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'var', False)
        # Obtaining the member 'data' of a type (line 517)
        data_125434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 20), var_125433, 'data')
        # Obtaining the member 'resize' of a type (line 517)
        resize_125435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 20), data_125434, 'resize')
        # Calling resize(args, kwargs) (line 517)
        resize_call_result_125438 = invoke(stypy.reporting.localization.Localization(__file__, 517, 20), resize_125435, *[shape_125436], **kwargs_125437)
        
        # SSA branch for the except part of a try statement (line 516)
        # SSA branch for the except 'ValueError' branch of a try statement (line 516)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Subscript (line 519):
        
        # Assigning a Call to a Subscript (line 519):
        
        # Call to astype(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'var' (line 519)
        var_125447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 77), 'var', False)
        # Obtaining the member 'data' of a type (line 519)
        data_125448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 77), var_125447, 'data')
        # Obtaining the member 'dtype' of a type (line 519)
        dtype_125449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 77), data_125448, 'dtype')
        # Processing the call keyword arguments (line 519)
        kwargs_125450 = {}
        
        # Call to resize(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'var' (line 519)
        var_125441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 53), 'var', False)
        # Obtaining the member 'data' of a type (line 519)
        data_125442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 53), var_125441, 'data')
        # Getting the type of 'shape' (line 519)
        shape_125443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 63), 'shape', False)
        # Processing the call keyword arguments (line 519)
        kwargs_125444 = {}
        # Getting the type of 'np' (line 519)
        np_125439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 43), 'np', False)
        # Obtaining the member 'resize' of a type (line 519)
        resize_125440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 43), np_125439, 'resize')
        # Calling resize(args, kwargs) (line 519)
        resize_call_result_125445 = invoke(stypy.reporting.localization.Localization(__file__, 519, 43), resize_125440, *[data_125442, shape_125443], **kwargs_125444)
        
        # Obtaining the member 'astype' of a type (line 519)
        astype_125446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 43), resize_call_result_125445, 'astype')
        # Calling astype(args, kwargs) (line 519)
        astype_call_result_125451 = invoke(stypy.reporting.localization.Localization(__file__, 519, 43), astype_125446, *[dtype_125449], **kwargs_125450)
        
        # Getting the type of 'var' (line 519)
        var_125452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 20), 'var')
        # Obtaining the member '__dict__' of a type (line 519)
        dict___125453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 20), var_125452, '__dict__')
        str_125454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 33), 'str', 'data')
        # Storing an element on a container (line 519)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 20), dict___125453, (str_125454, astype_call_result_125451))
        # SSA join for try-except statement (line 516)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 512)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Name (line 521):
        
        # Call to tell(...): (line 521)
        # Processing the call keyword arguments (line 521)
        kwargs_125458 = {}
        # Getting the type of 'self' (line 521)
        self_125455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 25), 'self', False)
        # Obtaining the member 'fp' of a type (line 521)
        fp_125456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 25), self_125455, 'fp')
        # Obtaining the member 'tell' of a type (line 521)
        tell_125457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 25), fp_125456, 'tell')
        # Calling tell(args, kwargs) (line 521)
        tell_call_result_125459 = invoke(stypy.reporting.localization.Localization(__file__, 521, 25), tell_125457, *[], **kwargs_125458)
        
        # Assigning a type to the variable 'pos' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 'pos', tell_call_result_125459)
        
        # Assigning a Name to a Name (line 521):
        # Getting the type of 'pos' (line 521)
        pos_125460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 'pos')
        # Assigning a type to the variable 'pos0' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'pos0', pos_125460)
        
        # Getting the type of 'var' (line 522)
        var_125461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 23), 'var')
        # Obtaining the member 'data' of a type (line 522)
        data_125462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 23), var_125461, 'data')
        # Testing the type of a for loop iterable (line 522)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 522, 12), data_125462)
        # Getting the type of the for loop variable (line 522)
        for_loop_var_125463 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 522, 12), data_125462)
        # Assigning a type to the variable 'rec' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'rec', for_loop_var_125463)
        # SSA begins for a for statement (line 522)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'rec' (line 526)
        rec_125464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 23), 'rec')
        # Obtaining the member 'shape' of a type (line 526)
        shape_125465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 23), rec_125464, 'shape')
        # Applying the 'not' unary operator (line 526)
        result_not__125466 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 19), 'not', shape_125465)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'rec' (line 526)
        rec_125467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 38), 'rec')
        # Obtaining the member 'dtype' of a type (line 526)
        dtype_125468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 38), rec_125467, 'dtype')
        # Obtaining the member 'byteorder' of a type (line 526)
        byteorder_125469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 38), dtype_125468, 'byteorder')
        str_125470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 61), 'str', '<')
        # Applying the binary operator '==' (line 526)
        result_eq_125471 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 38), '==', byteorder_125469, str_125470)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'rec' (line 527)
        rec_125472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 25), 'rec')
        # Obtaining the member 'dtype' of a type (line 527)
        dtype_125473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 25), rec_125472, 'dtype')
        # Obtaining the member 'byteorder' of a type (line 527)
        byteorder_125474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 25), dtype_125473, 'byteorder')
        str_125475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 48), 'str', '=')
        # Applying the binary operator '==' (line 527)
        result_eq_125476 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 25), '==', byteorder_125474, str_125475)
        
        # Getting the type of 'LITTLE_ENDIAN' (line 527)
        LITTLE_ENDIAN_125477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 56), 'LITTLE_ENDIAN')
        # Applying the binary operator 'and' (line 527)
        result_and_keyword_125478 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 25), 'and', result_eq_125476, LITTLE_ENDIAN_125477)
        
        # Applying the binary operator 'or' (line 526)
        result_or_keyword_125479 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 38), 'or', result_eq_125471, result_and_keyword_125478)
        
        # Applying the binary operator 'and' (line 526)
        result_and_keyword_125480 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 19), 'and', result_not__125466, result_or_keyword_125479)
        
        # Testing the type of an if condition (line 526)
        if_condition_125481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 526, 16), result_and_keyword_125480)
        # Assigning a type to the variable 'if_condition_125481' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'if_condition_125481', if_condition_125481)
        # SSA begins for if statement (line 526)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 528):
        
        # Assigning a Call to a Name (line 528):
        
        # Call to byteswap(...): (line 528)
        # Processing the call keyword arguments (line 528)
        kwargs_125484 = {}
        # Getting the type of 'rec' (line 528)
        rec_125482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 26), 'rec', False)
        # Obtaining the member 'byteswap' of a type (line 528)
        byteswap_125483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 26), rec_125482, 'byteswap')
        # Calling byteswap(args, kwargs) (line 528)
        byteswap_call_result_125485 = invoke(stypy.reporting.localization.Localization(__file__, 528, 26), byteswap_125483, *[], **kwargs_125484)
        
        # Assigning a type to the variable 'rec' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'rec', byteswap_call_result_125485)
        # SSA join for if statement (line 526)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 529)
        # Processing the call arguments (line 529)
        
        # Call to tostring(...): (line 529)
        # Processing the call keyword arguments (line 529)
        kwargs_125491 = {}
        # Getting the type of 'rec' (line 529)
        rec_125489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 30), 'rec', False)
        # Obtaining the member 'tostring' of a type (line 529)
        tostring_125490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 30), rec_125489, 'tostring')
        # Calling tostring(args, kwargs) (line 529)
        tostring_call_result_125492 = invoke(stypy.reporting.localization.Localization(__file__, 529, 30), tostring_125490, *[], **kwargs_125491)
        
        # Processing the call keyword arguments (line 529)
        kwargs_125493 = {}
        # Getting the type of 'self' (line 529)
        self_125486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 16), 'self', False)
        # Obtaining the member 'fp' of a type (line 529)
        fp_125487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 16), self_125486, 'fp')
        # Obtaining the member 'write' of a type (line 529)
        write_125488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 16), fp_125487, 'write')
        # Calling write(args, kwargs) (line 529)
        write_call_result_125494 = invoke(stypy.reporting.localization.Localization(__file__, 529, 16), write_125488, *[tostring_call_result_125492], **kwargs_125493)
        
        
        # Assigning a BinOp to a Name (line 531):
        
        # Assigning a BinOp to a Name (line 531):
        # Getting the type of 'rec' (line 531)
        rec_125495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 24), 'rec')
        # Obtaining the member 'size' of a type (line 531)
        size_125496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 24), rec_125495, 'size')
        # Getting the type of 'rec' (line 531)
        rec_125497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 35), 'rec')
        # Obtaining the member 'itemsize' of a type (line 531)
        itemsize_125498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 35), rec_125497, 'itemsize')
        # Applying the binary operator '*' (line 531)
        result_mul_125499 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 24), '*', size_125496, itemsize_125498)
        
        # Assigning a type to the variable 'count' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'count', result_mul_125499)
        
        # Call to write(...): (line 532)
        # Processing the call arguments (line 532)
        str_125503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 30), 'str', '0')
        # Getting the type of 'var' (line 532)
        var_125504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 38), 'var', False)
        # Obtaining the member '_vsize' of a type (line 532)
        _vsize_125505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 38), var_125504, '_vsize')
        # Getting the type of 'count' (line 532)
        count_125506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 51), 'count', False)
        # Applying the binary operator '-' (line 532)
        result_sub_125507 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 38), '-', _vsize_125505, count_125506)
        
        # Applying the binary operator '*' (line 532)
        result_mul_125508 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 30), '*', str_125503, result_sub_125507)
        
        # Processing the call keyword arguments (line 532)
        kwargs_125509 = {}
        # Getting the type of 'self' (line 532)
        self_125500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'self', False)
        # Obtaining the member 'fp' of a type (line 532)
        fp_125501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 16), self_125500, 'fp')
        # Obtaining the member 'write' of a type (line 532)
        write_125502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 16), fp_125501, 'write')
        # Calling write(args, kwargs) (line 532)
        write_call_result_125510 = invoke(stypy.reporting.localization.Localization(__file__, 532, 16), write_125502, *[result_mul_125508], **kwargs_125509)
        
        
        # Getting the type of 'pos' (line 533)
        pos_125511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'pos')
        # Getting the type of 'self' (line 533)
        self_125512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 23), 'self')
        # Obtaining the member '_recsize' of a type (line 533)
        _recsize_125513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 23), self_125512, '_recsize')
        # Applying the binary operator '+=' (line 533)
        result_iadd_125514 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 16), '+=', pos_125511, _recsize_125513)
        # Assigning a type to the variable 'pos' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'pos', result_iadd_125514)
        
        
        # Call to seek(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'pos' (line 534)
        pos_125518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 29), 'pos', False)
        # Processing the call keyword arguments (line 534)
        kwargs_125519 = {}
        # Getting the type of 'self' (line 534)
        self_125515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'self', False)
        # Obtaining the member 'fp' of a type (line 534)
        fp_125516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 16), self_125515, 'fp')
        # Obtaining the member 'seek' of a type (line 534)
        seek_125517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 16), fp_125516, 'seek')
        # Calling seek(args, kwargs) (line 534)
        seek_call_result_125520 = invoke(stypy.reporting.localization.Localization(__file__, 534, 16), seek_125517, *[pos_125518], **kwargs_125519)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to seek(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'pos0' (line 535)
        pos0_125524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'pos0', False)
        # Getting the type of 'var' (line 535)
        var_125525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 32), 'var', False)
        # Obtaining the member '_vsize' of a type (line 535)
        _vsize_125526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 32), var_125525, '_vsize')
        # Applying the binary operator '+' (line 535)
        result_add_125527 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 25), '+', pos0_125524, _vsize_125526)
        
        # Processing the call keyword arguments (line 535)
        kwargs_125528 = {}
        # Getting the type of 'self' (line 535)
        self_125521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 535)
        fp_125522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 12), self_125521, 'fp')
        # Obtaining the member 'seek' of a type (line 535)
        seek_125523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 12), fp_125522, 'seek')
        # Calling seek(args, kwargs) (line 535)
        seek_call_result_125529 = invoke(stypy.reporting.localization.Localization(__file__, 535, 12), seek_125523, *[result_add_125527], **kwargs_125528)
        
        # SSA join for if statement (line 506)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_write_var_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_var_data' in the type store
        # Getting the type of 'stypy_return_type' (line 496)
        stypy_return_type_125530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125530)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_var_data'
        return stypy_return_type_125530


    @norecursion
    def _write_values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_write_values'
        module_type_store = module_type_store.open_function_context('_write_values', 537, 4, False)
        # Assigning a type to the variable 'self' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._write_values.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._write_values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._write_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._write_values.__dict__.__setitem__('stypy_function_name', 'netcdf_file._write_values')
        netcdf_file._write_values.__dict__.__setitem__('stypy_param_names_list', ['values'])
        netcdf_file._write_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._write_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._write_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._write_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._write_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._write_values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._write_values', ['values'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_write_values', localization, ['values'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_write_values(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 538)
        str_125531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 27), 'str', 'dtype')
        # Getting the type of 'values' (line 538)
        values_125532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 19), 'values')
        
        (may_be_125533, more_types_in_union_125534) = may_provide_member(str_125531, values_125532)

        if may_be_125533:

            if more_types_in_union_125534:
                # Runtime conditional SSA (line 538)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'values' (line 538)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'values', remove_not_member_provider_from_union(values_125532, 'dtype'))
            
            # Assigning a Subscript to a Name (line 539):
            
            # Assigning a Subscript to a Name (line 539):
            
            # Obtaining the type of the subscript
            
            # Obtaining an instance of the builtin type 'tuple' (line 539)
            tuple_125535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 539)
            # Adding element type (line 539)
            # Getting the type of 'values' (line 539)
            values_125536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 30), 'values')
            # Obtaining the member 'dtype' of a type (line 539)
            dtype_125537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 30), values_125536, 'dtype')
            # Obtaining the member 'char' of a type (line 539)
            char_125538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 30), dtype_125537, 'char')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 30), tuple_125535, char_125538)
            # Adding element type (line 539)
            # Getting the type of 'values' (line 539)
            values_125539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 49), 'values')
            # Obtaining the member 'dtype' of a type (line 539)
            dtype_125540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 49), values_125539, 'dtype')
            # Obtaining the member 'itemsize' of a type (line 539)
            itemsize_125541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 49), dtype_125540, 'itemsize')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 30), tuple_125535, itemsize_125541)
            
            # Getting the type of 'REVERSE' (line 539)
            REVERSE_125542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 22), 'REVERSE')
            # Obtaining the member '__getitem__' of a type (line 539)
            getitem___125543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 22), REVERSE_125542, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 539)
            subscript_call_result_125544 = invoke(stypy.reporting.localization.Localization(__file__, 539, 22), getitem___125543, tuple_125535)
            
            # Assigning a type to the variable 'nc_type' (line 539)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'nc_type', subscript_call_result_125544)

            if more_types_in_union_125534:
                # Runtime conditional SSA for else branch (line 538)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_125533) or more_types_in_union_125534):
            # Assigning a type to the variable 'values' (line 538)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'values', remove_member_provider_from_union(values_125532, 'dtype'))
            
            # Assigning a ListComp to a Name (line 541):
            
            # Assigning a ListComp to a Name (line 541):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'integer_types' (line 541)
            integer_types_125548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 42), 'integer_types')
            comprehension_125549 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 21), integer_types_125548)
            # Assigning a type to the variable 't' (line 541)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 21), 't', comprehension_125549)
            
            # Obtaining an instance of the builtin type 'tuple' (line 541)
            tuple_125545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 22), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 541)
            # Adding element type (line 541)
            # Getting the type of 't' (line 541)
            t_125546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 22), 't')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 22), tuple_125545, t_125546)
            # Adding element type (line 541)
            # Getting the type of 'NC_INT' (line 541)
            NC_INT_125547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 25), 'NC_INT')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 22), tuple_125545, NC_INT_125547)
            
            list_125550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 21), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 21), list_125550, tuple_125545)
            # Assigning a type to the variable 'types' (line 541)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'types', list_125550)
            
            # Getting the type of 'types' (line 542)
            types_125551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'types')
            
            # Obtaining an instance of the builtin type 'list' (line 542)
            list_125552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 21), 'list')
            # Adding type elements to the builtin type 'list' instance (line 542)
            # Adding element type (line 542)
            
            # Obtaining an instance of the builtin type 'tuple' (line 543)
            tuple_125553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 543)
            # Adding element type (line 543)
            # Getting the type of 'float' (line 543)
            float_125554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 21), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 21), tuple_125553, float_125554)
            # Adding element type (line 543)
            # Getting the type of 'NC_FLOAT' (line 543)
            NC_FLOAT_125555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 28), 'NC_FLOAT')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 21), tuple_125553, NC_FLOAT_125555)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 21), list_125552, tuple_125553)
            # Adding element type (line 542)
            
            # Obtaining an instance of the builtin type 'tuple' (line 544)
            tuple_125556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 544)
            # Adding element type (line 544)
            # Getting the type of 'str' (line 544)
            str_125557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 21), 'str')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 21), tuple_125556, str_125557)
            # Adding element type (line 544)
            # Getting the type of 'NC_CHAR' (line 544)
            NC_CHAR_125558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 26), 'NC_CHAR')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 21), tuple_125556, NC_CHAR_125558)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 21), list_125552, tuple_125556)
            
            # Applying the binary operator '+=' (line 542)
            result_iadd_125559 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 12), '+=', types_125551, list_125552)
            # Assigning a type to the variable 'types' (line 542)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'types', result_iadd_125559)
            
            
            
            # Evaluating a boolean operation
            
            # Call to isinstance(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'values' (line 547)
            values_125561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 26), 'values', False)
            # Getting the type of 'text_type' (line 547)
            text_type_125562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 34), 'text_type', False)
            # Processing the call keyword arguments (line 547)
            kwargs_125563 = {}
            # Getting the type of 'isinstance' (line 547)
            isinstance_125560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 547)
            isinstance_call_result_125564 = invoke(stypy.reporting.localization.Localization(__file__, 547, 15), isinstance_125560, *[values_125561, text_type_125562], **kwargs_125563)
            
            
            # Call to isinstance(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'values' (line 547)
            values_125566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 59), 'values', False)
            # Getting the type of 'binary_type' (line 547)
            binary_type_125567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 67), 'binary_type', False)
            # Processing the call keyword arguments (line 547)
            kwargs_125568 = {}
            # Getting the type of 'isinstance' (line 547)
            isinstance_125565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 48), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 547)
            isinstance_call_result_125569 = invoke(stypy.reporting.localization.Localization(__file__, 547, 48), isinstance_125565, *[values_125566, binary_type_125567], **kwargs_125568)
            
            # Applying the binary operator 'or' (line 547)
            result_or_keyword_125570 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 15), 'or', isinstance_call_result_125564, isinstance_call_result_125569)
            
            # Testing the type of an if condition (line 547)
            if_condition_125571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 12), result_or_keyword_125570)
            # Assigning a type to the variable 'if_condition_125571' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'if_condition_125571', if_condition_125571)
            # SSA begins for if statement (line 547)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 548):
            
            # Assigning a Name to a Name (line 548):
            # Getting the type of 'values' (line 548)
            values_125572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 25), 'values')
            # Assigning a type to the variable 'sample' (line 548)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'sample', values_125572)
            # SSA branch for the else part of an if statement (line 547)
            module_type_store.open_ssa_branch('else')
            
            
            # SSA begins for try-except statement (line 550)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Name (line 551):
            
            # Assigning a Subscript to a Name (line 551):
            
            # Obtaining the type of the subscript
            int_125573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 36), 'int')
            # Getting the type of 'values' (line 551)
            values_125574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 29), 'values')
            # Obtaining the member '__getitem__' of a type (line 551)
            getitem___125575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 29), values_125574, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 551)
            subscript_call_result_125576 = invoke(stypy.reporting.localization.Localization(__file__, 551, 29), getitem___125575, int_125573)
            
            # Assigning a type to the variable 'sample' (line 551)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'sample', subscript_call_result_125576)
            # SSA branch for the except part of a try statement (line 550)
            # SSA branch for the except 'TypeError' branch of a try statement (line 550)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Name (line 553):
            
            # Assigning a Name to a Name (line 553):
            # Getting the type of 'values' (line 553)
            values_125577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 29), 'values')
            # Assigning a type to the variable 'sample' (line 553)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 20), 'sample', values_125577)
            # SSA join for try-except statement (line 550)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 547)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'types' (line 555)
            types_125578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 35), 'types')
            # Testing the type of a for loop iterable (line 555)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 555, 12), types_125578)
            # Getting the type of the for loop variable (line 555)
            for_loop_var_125579 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 555, 12), types_125578)
            # Assigning a type to the variable 'class_' (line 555)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'class_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 12), for_loop_var_125579))
            # Assigning a type to the variable 'nc_type' (line 555)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'nc_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 555, 12), for_loop_var_125579))
            # SSA begins for a for statement (line 555)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to isinstance(...): (line 556)
            # Processing the call arguments (line 556)
            # Getting the type of 'sample' (line 556)
            sample_125581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 30), 'sample', False)
            # Getting the type of 'class_' (line 556)
            class__125582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 38), 'class_', False)
            # Processing the call keyword arguments (line 556)
            kwargs_125583 = {}
            # Getting the type of 'isinstance' (line 556)
            isinstance_125580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 556)
            isinstance_call_result_125584 = invoke(stypy.reporting.localization.Localization(__file__, 556, 19), isinstance_125580, *[sample_125581, class__125582], **kwargs_125583)
            
            # Testing the type of an if condition (line 556)
            if_condition_125585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 16), isinstance_call_result_125584)
            # Assigning a type to the variable 'if_condition_125585' (line 556)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'if_condition_125585', if_condition_125585)
            # SSA begins for if statement (line 556)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 556)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_125533 and more_types_in_union_125534):
                # SSA join for if statement (line 538)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Subscript to a Tuple (line 559):
        
        # Assigning a Subscript to a Name (line 559):
        
        # Obtaining the type of the subscript
        int_125586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'nc_type' (line 559)
        nc_type_125587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 33), 'nc_type')
        # Getting the type of 'TYPEMAP' (line 559)
        TYPEMAP_125588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 25), 'TYPEMAP')
        # Obtaining the member '__getitem__' of a type (line 559)
        getitem___125589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 25), TYPEMAP_125588, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 559)
        subscript_call_result_125590 = invoke(stypy.reporting.localization.Localization(__file__, 559, 25), getitem___125589, nc_type_125587)
        
        # Obtaining the member '__getitem__' of a type (line 559)
        getitem___125591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), subscript_call_result_125590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 559)
        subscript_call_result_125592 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), getitem___125591, int_125586)
        
        # Assigning a type to the variable 'tuple_var_assignment_124546' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_124546', subscript_call_result_125592)
        
        # Assigning a Subscript to a Name (line 559):
        
        # Obtaining the type of the subscript
        int_125593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'nc_type' (line 559)
        nc_type_125594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 33), 'nc_type')
        # Getting the type of 'TYPEMAP' (line 559)
        TYPEMAP_125595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 25), 'TYPEMAP')
        # Obtaining the member '__getitem__' of a type (line 559)
        getitem___125596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 25), TYPEMAP_125595, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 559)
        subscript_call_result_125597 = invoke(stypy.reporting.localization.Localization(__file__, 559, 25), getitem___125596, nc_type_125594)
        
        # Obtaining the member '__getitem__' of a type (line 559)
        getitem___125598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), subscript_call_result_125597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 559)
        subscript_call_result_125599 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), getitem___125598, int_125593)
        
        # Assigning a type to the variable 'tuple_var_assignment_124547' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_124547', subscript_call_result_125599)
        
        # Assigning a Name to a Name (line 559):
        # Getting the type of 'tuple_var_assignment_124546' (line 559)
        tuple_var_assignment_124546_125600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_124546')
        # Assigning a type to the variable 'typecode' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'typecode', tuple_var_assignment_124546_125600)
        
        # Assigning a Name to a Name (line 559):
        # Getting the type of 'tuple_var_assignment_124547' (line 559)
        tuple_var_assignment_124547_125601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_124547')
        # Assigning a type to the variable 'size' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 18), 'size', tuple_var_assignment_124547_125601)
        
        # Assigning a BinOp to a Name (line 560):
        
        # Assigning a BinOp to a Name (line 560):
        str_125602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 17), 'str', '>%s')
        # Getting the type of 'typecode' (line 560)
        typecode_125603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 25), 'typecode')
        # Applying the binary operator '%' (line 560)
        result_mod_125604 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 17), '%', str_125602, typecode_125603)
        
        # Assigning a type to the variable 'dtype_' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'dtype_', result_mod_125604)
        
        # Assigning a IfExp to a Name (line 562):
        
        # Assigning a IfExp to a Name (line 562):
        
        
        # Getting the type of 'dtype_' (line 562)
        dtype__125605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 24), 'dtype_')
        str_125606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 34), 'str', '>c')
        # Applying the binary operator '==' (line 562)
        result_eq_125607 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 24), '==', dtype__125605, str_125606)
        
        # Testing the type of an if expression (line 562)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 17), result_eq_125607)
        # SSA begins for if expression (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        str_125608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 17), 'str', 'S')
        # SSA branch for the else part of an if expression (line 562)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'dtype_' (line 562)
        dtype__125609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 44), 'dtype_')
        # SSA join for if expression (line 562)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_125610 = union_type.UnionType.add(str_125608, dtype__125609)
        
        # Assigning a type to the variable 'dtype_' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'dtype_', if_exp_125610)
        
        # Assigning a Call to a Name (line 564):
        
        # Assigning a Call to a Name (line 564):
        
        # Call to asarray(...): (line 564)
        # Processing the call arguments (line 564)
        # Getting the type of 'values' (line 564)
        values_125612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 25), 'values', False)
        # Processing the call keyword arguments (line 564)
        # Getting the type of 'dtype_' (line 564)
        dtype__125613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 39), 'dtype_', False)
        keyword_125614 = dtype__125613
        kwargs_125615 = {'dtype': keyword_125614}
        # Getting the type of 'asarray' (line 564)
        asarray_125611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 17), 'asarray', False)
        # Calling asarray(args, kwargs) (line 564)
        asarray_call_result_125616 = invoke(stypy.reporting.localization.Localization(__file__, 564, 17), asarray_125611, *[values_125612], **kwargs_125615)
        
        # Assigning a type to the variable 'values' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'values', asarray_call_result_125616)
        
        # Call to write(...): (line 566)
        # Processing the call arguments (line 566)
        
        # Call to asbytes(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'nc_type' (line 566)
        nc_type_125621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 30), 'nc_type', False)
        # Processing the call keyword arguments (line 566)
        kwargs_125622 = {}
        # Getting the type of 'asbytes' (line 566)
        asbytes_125620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 22), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 566)
        asbytes_call_result_125623 = invoke(stypy.reporting.localization.Localization(__file__, 566, 22), asbytes_125620, *[nc_type_125621], **kwargs_125622)
        
        # Processing the call keyword arguments (line 566)
        kwargs_125624 = {}
        # Getting the type of 'self' (line 566)
        self_125617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 566)
        fp_125618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), self_125617, 'fp')
        # Obtaining the member 'write' of a type (line 566)
        write_125619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), fp_125618, 'write')
        # Calling write(args, kwargs) (line 566)
        write_call_result_125625 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), write_125619, *[asbytes_call_result_125623], **kwargs_125624)
        
        
        
        # Getting the type of 'values' (line 568)
        values_125626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 'values')
        # Obtaining the member 'dtype' of a type (line 568)
        dtype_125627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 11), values_125626, 'dtype')
        # Obtaining the member 'char' of a type (line 568)
        char_125628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 11), dtype_125627, 'char')
        str_125629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 32), 'str', 'S')
        # Applying the binary operator '==' (line 568)
        result_eq_125630 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 11), '==', char_125628, str_125629)
        
        # Testing the type of an if condition (line 568)
        if_condition_125631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), result_eq_125630)
        # Assigning a type to the variable 'if_condition_125631' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_125631', if_condition_125631)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 569):
        
        # Assigning a Attribute to a Name (line 569):
        # Getting the type of 'values' (line 569)
        values_125632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 21), 'values')
        # Obtaining the member 'itemsize' of a type (line 569)
        itemsize_125633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 21), values_125632, 'itemsize')
        # Assigning a type to the variable 'nelems' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'nelems', itemsize_125633)
        # SSA branch for the else part of an if statement (line 568)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 571):
        
        # Assigning a Attribute to a Name (line 571):
        # Getting the type of 'values' (line 571)
        values_125634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 21), 'values')
        # Obtaining the member 'size' of a type (line 571)
        size_125635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 21), values_125634, 'size')
        # Assigning a type to the variable 'nelems' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'nelems', size_125635)
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _pack_int(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'nelems' (line 572)
        nelems_125638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 23), 'nelems', False)
        # Processing the call keyword arguments (line 572)
        kwargs_125639 = {}
        # Getting the type of 'self' (line 572)
        self_125636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 572)
        _pack_int_125637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), self_125636, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 572)
        _pack_int_call_result_125640 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), _pack_int_125637, *[nelems_125638], **kwargs_125639)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'values' (line 574)
        values_125641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 15), 'values')
        # Obtaining the member 'shape' of a type (line 574)
        shape_125642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 15), values_125641, 'shape')
        # Applying the 'not' unary operator (line 574)
        result_not__125643 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), 'not', shape_125642)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'values' (line 574)
        values_125644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 33), 'values')
        # Obtaining the member 'dtype' of a type (line 574)
        dtype_125645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 33), values_125644, 'dtype')
        # Obtaining the member 'byteorder' of a type (line 574)
        byteorder_125646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 33), dtype_125645, 'byteorder')
        str_125647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 59), 'str', '<')
        # Applying the binary operator '==' (line 574)
        result_eq_125648 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 33), '==', byteorder_125646, str_125647)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'values' (line 575)
        values_125649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 17), 'values')
        # Obtaining the member 'dtype' of a type (line 575)
        dtype_125650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 17), values_125649, 'dtype')
        # Obtaining the member 'byteorder' of a type (line 575)
        byteorder_125651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 17), dtype_125650, 'byteorder')
        str_125652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 43), 'str', '=')
        # Applying the binary operator '==' (line 575)
        result_eq_125653 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 17), '==', byteorder_125651, str_125652)
        
        # Getting the type of 'LITTLE_ENDIAN' (line 575)
        LITTLE_ENDIAN_125654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 51), 'LITTLE_ENDIAN')
        # Applying the binary operator 'and' (line 575)
        result_and_keyword_125655 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 17), 'and', result_eq_125653, LITTLE_ENDIAN_125654)
        
        # Applying the binary operator 'or' (line 574)
        result_or_keyword_125656 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 33), 'or', result_eq_125648, result_and_keyword_125655)
        
        # Applying the binary operator 'and' (line 574)
        result_and_keyword_125657 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), 'and', result_not__125643, result_or_keyword_125656)
        
        # Testing the type of an if condition (line 574)
        if_condition_125658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 8), result_and_keyword_125657)
        # Assigning a type to the variable 'if_condition_125658' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'if_condition_125658', if_condition_125658)
        # SSA begins for if statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 576):
        
        # Assigning a Call to a Name (line 576):
        
        # Call to byteswap(...): (line 576)
        # Processing the call keyword arguments (line 576)
        kwargs_125661 = {}
        # Getting the type of 'values' (line 576)
        values_125659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 21), 'values', False)
        # Obtaining the member 'byteswap' of a type (line 576)
        byteswap_125660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 21), values_125659, 'byteswap')
        # Calling byteswap(args, kwargs) (line 576)
        byteswap_call_result_125662 = invoke(stypy.reporting.localization.Localization(__file__, 576, 21), byteswap_125660, *[], **kwargs_125661)
        
        # Assigning a type to the variable 'values' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'values', byteswap_call_result_125662)
        # SSA join for if statement (line 574)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 577)
        # Processing the call arguments (line 577)
        
        # Call to tostring(...): (line 577)
        # Processing the call keyword arguments (line 577)
        kwargs_125668 = {}
        # Getting the type of 'values' (line 577)
        values_125666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 22), 'values', False)
        # Obtaining the member 'tostring' of a type (line 577)
        tostring_125667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 22), values_125666, 'tostring')
        # Calling tostring(args, kwargs) (line 577)
        tostring_call_result_125669 = invoke(stypy.reporting.localization.Localization(__file__, 577, 22), tostring_125667, *[], **kwargs_125668)
        
        # Processing the call keyword arguments (line 577)
        kwargs_125670 = {}
        # Getting the type of 'self' (line 577)
        self_125663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 577)
        fp_125664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), self_125663, 'fp')
        # Obtaining the member 'write' of a type (line 577)
        write_125665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), fp_125664, 'write')
        # Calling write(args, kwargs) (line 577)
        write_call_result_125671 = invoke(stypy.reporting.localization.Localization(__file__, 577, 8), write_125665, *[tostring_call_result_125669], **kwargs_125670)
        
        
        # Assigning a BinOp to a Name (line 578):
        
        # Assigning a BinOp to a Name (line 578):
        # Getting the type of 'values' (line 578)
        values_125672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'values')
        # Obtaining the member 'size' of a type (line 578)
        size_125673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 16), values_125672, 'size')
        # Getting the type of 'values' (line 578)
        values_125674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 30), 'values')
        # Obtaining the member 'itemsize' of a type (line 578)
        itemsize_125675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 30), values_125674, 'itemsize')
        # Applying the binary operator '*' (line 578)
        result_mul_125676 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 16), '*', size_125673, itemsize_125675)
        
        # Assigning a type to the variable 'count' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'count', result_mul_125676)
        
        # Call to write(...): (line 579)
        # Processing the call arguments (line 579)
        str_125680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 22), 'str', '0')
        
        # Getting the type of 'count' (line 579)
        count_125681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 31), 'count', False)
        # Applying the 'usub' unary operator (line 579)
        result___neg___125682 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 30), 'usub', count_125681)
        
        int_125683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 39), 'int')
        # Applying the binary operator '%' (line 579)
        result_mod_125684 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 30), '%', result___neg___125682, int_125683)
        
        # Applying the binary operator '*' (line 579)
        result_mul_125685 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 22), '*', str_125680, result_mod_125684)
        
        # Processing the call keyword arguments (line 579)
        kwargs_125686 = {}
        # Getting the type of 'self' (line 579)
        self_125677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 579)
        fp_125678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), self_125677, 'fp')
        # Obtaining the member 'write' of a type (line 579)
        write_125679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), fp_125678, 'write')
        # Calling write(args, kwargs) (line 579)
        write_call_result_125687 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), write_125679, *[result_mul_125685], **kwargs_125686)
        
        
        # ################# End of '_write_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_write_values' in the type store
        # Getting the type of 'stypy_return_type' (line 537)
        stypy_return_type_125688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125688)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_write_values'
        return stypy_return_type_125688


    @norecursion
    def _read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read'
        module_type_store = module_type_store.open_function_context('_read', 581, 4, False)
        # Assigning a type to the variable 'self' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read')
        netcdf_file._read.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read(...)' code ##################

        
        # Assigning a Call to a Name (line 583):
        
        # Assigning a Call to a Name (line 583):
        
        # Call to read(...): (line 583)
        # Processing the call arguments (line 583)
        int_125692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 29), 'int')
        # Processing the call keyword arguments (line 583)
        kwargs_125693 = {}
        # Getting the type of 'self' (line 583)
        self_125689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 16), 'self', False)
        # Obtaining the member 'fp' of a type (line 583)
        fp_125690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 16), self_125689, 'fp')
        # Obtaining the member 'read' of a type (line 583)
        read_125691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 16), fp_125690, 'read')
        # Calling read(args, kwargs) (line 583)
        read_call_result_125694 = invoke(stypy.reporting.localization.Localization(__file__, 583, 16), read_125691, *[int_125692], **kwargs_125693)
        
        # Assigning a type to the variable 'magic' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'magic', read_call_result_125694)
        
        
        
        # Getting the type of 'magic' (line 584)
        magic_125695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 15), 'magic')
        str_125696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 24), 'str', 'CDF')
        # Applying the binary operator '==' (line 584)
        result_eq_125697 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 15), '==', magic_125695, str_125696)
        
        # Applying the 'not' unary operator (line 584)
        result_not__125698 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 11), 'not', result_eq_125697)
        
        # Testing the type of an if condition (line 584)
        if_condition_125699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 584, 8), result_not__125698)
        # Assigning a type to the variable 'if_condition_125699' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'if_condition_125699', if_condition_125699)
        # SSA begins for if statement (line 584)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 585)
        # Processing the call arguments (line 585)
        str_125701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 28), 'str', 'Error: %s is not a valid NetCDF 3 file')
        # Getting the type of 'self' (line 586)
        self_125702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'self', False)
        # Obtaining the member 'filename' of a type (line 586)
        filename_125703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 28), self_125702, 'filename')
        # Applying the binary operator '%' (line 585)
        result_mod_125704 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 28), '%', str_125701, filename_125703)
        
        # Processing the call keyword arguments (line 585)
        kwargs_125705 = {}
        # Getting the type of 'TypeError' (line 585)
        TypeError_125700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 585)
        TypeError_call_result_125706 = invoke(stypy.reporting.localization.Localization(__file__, 585, 18), TypeError_125700, *[result_mod_125704], **kwargs_125705)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 585, 12), TypeError_call_result_125706, 'raise parameter', BaseException)
        # SSA join for if statement (line 584)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Subscript (line 587):
        
        # Assigning a Subscript to a Subscript (line 587):
        
        # Obtaining the type of the subscript
        int_125707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 74), 'int')
        
        # Call to fromstring(...): (line 587)
        # Processing the call arguments (line 587)
        
        # Call to read(...): (line 587)
        # Processing the call arguments (line 587)
        int_125712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 64), 'int')
        # Processing the call keyword arguments (line 587)
        kwargs_125713 = {}
        # Getting the type of 'self' (line 587)
        self_125709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 51), 'self', False)
        # Obtaining the member 'fp' of a type (line 587)
        fp_125710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 51), self_125709, 'fp')
        # Obtaining the member 'read' of a type (line 587)
        read_125711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 51), fp_125710, 'read')
        # Calling read(args, kwargs) (line 587)
        read_call_result_125714 = invoke(stypy.reporting.localization.Localization(__file__, 587, 51), read_125711, *[int_125712], **kwargs_125713)
        
        str_125715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 68), 'str', '>b')
        # Processing the call keyword arguments (line 587)
        kwargs_125716 = {}
        # Getting the type of 'fromstring' (line 587)
        fromstring_125708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 40), 'fromstring', False)
        # Calling fromstring(args, kwargs) (line 587)
        fromstring_call_result_125717 = invoke(stypy.reporting.localization.Localization(__file__, 587, 40), fromstring_125708, *[read_call_result_125714, str_125715], **kwargs_125716)
        
        # Obtaining the member '__getitem__' of a type (line 587)
        getitem___125718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 40), fromstring_call_result_125717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 587)
        subscript_call_result_125719 = invoke(stypy.reporting.localization.Localization(__file__, 587, 40), getitem___125718, int_125707)
        
        # Getting the type of 'self' (line 587)
        self_125720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 587)
        dict___125721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 8), self_125720, '__dict__')
        str_125722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 22), 'str', 'version_byte')
        # Storing an element on a container (line 587)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 8), dict___125721, (str_125722, subscript_call_result_125719))
        
        # Call to _read_numrecs(...): (line 590)
        # Processing the call keyword arguments (line 590)
        kwargs_125725 = {}
        # Getting the type of 'self' (line 590)
        self_125723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'self', False)
        # Obtaining the member '_read_numrecs' of a type (line 590)
        _read_numrecs_125724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 8), self_125723, '_read_numrecs')
        # Calling _read_numrecs(args, kwargs) (line 590)
        _read_numrecs_call_result_125726 = invoke(stypy.reporting.localization.Localization(__file__, 590, 8), _read_numrecs_125724, *[], **kwargs_125725)
        
        
        # Call to _read_dim_array(...): (line 591)
        # Processing the call keyword arguments (line 591)
        kwargs_125729 = {}
        # Getting the type of 'self' (line 591)
        self_125727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'self', False)
        # Obtaining the member '_read_dim_array' of a type (line 591)
        _read_dim_array_125728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 8), self_125727, '_read_dim_array')
        # Calling _read_dim_array(args, kwargs) (line 591)
        _read_dim_array_call_result_125730 = invoke(stypy.reporting.localization.Localization(__file__, 591, 8), _read_dim_array_125728, *[], **kwargs_125729)
        
        
        # Call to _read_gatt_array(...): (line 592)
        # Processing the call keyword arguments (line 592)
        kwargs_125733 = {}
        # Getting the type of 'self' (line 592)
        self_125731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'self', False)
        # Obtaining the member '_read_gatt_array' of a type (line 592)
        _read_gatt_array_125732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 8), self_125731, '_read_gatt_array')
        # Calling _read_gatt_array(args, kwargs) (line 592)
        _read_gatt_array_call_result_125734 = invoke(stypy.reporting.localization.Localization(__file__, 592, 8), _read_gatt_array_125732, *[], **kwargs_125733)
        
        
        # Call to _read_var_array(...): (line 593)
        # Processing the call keyword arguments (line 593)
        kwargs_125737 = {}
        # Getting the type of 'self' (line 593)
        self_125735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'self', False)
        # Obtaining the member '_read_var_array' of a type (line 593)
        _read_var_array_125736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), self_125735, '_read_var_array')
        # Calling _read_var_array(args, kwargs) (line 593)
        _read_var_array_call_result_125738 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), _read_var_array_125736, *[], **kwargs_125737)
        
        
        # ################# End of '_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read' in the type store
        # Getting the type of 'stypy_return_type' (line 581)
        stypy_return_type_125739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125739)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read'
        return stypy_return_type_125739


    @norecursion
    def _read_numrecs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_numrecs'
        module_type_store = module_type_store.open_function_context('_read_numrecs', 595, 4, False)
        # Assigning a type to the variable 'self' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read_numrecs')
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read_numrecs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read_numrecs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_numrecs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_numrecs(...)' code ##################

        
        # Assigning a Call to a Subscript (line 596):
        
        # Assigning a Call to a Subscript (line 596):
        
        # Call to _unpack_int(...): (line 596)
        # Processing the call keyword arguments (line 596)
        kwargs_125742 = {}
        # Getting the type of 'self' (line 596)
        self_125740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 33), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 596)
        _unpack_int_125741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 33), self_125740, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 596)
        _unpack_int_call_result_125743 = invoke(stypy.reporting.localization.Localization(__file__, 596, 33), _unpack_int_125741, *[], **kwargs_125742)
        
        # Getting the type of 'self' (line 596)
        self_125744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 596)
        dict___125745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 8), self_125744, '__dict__')
        str_125746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 22), 'str', '_recs')
        # Storing an element on a container (line 596)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 596, 8), dict___125745, (str_125746, _unpack_int_call_result_125743))
        
        # ################# End of '_read_numrecs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_numrecs' in the type store
        # Getting the type of 'stypy_return_type' (line 595)
        stypy_return_type_125747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_numrecs'
        return stypy_return_type_125747


    @norecursion
    def _read_dim_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_dim_array'
        module_type_store = module_type_store.open_function_context('_read_dim_array', 598, 4, False)
        # Assigning a type to the variable 'self' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read_dim_array')
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read_dim_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read_dim_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_dim_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_dim_array(...)' code ##################

        
        # Assigning a Call to a Name (line 599):
        
        # Assigning a Call to a Name (line 599):
        
        # Call to read(...): (line 599)
        # Processing the call arguments (line 599)
        int_125751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 30), 'int')
        # Processing the call keyword arguments (line 599)
        kwargs_125752 = {}
        # Getting the type of 'self' (line 599)
        self_125748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'self', False)
        # Obtaining the member 'fp' of a type (line 599)
        fp_125749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 17), self_125748, 'fp')
        # Obtaining the member 'read' of a type (line 599)
        read_125750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 17), fp_125749, 'read')
        # Calling read(args, kwargs) (line 599)
        read_call_result_125753 = invoke(stypy.reporting.localization.Localization(__file__, 599, 17), read_125750, *[int_125751], **kwargs_125752)
        
        # Assigning a type to the variable 'header' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'header', read_call_result_125753)
        
        
        # Getting the type of 'header' (line 600)
        header_125754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 11), 'header')
        
        # Obtaining an instance of the builtin type 'list' (line 600)
        list_125755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 600)
        # Adding element type (line 600)
        # Getting the type of 'ZERO' (line 600)
        ZERO_125756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 26), 'ZERO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 25), list_125755, ZERO_125756)
        # Adding element type (line 600)
        # Getting the type of 'NC_DIMENSION' (line 600)
        NC_DIMENSION_125757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 32), 'NC_DIMENSION')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 25), list_125755, NC_DIMENSION_125757)
        
        # Applying the binary operator 'notin' (line 600)
        result_contains_125758 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 11), 'notin', header_125754, list_125755)
        
        # Testing the type of an if condition (line 600)
        if_condition_125759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 8), result_contains_125758)
        # Assigning a type to the variable 'if_condition_125759' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'if_condition_125759', if_condition_125759)
        # SSA begins for if statement (line 600)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 601)
        # Processing the call arguments (line 601)
        str_125761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 29), 'str', 'Unexpected header.')
        # Processing the call keyword arguments (line 601)
        kwargs_125762 = {}
        # Getting the type of 'ValueError' (line 601)
        ValueError_125760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 601)
        ValueError_call_result_125763 = invoke(stypy.reporting.localization.Localization(__file__, 601, 18), ValueError_125760, *[str_125761], **kwargs_125762)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 601, 12), ValueError_call_result_125763, 'raise parameter', BaseException)
        # SSA join for if statement (line 600)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 602):
        
        # Assigning a Call to a Name (line 602):
        
        # Call to _unpack_int(...): (line 602)
        # Processing the call keyword arguments (line 602)
        kwargs_125766 = {}
        # Getting the type of 'self' (line 602)
        self_125764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 16), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 602)
        _unpack_int_125765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 16), self_125764, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 602)
        _unpack_int_call_result_125767 = invoke(stypy.reporting.localization.Localization(__file__, 602, 16), _unpack_int_125765, *[], **kwargs_125766)
        
        # Assigning a type to the variable 'count' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'count', _unpack_int_call_result_125767)
        
        
        # Call to range(...): (line 604)
        # Processing the call arguments (line 604)
        # Getting the type of 'count' (line 604)
        count_125769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 25), 'count', False)
        # Processing the call keyword arguments (line 604)
        kwargs_125770 = {}
        # Getting the type of 'range' (line 604)
        range_125768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 19), 'range', False)
        # Calling range(args, kwargs) (line 604)
        range_call_result_125771 = invoke(stypy.reporting.localization.Localization(__file__, 604, 19), range_125768, *[count_125769], **kwargs_125770)
        
        # Testing the type of a for loop iterable (line 604)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 604, 8), range_call_result_125771)
        # Getting the type of the for loop variable (line 604)
        for_loop_var_125772 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 604, 8), range_call_result_125771)
        # Assigning a type to the variable 'dim' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'dim', for_loop_var_125772)
        # SSA begins for a for statement (line 604)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 605):
        
        # Assigning a Call to a Name (line 605):
        
        # Call to asstr(...): (line 605)
        # Processing the call arguments (line 605)
        
        # Call to _unpack_string(...): (line 605)
        # Processing the call keyword arguments (line 605)
        kwargs_125776 = {}
        # Getting the type of 'self' (line 605)
        self_125774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 25), 'self', False)
        # Obtaining the member '_unpack_string' of a type (line 605)
        _unpack_string_125775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 25), self_125774, '_unpack_string')
        # Calling _unpack_string(args, kwargs) (line 605)
        _unpack_string_call_result_125777 = invoke(stypy.reporting.localization.Localization(__file__, 605, 25), _unpack_string_125775, *[], **kwargs_125776)
        
        # Processing the call keyword arguments (line 605)
        kwargs_125778 = {}
        # Getting the type of 'asstr' (line 605)
        asstr_125773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 19), 'asstr', False)
        # Calling asstr(args, kwargs) (line 605)
        asstr_call_result_125779 = invoke(stypy.reporting.localization.Localization(__file__, 605, 19), asstr_125773, *[_unpack_string_call_result_125777], **kwargs_125778)
        
        # Assigning a type to the variable 'name' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'name', asstr_call_result_125779)
        
        # Assigning a BoolOp to a Name (line 606):
        
        # Assigning a BoolOp to a Name (line 606):
        
        # Evaluating a boolean operation
        
        # Call to _unpack_int(...): (line 606)
        # Processing the call keyword arguments (line 606)
        kwargs_125782 = {}
        # Getting the type of 'self' (line 606)
        self_125780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 21), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 606)
        _unpack_int_125781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 21), self_125780, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 606)
        _unpack_int_call_result_125783 = invoke(stypy.reporting.localization.Localization(__file__, 606, 21), _unpack_int_125781, *[], **kwargs_125782)
        
        # Getting the type of 'None' (line 606)
        None_125784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 43), 'None')
        # Applying the binary operator 'or' (line 606)
        result_or_keyword_125785 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 21), 'or', _unpack_int_call_result_125783, None_125784)
        
        # Assigning a type to the variable 'length' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'length', result_or_keyword_125785)
        
        # Assigning a Name to a Subscript (line 607):
        
        # Assigning a Name to a Subscript (line 607):
        # Getting the type of 'length' (line 607)
        length_125786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 36), 'length')
        # Getting the type of 'self' (line 607)
        self_125787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'self')
        # Obtaining the member 'dimensions' of a type (line 607)
        dimensions_125788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 12), self_125787, 'dimensions')
        # Getting the type of 'name' (line 607)
        name_125789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 28), 'name')
        # Storing an element on a container (line 607)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 12), dimensions_125788, (name_125789, length_125786))
        
        # Call to append(...): (line 608)
        # Processing the call arguments (line 608)
        # Getting the type of 'name' (line 608)
        name_125793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 30), 'name', False)
        # Processing the call keyword arguments (line 608)
        kwargs_125794 = {}
        # Getting the type of 'self' (line 608)
        self_125790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'self', False)
        # Obtaining the member '_dims' of a type (line 608)
        _dims_125791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 12), self_125790, '_dims')
        # Obtaining the member 'append' of a type (line 608)
        append_125792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 12), _dims_125791, 'append')
        # Calling append(args, kwargs) (line 608)
        append_call_result_125795 = invoke(stypy.reporting.localization.Localization(__file__, 608, 12), append_125792, *[name_125793], **kwargs_125794)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_read_dim_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_dim_array' in the type store
        # Getting the type of 'stypy_return_type' (line 598)
        stypy_return_type_125796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125796)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_dim_array'
        return stypy_return_type_125796


    @norecursion
    def _read_gatt_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_gatt_array'
        module_type_store = module_type_store.open_function_context('_read_gatt_array', 610, 4, False)
        # Assigning a type to the variable 'self' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read_gatt_array')
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read_gatt_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read_gatt_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_gatt_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_gatt_array(...)' code ##################

        
        
        # Call to items(...): (line 611)
        # Processing the call keyword arguments (line 611)
        kwargs_125802 = {}
        
        # Call to _read_att_array(...): (line 611)
        # Processing the call keyword arguments (line 611)
        kwargs_125799 = {}
        # Getting the type of 'self' (line 611)
        self_125797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 20), 'self', False)
        # Obtaining the member '_read_att_array' of a type (line 611)
        _read_att_array_125798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 20), self_125797, '_read_att_array')
        # Calling _read_att_array(args, kwargs) (line 611)
        _read_att_array_call_result_125800 = invoke(stypy.reporting.localization.Localization(__file__, 611, 20), _read_att_array_125798, *[], **kwargs_125799)
        
        # Obtaining the member 'items' of a type (line 611)
        items_125801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 20), _read_att_array_call_result_125800, 'items')
        # Calling items(args, kwargs) (line 611)
        items_call_result_125803 = invoke(stypy.reporting.localization.Localization(__file__, 611, 20), items_125801, *[], **kwargs_125802)
        
        # Testing the type of a for loop iterable (line 611)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 611, 8), items_call_result_125803)
        # Getting the type of the for loop variable (line 611)
        for_loop_var_125804 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 611, 8), items_call_result_125803)
        # Assigning a type to the variable 'k' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 8), for_loop_var_125804))
        # Assigning a type to the variable 'v' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 8), for_loop_var_125804))
        # SSA begins for a for statement (line 611)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to __setattr__(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'k' (line 612)
        k_125807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 29), 'k', False)
        # Getting the type of 'v' (line 612)
        v_125808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 32), 'v', False)
        # Processing the call keyword arguments (line 612)
        kwargs_125809 = {}
        # Getting the type of 'self' (line 612)
        self_125805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'self', False)
        # Obtaining the member '__setattr__' of a type (line 612)
        setattr___125806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 12), self_125805, '__setattr__')
        # Calling __setattr__(args, kwargs) (line 612)
        setattr___call_result_125810 = invoke(stypy.reporting.localization.Localization(__file__, 612, 12), setattr___125806, *[k_125807, v_125808], **kwargs_125809)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_read_gatt_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_gatt_array' in the type store
        # Getting the type of 'stypy_return_type' (line 610)
        stypy_return_type_125811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125811)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_gatt_array'
        return stypy_return_type_125811


    @norecursion
    def _read_att_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_att_array'
        module_type_store = module_type_store.open_function_context('_read_att_array', 614, 4, False)
        # Assigning a type to the variable 'self' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read_att_array')
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read_att_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read_att_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_att_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_att_array(...)' code ##################

        
        # Assigning a Call to a Name (line 615):
        
        # Assigning a Call to a Name (line 615):
        
        # Call to read(...): (line 615)
        # Processing the call arguments (line 615)
        int_125815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 30), 'int')
        # Processing the call keyword arguments (line 615)
        kwargs_125816 = {}
        # Getting the type of 'self' (line 615)
        self_125812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'self', False)
        # Obtaining the member 'fp' of a type (line 615)
        fp_125813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 17), self_125812, 'fp')
        # Obtaining the member 'read' of a type (line 615)
        read_125814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 17), fp_125813, 'read')
        # Calling read(args, kwargs) (line 615)
        read_call_result_125817 = invoke(stypy.reporting.localization.Localization(__file__, 615, 17), read_125814, *[int_125815], **kwargs_125816)
        
        # Assigning a type to the variable 'header' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'header', read_call_result_125817)
        
        
        # Getting the type of 'header' (line 616)
        header_125818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'header')
        
        # Obtaining an instance of the builtin type 'list' (line 616)
        list_125819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 616)
        # Adding element type (line 616)
        # Getting the type of 'ZERO' (line 616)
        ZERO_125820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 26), 'ZERO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 25), list_125819, ZERO_125820)
        # Adding element type (line 616)
        # Getting the type of 'NC_ATTRIBUTE' (line 616)
        NC_ATTRIBUTE_125821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 32), 'NC_ATTRIBUTE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 25), list_125819, NC_ATTRIBUTE_125821)
        
        # Applying the binary operator 'notin' (line 616)
        result_contains_125822 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 11), 'notin', header_125818, list_125819)
        
        # Testing the type of an if condition (line 616)
        if_condition_125823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 8), result_contains_125822)
        # Assigning a type to the variable 'if_condition_125823' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'if_condition_125823', if_condition_125823)
        # SSA begins for if statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 617)
        # Processing the call arguments (line 617)
        str_125825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 29), 'str', 'Unexpected header.')
        # Processing the call keyword arguments (line 617)
        kwargs_125826 = {}
        # Getting the type of 'ValueError' (line 617)
        ValueError_125824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 617)
        ValueError_call_result_125827 = invoke(stypy.reporting.localization.Localization(__file__, 617, 18), ValueError_125824, *[str_125825], **kwargs_125826)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 617, 12), ValueError_call_result_125827, 'raise parameter', BaseException)
        # SSA join for if statement (line 616)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 618):
        
        # Assigning a Call to a Name (line 618):
        
        # Call to _unpack_int(...): (line 618)
        # Processing the call keyword arguments (line 618)
        kwargs_125830 = {}
        # Getting the type of 'self' (line 618)
        self_125828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 618)
        _unpack_int_125829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 16), self_125828, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 618)
        _unpack_int_call_result_125831 = invoke(stypy.reporting.localization.Localization(__file__, 618, 16), _unpack_int_125829, *[], **kwargs_125830)
        
        # Assigning a type to the variable 'count' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'count', _unpack_int_call_result_125831)
        
        # Assigning a Call to a Name (line 620):
        
        # Assigning a Call to a Name (line 620):
        
        # Call to OrderedDict(...): (line 620)
        # Processing the call keyword arguments (line 620)
        kwargs_125833 = {}
        # Getting the type of 'OrderedDict' (line 620)
        OrderedDict_125832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 620)
        OrderedDict_call_result_125834 = invoke(stypy.reporting.localization.Localization(__file__, 620, 21), OrderedDict_125832, *[], **kwargs_125833)
        
        # Assigning a type to the variable 'attributes' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'attributes', OrderedDict_call_result_125834)
        
        
        # Call to range(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'count' (line 621)
        count_125836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 26), 'count', False)
        # Processing the call keyword arguments (line 621)
        kwargs_125837 = {}
        # Getting the type of 'range' (line 621)
        range_125835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'range', False)
        # Calling range(args, kwargs) (line 621)
        range_call_result_125838 = invoke(stypy.reporting.localization.Localization(__file__, 621, 20), range_125835, *[count_125836], **kwargs_125837)
        
        # Testing the type of a for loop iterable (line 621)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 621, 8), range_call_result_125838)
        # Getting the type of the for loop variable (line 621)
        for_loop_var_125839 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 621, 8), range_call_result_125838)
        # Assigning a type to the variable 'attr' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'attr', for_loop_var_125839)
        # SSA begins for a for statement (line 621)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 622):
        
        # Assigning a Call to a Name (line 622):
        
        # Call to asstr(...): (line 622)
        # Processing the call arguments (line 622)
        
        # Call to _unpack_string(...): (line 622)
        # Processing the call keyword arguments (line 622)
        kwargs_125843 = {}
        # Getting the type of 'self' (line 622)
        self_125841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 25), 'self', False)
        # Obtaining the member '_unpack_string' of a type (line 622)
        _unpack_string_125842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 25), self_125841, '_unpack_string')
        # Calling _unpack_string(args, kwargs) (line 622)
        _unpack_string_call_result_125844 = invoke(stypy.reporting.localization.Localization(__file__, 622, 25), _unpack_string_125842, *[], **kwargs_125843)
        
        # Processing the call keyword arguments (line 622)
        kwargs_125845 = {}
        # Getting the type of 'asstr' (line 622)
        asstr_125840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 'asstr', False)
        # Calling asstr(args, kwargs) (line 622)
        asstr_call_result_125846 = invoke(stypy.reporting.localization.Localization(__file__, 622, 19), asstr_125840, *[_unpack_string_call_result_125844], **kwargs_125845)
        
        # Assigning a type to the variable 'name' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 12), 'name', asstr_call_result_125846)
        
        # Assigning a Call to a Subscript (line 623):
        
        # Assigning a Call to a Subscript (line 623):
        
        # Call to _read_values(...): (line 623)
        # Processing the call keyword arguments (line 623)
        kwargs_125849 = {}
        # Getting the type of 'self' (line 623)
        self_125847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 31), 'self', False)
        # Obtaining the member '_read_values' of a type (line 623)
        _read_values_125848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 31), self_125847, '_read_values')
        # Calling _read_values(args, kwargs) (line 623)
        _read_values_call_result_125850 = invoke(stypy.reporting.localization.Localization(__file__, 623, 31), _read_values_125848, *[], **kwargs_125849)
        
        # Getting the type of 'attributes' (line 623)
        attributes_125851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'attributes')
        # Getting the type of 'name' (line 623)
        name_125852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 23), 'name')
        # Storing an element on a container (line 623)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 12), attributes_125851, (name_125852, _read_values_call_result_125850))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'attributes' (line 624)
        attributes_125853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'attributes')
        # Assigning a type to the variable 'stypy_return_type' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'stypy_return_type', attributes_125853)
        
        # ################# End of '_read_att_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_att_array' in the type store
        # Getting the type of 'stypy_return_type' (line 614)
        stypy_return_type_125854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_125854)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_att_array'
        return stypy_return_type_125854


    @norecursion
    def _read_var_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_var_array'
        module_type_store = module_type_store.open_function_context('_read_var_array', 626, 4, False)
        # Assigning a type to the variable 'self' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read_var_array')
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read_var_array.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read_var_array', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_var_array', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_var_array(...)' code ##################

        
        # Assigning a Call to a Name (line 627):
        
        # Assigning a Call to a Name (line 627):
        
        # Call to read(...): (line 627)
        # Processing the call arguments (line 627)
        int_125858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 30), 'int')
        # Processing the call keyword arguments (line 627)
        kwargs_125859 = {}
        # Getting the type of 'self' (line 627)
        self_125855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 17), 'self', False)
        # Obtaining the member 'fp' of a type (line 627)
        fp_125856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 17), self_125855, 'fp')
        # Obtaining the member 'read' of a type (line 627)
        read_125857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 17), fp_125856, 'read')
        # Calling read(args, kwargs) (line 627)
        read_call_result_125860 = invoke(stypy.reporting.localization.Localization(__file__, 627, 17), read_125857, *[int_125858], **kwargs_125859)
        
        # Assigning a type to the variable 'header' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'header', read_call_result_125860)
        
        
        # Getting the type of 'header' (line 628)
        header_125861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 11), 'header')
        
        # Obtaining an instance of the builtin type 'list' (line 628)
        list_125862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 628)
        # Adding element type (line 628)
        # Getting the type of 'ZERO' (line 628)
        ZERO_125863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 26), 'ZERO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 25), list_125862, ZERO_125863)
        # Adding element type (line 628)
        # Getting the type of 'NC_VARIABLE' (line 628)
        NC_VARIABLE_125864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 32), 'NC_VARIABLE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 628, 25), list_125862, NC_VARIABLE_125864)
        
        # Applying the binary operator 'notin' (line 628)
        result_contains_125865 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 11), 'notin', header_125861, list_125862)
        
        # Testing the type of an if condition (line 628)
        if_condition_125866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 8), result_contains_125865)
        # Assigning a type to the variable 'if_condition_125866' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'if_condition_125866', if_condition_125866)
        # SSA begins for if statement (line 628)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 629)
        # Processing the call arguments (line 629)
        str_125868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 29), 'str', 'Unexpected header.')
        # Processing the call keyword arguments (line 629)
        kwargs_125869 = {}
        # Getting the type of 'ValueError' (line 629)
        ValueError_125867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 629)
        ValueError_call_result_125870 = invoke(stypy.reporting.localization.Localization(__file__, 629, 18), ValueError_125867, *[str_125868], **kwargs_125869)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 629, 12), ValueError_call_result_125870, 'raise parameter', BaseException)
        # SSA join for if statement (line 628)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 631):
        
        # Assigning a Num to a Name (line 631):
        int_125871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 16), 'int')
        # Assigning a type to the variable 'begin' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'begin', int_125871)
        
        # Assigning a Dict to a Name (line 632):
        
        # Assigning a Dict to a Name (line 632):
        
        # Obtaining an instance of the builtin type 'dict' (line 632)
        dict_125872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 632)
        # Adding element type (key, value) (line 632)
        str_125873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 18), 'str', 'names')
        
        # Obtaining an instance of the builtin type 'list' (line 632)
        list_125874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 632)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 17), dict_125872, (str_125873, list_125874))
        # Adding element type (key, value) (line 632)
        str_125875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 31), 'str', 'formats')
        
        # Obtaining an instance of the builtin type 'list' (line 632)
        list_125876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 632)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 17), dict_125872, (str_125875, list_125876))
        
        # Assigning a type to the variable 'dtypes' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'dtypes', dict_125872)
        
        # Assigning a List to a Name (line 633):
        
        # Assigning a List to a Name (line 633):
        
        # Obtaining an instance of the builtin type 'list' (line 633)
        list_125877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 633)
        
        # Assigning a type to the variable 'rec_vars' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'rec_vars', list_125877)
        
        # Assigning a Call to a Name (line 634):
        
        # Assigning a Call to a Name (line 634):
        
        # Call to _unpack_int(...): (line 634)
        # Processing the call keyword arguments (line 634)
        kwargs_125880 = {}
        # Getting the type of 'self' (line 634)
        self_125878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 16), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 634)
        _unpack_int_125879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 16), self_125878, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 634)
        _unpack_int_call_result_125881 = invoke(stypy.reporting.localization.Localization(__file__, 634, 16), _unpack_int_125879, *[], **kwargs_125880)
        
        # Assigning a type to the variable 'count' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'count', _unpack_int_call_result_125881)
        
        
        # Call to range(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'count' (line 635)
        count_125883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 25), 'count', False)
        # Processing the call keyword arguments (line 635)
        kwargs_125884 = {}
        # Getting the type of 'range' (line 635)
        range_125882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 19), 'range', False)
        # Calling range(args, kwargs) (line 635)
        range_call_result_125885 = invoke(stypy.reporting.localization.Localization(__file__, 635, 19), range_125882, *[count_125883], **kwargs_125884)
        
        # Testing the type of a for loop iterable (line 635)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 635, 8), range_call_result_125885)
        # Getting the type of the for loop variable (line 635)
        for_loop_var_125886 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 635, 8), range_call_result_125885)
        # Assigning a type to the variable 'var' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'var', for_loop_var_125886)
        # SSA begins for a for statement (line 635)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 636):
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125890 = {}
        # Getting the type of 'self' (line 637)
        self_125888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125888, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125891 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125889, *[], **kwargs_125890)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125893 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125892, int_125887)
        
        # Assigning a type to the variable 'tuple_var_assignment_124548' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124548', subscript_call_result_125893)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125897 = {}
        # Getting the type of 'self' (line 637)
        self_125895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125895, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125898 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125896, *[], **kwargs_125897)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125900 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125899, int_125894)
        
        # Assigning a type to the variable 'tuple_var_assignment_124549' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124549', subscript_call_result_125900)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125904 = {}
        # Getting the type of 'self' (line 637)
        self_125902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125902, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125905 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125903, *[], **kwargs_125904)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125905, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125907 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125906, int_125901)
        
        # Assigning a type to the variable 'tuple_var_assignment_124550' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124550', subscript_call_result_125907)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125911 = {}
        # Getting the type of 'self' (line 637)
        self_125909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125909, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125912 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125910, *[], **kwargs_125911)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125912, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125914 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125913, int_125908)
        
        # Assigning a type to the variable 'tuple_var_assignment_124551' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124551', subscript_call_result_125914)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125918 = {}
        # Getting the type of 'self' (line 637)
        self_125916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125916, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125919 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125917, *[], **kwargs_125918)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125921 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125920, int_125915)
        
        # Assigning a type to the variable 'tuple_var_assignment_124552' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124552', subscript_call_result_125921)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125925 = {}
        # Getting the type of 'self' (line 637)
        self_125923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125923, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125926 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125924, *[], **kwargs_125925)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125928 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125927, int_125922)
        
        # Assigning a type to the variable 'tuple_var_assignment_124553' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124553', subscript_call_result_125928)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125932 = {}
        # Getting the type of 'self' (line 637)
        self_125930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125930, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125933 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125931, *[], **kwargs_125932)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125933, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125935 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125934, int_125929)
        
        # Assigning a type to the variable 'tuple_var_assignment_124554' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124554', subscript_call_result_125935)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125939 = {}
        # Getting the type of 'self' (line 637)
        self_125937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125937, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125940 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125938, *[], **kwargs_125939)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125940, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125942 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125941, int_125936)
        
        # Assigning a type to the variable 'tuple_var_assignment_124555' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124555', subscript_call_result_125942)
        
        # Assigning a Subscript to a Name (line 636):
        
        # Obtaining the type of the subscript
        int_125943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 12), 'int')
        
        # Call to _read_var(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_125946 = {}
        # Getting the type of 'self' (line 637)
        self_125944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 54), 'self', False)
        # Obtaining the member '_read_var' of a type (line 637)
        _read_var_125945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 54), self_125944, '_read_var')
        # Calling _read_var(args, kwargs) (line 637)
        _read_var_call_result_125947 = invoke(stypy.reporting.localization.Localization(__file__, 637, 54), _read_var_125945, *[], **kwargs_125946)
        
        # Obtaining the member '__getitem__' of a type (line 636)
        getitem___125948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), _read_var_call_result_125947, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 636)
        subscript_call_result_125949 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), getitem___125948, int_125943)
        
        # Assigning a type to the variable 'tuple_var_assignment_124556' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124556', subscript_call_result_125949)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124548' (line 636)
        tuple_var_assignment_124548_125950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124548')
        # Assigning a type to the variable 'name' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 13), 'name', tuple_var_assignment_124548_125950)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124549' (line 636)
        tuple_var_assignment_124549_125951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124549')
        # Assigning a type to the variable 'dimensions' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 19), 'dimensions', tuple_var_assignment_124549_125951)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124550' (line 636)
        tuple_var_assignment_124550_125952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124550')
        # Assigning a type to the variable 'shape' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 31), 'shape', tuple_var_assignment_124550_125952)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124551' (line 636)
        tuple_var_assignment_124551_125953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124551')
        # Assigning a type to the variable 'attributes' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 38), 'attributes', tuple_var_assignment_124551_125953)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124552' (line 636)
        tuple_var_assignment_124552_125954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124552')
        # Assigning a type to the variable 'typecode' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 13), 'typecode', tuple_var_assignment_124552_125954)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124553' (line 636)
        tuple_var_assignment_124553_125955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124553')
        # Assigning a type to the variable 'size' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 23), 'size', tuple_var_assignment_124553_125955)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124554' (line 636)
        tuple_var_assignment_124554_125956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124554')
        # Assigning a type to the variable 'dtype_' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 29), 'dtype_', tuple_var_assignment_124554_125956)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124555' (line 636)
        tuple_var_assignment_124555_125957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124555')
        # Assigning a type to the variable 'begin_' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 37), 'begin_', tuple_var_assignment_124555_125957)
        
        # Assigning a Name to a Name (line 636):
        # Getting the type of 'tuple_var_assignment_124556' (line 636)
        tuple_var_assignment_124556_125958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'tuple_var_assignment_124556')
        # Assigning a type to the variable 'vsize' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 45), 'vsize', tuple_var_assignment_124556_125958)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'shape' (line 652)
        shape_125959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 15), 'shape')
        
        
        # Obtaining the type of the subscript
        int_125960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 31), 'int')
        # Getting the type of 'shape' (line 652)
        shape_125961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 25), 'shape')
        # Obtaining the member '__getitem__' of a type (line 652)
        getitem___125962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 25), shape_125961, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 652)
        subscript_call_result_125963 = invoke(stypy.reporting.localization.Localization(__file__, 652, 25), getitem___125962, int_125960)
        
        # Getting the type of 'None' (line 652)
        None_125964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 37), 'None')
        # Applying the binary operator 'is' (line 652)
        result_is__125965 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 25), 'is', subscript_call_result_125963, None_125964)
        
        # Applying the binary operator 'and' (line 652)
        result_and_keyword_125966 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 15), 'and', shape_125959, result_is__125965)
        
        # Testing the type of an if condition (line 652)
        if_condition_125967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 652, 12), result_and_keyword_125966)
        # Assigning a type to the variable 'if_condition_125967' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'if_condition_125967', if_condition_125967)
        # SSA begins for if statement (line 652)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 653)
        # Processing the call arguments (line 653)
        # Getting the type of 'name' (line 653)
        name_125970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 32), 'name', False)
        # Processing the call keyword arguments (line 653)
        kwargs_125971 = {}
        # Getting the type of 'rec_vars' (line 653)
        rec_vars_125968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 16), 'rec_vars', False)
        # Obtaining the member 'append' of a type (line 653)
        append_125969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 16), rec_vars_125968, 'append')
        # Calling append(args, kwargs) (line 653)
        append_call_result_125972 = invoke(stypy.reporting.localization.Localization(__file__, 653, 16), append_125969, *[name_125970], **kwargs_125971)
        
        
        # Getting the type of 'self' (line 656)
        self_125973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'self')
        # Obtaining the member '__dict__' of a type (line 656)
        dict___125974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 16), self_125973, '__dict__')
        
        # Obtaining the type of the subscript
        str_125975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 30), 'str', '_recsize')
        # Getting the type of 'self' (line 656)
        self_125976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'self')
        # Obtaining the member '__dict__' of a type (line 656)
        dict___125977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 16), self_125976, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 656)
        getitem___125978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 16), dict___125977, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 656)
        subscript_call_result_125979 = invoke(stypy.reporting.localization.Localization(__file__, 656, 16), getitem___125978, str_125975)
        
        # Getting the type of 'vsize' (line 656)
        vsize_125980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 45), 'vsize')
        # Applying the binary operator '+=' (line 656)
        result_iadd_125981 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 16), '+=', subscript_call_result_125979, vsize_125980)
        # Getting the type of 'self' (line 656)
        self_125982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'self')
        # Obtaining the member '__dict__' of a type (line 656)
        dict___125983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 16), self_125982, '__dict__')
        str_125984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 30), 'str', '_recsize')
        # Storing an element on a container (line 656)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 16), dict___125983, (str_125984, result_iadd_125981))
        
        
        
        # Getting the type of 'begin' (line 657)
        begin_125985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 19), 'begin')
        int_125986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 28), 'int')
        # Applying the binary operator '==' (line 657)
        result_eq_125987 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 19), '==', begin_125985, int_125986)
        
        # Testing the type of an if condition (line 657)
        if_condition_125988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 16), result_eq_125987)
        # Assigning a type to the variable 'if_condition_125988' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'if_condition_125988', if_condition_125988)
        # SSA begins for if statement (line 657)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 658):
        
        # Assigning a Name to a Name (line 658):
        # Getting the type of 'begin_' (line 658)
        begin__125989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 28), 'begin_')
        # Assigning a type to the variable 'begin' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 20), 'begin', begin__125989)
        # SSA join for if statement (line 657)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 659)
        # Processing the call arguments (line 659)
        # Getting the type of 'name' (line 659)
        name_125995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 39), 'name', False)
        # Processing the call keyword arguments (line 659)
        kwargs_125996 = {}
        
        # Obtaining the type of the subscript
        str_125990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 23), 'str', 'names')
        # Getting the type of 'dtypes' (line 659)
        dtypes_125991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'dtypes', False)
        # Obtaining the member '__getitem__' of a type (line 659)
        getitem___125992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 16), dtypes_125991, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 659)
        subscript_call_result_125993 = invoke(stypy.reporting.localization.Localization(__file__, 659, 16), getitem___125992, str_125990)
        
        # Obtaining the member 'append' of a type (line 659)
        append_125994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 16), subscript_call_result_125993, 'append')
        # Calling append(args, kwargs) (line 659)
        append_call_result_125997 = invoke(stypy.reporting.localization.Localization(__file__, 659, 16), append_125994, *[name_125995], **kwargs_125996)
        
        
        # Call to append(...): (line 660)
        # Processing the call arguments (line 660)
        
        # Call to str(...): (line 660)
        # Processing the call arguments (line 660)
        
        # Obtaining the type of the subscript
        int_126004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 51), 'int')
        slice_126005 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 660, 45), int_126004, None, None)
        # Getting the type of 'shape' (line 660)
        shape_126006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 45), 'shape', False)
        # Obtaining the member '__getitem__' of a type (line 660)
        getitem___126007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 45), shape_126006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 660)
        subscript_call_result_126008 = invoke(stypy.reporting.localization.Localization(__file__, 660, 45), getitem___126007, slice_126005)
        
        # Processing the call keyword arguments (line 660)
        kwargs_126009 = {}
        # Getting the type of 'str' (line 660)
        str_126003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 41), 'str', False)
        # Calling str(args, kwargs) (line 660)
        str_call_result_126010 = invoke(stypy.reporting.localization.Localization(__file__, 660, 41), str_126003, *[subscript_call_result_126008], **kwargs_126009)
        
        # Getting the type of 'dtype_' (line 660)
        dtype__126011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 58), 'dtype_', False)
        # Applying the binary operator '+' (line 660)
        result_add_126012 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 41), '+', str_call_result_126010, dtype__126011)
        
        # Processing the call keyword arguments (line 660)
        kwargs_126013 = {}
        
        # Obtaining the type of the subscript
        str_125998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 23), 'str', 'formats')
        # Getting the type of 'dtypes' (line 660)
        dtypes_125999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'dtypes', False)
        # Obtaining the member '__getitem__' of a type (line 660)
        getitem___126000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 16), dtypes_125999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 660)
        subscript_call_result_126001 = invoke(stypy.reporting.localization.Localization(__file__, 660, 16), getitem___126000, str_125998)
        
        # Obtaining the member 'append' of a type (line 660)
        append_126002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 16), subscript_call_result_126001, 'append')
        # Calling append(args, kwargs) (line 660)
        append_call_result_126014 = invoke(stypy.reporting.localization.Localization(__file__, 660, 16), append_126002, *[result_add_126012], **kwargs_126013)
        
        
        
        # Getting the type of 'typecode' (line 663)
        typecode_126015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 19), 'typecode')
        str_126016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 31), 'str', 'bch')
        # Applying the binary operator 'in' (line 663)
        result_contains_126017 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 19), 'in', typecode_126015, str_126016)
        
        # Testing the type of an if condition (line 663)
        if_condition_126018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 16), result_contains_126017)
        # Assigning a type to the variable 'if_condition_126018' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'if_condition_126018', if_condition_126018)
        # SSA begins for if statement (line 663)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 664):
        
        # Assigning a BinOp to a Name (line 664):
        
        # Call to reduce(...): (line 664)
        # Processing the call arguments (line 664)
        # Getting the type of 'mul' (line 664)
        mul_126020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 41), 'mul', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 664)
        tuple_126021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 664)
        # Adding element type (line 664)
        int_126022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 47), tuple_126021, int_126022)
        
        
        # Obtaining the type of the subscript
        int_126023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 59), 'int')
        slice_126024 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 664, 53), int_126023, None, None)
        # Getting the type of 'shape' (line 664)
        shape_126025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 53), 'shape', False)
        # Obtaining the member '__getitem__' of a type (line 664)
        getitem___126026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 53), shape_126025, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 664)
        subscript_call_result_126027 = invoke(stypy.reporting.localization.Localization(__file__, 664, 53), getitem___126026, slice_126024)
        
        # Applying the binary operator '+' (line 664)
        result_add_126028 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 46), '+', tuple_126021, subscript_call_result_126027)
        
        # Processing the call keyword arguments (line 664)
        kwargs_126029 = {}
        # Getting the type of 'reduce' (line 664)
        reduce_126019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 34), 'reduce', False)
        # Calling reduce(args, kwargs) (line 664)
        reduce_call_result_126030 = invoke(stypy.reporting.localization.Localization(__file__, 664, 34), reduce_126019, *[mul_126020, result_add_126028], **kwargs_126029)
        
        # Getting the type of 'size' (line 664)
        size_126031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 66), 'size')
        # Applying the binary operator '*' (line 664)
        result_mul_126032 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 34), '*', reduce_call_result_126030, size_126031)
        
        # Assigning a type to the variable 'actual_size' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 20), 'actual_size', result_mul_126032)
        
        # Assigning a BinOp to a Name (line 665):
        
        # Assigning a BinOp to a Name (line 665):
        
        # Getting the type of 'actual_size' (line 665)
        actual_size_126033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 31), 'actual_size')
        # Applying the 'usub' unary operator (line 665)
        result___neg___126034 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 30), 'usub', actual_size_126033)
        
        int_126035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 45), 'int')
        # Applying the binary operator '%' (line 665)
        result_mod_126036 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 30), '%', result___neg___126034, int_126035)
        
        # Assigning a type to the variable 'padding' (line 665)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 20), 'padding', result_mod_126036)
        
        # Getting the type of 'padding' (line 666)
        padding_126037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 23), 'padding')
        # Testing the type of an if condition (line 666)
        if_condition_126038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 20), padding_126037)
        # Assigning a type to the variable 'if_condition_126038' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 20), 'if_condition_126038', if_condition_126038)
        # SSA begins for if statement (line 666)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 667)
        # Processing the call arguments (line 667)
        str_126044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 47), 'str', '_padding_%d')
        # Getting the type of 'var' (line 667)
        var_126045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 63), 'var', False)
        # Applying the binary operator '%' (line 667)
        result_mod_126046 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 47), '%', str_126044, var_126045)
        
        # Processing the call keyword arguments (line 667)
        kwargs_126047 = {}
        
        # Obtaining the type of the subscript
        str_126039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 31), 'str', 'names')
        # Getting the type of 'dtypes' (line 667)
        dtypes_126040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 24), 'dtypes', False)
        # Obtaining the member '__getitem__' of a type (line 667)
        getitem___126041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 24), dtypes_126040, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 667)
        subscript_call_result_126042 = invoke(stypy.reporting.localization.Localization(__file__, 667, 24), getitem___126041, str_126039)
        
        # Obtaining the member 'append' of a type (line 667)
        append_126043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 24), subscript_call_result_126042, 'append')
        # Calling append(args, kwargs) (line 667)
        append_call_result_126048 = invoke(stypy.reporting.localization.Localization(__file__, 667, 24), append_126043, *[result_mod_126046], **kwargs_126047)
        
        
        # Call to append(...): (line 668)
        # Processing the call arguments (line 668)
        str_126054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 49), 'str', '(%d,)>b')
        # Getting the type of 'padding' (line 668)
        padding_126055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 61), 'padding', False)
        # Applying the binary operator '%' (line 668)
        result_mod_126056 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 49), '%', str_126054, padding_126055)
        
        # Processing the call keyword arguments (line 668)
        kwargs_126057 = {}
        
        # Obtaining the type of the subscript
        str_126049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 31), 'str', 'formats')
        # Getting the type of 'dtypes' (line 668)
        dtypes_126050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 24), 'dtypes', False)
        # Obtaining the member '__getitem__' of a type (line 668)
        getitem___126051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 24), dtypes_126050, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 668)
        subscript_call_result_126052 = invoke(stypy.reporting.localization.Localization(__file__, 668, 24), getitem___126051, str_126049)
        
        # Obtaining the member 'append' of a type (line 668)
        append_126053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 24), subscript_call_result_126052, 'append')
        # Calling append(args, kwargs) (line 668)
        append_call_result_126058 = invoke(stypy.reporting.localization.Localization(__file__, 668, 24), append_126053, *[result_mod_126056], **kwargs_126057)
        
        # SSA join for if statement (line 666)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 663)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 671):
        
        # Assigning a Name to a Name (line 671):
        # Getting the type of 'None' (line 671)
        None_126059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 23), 'None')
        # Assigning a type to the variable 'data' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'data', None_126059)
        # SSA branch for the else part of an if statement (line 652)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 674):
        
        # Assigning a BinOp to a Name (line 674):
        
        # Call to reduce(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'mul' (line 674)
        mul_126061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 32), 'mul', False)
        # Getting the type of 'shape' (line 674)
        shape_126062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 37), 'shape', False)
        int_126063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 44), 'int')
        # Processing the call keyword arguments (line 674)
        kwargs_126064 = {}
        # Getting the type of 'reduce' (line 674)
        reduce_126060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 25), 'reduce', False)
        # Calling reduce(args, kwargs) (line 674)
        reduce_call_result_126065 = invoke(stypy.reporting.localization.Localization(__file__, 674, 25), reduce_126060, *[mul_126061, shape_126062, int_126063], **kwargs_126064)
        
        # Getting the type of 'size' (line 674)
        size_126066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 49), 'size')
        # Applying the binary operator '*' (line 674)
        result_mul_126067 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 25), '*', reduce_call_result_126065, size_126066)
        
        # Assigning a type to the variable 'a_size' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'a_size', result_mul_126067)
        
        # Getting the type of 'self' (line 675)
        self_126068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 19), 'self')
        # Obtaining the member 'use_mmap' of a type (line 675)
        use_mmap_126069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 19), self_126068, 'use_mmap')
        # Testing the type of an if condition (line 675)
        if_condition_126070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 675, 16), use_mmap_126069)
        # Assigning a type to the variable 'if_condition_126070' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'if_condition_126070', if_condition_126070)
        # SSA begins for if statement (line 675)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 676):
        
        # Assigning a Call to a Name (line 676):
        
        # Call to view(...): (line 676)
        # Processing the call keyword arguments (line 676)
        # Getting the type of 'dtype_' (line 676)
        dtype__126081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 73), 'dtype_', False)
        keyword_126082 = dtype__126081
        kwargs_126083 = {'dtype': keyword_126082}
        
        # Obtaining the type of the subscript
        # Getting the type of 'begin_' (line 676)
        begin__126071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 40), 'begin_', False)
        # Getting the type of 'begin_' (line 676)
        begin__126072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 47), 'begin_', False)
        # Getting the type of 'a_size' (line 676)
        a_size_126073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 54), 'a_size', False)
        # Applying the binary operator '+' (line 676)
        result_add_126074 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 47), '+', begin__126072, a_size_126073)
        
        slice_126075 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 27), begin__126071, result_add_126074, None)
        # Getting the type of 'self' (line 676)
        self_126076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 27), 'self', False)
        # Obtaining the member '_mm_buf' of a type (line 676)
        _mm_buf_126077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 27), self_126076, '_mm_buf')
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___126078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 27), _mm_buf_126077, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_126079 = invoke(stypy.reporting.localization.Localization(__file__, 676, 27), getitem___126078, slice_126075)
        
        # Obtaining the member 'view' of a type (line 676)
        view_126080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 27), subscript_call_result_126079, 'view')
        # Calling view(args, kwargs) (line 676)
        view_call_result_126084 = invoke(stypy.reporting.localization.Localization(__file__, 676, 27), view_126080, *[], **kwargs_126083)
        
        # Assigning a type to the variable 'data' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 20), 'data', view_call_result_126084)
        
        # Assigning a Name to a Attribute (line 677):
        
        # Assigning a Name to a Attribute (line 677):
        # Getting the type of 'shape' (line 677)
        shape_126085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 33), 'shape')
        # Getting the type of 'data' (line 677)
        data_126086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 20), 'data')
        # Setting the type of the member 'shape' of a type (line 677)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 20), data_126086, 'shape', shape_126085)
        # SSA branch for the else part of an if statement (line 675)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 679):
        
        # Assigning a Call to a Name (line 679):
        
        # Call to tell(...): (line 679)
        # Processing the call keyword arguments (line 679)
        kwargs_126090 = {}
        # Getting the type of 'self' (line 679)
        self_126087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 26), 'self', False)
        # Obtaining the member 'fp' of a type (line 679)
        fp_126088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 26), self_126087, 'fp')
        # Obtaining the member 'tell' of a type (line 679)
        tell_126089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 26), fp_126088, 'tell')
        # Calling tell(args, kwargs) (line 679)
        tell_call_result_126091 = invoke(stypy.reporting.localization.Localization(__file__, 679, 26), tell_126089, *[], **kwargs_126090)
        
        # Assigning a type to the variable 'pos' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 20), 'pos', tell_call_result_126091)
        
        # Call to seek(...): (line 680)
        # Processing the call arguments (line 680)
        # Getting the type of 'begin_' (line 680)
        begin__126095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 33), 'begin_', False)
        # Processing the call keyword arguments (line 680)
        kwargs_126096 = {}
        # Getting the type of 'self' (line 680)
        self_126092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 20), 'self', False)
        # Obtaining the member 'fp' of a type (line 680)
        fp_126093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 20), self_126092, 'fp')
        # Obtaining the member 'seek' of a type (line 680)
        seek_126094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 20), fp_126093, 'seek')
        # Calling seek(args, kwargs) (line 680)
        seek_call_result_126097 = invoke(stypy.reporting.localization.Localization(__file__, 680, 20), seek_126094, *[begin__126095], **kwargs_126096)
        
        
        # Assigning a Call to a Name (line 681):
        
        # Assigning a Call to a Name (line 681):
        
        # Call to fromstring(...): (line 681)
        # Processing the call arguments (line 681)
        
        # Call to read(...): (line 681)
        # Processing the call arguments (line 681)
        # Getting the type of 'a_size' (line 681)
        a_size_126102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 51), 'a_size', False)
        # Processing the call keyword arguments (line 681)
        kwargs_126103 = {}
        # Getting the type of 'self' (line 681)
        self_126099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 38), 'self', False)
        # Obtaining the member 'fp' of a type (line 681)
        fp_126100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 38), self_126099, 'fp')
        # Obtaining the member 'read' of a type (line 681)
        read_126101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 38), fp_126100, 'read')
        # Calling read(args, kwargs) (line 681)
        read_call_result_126104 = invoke(stypy.reporting.localization.Localization(__file__, 681, 38), read_126101, *[a_size_126102], **kwargs_126103)
        
        # Processing the call keyword arguments (line 681)
        # Getting the type of 'dtype_' (line 681)
        dtype__126105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 66), 'dtype_', False)
        keyword_126106 = dtype__126105
        kwargs_126107 = {'dtype': keyword_126106}
        # Getting the type of 'fromstring' (line 681)
        fromstring_126098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 27), 'fromstring', False)
        # Calling fromstring(args, kwargs) (line 681)
        fromstring_call_result_126108 = invoke(stypy.reporting.localization.Localization(__file__, 681, 27), fromstring_126098, *[read_call_result_126104], **kwargs_126107)
        
        # Assigning a type to the variable 'data' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 20), 'data', fromstring_call_result_126108)
        
        # Assigning a Name to a Attribute (line 682):
        
        # Assigning a Name to a Attribute (line 682):
        # Getting the type of 'shape' (line 682)
        shape_126109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 33), 'shape')
        # Getting the type of 'data' (line 682)
        data_126110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 20), 'data')
        # Setting the type of the member 'shape' of a type (line 682)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 20), data_126110, 'shape', shape_126109)
        
        # Call to seek(...): (line 683)
        # Processing the call arguments (line 683)
        # Getting the type of 'pos' (line 683)
        pos_126114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 33), 'pos', False)
        # Processing the call keyword arguments (line 683)
        kwargs_126115 = {}
        # Getting the type of 'self' (line 683)
        self_126111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 20), 'self', False)
        # Obtaining the member 'fp' of a type (line 683)
        fp_126112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 20), self_126111, 'fp')
        # Obtaining the member 'seek' of a type (line 683)
        seek_126113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 20), fp_126112, 'seek')
        # Calling seek(args, kwargs) (line 683)
        seek_call_result_126116 = invoke(stypy.reporting.localization.Localization(__file__, 683, 20), seek_126113, *[pos_126114], **kwargs_126115)
        
        # SSA join for if statement (line 675)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 652)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 686):
        
        # Assigning a Call to a Subscript (line 686):
        
        # Call to netcdf_variable(...): (line 686)
        # Processing the call arguments (line 686)
        # Getting the type of 'data' (line 687)
        data_126118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 20), 'data', False)
        # Getting the type of 'typecode' (line 687)
        typecode_126119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 26), 'typecode', False)
        # Getting the type of 'size' (line 687)
        size_126120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 36), 'size', False)
        # Getting the type of 'shape' (line 687)
        shape_126121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 42), 'shape', False)
        # Getting the type of 'dimensions' (line 687)
        dimensions_126122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 49), 'dimensions', False)
        # Getting the type of 'attributes' (line 687)
        attributes_126123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 61), 'attributes', False)
        # Processing the call keyword arguments (line 686)
        # Getting the type of 'self' (line 688)
        self_126124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 33), 'self', False)
        # Obtaining the member 'maskandscale' of a type (line 688)
        maskandscale_126125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 33), self_126124, 'maskandscale')
        keyword_126126 = maskandscale_126125
        kwargs_126127 = {'maskandscale': keyword_126126}
        # Getting the type of 'netcdf_variable' (line 686)
        netcdf_variable_126117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 35), 'netcdf_variable', False)
        # Calling netcdf_variable(args, kwargs) (line 686)
        netcdf_variable_call_result_126128 = invoke(stypy.reporting.localization.Localization(__file__, 686, 35), netcdf_variable_126117, *[data_126118, typecode_126119, size_126120, shape_126121, dimensions_126122, attributes_126123], **kwargs_126127)
        
        # Getting the type of 'self' (line 686)
        self_126129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'self')
        # Obtaining the member 'variables' of a type (line 686)
        variables_126130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 12), self_126129, 'variables')
        # Getting the type of 'name' (line 686)
        name_126131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 27), 'name')
        # Storing an element on a container (line 686)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 12), variables_126130, (name_126131, netcdf_variable_call_result_126128))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'rec_vars' (line 690)
        rec_vars_126132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 11), 'rec_vars')
        # Testing the type of an if condition (line 690)
        if_condition_126133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 690, 8), rec_vars_126132)
        # Assigning a type to the variable 'if_condition_126133' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'if_condition_126133', if_condition_126133)
        # SSA begins for if statement (line 690)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'rec_vars' (line 692)
        rec_vars_126135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 19), 'rec_vars', False)
        # Processing the call keyword arguments (line 692)
        kwargs_126136 = {}
        # Getting the type of 'len' (line 692)
        len_126134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 15), 'len', False)
        # Calling len(args, kwargs) (line 692)
        len_call_result_126137 = invoke(stypy.reporting.localization.Localization(__file__, 692, 15), len_126134, *[rec_vars_126135], **kwargs_126136)
        
        int_126138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 32), 'int')
        # Applying the binary operator '==' (line 692)
        result_eq_126139 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 15), '==', len_call_result_126137, int_126138)
        
        # Testing the type of an if condition (line 692)
        if_condition_126140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 12), result_eq_126139)
        # Assigning a type to the variable 'if_condition_126140' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 12), 'if_condition_126140', if_condition_126140)
        # SSA begins for if statement (line 692)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 693):
        
        # Assigning a Subscript to a Subscript (line 693):
        
        # Obtaining the type of the subscript
        int_126141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 51), 'int')
        slice_126142 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 693, 34), None, int_126141, None)
        
        # Obtaining the type of the subscript
        str_126143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 41), 'str', 'names')
        # Getting the type of 'dtypes' (line 693)
        dtypes_126144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 34), 'dtypes')
        # Obtaining the member '__getitem__' of a type (line 693)
        getitem___126145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 34), dtypes_126144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 693)
        subscript_call_result_126146 = invoke(stypy.reporting.localization.Localization(__file__, 693, 34), getitem___126145, str_126143)
        
        # Obtaining the member '__getitem__' of a type (line 693)
        getitem___126147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 34), subscript_call_result_126146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 693)
        subscript_call_result_126148 = invoke(stypy.reporting.localization.Localization(__file__, 693, 34), getitem___126147, slice_126142)
        
        # Getting the type of 'dtypes' (line 693)
        dtypes_126149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'dtypes')
        str_126150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 23), 'str', 'names')
        # Storing an element on a container (line 693)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 693, 16), dtypes_126149, (str_126150, subscript_call_result_126148))
        
        # Assigning a Subscript to a Subscript (line 694):
        
        # Assigning a Subscript to a Subscript (line 694):
        
        # Obtaining the type of the subscript
        int_126151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 55), 'int')
        slice_126152 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 694, 36), None, int_126151, None)
        
        # Obtaining the type of the subscript
        str_126153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 43), 'str', 'formats')
        # Getting the type of 'dtypes' (line 694)
        dtypes_126154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 36), 'dtypes')
        # Obtaining the member '__getitem__' of a type (line 694)
        getitem___126155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 36), dtypes_126154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 694)
        subscript_call_result_126156 = invoke(stypy.reporting.localization.Localization(__file__, 694, 36), getitem___126155, str_126153)
        
        # Obtaining the member '__getitem__' of a type (line 694)
        getitem___126157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 36), subscript_call_result_126156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 694)
        subscript_call_result_126158 = invoke(stypy.reporting.localization.Localization(__file__, 694, 36), getitem___126157, slice_126152)
        
        # Getting the type of 'dtypes' (line 694)
        dtypes_126159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'dtypes')
        str_126160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 23), 'str', 'formats')
        # Storing an element on a container (line 694)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 16), dtypes_126159, (str_126160, subscript_call_result_126158))
        # SSA join for if statement (line 692)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 697)
        self_126161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 15), 'self')
        # Obtaining the member 'use_mmap' of a type (line 697)
        use_mmap_126162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 15), self_126161, 'use_mmap')
        # Testing the type of an if condition (line 697)
        if_condition_126163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 12), use_mmap_126162)
        # Assigning a type to the variable 'if_condition_126163' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'if_condition_126163', if_condition_126163)
        # SSA begins for if statement (line 697)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 698):
        
        # Assigning a Call to a Name (line 698):
        
        # Call to view(...): (line 698)
        # Processing the call keyword arguments (line 698)
        # Getting the type of 'dtypes' (line 698)
        dtypes_126178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 90), 'dtypes', False)
        keyword_126179 = dtypes_126178
        kwargs_126180 = {'dtype': keyword_126179}
        
        # Obtaining the type of the subscript
        # Getting the type of 'begin' (line 698)
        begin_126164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 41), 'begin', False)
        # Getting the type of 'begin' (line 698)
        begin_126165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 47), 'begin', False)
        # Getting the type of 'self' (line 698)
        self_126166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 53), 'self', False)
        # Obtaining the member '_recs' of a type (line 698)
        _recs_126167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 53), self_126166, '_recs')
        # Getting the type of 'self' (line 698)
        self_126168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 64), 'self', False)
        # Obtaining the member '_recsize' of a type (line 698)
        _recsize_126169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 64), self_126168, '_recsize')
        # Applying the binary operator '*' (line 698)
        result_mul_126170 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 53), '*', _recs_126167, _recsize_126169)
        
        # Applying the binary operator '+' (line 698)
        result_add_126171 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 47), '+', begin_126165, result_mul_126170)
        
        slice_126172 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 698, 28), begin_126164, result_add_126171, None)
        # Getting the type of 'self' (line 698)
        self_126173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 28), 'self', False)
        # Obtaining the member '_mm_buf' of a type (line 698)
        _mm_buf_126174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 28), self_126173, '_mm_buf')
        # Obtaining the member '__getitem__' of a type (line 698)
        getitem___126175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 28), _mm_buf_126174, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 698)
        subscript_call_result_126176 = invoke(stypy.reporting.localization.Localization(__file__, 698, 28), getitem___126175, slice_126172)
        
        # Obtaining the member 'view' of a type (line 698)
        view_126177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 28), subscript_call_result_126176, 'view')
        # Calling view(args, kwargs) (line 698)
        view_call_result_126181 = invoke(stypy.reporting.localization.Localization(__file__, 698, 28), view_126177, *[], **kwargs_126180)
        
        # Assigning a type to the variable 'rec_array' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 16), 'rec_array', view_call_result_126181)
        
        # Assigning a Tuple to a Attribute (line 699):
        
        # Assigning a Tuple to a Attribute (line 699):
        
        # Obtaining an instance of the builtin type 'tuple' (line 699)
        tuple_126182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 699)
        # Adding element type (line 699)
        # Getting the type of 'self' (line 699)
        self_126183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 35), 'self')
        # Obtaining the member '_recs' of a type (line 699)
        _recs_126184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 35), self_126183, '_recs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 35), tuple_126182, _recs_126184)
        
        # Getting the type of 'rec_array' (line 699)
        rec_array_126185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 16), 'rec_array')
        # Setting the type of the member 'shape' of a type (line 699)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 16), rec_array_126185, 'shape', tuple_126182)
        # SSA branch for the else part of an if statement (line 697)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 701):
        
        # Assigning a Call to a Name (line 701):
        
        # Call to tell(...): (line 701)
        # Processing the call keyword arguments (line 701)
        kwargs_126189 = {}
        # Getting the type of 'self' (line 701)
        self_126186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 22), 'self', False)
        # Obtaining the member 'fp' of a type (line 701)
        fp_126187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 22), self_126186, 'fp')
        # Obtaining the member 'tell' of a type (line 701)
        tell_126188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 22), fp_126187, 'tell')
        # Calling tell(args, kwargs) (line 701)
        tell_call_result_126190 = invoke(stypy.reporting.localization.Localization(__file__, 701, 22), tell_126188, *[], **kwargs_126189)
        
        # Assigning a type to the variable 'pos' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'pos', tell_call_result_126190)
        
        # Call to seek(...): (line 702)
        # Processing the call arguments (line 702)
        # Getting the type of 'begin' (line 702)
        begin_126194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 29), 'begin', False)
        # Processing the call keyword arguments (line 702)
        kwargs_126195 = {}
        # Getting the type of 'self' (line 702)
        self_126191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'self', False)
        # Obtaining the member 'fp' of a type (line 702)
        fp_126192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 16), self_126191, 'fp')
        # Obtaining the member 'seek' of a type (line 702)
        seek_126193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 16), fp_126192, 'seek')
        # Calling seek(args, kwargs) (line 702)
        seek_call_result_126196 = invoke(stypy.reporting.localization.Localization(__file__, 702, 16), seek_126193, *[begin_126194], **kwargs_126195)
        
        
        # Assigning a Call to a Name (line 703):
        
        # Assigning a Call to a Name (line 703):
        
        # Call to fromstring(...): (line 703)
        # Processing the call arguments (line 703)
        
        # Call to read(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'self' (line 703)
        self_126201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 52), 'self', False)
        # Obtaining the member '_recs' of a type (line 703)
        _recs_126202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 52), self_126201, '_recs')
        # Getting the type of 'self' (line 703)
        self_126203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 63), 'self', False)
        # Obtaining the member '_recsize' of a type (line 703)
        _recsize_126204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 63), self_126203, '_recsize')
        # Applying the binary operator '*' (line 703)
        result_mul_126205 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 52), '*', _recs_126202, _recsize_126204)
        
        # Processing the call keyword arguments (line 703)
        kwargs_126206 = {}
        # Getting the type of 'self' (line 703)
        self_126198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 39), 'self', False)
        # Obtaining the member 'fp' of a type (line 703)
        fp_126199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 39), self_126198, 'fp')
        # Obtaining the member 'read' of a type (line 703)
        read_126200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 39), fp_126199, 'read')
        # Calling read(args, kwargs) (line 703)
        read_call_result_126207 = invoke(stypy.reporting.localization.Localization(__file__, 703, 39), read_126200, *[result_mul_126205], **kwargs_126206)
        
        # Processing the call keyword arguments (line 703)
        # Getting the type of 'dtypes' (line 703)
        dtypes_126208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 85), 'dtypes', False)
        keyword_126209 = dtypes_126208
        kwargs_126210 = {'dtype': keyword_126209}
        # Getting the type of 'fromstring' (line 703)
        fromstring_126197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 28), 'fromstring', False)
        # Calling fromstring(args, kwargs) (line 703)
        fromstring_call_result_126211 = invoke(stypy.reporting.localization.Localization(__file__, 703, 28), fromstring_126197, *[read_call_result_126207], **kwargs_126210)
        
        # Assigning a type to the variable 'rec_array' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), 'rec_array', fromstring_call_result_126211)
        
        # Assigning a Tuple to a Attribute (line 704):
        
        # Assigning a Tuple to a Attribute (line 704):
        
        # Obtaining an instance of the builtin type 'tuple' (line 704)
        tuple_126212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 704)
        # Adding element type (line 704)
        # Getting the type of 'self' (line 704)
        self_126213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 35), 'self')
        # Obtaining the member '_recs' of a type (line 704)
        _recs_126214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 35), self_126213, '_recs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 704, 35), tuple_126212, _recs_126214)
        
        # Getting the type of 'rec_array' (line 704)
        rec_array_126215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 16), 'rec_array')
        # Setting the type of the member 'shape' of a type (line 704)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 16), rec_array_126215, 'shape', tuple_126212)
        
        # Call to seek(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'pos' (line 705)
        pos_126219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 29), 'pos', False)
        # Processing the call keyword arguments (line 705)
        kwargs_126220 = {}
        # Getting the type of 'self' (line 705)
        self_126216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 16), 'self', False)
        # Obtaining the member 'fp' of a type (line 705)
        fp_126217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 16), self_126216, 'fp')
        # Obtaining the member 'seek' of a type (line 705)
        seek_126218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 16), fp_126217, 'seek')
        # Calling seek(args, kwargs) (line 705)
        seek_call_result_126221 = invoke(stypy.reporting.localization.Localization(__file__, 705, 16), seek_126218, *[pos_126219], **kwargs_126220)
        
        # SSA join for if statement (line 697)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'rec_vars' (line 707)
        rec_vars_126222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 23), 'rec_vars')
        # Testing the type of a for loop iterable (line 707)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 707, 12), rec_vars_126222)
        # Getting the type of the for loop variable (line 707)
        for_loop_var_126223 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 707, 12), rec_vars_126222)
        # Assigning a type to the variable 'var' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 12), 'var', for_loop_var_126223)
        # SSA begins for a for statement (line 707)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 708):
        
        # Assigning a Subscript to a Subscript (line 708):
        
        # Obtaining the type of the subscript
        # Getting the type of 'var' (line 708)
        var_126224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 65), 'var')
        # Getting the type of 'rec_array' (line 708)
        rec_array_126225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 55), 'rec_array')
        # Obtaining the member '__getitem__' of a type (line 708)
        getitem___126226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 55), rec_array_126225, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 708)
        subscript_call_result_126227 = invoke(stypy.reporting.localization.Localization(__file__, 708, 55), getitem___126226, var_126224)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'var' (line 708)
        var_126228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 31), 'var')
        # Getting the type of 'self' (line 708)
        self_126229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'self')
        # Obtaining the member 'variables' of a type (line 708)
        variables_126230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 16), self_126229, 'variables')
        # Obtaining the member '__getitem__' of a type (line 708)
        getitem___126231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 16), variables_126230, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 708)
        subscript_call_result_126232 = invoke(stypy.reporting.localization.Localization(__file__, 708, 16), getitem___126231, var_126228)
        
        # Obtaining the member '__dict__' of a type (line 708)
        dict___126233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 16), subscript_call_result_126232, '__dict__')
        str_126234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 45), 'str', 'data')
        # Storing an element on a container (line 708)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 16), dict___126233, (str_126234, subscript_call_result_126227))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 690)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_read_var_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_var_array' in the type store
        # Getting the type of 'stypy_return_type' (line 626)
        stypy_return_type_126235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_var_array'
        return stypy_return_type_126235


    @norecursion
    def _read_var(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_var'
        module_type_store = module_type_store.open_function_context('_read_var', 710, 4, False)
        # Assigning a type to the variable 'self' (line 711)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read_var.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read_var.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read_var.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read_var.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read_var')
        netcdf_file._read_var.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read_var.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read_var.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read_var.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read_var.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read_var.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read_var.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read_var', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_var', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_var(...)' code ##################

        
        # Assigning a Call to a Name (line 711):
        
        # Assigning a Call to a Name (line 711):
        
        # Call to asstr(...): (line 711)
        # Processing the call arguments (line 711)
        
        # Call to _unpack_string(...): (line 711)
        # Processing the call keyword arguments (line 711)
        kwargs_126239 = {}
        # Getting the type of 'self' (line 711)
        self_126237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 21), 'self', False)
        # Obtaining the member '_unpack_string' of a type (line 711)
        _unpack_string_126238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 21), self_126237, '_unpack_string')
        # Calling _unpack_string(args, kwargs) (line 711)
        _unpack_string_call_result_126240 = invoke(stypy.reporting.localization.Localization(__file__, 711, 21), _unpack_string_126238, *[], **kwargs_126239)
        
        # Processing the call keyword arguments (line 711)
        kwargs_126241 = {}
        # Getting the type of 'asstr' (line 711)
        asstr_126236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 15), 'asstr', False)
        # Calling asstr(args, kwargs) (line 711)
        asstr_call_result_126242 = invoke(stypy.reporting.localization.Localization(__file__, 711, 15), asstr_126236, *[_unpack_string_call_result_126240], **kwargs_126241)
        
        # Assigning a type to the variable 'name' (line 711)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'name', asstr_call_result_126242)
        
        # Assigning a List to a Name (line 712):
        
        # Assigning a List to a Name (line 712):
        
        # Obtaining an instance of the builtin type 'list' (line 712)
        list_126243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 712)
        
        # Assigning a type to the variable 'dimensions' (line 712)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'dimensions', list_126243)
        
        # Assigning a List to a Name (line 713):
        
        # Assigning a List to a Name (line 713):
        
        # Obtaining an instance of the builtin type 'list' (line 713)
        list_126244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 713)
        
        # Assigning a type to the variable 'shape' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'shape', list_126244)
        
        # Assigning a Call to a Name (line 714):
        
        # Assigning a Call to a Name (line 714):
        
        # Call to _unpack_int(...): (line 714)
        # Processing the call keyword arguments (line 714)
        kwargs_126247 = {}
        # Getting the type of 'self' (line 714)
        self_126245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 15), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 714)
        _unpack_int_126246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 15), self_126245, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 714)
        _unpack_int_call_result_126248 = invoke(stypy.reporting.localization.Localization(__file__, 714, 15), _unpack_int_126246, *[], **kwargs_126247)
        
        # Assigning a type to the variable 'dims' (line 714)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'dims', _unpack_int_call_result_126248)
        
        
        # Call to range(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'dims' (line 716)
        dims_126250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 23), 'dims', False)
        # Processing the call keyword arguments (line 716)
        kwargs_126251 = {}
        # Getting the type of 'range' (line 716)
        range_126249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 17), 'range', False)
        # Calling range(args, kwargs) (line 716)
        range_call_result_126252 = invoke(stypy.reporting.localization.Localization(__file__, 716, 17), range_126249, *[dims_126250], **kwargs_126251)
        
        # Testing the type of a for loop iterable (line 716)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 716, 8), range_call_result_126252)
        # Getting the type of the for loop variable (line 716)
        for_loop_var_126253 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 716, 8), range_call_result_126252)
        # Assigning a type to the variable 'i' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'i', for_loop_var_126253)
        # SSA begins for a for statement (line 716)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 717):
        
        # Assigning a Call to a Name (line 717):
        
        # Call to _unpack_int(...): (line 717)
        # Processing the call keyword arguments (line 717)
        kwargs_126256 = {}
        # Getting the type of 'self' (line 717)
        self_126254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 20), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 717)
        _unpack_int_126255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 20), self_126254, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 717)
        _unpack_int_call_result_126257 = invoke(stypy.reporting.localization.Localization(__file__, 717, 20), _unpack_int_126255, *[], **kwargs_126256)
        
        # Assigning a type to the variable 'dimid' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 12), 'dimid', _unpack_int_call_result_126257)
        
        # Assigning a Subscript to a Name (line 718):
        
        # Assigning a Subscript to a Name (line 718):
        
        # Obtaining the type of the subscript
        # Getting the type of 'dimid' (line 718)
        dimid_126258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 33), 'dimid')
        # Getting the type of 'self' (line 718)
        self_126259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 22), 'self')
        # Obtaining the member '_dims' of a type (line 718)
        _dims_126260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 22), self_126259, '_dims')
        # Obtaining the member '__getitem__' of a type (line 718)
        getitem___126261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 22), _dims_126260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 718)
        subscript_call_result_126262 = invoke(stypy.reporting.localization.Localization(__file__, 718, 22), getitem___126261, dimid_126258)
        
        # Assigning a type to the variable 'dimname' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 12), 'dimname', subscript_call_result_126262)
        
        # Call to append(...): (line 719)
        # Processing the call arguments (line 719)
        # Getting the type of 'dimname' (line 719)
        dimname_126265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 30), 'dimname', False)
        # Processing the call keyword arguments (line 719)
        kwargs_126266 = {}
        # Getting the type of 'dimensions' (line 719)
        dimensions_126263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'dimensions', False)
        # Obtaining the member 'append' of a type (line 719)
        append_126264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 12), dimensions_126263, 'append')
        # Calling append(args, kwargs) (line 719)
        append_call_result_126267 = invoke(stypy.reporting.localization.Localization(__file__, 719, 12), append_126264, *[dimname_126265], **kwargs_126266)
        
        
        # Assigning a Subscript to a Name (line 720):
        
        # Assigning a Subscript to a Name (line 720):
        
        # Obtaining the type of the subscript
        # Getting the type of 'dimname' (line 720)
        dimname_126268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 34), 'dimname')
        # Getting the type of 'self' (line 720)
        self_126269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 18), 'self')
        # Obtaining the member 'dimensions' of a type (line 720)
        dimensions_126270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 18), self_126269, 'dimensions')
        # Obtaining the member '__getitem__' of a type (line 720)
        getitem___126271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 18), dimensions_126270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 720)
        subscript_call_result_126272 = invoke(stypy.reporting.localization.Localization(__file__, 720, 18), getitem___126271, dimname_126268)
        
        # Assigning a type to the variable 'dim' (line 720)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 12), 'dim', subscript_call_result_126272)
        
        # Call to append(...): (line 721)
        # Processing the call arguments (line 721)
        # Getting the type of 'dim' (line 721)
        dim_126275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 25), 'dim', False)
        # Processing the call keyword arguments (line 721)
        kwargs_126276 = {}
        # Getting the type of 'shape' (line 721)
        shape_126273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'shape', False)
        # Obtaining the member 'append' of a type (line 721)
        append_126274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 12), shape_126273, 'append')
        # Calling append(args, kwargs) (line 721)
        append_call_result_126277 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), append_126274, *[dim_126275], **kwargs_126276)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 722):
        
        # Assigning a Call to a Name (line 722):
        
        # Call to tuple(...): (line 722)
        # Processing the call arguments (line 722)
        # Getting the type of 'dimensions' (line 722)
        dimensions_126279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 27), 'dimensions', False)
        # Processing the call keyword arguments (line 722)
        kwargs_126280 = {}
        # Getting the type of 'tuple' (line 722)
        tuple_126278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 21), 'tuple', False)
        # Calling tuple(args, kwargs) (line 722)
        tuple_call_result_126281 = invoke(stypy.reporting.localization.Localization(__file__, 722, 21), tuple_126278, *[dimensions_126279], **kwargs_126280)
        
        # Assigning a type to the variable 'dimensions' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'dimensions', tuple_call_result_126281)
        
        # Assigning a Call to a Name (line 723):
        
        # Assigning a Call to a Name (line 723):
        
        # Call to tuple(...): (line 723)
        # Processing the call arguments (line 723)
        # Getting the type of 'shape' (line 723)
        shape_126283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 22), 'shape', False)
        # Processing the call keyword arguments (line 723)
        kwargs_126284 = {}
        # Getting the type of 'tuple' (line 723)
        tuple_126282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 723)
        tuple_call_result_126285 = invoke(stypy.reporting.localization.Localization(__file__, 723, 16), tuple_126282, *[shape_126283], **kwargs_126284)
        
        # Assigning a type to the variable 'shape' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'shape', tuple_call_result_126285)
        
        # Assigning a Call to a Name (line 725):
        
        # Assigning a Call to a Name (line 725):
        
        # Call to _read_att_array(...): (line 725)
        # Processing the call keyword arguments (line 725)
        kwargs_126288 = {}
        # Getting the type of 'self' (line 725)
        self_126286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 21), 'self', False)
        # Obtaining the member '_read_att_array' of a type (line 725)
        _read_att_array_126287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 21), self_126286, '_read_att_array')
        # Calling _read_att_array(args, kwargs) (line 725)
        _read_att_array_call_result_126289 = invoke(stypy.reporting.localization.Localization(__file__, 725, 21), _read_att_array_126287, *[], **kwargs_126288)
        
        # Assigning a type to the variable 'attributes' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'attributes', _read_att_array_call_result_126289)
        
        # Assigning a Call to a Name (line 726):
        
        # Assigning a Call to a Name (line 726):
        
        # Call to read(...): (line 726)
        # Processing the call arguments (line 726)
        int_126293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 31), 'int')
        # Processing the call keyword arguments (line 726)
        kwargs_126294 = {}
        # Getting the type of 'self' (line 726)
        self_126290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 18), 'self', False)
        # Obtaining the member 'fp' of a type (line 726)
        fp_126291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 18), self_126290, 'fp')
        # Obtaining the member 'read' of a type (line 726)
        read_126292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 18), fp_126291, 'read')
        # Calling read(args, kwargs) (line 726)
        read_call_result_126295 = invoke(stypy.reporting.localization.Localization(__file__, 726, 18), read_126292, *[int_126293], **kwargs_126294)
        
        # Assigning a type to the variable 'nc_type' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'nc_type', read_call_result_126295)
        
        # Assigning a Call to a Name (line 727):
        
        # Assigning a Call to a Name (line 727):
        
        # Call to _unpack_int(...): (line 727)
        # Processing the call keyword arguments (line 727)
        kwargs_126298 = {}
        # Getting the type of 'self' (line 727)
        self_126296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 16), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 727)
        _unpack_int_126297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 16), self_126296, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 727)
        _unpack_int_call_result_126299 = invoke(stypy.reporting.localization.Localization(__file__, 727, 16), _unpack_int_126297, *[], **kwargs_126298)
        
        # Assigning a type to the variable 'vsize' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'vsize', _unpack_int_call_result_126299)
        
        # Assigning a Call to a Name (line 728):
        
        # Assigning a Call to a Name (line 728):
        
        # Call to (...): (line 728)
        # Processing the call keyword arguments (line 728)
        kwargs_126311 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 728)
        self_126300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 55), 'self', False)
        # Obtaining the member 'version_byte' of a type (line 728)
        version_byte_126301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 55), self_126300, 'version_byte')
        int_126302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 73), 'int')
        # Applying the binary operator '-' (line 728)
        result_sub_126303 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 55), '-', version_byte_126301, int_126302)
        
        
        # Obtaining an instance of the builtin type 'list' (line 728)
        list_126304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 728)
        # Adding element type (line 728)
        # Getting the type of 'self' (line 728)
        self_126305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 17), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 728)
        _unpack_int_126306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 17), self_126305, '_unpack_int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 728, 16), list_126304, _unpack_int_126306)
        # Adding element type (line 728)
        # Getting the type of 'self' (line 728)
        self_126307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 35), 'self', False)
        # Obtaining the member '_unpack_int64' of a type (line 728)
        _unpack_int64_126308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 35), self_126307, '_unpack_int64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 728, 16), list_126304, _unpack_int64_126308)
        
        # Obtaining the member '__getitem__' of a type (line 728)
        getitem___126309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 16), list_126304, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 728)
        subscript_call_result_126310 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), getitem___126309, result_sub_126303)
        
        # Calling (args, kwargs) (line 728)
        _call_result_126312 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), subscript_call_result_126310, *[], **kwargs_126311)
        
        # Assigning a type to the variable 'begin' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'begin', _call_result_126312)
        
        # Assigning a Subscript to a Tuple (line 730):
        
        # Assigning a Subscript to a Name (line 730):
        
        # Obtaining the type of the subscript
        int_126313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'nc_type' (line 730)
        nc_type_126314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 33), 'nc_type')
        # Getting the type of 'TYPEMAP' (line 730)
        TYPEMAP_126315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 25), 'TYPEMAP')
        # Obtaining the member '__getitem__' of a type (line 730)
        getitem___126316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 25), TYPEMAP_126315, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 730)
        subscript_call_result_126317 = invoke(stypy.reporting.localization.Localization(__file__, 730, 25), getitem___126316, nc_type_126314)
        
        # Obtaining the member '__getitem__' of a type (line 730)
        getitem___126318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 8), subscript_call_result_126317, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 730)
        subscript_call_result_126319 = invoke(stypy.reporting.localization.Localization(__file__, 730, 8), getitem___126318, int_126313)
        
        # Assigning a type to the variable 'tuple_var_assignment_124557' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'tuple_var_assignment_124557', subscript_call_result_126319)
        
        # Assigning a Subscript to a Name (line 730):
        
        # Obtaining the type of the subscript
        int_126320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'nc_type' (line 730)
        nc_type_126321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 33), 'nc_type')
        # Getting the type of 'TYPEMAP' (line 730)
        TYPEMAP_126322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 25), 'TYPEMAP')
        # Obtaining the member '__getitem__' of a type (line 730)
        getitem___126323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 25), TYPEMAP_126322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 730)
        subscript_call_result_126324 = invoke(stypy.reporting.localization.Localization(__file__, 730, 25), getitem___126323, nc_type_126321)
        
        # Obtaining the member '__getitem__' of a type (line 730)
        getitem___126325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 8), subscript_call_result_126324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 730)
        subscript_call_result_126326 = invoke(stypy.reporting.localization.Localization(__file__, 730, 8), getitem___126325, int_126320)
        
        # Assigning a type to the variable 'tuple_var_assignment_124558' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'tuple_var_assignment_124558', subscript_call_result_126326)
        
        # Assigning a Name to a Name (line 730):
        # Getting the type of 'tuple_var_assignment_124557' (line 730)
        tuple_var_assignment_124557_126327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'tuple_var_assignment_124557')
        # Assigning a type to the variable 'typecode' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'typecode', tuple_var_assignment_124557_126327)
        
        # Assigning a Name to a Name (line 730):
        # Getting the type of 'tuple_var_assignment_124558' (line 730)
        tuple_var_assignment_124558_126328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'tuple_var_assignment_124558')
        # Assigning a type to the variable 'size' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 18), 'size', tuple_var_assignment_124558_126328)
        
        # Assigning a BinOp to a Name (line 731):
        
        # Assigning a BinOp to a Name (line 731):
        str_126329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 17), 'str', '>%s')
        # Getting the type of 'typecode' (line 731)
        typecode_126330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 25), 'typecode')
        # Applying the binary operator '%' (line 731)
        result_mod_126331 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 17), '%', str_126329, typecode_126330)
        
        # Assigning a type to the variable 'dtype_' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'dtype_', result_mod_126331)
        
        # Obtaining an instance of the builtin type 'tuple' (line 733)
        tuple_126332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 733)
        # Adding element type (line 733)
        # Getting the type of 'name' (line 733)
        name_126333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 15), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, name_126333)
        # Adding element type (line 733)
        # Getting the type of 'dimensions' (line 733)
        dimensions_126334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 21), 'dimensions')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, dimensions_126334)
        # Adding element type (line 733)
        # Getting the type of 'shape' (line 733)
        shape_126335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 33), 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, shape_126335)
        # Adding element type (line 733)
        # Getting the type of 'attributes' (line 733)
        attributes_126336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 40), 'attributes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, attributes_126336)
        # Adding element type (line 733)
        # Getting the type of 'typecode' (line 733)
        typecode_126337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 52), 'typecode')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, typecode_126337)
        # Adding element type (line 733)
        # Getting the type of 'size' (line 733)
        size_126338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 62), 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, size_126338)
        # Adding element type (line 733)
        # Getting the type of 'dtype_' (line 733)
        dtype__126339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 68), 'dtype_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, dtype__126339)
        # Adding element type (line 733)
        # Getting the type of 'begin' (line 733)
        begin_126340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 76), 'begin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, begin_126340)
        # Adding element type (line 733)
        # Getting the type of 'vsize' (line 733)
        vsize_126341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 83), 'vsize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 15), tuple_126332, vsize_126341)
        
        # Assigning a type to the variable 'stypy_return_type' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'stypy_return_type', tuple_126332)
        
        # ################# End of '_read_var(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_var' in the type store
        # Getting the type of 'stypy_return_type' (line 710)
        stypy_return_type_126342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_var'
        return stypy_return_type_126342


    @norecursion
    def _read_values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_values'
        module_type_store = module_type_store.open_function_context('_read_values', 735, 4, False)
        # Assigning a type to the variable 'self' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._read_values.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._read_values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._read_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._read_values.__dict__.__setitem__('stypy_function_name', 'netcdf_file._read_values')
        netcdf_file._read_values.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._read_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._read_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._read_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._read_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._read_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._read_values.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._read_values', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_values', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_values(...)' code ##################

        
        # Assigning a Call to a Name (line 736):
        
        # Assigning a Call to a Name (line 736):
        
        # Call to read(...): (line 736)
        # Processing the call arguments (line 736)
        int_126346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 31), 'int')
        # Processing the call keyword arguments (line 736)
        kwargs_126347 = {}
        # Getting the type of 'self' (line 736)
        self_126343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 18), 'self', False)
        # Obtaining the member 'fp' of a type (line 736)
        fp_126344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 18), self_126343, 'fp')
        # Obtaining the member 'read' of a type (line 736)
        read_126345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 18), fp_126344, 'read')
        # Calling read(args, kwargs) (line 736)
        read_call_result_126348 = invoke(stypy.reporting.localization.Localization(__file__, 736, 18), read_126345, *[int_126346], **kwargs_126347)
        
        # Assigning a type to the variable 'nc_type' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 8), 'nc_type', read_call_result_126348)
        
        # Assigning a Call to a Name (line 737):
        
        # Assigning a Call to a Name (line 737):
        
        # Call to _unpack_int(...): (line 737)
        # Processing the call keyword arguments (line 737)
        kwargs_126351 = {}
        # Getting the type of 'self' (line 737)
        self_126349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 737)
        _unpack_int_126350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 12), self_126349, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 737)
        _unpack_int_call_result_126352 = invoke(stypy.reporting.localization.Localization(__file__, 737, 12), _unpack_int_126350, *[], **kwargs_126351)
        
        # Assigning a type to the variable 'n' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'n', _unpack_int_call_result_126352)
        
        # Assigning a Subscript to a Tuple (line 739):
        
        # Assigning a Subscript to a Name (line 739):
        
        # Obtaining the type of the subscript
        int_126353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'nc_type' (line 739)
        nc_type_126354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 33), 'nc_type')
        # Getting the type of 'TYPEMAP' (line 739)
        TYPEMAP_126355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 25), 'TYPEMAP')
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___126356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 25), TYPEMAP_126355, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_126357 = invoke(stypy.reporting.localization.Localization(__file__, 739, 25), getitem___126356, nc_type_126354)
        
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___126358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 8), subscript_call_result_126357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_126359 = invoke(stypy.reporting.localization.Localization(__file__, 739, 8), getitem___126358, int_126353)
        
        # Assigning a type to the variable 'tuple_var_assignment_124559' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'tuple_var_assignment_124559', subscript_call_result_126359)
        
        # Assigning a Subscript to a Name (line 739):
        
        # Obtaining the type of the subscript
        int_126360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 8), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'nc_type' (line 739)
        nc_type_126361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 33), 'nc_type')
        # Getting the type of 'TYPEMAP' (line 739)
        TYPEMAP_126362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 25), 'TYPEMAP')
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___126363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 25), TYPEMAP_126362, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_126364 = invoke(stypy.reporting.localization.Localization(__file__, 739, 25), getitem___126363, nc_type_126361)
        
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___126365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 8), subscript_call_result_126364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_126366 = invoke(stypy.reporting.localization.Localization(__file__, 739, 8), getitem___126365, int_126360)
        
        # Assigning a type to the variable 'tuple_var_assignment_124560' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'tuple_var_assignment_124560', subscript_call_result_126366)
        
        # Assigning a Name to a Name (line 739):
        # Getting the type of 'tuple_var_assignment_124559' (line 739)
        tuple_var_assignment_124559_126367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'tuple_var_assignment_124559')
        # Assigning a type to the variable 'typecode' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'typecode', tuple_var_assignment_124559_126367)
        
        # Assigning a Name to a Name (line 739):
        # Getting the type of 'tuple_var_assignment_124560' (line 739)
        tuple_var_assignment_124560_126368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'tuple_var_assignment_124560')
        # Assigning a type to the variable 'size' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 18), 'size', tuple_var_assignment_124560_126368)
        
        # Assigning a BinOp to a Name (line 741):
        
        # Assigning a BinOp to a Name (line 741):
        # Getting the type of 'n' (line 741)
        n_126369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'n')
        # Getting the type of 'size' (line 741)
        size_126370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 18), 'size')
        # Applying the binary operator '*' (line 741)
        result_mul_126371 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 16), '*', n_126369, size_126370)
        
        # Assigning a type to the variable 'count' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'count', result_mul_126371)
        
        # Assigning a Call to a Name (line 742):
        
        # Assigning a Call to a Name (line 742):
        
        # Call to read(...): (line 742)
        # Processing the call arguments (line 742)
        
        # Call to int(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'count' (line 742)
        count_126376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 34), 'count', False)
        # Processing the call keyword arguments (line 742)
        kwargs_126377 = {}
        # Getting the type of 'int' (line 742)
        int_126375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 30), 'int', False)
        # Calling int(args, kwargs) (line 742)
        int_call_result_126378 = invoke(stypy.reporting.localization.Localization(__file__, 742, 30), int_126375, *[count_126376], **kwargs_126377)
        
        # Processing the call keyword arguments (line 742)
        kwargs_126379 = {}
        # Getting the type of 'self' (line 742)
        self_126372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 17), 'self', False)
        # Obtaining the member 'fp' of a type (line 742)
        fp_126373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 17), self_126372, 'fp')
        # Obtaining the member 'read' of a type (line 742)
        read_126374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 17), fp_126373, 'read')
        # Calling read(args, kwargs) (line 742)
        read_call_result_126380 = invoke(stypy.reporting.localization.Localization(__file__, 742, 17), read_126374, *[int_call_result_126378], **kwargs_126379)
        
        # Assigning a type to the variable 'values' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'values', read_call_result_126380)
        
        # Call to read(...): (line 743)
        # Processing the call arguments (line 743)
        
        # Getting the type of 'count' (line 743)
        count_126384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 22), 'count', False)
        # Applying the 'usub' unary operator (line 743)
        result___neg___126385 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 21), 'usub', count_126384)
        
        int_126386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 30), 'int')
        # Applying the binary operator '%' (line 743)
        result_mod_126387 = python_operator(stypy.reporting.localization.Localization(__file__, 743, 21), '%', result___neg___126385, int_126386)
        
        # Processing the call keyword arguments (line 743)
        kwargs_126388 = {}
        # Getting the type of 'self' (line 743)
        self_126381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 743)
        fp_126382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), self_126381, 'fp')
        # Obtaining the member 'read' of a type (line 743)
        read_126383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), fp_126382, 'read')
        # Calling read(args, kwargs) (line 743)
        read_call_result_126389 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), read_126383, *[result_mod_126387], **kwargs_126388)
        
        
        
        # Getting the type of 'typecode' (line 745)
        typecode_126390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 11), 'typecode')
        str_126391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 27), 'str', 'c')
        # Applying the binary operator 'isnot' (line 745)
        result_is_not_126392 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 11), 'isnot', typecode_126390, str_126391)
        
        # Testing the type of an if condition (line 745)
        if_condition_126393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 745, 8), result_is_not_126392)
        # Assigning a type to the variable 'if_condition_126393' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'if_condition_126393', if_condition_126393)
        # SSA begins for if statement (line 745)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 746):
        
        # Assigning a Call to a Name (line 746):
        
        # Call to fromstring(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'values' (line 746)
        values_126395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 32), 'values', False)
        # Processing the call keyword arguments (line 746)
        str_126396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 46), 'str', '>%s')
        # Getting the type of 'typecode' (line 746)
        typecode_126397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 54), 'typecode', False)
        # Applying the binary operator '%' (line 746)
        result_mod_126398 = python_operator(stypy.reporting.localization.Localization(__file__, 746, 46), '%', str_126396, typecode_126397)
        
        keyword_126399 = result_mod_126398
        kwargs_126400 = {'dtype': keyword_126399}
        # Getting the type of 'fromstring' (line 746)
        fromstring_126394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 21), 'fromstring', False)
        # Calling fromstring(args, kwargs) (line 746)
        fromstring_call_result_126401 = invoke(stypy.reporting.localization.Localization(__file__, 746, 21), fromstring_126394, *[values_126395], **kwargs_126400)
        
        # Assigning a type to the variable 'values' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 12), 'values', fromstring_call_result_126401)
        
        
        # Getting the type of 'values' (line 747)
        values_126402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 15), 'values')
        # Obtaining the member 'shape' of a type (line 747)
        shape_126403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 15), values_126402, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 747)
        tuple_126404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 747)
        # Adding element type (line 747)
        int_126405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 747, 32), tuple_126404, int_126405)
        
        # Applying the binary operator '==' (line 747)
        result_eq_126406 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 15), '==', shape_126403, tuple_126404)
        
        # Testing the type of an if condition (line 747)
        if_condition_126407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 747, 12), result_eq_126406)
        # Assigning a type to the variable 'if_condition_126407' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'if_condition_126407', if_condition_126407)
        # SSA begins for if statement (line 747)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 748):
        
        # Assigning a Subscript to a Name (line 748):
        
        # Obtaining the type of the subscript
        int_126408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 32), 'int')
        # Getting the type of 'values' (line 748)
        values_126409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 25), 'values')
        # Obtaining the member '__getitem__' of a type (line 748)
        getitem___126410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 25), values_126409, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 748)
        subscript_call_result_126411 = invoke(stypy.reporting.localization.Localization(__file__, 748, 25), getitem___126410, int_126408)
        
        # Assigning a type to the variable 'values' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 16), 'values', subscript_call_result_126411)
        # SSA join for if statement (line 747)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 745)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 750):
        
        # Assigning a Call to a Name (line 750):
        
        # Call to rstrip(...): (line 750)
        # Processing the call arguments (line 750)
        str_126414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 35), 'str', '\x00')
        # Processing the call keyword arguments (line 750)
        kwargs_126415 = {}
        # Getting the type of 'values' (line 750)
        values_126412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 21), 'values', False)
        # Obtaining the member 'rstrip' of a type (line 750)
        rstrip_126413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 21), values_126412, 'rstrip')
        # Calling rstrip(args, kwargs) (line 750)
        rstrip_call_result_126416 = invoke(stypy.reporting.localization.Localization(__file__, 750, 21), rstrip_126413, *[str_126414], **kwargs_126415)
        
        # Assigning a type to the variable 'values' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'values', rstrip_call_result_126416)
        # SSA join for if statement (line 745)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'values' (line 751)
        values_126417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 15), 'values')
        # Assigning a type to the variable 'stypy_return_type' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'stypy_return_type', values_126417)
        
        # ################# End of '_read_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_values' in the type store
        # Getting the type of 'stypy_return_type' (line 735)
        stypy_return_type_126418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_values'
        return stypy_return_type_126418


    @norecursion
    def _pack_begin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pack_begin'
        module_type_store = module_type_store.open_function_context('_pack_begin', 753, 4, False)
        # Assigning a type to the variable 'self' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_function_name', 'netcdf_file._pack_begin')
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_param_names_list', ['begin'])
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._pack_begin.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._pack_begin', ['begin'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pack_begin', localization, ['begin'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pack_begin(...)' code ##################

        
        
        # Getting the type of 'self' (line 754)
        self_126419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 11), 'self')
        # Obtaining the member 'version_byte' of a type (line 754)
        version_byte_126420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 11), self_126419, 'version_byte')
        int_126421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 32), 'int')
        # Applying the binary operator '==' (line 754)
        result_eq_126422 = python_operator(stypy.reporting.localization.Localization(__file__, 754, 11), '==', version_byte_126420, int_126421)
        
        # Testing the type of an if condition (line 754)
        if_condition_126423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 754, 8), result_eq_126422)
        # Assigning a type to the variable 'if_condition_126423' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'if_condition_126423', if_condition_126423)
        # SSA begins for if statement (line 754)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _pack_int(...): (line 755)
        # Processing the call arguments (line 755)
        # Getting the type of 'begin' (line 755)
        begin_126426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 27), 'begin', False)
        # Processing the call keyword arguments (line 755)
        kwargs_126427 = {}
        # Getting the type of 'self' (line 755)
        self_126424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 755)
        _pack_int_126425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 12), self_126424, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 755)
        _pack_int_call_result_126428 = invoke(stypy.reporting.localization.Localization(__file__, 755, 12), _pack_int_126425, *[begin_126426], **kwargs_126427)
        
        # SSA branch for the else part of an if statement (line 754)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 756)
        self_126429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 13), 'self')
        # Obtaining the member 'version_byte' of a type (line 756)
        version_byte_126430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 13), self_126429, 'version_byte')
        int_126431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 34), 'int')
        # Applying the binary operator '==' (line 756)
        result_eq_126432 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 13), '==', version_byte_126430, int_126431)
        
        # Testing the type of an if condition (line 756)
        if_condition_126433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 756, 13), result_eq_126432)
        # Assigning a type to the variable 'if_condition_126433' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 13), 'if_condition_126433', if_condition_126433)
        # SSA begins for if statement (line 756)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _pack_int64(...): (line 757)
        # Processing the call arguments (line 757)
        # Getting the type of 'begin' (line 757)
        begin_126436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 29), 'begin', False)
        # Processing the call keyword arguments (line 757)
        kwargs_126437 = {}
        # Getting the type of 'self' (line 757)
        self_126434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 12), 'self', False)
        # Obtaining the member '_pack_int64' of a type (line 757)
        _pack_int64_126435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 12), self_126434, '_pack_int64')
        # Calling _pack_int64(args, kwargs) (line 757)
        _pack_int64_call_result_126438 = invoke(stypy.reporting.localization.Localization(__file__, 757, 12), _pack_int64_126435, *[begin_126436], **kwargs_126437)
        
        # SSA join for if statement (line 756)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 754)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_pack_begin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pack_begin' in the type store
        # Getting the type of 'stypy_return_type' (line 753)
        stypy_return_type_126439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126439)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pack_begin'
        return stypy_return_type_126439


    @norecursion
    def _pack_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pack_int'
        module_type_store = module_type_store.open_function_context('_pack_int', 759, 4, False)
        # Assigning a type to the variable 'self' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._pack_int.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_function_name', 'netcdf_file._pack_int')
        netcdf_file._pack_int.__dict__.__setitem__('stypy_param_names_list', ['value'])
        netcdf_file._pack_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._pack_int.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._pack_int', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pack_int', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pack_int(...)' code ##################

        
        # Call to write(...): (line 760)
        # Processing the call arguments (line 760)
        
        # Call to tostring(...): (line 760)
        # Processing the call keyword arguments (line 760)
        kwargs_126449 = {}
        
        # Call to array(...): (line 760)
        # Processing the call arguments (line 760)
        # Getting the type of 'value' (line 760)
        value_126444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 28), 'value', False)
        str_126445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 35), 'str', '>i')
        # Processing the call keyword arguments (line 760)
        kwargs_126446 = {}
        # Getting the type of 'array' (line 760)
        array_126443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 22), 'array', False)
        # Calling array(args, kwargs) (line 760)
        array_call_result_126447 = invoke(stypy.reporting.localization.Localization(__file__, 760, 22), array_126443, *[value_126444, str_126445], **kwargs_126446)
        
        # Obtaining the member 'tostring' of a type (line 760)
        tostring_126448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 22), array_call_result_126447, 'tostring')
        # Calling tostring(args, kwargs) (line 760)
        tostring_call_result_126450 = invoke(stypy.reporting.localization.Localization(__file__, 760, 22), tostring_126448, *[], **kwargs_126449)
        
        # Processing the call keyword arguments (line 760)
        kwargs_126451 = {}
        # Getting the type of 'self' (line 760)
        self_126440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 760)
        fp_126441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), self_126440, 'fp')
        # Obtaining the member 'write' of a type (line 760)
        write_126442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), fp_126441, 'write')
        # Calling write(args, kwargs) (line 760)
        write_call_result_126452 = invoke(stypy.reporting.localization.Localization(__file__, 760, 8), write_126442, *[tostring_call_result_126450], **kwargs_126451)
        
        
        # ################# End of '_pack_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pack_int' in the type store
        # Getting the type of 'stypy_return_type' (line 759)
        stypy_return_type_126453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pack_int'
        return stypy_return_type_126453

    
    # Assigning a Name to a Name (line 761):

    @norecursion
    def _unpack_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_unpack_int'
        module_type_store = module_type_store.open_function_context('_unpack_int', 763, 4, False)
        # Assigning a type to the variable 'self' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_function_name', 'netcdf_file._unpack_int')
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._unpack_int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._unpack_int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_unpack_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_unpack_int(...)' code ##################

        
        # Call to int(...): (line 764)
        # Processing the call arguments (line 764)
        
        # Obtaining the type of the subscript
        int_126455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 53), 'int')
        
        # Call to fromstring(...): (line 764)
        # Processing the call arguments (line 764)
        
        # Call to read(...): (line 764)
        # Processing the call arguments (line 764)
        int_126460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 43), 'int')
        # Processing the call keyword arguments (line 764)
        kwargs_126461 = {}
        # Getting the type of 'self' (line 764)
        self_126457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 30), 'self', False)
        # Obtaining the member 'fp' of a type (line 764)
        fp_126458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 30), self_126457, 'fp')
        # Obtaining the member 'read' of a type (line 764)
        read_126459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 30), fp_126458, 'read')
        # Calling read(args, kwargs) (line 764)
        read_call_result_126462 = invoke(stypy.reporting.localization.Localization(__file__, 764, 30), read_126459, *[int_126460], **kwargs_126461)
        
        str_126463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 47), 'str', '>i')
        # Processing the call keyword arguments (line 764)
        kwargs_126464 = {}
        # Getting the type of 'fromstring' (line 764)
        fromstring_126456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 19), 'fromstring', False)
        # Calling fromstring(args, kwargs) (line 764)
        fromstring_call_result_126465 = invoke(stypy.reporting.localization.Localization(__file__, 764, 19), fromstring_126456, *[read_call_result_126462, str_126463], **kwargs_126464)
        
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___126466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 19), fromstring_call_result_126465, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_126467 = invoke(stypy.reporting.localization.Localization(__file__, 764, 19), getitem___126466, int_126455)
        
        # Processing the call keyword arguments (line 764)
        kwargs_126468 = {}
        # Getting the type of 'int' (line 764)
        int_126454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 15), 'int', False)
        # Calling int(args, kwargs) (line 764)
        int_call_result_126469 = invoke(stypy.reporting.localization.Localization(__file__, 764, 15), int_126454, *[subscript_call_result_126467], **kwargs_126468)
        
        # Assigning a type to the variable 'stypy_return_type' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'stypy_return_type', int_call_result_126469)
        
        # ################# End of '_unpack_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_unpack_int' in the type store
        # Getting the type of 'stypy_return_type' (line 763)
        stypy_return_type_126470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_unpack_int'
        return stypy_return_type_126470

    
    # Assigning a Name to a Name (line 765):

    @norecursion
    def _pack_int64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pack_int64'
        module_type_store = module_type_store.open_function_context('_pack_int64', 767, 4, False)
        # Assigning a type to the variable 'self' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_function_name', 'netcdf_file._pack_int64')
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_param_names_list', ['value'])
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._pack_int64.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._pack_int64', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pack_int64', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pack_int64(...)' code ##################

        
        # Call to write(...): (line 768)
        # Processing the call arguments (line 768)
        
        # Call to tostring(...): (line 768)
        # Processing the call keyword arguments (line 768)
        kwargs_126480 = {}
        
        # Call to array(...): (line 768)
        # Processing the call arguments (line 768)
        # Getting the type of 'value' (line 768)
        value_126475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 28), 'value', False)
        str_126476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 35), 'str', '>q')
        # Processing the call keyword arguments (line 768)
        kwargs_126477 = {}
        # Getting the type of 'array' (line 768)
        array_126474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 22), 'array', False)
        # Calling array(args, kwargs) (line 768)
        array_call_result_126478 = invoke(stypy.reporting.localization.Localization(__file__, 768, 22), array_126474, *[value_126475, str_126476], **kwargs_126477)
        
        # Obtaining the member 'tostring' of a type (line 768)
        tostring_126479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 22), array_call_result_126478, 'tostring')
        # Calling tostring(args, kwargs) (line 768)
        tostring_call_result_126481 = invoke(stypy.reporting.localization.Localization(__file__, 768, 22), tostring_126479, *[], **kwargs_126480)
        
        # Processing the call keyword arguments (line 768)
        kwargs_126482 = {}
        # Getting the type of 'self' (line 768)
        self_126471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 768)
        fp_126472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 8), self_126471, 'fp')
        # Obtaining the member 'write' of a type (line 768)
        write_126473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 8), fp_126472, 'write')
        # Calling write(args, kwargs) (line 768)
        write_call_result_126483 = invoke(stypy.reporting.localization.Localization(__file__, 768, 8), write_126473, *[tostring_call_result_126481], **kwargs_126482)
        
        
        # ################# End of '_pack_int64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pack_int64' in the type store
        # Getting the type of 'stypy_return_type' (line 767)
        stypy_return_type_126484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126484)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pack_int64'
        return stypy_return_type_126484


    @norecursion
    def _unpack_int64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_unpack_int64'
        module_type_store = module_type_store.open_function_context('_unpack_int64', 770, 4, False)
        # Assigning a type to the variable 'self' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_function_name', 'netcdf_file._unpack_int64')
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._unpack_int64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._unpack_int64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_unpack_int64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_unpack_int64(...)' code ##################

        
        # Obtaining the type of the subscript
        int_126485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 49), 'int')
        
        # Call to fromstring(...): (line 771)
        # Processing the call arguments (line 771)
        
        # Call to read(...): (line 771)
        # Processing the call arguments (line 771)
        int_126490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 39), 'int')
        # Processing the call keyword arguments (line 771)
        kwargs_126491 = {}
        # Getting the type of 'self' (line 771)
        self_126487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 26), 'self', False)
        # Obtaining the member 'fp' of a type (line 771)
        fp_126488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 26), self_126487, 'fp')
        # Obtaining the member 'read' of a type (line 771)
        read_126489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 26), fp_126488, 'read')
        # Calling read(args, kwargs) (line 771)
        read_call_result_126492 = invoke(stypy.reporting.localization.Localization(__file__, 771, 26), read_126489, *[int_126490], **kwargs_126491)
        
        str_126493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 43), 'str', '>q')
        # Processing the call keyword arguments (line 771)
        kwargs_126494 = {}
        # Getting the type of 'fromstring' (line 771)
        fromstring_126486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 15), 'fromstring', False)
        # Calling fromstring(args, kwargs) (line 771)
        fromstring_call_result_126495 = invoke(stypy.reporting.localization.Localization(__file__, 771, 15), fromstring_126486, *[read_call_result_126492, str_126493], **kwargs_126494)
        
        # Obtaining the member '__getitem__' of a type (line 771)
        getitem___126496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 15), fromstring_call_result_126495, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 771)
        subscript_call_result_126497 = invoke(stypy.reporting.localization.Localization(__file__, 771, 15), getitem___126496, int_126485)
        
        # Assigning a type to the variable 'stypy_return_type' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'stypy_return_type', subscript_call_result_126497)
        
        # ################# End of '_unpack_int64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_unpack_int64' in the type store
        # Getting the type of 'stypy_return_type' (line 770)
        stypy_return_type_126498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126498)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_unpack_int64'
        return stypy_return_type_126498


    @norecursion
    def _pack_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pack_string'
        module_type_store = module_type_store.open_function_context('_pack_string', 773, 4, False)
        # Assigning a type to the variable 'self' (line 774)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._pack_string.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_function_name', 'netcdf_file._pack_string')
        netcdf_file._pack_string.__dict__.__setitem__('stypy_param_names_list', ['s'])
        netcdf_file._pack_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._pack_string.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._pack_string', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pack_string', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pack_string(...)' code ##################

        
        # Assigning a Call to a Name (line 774):
        
        # Assigning a Call to a Name (line 774):
        
        # Call to len(...): (line 774)
        # Processing the call arguments (line 774)
        # Getting the type of 's' (line 774)
        s_126500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 20), 's', False)
        # Processing the call keyword arguments (line 774)
        kwargs_126501 = {}
        # Getting the type of 'len' (line 774)
        len_126499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 16), 'len', False)
        # Calling len(args, kwargs) (line 774)
        len_call_result_126502 = invoke(stypy.reporting.localization.Localization(__file__, 774, 16), len_126499, *[s_126500], **kwargs_126501)
        
        # Assigning a type to the variable 'count' (line 774)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'count', len_call_result_126502)
        
        # Call to _pack_int(...): (line 775)
        # Processing the call arguments (line 775)
        # Getting the type of 'count' (line 775)
        count_126505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 23), 'count', False)
        # Processing the call keyword arguments (line 775)
        kwargs_126506 = {}
        # Getting the type of 'self' (line 775)
        self_126503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'self', False)
        # Obtaining the member '_pack_int' of a type (line 775)
        _pack_int_126504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 8), self_126503, '_pack_int')
        # Calling _pack_int(args, kwargs) (line 775)
        _pack_int_call_result_126507 = invoke(stypy.reporting.localization.Localization(__file__, 775, 8), _pack_int_126504, *[count_126505], **kwargs_126506)
        
        
        # Call to write(...): (line 776)
        # Processing the call arguments (line 776)
        
        # Call to asbytes(...): (line 776)
        # Processing the call arguments (line 776)
        # Getting the type of 's' (line 776)
        s_126512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 30), 's', False)
        # Processing the call keyword arguments (line 776)
        kwargs_126513 = {}
        # Getting the type of 'asbytes' (line 776)
        asbytes_126511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 22), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 776)
        asbytes_call_result_126514 = invoke(stypy.reporting.localization.Localization(__file__, 776, 22), asbytes_126511, *[s_126512], **kwargs_126513)
        
        # Processing the call keyword arguments (line 776)
        kwargs_126515 = {}
        # Getting the type of 'self' (line 776)
        self_126508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 776)
        fp_126509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 8), self_126508, 'fp')
        # Obtaining the member 'write' of a type (line 776)
        write_126510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 8), fp_126509, 'write')
        # Calling write(args, kwargs) (line 776)
        write_call_result_126516 = invoke(stypy.reporting.localization.Localization(__file__, 776, 8), write_126510, *[asbytes_call_result_126514], **kwargs_126515)
        
        
        # Call to write(...): (line 777)
        # Processing the call arguments (line 777)
        str_126520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 22), 'str', '0')
        
        # Getting the type of 'count' (line 777)
        count_126521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 31), 'count', False)
        # Applying the 'usub' unary operator (line 777)
        result___neg___126522 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 30), 'usub', count_126521)
        
        int_126523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 39), 'int')
        # Applying the binary operator '%' (line 777)
        result_mod_126524 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 30), '%', result___neg___126522, int_126523)
        
        # Applying the binary operator '*' (line 777)
        result_mul_126525 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 22), '*', str_126520, result_mod_126524)
        
        # Processing the call keyword arguments (line 777)
        kwargs_126526 = {}
        # Getting the type of 'self' (line 777)
        self_126517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 777)
        fp_126518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 8), self_126517, 'fp')
        # Obtaining the member 'write' of a type (line 777)
        write_126519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 8), fp_126518, 'write')
        # Calling write(args, kwargs) (line 777)
        write_call_result_126527 = invoke(stypy.reporting.localization.Localization(__file__, 777, 8), write_126519, *[result_mul_126525], **kwargs_126526)
        
        
        # ################# End of '_pack_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pack_string' in the type store
        # Getting the type of 'stypy_return_type' (line 773)
        stypy_return_type_126528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pack_string'
        return stypy_return_type_126528


    @norecursion
    def _unpack_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_unpack_string'
        module_type_store = module_type_store.open_function_context('_unpack_string', 779, 4, False)
        # Assigning a type to the variable 'self' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_localization', localization)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_function_name', 'netcdf_file._unpack_string')
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_file._unpack_string.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_file._unpack_string', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_unpack_string', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_unpack_string(...)' code ##################

        
        # Assigning a Call to a Name (line 780):
        
        # Assigning a Call to a Name (line 780):
        
        # Call to _unpack_int(...): (line 780)
        # Processing the call keyword arguments (line 780)
        kwargs_126531 = {}
        # Getting the type of 'self' (line 780)
        self_126529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 16), 'self', False)
        # Obtaining the member '_unpack_int' of a type (line 780)
        _unpack_int_126530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 16), self_126529, '_unpack_int')
        # Calling _unpack_int(args, kwargs) (line 780)
        _unpack_int_call_result_126532 = invoke(stypy.reporting.localization.Localization(__file__, 780, 16), _unpack_int_126530, *[], **kwargs_126531)
        
        # Assigning a type to the variable 'count' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'count', _unpack_int_call_result_126532)
        
        # Assigning a Call to a Name (line 781):
        
        # Assigning a Call to a Name (line 781):
        
        # Call to rstrip(...): (line 781)
        # Processing the call arguments (line 781)
        str_126540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 39), 'str', '\x00')
        # Processing the call keyword arguments (line 781)
        kwargs_126541 = {}
        
        # Call to read(...): (line 781)
        # Processing the call arguments (line 781)
        # Getting the type of 'count' (line 781)
        count_126536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 25), 'count', False)
        # Processing the call keyword arguments (line 781)
        kwargs_126537 = {}
        # Getting the type of 'self' (line 781)
        self_126533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'self', False)
        # Obtaining the member 'fp' of a type (line 781)
        fp_126534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), self_126533, 'fp')
        # Obtaining the member 'read' of a type (line 781)
        read_126535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), fp_126534, 'read')
        # Calling read(args, kwargs) (line 781)
        read_call_result_126538 = invoke(stypy.reporting.localization.Localization(__file__, 781, 12), read_126535, *[count_126536], **kwargs_126537)
        
        # Obtaining the member 'rstrip' of a type (line 781)
        rstrip_126539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), read_call_result_126538, 'rstrip')
        # Calling rstrip(args, kwargs) (line 781)
        rstrip_call_result_126542 = invoke(stypy.reporting.localization.Localization(__file__, 781, 12), rstrip_126539, *[str_126540], **kwargs_126541)
        
        # Assigning a type to the variable 's' (line 781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 8), 's', rstrip_call_result_126542)
        
        # Call to read(...): (line 782)
        # Processing the call arguments (line 782)
        
        # Getting the type of 'count' (line 782)
        count_126546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 22), 'count', False)
        # Applying the 'usub' unary operator (line 782)
        result___neg___126547 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 21), 'usub', count_126546)
        
        int_126548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 30), 'int')
        # Applying the binary operator '%' (line 782)
        result_mod_126549 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 21), '%', result___neg___126547, int_126548)
        
        # Processing the call keyword arguments (line 782)
        kwargs_126550 = {}
        # Getting the type of 'self' (line 782)
        self_126543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'self', False)
        # Obtaining the member 'fp' of a type (line 782)
        fp_126544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 8), self_126543, 'fp')
        # Obtaining the member 'read' of a type (line 782)
        read_126545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 8), fp_126544, 'read')
        # Calling read(args, kwargs) (line 782)
        read_call_result_126551 = invoke(stypy.reporting.localization.Localization(__file__, 782, 8), read_126545, *[result_mod_126549], **kwargs_126550)
        
        # Getting the type of 's' (line 783)
        s_126552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'stypy_return_type', s_126552)
        
        # ################# End of '_unpack_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_unpack_string' in the type store
        # Getting the type of 'stypy_return_type' (line 779)
        stypy_return_type_126553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126553)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_unpack_string'
        return stypy_return_type_126553


# Assigning a type to the variable 'netcdf_file' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'netcdf_file', netcdf_file)

# Assigning a Name to a Name (line 302):
# Getting the type of 'netcdf_file'
netcdf_file_126554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Obtaining the member 'close' of a type
close_126555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126554, 'close')
# Getting the type of 'netcdf_file'
netcdf_file_126556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Setting the type of the member '__del__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126556, '__del__', close_126555)

# Assigning a Name to a Name (line 392):
# Getting the type of 'netcdf_file'
netcdf_file_126557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Obtaining the member 'flush' of a type
flush_126558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126557, 'flush')
# Getting the type of 'netcdf_file'
netcdf_file_126559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Setting the type of the member 'sync' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126559, 'sync', flush_126558)

# Assigning a Name to a Name (line 761):
# Getting the type of 'netcdf_file'
netcdf_file_126560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Obtaining the member '_pack_int' of a type
_pack_int_126561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126560, '_pack_int')
# Getting the type of 'netcdf_file'
netcdf_file_126562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Setting the type of the member '_pack_int32' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126562, '_pack_int32', _pack_int_126561)

# Assigning a Name to a Name (line 765):
# Getting the type of 'netcdf_file'
netcdf_file_126563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Obtaining the member '_unpack_int' of a type
_unpack_int_126564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126563, '_unpack_int')
# Getting the type of 'netcdf_file'
netcdf_file_126565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_file')
# Setting the type of the member '_unpack_int32' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_file_126565, '_unpack_int32', _unpack_int_126564)
# Declaration of the 'netcdf_variable' class

class netcdf_variable(object, ):
    str_126566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, (-1)), 'str', "\n    A data object for the `netcdf` module.\n\n    `netcdf_variable` objects are constructed by calling the method\n    `netcdf_file.createVariable` on the `netcdf_file` object. `netcdf_variable`\n    objects behave much like array objects defined in numpy, except that their\n    data resides in a file. Data is read by indexing and written by assigning\n    to an indexed subset; the entire array can be accessed by the index ``[:]``\n    or (for scalars) by using the methods `getValue` and `assignValue`.\n    `netcdf_variable` objects also have attribute `shape` with the same meaning\n    as for arrays, but the shape cannot be modified. There is another read-only\n    attribute `dimensions`, whose value is the tuple of dimension names.\n\n    All other attributes correspond to variable attributes defined in\n    the NetCDF file. Variable attributes are created by assigning to an\n    attribute of the `netcdf_variable` object.\n\n    Parameters\n    ----------\n    data : array_like\n        The data array that holds the values for the variable.\n        Typically, this is initialized as empty, but with the proper shape.\n    typecode : dtype character code\n        Desired data-type for the data array.\n    size : int\n        Desired element size for the data array.\n    shape : sequence of ints\n        The shape of the array.  This should match the lengths of the\n        variable's dimensions.\n    dimensions : sequence of strings\n        The names of the dimensions used by the variable.  Must be in the\n        same order of the dimension lengths given by `shape`.\n    attributes : dict, optional\n        Attribute values (any type) keyed by string names.  These attributes\n        become attributes for the netcdf_variable object.\n    maskandscale : bool, optional\n        Whether to automatically scale and/or mask data based on attributes.\n        Default is False.\n\n\n    Attributes\n    ----------\n    dimensions : list of str\n        List of names of dimensions used by the variable object.\n    isrec, shape\n        Properties\n\n    See also\n    --------\n    isrec, shape\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 840)
        None_126567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 28), 'None')
        # Getting the type of 'False' (line 841)
        False_126568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 30), 'False')
        defaults = [None_126567, False_126568]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 839, 4, False)
        # Assigning a type to the variable 'self' (line 840)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.__init__', ['data', 'typecode', 'size', 'shape', 'dimensions', 'attributes', 'maskandscale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data', 'typecode', 'size', 'shape', 'dimensions', 'attributes', 'maskandscale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 842):
        
        # Assigning a Name to a Attribute (line 842):
        # Getting the type of 'data' (line 842)
        data_126569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 20), 'data')
        # Getting the type of 'self' (line 842)
        self_126570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'self')
        # Setting the type of the member 'data' of a type (line 842)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 8), self_126570, 'data', data_126569)
        
        # Assigning a Name to a Attribute (line 843):
        
        # Assigning a Name to a Attribute (line 843):
        # Getting the type of 'typecode' (line 843)
        typecode_126571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 25), 'typecode')
        # Getting the type of 'self' (line 843)
        self_126572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'self')
        # Setting the type of the member '_typecode' of a type (line 843)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 8), self_126572, '_typecode', typecode_126571)
        
        # Assigning a Name to a Attribute (line 844):
        
        # Assigning a Name to a Attribute (line 844):
        # Getting the type of 'size' (line 844)
        size_126573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 21), 'size')
        # Getting the type of 'self' (line 844)
        self_126574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'self')
        # Setting the type of the member '_size' of a type (line 844)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 8), self_126574, '_size', size_126573)
        
        # Assigning a Name to a Attribute (line 845):
        
        # Assigning a Name to a Attribute (line 845):
        # Getting the type of 'shape' (line 845)
        shape_126575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 22), 'shape')
        # Getting the type of 'self' (line 845)
        self_126576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'self')
        # Setting the type of the member '_shape' of a type (line 845)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 8), self_126576, '_shape', shape_126575)
        
        # Assigning a Name to a Attribute (line 846):
        
        # Assigning a Name to a Attribute (line 846):
        # Getting the type of 'dimensions' (line 846)
        dimensions_126577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 26), 'dimensions')
        # Getting the type of 'self' (line 846)
        self_126578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'self')
        # Setting the type of the member 'dimensions' of a type (line 846)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), self_126578, 'dimensions', dimensions_126577)
        
        # Assigning a Name to a Attribute (line 847):
        
        # Assigning a Name to a Attribute (line 847):
        # Getting the type of 'maskandscale' (line 847)
        maskandscale_126579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 28), 'maskandscale')
        # Getting the type of 'self' (line 847)
        self_126580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 8), 'self')
        # Setting the type of the member 'maskandscale' of a type (line 847)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 8), self_126580, 'maskandscale', maskandscale_126579)
        
        # Assigning a BoolOp to a Attribute (line 849):
        
        # Assigning a BoolOp to a Attribute (line 849):
        
        # Evaluating a boolean operation
        # Getting the type of 'attributes' (line 849)
        attributes_126581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 27), 'attributes')
        
        # Call to OrderedDict(...): (line 849)
        # Processing the call keyword arguments (line 849)
        kwargs_126583 = {}
        # Getting the type of 'OrderedDict' (line 849)
        OrderedDict_126582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 41), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 849)
        OrderedDict_call_result_126584 = invoke(stypy.reporting.localization.Localization(__file__, 849, 41), OrderedDict_126582, *[], **kwargs_126583)
        
        # Applying the binary operator 'or' (line 849)
        result_or_keyword_126585 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 27), 'or', attributes_126581, OrderedDict_call_result_126584)
        
        # Getting the type of 'self' (line 849)
        self_126586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'self')
        # Setting the type of the member '_attributes' of a type (line 849)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 8), self_126586, '_attributes', result_or_keyword_126585)
        
        
        # Call to items(...): (line 850)
        # Processing the call keyword arguments (line 850)
        kwargs_126590 = {}
        # Getting the type of 'self' (line 850)
        self_126587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 20), 'self', False)
        # Obtaining the member '_attributes' of a type (line 850)
        _attributes_126588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 20), self_126587, '_attributes')
        # Obtaining the member 'items' of a type (line 850)
        items_126589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 20), _attributes_126588, 'items')
        # Calling items(args, kwargs) (line 850)
        items_call_result_126591 = invoke(stypy.reporting.localization.Localization(__file__, 850, 20), items_126589, *[], **kwargs_126590)
        
        # Testing the type of a for loop iterable (line 850)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 850, 8), items_call_result_126591)
        # Getting the type of the for loop variable (line 850)
        for_loop_var_126592 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 850, 8), items_call_result_126591)
        # Assigning a type to the variable 'k' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 850, 8), for_loop_var_126592))
        # Assigning a type to the variable 'v' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 850, 8), for_loop_var_126592))
        # SSA begins for a for statement (line 850)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 851):
        
        # Assigning a Name to a Subscript (line 851):
        # Getting the type of 'v' (line 851)
        v_126593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 31), 'v')
        # Getting the type of 'self' (line 851)
        self_126594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 12), 'self')
        # Obtaining the member '__dict__' of a type (line 851)
        dict___126595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 12), self_126594, '__dict__')
        # Getting the type of 'k' (line 851)
        k_126596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 26), 'k')
        # Storing an element on a container (line 851)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 851, 12), dict___126595, (k_126596, v_126593))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __setattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setattr__'
        module_type_store = module_type_store.open_function_context('__setattr__', 853, 4, False)
        # Assigning a type to the variable 'self' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.__setattr__')
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_param_names_list', ['attr', 'value'])
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.__setattr__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.__setattr__', ['attr', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setattr__', localization, ['attr', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setattr__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 856)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Subscript (line 857):
        
        # Assigning a Name to a Subscript (line 857):
        # Getting the type of 'value' (line 857)
        value_126597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 37), 'value')
        # Getting the type of 'self' (line 857)
        self_126598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 12), 'self')
        # Obtaining the member '_attributes' of a type (line 857)
        _attributes_126599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 12), self_126598, '_attributes')
        # Getting the type of 'attr' (line 857)
        attr_126600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 29), 'attr')
        # Storing an element on a container (line 857)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 857, 12), _attributes_126599, (attr_126600, value_126597))
        # SSA branch for the except part of a try statement (line 856)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 856)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 856)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 860):
        
        # Assigning a Name to a Subscript (line 860):
        # Getting the type of 'value' (line 860)
        value_126601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 30), 'value')
        # Getting the type of 'self' (line 860)
        self_126602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'self')
        # Obtaining the member '__dict__' of a type (line 860)
        dict___126603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 8), self_126602, '__dict__')
        # Getting the type of 'attr' (line 860)
        attr_126604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 22), 'attr')
        # Storing an element on a container (line 860)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 860, 8), dict___126603, (attr_126604, value_126601))
        
        # ################# End of '__setattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 853)
        stypy_return_type_126605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setattr__'
        return stypy_return_type_126605


    @norecursion
    def isrec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isrec'
        module_type_store = module_type_store.open_function_context('isrec', 862, 4, False)
        # Assigning a type to the variable 'self' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.isrec.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.isrec')
        netcdf_variable.isrec.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_variable.isrec.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.isrec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.isrec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isrec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isrec(...)' code ##################

        str_126606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, (-1)), 'str', 'Returns whether the variable has a record dimension or not.\n\n        A record dimension is a dimension along which additional data could be\n        easily appended in the netcdf data structure without much rewriting of\n        the data file. This attribute is a read-only property of the\n        `netcdf_variable`.\n\n        ')
        
        # Evaluating a boolean operation
        
        # Call to bool(...): (line 871)
        # Processing the call arguments (line 871)
        # Getting the type of 'self' (line 871)
        self_126608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 20), 'self', False)
        # Obtaining the member 'data' of a type (line 871)
        data_126609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 20), self_126608, 'data')
        # Obtaining the member 'shape' of a type (line 871)
        shape_126610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 20), data_126609, 'shape')
        # Processing the call keyword arguments (line 871)
        kwargs_126611 = {}
        # Getting the type of 'bool' (line 871)
        bool_126607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 15), 'bool', False)
        # Calling bool(args, kwargs) (line 871)
        bool_call_result_126612 = invoke(stypy.reporting.localization.Localization(__file__, 871, 15), bool_126607, *[shape_126610], **kwargs_126611)
        
        
        
        # Obtaining the type of the subscript
        int_126613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 57), 'int')
        # Getting the type of 'self' (line 871)
        self_126614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 45), 'self')
        # Obtaining the member '_shape' of a type (line 871)
        _shape_126615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 45), self_126614, '_shape')
        # Obtaining the member '__getitem__' of a type (line 871)
        getitem___126616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 45), _shape_126615, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 871)
        subscript_call_result_126617 = invoke(stypy.reporting.localization.Localization(__file__, 871, 45), getitem___126616, int_126613)
        
        # Applying the 'not' unary operator (line 871)
        result_not__126618 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 41), 'not', subscript_call_result_126617)
        
        # Applying the binary operator 'and' (line 871)
        result_and_keyword_126619 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 15), 'and', bool_call_result_126612, result_not__126618)
        
        # Assigning a type to the variable 'stypy_return_type' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'stypy_return_type', result_and_keyword_126619)
        
        # ################# End of 'isrec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isrec' in the type store
        # Getting the type of 'stypy_return_type' (line 862)
        stypy_return_type_126620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126620)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isrec'
        return stypy_return_type_126620

    
    # Assigning a Call to a Name (line 872):

    @norecursion
    def shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shape'
        module_type_store = module_type_store.open_function_context('shape', 874, 4, False)
        # Assigning a type to the variable 'self' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.shape.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.shape.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.shape')
        netcdf_variable.shape.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_variable.shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shape(...)' code ##################

        str_126621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, (-1)), 'str', 'Returns the shape tuple of the data variable.\n\n        This is a read-only attribute and can not be modified in the\n        same manner of other numpy arrays.\n        ')
        # Getting the type of 'self' (line 880)
        self_126622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 15), 'self')
        # Obtaining the member 'data' of a type (line 880)
        data_126623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 15), self_126622, 'data')
        # Obtaining the member 'shape' of a type (line 880)
        shape_126624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 15), data_126623, 'shape')
        # Assigning a type to the variable 'stypy_return_type' (line 880)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 8), 'stypy_return_type', shape_126624)
        
        # ################# End of 'shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shape' in the type store
        # Getting the type of 'stypy_return_type' (line 874)
        stypy_return_type_126625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shape'
        return stypy_return_type_126625

    
    # Assigning a Call to a Name (line 881):

    @norecursion
    def getValue(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getValue'
        module_type_store = module_type_store.open_function_context('getValue', 883, 4, False)
        # Assigning a type to the variable 'self' (line 884)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.getValue.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.getValue')
        netcdf_variable.getValue.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_variable.getValue.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.getValue.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.getValue', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getValue', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getValue(...)' code ##################

        str_126626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, (-1)), 'str', '\n        Retrieve a scalar value from a `netcdf_variable` of length one.\n\n        Raises\n        ------\n        ValueError\n            If the netcdf variable is an array of length greater than one,\n            this exception will be raised.\n\n        ')
        
        # Call to item(...): (line 894)
        # Processing the call keyword arguments (line 894)
        kwargs_126630 = {}
        # Getting the type of 'self' (line 894)
        self_126627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 15), 'self', False)
        # Obtaining the member 'data' of a type (line 894)
        data_126628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 15), self_126627, 'data')
        # Obtaining the member 'item' of a type (line 894)
        item_126629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 15), data_126628, 'item')
        # Calling item(args, kwargs) (line 894)
        item_call_result_126631 = invoke(stypy.reporting.localization.Localization(__file__, 894, 15), item_126629, *[], **kwargs_126630)
        
        # Assigning a type to the variable 'stypy_return_type' (line 894)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'stypy_return_type', item_call_result_126631)
        
        # ################# End of 'getValue(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getValue' in the type store
        # Getting the type of 'stypy_return_type' (line 883)
        stypy_return_type_126632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getValue'
        return stypy_return_type_126632


    @norecursion
    def assignValue(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assignValue'
        module_type_store = module_type_store.open_function_context('assignValue', 896, 4, False)
        # Assigning a type to the variable 'self' (line 897)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.assignValue')
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_param_names_list', ['value'])
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.assignValue.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.assignValue', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assignValue', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assignValue(...)' code ##################

        str_126633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, (-1)), 'str', '\n        Assign a scalar value to a `netcdf_variable` of length one.\n\n        Parameters\n        ----------\n        value : scalar\n            Scalar value (of compatible type) to assign to a length-one netcdf\n            variable. This value will be written to file.\n\n        Raises\n        ------\n        ValueError\n            If the input is not a scalar, or if the destination is not a length-one\n            netcdf variable.\n\n        ')
        
        
        # Getting the type of 'self' (line 913)
        self_126634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 15), 'self')
        # Obtaining the member 'data' of a type (line 913)
        data_126635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 15), self_126634, 'data')
        # Obtaining the member 'flags' of a type (line 913)
        flags_126636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 15), data_126635, 'flags')
        # Obtaining the member 'writeable' of a type (line 913)
        writeable_126637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 15), flags_126636, 'writeable')
        # Applying the 'not' unary operator (line 913)
        result_not__126638 = python_operator(stypy.reporting.localization.Localization(__file__, 913, 11), 'not', writeable_126637)
        
        # Testing the type of an if condition (line 913)
        if_condition_126639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 913, 8), result_not__126638)
        # Assigning a type to the variable 'if_condition_126639' (line 913)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 8), 'if_condition_126639', if_condition_126639)
        # SSA begins for if statement (line 913)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 919)
        # Processing the call arguments (line 919)
        str_126641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 31), 'str', 'variable is not writeable')
        # Processing the call keyword arguments (line 919)
        kwargs_126642 = {}
        # Getting the type of 'RuntimeError' (line 919)
        RuntimeError_126640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 919)
        RuntimeError_call_result_126643 = invoke(stypy.reporting.localization.Localization(__file__, 919, 18), RuntimeError_126640, *[str_126641], **kwargs_126642)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 919, 12), RuntimeError_call_result_126643, 'raise parameter', BaseException)
        # SSA join for if statement (line 913)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to itemset(...): (line 921)
        # Processing the call arguments (line 921)
        # Getting the type of 'value' (line 921)
        value_126647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 26), 'value', False)
        # Processing the call keyword arguments (line 921)
        kwargs_126648 = {}
        # Getting the type of 'self' (line 921)
        self_126644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 8), 'self', False)
        # Obtaining the member 'data' of a type (line 921)
        data_126645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 8), self_126644, 'data')
        # Obtaining the member 'itemset' of a type (line 921)
        itemset_126646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 8), data_126645, 'itemset')
        # Calling itemset(args, kwargs) (line 921)
        itemset_call_result_126649 = invoke(stypy.reporting.localization.Localization(__file__, 921, 8), itemset_126646, *[value_126647], **kwargs_126648)
        
        
        # ################# End of 'assignValue(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assignValue' in the type store
        # Getting the type of 'stypy_return_type' (line 896)
        stypy_return_type_126650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assignValue'
        return stypy_return_type_126650


    @norecursion
    def typecode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'typecode'
        module_type_store = module_type_store.open_function_context('typecode', 923, 4, False)
        # Assigning a type to the variable 'self' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.typecode.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.typecode')
        netcdf_variable.typecode.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_variable.typecode.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.typecode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.typecode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'typecode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'typecode(...)' code ##################

        str_126651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, (-1)), 'str', "\n        Return the typecode of the variable.\n\n        Returns\n        -------\n        typecode : char\n            The character typecode of the variable (eg, 'i' for int).\n\n        ")
        # Getting the type of 'self' (line 933)
        self_126652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 15), 'self')
        # Obtaining the member '_typecode' of a type (line 933)
        _typecode_126653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 15), self_126652, '_typecode')
        # Assigning a type to the variable 'stypy_return_type' (line 933)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 8), 'stypy_return_type', _typecode_126653)
        
        # ################# End of 'typecode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'typecode' in the type store
        # Getting the type of 'stypy_return_type' (line 923)
        stypy_return_type_126654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126654)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'typecode'
        return stypy_return_type_126654


    @norecursion
    def itemsize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'itemsize'
        module_type_store = module_type_store.open_function_context('itemsize', 935, 4, False)
        # Assigning a type to the variable 'self' (line 936)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.itemsize')
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.itemsize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.itemsize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'itemsize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'itemsize(...)' code ##################

        str_126655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, (-1)), 'str', '\n        Return the itemsize of the variable.\n\n        Returns\n        -------\n        itemsize : int\n            The element size of the variable (eg, 8 for float64).\n\n        ')
        # Getting the type of 'self' (line 945)
        self_126656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 15), 'self')
        # Obtaining the member '_size' of a type (line 945)
        _size_126657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 945, 15), self_126656, '_size')
        # Assigning a type to the variable 'stypy_return_type' (line 945)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 8), 'stypy_return_type', _size_126657)
        
        # ################# End of 'itemsize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'itemsize' in the type store
        # Getting the type of 'stypy_return_type' (line 935)
        stypy_return_type_126658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126658)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'itemsize'
        return stypy_return_type_126658


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 947, 4, False)
        # Assigning a type to the variable 'self' (line 948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.__getitem__')
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['index'])
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.__getitem__', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        
        # Getting the type of 'self' (line 948)
        self_126659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 15), 'self')
        # Obtaining the member 'maskandscale' of a type (line 948)
        maskandscale_126660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 15), self_126659, 'maskandscale')
        # Applying the 'not' unary operator (line 948)
        result_not__126661 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 11), 'not', maskandscale_126660)
        
        # Testing the type of an if condition (line 948)
        if_condition_126662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 948, 8), result_not__126661)
        # Assigning a type to the variable 'if_condition_126662' (line 948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 8), 'if_condition_126662', if_condition_126662)
        # SSA begins for if statement (line 948)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 949)
        index_126663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 29), 'index')
        # Getting the type of 'self' (line 949)
        self_126664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 19), 'self')
        # Obtaining the member 'data' of a type (line 949)
        data_126665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 949, 19), self_126664, 'data')
        # Obtaining the member '__getitem__' of a type (line 949)
        getitem___126666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 949, 19), data_126665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 949)
        subscript_call_result_126667 = invoke(stypy.reporting.localization.Localization(__file__, 949, 19), getitem___126666, index_126663)
        
        # Assigning a type to the variable 'stypy_return_type' (line 949)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 12), 'stypy_return_type', subscript_call_result_126667)
        # SSA join for if statement (line 948)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 951):
        
        # Assigning a Call to a Name (line 951):
        
        # Call to copy(...): (line 951)
        # Processing the call keyword arguments (line 951)
        kwargs_126674 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 951)
        index_126668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 25), 'index', False)
        # Getting the type of 'self' (line 951)
        self_126669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 15), 'self', False)
        # Obtaining the member 'data' of a type (line 951)
        data_126670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 15), self_126669, 'data')
        # Obtaining the member '__getitem__' of a type (line 951)
        getitem___126671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 15), data_126670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 951)
        subscript_call_result_126672 = invoke(stypy.reporting.localization.Localization(__file__, 951, 15), getitem___126671, index_126668)
        
        # Obtaining the member 'copy' of a type (line 951)
        copy_126673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 15), subscript_call_result_126672, 'copy')
        # Calling copy(args, kwargs) (line 951)
        copy_call_result_126675 = invoke(stypy.reporting.localization.Localization(__file__, 951, 15), copy_126673, *[], **kwargs_126674)
        
        # Assigning a type to the variable 'data' (line 951)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 8), 'data', copy_call_result_126675)
        
        # Assigning a Call to a Name (line 952):
        
        # Assigning a Call to a Name (line 952):
        
        # Call to _get_missing_value(...): (line 952)
        # Processing the call keyword arguments (line 952)
        kwargs_126678 = {}
        # Getting the type of 'self' (line 952)
        self_126676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 24), 'self', False)
        # Obtaining the member '_get_missing_value' of a type (line 952)
        _get_missing_value_126677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 24), self_126676, '_get_missing_value')
        # Calling _get_missing_value(args, kwargs) (line 952)
        _get_missing_value_call_result_126679 = invoke(stypy.reporting.localization.Localization(__file__, 952, 24), _get_missing_value_126677, *[], **kwargs_126678)
        
        # Assigning a type to the variable 'missing_value' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'missing_value', _get_missing_value_call_result_126679)
        
        # Assigning a Call to a Name (line 953):
        
        # Assigning a Call to a Name (line 953):
        
        # Call to _apply_missing_value(...): (line 953)
        # Processing the call arguments (line 953)
        # Getting the type of 'data' (line 953)
        data_126682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 41), 'data', False)
        # Getting the type of 'missing_value' (line 953)
        missing_value_126683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 47), 'missing_value', False)
        # Processing the call keyword arguments (line 953)
        kwargs_126684 = {}
        # Getting the type of 'self' (line 953)
        self_126680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 15), 'self', False)
        # Obtaining the member '_apply_missing_value' of a type (line 953)
        _apply_missing_value_126681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 953, 15), self_126680, '_apply_missing_value')
        # Calling _apply_missing_value(args, kwargs) (line 953)
        _apply_missing_value_call_result_126685 = invoke(stypy.reporting.localization.Localization(__file__, 953, 15), _apply_missing_value_126681, *[data_126682, missing_value_126683], **kwargs_126684)
        
        # Assigning a type to the variable 'data' (line 953)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 8), 'data', _apply_missing_value_call_result_126685)
        
        # Assigning a Call to a Name (line 954):
        
        # Assigning a Call to a Name (line 954):
        
        # Call to get(...): (line 954)
        # Processing the call arguments (line 954)
        str_126689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 44), 'str', 'scale_factor')
        # Processing the call keyword arguments (line 954)
        kwargs_126690 = {}
        # Getting the type of 'self' (line 954)
        self_126686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 23), 'self', False)
        # Obtaining the member '_attributes' of a type (line 954)
        _attributes_126687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 23), self_126686, '_attributes')
        # Obtaining the member 'get' of a type (line 954)
        get_126688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 23), _attributes_126687, 'get')
        # Calling get(args, kwargs) (line 954)
        get_call_result_126691 = invoke(stypy.reporting.localization.Localization(__file__, 954, 23), get_126688, *[str_126689], **kwargs_126690)
        
        # Assigning a type to the variable 'scale_factor' (line 954)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 8), 'scale_factor', get_call_result_126691)
        
        # Assigning a Call to a Name (line 955):
        
        # Assigning a Call to a Name (line 955):
        
        # Call to get(...): (line 955)
        # Processing the call arguments (line 955)
        str_126695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 42), 'str', 'add_offset')
        # Processing the call keyword arguments (line 955)
        kwargs_126696 = {}
        # Getting the type of 'self' (line 955)
        self_126692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 21), 'self', False)
        # Obtaining the member '_attributes' of a type (line 955)
        _attributes_126693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 21), self_126692, '_attributes')
        # Obtaining the member 'get' of a type (line 955)
        get_126694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 21), _attributes_126693, 'get')
        # Calling get(args, kwargs) (line 955)
        get_call_result_126697 = invoke(stypy.reporting.localization.Localization(__file__, 955, 21), get_126694, *[str_126695], **kwargs_126696)
        
        # Assigning a type to the variable 'add_offset' (line 955)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 8), 'add_offset', get_call_result_126697)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'add_offset' (line 956)
        add_offset_126698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 11), 'add_offset')
        # Getting the type of 'None' (line 956)
        None_126699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 29), 'None')
        # Applying the binary operator 'isnot' (line 956)
        result_is_not_126700 = python_operator(stypy.reporting.localization.Localization(__file__, 956, 11), 'isnot', add_offset_126698, None_126699)
        
        
        # Getting the type of 'scale_factor' (line 956)
        scale_factor_126701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 37), 'scale_factor')
        # Getting the type of 'None' (line 956)
        None_126702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 57), 'None')
        # Applying the binary operator 'isnot' (line 956)
        result_is_not_126703 = python_operator(stypy.reporting.localization.Localization(__file__, 956, 37), 'isnot', scale_factor_126701, None_126702)
        
        # Applying the binary operator 'or' (line 956)
        result_or_keyword_126704 = python_operator(stypy.reporting.localization.Localization(__file__, 956, 11), 'or', result_is_not_126700, result_is_not_126703)
        
        # Testing the type of an if condition (line 956)
        if_condition_126705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 956, 8), result_or_keyword_126704)
        # Assigning a type to the variable 'if_condition_126705' (line 956)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 8), 'if_condition_126705', if_condition_126705)
        # SSA begins for if statement (line 956)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 957):
        
        # Assigning a Call to a Name (line 957):
        
        # Call to astype(...): (line 957)
        # Processing the call arguments (line 957)
        # Getting the type of 'np' (line 957)
        np_126708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 31), 'np', False)
        # Obtaining the member 'float64' of a type (line 957)
        float64_126709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 957, 31), np_126708, 'float64')
        # Processing the call keyword arguments (line 957)
        kwargs_126710 = {}
        # Getting the type of 'data' (line 957)
        data_126706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 19), 'data', False)
        # Obtaining the member 'astype' of a type (line 957)
        astype_126707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 957, 19), data_126706, 'astype')
        # Calling astype(args, kwargs) (line 957)
        astype_call_result_126711 = invoke(stypy.reporting.localization.Localization(__file__, 957, 19), astype_126707, *[float64_126709], **kwargs_126710)
        
        # Assigning a type to the variable 'data' (line 957)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 12), 'data', astype_call_result_126711)
        # SSA join for if statement (line 956)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 958)
        # Getting the type of 'scale_factor' (line 958)
        scale_factor_126712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 8), 'scale_factor')
        # Getting the type of 'None' (line 958)
        None_126713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 31), 'None')
        
        (may_be_126714, more_types_in_union_126715) = may_not_be_none(scale_factor_126712, None_126713)

        if may_be_126714:

            if more_types_in_union_126715:
                # Runtime conditional SSA (line 958)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 959):
            
            # Assigning a BinOp to a Name (line 959):
            # Getting the type of 'data' (line 959)
            data_126716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 19), 'data')
            # Getting the type of 'scale_factor' (line 959)
            scale_factor_126717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 26), 'scale_factor')
            # Applying the binary operator '*' (line 959)
            result_mul_126718 = python_operator(stypy.reporting.localization.Localization(__file__, 959, 19), '*', data_126716, scale_factor_126717)
            
            # Assigning a type to the variable 'data' (line 959)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 12), 'data', result_mul_126718)

            if more_types_in_union_126715:
                # SSA join for if statement (line 958)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 960)
        # Getting the type of 'add_offset' (line 960)
        add_offset_126719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 8), 'add_offset')
        # Getting the type of 'None' (line 960)
        None_126720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 29), 'None')
        
        (may_be_126721, more_types_in_union_126722) = may_not_be_none(add_offset_126719, None_126720)

        if may_be_126721:

            if more_types_in_union_126722:
                # Runtime conditional SSA (line 960)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'data' (line 961)
            data_126723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 12), 'data')
            # Getting the type of 'add_offset' (line 961)
            add_offset_126724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 20), 'add_offset')
            # Applying the binary operator '+=' (line 961)
            result_iadd_126725 = python_operator(stypy.reporting.localization.Localization(__file__, 961, 12), '+=', data_126723, add_offset_126724)
            # Assigning a type to the variable 'data' (line 961)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 12), 'data', result_iadd_126725)
            

            if more_types_in_union_126722:
                # SSA join for if statement (line 960)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'data' (line 963)
        data_126726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 15), 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 963)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 8), 'stypy_return_type', data_126726)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 947)
        stypy_return_type_126727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_126727


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 965, 4, False)
        # Assigning a type to the variable 'self' (line 966)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_function_name', 'netcdf_variable.__setitem__')
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['index', 'data'])
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable.__setitem__', ['index', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['index', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        
        # Getting the type of 'self' (line 966)
        self_126728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 11), 'self')
        # Obtaining the member 'maskandscale' of a type (line 966)
        maskandscale_126729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 11), self_126728, 'maskandscale')
        # Testing the type of an if condition (line 966)
        if_condition_126730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 966, 8), maskandscale_126729)
        # Assigning a type to the variable 'if_condition_126730' (line 966)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'if_condition_126730', if_condition_126730)
        # SSA begins for if statement (line 966)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Name (line 967):
        
        # Assigning a BoolOp to a Name (line 967):
        
        # Evaluating a boolean operation
        
        # Call to _get_missing_value(...): (line 968)
        # Processing the call keyword arguments (line 968)
        kwargs_126733 = {}
        # Getting the type of 'self' (line 968)
        self_126731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 20), 'self', False)
        # Obtaining the member '_get_missing_value' of a type (line 968)
        _get_missing_value_126732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 968, 20), self_126731, '_get_missing_value')
        # Calling _get_missing_value(args, kwargs) (line 968)
        _get_missing_value_call_result_126734 = invoke(stypy.reporting.localization.Localization(__file__, 968, 20), _get_missing_value_126732, *[], **kwargs_126733)
        
        
        # Call to getattr(...): (line 969)
        # Processing the call arguments (line 969)
        # Getting the type of 'data' (line 969)
        data_126736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 28), 'data', False)
        str_126737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 34), 'str', 'fill_value')
        int_126738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 48), 'int')
        # Processing the call keyword arguments (line 969)
        kwargs_126739 = {}
        # Getting the type of 'getattr' (line 969)
        getattr_126735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 969)
        getattr_call_result_126740 = invoke(stypy.reporting.localization.Localization(__file__, 969, 20), getattr_126735, *[data_126736, str_126737, int_126738], **kwargs_126739)
        
        # Applying the binary operator 'or' (line 968)
        result_or_keyword_126741 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 20), 'or', _get_missing_value_call_result_126734, getattr_call_result_126740)
        
        # Assigning a type to the variable 'missing_value' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 12), 'missing_value', result_or_keyword_126741)
        
        # Call to setdefault(...): (line 970)
        # Processing the call arguments (line 970)
        str_126745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 40), 'str', 'missing_value')
        # Getting the type of 'missing_value' (line 970)
        missing_value_126746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 57), 'missing_value', False)
        # Processing the call keyword arguments (line 970)
        kwargs_126747 = {}
        # Getting the type of 'self' (line 970)
        self_126742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 12), 'self', False)
        # Obtaining the member '_attributes' of a type (line 970)
        _attributes_126743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 12), self_126742, '_attributes')
        # Obtaining the member 'setdefault' of a type (line 970)
        setdefault_126744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 12), _attributes_126743, 'setdefault')
        # Calling setdefault(args, kwargs) (line 970)
        setdefault_call_result_126748 = invoke(stypy.reporting.localization.Localization(__file__, 970, 12), setdefault_126744, *[str_126745, missing_value_126746], **kwargs_126747)
        
        
        # Call to setdefault(...): (line 971)
        # Processing the call arguments (line 971)
        str_126752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 40), 'str', '_FillValue')
        # Getting the type of 'missing_value' (line 971)
        missing_value_126753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 54), 'missing_value', False)
        # Processing the call keyword arguments (line 971)
        kwargs_126754 = {}
        # Getting the type of 'self' (line 971)
        self_126749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 12), 'self', False)
        # Obtaining the member '_attributes' of a type (line 971)
        _attributes_126750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 12), self_126749, '_attributes')
        # Obtaining the member 'setdefault' of a type (line 971)
        setdefault_126751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 12), _attributes_126750, 'setdefault')
        # Calling setdefault(args, kwargs) (line 971)
        setdefault_call_result_126755 = invoke(stypy.reporting.localization.Localization(__file__, 971, 12), setdefault_126751, *[str_126752, missing_value_126753], **kwargs_126754)
        
        
        # Assigning a BinOp to a Name (line 972):
        
        # Assigning a BinOp to a Name (line 972):
        # Getting the type of 'data' (line 972)
        data_126756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 21), 'data')
        
        # Call to get(...): (line 972)
        # Processing the call arguments (line 972)
        str_126760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 49), 'str', 'add_offset')
        float_126761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 63), 'float')
        # Processing the call keyword arguments (line 972)
        kwargs_126762 = {}
        # Getting the type of 'self' (line 972)
        self_126757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 28), 'self', False)
        # Obtaining the member '_attributes' of a type (line 972)
        _attributes_126758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 28), self_126757, '_attributes')
        # Obtaining the member 'get' of a type (line 972)
        get_126759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 28), _attributes_126758, 'get')
        # Calling get(args, kwargs) (line 972)
        get_call_result_126763 = invoke(stypy.reporting.localization.Localization(__file__, 972, 28), get_126759, *[str_126760, float_126761], **kwargs_126762)
        
        # Applying the binary operator '-' (line 972)
        result_sub_126764 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 21), '-', data_126756, get_call_result_126763)
        
        
        # Call to get(...): (line 973)
        # Processing the call arguments (line 973)
        str_126768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 41), 'str', 'scale_factor')
        float_126769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 57), 'float')
        # Processing the call keyword arguments (line 973)
        kwargs_126770 = {}
        # Getting the type of 'self' (line 973)
        self_126765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 20), 'self', False)
        # Obtaining the member '_attributes' of a type (line 973)
        _attributes_126766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 20), self_126765, '_attributes')
        # Obtaining the member 'get' of a type (line 973)
        get_126767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 20), _attributes_126766, 'get')
        # Calling get(args, kwargs) (line 973)
        get_call_result_126771 = invoke(stypy.reporting.localization.Localization(__file__, 973, 20), get_126767, *[str_126768, float_126769], **kwargs_126770)
        
        # Applying the binary operator 'div' (line 972)
        result_div_126772 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 20), 'div', result_sub_126764, get_call_result_126771)
        
        # Assigning a type to the variable 'data' (line 972)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), 'data', result_div_126772)
        
        # Assigning a Call to a Name (line 974):
        
        # Assigning a Call to a Name (line 974):
        
        # Call to filled(...): (line 974)
        # Processing the call arguments (line 974)
        # Getting the type of 'missing_value' (line 974)
        missing_value_126780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 46), 'missing_value', False)
        # Processing the call keyword arguments (line 974)
        kwargs_126781 = {}
        
        # Call to asarray(...): (line 974)
        # Processing the call arguments (line 974)
        # Getting the type of 'data' (line 974)
        data_126776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 33), 'data', False)
        # Processing the call keyword arguments (line 974)
        kwargs_126777 = {}
        # Getting the type of 'np' (line 974)
        np_126773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 19), 'np', False)
        # Obtaining the member 'ma' of a type (line 974)
        ma_126774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 19), np_126773, 'ma')
        # Obtaining the member 'asarray' of a type (line 974)
        asarray_126775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 19), ma_126774, 'asarray')
        # Calling asarray(args, kwargs) (line 974)
        asarray_call_result_126778 = invoke(stypy.reporting.localization.Localization(__file__, 974, 19), asarray_126775, *[data_126776], **kwargs_126777)
        
        # Obtaining the member 'filled' of a type (line 974)
        filled_126779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 19), asarray_call_result_126778, 'filled')
        # Calling filled(args, kwargs) (line 974)
        filled_call_result_126782 = invoke(stypy.reporting.localization.Localization(__file__, 974, 19), filled_126779, *[missing_value_126780], **kwargs_126781)
        
        # Assigning a type to the variable 'data' (line 974)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 12), 'data', filled_call_result_126782)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 975)
        self_126783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 15), 'self')
        # Obtaining the member '_typecode' of a type (line 975)
        _typecode_126784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 15), self_126783, '_typecode')
        str_126785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 37), 'str', 'fd')
        # Applying the binary operator 'notin' (line 975)
        result_contains_126786 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 15), 'notin', _typecode_126784, str_126785)
        
        
        # Getting the type of 'data' (line 975)
        data_126787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 46), 'data')
        # Obtaining the member 'dtype' of a type (line 975)
        dtype_126788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 46), data_126787, 'dtype')
        # Obtaining the member 'kind' of a type (line 975)
        kind_126789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 46), dtype_126788, 'kind')
        str_126790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 65), 'str', 'f')
        # Applying the binary operator '==' (line 975)
        result_eq_126791 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 46), '==', kind_126789, str_126790)
        
        # Applying the binary operator 'and' (line 975)
        result_and_keyword_126792 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 15), 'and', result_contains_126786, result_eq_126791)
        
        # Testing the type of an if condition (line 975)
        if_condition_126793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 975, 12), result_and_keyword_126792)
        # Assigning a type to the variable 'if_condition_126793' (line 975)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 12), 'if_condition_126793', if_condition_126793)
        # SSA begins for if statement (line 975)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 976):
        
        # Assigning a Call to a Name (line 976):
        
        # Call to round(...): (line 976)
        # Processing the call arguments (line 976)
        # Getting the type of 'data' (line 976)
        data_126796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 32), 'data', False)
        # Processing the call keyword arguments (line 976)
        kwargs_126797 = {}
        # Getting the type of 'np' (line 976)
        np_126794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 23), 'np', False)
        # Obtaining the member 'round' of a type (line 976)
        round_126795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 23), np_126794, 'round')
        # Calling round(args, kwargs) (line 976)
        round_call_result_126798 = invoke(stypy.reporting.localization.Localization(__file__, 976, 23), round_126795, *[data_126796], **kwargs_126797)
        
        # Assigning a type to the variable 'data' (line 976)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 16), 'data', round_call_result_126798)
        # SSA join for if statement (line 975)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 966)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 979)
        self_126799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 11), 'self')
        # Obtaining the member 'isrec' of a type (line 979)
        isrec_126800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 11), self_126799, 'isrec')
        # Testing the type of an if condition (line 979)
        if_condition_126801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 979, 8), isrec_126800)
        # Assigning a type to the variable 'if_condition_126801' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'if_condition_126801', if_condition_126801)
        # SSA begins for if statement (line 979)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 980)
        # Getting the type of 'tuple' (line 980)
        tuple_126802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 33), 'tuple')
        # Getting the type of 'index' (line 980)
        index_126803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 26), 'index')
        
        (may_be_126804, more_types_in_union_126805) = may_be_subtype(tuple_126802, index_126803)

        if may_be_126804:

            if more_types_in_union_126805:
                # Runtime conditional SSA (line 980)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'index' (line 980)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 12), 'index', remove_not_subtype_from_union(index_126803, tuple))
            
            # Assigning a Subscript to a Name (line 981):
            
            # Assigning a Subscript to a Name (line 981):
            
            # Obtaining the type of the subscript
            int_126806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 34), 'int')
            # Getting the type of 'index' (line 981)
            index_126807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 28), 'index')
            # Obtaining the member '__getitem__' of a type (line 981)
            getitem___126808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 28), index_126807, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 981)
            subscript_call_result_126809 = invoke(stypy.reporting.localization.Localization(__file__, 981, 28), getitem___126808, int_126806)
            
            # Assigning a type to the variable 'rec_index' (line 981)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 16), 'rec_index', subscript_call_result_126809)

            if more_types_in_union_126805:
                # Runtime conditional SSA for else branch (line 980)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_126804) or more_types_in_union_126805):
            # Assigning a type to the variable 'index' (line 980)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 12), 'index', remove_subtype_from_union(index_126803, tuple))
            
            # Assigning a Name to a Name (line 983):
            
            # Assigning a Name to a Name (line 983):
            # Getting the type of 'index' (line 983)
            index_126810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 28), 'index')
            # Assigning a type to the variable 'rec_index' (line 983)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 16), 'rec_index', index_126810)

            if (may_be_126804 and more_types_in_union_126805):
                # SSA join for if statement (line 980)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 984)
        # Getting the type of 'slice' (line 984)
        slice_126811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 37), 'slice')
        # Getting the type of 'rec_index' (line 984)
        rec_index_126812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 26), 'rec_index')
        
        (may_be_126813, more_types_in_union_126814) = may_be_subtype(slice_126811, rec_index_126812)

        if may_be_126813:

            if more_types_in_union_126814:
                # Runtime conditional SSA (line 984)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'rec_index' (line 984)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 12), 'rec_index', remove_not_subtype_from_union(rec_index_126812, slice))
            
            # Assigning a BinOp to a Name (line 985):
            
            # Assigning a BinOp to a Name (line 985):
            
            # Evaluating a boolean operation
            # Getting the type of 'rec_index' (line 985)
            rec_index_126815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 24), 'rec_index')
            # Obtaining the member 'start' of a type (line 985)
            start_126816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 985, 24), rec_index_126815, 'start')
            int_126817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 43), 'int')
            # Applying the binary operator 'or' (line 985)
            result_or_keyword_126818 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 24), 'or', start_126816, int_126817)
            
            
            # Call to len(...): (line 985)
            # Processing the call arguments (line 985)
            # Getting the type of 'data' (line 985)
            data_126820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 52), 'data', False)
            # Processing the call keyword arguments (line 985)
            kwargs_126821 = {}
            # Getting the type of 'len' (line 985)
            len_126819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 48), 'len', False)
            # Calling len(args, kwargs) (line 985)
            len_call_result_126822 = invoke(stypy.reporting.localization.Localization(__file__, 985, 48), len_126819, *[data_126820], **kwargs_126821)
            
            # Applying the binary operator '+' (line 985)
            result_add_126823 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 23), '+', result_or_keyword_126818, len_call_result_126822)
            
            # Assigning a type to the variable 'recs' (line 985)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 16), 'recs', result_add_126823)

            if more_types_in_union_126814:
                # Runtime conditional SSA for else branch (line 984)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_126813) or more_types_in_union_126814):
            # Assigning a type to the variable 'rec_index' (line 984)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 12), 'rec_index', remove_subtype_from_union(rec_index_126812, slice))
            
            # Assigning a BinOp to a Name (line 987):
            
            # Assigning a BinOp to a Name (line 987):
            # Getting the type of 'rec_index' (line 987)
            rec_index_126824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 23), 'rec_index')
            int_126825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 35), 'int')
            # Applying the binary operator '+' (line 987)
            result_add_126826 = python_operator(stypy.reporting.localization.Localization(__file__, 987, 23), '+', rec_index_126824, int_126825)
            
            # Assigning a type to the variable 'recs' (line 987)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 987, 16), 'recs', result_add_126826)

            if (may_be_126813 and more_types_in_union_126814):
                # SSA join for if statement (line 984)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'recs' (line 988)
        recs_126827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 15), 'recs')
        
        # Call to len(...): (line 988)
        # Processing the call arguments (line 988)
        # Getting the type of 'self' (line 988)
        self_126829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 26), 'self', False)
        # Obtaining the member 'data' of a type (line 988)
        data_126830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 26), self_126829, 'data')
        # Processing the call keyword arguments (line 988)
        kwargs_126831 = {}
        # Getting the type of 'len' (line 988)
        len_126828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 22), 'len', False)
        # Calling len(args, kwargs) (line 988)
        len_call_result_126832 = invoke(stypy.reporting.localization.Localization(__file__, 988, 22), len_126828, *[data_126830], **kwargs_126831)
        
        # Applying the binary operator '>' (line 988)
        result_gt_126833 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 15), '>', recs_126827, len_call_result_126832)
        
        # Testing the type of an if condition (line 988)
        if_condition_126834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 988, 12), result_gt_126833)
        # Assigning a type to the variable 'if_condition_126834' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 12), 'if_condition_126834', if_condition_126834)
        # SSA begins for if statement (line 988)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 989):
        
        # Assigning a BinOp to a Name (line 989):
        
        # Obtaining an instance of the builtin type 'tuple' (line 989)
        tuple_126835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 989)
        # Adding element type (line 989)
        # Getting the type of 'recs' (line 989)
        recs_126836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 25), 'recs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 989, 25), tuple_126835, recs_126836)
        
        
        # Obtaining the type of the subscript
        int_126837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 46), 'int')
        slice_126838 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 989, 34), int_126837, None, None)
        # Getting the type of 'self' (line 989)
        self_126839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 34), 'self')
        # Obtaining the member '_shape' of a type (line 989)
        _shape_126840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 34), self_126839, '_shape')
        # Obtaining the member '__getitem__' of a type (line 989)
        getitem___126841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 34), _shape_126840, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 989)
        subscript_call_result_126842 = invoke(stypy.reporting.localization.Localization(__file__, 989, 34), getitem___126841, slice_126838)
        
        # Applying the binary operator '+' (line 989)
        result_add_126843 = python_operator(stypy.reporting.localization.Localization(__file__, 989, 24), '+', tuple_126835, subscript_call_result_126842)
        
        # Assigning a type to the variable 'shape' (line 989)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 989, 16), 'shape', result_add_126843)
        
        
        # SSA begins for try-except statement (line 992)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to resize(...): (line 993)
        # Processing the call arguments (line 993)
        # Getting the type of 'shape' (line 993)
        shape_126847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 37), 'shape', False)
        # Processing the call keyword arguments (line 993)
        kwargs_126848 = {}
        # Getting the type of 'self' (line 993)
        self_126844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 20), 'self', False)
        # Obtaining the member 'data' of a type (line 993)
        data_126845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 20), self_126844, 'data')
        # Obtaining the member 'resize' of a type (line 993)
        resize_126846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 20), data_126845, 'resize')
        # Calling resize(args, kwargs) (line 993)
        resize_call_result_126849 = invoke(stypy.reporting.localization.Localization(__file__, 993, 20), resize_126846, *[shape_126847], **kwargs_126848)
        
        # SSA branch for the except part of a try statement (line 992)
        # SSA branch for the except 'ValueError' branch of a try statement (line 992)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Subscript (line 995):
        
        # Assigning a Call to a Subscript (line 995):
        
        # Call to astype(...): (line 995)
        # Processing the call arguments (line 995)
        # Getting the type of 'self' (line 995)
        self_126858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 79), 'self', False)
        # Obtaining the member 'data' of a type (line 995)
        data_126859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 79), self_126858, 'data')
        # Obtaining the member 'dtype' of a type (line 995)
        dtype_126860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 79), data_126859, 'dtype')
        # Processing the call keyword arguments (line 995)
        kwargs_126861 = {}
        
        # Call to resize(...): (line 995)
        # Processing the call arguments (line 995)
        # Getting the type of 'self' (line 995)
        self_126852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 54), 'self', False)
        # Obtaining the member 'data' of a type (line 995)
        data_126853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 54), self_126852, 'data')
        # Getting the type of 'shape' (line 995)
        shape_126854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 65), 'shape', False)
        # Processing the call keyword arguments (line 995)
        kwargs_126855 = {}
        # Getting the type of 'np' (line 995)
        np_126850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 44), 'np', False)
        # Obtaining the member 'resize' of a type (line 995)
        resize_126851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 44), np_126850, 'resize')
        # Calling resize(args, kwargs) (line 995)
        resize_call_result_126856 = invoke(stypy.reporting.localization.Localization(__file__, 995, 44), resize_126851, *[data_126853, shape_126854], **kwargs_126855)
        
        # Obtaining the member 'astype' of a type (line 995)
        astype_126857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 44), resize_call_result_126856, 'astype')
        # Calling astype(args, kwargs) (line 995)
        astype_call_result_126862 = invoke(stypy.reporting.localization.Localization(__file__, 995, 44), astype_126857, *[dtype_126860], **kwargs_126861)
        
        # Getting the type of 'self' (line 995)
        self_126863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 20), 'self')
        # Obtaining the member '__dict__' of a type (line 995)
        dict___126864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 20), self_126863, '__dict__')
        str_126865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 34), 'str', 'data')
        # Storing an element on a container (line 995)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 20), dict___126864, (str_126865, astype_call_result_126862))
        # SSA join for try-except statement (line 992)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 988)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 979)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 996):
        
        # Assigning a Name to a Subscript (line 996):
        # Getting the type of 'data' (line 996)
        data_126866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 27), 'data')
        # Getting the type of 'self' (line 996)
        self_126867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 8), 'self')
        # Obtaining the member 'data' of a type (line 996)
        data_126868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 996, 8), self_126867, 'data')
        # Getting the type of 'index' (line 996)
        index_126869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 18), 'index')
        # Storing an element on a container (line 996)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 8), data_126868, (index_126869, data_126866))
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 965)
        stypy_return_type_126870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126870)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_126870


    @norecursion
    def _get_missing_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_missing_value'
        module_type_store = module_type_store.open_function_context('_get_missing_value', 998, 4, False)
        # Assigning a type to the variable 'self' (line 999)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_function_name', 'netcdf_variable._get_missing_value')
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_param_names_list', [])
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable._get_missing_value.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'netcdf_variable._get_missing_value', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_missing_value', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_missing_value(...)' code ##################

        str_126871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, (-1)), 'str', '\n        Returns the value denoting "no data" for this variable.\n\n        If this variable does not have a missing/fill value, returns None.\n\n        If both _FillValue and missing_value are given, give precedence to\n        _FillValue. The netCDF standard gives special meaning to _FillValue;\n        missing_value is  just used for compatibility with old datasets.\n        ')
        
        
        str_126872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 11), 'str', '_FillValue')
        # Getting the type of 'self' (line 1009)
        self_126873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 27), 'self')
        # Obtaining the member '_attributes' of a type (line 1009)
        _attributes_126874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1009, 27), self_126873, '_attributes')
        # Applying the binary operator 'in' (line 1009)
        result_contains_126875 = python_operator(stypy.reporting.localization.Localization(__file__, 1009, 11), 'in', str_126872, _attributes_126874)
        
        # Testing the type of an if condition (line 1009)
        if_condition_126876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1009, 8), result_contains_126875)
        # Assigning a type to the variable 'if_condition_126876' (line 1009)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'if_condition_126876', if_condition_126876)
        # SSA begins for if statement (line 1009)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 1010):
        
        # Assigning a Subscript to a Name (line 1010):
        
        # Obtaining the type of the subscript
        str_126877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 45), 'str', '_FillValue')
        # Getting the type of 'self' (line 1010)
        self_126878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 28), 'self')
        # Obtaining the member '_attributes' of a type (line 1010)
        _attributes_126879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 28), self_126878, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 1010)
        getitem___126880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 28), _attributes_126879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1010)
        subscript_call_result_126881 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 28), getitem___126880, str_126877)
        
        # Assigning a type to the variable 'missing_value' (line 1010)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 12), 'missing_value', subscript_call_result_126881)
        # SSA branch for the else part of an if statement (line 1009)
        module_type_store.open_ssa_branch('else')
        
        
        str_126882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 13), 'str', 'missing_value')
        # Getting the type of 'self' (line 1011)
        self_126883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 32), 'self')
        # Obtaining the member '_attributes' of a type (line 1011)
        _attributes_126884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 32), self_126883, '_attributes')
        # Applying the binary operator 'in' (line 1011)
        result_contains_126885 = python_operator(stypy.reporting.localization.Localization(__file__, 1011, 13), 'in', str_126882, _attributes_126884)
        
        # Testing the type of an if condition (line 1011)
        if_condition_126886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1011, 13), result_contains_126885)
        # Assigning a type to the variable 'if_condition_126886' (line 1011)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 13), 'if_condition_126886', if_condition_126886)
        # SSA begins for if statement (line 1011)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 1012):
        
        # Assigning a Subscript to a Name (line 1012):
        
        # Obtaining the type of the subscript
        str_126887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 45), 'str', 'missing_value')
        # Getting the type of 'self' (line 1012)
        self_126888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 28), 'self')
        # Obtaining the member '_attributes' of a type (line 1012)
        _attributes_126889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 28), self_126888, '_attributes')
        # Obtaining the member '__getitem__' of a type (line 1012)
        getitem___126890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 28), _attributes_126889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1012)
        subscript_call_result_126891 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 28), getitem___126890, str_126887)
        
        # Assigning a type to the variable 'missing_value' (line 1012)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 12), 'missing_value', subscript_call_result_126891)
        # SSA branch for the else part of an if statement (line 1011)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1014):
        
        # Assigning a Name to a Name (line 1014):
        # Getting the type of 'None' (line 1014)
        None_126892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 28), 'None')
        # Assigning a type to the variable 'missing_value' (line 1014)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1014, 12), 'missing_value', None_126892)
        # SSA join for if statement (line 1011)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1009)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'missing_value' (line 1016)
        missing_value_126893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 15), 'missing_value')
        # Assigning a type to the variable 'stypy_return_type' (line 1016)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1016, 8), 'stypy_return_type', missing_value_126893)
        
        # ################# End of '_get_missing_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_missing_value' in the type store
        # Getting the type of 'stypy_return_type' (line 998)
        stypy_return_type_126894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126894)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_missing_value'
        return stypy_return_type_126894


    @staticmethod
    @norecursion
    def _apply_missing_value(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_apply_missing_value'
        module_type_store = module_type_store.open_function_context('_apply_missing_value', 1018, 4, False)
        
        # Passed parameters checking function
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_localization', localization)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_type_of_self', None)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_function_name', '_apply_missing_value')
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_param_names_list', ['data', 'missing_value'])
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        netcdf_variable._apply_missing_value.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '_apply_missing_value', ['data', 'missing_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_apply_missing_value', localization, ['missing_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_apply_missing_value(...)' code ##################

        str_126895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, (-1)), 'str', '\n        Applies the given missing value to the data array.\n\n        Returns a numpy.ma array, with any value equal to missing_value masked\n        out (unless missing_value is None, in which case the original array is\n        returned).\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 1028)
        # Getting the type of 'missing_value' (line 1028)
        missing_value_126896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 11), 'missing_value')
        # Getting the type of 'None' (line 1028)
        None_126897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 28), 'None')
        
        (may_be_126898, more_types_in_union_126899) = may_be_none(missing_value_126896, None_126897)

        if may_be_126898:

            if more_types_in_union_126899:
                # Runtime conditional SSA (line 1028)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 1029):
            
            # Assigning a Name to a Name (line 1029):
            # Getting the type of 'data' (line 1029)
            data_126900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 22), 'data')
            # Assigning a type to the variable 'newdata' (line 1029)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1029, 12), 'newdata', data_126900)

            if more_types_in_union_126899:
                # Runtime conditional SSA for else branch (line 1028)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_126898) or more_types_in_union_126899):
            
            
            # SSA begins for try-except statement (line 1031)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 1032):
            
            # Assigning a Call to a Name (line 1032):
            
            # Call to isnan(...): (line 1032)
            # Processing the call arguments (line 1032)
            # Getting the type of 'missing_value' (line 1032)
            missing_value_126903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 47), 'missing_value', False)
            # Processing the call keyword arguments (line 1032)
            kwargs_126904 = {}
            # Getting the type of 'np' (line 1032)
            np_126901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 38), 'np', False)
            # Obtaining the member 'isnan' of a type (line 1032)
            isnan_126902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 38), np_126901, 'isnan')
            # Calling isnan(args, kwargs) (line 1032)
            isnan_call_result_126905 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 38), isnan_126902, *[missing_value_126903], **kwargs_126904)
            
            # Assigning a type to the variable 'missing_value_isnan' (line 1032)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 16), 'missing_value_isnan', isnan_call_result_126905)
            # SSA branch for the except part of a try statement (line 1031)
            # SSA branch for the except 'Tuple' branch of a try statement (line 1031)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Name (line 1035):
            
            # Assigning a Name to a Name (line 1035):
            # Getting the type of 'False' (line 1035)
            False_126906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 38), 'False')
            # Assigning a type to the variable 'missing_value_isnan' (line 1035)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 16), 'missing_value_isnan', False_126906)
            # SSA join for try-except statement (line 1031)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'missing_value_isnan' (line 1037)
            missing_value_isnan_126907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 15), 'missing_value_isnan')
            # Testing the type of an if condition (line 1037)
            if_condition_126908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1037, 12), missing_value_isnan_126907)
            # Assigning a type to the variable 'if_condition_126908' (line 1037)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 12), 'if_condition_126908', if_condition_126908)
            # SSA begins for if statement (line 1037)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 1038):
            
            # Assigning a Call to a Name (line 1038):
            
            # Call to isnan(...): (line 1038)
            # Processing the call arguments (line 1038)
            # Getting the type of 'data' (line 1038)
            data_126911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 34), 'data', False)
            # Processing the call keyword arguments (line 1038)
            kwargs_126912 = {}
            # Getting the type of 'np' (line 1038)
            np_126909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 25), 'np', False)
            # Obtaining the member 'isnan' of a type (line 1038)
            isnan_126910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 25), np_126909, 'isnan')
            # Calling isnan(args, kwargs) (line 1038)
            isnan_call_result_126913 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 25), isnan_126910, *[data_126911], **kwargs_126912)
            
            # Assigning a type to the variable 'mymask' (line 1038)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 16), 'mymask', isnan_call_result_126913)
            # SSA branch for the else part of an if statement (line 1037)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Compare to a Name (line 1040):
            
            # Assigning a Compare to a Name (line 1040):
            
            # Getting the type of 'data' (line 1040)
            data_126914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 26), 'data')
            # Getting the type of 'missing_value' (line 1040)
            missing_value_126915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 34), 'missing_value')
            # Applying the binary operator '==' (line 1040)
            result_eq_126916 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 26), '==', data_126914, missing_value_126915)
            
            # Assigning a type to the variable 'mymask' (line 1040)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 16), 'mymask', result_eq_126916)
            # SSA join for if statement (line 1037)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 1042):
            
            # Assigning a Call to a Name (line 1042):
            
            # Call to masked_where(...): (line 1042)
            # Processing the call arguments (line 1042)
            # Getting the type of 'mymask' (line 1042)
            mymask_126920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 41), 'mymask', False)
            # Getting the type of 'data' (line 1042)
            data_126921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 49), 'data', False)
            # Processing the call keyword arguments (line 1042)
            kwargs_126922 = {}
            # Getting the type of 'np' (line 1042)
            np_126917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 22), 'np', False)
            # Obtaining the member 'ma' of a type (line 1042)
            ma_126918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 22), np_126917, 'ma')
            # Obtaining the member 'masked_where' of a type (line 1042)
            masked_where_126919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 22), ma_126918, 'masked_where')
            # Calling masked_where(args, kwargs) (line 1042)
            masked_where_call_result_126923 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 22), masked_where_126919, *[mymask_126920, data_126921], **kwargs_126922)
            
            # Assigning a type to the variable 'newdata' (line 1042)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 12), 'newdata', masked_where_call_result_126923)

            if (may_be_126898 and more_types_in_union_126899):
                # SSA join for if statement (line 1028)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'newdata' (line 1044)
        newdata_126924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 15), 'newdata')
        # Assigning a type to the variable 'stypy_return_type' (line 1044)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1044, 8), 'stypy_return_type', newdata_126924)
        
        # ################# End of '_apply_missing_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_apply_missing_value' in the type store
        # Getting the type of 'stypy_return_type' (line 1018)
        stypy_return_type_126925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_126925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_apply_missing_value'
        return stypy_return_type_126925


# Assigning a type to the variable 'netcdf_variable' (line 786)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 0), 'netcdf_variable', netcdf_variable)

# Assigning a Call to a Name (line 872):

# Call to property(...): (line 872)
# Processing the call arguments (line 872)
# Getting the type of 'netcdf_variable'
netcdf_variable_126927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_variable', False)
# Obtaining the member 'isrec' of a type
isrec_126928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_variable_126927, 'isrec')
# Processing the call keyword arguments (line 872)
kwargs_126929 = {}
# Getting the type of 'property' (line 872)
property_126926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'property', False)
# Calling property(args, kwargs) (line 872)
property_call_result_126930 = invoke(stypy.reporting.localization.Localization(__file__, 872, 12), property_126926, *[isrec_126928], **kwargs_126929)

# Getting the type of 'netcdf_variable'
netcdf_variable_126931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_variable')
# Setting the type of the member 'isrec' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_variable_126931, 'isrec', property_call_result_126930)

# Assigning a Call to a Name (line 881):

# Call to property(...): (line 881)
# Processing the call arguments (line 881)
# Getting the type of 'netcdf_variable'
netcdf_variable_126933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_variable', False)
# Obtaining the member 'shape' of a type
shape_126934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_variable_126933, 'shape')
# Processing the call keyword arguments (line 881)
kwargs_126935 = {}
# Getting the type of 'property' (line 881)
property_126932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'property', False)
# Calling property(args, kwargs) (line 881)
property_call_result_126936 = invoke(stypy.reporting.localization.Localization(__file__, 881, 12), property_126932, *[shape_126934], **kwargs_126935)

# Getting the type of 'netcdf_variable'
netcdf_variable_126937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'netcdf_variable')
# Setting the type of the member 'shape' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), netcdf_variable_126937, 'shape', property_call_result_126936)

# Assigning a Name to a Name (line 1047):

# Assigning a Name to a Name (line 1047):
# Getting the type of 'netcdf_file' (line 1047)
netcdf_file_126938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 13), 'netcdf_file')
# Assigning a type to the variable 'NetCDFFile' (line 1047)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 0), 'NetCDFFile', netcdf_file_126938)

# Assigning a Name to a Name (line 1048):

# Assigning a Name to a Name (line 1048):
# Getting the type of 'netcdf_variable' (line 1048)
netcdf_variable_126939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 17), 'netcdf_variable')
# Assigning a type to the variable 'NetCDFVariable' (line 1048)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1048, 0), 'NetCDFVariable', netcdf_variable_126939)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
