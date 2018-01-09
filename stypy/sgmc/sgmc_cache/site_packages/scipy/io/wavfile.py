
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module to read / write wav files using numpy arrays
3: 
4: Functions
5: ---------
6: `read`: Return the sample rate (in samples/sec) and data from a WAV file.
7: 
8: `write`: Write a numpy array as a WAV file.
9: 
10: '''
11: from __future__ import division, print_function, absolute_import
12: 
13: import sys
14: import numpy
15: import struct
16: import warnings
17: 
18: 
19: __all__ = [
20:     'WavFileWarning',
21:     'read',
22:     'write'
23: ]
24: 
25: 
26: class WavFileWarning(UserWarning):
27:     pass
28: 
29: 
30: WAVE_FORMAT_PCM = 0x0001
31: WAVE_FORMAT_IEEE_FLOAT = 0x0003
32: WAVE_FORMAT_EXTENSIBLE = 0xfffe
33: KNOWN_WAVE_FORMATS = (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)
34: 
35: # assumes file pointer is immediately
36: #  after the 'fmt ' id
37: 
38: 
39: def _read_fmt_chunk(fid, is_big_endian):
40:     '''
41:     Returns
42:     -------
43:     size : int
44:         size of format subchunk in bytes (minus 8 for "fmt " and itself)
45:     format_tag : int
46:         PCM, float, or compressed format
47:     channels : int
48:         number of channels
49:     fs : int
50:         sampling frequency in samples per second
51:     bytes_per_second : int
52:         overall byte rate for the file
53:     block_align : int
54:         bytes per sample, including all channels
55:     bit_depth : int
56:         bits per sample
57:     '''
58:     if is_big_endian:
59:         fmt = '>'
60:     else:
61:         fmt = '<'
62: 
63:     size = res = struct.unpack(fmt+'I', fid.read(4))[0]
64:     bytes_read = 0
65: 
66:     if size < 16:
67:         raise ValueError("Binary structure of wave file is not compliant")
68: 
69:     res = struct.unpack(fmt+'HHIIHH', fid.read(16))
70:     bytes_read += 16
71: 
72:     format_tag, channels, fs, bytes_per_second, block_align, bit_depth = res
73: 
74:     if format_tag == WAVE_FORMAT_EXTENSIBLE and size >= (16+2):
75:         ext_chunk_size = struct.unpack(fmt+'H', fid.read(2))[0]
76:         bytes_read += 2
77:         if ext_chunk_size >= 22:
78:             extensible_chunk_data = fid.read(22)
79:             bytes_read += 22
80:             raw_guid = extensible_chunk_data[2+4:2+4+16]
81:             # GUID template {XXXXXXXX-0000-0010-8000-00AA00389B71} (RFC-2361)
82:             # MS GUID byte order: first three groups are native byte order,
83:             # rest is Big Endian
84:             if is_big_endian:
85:                 tail = b'\x00\x00\x00\x10\x80\x00\x00\xAA\x00\x38\x9B\x71'
86:             else:
87:                 tail = b'\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71'
88:             if raw_guid.endswith(tail):
89:                 format_tag = struct.unpack(fmt+'I', raw_guid[:4])[0]
90:         else:
91:             raise ValueError("Binary structure of wave file is not compliant")
92: 
93:     if format_tag not in KNOWN_WAVE_FORMATS:
94:         raise ValueError("Unknown wave file format")
95: 
96:     # move file pointer to next chunk
97:     if size > (bytes_read):
98:         fid.read(size - bytes_read)
99: 
100:     return (size, format_tag, channels, fs, bytes_per_second, block_align,
101:             bit_depth)
102: 
103: 
104: # assumes file pointer is immediately after the 'data' id
105: def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian,
106:                      mmap=False):
107:     if is_big_endian:
108:         fmt = '>I'
109:     else:
110:         fmt = '<I'
111: 
112:     # Size of the data subchunk in bytes
113:     size = struct.unpack(fmt, fid.read(4))[0]
114: 
115:     # Number of bytes per sample
116:     bytes_per_sample = bit_depth//8
117:     if bit_depth == 8:
118:         dtype = 'u1'
119:     else:
120:         if is_big_endian:
121:             dtype = '>'
122:         else:
123:             dtype = '<'
124:         if format_tag == WAVE_FORMAT_PCM:
125:             dtype += 'i%d' % bytes_per_sample
126:         else:
127:             dtype += 'f%d' % bytes_per_sample
128:     if not mmap:
129:         data = numpy.fromstring(fid.read(size), dtype=dtype)
130:     else:
131:         start = fid.tell()
132:         data = numpy.memmap(fid, dtype=dtype, mode='c', offset=start,
133:                             shape=(size//bytes_per_sample,))
134:         fid.seek(start + size)
135: 
136:     if channels > 1:
137:         data = data.reshape(-1, channels)
138:     return data
139: 
140: 
141: def _skip_unknown_chunk(fid, is_big_endian):
142:     if is_big_endian:
143:         fmt = '>I'
144:     else:
145:         fmt = '<I'
146: 
147:     data = fid.read(4)
148:     # call unpack() and seek() only if we have really read data from file
149:     # otherwise empty read at the end of the file would trigger
150:     # unnecessary exception at unpack() call
151:     # in case data equals somehow to 0, there is no need for seek() anyway
152:     if data:
153:         size = struct.unpack(fmt, data)[0]
154:         fid.seek(size, 1)
155: 
156: 
157: def _read_riff_chunk(fid):
158:     str1 = fid.read(4)  # File signature
159:     if str1 == b'RIFF':
160:         is_big_endian = False
161:         fmt = '<I'
162:     elif str1 == b'RIFX':
163:         is_big_endian = True
164:         fmt = '>I'
165:     else:
166:         # There are also .wav files with "FFIR" or "XFIR" signatures?
167:         raise ValueError("File format {}... not "
168:                          "understood.".format(repr(str1)))
169: 
170:     # Size of entire file
171:     file_size = struct.unpack(fmt, fid.read(4))[0] + 8
172: 
173:     str2 = fid.read(4)
174:     if str2 != b'WAVE':
175:         raise ValueError("Not a WAV file.")
176: 
177:     return file_size, is_big_endian
178: 
179: 
180: def read(filename, mmap=False):
181:     '''
182:     Open a WAV file
183: 
184:     Return the sample rate (in samples/sec) and data from a WAV file.
185: 
186:     Parameters
187:     ----------
188:     filename : string or open file handle
189:         Input wav file.
190:     mmap : bool, optional
191:         Whether to read data as memory-mapped.
192:         Only to be used on real files (Default: False).
193: 
194:         .. versionadded:: 0.12.0
195: 
196:     Returns
197:     -------
198:     rate : int
199:         Sample rate of wav file.
200:     data : numpy array
201:         Data read from wav file.  Data-type is determined from the file;
202:         see Notes.
203: 
204:     Notes
205:     -----
206:     This function cannot read wav files with 24-bit data.
207: 
208:     Common data types: [1]_
209: 
210:     =====================  ===========  ===========  =============
211:          WAV format            Min          Max       NumPy dtype
212:     =====================  ===========  ===========  =============
213:     32-bit floating-point  -1.0         +1.0         float32
214:     32-bit PCM             -2147483648  +2147483647  int32
215:     16-bit PCM             -32768       +32767       int16
216:     8-bit PCM              0            255          uint8
217:     =====================  ===========  ===========  =============
218: 
219:     Note that 8-bit PCM is unsigned.
220: 
221:     References
222:     ----------
223:     .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
224:        Interface and Data Specifications 1.0", section "Data Format of the
225:        Samples", August 1991
226:        http://www.tactilemedia.com/info/MCI_Control_Info.html
227: 
228:     '''
229:     if hasattr(filename, 'read'):
230:         fid = filename
231:         mmap = False
232:     else:
233:         fid = open(filename, 'rb')
234: 
235:     try:
236:         file_size, is_big_endian = _read_riff_chunk(fid)
237:         fmt_chunk_received = False
238:         channels = 1
239:         bit_depth = 8
240:         format_tag = WAVE_FORMAT_PCM
241:         while fid.tell() < file_size:
242:             # read the next chunk
243:             chunk_id = fid.read(4)
244: 
245:             if not chunk_id:
246:                 raise ValueError("Unexpected end of file.")
247:             elif len(chunk_id) < 4:
248:                 raise ValueError("Incomplete wav chunk.")
249: 
250:             if chunk_id == b'fmt ':
251:                 fmt_chunk_received = True
252:                 fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
253:                 format_tag, channels, fs = fmt_chunk[1:4]
254:                 bit_depth = fmt_chunk[6]
255:                 if bit_depth not in (8, 16, 32, 64, 96, 128):
256:                     raise ValueError("Unsupported bit depth: the wav file "
257:                                      "has {}-bit data.".format(bit_depth))
258:             elif chunk_id == b'fact':
259:                 _skip_unknown_chunk(fid, is_big_endian)
260:             elif chunk_id == b'data':
261:                 if not fmt_chunk_received:
262:                     raise ValueError("No fmt chunk before data")
263:                 data = _read_data_chunk(fid, format_tag, channels, bit_depth,
264:                                         is_big_endian, mmap)
265:             elif chunk_id == b'LIST':
266:                 # Someday this could be handled properly but for now skip it
267:                 _skip_unknown_chunk(fid, is_big_endian)
268:             elif chunk_id in (b'JUNK', b'Fake'):
269:                 # Skip alignment chunks without warning
270:                 _skip_unknown_chunk(fid, is_big_endian)
271:             else:
272:                 warnings.warn("Chunk (non-data) not understood, skipping it.",
273:                               WavFileWarning)
274:                 _skip_unknown_chunk(fid, is_big_endian)
275:     finally:
276:         if not hasattr(filename, 'read'):
277:             fid.close()
278:         else:
279:             fid.seek(0)
280: 
281:     return fs, data
282: 
283: 
284: def write(filename, rate, data):
285:     '''
286:     Write a numpy array as a WAV file.
287: 
288:     Parameters
289:     ----------
290:     filename : string or open file handle
291:         Output wav file.
292:     rate : int
293:         The sample rate (in samples/sec).
294:     data : ndarray
295:         A 1-D or 2-D numpy array of either integer or float data-type.
296: 
297:     Notes
298:     -----
299:     * Writes a simple uncompressed WAV file.
300:     * To write multiple-channels, use a 2-D array of shape
301:       (Nsamples, Nchannels).
302:     * The bits-per-sample and PCM/float will be determined by the data-type.
303: 
304:     Common data types: [1]_
305: 
306:     =====================  ===========  ===========  =============
307:          WAV format            Min          Max       NumPy dtype
308:     =====================  ===========  ===========  =============
309:     32-bit floating-point  -1.0         +1.0         float32
310:     32-bit PCM             -2147483648  +2147483647  int32
311:     16-bit PCM             -32768       +32767       int16
312:     8-bit PCM              0            255          uint8
313:     =====================  ===========  ===========  =============
314: 
315:     Note that 8-bit PCM is unsigned.
316: 
317:     References
318:     ----------
319:     .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
320:        Interface and Data Specifications 1.0", section "Data Format of the
321:        Samples", August 1991
322:        http://www.tactilemedia.com/info/MCI_Control_Info.html
323: 
324:     '''
325:     if hasattr(filename, 'write'):
326:         fid = filename
327:     else:
328:         fid = open(filename, 'wb')
329: 
330:     fs = rate
331: 
332:     try:
333:         dkind = data.dtype.kind
334:         if not (dkind == 'i' or dkind == 'f' or (dkind == 'u' and
335:                                                  data.dtype.itemsize == 1)):
336:             raise ValueError("Unsupported data type '%s'" % data.dtype)
337: 
338:         header_data = b''
339: 
340:         header_data += b'RIFF'
341:         header_data += b'\x00\x00\x00\x00'
342:         header_data += b'WAVE'
343: 
344:         # fmt chunk
345:         header_data += b'fmt '
346:         if dkind == 'f':
347:             format_tag = WAVE_FORMAT_IEEE_FLOAT
348:         else:
349:             format_tag = WAVE_FORMAT_PCM
350:         if data.ndim == 1:
351:             channels = 1
352:         else:
353:             channels = data.shape[1]
354:         bit_depth = data.dtype.itemsize * 8
355:         bytes_per_second = fs*(bit_depth // 8)*channels
356:         block_align = channels * (bit_depth // 8)
357: 
358:         fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs,
359:                                      bytes_per_second, block_align, bit_depth)
360:         if not (dkind == 'i' or dkind == 'u'):
361:             # add cbSize field for non-PCM files
362:             fmt_chunk_data += b'\x00\x00'
363: 
364:         header_data += struct.pack('<I', len(fmt_chunk_data))
365:         header_data += fmt_chunk_data
366: 
367:         # fact chunk (non-PCM files)
368:         if not (dkind == 'i' or dkind == 'u'):
369:             header_data += b'fact'
370:             header_data += struct.pack('<II', 4, data.shape[0])
371: 
372:         # check data size (needs to be immediately before the data chunk)
373:         if ((len(header_data)-4-4) + (4+4+data.nbytes)) > 0xFFFFFFFF:
374:             raise ValueError("Data exceeds wave file size limit")
375: 
376:         fid.write(header_data)
377: 
378:         # data chunk
379:         fid.write(b'data')
380:         fid.write(struct.pack('<I', data.nbytes))
381:         if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and
382:                                            sys.byteorder == 'big'):
383:             data = data.byteswap()
384:         _array_tofile(fid, data)
385: 
386:         # Determine file size and place it in correct
387:         #  position at start of the file.
388:         size = fid.tell()
389:         fid.seek(4)
390:         fid.write(struct.pack('<I', size-8))
391: 
392:     finally:
393:         if not hasattr(filename, 'write'):
394:             fid.close()
395:         else:
396:             fid.seek(0)
397: 
398: 
399: if sys.version_info[0] >= 3:
400:     def _array_tofile(fid, data):
401:         # ravel gives a c-contiguous buffer
402:         fid.write(data.ravel().view('b').data)
403: else:
404:     def _array_tofile(fid, data):
405:         fid.write(data.tostring())
406: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_127005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', '\nModule to read / write wav files using numpy arrays\n\nFunctions\n---------\n`read`: Return the sample rate (in samples/sec) and data from a WAV file.\n\n`write`: Write a numpy array as a WAV file.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import sys' statement (line 13)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import numpy' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_127006 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy')

if (type(import_127006) is not StypyTypeError):

    if (import_127006 != 'pyd_module'):
        __import__(import_127006)
        sys_modules_127007 = sys.modules[import_127006]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', sys_modules_127007.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy', import_127006)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import struct' statement (line 15)
import struct

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'struct', struct, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import warnings' statement (line 16)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'warnings', warnings, module_type_store)


# Assigning a List to a Name (line 19):

# Assigning a List to a Name (line 19):
__all__ = ['WavFileWarning', 'read', 'write']
module_type_store.set_exportable_members(['WavFileWarning', 'read', 'write'])

# Obtaining an instance of the builtin type 'list' (line 19)
list_127008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_127009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', 'WavFileWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_127008, str_127009)
# Adding element type (line 19)
str_127010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', 'read')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_127008, str_127010)
# Adding element type (line 19)
str_127011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', 'write')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_127008, str_127011)

# Assigning a type to the variable '__all__' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__all__', list_127008)
# Declaration of the 'WavFileWarning' class
# Getting the type of 'UserWarning' (line 26)
UserWarning_127012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'UserWarning')

class WavFileWarning(UserWarning_127012, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 0, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WavFileWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'WavFileWarning' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'WavFileWarning', WavFileWarning)

# Assigning a Num to a Name (line 30):

# Assigning a Num to a Name (line 30):
int_127013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'int')
# Assigning a type to the variable 'WAVE_FORMAT_PCM' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'WAVE_FORMAT_PCM', int_127013)

# Assigning a Num to a Name (line 31):

# Assigning a Num to a Name (line 31):
int_127014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'int')
# Assigning a type to the variable 'WAVE_FORMAT_IEEE_FLOAT' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'WAVE_FORMAT_IEEE_FLOAT', int_127014)

# Assigning a Num to a Name (line 32):

# Assigning a Num to a Name (line 32):
int_127015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'int')
# Assigning a type to the variable 'WAVE_FORMAT_EXTENSIBLE' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'WAVE_FORMAT_EXTENSIBLE', int_127015)

# Assigning a Tuple to a Name (line 33):

# Assigning a Tuple to a Name (line 33):

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_127016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
# Getting the type of 'WAVE_FORMAT_PCM' (line 33)
WAVE_FORMAT_PCM_127017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'WAVE_FORMAT_PCM')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 22), tuple_127016, WAVE_FORMAT_PCM_127017)
# Adding element type (line 33)
# Getting the type of 'WAVE_FORMAT_IEEE_FLOAT' (line 33)
WAVE_FORMAT_IEEE_FLOAT_127018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 39), 'WAVE_FORMAT_IEEE_FLOAT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 22), tuple_127016, WAVE_FORMAT_IEEE_FLOAT_127018)

# Assigning a type to the variable 'KNOWN_WAVE_FORMATS' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'KNOWN_WAVE_FORMATS', tuple_127016)

@norecursion
def _read_fmt_chunk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_fmt_chunk'
    module_type_store = module_type_store.open_function_context('_read_fmt_chunk', 39, 0, False)
    
    # Passed parameters checking function
    _read_fmt_chunk.stypy_localization = localization
    _read_fmt_chunk.stypy_type_of_self = None
    _read_fmt_chunk.stypy_type_store = module_type_store
    _read_fmt_chunk.stypy_function_name = '_read_fmt_chunk'
    _read_fmt_chunk.stypy_param_names_list = ['fid', 'is_big_endian']
    _read_fmt_chunk.stypy_varargs_param_name = None
    _read_fmt_chunk.stypy_kwargs_param_name = None
    _read_fmt_chunk.stypy_call_defaults = defaults
    _read_fmt_chunk.stypy_call_varargs = varargs
    _read_fmt_chunk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_fmt_chunk', ['fid', 'is_big_endian'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_fmt_chunk', localization, ['fid', 'is_big_endian'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_fmt_chunk(...)' code ##################

    str_127019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', '\n    Returns\n    -------\n    size : int\n        size of format subchunk in bytes (minus 8 for "fmt " and itself)\n    format_tag : int\n        PCM, float, or compressed format\n    channels : int\n        number of channels\n    fs : int\n        sampling frequency in samples per second\n    bytes_per_second : int\n        overall byte rate for the file\n    block_align : int\n        bytes per sample, including all channels\n    bit_depth : int\n        bits per sample\n    ')
    
    # Getting the type of 'is_big_endian' (line 58)
    is_big_endian_127020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'is_big_endian')
    # Testing the type of an if condition (line 58)
    if_condition_127021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), is_big_endian_127020)
    # Assigning a type to the variable 'if_condition_127021' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_127021', if_condition_127021)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 59):
    
    # Assigning a Str to a Name (line 59):
    str_127022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 14), 'str', '>')
    # Assigning a type to the variable 'fmt' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'fmt', str_127022)
    # SSA branch for the else part of an if statement (line 58)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 61):
    
    # Assigning a Str to a Name (line 61):
    str_127023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 14), 'str', '<')
    # Assigning a type to the variable 'fmt' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'fmt', str_127023)
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Subscript to a Name (line 63):
    
    # Obtaining the type of the subscript
    int_127024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 53), 'int')
    
    # Call to unpack(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'fmt' (line 63)
    fmt_127027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'fmt', False)
    str_127028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 35), 'str', 'I')
    # Applying the binary operator '+' (line 63)
    result_add_127029 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 31), '+', fmt_127027, str_127028)
    
    
    # Call to read(...): (line 63)
    # Processing the call arguments (line 63)
    int_127032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 49), 'int')
    # Processing the call keyword arguments (line 63)
    kwargs_127033 = {}
    # Getting the type of 'fid' (line 63)
    fid_127030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'fid', False)
    # Obtaining the member 'read' of a type (line 63)
    read_127031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 40), fid_127030, 'read')
    # Calling read(args, kwargs) (line 63)
    read_call_result_127034 = invoke(stypy.reporting.localization.Localization(__file__, 63, 40), read_127031, *[int_127032], **kwargs_127033)
    
    # Processing the call keyword arguments (line 63)
    kwargs_127035 = {}
    # Getting the type of 'struct' (line 63)
    struct_127025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 63)
    unpack_127026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 17), struct_127025, 'unpack')
    # Calling unpack(args, kwargs) (line 63)
    unpack_call_result_127036 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), unpack_127026, *[result_add_127029, read_call_result_127034], **kwargs_127035)
    
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___127037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 17), unpack_call_result_127036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_127038 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), getitem___127037, int_127024)
    
    # Assigning a type to the variable 'res' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'res', subscript_call_result_127038)
    
    # Assigning a Name to a Name (line 63):
    # Getting the type of 'res' (line 63)
    res_127039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'res')
    # Assigning a type to the variable 'size' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'size', res_127039)
    
    # Assigning a Num to a Name (line 64):
    
    # Assigning a Num to a Name (line 64):
    int_127040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 17), 'int')
    # Assigning a type to the variable 'bytes_read' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'bytes_read', int_127040)
    
    
    # Getting the type of 'size' (line 66)
    size_127041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'size')
    int_127042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'int')
    # Applying the binary operator '<' (line 66)
    result_lt_127043 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), '<', size_127041, int_127042)
    
    # Testing the type of an if condition (line 66)
    if_condition_127044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_lt_127043)
    # Assigning a type to the variable 'if_condition_127044' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_127044', if_condition_127044)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 67)
    # Processing the call arguments (line 67)
    str_127046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'str', 'Binary structure of wave file is not compliant')
    # Processing the call keyword arguments (line 67)
    kwargs_127047 = {}
    # Getting the type of 'ValueError' (line 67)
    ValueError_127045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 67)
    ValueError_call_result_127048 = invoke(stypy.reporting.localization.Localization(__file__, 67, 14), ValueError_127045, *[str_127046], **kwargs_127047)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 67, 8), ValueError_call_result_127048, 'raise parameter', BaseException)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to unpack(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'fmt' (line 69)
    fmt_127051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'fmt', False)
    str_127052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'str', 'HHIIHH')
    # Applying the binary operator '+' (line 69)
    result_add_127053 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 24), '+', fmt_127051, str_127052)
    
    
    # Call to read(...): (line 69)
    # Processing the call arguments (line 69)
    int_127056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 47), 'int')
    # Processing the call keyword arguments (line 69)
    kwargs_127057 = {}
    # Getting the type of 'fid' (line 69)
    fid_127054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'fid', False)
    # Obtaining the member 'read' of a type (line 69)
    read_127055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 38), fid_127054, 'read')
    # Calling read(args, kwargs) (line 69)
    read_call_result_127058 = invoke(stypy.reporting.localization.Localization(__file__, 69, 38), read_127055, *[int_127056], **kwargs_127057)
    
    # Processing the call keyword arguments (line 69)
    kwargs_127059 = {}
    # Getting the type of 'struct' (line 69)
    struct_127049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 10), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 69)
    unpack_127050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 10), struct_127049, 'unpack')
    # Calling unpack(args, kwargs) (line 69)
    unpack_call_result_127060 = invoke(stypy.reporting.localization.Localization(__file__, 69, 10), unpack_127050, *[result_add_127053, read_call_result_127058], **kwargs_127059)
    
    # Assigning a type to the variable 'res' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'res', unpack_call_result_127060)
    
    # Getting the type of 'bytes_read' (line 70)
    bytes_read_127061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'bytes_read')
    int_127062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'int')
    # Applying the binary operator '+=' (line 70)
    result_iadd_127063 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 4), '+=', bytes_read_127061, int_127062)
    # Assigning a type to the variable 'bytes_read' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'bytes_read', result_iadd_127063)
    
    
    # Assigning a Name to a Tuple (line 72):
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_127064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'res' (line 72)
    res_127065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 73), 'res')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___127066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), res_127065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_127067 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___127066, int_127064)
    
    # Assigning a type to the variable 'tuple_var_assignment_126994' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126994', subscript_call_result_127067)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_127068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'res' (line 72)
    res_127069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 73), 'res')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___127070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), res_127069, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_127071 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___127070, int_127068)
    
    # Assigning a type to the variable 'tuple_var_assignment_126995' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126995', subscript_call_result_127071)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_127072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'res' (line 72)
    res_127073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 73), 'res')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___127074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), res_127073, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_127075 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___127074, int_127072)
    
    # Assigning a type to the variable 'tuple_var_assignment_126996' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126996', subscript_call_result_127075)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_127076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'res' (line 72)
    res_127077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 73), 'res')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___127078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), res_127077, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_127079 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___127078, int_127076)
    
    # Assigning a type to the variable 'tuple_var_assignment_126997' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126997', subscript_call_result_127079)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_127080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'res' (line 72)
    res_127081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 73), 'res')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___127082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), res_127081, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_127083 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___127082, int_127080)
    
    # Assigning a type to the variable 'tuple_var_assignment_126998' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126998', subscript_call_result_127083)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_127084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    # Getting the type of 'res' (line 72)
    res_127085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 73), 'res')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___127086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), res_127085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_127087 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___127086, int_127084)
    
    # Assigning a type to the variable 'tuple_var_assignment_126999' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126999', subscript_call_result_127087)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_126994' (line 72)
    tuple_var_assignment_126994_127088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126994')
    # Assigning a type to the variable 'format_tag' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'format_tag', tuple_var_assignment_126994_127088)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_126995' (line 72)
    tuple_var_assignment_126995_127089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126995')
    # Assigning a type to the variable 'channels' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'channels', tuple_var_assignment_126995_127089)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_126996' (line 72)
    tuple_var_assignment_126996_127090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126996')
    # Assigning a type to the variable 'fs' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'fs', tuple_var_assignment_126996_127090)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_126997' (line 72)
    tuple_var_assignment_126997_127091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126997')
    # Assigning a type to the variable 'bytes_per_second' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'bytes_per_second', tuple_var_assignment_126997_127091)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_126998' (line 72)
    tuple_var_assignment_126998_127092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126998')
    # Assigning a type to the variable 'block_align' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 48), 'block_align', tuple_var_assignment_126998_127092)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_126999' (line 72)
    tuple_var_assignment_126999_127093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_126999')
    # Assigning a type to the variable 'bit_depth' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 61), 'bit_depth', tuple_var_assignment_126999_127093)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'format_tag' (line 74)
    format_tag_127094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'format_tag')
    # Getting the type of 'WAVE_FORMAT_EXTENSIBLE' (line 74)
    WAVE_FORMAT_EXTENSIBLE_127095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'WAVE_FORMAT_EXTENSIBLE')
    # Applying the binary operator '==' (line 74)
    result_eq_127096 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 7), '==', format_tag_127094, WAVE_FORMAT_EXTENSIBLE_127095)
    
    
    # Getting the type of 'size' (line 74)
    size_127097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 48), 'size')
    int_127098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 57), 'int')
    int_127099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 60), 'int')
    # Applying the binary operator '+' (line 74)
    result_add_127100 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 57), '+', int_127098, int_127099)
    
    # Applying the binary operator '>=' (line 74)
    result_ge_127101 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 48), '>=', size_127097, result_add_127100)
    
    # Applying the binary operator 'and' (line 74)
    result_and_keyword_127102 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 7), 'and', result_eq_127096, result_ge_127101)
    
    # Testing the type of an if condition (line 74)
    if_condition_127103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), result_and_keyword_127102)
    # Assigning a type to the variable 'if_condition_127103' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_127103', if_condition_127103)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 75):
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_127104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 61), 'int')
    
    # Call to unpack(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'fmt' (line 75)
    fmt_127107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 39), 'fmt', False)
    str_127108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 43), 'str', 'H')
    # Applying the binary operator '+' (line 75)
    result_add_127109 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 39), '+', fmt_127107, str_127108)
    
    
    # Call to read(...): (line 75)
    # Processing the call arguments (line 75)
    int_127112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 57), 'int')
    # Processing the call keyword arguments (line 75)
    kwargs_127113 = {}
    # Getting the type of 'fid' (line 75)
    fid_127110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 48), 'fid', False)
    # Obtaining the member 'read' of a type (line 75)
    read_127111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 48), fid_127110, 'read')
    # Calling read(args, kwargs) (line 75)
    read_call_result_127114 = invoke(stypy.reporting.localization.Localization(__file__, 75, 48), read_127111, *[int_127112], **kwargs_127113)
    
    # Processing the call keyword arguments (line 75)
    kwargs_127115 = {}
    # Getting the type of 'struct' (line 75)
    struct_127105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 75)
    unpack_127106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), struct_127105, 'unpack')
    # Calling unpack(args, kwargs) (line 75)
    unpack_call_result_127116 = invoke(stypy.reporting.localization.Localization(__file__, 75, 25), unpack_127106, *[result_add_127109, read_call_result_127114], **kwargs_127115)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___127117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), unpack_call_result_127116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_127118 = invoke(stypy.reporting.localization.Localization(__file__, 75, 25), getitem___127117, int_127104)
    
    # Assigning a type to the variable 'ext_chunk_size' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'ext_chunk_size', subscript_call_result_127118)
    
    # Getting the type of 'bytes_read' (line 76)
    bytes_read_127119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'bytes_read')
    int_127120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'int')
    # Applying the binary operator '+=' (line 76)
    result_iadd_127121 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 8), '+=', bytes_read_127119, int_127120)
    # Assigning a type to the variable 'bytes_read' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'bytes_read', result_iadd_127121)
    
    
    
    # Getting the type of 'ext_chunk_size' (line 77)
    ext_chunk_size_127122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'ext_chunk_size')
    int_127123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 29), 'int')
    # Applying the binary operator '>=' (line 77)
    result_ge_127124 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), '>=', ext_chunk_size_127122, int_127123)
    
    # Testing the type of an if condition (line 77)
    if_condition_127125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), result_ge_127124)
    # Assigning a type to the variable 'if_condition_127125' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_127125', if_condition_127125)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to read(...): (line 78)
    # Processing the call arguments (line 78)
    int_127128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 45), 'int')
    # Processing the call keyword arguments (line 78)
    kwargs_127129 = {}
    # Getting the type of 'fid' (line 78)
    fid_127126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 36), 'fid', False)
    # Obtaining the member 'read' of a type (line 78)
    read_127127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 36), fid_127126, 'read')
    # Calling read(args, kwargs) (line 78)
    read_call_result_127130 = invoke(stypy.reporting.localization.Localization(__file__, 78, 36), read_127127, *[int_127128], **kwargs_127129)
    
    # Assigning a type to the variable 'extensible_chunk_data' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'extensible_chunk_data', read_call_result_127130)
    
    # Getting the type of 'bytes_read' (line 79)
    bytes_read_127131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'bytes_read')
    int_127132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'int')
    # Applying the binary operator '+=' (line 79)
    result_iadd_127133 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 12), '+=', bytes_read_127131, int_127132)
    # Assigning a type to the variable 'bytes_read' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'bytes_read', result_iadd_127133)
    
    
    # Assigning a Subscript to a Name (line 80):
    
    # Assigning a Subscript to a Name (line 80):
    
    # Obtaining the type of the subscript
    int_127134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 45), 'int')
    int_127135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 47), 'int')
    # Applying the binary operator '+' (line 80)
    result_add_127136 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 45), '+', int_127134, int_127135)
    
    int_127137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 49), 'int')
    int_127138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 51), 'int')
    # Applying the binary operator '+' (line 80)
    result_add_127139 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 49), '+', int_127137, int_127138)
    
    int_127140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 53), 'int')
    # Applying the binary operator '+' (line 80)
    result_add_127141 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 52), '+', result_add_127139, int_127140)
    
    slice_127142 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 23), result_add_127136, result_add_127141, None)
    # Getting the type of 'extensible_chunk_data' (line 80)
    extensible_chunk_data_127143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'extensible_chunk_data')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___127144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 23), extensible_chunk_data_127143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_127145 = invoke(stypy.reporting.localization.Localization(__file__, 80, 23), getitem___127144, slice_127142)
    
    # Assigning a type to the variable 'raw_guid' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'raw_guid', subscript_call_result_127145)
    
    # Getting the type of 'is_big_endian' (line 84)
    is_big_endian_127146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'is_big_endian')
    # Testing the type of an if condition (line 84)
    if_condition_127147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 12), is_big_endian_127146)
    # Assigning a type to the variable 'if_condition_127147' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'if_condition_127147', if_condition_127147)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 85):
    
    # Assigning a Str to a Name (line 85):
    str_127148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'str', '\x00\x00\x00\x10\x80\x00\x00\xaa\x008\x9bq')
    # Assigning a type to the variable 'tail' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'tail', str_127148)
    # SSA branch for the else part of an if statement (line 84)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 87):
    
    # Assigning a Str to a Name (line 87):
    str_127149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'str', '\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq')
    # Assigning a type to the variable 'tail' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'tail', str_127149)
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to endswith(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'tail' (line 88)
    tail_127152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'tail', False)
    # Processing the call keyword arguments (line 88)
    kwargs_127153 = {}
    # Getting the type of 'raw_guid' (line 88)
    raw_guid_127150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'raw_guid', False)
    # Obtaining the member 'endswith' of a type (line 88)
    endswith_127151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), raw_guid_127150, 'endswith')
    # Calling endswith(args, kwargs) (line 88)
    endswith_call_result_127154 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), endswith_127151, *[tail_127152], **kwargs_127153)
    
    # Testing the type of an if condition (line 88)
    if_condition_127155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 12), endswith_call_result_127154)
    # Assigning a type to the variable 'if_condition_127155' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'if_condition_127155', if_condition_127155)
    # SSA begins for if statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 89):
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_127156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 66), 'int')
    
    # Call to unpack(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'fmt' (line 89)
    fmt_127159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 43), 'fmt', False)
    str_127160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 47), 'str', 'I')
    # Applying the binary operator '+' (line 89)
    result_add_127161 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 43), '+', fmt_127159, str_127160)
    
    
    # Obtaining the type of the subscript
    int_127162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 62), 'int')
    slice_127163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 52), None, int_127162, None)
    # Getting the type of 'raw_guid' (line 89)
    raw_guid_127164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 52), 'raw_guid', False)
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___127165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 52), raw_guid_127164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_127166 = invoke(stypy.reporting.localization.Localization(__file__, 89, 52), getitem___127165, slice_127163)
    
    # Processing the call keyword arguments (line 89)
    kwargs_127167 = {}
    # Getting the type of 'struct' (line 89)
    struct_127157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 89)
    unpack_127158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 29), struct_127157, 'unpack')
    # Calling unpack(args, kwargs) (line 89)
    unpack_call_result_127168 = invoke(stypy.reporting.localization.Localization(__file__, 89, 29), unpack_127158, *[result_add_127161, subscript_call_result_127166], **kwargs_127167)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___127169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 29), unpack_call_result_127168, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_127170 = invoke(stypy.reporting.localization.Localization(__file__, 89, 29), getitem___127169, int_127156)
    
    # Assigning a type to the variable 'format_tag' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'format_tag', subscript_call_result_127170)
    # SSA join for if statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 91)
    # Processing the call arguments (line 91)
    str_127172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'str', 'Binary structure of wave file is not compliant')
    # Processing the call keyword arguments (line 91)
    kwargs_127173 = {}
    # Getting the type of 'ValueError' (line 91)
    ValueError_127171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 91)
    ValueError_call_result_127174 = invoke(stypy.reporting.localization.Localization(__file__, 91, 18), ValueError_127171, *[str_127172], **kwargs_127173)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 91, 12), ValueError_call_result_127174, 'raise parameter', BaseException)
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'format_tag' (line 93)
    format_tag_127175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 7), 'format_tag')
    # Getting the type of 'KNOWN_WAVE_FORMATS' (line 93)
    KNOWN_WAVE_FORMATS_127176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'KNOWN_WAVE_FORMATS')
    # Applying the binary operator 'notin' (line 93)
    result_contains_127177 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 7), 'notin', format_tag_127175, KNOWN_WAVE_FORMATS_127176)
    
    # Testing the type of an if condition (line 93)
    if_condition_127178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), result_contains_127177)
    # Assigning a type to the variable 'if_condition_127178' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_127178', if_condition_127178)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 94)
    # Processing the call arguments (line 94)
    str_127180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'str', 'Unknown wave file format')
    # Processing the call keyword arguments (line 94)
    kwargs_127181 = {}
    # Getting the type of 'ValueError' (line 94)
    ValueError_127179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 94)
    ValueError_call_result_127182 = invoke(stypy.reporting.localization.Localization(__file__, 94, 14), ValueError_127179, *[str_127180], **kwargs_127181)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 94, 8), ValueError_call_result_127182, 'raise parameter', BaseException)
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'size' (line 97)
    size_127183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 7), 'size')
    # Getting the type of 'bytes_read' (line 97)
    bytes_read_127184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'bytes_read')
    # Applying the binary operator '>' (line 97)
    result_gt_127185 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 7), '>', size_127183, bytes_read_127184)
    
    # Testing the type of an if condition (line 97)
    if_condition_127186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 4), result_gt_127185)
    # Assigning a type to the variable 'if_condition_127186' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'if_condition_127186', if_condition_127186)
    # SSA begins for if statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to read(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'size' (line 98)
    size_127189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'size', False)
    # Getting the type of 'bytes_read' (line 98)
    bytes_read_127190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'bytes_read', False)
    # Applying the binary operator '-' (line 98)
    result_sub_127191 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 17), '-', size_127189, bytes_read_127190)
    
    # Processing the call keyword arguments (line 98)
    kwargs_127192 = {}
    # Getting the type of 'fid' (line 98)
    fid_127187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'fid', False)
    # Obtaining the member 'read' of a type (line 98)
    read_127188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), fid_127187, 'read')
    # Calling read(args, kwargs) (line 98)
    read_call_result_127193 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), read_127188, *[result_sub_127191], **kwargs_127192)
    
    # SSA join for if statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 100)
    tuple_127194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 100)
    # Adding element type (line 100)
    # Getting the type of 'size' (line 100)
    size_127195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), tuple_127194, size_127195)
    # Adding element type (line 100)
    # Getting the type of 'format_tag' (line 100)
    format_tag_127196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'format_tag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), tuple_127194, format_tag_127196)
    # Adding element type (line 100)
    # Getting the type of 'channels' (line 100)
    channels_127197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'channels')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), tuple_127194, channels_127197)
    # Adding element type (line 100)
    # Getting the type of 'fs' (line 100)
    fs_127198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'fs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), tuple_127194, fs_127198)
    # Adding element type (line 100)
    # Getting the type of 'bytes_per_second' (line 100)
    bytes_per_second_127199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'bytes_per_second')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), tuple_127194, bytes_per_second_127199)
    # Adding element type (line 100)
    # Getting the type of 'block_align' (line 100)
    block_align_127200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 62), 'block_align')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), tuple_127194, block_align_127200)
    # Adding element type (line 100)
    # Getting the type of 'bit_depth' (line 101)
    bit_depth_127201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'bit_depth')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), tuple_127194, bit_depth_127201)
    
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type', tuple_127194)
    
    # ################# End of '_read_fmt_chunk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_fmt_chunk' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_127202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127202)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_fmt_chunk'
    return stypy_return_type_127202

# Assigning a type to the variable '_read_fmt_chunk' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_read_fmt_chunk', _read_fmt_chunk)

@norecursion
def _read_data_chunk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 106)
    False_127203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'False')
    defaults = [False_127203]
    # Create a new context for function '_read_data_chunk'
    module_type_store = module_type_store.open_function_context('_read_data_chunk', 105, 0, False)
    
    # Passed parameters checking function
    _read_data_chunk.stypy_localization = localization
    _read_data_chunk.stypy_type_of_self = None
    _read_data_chunk.stypy_type_store = module_type_store
    _read_data_chunk.stypy_function_name = '_read_data_chunk'
    _read_data_chunk.stypy_param_names_list = ['fid', 'format_tag', 'channels', 'bit_depth', 'is_big_endian', 'mmap']
    _read_data_chunk.stypy_varargs_param_name = None
    _read_data_chunk.stypy_kwargs_param_name = None
    _read_data_chunk.stypy_call_defaults = defaults
    _read_data_chunk.stypy_call_varargs = varargs
    _read_data_chunk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_data_chunk', ['fid', 'format_tag', 'channels', 'bit_depth', 'is_big_endian', 'mmap'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_data_chunk', localization, ['fid', 'format_tag', 'channels', 'bit_depth', 'is_big_endian', 'mmap'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_data_chunk(...)' code ##################

    
    # Getting the type of 'is_big_endian' (line 107)
    is_big_endian_127204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 7), 'is_big_endian')
    # Testing the type of an if condition (line 107)
    if_condition_127205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 4), is_big_endian_127204)
    # Assigning a type to the variable 'if_condition_127205' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'if_condition_127205', if_condition_127205)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 108):
    
    # Assigning a Str to a Name (line 108):
    str_127206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 14), 'str', '>I')
    # Assigning a type to the variable 'fmt' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'fmt', str_127206)
    # SSA branch for the else part of an if statement (line 107)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 110):
    
    # Assigning a Str to a Name (line 110):
    str_127207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 14), 'str', '<I')
    # Assigning a type to the variable 'fmt' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'fmt', str_127207)
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 113):
    
    # Assigning a Subscript to a Name (line 113):
    
    # Obtaining the type of the subscript
    int_127208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 43), 'int')
    
    # Call to unpack(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'fmt' (line 113)
    fmt_127211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'fmt', False)
    
    # Call to read(...): (line 113)
    # Processing the call arguments (line 113)
    int_127214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 39), 'int')
    # Processing the call keyword arguments (line 113)
    kwargs_127215 = {}
    # Getting the type of 'fid' (line 113)
    fid_127212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'fid', False)
    # Obtaining the member 'read' of a type (line 113)
    read_127213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 30), fid_127212, 'read')
    # Calling read(args, kwargs) (line 113)
    read_call_result_127216 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), read_127213, *[int_127214], **kwargs_127215)
    
    # Processing the call keyword arguments (line 113)
    kwargs_127217 = {}
    # Getting the type of 'struct' (line 113)
    struct_127209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 113)
    unpack_127210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), struct_127209, 'unpack')
    # Calling unpack(args, kwargs) (line 113)
    unpack_call_result_127218 = invoke(stypy.reporting.localization.Localization(__file__, 113, 11), unpack_127210, *[fmt_127211, read_call_result_127216], **kwargs_127217)
    
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___127219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), unpack_call_result_127218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_127220 = invoke(stypy.reporting.localization.Localization(__file__, 113, 11), getitem___127219, int_127208)
    
    # Assigning a type to the variable 'size' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'size', subscript_call_result_127220)
    
    # Assigning a BinOp to a Name (line 116):
    
    # Assigning a BinOp to a Name (line 116):
    # Getting the type of 'bit_depth' (line 116)
    bit_depth_127221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'bit_depth')
    int_127222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 34), 'int')
    # Applying the binary operator '//' (line 116)
    result_floordiv_127223 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 23), '//', bit_depth_127221, int_127222)
    
    # Assigning a type to the variable 'bytes_per_sample' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'bytes_per_sample', result_floordiv_127223)
    
    
    # Getting the type of 'bit_depth' (line 117)
    bit_depth_127224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 7), 'bit_depth')
    int_127225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'int')
    # Applying the binary operator '==' (line 117)
    result_eq_127226 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 7), '==', bit_depth_127224, int_127225)
    
    # Testing the type of an if condition (line 117)
    if_condition_127227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_eq_127226)
    # Assigning a type to the variable 'if_condition_127227' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'if_condition_127227', if_condition_127227)
    # SSA begins for if statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 118):
    
    # Assigning a Str to a Name (line 118):
    str_127228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'str', 'u1')
    # Assigning a type to the variable 'dtype' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'dtype', str_127228)
    # SSA branch for the else part of an if statement (line 117)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'is_big_endian' (line 120)
    is_big_endian_127229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'is_big_endian')
    # Testing the type of an if condition (line 120)
    if_condition_127230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), is_big_endian_127229)
    # Assigning a type to the variable 'if_condition_127230' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_127230', if_condition_127230)
    # SSA begins for if statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 121):
    
    # Assigning a Str to a Name (line 121):
    str_127231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'str', '>')
    # Assigning a type to the variable 'dtype' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'dtype', str_127231)
    # SSA branch for the else part of an if statement (line 120)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 123):
    
    # Assigning a Str to a Name (line 123):
    str_127232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'str', '<')
    # Assigning a type to the variable 'dtype' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'dtype', str_127232)
    # SSA join for if statement (line 120)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'format_tag' (line 124)
    format_tag_127233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'format_tag')
    # Getting the type of 'WAVE_FORMAT_PCM' (line 124)
    WAVE_FORMAT_PCM_127234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'WAVE_FORMAT_PCM')
    # Applying the binary operator '==' (line 124)
    result_eq_127235 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), '==', format_tag_127233, WAVE_FORMAT_PCM_127234)
    
    # Testing the type of an if condition (line 124)
    if_condition_127236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_eq_127235)
    # Assigning a type to the variable 'if_condition_127236' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_127236', if_condition_127236)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'dtype' (line 125)
    dtype_127237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'dtype')
    str_127238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 21), 'str', 'i%d')
    # Getting the type of 'bytes_per_sample' (line 125)
    bytes_per_sample_127239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'bytes_per_sample')
    # Applying the binary operator '%' (line 125)
    result_mod_127240 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 21), '%', str_127238, bytes_per_sample_127239)
    
    # Applying the binary operator '+=' (line 125)
    result_iadd_127241 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 12), '+=', dtype_127237, result_mod_127240)
    # Assigning a type to the variable 'dtype' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'dtype', result_iadd_127241)
    
    # SSA branch for the else part of an if statement (line 124)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'dtype' (line 127)
    dtype_127242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'dtype')
    str_127243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 21), 'str', 'f%d')
    # Getting the type of 'bytes_per_sample' (line 127)
    bytes_per_sample_127244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'bytes_per_sample')
    # Applying the binary operator '%' (line 127)
    result_mod_127245 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 21), '%', str_127243, bytes_per_sample_127244)
    
    # Applying the binary operator '+=' (line 127)
    result_iadd_127246 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 12), '+=', dtype_127242, result_mod_127245)
    # Assigning a type to the variable 'dtype' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'dtype', result_iadd_127246)
    
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 117)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mmap' (line 128)
    mmap_127247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'mmap')
    # Applying the 'not' unary operator (line 128)
    result_not__127248 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 7), 'not', mmap_127247)
    
    # Testing the type of an if condition (line 128)
    if_condition_127249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 4), result_not__127248)
    # Assigning a type to the variable 'if_condition_127249' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'if_condition_127249', if_condition_127249)
    # SSA begins for if statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to fromstring(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Call to read(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'size' (line 129)
    size_127254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 41), 'size', False)
    # Processing the call keyword arguments (line 129)
    kwargs_127255 = {}
    # Getting the type of 'fid' (line 129)
    fid_127252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'fid', False)
    # Obtaining the member 'read' of a type (line 129)
    read_127253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 32), fid_127252, 'read')
    # Calling read(args, kwargs) (line 129)
    read_call_result_127256 = invoke(stypy.reporting.localization.Localization(__file__, 129, 32), read_127253, *[size_127254], **kwargs_127255)
    
    # Processing the call keyword arguments (line 129)
    # Getting the type of 'dtype' (line 129)
    dtype_127257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 54), 'dtype', False)
    keyword_127258 = dtype_127257
    kwargs_127259 = {'dtype': keyword_127258}
    # Getting the type of 'numpy' (line 129)
    numpy_127250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'numpy', False)
    # Obtaining the member 'fromstring' of a type (line 129)
    fromstring_127251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 15), numpy_127250, 'fromstring')
    # Calling fromstring(args, kwargs) (line 129)
    fromstring_call_result_127260 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), fromstring_127251, *[read_call_result_127256], **kwargs_127259)
    
    # Assigning a type to the variable 'data' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'data', fromstring_call_result_127260)
    # SSA branch for the else part of an if statement (line 128)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to tell(...): (line 131)
    # Processing the call keyword arguments (line 131)
    kwargs_127263 = {}
    # Getting the type of 'fid' (line 131)
    fid_127261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'fid', False)
    # Obtaining the member 'tell' of a type (line 131)
    tell_127262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), fid_127261, 'tell')
    # Calling tell(args, kwargs) (line 131)
    tell_call_result_127264 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), tell_127262, *[], **kwargs_127263)
    
    # Assigning a type to the variable 'start' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'start', tell_call_result_127264)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to memmap(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'fid' (line 132)
    fid_127267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'fid', False)
    # Processing the call keyword arguments (line 132)
    # Getting the type of 'dtype' (line 132)
    dtype_127268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'dtype', False)
    keyword_127269 = dtype_127268
    str_127270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 51), 'str', 'c')
    keyword_127271 = str_127270
    # Getting the type of 'start' (line 132)
    start_127272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 63), 'start', False)
    keyword_127273 = start_127272
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_127274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    # Getting the type of 'size' (line 133)
    size_127275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 35), 'size', False)
    # Getting the type of 'bytes_per_sample' (line 133)
    bytes_per_sample_127276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'bytes_per_sample', False)
    # Applying the binary operator '//' (line 133)
    result_floordiv_127277 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 35), '//', size_127275, bytes_per_sample_127276)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 35), tuple_127274, result_floordiv_127277)
    
    keyword_127278 = tuple_127274
    kwargs_127279 = {'dtype': keyword_127269, 'shape': keyword_127278, 'mode': keyword_127271, 'offset': keyword_127273}
    # Getting the type of 'numpy' (line 132)
    numpy_127265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'numpy', False)
    # Obtaining the member 'memmap' of a type (line 132)
    memmap_127266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), numpy_127265, 'memmap')
    # Calling memmap(args, kwargs) (line 132)
    memmap_call_result_127280 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), memmap_127266, *[fid_127267], **kwargs_127279)
    
    # Assigning a type to the variable 'data' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'data', memmap_call_result_127280)
    
    # Call to seek(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'start' (line 134)
    start_127283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'start', False)
    # Getting the type of 'size' (line 134)
    size_127284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'size', False)
    # Applying the binary operator '+' (line 134)
    result_add_127285 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 17), '+', start_127283, size_127284)
    
    # Processing the call keyword arguments (line 134)
    kwargs_127286 = {}
    # Getting the type of 'fid' (line 134)
    fid_127281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'fid', False)
    # Obtaining the member 'seek' of a type (line 134)
    seek_127282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), fid_127281, 'seek')
    # Calling seek(args, kwargs) (line 134)
    seek_call_result_127287 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), seek_127282, *[result_add_127285], **kwargs_127286)
    
    # SSA join for if statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'channels' (line 136)
    channels_127288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 7), 'channels')
    int_127289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 18), 'int')
    # Applying the binary operator '>' (line 136)
    result_gt_127290 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 7), '>', channels_127288, int_127289)
    
    # Testing the type of an if condition (line 136)
    if_condition_127291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), result_gt_127290)
    # Assigning a type to the variable 'if_condition_127291' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_127291', if_condition_127291)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to reshape(...): (line 137)
    # Processing the call arguments (line 137)
    int_127294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 28), 'int')
    # Getting the type of 'channels' (line 137)
    channels_127295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'channels', False)
    # Processing the call keyword arguments (line 137)
    kwargs_127296 = {}
    # Getting the type of 'data' (line 137)
    data_127292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'data', False)
    # Obtaining the member 'reshape' of a type (line 137)
    reshape_127293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), data_127292, 'reshape')
    # Calling reshape(args, kwargs) (line 137)
    reshape_call_result_127297 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), reshape_127293, *[int_127294, channels_127295], **kwargs_127296)
    
    # Assigning a type to the variable 'data' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'data', reshape_call_result_127297)
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'data' (line 138)
    data_127298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'data')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', data_127298)
    
    # ################# End of '_read_data_chunk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_data_chunk' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_127299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_data_chunk'
    return stypy_return_type_127299

# Assigning a type to the variable '_read_data_chunk' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), '_read_data_chunk', _read_data_chunk)

@norecursion
def _skip_unknown_chunk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_skip_unknown_chunk'
    module_type_store = module_type_store.open_function_context('_skip_unknown_chunk', 141, 0, False)
    
    # Passed parameters checking function
    _skip_unknown_chunk.stypy_localization = localization
    _skip_unknown_chunk.stypy_type_of_self = None
    _skip_unknown_chunk.stypy_type_store = module_type_store
    _skip_unknown_chunk.stypy_function_name = '_skip_unknown_chunk'
    _skip_unknown_chunk.stypy_param_names_list = ['fid', 'is_big_endian']
    _skip_unknown_chunk.stypy_varargs_param_name = None
    _skip_unknown_chunk.stypy_kwargs_param_name = None
    _skip_unknown_chunk.stypy_call_defaults = defaults
    _skip_unknown_chunk.stypy_call_varargs = varargs
    _skip_unknown_chunk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_skip_unknown_chunk', ['fid', 'is_big_endian'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_skip_unknown_chunk', localization, ['fid', 'is_big_endian'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_skip_unknown_chunk(...)' code ##################

    
    # Getting the type of 'is_big_endian' (line 142)
    is_big_endian_127300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'is_big_endian')
    # Testing the type of an if condition (line 142)
    if_condition_127301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 4), is_big_endian_127300)
    # Assigning a type to the variable 'if_condition_127301' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'if_condition_127301', if_condition_127301)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 143):
    
    # Assigning a Str to a Name (line 143):
    str_127302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 14), 'str', '>I')
    # Assigning a type to the variable 'fmt' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'fmt', str_127302)
    # SSA branch for the else part of an if statement (line 142)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 145):
    
    # Assigning a Str to a Name (line 145):
    str_127303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 14), 'str', '<I')
    # Assigning a type to the variable 'fmt' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'fmt', str_127303)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to read(...): (line 147)
    # Processing the call arguments (line 147)
    int_127306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'int')
    # Processing the call keyword arguments (line 147)
    kwargs_127307 = {}
    # Getting the type of 'fid' (line 147)
    fid_127304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'fid', False)
    # Obtaining the member 'read' of a type (line 147)
    read_127305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 11), fid_127304, 'read')
    # Calling read(args, kwargs) (line 147)
    read_call_result_127308 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), read_127305, *[int_127306], **kwargs_127307)
    
    # Assigning a type to the variable 'data' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'data', read_call_result_127308)
    
    # Getting the type of 'data' (line 152)
    data_127309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 7), 'data')
    # Testing the type of an if condition (line 152)
    if_condition_127310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 4), data_127309)
    # Assigning a type to the variable 'if_condition_127310' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'if_condition_127310', if_condition_127310)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 153):
    
    # Assigning a Subscript to a Name (line 153):
    
    # Obtaining the type of the subscript
    int_127311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 40), 'int')
    
    # Call to unpack(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'fmt' (line 153)
    fmt_127314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'fmt', False)
    # Getting the type of 'data' (line 153)
    data_127315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'data', False)
    # Processing the call keyword arguments (line 153)
    kwargs_127316 = {}
    # Getting the type of 'struct' (line 153)
    struct_127312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 153)
    unpack_127313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), struct_127312, 'unpack')
    # Calling unpack(args, kwargs) (line 153)
    unpack_call_result_127317 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), unpack_127313, *[fmt_127314, data_127315], **kwargs_127316)
    
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___127318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), unpack_call_result_127317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_127319 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), getitem___127318, int_127311)
    
    # Assigning a type to the variable 'size' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'size', subscript_call_result_127319)
    
    # Call to seek(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'size' (line 154)
    size_127322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'size', False)
    int_127323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'int')
    # Processing the call keyword arguments (line 154)
    kwargs_127324 = {}
    # Getting the type of 'fid' (line 154)
    fid_127320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'fid', False)
    # Obtaining the member 'seek' of a type (line 154)
    seek_127321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), fid_127320, 'seek')
    # Calling seek(args, kwargs) (line 154)
    seek_call_result_127325 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), seek_127321, *[size_127322, int_127323], **kwargs_127324)
    
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_skip_unknown_chunk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_skip_unknown_chunk' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_127326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_skip_unknown_chunk'
    return stypy_return_type_127326

# Assigning a type to the variable '_skip_unknown_chunk' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), '_skip_unknown_chunk', _skip_unknown_chunk)

@norecursion
def _read_riff_chunk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_read_riff_chunk'
    module_type_store = module_type_store.open_function_context('_read_riff_chunk', 157, 0, False)
    
    # Passed parameters checking function
    _read_riff_chunk.stypy_localization = localization
    _read_riff_chunk.stypy_type_of_self = None
    _read_riff_chunk.stypy_type_store = module_type_store
    _read_riff_chunk.stypy_function_name = '_read_riff_chunk'
    _read_riff_chunk.stypy_param_names_list = ['fid']
    _read_riff_chunk.stypy_varargs_param_name = None
    _read_riff_chunk.stypy_kwargs_param_name = None
    _read_riff_chunk.stypy_call_defaults = defaults
    _read_riff_chunk.stypy_call_varargs = varargs
    _read_riff_chunk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_riff_chunk', ['fid'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_riff_chunk', localization, ['fid'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_riff_chunk(...)' code ##################

    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to read(...): (line 158)
    # Processing the call arguments (line 158)
    int_127329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'int')
    # Processing the call keyword arguments (line 158)
    kwargs_127330 = {}
    # Getting the type of 'fid' (line 158)
    fid_127327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'fid', False)
    # Obtaining the member 'read' of a type (line 158)
    read_127328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), fid_127327, 'read')
    # Calling read(args, kwargs) (line 158)
    read_call_result_127331 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), read_127328, *[int_127329], **kwargs_127330)
    
    # Assigning a type to the variable 'str1' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'str1', read_call_result_127331)
    
    
    # Getting the type of 'str1' (line 159)
    str1_127332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'str1')
    str_127333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 15), 'str', 'RIFF')
    # Applying the binary operator '==' (line 159)
    result_eq_127334 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), '==', str1_127332, str_127333)
    
    # Testing the type of an if condition (line 159)
    if_condition_127335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), result_eq_127334)
    # Assigning a type to the variable 'if_condition_127335' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_127335', if_condition_127335)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 160):
    
    # Assigning a Name to a Name (line 160):
    # Getting the type of 'False' (line 160)
    False_127336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'False')
    # Assigning a type to the variable 'is_big_endian' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'is_big_endian', False_127336)
    
    # Assigning a Str to a Name (line 161):
    
    # Assigning a Str to a Name (line 161):
    str_127337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 14), 'str', '<I')
    # Assigning a type to the variable 'fmt' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'fmt', str_127337)
    # SSA branch for the else part of an if statement (line 159)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'str1' (line 162)
    str1_127338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 9), 'str1')
    str_127339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 17), 'str', 'RIFX')
    # Applying the binary operator '==' (line 162)
    result_eq_127340 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 9), '==', str1_127338, str_127339)
    
    # Testing the type of an if condition (line 162)
    if_condition_127341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 9), result_eq_127340)
    # Assigning a type to the variable 'if_condition_127341' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 9), 'if_condition_127341', if_condition_127341)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 163):
    
    # Assigning a Name to a Name (line 163):
    # Getting the type of 'True' (line 163)
    True_127342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'True')
    # Assigning a type to the variable 'is_big_endian' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'is_big_endian', True_127342)
    
    # Assigning a Str to a Name (line 164):
    
    # Assigning a Str to a Name (line 164):
    str_127343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 14), 'str', '>I')
    # Assigning a type to the variable 'fmt' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'fmt', str_127343)
    # SSA branch for the else part of an if statement (line 162)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Call to format(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Call to repr(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'str1' (line 168)
    str1_127348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 51), 'str1', False)
    # Processing the call keyword arguments (line 168)
    kwargs_127349 = {}
    # Getting the type of 'repr' (line 168)
    repr_127347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'repr', False)
    # Calling repr(args, kwargs) (line 168)
    repr_call_result_127350 = invoke(stypy.reporting.localization.Localization(__file__, 168, 46), repr_127347, *[str1_127348], **kwargs_127349)
    
    # Processing the call keyword arguments (line 167)
    kwargs_127351 = {}
    str_127345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'str', 'File format {}... not understood.')
    # Obtaining the member 'format' of a type (line 167)
    format_127346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 25), str_127345, 'format')
    # Calling format(args, kwargs) (line 167)
    format_call_result_127352 = invoke(stypy.reporting.localization.Localization(__file__, 167, 25), format_127346, *[repr_call_result_127350], **kwargs_127351)
    
    # Processing the call keyword arguments (line 167)
    kwargs_127353 = {}
    # Getting the type of 'ValueError' (line 167)
    ValueError_127344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 167)
    ValueError_call_result_127354 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), ValueError_127344, *[format_call_result_127352], **kwargs_127353)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 167, 8), ValueError_call_result_127354, 'raise parameter', BaseException)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 171):
    
    # Assigning a BinOp to a Name (line 171):
    
    # Obtaining the type of the subscript
    int_127355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 48), 'int')
    
    # Call to unpack(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'fmt' (line 171)
    fmt_127358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'fmt', False)
    
    # Call to read(...): (line 171)
    # Processing the call arguments (line 171)
    int_127361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 44), 'int')
    # Processing the call keyword arguments (line 171)
    kwargs_127362 = {}
    # Getting the type of 'fid' (line 171)
    fid_127359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'fid', False)
    # Obtaining the member 'read' of a type (line 171)
    read_127360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 35), fid_127359, 'read')
    # Calling read(args, kwargs) (line 171)
    read_call_result_127363 = invoke(stypy.reporting.localization.Localization(__file__, 171, 35), read_127360, *[int_127361], **kwargs_127362)
    
    # Processing the call keyword arguments (line 171)
    kwargs_127364 = {}
    # Getting the type of 'struct' (line 171)
    struct_127356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'struct', False)
    # Obtaining the member 'unpack' of a type (line 171)
    unpack_127357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), struct_127356, 'unpack')
    # Calling unpack(args, kwargs) (line 171)
    unpack_call_result_127365 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), unpack_127357, *[fmt_127358, read_call_result_127363], **kwargs_127364)
    
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___127366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), unpack_call_result_127365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_127367 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), getitem___127366, int_127355)
    
    int_127368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 53), 'int')
    # Applying the binary operator '+' (line 171)
    result_add_127369 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 16), '+', subscript_call_result_127367, int_127368)
    
    # Assigning a type to the variable 'file_size' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'file_size', result_add_127369)
    
    # Assigning a Call to a Name (line 173):
    
    # Assigning a Call to a Name (line 173):
    
    # Call to read(...): (line 173)
    # Processing the call arguments (line 173)
    int_127372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 20), 'int')
    # Processing the call keyword arguments (line 173)
    kwargs_127373 = {}
    # Getting the type of 'fid' (line 173)
    fid_127370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'fid', False)
    # Obtaining the member 'read' of a type (line 173)
    read_127371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 11), fid_127370, 'read')
    # Calling read(args, kwargs) (line 173)
    read_call_result_127374 = invoke(stypy.reporting.localization.Localization(__file__, 173, 11), read_127371, *[int_127372], **kwargs_127373)
    
    # Assigning a type to the variable 'str2' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'str2', read_call_result_127374)
    
    
    # Getting the type of 'str2' (line 174)
    str2_127375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'str2')
    str_127376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'str', 'WAVE')
    # Applying the binary operator '!=' (line 174)
    result_ne_127377 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), '!=', str2_127375, str_127376)
    
    # Testing the type of an if condition (line 174)
    if_condition_127378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_ne_127377)
    # Assigning a type to the variable 'if_condition_127378' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_127378', if_condition_127378)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 175)
    # Processing the call arguments (line 175)
    str_127380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 25), 'str', 'Not a WAV file.')
    # Processing the call keyword arguments (line 175)
    kwargs_127381 = {}
    # Getting the type of 'ValueError' (line 175)
    ValueError_127379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 175)
    ValueError_call_result_127382 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), ValueError_127379, *[str_127380], **kwargs_127381)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 8), ValueError_call_result_127382, 'raise parameter', BaseException)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 177)
    tuple_127383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 177)
    # Adding element type (line 177)
    # Getting the type of 'file_size' (line 177)
    file_size_127384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'file_size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 11), tuple_127383, file_size_127384)
    # Adding element type (line 177)
    # Getting the type of 'is_big_endian' (line 177)
    is_big_endian_127385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 22), 'is_big_endian')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 11), tuple_127383, is_big_endian_127385)
    
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', tuple_127383)
    
    # ################# End of '_read_riff_chunk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_riff_chunk' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_127386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127386)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_riff_chunk'
    return stypy_return_type_127386

# Assigning a type to the variable '_read_riff_chunk' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), '_read_riff_chunk', _read_riff_chunk)

@norecursion
def read(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 180)
    False_127387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'False')
    defaults = [False_127387]
    # Create a new context for function 'read'
    module_type_store = module_type_store.open_function_context('read', 180, 0, False)
    
    # Passed parameters checking function
    read.stypy_localization = localization
    read.stypy_type_of_self = None
    read.stypy_type_store = module_type_store
    read.stypy_function_name = 'read'
    read.stypy_param_names_list = ['filename', 'mmap']
    read.stypy_varargs_param_name = None
    read.stypy_kwargs_param_name = None
    read.stypy_call_defaults = defaults
    read.stypy_call_varargs = varargs
    read.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read', ['filename', 'mmap'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read', localization, ['filename', 'mmap'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read(...)' code ##################

    str_127388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, (-1)), 'str', '\n    Open a WAV file\n\n    Return the sample rate (in samples/sec) and data from a WAV file.\n\n    Parameters\n    ----------\n    filename : string or open file handle\n        Input wav file.\n    mmap : bool, optional\n        Whether to read data as memory-mapped.\n        Only to be used on real files (Default: False).\n\n        .. versionadded:: 0.12.0\n\n    Returns\n    -------\n    rate : int\n        Sample rate of wav file.\n    data : numpy array\n        Data read from wav file.  Data-type is determined from the file;\n        see Notes.\n\n    Notes\n    -----\n    This function cannot read wav files with 24-bit data.\n\n    Common data types: [1]_\n\n    =====================  ===========  ===========  =============\n         WAV format            Min          Max       NumPy dtype\n    =====================  ===========  ===========  =============\n    32-bit floating-point  -1.0         +1.0         float32\n    32-bit PCM             -2147483648  +2147483647  int32\n    16-bit PCM             -32768       +32767       int16\n    8-bit PCM              0            255          uint8\n    =====================  ===========  ===========  =============\n\n    Note that 8-bit PCM is unsigned.\n\n    References\n    ----------\n    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming\n       Interface and Data Specifications 1.0", section "Data Format of the\n       Samples", August 1991\n       http://www.tactilemedia.com/info/MCI_Control_Info.html\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 229)
    str_127389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 25), 'str', 'read')
    # Getting the type of 'filename' (line 229)
    filename_127390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'filename')
    
    (may_be_127391, more_types_in_union_127392) = may_provide_member(str_127389, filename_127390)

    if may_be_127391:

        if more_types_in_union_127392:
            # Runtime conditional SSA (line 229)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'filename' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'filename', remove_not_member_provider_from_union(filename_127390, 'read'))
        
        # Assigning a Name to a Name (line 230):
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'filename' (line 230)
        filename_127393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 14), 'filename')
        # Assigning a type to the variable 'fid' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'fid', filename_127393)
        
        # Assigning a Name to a Name (line 231):
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'False' (line 231)
        False_127394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'False')
        # Assigning a type to the variable 'mmap' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'mmap', False_127394)

        if more_types_in_union_127392:
            # Runtime conditional SSA for else branch (line 229)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_127391) or more_types_in_union_127392):
        # Assigning a type to the variable 'filename' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'filename', remove_member_provider_from_union(filename_127390, 'read'))
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to open(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'filename' (line 233)
        filename_127396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'filename', False)
        str_127397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'str', 'rb')
        # Processing the call keyword arguments (line 233)
        kwargs_127398 = {}
        # Getting the type of 'open' (line 233)
        open_127395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 14), 'open', False)
        # Calling open(args, kwargs) (line 233)
        open_call_result_127399 = invoke(stypy.reporting.localization.Localization(__file__, 233, 14), open_127395, *[filename_127396, str_127397], **kwargs_127398)
        
        # Assigning a type to the variable 'fid' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'fid', open_call_result_127399)

        if (may_be_127391 and more_types_in_union_127392):
            # SSA join for if statement (line 229)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Try-finally block (line 235)
    
    # Assigning a Call to a Tuple (line 236):
    
    # Assigning a Subscript to a Name (line 236):
    
    # Obtaining the type of the subscript
    int_127400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 8), 'int')
    
    # Call to _read_riff_chunk(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'fid' (line 236)
    fid_127402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 52), 'fid', False)
    # Processing the call keyword arguments (line 236)
    kwargs_127403 = {}
    # Getting the type of '_read_riff_chunk' (line 236)
    _read_riff_chunk_127401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 35), '_read_riff_chunk', False)
    # Calling _read_riff_chunk(args, kwargs) (line 236)
    _read_riff_chunk_call_result_127404 = invoke(stypy.reporting.localization.Localization(__file__, 236, 35), _read_riff_chunk_127401, *[fid_127402], **kwargs_127403)
    
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___127405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), _read_riff_chunk_call_result_127404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_127406 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), getitem___127405, int_127400)
    
    # Assigning a type to the variable 'tuple_var_assignment_127000' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'tuple_var_assignment_127000', subscript_call_result_127406)
    
    # Assigning a Subscript to a Name (line 236):
    
    # Obtaining the type of the subscript
    int_127407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 8), 'int')
    
    # Call to _read_riff_chunk(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'fid' (line 236)
    fid_127409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 52), 'fid', False)
    # Processing the call keyword arguments (line 236)
    kwargs_127410 = {}
    # Getting the type of '_read_riff_chunk' (line 236)
    _read_riff_chunk_127408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 35), '_read_riff_chunk', False)
    # Calling _read_riff_chunk(args, kwargs) (line 236)
    _read_riff_chunk_call_result_127411 = invoke(stypy.reporting.localization.Localization(__file__, 236, 35), _read_riff_chunk_127408, *[fid_127409], **kwargs_127410)
    
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___127412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), _read_riff_chunk_call_result_127411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_127413 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), getitem___127412, int_127407)
    
    # Assigning a type to the variable 'tuple_var_assignment_127001' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'tuple_var_assignment_127001', subscript_call_result_127413)
    
    # Assigning a Name to a Name (line 236):
    # Getting the type of 'tuple_var_assignment_127000' (line 236)
    tuple_var_assignment_127000_127414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'tuple_var_assignment_127000')
    # Assigning a type to the variable 'file_size' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'file_size', tuple_var_assignment_127000_127414)
    
    # Assigning a Name to a Name (line 236):
    # Getting the type of 'tuple_var_assignment_127001' (line 236)
    tuple_var_assignment_127001_127415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'tuple_var_assignment_127001')
    # Assigning a type to the variable 'is_big_endian' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'is_big_endian', tuple_var_assignment_127001_127415)
    
    # Assigning a Name to a Name (line 237):
    
    # Assigning a Name to a Name (line 237):
    # Getting the type of 'False' (line 237)
    False_127416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 29), 'False')
    # Assigning a type to the variable 'fmt_chunk_received' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'fmt_chunk_received', False_127416)
    
    # Assigning a Num to a Name (line 238):
    
    # Assigning a Num to a Name (line 238):
    int_127417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 19), 'int')
    # Assigning a type to the variable 'channels' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'channels', int_127417)
    
    # Assigning a Num to a Name (line 239):
    
    # Assigning a Num to a Name (line 239):
    int_127418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 20), 'int')
    # Assigning a type to the variable 'bit_depth' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'bit_depth', int_127418)
    
    # Assigning a Name to a Name (line 240):
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'WAVE_FORMAT_PCM' (line 240)
    WAVE_FORMAT_PCM_127419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'WAVE_FORMAT_PCM')
    # Assigning a type to the variable 'format_tag' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'format_tag', WAVE_FORMAT_PCM_127419)
    
    
    
    # Call to tell(...): (line 241)
    # Processing the call keyword arguments (line 241)
    kwargs_127422 = {}
    # Getting the type of 'fid' (line 241)
    fid_127420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 14), 'fid', False)
    # Obtaining the member 'tell' of a type (line 241)
    tell_127421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 14), fid_127420, 'tell')
    # Calling tell(args, kwargs) (line 241)
    tell_call_result_127423 = invoke(stypy.reporting.localization.Localization(__file__, 241, 14), tell_127421, *[], **kwargs_127422)
    
    # Getting the type of 'file_size' (line 241)
    file_size_127424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 27), 'file_size')
    # Applying the binary operator '<' (line 241)
    result_lt_127425 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 14), '<', tell_call_result_127423, file_size_127424)
    
    # Testing the type of an if condition (line 241)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), result_lt_127425)
    # SSA begins for while statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 243):
    
    # Assigning a Call to a Name (line 243):
    
    # Call to read(...): (line 243)
    # Processing the call arguments (line 243)
    int_127428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 32), 'int')
    # Processing the call keyword arguments (line 243)
    kwargs_127429 = {}
    # Getting the type of 'fid' (line 243)
    fid_127426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 23), 'fid', False)
    # Obtaining the member 'read' of a type (line 243)
    read_127427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 23), fid_127426, 'read')
    # Calling read(args, kwargs) (line 243)
    read_call_result_127430 = invoke(stypy.reporting.localization.Localization(__file__, 243, 23), read_127427, *[int_127428], **kwargs_127429)
    
    # Assigning a type to the variable 'chunk_id' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'chunk_id', read_call_result_127430)
    
    
    # Getting the type of 'chunk_id' (line 245)
    chunk_id_127431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'chunk_id')
    # Applying the 'not' unary operator (line 245)
    result_not__127432 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 15), 'not', chunk_id_127431)
    
    # Testing the type of an if condition (line 245)
    if_condition_127433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 12), result_not__127432)
    # Assigning a type to the variable 'if_condition_127433' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'if_condition_127433', if_condition_127433)
    # SSA begins for if statement (line 245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 246)
    # Processing the call arguments (line 246)
    str_127435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 33), 'str', 'Unexpected end of file.')
    # Processing the call keyword arguments (line 246)
    kwargs_127436 = {}
    # Getting the type of 'ValueError' (line 246)
    ValueError_127434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 246)
    ValueError_call_result_127437 = invoke(stypy.reporting.localization.Localization(__file__, 246, 22), ValueError_127434, *[str_127435], **kwargs_127436)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 246, 16), ValueError_call_result_127437, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 245)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'chunk_id' (line 247)
    chunk_id_127439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 21), 'chunk_id', False)
    # Processing the call keyword arguments (line 247)
    kwargs_127440 = {}
    # Getting the type of 'len' (line 247)
    len_127438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'len', False)
    # Calling len(args, kwargs) (line 247)
    len_call_result_127441 = invoke(stypy.reporting.localization.Localization(__file__, 247, 17), len_127438, *[chunk_id_127439], **kwargs_127440)
    
    int_127442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 33), 'int')
    # Applying the binary operator '<' (line 247)
    result_lt_127443 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 17), '<', len_call_result_127441, int_127442)
    
    # Testing the type of an if condition (line 247)
    if_condition_127444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 17), result_lt_127443)
    # Assigning a type to the variable 'if_condition_127444' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'if_condition_127444', if_condition_127444)
    # SSA begins for if statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 248)
    # Processing the call arguments (line 248)
    str_127446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 33), 'str', 'Incomplete wav chunk.')
    # Processing the call keyword arguments (line 248)
    kwargs_127447 = {}
    # Getting the type of 'ValueError' (line 248)
    ValueError_127445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 248)
    ValueError_call_result_127448 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), ValueError_127445, *[str_127446], **kwargs_127447)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 248, 16), ValueError_call_result_127448, 'raise parameter', BaseException)
    # SSA join for if statement (line 247)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 245)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'chunk_id' (line 250)
    chunk_id_127449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'chunk_id')
    str_127450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 27), 'str', 'fmt ')
    # Applying the binary operator '==' (line 250)
    result_eq_127451 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 15), '==', chunk_id_127449, str_127450)
    
    # Testing the type of an if condition (line 250)
    if_condition_127452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 12), result_eq_127451)
    # Assigning a type to the variable 'if_condition_127452' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'if_condition_127452', if_condition_127452)
    # SSA begins for if statement (line 250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 251):
    
    # Assigning a Name to a Name (line 251):
    # Getting the type of 'True' (line 251)
    True_127453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 37), 'True')
    # Assigning a type to the variable 'fmt_chunk_received' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'fmt_chunk_received', True_127453)
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to _read_fmt_chunk(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'fid' (line 252)
    fid_127455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 44), 'fid', False)
    # Getting the type of 'is_big_endian' (line 252)
    is_big_endian_127456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 49), 'is_big_endian', False)
    # Processing the call keyword arguments (line 252)
    kwargs_127457 = {}
    # Getting the type of '_read_fmt_chunk' (line 252)
    _read_fmt_chunk_127454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), '_read_fmt_chunk', False)
    # Calling _read_fmt_chunk(args, kwargs) (line 252)
    _read_fmt_chunk_call_result_127458 = invoke(stypy.reporting.localization.Localization(__file__, 252, 28), _read_fmt_chunk_127454, *[fid_127455, is_big_endian_127456], **kwargs_127457)
    
    # Assigning a type to the variable 'fmt_chunk' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'fmt_chunk', _read_fmt_chunk_call_result_127458)
    
    # Assigning a Subscript to a Tuple (line 253):
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_127459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 16), 'int')
    
    # Obtaining the type of the subscript
    int_127460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 53), 'int')
    int_127461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 55), 'int')
    slice_127462 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 253, 43), int_127460, int_127461, None)
    # Getting the type of 'fmt_chunk' (line 253)
    fmt_chunk_127463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 'fmt_chunk')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___127464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 43), fmt_chunk_127463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_127465 = invoke(stypy.reporting.localization.Localization(__file__, 253, 43), getitem___127464, slice_127462)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___127466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 16), subscript_call_result_127465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_127467 = invoke(stypy.reporting.localization.Localization(__file__, 253, 16), getitem___127466, int_127459)
    
    # Assigning a type to the variable 'tuple_var_assignment_127002' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'tuple_var_assignment_127002', subscript_call_result_127467)
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_127468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 16), 'int')
    
    # Obtaining the type of the subscript
    int_127469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 53), 'int')
    int_127470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 55), 'int')
    slice_127471 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 253, 43), int_127469, int_127470, None)
    # Getting the type of 'fmt_chunk' (line 253)
    fmt_chunk_127472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 'fmt_chunk')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___127473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 43), fmt_chunk_127472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_127474 = invoke(stypy.reporting.localization.Localization(__file__, 253, 43), getitem___127473, slice_127471)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___127475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 16), subscript_call_result_127474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_127476 = invoke(stypy.reporting.localization.Localization(__file__, 253, 16), getitem___127475, int_127468)
    
    # Assigning a type to the variable 'tuple_var_assignment_127003' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'tuple_var_assignment_127003', subscript_call_result_127476)
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_127477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 16), 'int')
    
    # Obtaining the type of the subscript
    int_127478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 53), 'int')
    int_127479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 55), 'int')
    slice_127480 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 253, 43), int_127478, int_127479, None)
    # Getting the type of 'fmt_chunk' (line 253)
    fmt_chunk_127481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 'fmt_chunk')
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___127482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 43), fmt_chunk_127481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_127483 = invoke(stypy.reporting.localization.Localization(__file__, 253, 43), getitem___127482, slice_127480)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___127484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 16), subscript_call_result_127483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_127485 = invoke(stypy.reporting.localization.Localization(__file__, 253, 16), getitem___127484, int_127477)
    
    # Assigning a type to the variable 'tuple_var_assignment_127004' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'tuple_var_assignment_127004', subscript_call_result_127485)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_127002' (line 253)
    tuple_var_assignment_127002_127486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'tuple_var_assignment_127002')
    # Assigning a type to the variable 'format_tag' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'format_tag', tuple_var_assignment_127002_127486)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_127003' (line 253)
    tuple_var_assignment_127003_127487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'tuple_var_assignment_127003')
    # Assigning a type to the variable 'channels' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'channels', tuple_var_assignment_127003_127487)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_127004' (line 253)
    tuple_var_assignment_127004_127488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'tuple_var_assignment_127004')
    # Assigning a type to the variable 'fs' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 38), 'fs', tuple_var_assignment_127004_127488)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_127489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'int')
    # Getting the type of 'fmt_chunk' (line 254)
    fmt_chunk_127490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'fmt_chunk')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___127491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 28), fmt_chunk_127490, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_127492 = invoke(stypy.reporting.localization.Localization(__file__, 254, 28), getitem___127491, int_127489)
    
    # Assigning a type to the variable 'bit_depth' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'bit_depth', subscript_call_result_127492)
    
    
    # Getting the type of 'bit_depth' (line 255)
    bit_depth_127493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 'bit_depth')
    
    # Obtaining an instance of the builtin type 'tuple' (line 255)
    tuple_127494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 255)
    # Adding element type (line 255)
    int_127495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), tuple_127494, int_127495)
    # Adding element type (line 255)
    int_127496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), tuple_127494, int_127496)
    # Adding element type (line 255)
    int_127497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), tuple_127494, int_127497)
    # Adding element type (line 255)
    int_127498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), tuple_127494, int_127498)
    # Adding element type (line 255)
    int_127499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), tuple_127494, int_127499)
    # Adding element type (line 255)
    int_127500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 37), tuple_127494, int_127500)
    
    # Applying the binary operator 'notin' (line 255)
    result_contains_127501 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 19), 'notin', bit_depth_127493, tuple_127494)
    
    # Testing the type of an if condition (line 255)
    if_condition_127502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 16), result_contains_127501)
    # Assigning a type to the variable 'if_condition_127502' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'if_condition_127502', if_condition_127502)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 256)
    # Processing the call arguments (line 256)
    
    # Call to format(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'bit_depth' (line 257)
    bit_depth_127506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 63), 'bit_depth', False)
    # Processing the call keyword arguments (line 256)
    kwargs_127507 = {}
    str_127504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 37), 'str', 'Unsupported bit depth: the wav file has {}-bit data.')
    # Obtaining the member 'format' of a type (line 256)
    format_127505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 37), str_127504, 'format')
    # Calling format(args, kwargs) (line 256)
    format_call_result_127508 = invoke(stypy.reporting.localization.Localization(__file__, 256, 37), format_127505, *[bit_depth_127506], **kwargs_127507)
    
    # Processing the call keyword arguments (line 256)
    kwargs_127509 = {}
    # Getting the type of 'ValueError' (line 256)
    ValueError_127503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 256)
    ValueError_call_result_127510 = invoke(stypy.reporting.localization.Localization(__file__, 256, 26), ValueError_127503, *[format_call_result_127508], **kwargs_127509)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 20), ValueError_call_result_127510, 'raise parameter', BaseException)
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 250)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'chunk_id' (line 258)
    chunk_id_127511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'chunk_id')
    str_127512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'str', 'fact')
    # Applying the binary operator '==' (line 258)
    result_eq_127513 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 17), '==', chunk_id_127511, str_127512)
    
    # Testing the type of an if condition (line 258)
    if_condition_127514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 17), result_eq_127513)
    # Assigning a type to the variable 'if_condition_127514' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'if_condition_127514', if_condition_127514)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _skip_unknown_chunk(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'fid' (line 259)
    fid_127516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 36), 'fid', False)
    # Getting the type of 'is_big_endian' (line 259)
    is_big_endian_127517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 41), 'is_big_endian', False)
    # Processing the call keyword arguments (line 259)
    kwargs_127518 = {}
    # Getting the type of '_skip_unknown_chunk' (line 259)
    _skip_unknown_chunk_127515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), '_skip_unknown_chunk', False)
    # Calling _skip_unknown_chunk(args, kwargs) (line 259)
    _skip_unknown_chunk_call_result_127519 = invoke(stypy.reporting.localization.Localization(__file__, 259, 16), _skip_unknown_chunk_127515, *[fid_127516, is_big_endian_127517], **kwargs_127518)
    
    # SSA branch for the else part of an if statement (line 258)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'chunk_id' (line 260)
    chunk_id_127520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'chunk_id')
    str_127521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'str', 'data')
    # Applying the binary operator '==' (line 260)
    result_eq_127522 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 17), '==', chunk_id_127520, str_127521)
    
    # Testing the type of an if condition (line 260)
    if_condition_127523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 17), result_eq_127522)
    # Assigning a type to the variable 'if_condition_127523' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'if_condition_127523', if_condition_127523)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'fmt_chunk_received' (line 261)
    fmt_chunk_received_127524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'fmt_chunk_received')
    # Applying the 'not' unary operator (line 261)
    result_not__127525 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 19), 'not', fmt_chunk_received_127524)
    
    # Testing the type of an if condition (line 261)
    if_condition_127526 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 16), result_not__127525)
    # Assigning a type to the variable 'if_condition_127526' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'if_condition_127526', if_condition_127526)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 262)
    # Processing the call arguments (line 262)
    str_127528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 37), 'str', 'No fmt chunk before data')
    # Processing the call keyword arguments (line 262)
    kwargs_127529 = {}
    # Getting the type of 'ValueError' (line 262)
    ValueError_127527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 262)
    ValueError_call_result_127530 = invoke(stypy.reporting.localization.Localization(__file__, 262, 26), ValueError_127527, *[str_127528], **kwargs_127529)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 262, 20), ValueError_call_result_127530, 'raise parameter', BaseException)
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 263):
    
    # Assigning a Call to a Name (line 263):
    
    # Call to _read_data_chunk(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'fid' (line 263)
    fid_127532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 40), 'fid', False)
    # Getting the type of 'format_tag' (line 263)
    format_tag_127533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'format_tag', False)
    # Getting the type of 'channels' (line 263)
    channels_127534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 57), 'channels', False)
    # Getting the type of 'bit_depth' (line 263)
    bit_depth_127535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 67), 'bit_depth', False)
    # Getting the type of 'is_big_endian' (line 264)
    is_big_endian_127536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 40), 'is_big_endian', False)
    # Getting the type of 'mmap' (line 264)
    mmap_127537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 55), 'mmap', False)
    # Processing the call keyword arguments (line 263)
    kwargs_127538 = {}
    # Getting the type of '_read_data_chunk' (line 263)
    _read_data_chunk_127531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), '_read_data_chunk', False)
    # Calling _read_data_chunk(args, kwargs) (line 263)
    _read_data_chunk_call_result_127539 = invoke(stypy.reporting.localization.Localization(__file__, 263, 23), _read_data_chunk_127531, *[fid_127532, format_tag_127533, channels_127534, bit_depth_127535, is_big_endian_127536, mmap_127537], **kwargs_127538)
    
    # Assigning a type to the variable 'data' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'data', _read_data_chunk_call_result_127539)
    # SSA branch for the else part of an if statement (line 260)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'chunk_id' (line 265)
    chunk_id_127540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'chunk_id')
    str_127541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 29), 'str', 'LIST')
    # Applying the binary operator '==' (line 265)
    result_eq_127542 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 17), '==', chunk_id_127540, str_127541)
    
    # Testing the type of an if condition (line 265)
    if_condition_127543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 17), result_eq_127542)
    # Assigning a type to the variable 'if_condition_127543' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'if_condition_127543', if_condition_127543)
    # SSA begins for if statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _skip_unknown_chunk(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'fid' (line 267)
    fid_127545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 36), 'fid', False)
    # Getting the type of 'is_big_endian' (line 267)
    is_big_endian_127546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 41), 'is_big_endian', False)
    # Processing the call keyword arguments (line 267)
    kwargs_127547 = {}
    # Getting the type of '_skip_unknown_chunk' (line 267)
    _skip_unknown_chunk_127544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), '_skip_unknown_chunk', False)
    # Calling _skip_unknown_chunk(args, kwargs) (line 267)
    _skip_unknown_chunk_call_result_127548 = invoke(stypy.reporting.localization.Localization(__file__, 267, 16), _skip_unknown_chunk_127544, *[fid_127545, is_big_endian_127546], **kwargs_127547)
    
    # SSA branch for the else part of an if statement (line 265)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'chunk_id' (line 268)
    chunk_id_127549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), 'chunk_id')
    
    # Obtaining an instance of the builtin type 'tuple' (line 268)
    tuple_127550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 268)
    # Adding element type (line 268)
    str_127551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 30), 'str', 'JUNK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 30), tuple_127550, str_127551)
    # Adding element type (line 268)
    str_127552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 39), 'str', 'Fake')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 30), tuple_127550, str_127552)
    
    # Applying the binary operator 'in' (line 268)
    result_contains_127553 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 17), 'in', chunk_id_127549, tuple_127550)
    
    # Testing the type of an if condition (line 268)
    if_condition_127554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 17), result_contains_127553)
    # Assigning a type to the variable 'if_condition_127554' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), 'if_condition_127554', if_condition_127554)
    # SSA begins for if statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _skip_unknown_chunk(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'fid' (line 270)
    fid_127556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 36), 'fid', False)
    # Getting the type of 'is_big_endian' (line 270)
    is_big_endian_127557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'is_big_endian', False)
    # Processing the call keyword arguments (line 270)
    kwargs_127558 = {}
    # Getting the type of '_skip_unknown_chunk' (line 270)
    _skip_unknown_chunk_127555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), '_skip_unknown_chunk', False)
    # Calling _skip_unknown_chunk(args, kwargs) (line 270)
    _skip_unknown_chunk_call_result_127559 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), _skip_unknown_chunk_127555, *[fid_127556, is_big_endian_127557], **kwargs_127558)
    
    # SSA branch for the else part of an if statement (line 268)
    module_type_store.open_ssa_branch('else')
    
    # Call to warn(...): (line 272)
    # Processing the call arguments (line 272)
    str_127562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 30), 'str', 'Chunk (non-data) not understood, skipping it.')
    # Getting the type of 'WavFileWarning' (line 273)
    WavFileWarning_127563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'WavFileWarning', False)
    # Processing the call keyword arguments (line 272)
    kwargs_127564 = {}
    # Getting the type of 'warnings' (line 272)
    warnings_127560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 272)
    warn_127561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), warnings_127560, 'warn')
    # Calling warn(args, kwargs) (line 272)
    warn_call_result_127565 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), warn_127561, *[str_127562, WavFileWarning_127563], **kwargs_127564)
    
    
    # Call to _skip_unknown_chunk(...): (line 274)
    # Processing the call arguments (line 274)
    # Getting the type of 'fid' (line 274)
    fid_127567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'fid', False)
    # Getting the type of 'is_big_endian' (line 274)
    is_big_endian_127568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 41), 'is_big_endian', False)
    # Processing the call keyword arguments (line 274)
    kwargs_127569 = {}
    # Getting the type of '_skip_unknown_chunk' (line 274)
    _skip_unknown_chunk_127566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), '_skip_unknown_chunk', False)
    # Calling _skip_unknown_chunk(args, kwargs) (line 274)
    _skip_unknown_chunk_call_result_127570 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), _skip_unknown_chunk_127566, *[fid_127567, is_big_endian_127568], **kwargs_127569)
    
    # SSA join for if statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 265)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 250)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 235)
    
    # Type idiom detected: calculating its left and rigth part (line 276)
    str_127571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 33), 'str', 'read')
    # Getting the type of 'filename' (line 276)
    filename_127572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'filename')
    
    (may_be_127573, more_types_in_union_127574) = may_not_provide_member(str_127571, filename_127572)

    if may_be_127573:

        if more_types_in_union_127574:
            # Runtime conditional SSA (line 276)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'filename' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'filename', remove_member_provider_from_union(filename_127572, 'read'))
        
        # Call to close(...): (line 277)
        # Processing the call keyword arguments (line 277)
        kwargs_127577 = {}
        # Getting the type of 'fid' (line 277)
        fid_127575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'fid', False)
        # Obtaining the member 'close' of a type (line 277)
        close_127576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), fid_127575, 'close')
        # Calling close(args, kwargs) (line 277)
        close_call_result_127578 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), close_127576, *[], **kwargs_127577)
        

        if more_types_in_union_127574:
            # Runtime conditional SSA for else branch (line 276)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_127573) or more_types_in_union_127574):
        # Assigning a type to the variable 'filename' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'filename', remove_not_member_provider_from_union(filename_127572, 'read'))
        
        # Call to seek(...): (line 279)
        # Processing the call arguments (line 279)
        int_127581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 21), 'int')
        # Processing the call keyword arguments (line 279)
        kwargs_127582 = {}
        # Getting the type of 'fid' (line 279)
        fid_127579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'fid', False)
        # Obtaining the member 'seek' of a type (line 279)
        seek_127580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), fid_127579, 'seek')
        # Calling seek(args, kwargs) (line 279)
        seek_call_result_127583 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), seek_127580, *[int_127581], **kwargs_127582)
        

        if (may_be_127573 and more_types_in_union_127574):
            # SSA join for if statement (line 276)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 281)
    tuple_127584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 281)
    # Adding element type (line 281)
    # Getting the type of 'fs' (line 281)
    fs_127585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'fs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 11), tuple_127584, fs_127585)
    # Adding element type (line 281)
    # Getting the type of 'data' (line 281)
    data_127586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'data')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 11), tuple_127584, data_127586)
    
    # Assigning a type to the variable 'stypy_return_type' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type', tuple_127584)
    
    # ################# End of 'read(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_127587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127587)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read'
    return stypy_return_type_127587

# Assigning a type to the variable 'read' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'read', read)

@norecursion
def write(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'write'
    module_type_store = module_type_store.open_function_context('write', 284, 0, False)
    
    # Passed parameters checking function
    write.stypy_localization = localization
    write.stypy_type_of_self = None
    write.stypy_type_store = module_type_store
    write.stypy_function_name = 'write'
    write.stypy_param_names_list = ['filename', 'rate', 'data']
    write.stypy_varargs_param_name = None
    write.stypy_kwargs_param_name = None
    write.stypy_call_defaults = defaults
    write.stypy_call_varargs = varargs
    write.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'write', ['filename', 'rate', 'data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'write', localization, ['filename', 'rate', 'data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'write(...)' code ##################

    str_127588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, (-1)), 'str', '\n    Write a numpy array as a WAV file.\n\n    Parameters\n    ----------\n    filename : string or open file handle\n        Output wav file.\n    rate : int\n        The sample rate (in samples/sec).\n    data : ndarray\n        A 1-D or 2-D numpy array of either integer or float data-type.\n\n    Notes\n    -----\n    * Writes a simple uncompressed WAV file.\n    * To write multiple-channels, use a 2-D array of shape\n      (Nsamples, Nchannels).\n    * The bits-per-sample and PCM/float will be determined by the data-type.\n\n    Common data types: [1]_\n\n    =====================  ===========  ===========  =============\n         WAV format            Min          Max       NumPy dtype\n    =====================  ===========  ===========  =============\n    32-bit floating-point  -1.0         +1.0         float32\n    32-bit PCM             -2147483648  +2147483647  int32\n    16-bit PCM             -32768       +32767       int16\n    8-bit PCM              0            255          uint8\n    =====================  ===========  ===========  =============\n\n    Note that 8-bit PCM is unsigned.\n\n    References\n    ----------\n    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming\n       Interface and Data Specifications 1.0", section "Data Format of the\n       Samples", August 1991\n       http://www.tactilemedia.com/info/MCI_Control_Info.html\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 325)
    str_127589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 25), 'str', 'write')
    # Getting the type of 'filename' (line 325)
    filename_127590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'filename')
    
    (may_be_127591, more_types_in_union_127592) = may_provide_member(str_127589, filename_127590)

    if may_be_127591:

        if more_types_in_union_127592:
            # Runtime conditional SSA (line 325)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'filename' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'filename', remove_not_member_provider_from_union(filename_127590, 'write'))
        
        # Assigning a Name to a Name (line 326):
        
        # Assigning a Name to a Name (line 326):
        # Getting the type of 'filename' (line 326)
        filename_127593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 14), 'filename')
        # Assigning a type to the variable 'fid' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'fid', filename_127593)

        if more_types_in_union_127592:
            # Runtime conditional SSA for else branch (line 325)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_127591) or more_types_in_union_127592):
        # Assigning a type to the variable 'filename' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'filename', remove_member_provider_from_union(filename_127590, 'write'))
        
        # Assigning a Call to a Name (line 328):
        
        # Assigning a Call to a Name (line 328):
        
        # Call to open(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'filename' (line 328)
        filename_127595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'filename', False)
        str_127596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 29), 'str', 'wb')
        # Processing the call keyword arguments (line 328)
        kwargs_127597 = {}
        # Getting the type of 'open' (line 328)
        open_127594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'open', False)
        # Calling open(args, kwargs) (line 328)
        open_call_result_127598 = invoke(stypy.reporting.localization.Localization(__file__, 328, 14), open_127594, *[filename_127595, str_127596], **kwargs_127597)
        
        # Assigning a type to the variable 'fid' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'fid', open_call_result_127598)

        if (may_be_127591 and more_types_in_union_127592):
            # SSA join for if statement (line 325)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 330):
    
    # Assigning a Name to a Name (line 330):
    # Getting the type of 'rate' (line 330)
    rate_127599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 9), 'rate')
    # Assigning a type to the variable 'fs' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'fs', rate_127599)
    
    # Try-finally block (line 332)
    
    # Assigning a Attribute to a Name (line 333):
    
    # Assigning a Attribute to a Name (line 333):
    # Getting the type of 'data' (line 333)
    data_127600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'data')
    # Obtaining the member 'dtype' of a type (line 333)
    dtype_127601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), data_127600, 'dtype')
    # Obtaining the member 'kind' of a type (line 333)
    kind_127602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), dtype_127601, 'kind')
    # Assigning a type to the variable 'dkind' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'dkind', kind_127602)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dkind' (line 334)
    dkind_127603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'dkind')
    str_127604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 25), 'str', 'i')
    # Applying the binary operator '==' (line 334)
    result_eq_127605 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 16), '==', dkind_127603, str_127604)
    
    
    # Getting the type of 'dkind' (line 334)
    dkind_127606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'dkind')
    str_127607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 41), 'str', 'f')
    # Applying the binary operator '==' (line 334)
    result_eq_127608 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 32), '==', dkind_127606, str_127607)
    
    # Applying the binary operator 'or' (line 334)
    result_or_keyword_127609 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 16), 'or', result_eq_127605, result_eq_127608)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dkind' (line 334)
    dkind_127610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 49), 'dkind')
    str_127611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 58), 'str', 'u')
    # Applying the binary operator '==' (line 334)
    result_eq_127612 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 49), '==', dkind_127610, str_127611)
    
    
    # Getting the type of 'data' (line 335)
    data_127613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 49), 'data')
    # Obtaining the member 'dtype' of a type (line 335)
    dtype_127614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 49), data_127613, 'dtype')
    # Obtaining the member 'itemsize' of a type (line 335)
    itemsize_127615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 49), dtype_127614, 'itemsize')
    int_127616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 72), 'int')
    # Applying the binary operator '==' (line 335)
    result_eq_127617 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 49), '==', itemsize_127615, int_127616)
    
    # Applying the binary operator 'and' (line 334)
    result_and_keyword_127618 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 49), 'and', result_eq_127612, result_eq_127617)
    
    # Applying the binary operator 'or' (line 334)
    result_or_keyword_127619 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 16), 'or', result_or_keyword_127609, result_and_keyword_127618)
    
    # Applying the 'not' unary operator (line 334)
    result_not__127620 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), 'not', result_or_keyword_127619)
    
    # Testing the type of an if condition (line 334)
    if_condition_127621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_not__127620)
    # Assigning a type to the variable 'if_condition_127621' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_127621', if_condition_127621)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 336)
    # Processing the call arguments (line 336)
    str_127623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 29), 'str', "Unsupported data type '%s'")
    # Getting the type of 'data' (line 336)
    data_127624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 60), 'data', False)
    # Obtaining the member 'dtype' of a type (line 336)
    dtype_127625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 60), data_127624, 'dtype')
    # Applying the binary operator '%' (line 336)
    result_mod_127626 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 29), '%', str_127623, dtype_127625)
    
    # Processing the call keyword arguments (line 336)
    kwargs_127627 = {}
    # Getting the type of 'ValueError' (line 336)
    ValueError_127622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 336)
    ValueError_call_result_127628 = invoke(stypy.reporting.localization.Localization(__file__, 336, 18), ValueError_127622, *[result_mod_127626], **kwargs_127627)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 336, 12), ValueError_call_result_127628, 'raise parameter', BaseException)
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Str to a Name (line 338):
    
    # Assigning a Str to a Name (line 338):
    str_127629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 22), 'str', '')
    # Assigning a type to the variable 'header_data' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'header_data', str_127629)
    
    # Getting the type of 'header_data' (line 340)
    header_data_127630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'header_data')
    str_127631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 23), 'str', 'RIFF')
    # Applying the binary operator '+=' (line 340)
    result_iadd_127632 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 8), '+=', header_data_127630, str_127631)
    # Assigning a type to the variable 'header_data' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'header_data', result_iadd_127632)
    
    
    # Getting the type of 'header_data' (line 341)
    header_data_127633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'header_data')
    str_127634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 23), 'str', '\x00\x00\x00\x00')
    # Applying the binary operator '+=' (line 341)
    result_iadd_127635 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 8), '+=', header_data_127633, str_127634)
    # Assigning a type to the variable 'header_data' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'header_data', result_iadd_127635)
    
    
    # Getting the type of 'header_data' (line 342)
    header_data_127636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'header_data')
    str_127637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'str', 'WAVE')
    # Applying the binary operator '+=' (line 342)
    result_iadd_127638 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 8), '+=', header_data_127636, str_127637)
    # Assigning a type to the variable 'header_data' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'header_data', result_iadd_127638)
    
    
    # Getting the type of 'header_data' (line 345)
    header_data_127639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'header_data')
    str_127640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 23), 'str', 'fmt ')
    # Applying the binary operator '+=' (line 345)
    result_iadd_127641 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 8), '+=', header_data_127639, str_127640)
    # Assigning a type to the variable 'header_data' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'header_data', result_iadd_127641)
    
    
    
    # Getting the type of 'dkind' (line 346)
    dkind_127642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'dkind')
    str_127643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 20), 'str', 'f')
    # Applying the binary operator '==' (line 346)
    result_eq_127644 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 11), '==', dkind_127642, str_127643)
    
    # Testing the type of an if condition (line 346)
    if_condition_127645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 8), result_eq_127644)
    # Assigning a type to the variable 'if_condition_127645' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'if_condition_127645', if_condition_127645)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 347):
    
    # Assigning a Name to a Name (line 347):
    # Getting the type of 'WAVE_FORMAT_IEEE_FLOAT' (line 347)
    WAVE_FORMAT_IEEE_FLOAT_127646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 25), 'WAVE_FORMAT_IEEE_FLOAT')
    # Assigning a type to the variable 'format_tag' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'format_tag', WAVE_FORMAT_IEEE_FLOAT_127646)
    # SSA branch for the else part of an if statement (line 346)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 349):
    
    # Assigning a Name to a Name (line 349):
    # Getting the type of 'WAVE_FORMAT_PCM' (line 349)
    WAVE_FORMAT_PCM_127647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), 'WAVE_FORMAT_PCM')
    # Assigning a type to the variable 'format_tag' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'format_tag', WAVE_FORMAT_PCM_127647)
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'data' (line 350)
    data_127648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'data')
    # Obtaining the member 'ndim' of a type (line 350)
    ndim_127649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 11), data_127648, 'ndim')
    int_127650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 24), 'int')
    # Applying the binary operator '==' (line 350)
    result_eq_127651 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 11), '==', ndim_127649, int_127650)
    
    # Testing the type of an if condition (line 350)
    if_condition_127652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 8), result_eq_127651)
    # Assigning a type to the variable 'if_condition_127652' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'if_condition_127652', if_condition_127652)
    # SSA begins for if statement (line 350)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 351):
    
    # Assigning a Num to a Name (line 351):
    int_127653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 23), 'int')
    # Assigning a type to the variable 'channels' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'channels', int_127653)
    # SSA branch for the else part of an if statement (line 350)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 353):
    
    # Assigning a Subscript to a Name (line 353):
    
    # Obtaining the type of the subscript
    int_127654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 34), 'int')
    # Getting the type of 'data' (line 353)
    data_127655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 23), 'data')
    # Obtaining the member 'shape' of a type (line 353)
    shape_127656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 23), data_127655, 'shape')
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___127657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 23), shape_127656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_127658 = invoke(stypy.reporting.localization.Localization(__file__, 353, 23), getitem___127657, int_127654)
    
    # Assigning a type to the variable 'channels' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'channels', subscript_call_result_127658)
    # SSA join for if statement (line 350)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 354):
    
    # Assigning a BinOp to a Name (line 354):
    # Getting the type of 'data' (line 354)
    data_127659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 20), 'data')
    # Obtaining the member 'dtype' of a type (line 354)
    dtype_127660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 20), data_127659, 'dtype')
    # Obtaining the member 'itemsize' of a type (line 354)
    itemsize_127661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 20), dtype_127660, 'itemsize')
    int_127662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 42), 'int')
    # Applying the binary operator '*' (line 354)
    result_mul_127663 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 20), '*', itemsize_127661, int_127662)
    
    # Assigning a type to the variable 'bit_depth' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'bit_depth', result_mul_127663)
    
    # Assigning a BinOp to a Name (line 355):
    
    # Assigning a BinOp to a Name (line 355):
    # Getting the type of 'fs' (line 355)
    fs_127664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 27), 'fs')
    # Getting the type of 'bit_depth' (line 355)
    bit_depth_127665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 31), 'bit_depth')
    int_127666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 44), 'int')
    # Applying the binary operator '//' (line 355)
    result_floordiv_127667 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 31), '//', bit_depth_127665, int_127666)
    
    # Applying the binary operator '*' (line 355)
    result_mul_127668 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 27), '*', fs_127664, result_floordiv_127667)
    
    # Getting the type of 'channels' (line 355)
    channels_127669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 47), 'channels')
    # Applying the binary operator '*' (line 355)
    result_mul_127670 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 46), '*', result_mul_127668, channels_127669)
    
    # Assigning a type to the variable 'bytes_per_second' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'bytes_per_second', result_mul_127670)
    
    # Assigning a BinOp to a Name (line 356):
    
    # Assigning a BinOp to a Name (line 356):
    # Getting the type of 'channels' (line 356)
    channels_127671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'channels')
    # Getting the type of 'bit_depth' (line 356)
    bit_depth_127672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'bit_depth')
    int_127673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 47), 'int')
    # Applying the binary operator '//' (line 356)
    result_floordiv_127674 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 34), '//', bit_depth_127672, int_127673)
    
    # Applying the binary operator '*' (line 356)
    result_mul_127675 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 22), '*', channels_127671, result_floordiv_127674)
    
    # Assigning a type to the variable 'block_align' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'block_align', result_mul_127675)
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to pack(...): (line 358)
    # Processing the call arguments (line 358)
    str_127678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 37), 'str', '<HHIIHH')
    # Getting the type of 'format_tag' (line 358)
    format_tag_127679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 48), 'format_tag', False)
    # Getting the type of 'channels' (line 358)
    channels_127680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 60), 'channels', False)
    # Getting the type of 'fs' (line 358)
    fs_127681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 70), 'fs', False)
    # Getting the type of 'bytes_per_second' (line 359)
    bytes_per_second_127682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 37), 'bytes_per_second', False)
    # Getting the type of 'block_align' (line 359)
    block_align_127683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 55), 'block_align', False)
    # Getting the type of 'bit_depth' (line 359)
    bit_depth_127684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 68), 'bit_depth', False)
    # Processing the call keyword arguments (line 358)
    kwargs_127685 = {}
    # Getting the type of 'struct' (line 358)
    struct_127676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 25), 'struct', False)
    # Obtaining the member 'pack' of a type (line 358)
    pack_127677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 25), struct_127676, 'pack')
    # Calling pack(args, kwargs) (line 358)
    pack_call_result_127686 = invoke(stypy.reporting.localization.Localization(__file__, 358, 25), pack_127677, *[str_127678, format_tag_127679, channels_127680, fs_127681, bytes_per_second_127682, block_align_127683, bit_depth_127684], **kwargs_127685)
    
    # Assigning a type to the variable 'fmt_chunk_data' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'fmt_chunk_data', pack_call_result_127686)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dkind' (line 360)
    dkind_127687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'dkind')
    str_127688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 25), 'str', 'i')
    # Applying the binary operator '==' (line 360)
    result_eq_127689 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 16), '==', dkind_127687, str_127688)
    
    
    # Getting the type of 'dkind' (line 360)
    dkind_127690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 32), 'dkind')
    str_127691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 41), 'str', 'u')
    # Applying the binary operator '==' (line 360)
    result_eq_127692 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 32), '==', dkind_127690, str_127691)
    
    # Applying the binary operator 'or' (line 360)
    result_or_keyword_127693 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 16), 'or', result_eq_127689, result_eq_127692)
    
    # Applying the 'not' unary operator (line 360)
    result_not__127694 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 11), 'not', result_or_keyword_127693)
    
    # Testing the type of an if condition (line 360)
    if_condition_127695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 8), result_not__127694)
    # Assigning a type to the variable 'if_condition_127695' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'if_condition_127695', if_condition_127695)
    # SSA begins for if statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'fmt_chunk_data' (line 362)
    fmt_chunk_data_127696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'fmt_chunk_data')
    str_127697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 30), 'str', '\x00\x00')
    # Applying the binary operator '+=' (line 362)
    result_iadd_127698 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 12), '+=', fmt_chunk_data_127696, str_127697)
    # Assigning a type to the variable 'fmt_chunk_data' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'fmt_chunk_data', result_iadd_127698)
    
    # SSA join for if statement (line 360)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'header_data' (line 364)
    header_data_127699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'header_data')
    
    # Call to pack(...): (line 364)
    # Processing the call arguments (line 364)
    str_127702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 35), 'str', '<I')
    
    # Call to len(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'fmt_chunk_data' (line 364)
    fmt_chunk_data_127704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 45), 'fmt_chunk_data', False)
    # Processing the call keyword arguments (line 364)
    kwargs_127705 = {}
    # Getting the type of 'len' (line 364)
    len_127703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 41), 'len', False)
    # Calling len(args, kwargs) (line 364)
    len_call_result_127706 = invoke(stypy.reporting.localization.Localization(__file__, 364, 41), len_127703, *[fmt_chunk_data_127704], **kwargs_127705)
    
    # Processing the call keyword arguments (line 364)
    kwargs_127707 = {}
    # Getting the type of 'struct' (line 364)
    struct_127700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'struct', False)
    # Obtaining the member 'pack' of a type (line 364)
    pack_127701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 23), struct_127700, 'pack')
    # Calling pack(args, kwargs) (line 364)
    pack_call_result_127708 = invoke(stypy.reporting.localization.Localization(__file__, 364, 23), pack_127701, *[str_127702, len_call_result_127706], **kwargs_127707)
    
    # Applying the binary operator '+=' (line 364)
    result_iadd_127709 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 8), '+=', header_data_127699, pack_call_result_127708)
    # Assigning a type to the variable 'header_data' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'header_data', result_iadd_127709)
    
    
    # Getting the type of 'header_data' (line 365)
    header_data_127710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'header_data')
    # Getting the type of 'fmt_chunk_data' (line 365)
    fmt_chunk_data_127711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 23), 'fmt_chunk_data')
    # Applying the binary operator '+=' (line 365)
    result_iadd_127712 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 8), '+=', header_data_127710, fmt_chunk_data_127711)
    # Assigning a type to the variable 'header_data' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'header_data', result_iadd_127712)
    
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dkind' (line 368)
    dkind_127713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'dkind')
    str_127714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 25), 'str', 'i')
    # Applying the binary operator '==' (line 368)
    result_eq_127715 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 16), '==', dkind_127713, str_127714)
    
    
    # Getting the type of 'dkind' (line 368)
    dkind_127716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 32), 'dkind')
    str_127717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 41), 'str', 'u')
    # Applying the binary operator '==' (line 368)
    result_eq_127718 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 32), '==', dkind_127716, str_127717)
    
    # Applying the binary operator 'or' (line 368)
    result_or_keyword_127719 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 16), 'or', result_eq_127715, result_eq_127718)
    
    # Applying the 'not' unary operator (line 368)
    result_not__127720 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 11), 'not', result_or_keyword_127719)
    
    # Testing the type of an if condition (line 368)
    if_condition_127721 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 8), result_not__127720)
    # Assigning a type to the variable 'if_condition_127721' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'if_condition_127721', if_condition_127721)
    # SSA begins for if statement (line 368)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'header_data' (line 369)
    header_data_127722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'header_data')
    str_127723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 27), 'str', 'fact')
    # Applying the binary operator '+=' (line 369)
    result_iadd_127724 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 12), '+=', header_data_127722, str_127723)
    # Assigning a type to the variable 'header_data' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'header_data', result_iadd_127724)
    
    
    # Getting the type of 'header_data' (line 370)
    header_data_127725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'header_data')
    
    # Call to pack(...): (line 370)
    # Processing the call arguments (line 370)
    str_127728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 39), 'str', '<II')
    int_127729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 46), 'int')
    
    # Obtaining the type of the subscript
    int_127730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 60), 'int')
    # Getting the type of 'data' (line 370)
    data_127731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 49), 'data', False)
    # Obtaining the member 'shape' of a type (line 370)
    shape_127732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 49), data_127731, 'shape')
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___127733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 49), shape_127732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 370)
    subscript_call_result_127734 = invoke(stypy.reporting.localization.Localization(__file__, 370, 49), getitem___127733, int_127730)
    
    # Processing the call keyword arguments (line 370)
    kwargs_127735 = {}
    # Getting the type of 'struct' (line 370)
    struct_127726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'struct', False)
    # Obtaining the member 'pack' of a type (line 370)
    pack_127727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 27), struct_127726, 'pack')
    # Calling pack(args, kwargs) (line 370)
    pack_call_result_127736 = invoke(stypy.reporting.localization.Localization(__file__, 370, 27), pack_127727, *[str_127728, int_127729, subscript_call_result_127734], **kwargs_127735)
    
    # Applying the binary operator '+=' (line 370)
    result_iadd_127737 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 12), '+=', header_data_127725, pack_call_result_127736)
    # Assigning a type to the variable 'header_data' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'header_data', result_iadd_127737)
    
    # SSA join for if statement (line 368)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'header_data' (line 373)
    header_data_127739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'header_data', False)
    # Processing the call keyword arguments (line 373)
    kwargs_127740 = {}
    # Getting the type of 'len' (line 373)
    len_127738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 13), 'len', False)
    # Calling len(args, kwargs) (line 373)
    len_call_result_127741 = invoke(stypy.reporting.localization.Localization(__file__, 373, 13), len_127738, *[header_data_127739], **kwargs_127740)
    
    int_127742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 30), 'int')
    # Applying the binary operator '-' (line 373)
    result_sub_127743 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 13), '-', len_call_result_127741, int_127742)
    
    int_127744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 32), 'int')
    # Applying the binary operator '-' (line 373)
    result_sub_127745 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 31), '-', result_sub_127743, int_127744)
    
    int_127746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 38), 'int')
    int_127747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 40), 'int')
    # Applying the binary operator '+' (line 373)
    result_add_127748 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 38), '+', int_127746, int_127747)
    
    # Getting the type of 'data' (line 373)
    data_127749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 42), 'data')
    # Obtaining the member 'nbytes' of a type (line 373)
    nbytes_127750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 42), data_127749, 'nbytes')
    # Applying the binary operator '+' (line 373)
    result_add_127751 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 41), '+', result_add_127748, nbytes_127750)
    
    # Applying the binary operator '+' (line 373)
    result_add_127752 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 12), '+', result_sub_127745, result_add_127751)
    
    long_127753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 58), 'long')
    # Applying the binary operator '>' (line 373)
    result_gt_127754 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 11), '>', result_add_127752, long_127753)
    
    # Testing the type of an if condition (line 373)
    if_condition_127755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 8), result_gt_127754)
    # Assigning a type to the variable 'if_condition_127755' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'if_condition_127755', if_condition_127755)
    # SSA begins for if statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 374)
    # Processing the call arguments (line 374)
    str_127757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 29), 'str', 'Data exceeds wave file size limit')
    # Processing the call keyword arguments (line 374)
    kwargs_127758 = {}
    # Getting the type of 'ValueError' (line 374)
    ValueError_127756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 374)
    ValueError_call_result_127759 = invoke(stypy.reporting.localization.Localization(__file__, 374, 18), ValueError_127756, *[str_127757], **kwargs_127758)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 374, 12), ValueError_call_result_127759, 'raise parameter', BaseException)
    # SSA join for if statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'header_data' (line 376)
    header_data_127762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), 'header_data', False)
    # Processing the call keyword arguments (line 376)
    kwargs_127763 = {}
    # Getting the type of 'fid' (line 376)
    fid_127760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 376)
    write_127761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), fid_127760, 'write')
    # Calling write(args, kwargs) (line 376)
    write_call_result_127764 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), write_127761, *[header_data_127762], **kwargs_127763)
    
    
    # Call to write(...): (line 379)
    # Processing the call arguments (line 379)
    str_127767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 18), 'str', 'data')
    # Processing the call keyword arguments (line 379)
    kwargs_127768 = {}
    # Getting the type of 'fid' (line 379)
    fid_127765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 379)
    write_127766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), fid_127765, 'write')
    # Calling write(args, kwargs) (line 379)
    write_call_result_127769 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), write_127766, *[str_127767], **kwargs_127768)
    
    
    # Call to write(...): (line 380)
    # Processing the call arguments (line 380)
    
    # Call to pack(...): (line 380)
    # Processing the call arguments (line 380)
    str_127774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 30), 'str', '<I')
    # Getting the type of 'data' (line 380)
    data_127775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 36), 'data', False)
    # Obtaining the member 'nbytes' of a type (line 380)
    nbytes_127776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 36), data_127775, 'nbytes')
    # Processing the call keyword arguments (line 380)
    kwargs_127777 = {}
    # Getting the type of 'struct' (line 380)
    struct_127772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 18), 'struct', False)
    # Obtaining the member 'pack' of a type (line 380)
    pack_127773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 18), struct_127772, 'pack')
    # Calling pack(args, kwargs) (line 380)
    pack_call_result_127778 = invoke(stypy.reporting.localization.Localization(__file__, 380, 18), pack_127773, *[str_127774, nbytes_127776], **kwargs_127777)
    
    # Processing the call keyword arguments (line 380)
    kwargs_127779 = {}
    # Getting the type of 'fid' (line 380)
    fid_127770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 380)
    write_127771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), fid_127770, 'write')
    # Calling write(args, kwargs) (line 380)
    write_call_result_127780 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), write_127771, *[pack_call_result_127778], **kwargs_127779)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'data' (line 381)
    data_127781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'data')
    # Obtaining the member 'dtype' of a type (line 381)
    dtype_127782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 11), data_127781, 'dtype')
    # Obtaining the member 'byteorder' of a type (line 381)
    byteorder_127783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 11), dtype_127782, 'byteorder')
    str_127784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 35), 'str', '>')
    # Applying the binary operator '==' (line 381)
    result_eq_127785 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 11), '==', byteorder_127783, str_127784)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'data' (line 381)
    data_127786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 43), 'data')
    # Obtaining the member 'dtype' of a type (line 381)
    dtype_127787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 43), data_127786, 'dtype')
    # Obtaining the member 'byteorder' of a type (line 381)
    byteorder_127788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 43), dtype_127787, 'byteorder')
    str_127789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 67), 'str', '=')
    # Applying the binary operator '==' (line 381)
    result_eq_127790 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 43), '==', byteorder_127788, str_127789)
    
    
    # Getting the type of 'sys' (line 382)
    sys_127791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 43), 'sys')
    # Obtaining the member 'byteorder' of a type (line 382)
    byteorder_127792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 43), sys_127791, 'byteorder')
    str_127793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 60), 'str', 'big')
    # Applying the binary operator '==' (line 382)
    result_eq_127794 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 43), '==', byteorder_127792, str_127793)
    
    # Applying the binary operator 'and' (line 381)
    result_and_keyword_127795 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 43), 'and', result_eq_127790, result_eq_127794)
    
    # Applying the binary operator 'or' (line 381)
    result_or_keyword_127796 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 11), 'or', result_eq_127785, result_and_keyword_127795)
    
    # Testing the type of an if condition (line 381)
    if_condition_127797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 8), result_or_keyword_127796)
    # Assigning a type to the variable 'if_condition_127797' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'if_condition_127797', if_condition_127797)
    # SSA begins for if statement (line 381)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 383):
    
    # Assigning a Call to a Name (line 383):
    
    # Call to byteswap(...): (line 383)
    # Processing the call keyword arguments (line 383)
    kwargs_127800 = {}
    # Getting the type of 'data' (line 383)
    data_127798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'data', False)
    # Obtaining the member 'byteswap' of a type (line 383)
    byteswap_127799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), data_127798, 'byteswap')
    # Calling byteswap(args, kwargs) (line 383)
    byteswap_call_result_127801 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), byteswap_127799, *[], **kwargs_127800)
    
    # Assigning a type to the variable 'data' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'data', byteswap_call_result_127801)
    # SSA join for if statement (line 381)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _array_tofile(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'fid' (line 384)
    fid_127803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 22), 'fid', False)
    # Getting the type of 'data' (line 384)
    data_127804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'data', False)
    # Processing the call keyword arguments (line 384)
    kwargs_127805 = {}
    # Getting the type of '_array_tofile' (line 384)
    _array_tofile_127802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), '_array_tofile', False)
    # Calling _array_tofile(args, kwargs) (line 384)
    _array_tofile_call_result_127806 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), _array_tofile_127802, *[fid_127803, data_127804], **kwargs_127805)
    
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to tell(...): (line 388)
    # Processing the call keyword arguments (line 388)
    kwargs_127809 = {}
    # Getting the type of 'fid' (line 388)
    fid_127807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'fid', False)
    # Obtaining the member 'tell' of a type (line 388)
    tell_127808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), fid_127807, 'tell')
    # Calling tell(args, kwargs) (line 388)
    tell_call_result_127810 = invoke(stypy.reporting.localization.Localization(__file__, 388, 15), tell_127808, *[], **kwargs_127809)
    
    # Assigning a type to the variable 'size' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'size', tell_call_result_127810)
    
    # Call to seek(...): (line 389)
    # Processing the call arguments (line 389)
    int_127813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 17), 'int')
    # Processing the call keyword arguments (line 389)
    kwargs_127814 = {}
    # Getting the type of 'fid' (line 389)
    fid_127811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'fid', False)
    # Obtaining the member 'seek' of a type (line 389)
    seek_127812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), fid_127811, 'seek')
    # Calling seek(args, kwargs) (line 389)
    seek_call_result_127815 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), seek_127812, *[int_127813], **kwargs_127814)
    
    
    # Call to write(...): (line 390)
    # Processing the call arguments (line 390)
    
    # Call to pack(...): (line 390)
    # Processing the call arguments (line 390)
    str_127820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 30), 'str', '<I')
    # Getting the type of 'size' (line 390)
    size_127821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 36), 'size', False)
    int_127822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 41), 'int')
    # Applying the binary operator '-' (line 390)
    result_sub_127823 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 36), '-', size_127821, int_127822)
    
    # Processing the call keyword arguments (line 390)
    kwargs_127824 = {}
    # Getting the type of 'struct' (line 390)
    struct_127818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 18), 'struct', False)
    # Obtaining the member 'pack' of a type (line 390)
    pack_127819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 18), struct_127818, 'pack')
    # Calling pack(args, kwargs) (line 390)
    pack_call_result_127825 = invoke(stypy.reporting.localization.Localization(__file__, 390, 18), pack_127819, *[str_127820, result_sub_127823], **kwargs_127824)
    
    # Processing the call keyword arguments (line 390)
    kwargs_127826 = {}
    # Getting the type of 'fid' (line 390)
    fid_127816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 390)
    write_127817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), fid_127816, 'write')
    # Calling write(args, kwargs) (line 390)
    write_call_result_127827 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), write_127817, *[pack_call_result_127825], **kwargs_127826)
    
    
    # finally branch of the try-finally block (line 332)
    
    # Type idiom detected: calculating its left and rigth part (line 393)
    str_127828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 33), 'str', 'write')
    # Getting the type of 'filename' (line 393)
    filename_127829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 23), 'filename')
    
    (may_be_127830, more_types_in_union_127831) = may_not_provide_member(str_127828, filename_127829)

    if may_be_127830:

        if more_types_in_union_127831:
            # Runtime conditional SSA (line 393)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'filename' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'filename', remove_member_provider_from_union(filename_127829, 'write'))
        
        # Call to close(...): (line 394)
        # Processing the call keyword arguments (line 394)
        kwargs_127834 = {}
        # Getting the type of 'fid' (line 394)
        fid_127832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'fid', False)
        # Obtaining the member 'close' of a type (line 394)
        close_127833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), fid_127832, 'close')
        # Calling close(args, kwargs) (line 394)
        close_call_result_127835 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), close_127833, *[], **kwargs_127834)
        

        if more_types_in_union_127831:
            # Runtime conditional SSA for else branch (line 393)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_127830) or more_types_in_union_127831):
        # Assigning a type to the variable 'filename' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'filename', remove_not_member_provider_from_union(filename_127829, 'write'))
        
        # Call to seek(...): (line 396)
        # Processing the call arguments (line 396)
        int_127838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 21), 'int')
        # Processing the call keyword arguments (line 396)
        kwargs_127839 = {}
        # Getting the type of 'fid' (line 396)
        fid_127836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'fid', False)
        # Obtaining the member 'seek' of a type (line 396)
        seek_127837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 12), fid_127836, 'seek')
        # Calling seek(args, kwargs) (line 396)
        seek_call_result_127840 = invoke(stypy.reporting.localization.Localization(__file__, 396, 12), seek_127837, *[int_127838], **kwargs_127839)
        

        if (may_be_127830 and more_types_in_union_127831):
            # SSA join for if statement (line 393)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # ################# End of 'write(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'write' in the type store
    # Getting the type of 'stypy_return_type' (line 284)
    stypy_return_type_127841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127841)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'write'
    return stypy_return_type_127841

# Assigning a type to the variable 'write' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'write', write)



# Obtaining the type of the subscript
int_127842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 20), 'int')
# Getting the type of 'sys' (line 399)
sys_127843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 399)
version_info_127844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 3), sys_127843, 'version_info')
# Obtaining the member '__getitem__' of a type (line 399)
getitem___127845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 3), version_info_127844, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 399)
subscript_call_result_127846 = invoke(stypy.reporting.localization.Localization(__file__, 399, 3), getitem___127845, int_127842)

int_127847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 26), 'int')
# Applying the binary operator '>=' (line 399)
result_ge_127848 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 3), '>=', subscript_call_result_127846, int_127847)

# Testing the type of an if condition (line 399)
if_condition_127849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 0), result_ge_127848)
# Assigning a type to the variable 'if_condition_127849' (line 399)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'if_condition_127849', if_condition_127849)
# SSA begins for if statement (line 399)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def _array_tofile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_array_tofile'
    module_type_store = module_type_store.open_function_context('_array_tofile', 400, 4, False)
    
    # Passed parameters checking function
    _array_tofile.stypy_localization = localization
    _array_tofile.stypy_type_of_self = None
    _array_tofile.stypy_type_store = module_type_store
    _array_tofile.stypy_function_name = '_array_tofile'
    _array_tofile.stypy_param_names_list = ['fid', 'data']
    _array_tofile.stypy_varargs_param_name = None
    _array_tofile.stypy_kwargs_param_name = None
    _array_tofile.stypy_call_defaults = defaults
    _array_tofile.stypy_call_varargs = varargs
    _array_tofile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_array_tofile', ['fid', 'data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_array_tofile', localization, ['fid', 'data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_array_tofile(...)' code ##################

    
    # Call to write(...): (line 402)
    # Processing the call arguments (line 402)
    
    # Call to view(...): (line 402)
    # Processing the call arguments (line 402)
    str_127857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 36), 'str', 'b')
    # Processing the call keyword arguments (line 402)
    kwargs_127858 = {}
    
    # Call to ravel(...): (line 402)
    # Processing the call keyword arguments (line 402)
    kwargs_127854 = {}
    # Getting the type of 'data' (line 402)
    data_127852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'data', False)
    # Obtaining the member 'ravel' of a type (line 402)
    ravel_127853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 18), data_127852, 'ravel')
    # Calling ravel(args, kwargs) (line 402)
    ravel_call_result_127855 = invoke(stypy.reporting.localization.Localization(__file__, 402, 18), ravel_127853, *[], **kwargs_127854)
    
    # Obtaining the member 'view' of a type (line 402)
    view_127856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 18), ravel_call_result_127855, 'view')
    # Calling view(args, kwargs) (line 402)
    view_call_result_127859 = invoke(stypy.reporting.localization.Localization(__file__, 402, 18), view_127856, *[str_127857], **kwargs_127858)
    
    # Obtaining the member 'data' of a type (line 402)
    data_127860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 18), view_call_result_127859, 'data')
    # Processing the call keyword arguments (line 402)
    kwargs_127861 = {}
    # Getting the type of 'fid' (line 402)
    fid_127850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 402)
    write_127851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), fid_127850, 'write')
    # Calling write(args, kwargs) (line 402)
    write_call_result_127862 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), write_127851, *[data_127860], **kwargs_127861)
    
    
    # ################# End of '_array_tofile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_array_tofile' in the type store
    # Getting the type of 'stypy_return_type' (line 400)
    stypy_return_type_127863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_array_tofile'
    return stypy_return_type_127863

# Assigning a type to the variable '_array_tofile' (line 400)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), '_array_tofile', _array_tofile)
# SSA branch for the else part of an if statement (line 399)
module_type_store.open_ssa_branch('else')

@norecursion
def _array_tofile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_array_tofile'
    module_type_store = module_type_store.open_function_context('_array_tofile', 404, 4, False)
    
    # Passed parameters checking function
    _array_tofile.stypy_localization = localization
    _array_tofile.stypy_type_of_self = None
    _array_tofile.stypy_type_store = module_type_store
    _array_tofile.stypy_function_name = '_array_tofile'
    _array_tofile.stypy_param_names_list = ['fid', 'data']
    _array_tofile.stypy_varargs_param_name = None
    _array_tofile.stypy_kwargs_param_name = None
    _array_tofile.stypy_call_defaults = defaults
    _array_tofile.stypy_call_varargs = varargs
    _array_tofile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_array_tofile', ['fid', 'data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_array_tofile', localization, ['fid', 'data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_array_tofile(...)' code ##################

    
    # Call to write(...): (line 405)
    # Processing the call arguments (line 405)
    
    # Call to tostring(...): (line 405)
    # Processing the call keyword arguments (line 405)
    kwargs_127868 = {}
    # Getting the type of 'data' (line 405)
    data_127866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 18), 'data', False)
    # Obtaining the member 'tostring' of a type (line 405)
    tostring_127867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 18), data_127866, 'tostring')
    # Calling tostring(args, kwargs) (line 405)
    tostring_call_result_127869 = invoke(stypy.reporting.localization.Localization(__file__, 405, 18), tostring_127867, *[], **kwargs_127868)
    
    # Processing the call keyword arguments (line 405)
    kwargs_127870 = {}
    # Getting the type of 'fid' (line 405)
    fid_127864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 405)
    write_127865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), fid_127864, 'write')
    # Calling write(args, kwargs) (line 405)
    write_call_result_127871 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), write_127865, *[tostring_call_result_127869], **kwargs_127870)
    
    
    # ################# End of '_array_tofile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_array_tofile' in the type store
    # Getting the type of 'stypy_return_type' (line 404)
    stypy_return_type_127872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_127872)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_array_tofile'
    return stypy_return_type_127872

# Assigning a type to the variable '_array_tofile' (line 404)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), '_array_tofile', _array_tofile)
# SSA join for if statement (line 399)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
