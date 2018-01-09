
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Authors: Travis Oliphant, Matthew Brett
2: 
3: '''
4: Base classes for MATLAB file stream reading.
5: 
6: MATLAB is a registered trademark of the Mathworks inc.
7: '''
8: from __future__ import division, print_function, absolute_import
9: 
10: import sys
11: import operator
12: 
13: from scipy._lib.six import reduce
14: 
15: import numpy as np
16: 
17: if sys.version_info[0] >= 3:
18:     byteord = int
19: else:
20:     byteord = ord
21: 
22: from scipy.misc import doccer
23: 
24: from . import byteordercodes as boc
25: 
26: 
27: class MatReadError(Exception):
28:     pass
29: 
30: 
31: class MatWriteError(Exception):
32:     pass
33: 
34: 
35: class MatReadWarning(UserWarning):
36:     pass
37: 
38: 
39: doc_dict = \
40:     {'file_arg':
41:          '''file_name : str
42:    Name of the mat file (do not need .mat extension if
43:    appendmat==True) Can also pass open file-like object.''',
44:      'append_arg':
45:          '''appendmat : bool, optional
46:    True to append the .mat extension to the end of the given
47:    filename, if not already present.''',
48:      'load_args':
49:          '''byte_order : str or None, optional
50:    None by default, implying byte order guessed from mat
51:    file. Otherwise can be one of ('native', '=', 'little', '<',
52:    'BIG', '>').
53: mat_dtype : bool, optional
54:    If True, return arrays in same dtype as would be loaded into
55:    MATLAB (instead of the dtype with which they are saved).
56: squeeze_me : bool, optional
57:    Whether to squeeze unit matrix dimensions or not.
58: chars_as_strings : bool, optional
59:    Whether to convert char arrays to string arrays.
60: matlab_compatible : bool, optional
61:    Returns matrices as would be loaded by MATLAB (implies
62:    squeeze_me=False, chars_as_strings=False, mat_dtype=True,
63:    struct_as_record=True).''',
64:      'struct_arg':
65:          '''struct_as_record : bool, optional
66:    Whether to load MATLAB structs as numpy record arrays, or as
67:    old-style numpy arrays with dtype=object.  Setting this flag to
68:    False replicates the behavior of scipy version 0.7.x (returning
69:    numpy object arrays).  The default setting is True, because it
70:    allows easier round-trip load and save of MATLAB files.''',
71:      'matstream_arg':
72:          '''mat_stream : file-like
73:    Object with file API, open for reading.''',
74:      'long_fields':
75:          '''long_field_names : bool, optional
76:    * False - maximum field name length in a structure is 31 characters
77:      which is the documented maximum length. This is the default.
78:    * True - maximum field name length in a structure is 63 characters
79:      which works for MATLAB 7.6''',
80:      'do_compression':
81:          '''do_compression : bool, optional
82:    Whether to compress matrices on write. Default is False.''',
83:      'oned_as':
84:          '''oned_as : {'row', 'column'}, optional
85:    If 'column', write 1-D numpy arrays as column vectors.
86:    If 'row', write 1D numpy arrays as row vectors.''',
87:      'unicode_strings':
88:          '''unicode_strings : bool, optional
89:    If True, write strings as Unicode, else MATLAB usual encoding.'''}
90: 
91: docfiller = doccer.filldoc(doc_dict)
92: 
93: '''
94: 
95:  Note on architecture
96: ======================
97: 
98: There are three sets of parameters relevant for reading files.  The
99: first are *file read parameters* - containing options that are common
100: for reading the whole file, and therefore every variable within that
101: file. At the moment these are:
102: 
103: * mat_stream
104: * dtypes (derived from byte code)
105: * byte_order
106: * chars_as_strings
107: * squeeze_me
108: * struct_as_record (MATLAB 5 files)
109: * class_dtypes (derived from order code, MATLAB 5 files)
110: * codecs (MATLAB 5 files)
111: * uint16_codec (MATLAB 5 files)
112: 
113: Another set of parameters are those that apply only to the current
114: variable being read - the *header*:
115: 
116: * header related variables (different for v4 and v5 mat files)
117: * is_complex
118: * mclass
119: * var_stream
120: 
121: With the header, we need ``next_position`` to tell us where the next
122: variable in the stream is.
123: 
124: Then, for each element in a matrix, there can be *element read
125: parameters*.  An element is, for example, one element in a MATLAB cell
126: array.  At the moment these are:
127: 
128: * mat_dtype
129: 
130: The file-reading object contains the *file read parameters*.  The
131: *header* is passed around as a data object, or may be read and discarded
132: in a single function.  The *element read parameters* - the mat_dtype in
133: this instance, is passed into a general post-processing function - see
134: ``mio_utils`` for details.
135: '''
136: 
137: 
138: def convert_dtypes(dtype_template, order_code):
139:     ''' Convert dtypes in mapping to given order
140: 
141:     Parameters
142:     ----------
143:     dtype_template : mapping
144:        mapping with values returning numpy dtype from ``np.dtype(val)``
145:     order_code : str
146:        an order code suitable for using in ``dtype.newbyteorder()``
147: 
148:     Returns
149:     -------
150:     dtypes : mapping
151:        mapping where values have been replaced by
152:        ``np.dtype(val).newbyteorder(order_code)``
153: 
154:     '''
155:     dtypes = dtype_template.copy()
156:     for k in dtypes:
157:         dtypes[k] = np.dtype(dtypes[k]).newbyteorder(order_code)
158:     return dtypes
159: 
160: 
161: def read_dtype(mat_stream, a_dtype):
162:     '''
163:     Generic get of byte stream data of known type
164: 
165:     Parameters
166:     ----------
167:     mat_stream : file_like object
168:         MATLAB (tm) mat file stream
169:     a_dtype : dtype
170:         dtype of array to read.  `a_dtype` is assumed to be correct
171:         endianness.
172: 
173:     Returns
174:     -------
175:     arr : ndarray
176:         Array of dtype `a_dtype` read from stream.
177: 
178:     '''
179:     num_bytes = a_dtype.itemsize
180:     arr = np.ndarray(shape=(),
181:                      dtype=a_dtype,
182:                      buffer=mat_stream.read(num_bytes),
183:                      order='F')
184:     return arr
185: 
186: 
187: def get_matfile_version(fileobj):
188:     '''
189:     Return major, minor tuple depending on apparent mat file type
190: 
191:     Where:
192: 
193:      #. 0,x -> version 4 format mat files
194:      #. 1,x -> version 5 format mat files
195:      #. 2,x -> version 7.3 format mat files (HDF format)
196: 
197:     Parameters
198:     ----------
199:     fileobj : file_like
200:         object implementing seek() and read()
201: 
202:     Returns
203:     -------
204:     major_version : {0, 1, 2}
205:         major MATLAB File format version
206:     minor_version : int
207:         minor MATLAB file format version
208: 
209:     Raises
210:     ------
211:     MatReadError
212:         If the file is empty.
213:     ValueError
214:         The matfile version is unknown.
215: 
216:     Notes
217:     -----
218:     Has the side effect of setting the file read pointer to 0
219:     '''
220:     # Mat4 files have a zero somewhere in first 4 bytes
221:     fileobj.seek(0)
222:     mopt_bytes = fileobj.read(4)
223:     if len(mopt_bytes) == 0:
224:         raise MatReadError("Mat file appears to be empty")
225:     mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=mopt_bytes)
226:     if 0 in mopt_ints:
227:         fileobj.seek(0)
228:         return (0,0)
229:     # For 5 format or 7.3 format we need to read an integer in the
230:     # header. Bytes 124 through 128 contain a version integer and an
231:     # endian test string
232:     fileobj.seek(124)
233:     tst_str = fileobj.read(4)
234:     fileobj.seek(0)
235:     maj_ind = int(tst_str[2] == b'I'[0])
236:     maj_val = byteord(tst_str[maj_ind])
237:     min_val = byteord(tst_str[1-maj_ind])
238:     ret = (maj_val, min_val)
239:     if maj_val in (1, 2):
240:         return ret
241:     raise ValueError('Unknown mat file type, version %s, %s' % ret)
242: 
243: 
244: def matdims(arr, oned_as='column'):
245:     '''
246:     Determine equivalent MATLAB dimensions for given array
247: 
248:     Parameters
249:     ----------
250:     arr : ndarray
251:         Input array
252:     oned_as : {'column', 'row'}, optional
253:         Whether 1-D arrays are returned as MATLAB row or column matrices.
254:         Default is 'column'.
255: 
256:     Returns
257:     -------
258:     dims : tuple
259:         Shape tuple, in the form MATLAB expects it.
260: 
261:     Notes
262:     -----
263:     We had to decide what shape a 1 dimensional array would be by
264:     default.  ``np.atleast_2d`` thinks it is a row vector.  The
265:     default for a vector in MATLAB (e.g. ``>> 1:12``) is a row vector.
266: 
267:     Versions of scipy up to and including 0.11 resulted (accidentally)
268:     in 1-D arrays being read as column vectors.  For the moment, we
269:     maintain the same tradition here.
270: 
271:     Examples
272:     --------
273:     >>> matdims(np.array(1)) # numpy scalar
274:     (1, 1)
275:     >>> matdims(np.array([1])) # 1d array, 1 element
276:     (1, 1)
277:     >>> matdims(np.array([1,2])) # 1d array, 2 elements
278:     (2, 1)
279:     >>> matdims(np.array([[2],[3]])) # 2d array, column vector
280:     (2, 1)
281:     >>> matdims(np.array([[2,3]])) # 2d array, row vector
282:     (1, 2)
283:     >>> matdims(np.array([[[2,3]]])) # 3d array, rowish vector
284:     (1, 1, 2)
285:     >>> matdims(np.array([])) # empty 1d array
286:     (0, 0)
287:     >>> matdims(np.array([[]])) # empty 2d
288:     (0, 0)
289:     >>> matdims(np.array([[[]]])) # empty 3d
290:     (0, 0, 0)
291: 
292:     Optional argument flips 1-D shape behavior.
293: 
294:     >>> matdims(np.array([1,2]), 'row') # 1d array, 2 elements
295:     (1, 2)
296: 
297:     The argument has to make sense though
298: 
299:     >>> matdims(np.array([1,2]), 'bizarre')
300:     Traceback (most recent call last):
301:        ...
302:     ValueError: 1D option "bizarre" is strange
303: 
304:     '''
305:     shape = arr.shape
306:     if shape == ():  # scalar
307:         return (1,1)
308:     if reduce(operator.mul, shape) == 0:  # zero elememts
309:         return (0,) * np.max([arr.ndim, 2])
310:     if len(shape) == 1:  # 1D
311:         if oned_as == 'column':
312:             return shape + (1,)
313:         elif oned_as == 'row':
314:             return (1,) + shape
315:         else:
316:             raise ValueError('1D option "%s" is strange'
317:                              % oned_as)
318:     return shape
319: 
320: 
321: class MatVarReader(object):
322:     ''' Abstract class defining required interface for var readers'''
323:     def __init__(self, file_reader):
324:         pass
325: 
326:     def read_header(self):
327:         ''' Returns header '''
328:         pass
329: 
330:     def array_from_header(self, header):
331:         ''' Reads array given header '''
332:         pass
333: 
334: 
335: class MatFileReader(object):
336:     ''' Base object for reading mat files
337: 
338:     To make this class functional, you will need to override the
339:     following methods:
340: 
341:     matrix_getter_factory   - gives object to fetch next matrix from stream
342:     guess_byte_order        - guesses file byte order from file
343:     '''
344: 
345:     @docfiller
346:     def __init__(self, mat_stream,
347:                  byte_order=None,
348:                  mat_dtype=False,
349:                  squeeze_me=False,
350:                  chars_as_strings=True,
351:                  matlab_compatible=False,
352:                  struct_as_record=True,
353:                  verify_compressed_data_integrity=True
354:                  ):
355:         '''
356:         Initializer for mat file reader
357: 
358:         mat_stream : file-like
359:             object with file API, open for reading
360:     %(load_args)s
361:         '''
362:         # Initialize stream
363:         self.mat_stream = mat_stream
364:         self.dtypes = {}
365:         if not byte_order:
366:             byte_order = self.guess_byte_order()
367:         else:
368:             byte_order = boc.to_numpy_code(byte_order)
369:         self.byte_order = byte_order
370:         self.struct_as_record = struct_as_record
371:         if matlab_compatible:
372:             self.set_matlab_compatible()
373:         else:
374:             self.squeeze_me = squeeze_me
375:             self.chars_as_strings = chars_as_strings
376:             self.mat_dtype = mat_dtype
377:         self.verify_compressed_data_integrity = verify_compressed_data_integrity
378: 
379:     def set_matlab_compatible(self):
380:         ''' Sets options to return arrays as MATLAB loads them '''
381:         self.mat_dtype = True
382:         self.squeeze_me = False
383:         self.chars_as_strings = False
384: 
385:     def guess_byte_order(self):
386:         ''' As we do not know what file type we have, assume native '''
387:         return boc.native_code
388: 
389:     def end_of_stream(self):
390:         b = self.mat_stream.read(1)
391:         curpos = self.mat_stream.tell()
392:         self.mat_stream.seek(curpos-1)
393:         return len(b) == 0
394: 
395: 
396: def arr_dtype_number(arr, num):
397:     ''' Return dtype for given number of items per element'''
398:     return np.dtype(arr.dtype.str[:2] + str(num))
399: 
400: 
401: def arr_to_chars(arr):
402:     ''' Convert string array to char array '''
403:     dims = list(arr.shape)
404:     if not dims:
405:         dims = [1]
406:     dims.append(int(arr.dtype.str[2:]))
407:     arr = np.ndarray(shape=dims,
408:                      dtype=arr_dtype_number(arr, 1),
409:                      buffer=arr)
410:     empties = [arr == '']
411:     if not np.any(empties):
412:         return arr
413:     arr = arr.copy()
414:     arr[empties] = ' '
415:     return arr
416: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_137524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nBase classes for MATLAB file stream reading.\n\nMATLAB is a registered trademark of the Mathworks inc.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import operator' statement (line 11)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib.six import reduce' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_137525 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six')

if (type(import_137525) is not StypyTypeError):

    if (import_137525 != 'pyd_module'):
        __import__(import_137525)
        sys_modules_137526 = sys.modules[import_137525]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', sys_modules_137526.module_type_store, module_type_store, ['reduce'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_137526, sys_modules_137526.module_type_store, module_type_store)
    else:
        from scipy._lib.six import reduce

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', None, module_type_store, ['reduce'], [reduce])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', import_137525)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import numpy' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_137527 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy')

if (type(import_137527) is not StypyTypeError):

    if (import_137527 != 'pyd_module'):
        __import__(import_137527)
        sys_modules_137528 = sys.modules[import_137527]
        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'np', sys_modules_137528.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', import_137527)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')




# Obtaining the type of the subscript
int_137529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
# Getting the type of 'sys' (line 17)
sys_137530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 17)
version_info_137531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 3), sys_137530, 'version_info')
# Obtaining the member '__getitem__' of a type (line 17)
getitem___137532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 3), version_info_137531, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 17)
subscript_call_result_137533 = invoke(stypy.reporting.localization.Localization(__file__, 17, 3), getitem___137532, int_137529)

int_137534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
# Applying the binary operator '>=' (line 17)
result_ge_137535 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 3), '>=', subscript_call_result_137533, int_137534)

# Testing the type of an if condition (line 17)
if_condition_137536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 0), result_ge_137535)
# Assigning a type to the variable 'if_condition_137536' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'if_condition_137536', if_condition_137536)
# SSA begins for if statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 18):
# Getting the type of 'int' (line 18)
int_137537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'int')
# Assigning a type to the variable 'byteord' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'byteord', int_137537)
# SSA branch for the else part of an if statement (line 17)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 20):
# Getting the type of 'ord' (line 20)
ord_137538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'ord')
# Assigning a type to the variable 'byteord' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'byteord', ord_137538)
# SSA join for if statement (line 17)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from scipy.misc import doccer' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_137539 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.misc')

if (type(import_137539) is not StypyTypeError):

    if (import_137539 != 'pyd_module'):
        __import__(import_137539)
        sys_modules_137540 = sys.modules[import_137539]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.misc', sys_modules_137540.module_type_store, module_type_store, ['doccer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_137540, sys_modules_137540.module_type_store, module_type_store)
    else:
        from scipy.misc import doccer

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.misc', None, module_type_store, ['doccer'], [doccer])

else:
    # Assigning a type to the variable 'scipy.misc' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'scipy.misc', import_137539)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy.io.matlab import boc' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_137541 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.io.matlab')

if (type(import_137541) is not StypyTypeError):

    if (import_137541 != 'pyd_module'):
        __import__(import_137541)
        sys_modules_137542 = sys.modules[import_137541]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.io.matlab', sys_modules_137542.module_type_store, module_type_store, ['byteordercodes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_137542, sys_modules_137542.module_type_store, module_type_store)
    else:
        from scipy.io.matlab import byteordercodes as boc

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.io.matlab', None, module_type_store, ['byteordercodes'], [boc])

else:
    # Assigning a type to the variable 'scipy.io.matlab' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.io.matlab', import_137541)

# Adding an alias
module_type_store.add_alias('boc', 'byteordercodes')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

# Declaration of the 'MatReadError' class
# Getting the type of 'Exception' (line 27)
Exception_137543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'Exception')

class MatReadError(Exception_137543, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 27, 0, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatReadError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatReadError' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'MatReadError', MatReadError)
# Declaration of the 'MatWriteError' class
# Getting the type of 'Exception' (line 31)
Exception_137544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'Exception')

class MatWriteError(Exception_137544, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 31, 0, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatWriteError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatWriteError' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'MatWriteError', MatWriteError)
# Declaration of the 'MatReadWarning' class
# Getting the type of 'UserWarning' (line 35)
UserWarning_137545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'UserWarning')

class MatReadWarning(UserWarning_137545, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 35, 0, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatReadWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatReadWarning' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'MatReadWarning', MatReadWarning)

# Assigning a Dict to a Name (line 39):

# Obtaining an instance of the builtin type 'dict' (line 40)
dict_137546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 40)
# Adding element type (key, value) (line 40)
str_137547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 5), 'str', 'file_arg')
str_137548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', 'file_name : str\n   Name of the mat file (do not need .mat extension if\n   appendmat==True) Can also pass open file-like object.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137547, str_137548))
# Adding element type (key, value) (line 40)
str_137549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 5), 'str', 'append_arg')
str_137550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', 'appendmat : bool, optional\n   True to append the .mat extension to the end of the given\n   filename, if not already present.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137549, str_137550))
# Adding element type (key, value) (line 40)
str_137551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 5), 'str', 'load_args')
str_137552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', "byte_order : str or None, optional\n   None by default, implying byte order guessed from mat\n   file. Otherwise can be one of ('native', '=', 'little', '<',\n   'BIG', '>').\nmat_dtype : bool, optional\n   If True, return arrays in same dtype as would be loaded into\n   MATLAB (instead of the dtype with which they are saved).\nsqueeze_me : bool, optional\n   Whether to squeeze unit matrix dimensions or not.\nchars_as_strings : bool, optional\n   Whether to convert char arrays to string arrays.\nmatlab_compatible : bool, optional\n   Returns matrices as would be loaded by MATLAB (implies\n   squeeze_me=False, chars_as_strings=False, mat_dtype=True,\n   struct_as_record=True).")
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137551, str_137552))
# Adding element type (key, value) (line 40)
str_137553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 5), 'str', 'struct_arg')
str_137554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, (-1)), 'str', 'struct_as_record : bool, optional\n   Whether to load MATLAB structs as numpy record arrays, or as\n   old-style numpy arrays with dtype=object.  Setting this flag to\n   False replicates the behavior of scipy version 0.7.x (returning\n   numpy object arrays).  The default setting is True, because it\n   allows easier round-trip load and save of MATLAB files.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137553, str_137554))
# Adding element type (key, value) (line 40)
str_137555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 5), 'str', 'matstream_arg')
str_137556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', 'mat_stream : file-like\n   Object with file API, open for reading.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137555, str_137556))
# Adding element type (key, value) (line 40)
str_137557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 5), 'str', 'long_fields')
str_137558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', 'long_field_names : bool, optional\n   * False - maximum field name length in a structure is 31 characters\n     which is the documented maximum length. This is the default.\n   * True - maximum field name length in a structure is 63 characters\n     which works for MATLAB 7.6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137557, str_137558))
# Adding element type (key, value) (line 40)
str_137559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 5), 'str', 'do_compression')
str_137560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'str', 'do_compression : bool, optional\n   Whether to compress matrices on write. Default is False.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137559, str_137560))
# Adding element type (key, value) (line 40)
str_137561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 5), 'str', 'oned_as')
str_137562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', "oned_as : {'row', 'column'}, optional\n   If 'column', write 1-D numpy arrays as column vectors.\n   If 'row', write 1D numpy arrays as row vectors.")
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137561, str_137562))
# Adding element type (key, value) (line 40)
str_137563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 5), 'str', 'unicode_strings')
str_137564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', 'unicode_strings : bool, optional\n   If True, write strings as Unicode, else MATLAB usual encoding.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), dict_137546, (str_137563, str_137564))

# Assigning a type to the variable 'doc_dict' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'doc_dict', dict_137546)

# Assigning a Call to a Name (line 91):

# Call to filldoc(...): (line 91)
# Processing the call arguments (line 91)
# Getting the type of 'doc_dict' (line 91)
doc_dict_137567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'doc_dict', False)
# Processing the call keyword arguments (line 91)
kwargs_137568 = {}
# Getting the type of 'doccer' (line 91)
doccer_137565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'doccer', False)
# Obtaining the member 'filldoc' of a type (line 91)
filldoc_137566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), doccer_137565, 'filldoc')
# Calling filldoc(args, kwargs) (line 91)
filldoc_call_result_137569 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), filldoc_137566, *[doc_dict_137567], **kwargs_137568)

# Assigning a type to the variable 'docfiller' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'docfiller', filldoc_call_result_137569)
str_137570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', '\n\n Note on architecture\n======================\n\nThere are three sets of parameters relevant for reading files.  The\nfirst are *file read parameters* - containing options that are common\nfor reading the whole file, and therefore every variable within that\nfile. At the moment these are:\n\n* mat_stream\n* dtypes (derived from byte code)\n* byte_order\n* chars_as_strings\n* squeeze_me\n* struct_as_record (MATLAB 5 files)\n* class_dtypes (derived from order code, MATLAB 5 files)\n* codecs (MATLAB 5 files)\n* uint16_codec (MATLAB 5 files)\n\nAnother set of parameters are those that apply only to the current\nvariable being read - the *header*:\n\n* header related variables (different for v4 and v5 mat files)\n* is_complex\n* mclass\n* var_stream\n\nWith the header, we need ``next_position`` to tell us where the next\nvariable in the stream is.\n\nThen, for each element in a matrix, there can be *element read\nparameters*.  An element is, for example, one element in a MATLAB cell\narray.  At the moment these are:\n\n* mat_dtype\n\nThe file-reading object contains the *file read parameters*.  The\n*header* is passed around as a data object, or may be read and discarded\nin a single function.  The *element read parameters* - the mat_dtype in\nthis instance, is passed into a general post-processing function - see\n``mio_utils`` for details.\n')

@norecursion
def convert_dtypes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'convert_dtypes'
    module_type_store = module_type_store.open_function_context('convert_dtypes', 138, 0, False)
    
    # Passed parameters checking function
    convert_dtypes.stypy_localization = localization
    convert_dtypes.stypy_type_of_self = None
    convert_dtypes.stypy_type_store = module_type_store
    convert_dtypes.stypy_function_name = 'convert_dtypes'
    convert_dtypes.stypy_param_names_list = ['dtype_template', 'order_code']
    convert_dtypes.stypy_varargs_param_name = None
    convert_dtypes.stypy_kwargs_param_name = None
    convert_dtypes.stypy_call_defaults = defaults
    convert_dtypes.stypy_call_varargs = varargs
    convert_dtypes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convert_dtypes', ['dtype_template', 'order_code'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convert_dtypes', localization, ['dtype_template', 'order_code'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convert_dtypes(...)' code ##################

    str_137571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, (-1)), 'str', ' Convert dtypes in mapping to given order\n\n    Parameters\n    ----------\n    dtype_template : mapping\n       mapping with values returning numpy dtype from ``np.dtype(val)``\n    order_code : str\n       an order code suitable for using in ``dtype.newbyteorder()``\n\n    Returns\n    -------\n    dtypes : mapping\n       mapping where values have been replaced by\n       ``np.dtype(val).newbyteorder(order_code)``\n\n    ')
    
    # Assigning a Call to a Name (line 155):
    
    # Call to copy(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_137574 = {}
    # Getting the type of 'dtype_template' (line 155)
    dtype_template_137572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 13), 'dtype_template', False)
    # Obtaining the member 'copy' of a type (line 155)
    copy_137573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 13), dtype_template_137572, 'copy')
    # Calling copy(args, kwargs) (line 155)
    copy_call_result_137575 = invoke(stypy.reporting.localization.Localization(__file__, 155, 13), copy_137573, *[], **kwargs_137574)
    
    # Assigning a type to the variable 'dtypes' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'dtypes', copy_call_result_137575)
    
    # Getting the type of 'dtypes' (line 156)
    dtypes_137576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'dtypes')
    # Testing the type of a for loop iterable (line 156)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 4), dtypes_137576)
    # Getting the type of the for loop variable (line 156)
    for_loop_var_137577 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 4), dtypes_137576)
    # Assigning a type to the variable 'k' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'k', for_loop_var_137577)
    # SSA begins for a for statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 157):
    
    # Call to newbyteorder(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'order_code' (line 157)
    order_code_137587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 53), 'order_code', False)
    # Processing the call keyword arguments (line 157)
    kwargs_137588 = {}
    
    # Call to dtype(...): (line 157)
    # Processing the call arguments (line 157)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 157)
    k_137580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'k', False)
    # Getting the type of 'dtypes' (line 157)
    dtypes_137581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'dtypes', False)
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___137582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 29), dtypes_137581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_137583 = invoke(stypy.reporting.localization.Localization(__file__, 157, 29), getitem___137582, k_137580)
    
    # Processing the call keyword arguments (line 157)
    kwargs_137584 = {}
    # Getting the type of 'np' (line 157)
    np_137578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'np', False)
    # Obtaining the member 'dtype' of a type (line 157)
    dtype_137579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 20), np_137578, 'dtype')
    # Calling dtype(args, kwargs) (line 157)
    dtype_call_result_137585 = invoke(stypy.reporting.localization.Localization(__file__, 157, 20), dtype_137579, *[subscript_call_result_137583], **kwargs_137584)
    
    # Obtaining the member 'newbyteorder' of a type (line 157)
    newbyteorder_137586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 20), dtype_call_result_137585, 'newbyteorder')
    # Calling newbyteorder(args, kwargs) (line 157)
    newbyteorder_call_result_137589 = invoke(stypy.reporting.localization.Localization(__file__, 157, 20), newbyteorder_137586, *[order_code_137587], **kwargs_137588)
    
    # Getting the type of 'dtypes' (line 157)
    dtypes_137590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'dtypes')
    # Getting the type of 'k' (line 157)
    k_137591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'k')
    # Storing an element on a container (line 157)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), dtypes_137590, (k_137591, newbyteorder_call_result_137589))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dtypes' (line 158)
    dtypes_137592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'dtypes')
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type', dtypes_137592)
    
    # ################# End of 'convert_dtypes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convert_dtypes' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_137593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137593)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convert_dtypes'
    return stypy_return_type_137593

# Assigning a type to the variable 'convert_dtypes' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'convert_dtypes', convert_dtypes)

@norecursion
def read_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_dtype'
    module_type_store = module_type_store.open_function_context('read_dtype', 161, 0, False)
    
    # Passed parameters checking function
    read_dtype.stypy_localization = localization
    read_dtype.stypy_type_of_self = None
    read_dtype.stypy_type_store = module_type_store
    read_dtype.stypy_function_name = 'read_dtype'
    read_dtype.stypy_param_names_list = ['mat_stream', 'a_dtype']
    read_dtype.stypy_varargs_param_name = None
    read_dtype.stypy_kwargs_param_name = None
    read_dtype.stypy_call_defaults = defaults
    read_dtype.stypy_call_varargs = varargs
    read_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_dtype', ['mat_stream', 'a_dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_dtype', localization, ['mat_stream', 'a_dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_dtype(...)' code ##################

    str_137594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, (-1)), 'str', '\n    Generic get of byte stream data of known type\n\n    Parameters\n    ----------\n    mat_stream : file_like object\n        MATLAB (tm) mat file stream\n    a_dtype : dtype\n        dtype of array to read.  `a_dtype` is assumed to be correct\n        endianness.\n\n    Returns\n    -------\n    arr : ndarray\n        Array of dtype `a_dtype` read from stream.\n\n    ')
    
    # Assigning a Attribute to a Name (line 179):
    # Getting the type of 'a_dtype' (line 179)
    a_dtype_137595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'a_dtype')
    # Obtaining the member 'itemsize' of a type (line 179)
    itemsize_137596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), a_dtype_137595, 'itemsize')
    # Assigning a type to the variable 'num_bytes' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'num_bytes', itemsize_137596)
    
    # Assigning a Call to a Name (line 180):
    
    # Call to ndarray(...): (line 180)
    # Processing the call keyword arguments (line 180)
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_137599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    
    keyword_137600 = tuple_137599
    # Getting the type of 'a_dtype' (line 181)
    a_dtype_137601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 27), 'a_dtype', False)
    keyword_137602 = a_dtype_137601
    
    # Call to read(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'num_bytes' (line 182)
    num_bytes_137605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'num_bytes', False)
    # Processing the call keyword arguments (line 182)
    kwargs_137606 = {}
    # Getting the type of 'mat_stream' (line 182)
    mat_stream_137603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 28), 'mat_stream', False)
    # Obtaining the member 'read' of a type (line 182)
    read_137604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 28), mat_stream_137603, 'read')
    # Calling read(args, kwargs) (line 182)
    read_call_result_137607 = invoke(stypy.reporting.localization.Localization(__file__, 182, 28), read_137604, *[num_bytes_137605], **kwargs_137606)
    
    keyword_137608 = read_call_result_137607
    str_137609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 27), 'str', 'F')
    keyword_137610 = str_137609
    kwargs_137611 = {'buffer': keyword_137608, 'dtype': keyword_137602, 'shape': keyword_137600, 'order': keyword_137610}
    # Getting the type of 'np' (line 180)
    np_137597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 10), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 180)
    ndarray_137598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 10), np_137597, 'ndarray')
    # Calling ndarray(args, kwargs) (line 180)
    ndarray_call_result_137612 = invoke(stypy.reporting.localization.Localization(__file__, 180, 10), ndarray_137598, *[], **kwargs_137611)
    
    # Assigning a type to the variable 'arr' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'arr', ndarray_call_result_137612)
    # Getting the type of 'arr' (line 184)
    arr_137613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type', arr_137613)
    
    # ################# End of 'read_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_137614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137614)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_dtype'
    return stypy_return_type_137614

# Assigning a type to the variable 'read_dtype' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'read_dtype', read_dtype)

@norecursion
def get_matfile_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_matfile_version'
    module_type_store = module_type_store.open_function_context('get_matfile_version', 187, 0, False)
    
    # Passed parameters checking function
    get_matfile_version.stypy_localization = localization
    get_matfile_version.stypy_type_of_self = None
    get_matfile_version.stypy_type_store = module_type_store
    get_matfile_version.stypy_function_name = 'get_matfile_version'
    get_matfile_version.stypy_param_names_list = ['fileobj']
    get_matfile_version.stypy_varargs_param_name = None
    get_matfile_version.stypy_kwargs_param_name = None
    get_matfile_version.stypy_call_defaults = defaults
    get_matfile_version.stypy_call_varargs = varargs
    get_matfile_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_matfile_version', ['fileobj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_matfile_version', localization, ['fileobj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_matfile_version(...)' code ##################

    str_137615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', '\n    Return major, minor tuple depending on apparent mat file type\n\n    Where:\n\n     #. 0,x -> version 4 format mat files\n     #. 1,x -> version 5 format mat files\n     #. 2,x -> version 7.3 format mat files (HDF format)\n\n    Parameters\n    ----------\n    fileobj : file_like\n        object implementing seek() and read()\n\n    Returns\n    -------\n    major_version : {0, 1, 2}\n        major MATLAB File format version\n    minor_version : int\n        minor MATLAB file format version\n\n    Raises\n    ------\n    MatReadError\n        If the file is empty.\n    ValueError\n        The matfile version is unknown.\n\n    Notes\n    -----\n    Has the side effect of setting the file read pointer to 0\n    ')
    
    # Call to seek(...): (line 221)
    # Processing the call arguments (line 221)
    int_137618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 17), 'int')
    # Processing the call keyword arguments (line 221)
    kwargs_137619 = {}
    # Getting the type of 'fileobj' (line 221)
    fileobj_137616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'fileobj', False)
    # Obtaining the member 'seek' of a type (line 221)
    seek_137617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 4), fileobj_137616, 'seek')
    # Calling seek(args, kwargs) (line 221)
    seek_call_result_137620 = invoke(stypy.reporting.localization.Localization(__file__, 221, 4), seek_137617, *[int_137618], **kwargs_137619)
    
    
    # Assigning a Call to a Name (line 222):
    
    # Call to read(...): (line 222)
    # Processing the call arguments (line 222)
    int_137623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 30), 'int')
    # Processing the call keyword arguments (line 222)
    kwargs_137624 = {}
    # Getting the type of 'fileobj' (line 222)
    fileobj_137621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 17), 'fileobj', False)
    # Obtaining the member 'read' of a type (line 222)
    read_137622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 17), fileobj_137621, 'read')
    # Calling read(args, kwargs) (line 222)
    read_call_result_137625 = invoke(stypy.reporting.localization.Localization(__file__, 222, 17), read_137622, *[int_137623], **kwargs_137624)
    
    # Assigning a type to the variable 'mopt_bytes' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'mopt_bytes', read_call_result_137625)
    
    
    
    # Call to len(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'mopt_bytes' (line 223)
    mopt_bytes_137627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'mopt_bytes', False)
    # Processing the call keyword arguments (line 223)
    kwargs_137628 = {}
    # Getting the type of 'len' (line 223)
    len_137626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 7), 'len', False)
    # Calling len(args, kwargs) (line 223)
    len_call_result_137629 = invoke(stypy.reporting.localization.Localization(__file__, 223, 7), len_137626, *[mopt_bytes_137627], **kwargs_137628)
    
    int_137630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 26), 'int')
    # Applying the binary operator '==' (line 223)
    result_eq_137631 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 7), '==', len_call_result_137629, int_137630)
    
    # Testing the type of an if condition (line 223)
    if_condition_137632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 4), result_eq_137631)
    # Assigning a type to the variable 'if_condition_137632' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'if_condition_137632', if_condition_137632)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to MatReadError(...): (line 224)
    # Processing the call arguments (line 224)
    str_137634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'str', 'Mat file appears to be empty')
    # Processing the call keyword arguments (line 224)
    kwargs_137635 = {}
    # Getting the type of 'MatReadError' (line 224)
    MatReadError_137633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'MatReadError', False)
    # Calling MatReadError(args, kwargs) (line 224)
    MatReadError_call_result_137636 = invoke(stypy.reporting.localization.Localization(__file__, 224, 14), MatReadError_137633, *[str_137634], **kwargs_137635)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 224, 8), MatReadError_call_result_137636, 'raise parameter', BaseException)
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 225):
    
    # Call to ndarray(...): (line 225)
    # Processing the call keyword arguments (line 225)
    
    # Obtaining an instance of the builtin type 'tuple' (line 225)
    tuple_137639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 225)
    # Adding element type (line 225)
    int_137640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 34), tuple_137639, int_137640)
    
    keyword_137641 = tuple_137639
    # Getting the type of 'np' (line 225)
    np_137642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 45), 'np', False)
    # Obtaining the member 'uint8' of a type (line 225)
    uint8_137643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 45), np_137642, 'uint8')
    keyword_137644 = uint8_137643
    # Getting the type of 'mopt_bytes' (line 225)
    mopt_bytes_137645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 62), 'mopt_bytes', False)
    keyword_137646 = mopt_bytes_137645
    kwargs_137647 = {'buffer': keyword_137646, 'dtype': keyword_137644, 'shape': keyword_137641}
    # Getting the type of 'np' (line 225)
    np_137637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 225)
    ndarray_137638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), np_137637, 'ndarray')
    # Calling ndarray(args, kwargs) (line 225)
    ndarray_call_result_137648 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), ndarray_137638, *[], **kwargs_137647)
    
    # Assigning a type to the variable 'mopt_ints' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'mopt_ints', ndarray_call_result_137648)
    
    
    int_137649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 7), 'int')
    # Getting the type of 'mopt_ints' (line 226)
    mopt_ints_137650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'mopt_ints')
    # Applying the binary operator 'in' (line 226)
    result_contains_137651 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 7), 'in', int_137649, mopt_ints_137650)
    
    # Testing the type of an if condition (line 226)
    if_condition_137652 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 4), result_contains_137651)
    # Assigning a type to the variable 'if_condition_137652' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'if_condition_137652', if_condition_137652)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to seek(...): (line 227)
    # Processing the call arguments (line 227)
    int_137655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 21), 'int')
    # Processing the call keyword arguments (line 227)
    kwargs_137656 = {}
    # Getting the type of 'fileobj' (line 227)
    fileobj_137653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'fileobj', False)
    # Obtaining the member 'seek' of a type (line 227)
    seek_137654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), fileobj_137653, 'seek')
    # Calling seek(args, kwargs) (line 227)
    seek_call_result_137657 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), seek_137654, *[int_137655], **kwargs_137656)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 228)
    tuple_137658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 228)
    # Adding element type (line 228)
    int_137659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 16), tuple_137658, int_137659)
    # Adding element type (line 228)
    int_137660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 16), tuple_137658, int_137660)
    
    # Assigning a type to the variable 'stypy_return_type' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', tuple_137658)
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to seek(...): (line 232)
    # Processing the call arguments (line 232)
    int_137663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'int')
    # Processing the call keyword arguments (line 232)
    kwargs_137664 = {}
    # Getting the type of 'fileobj' (line 232)
    fileobj_137661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'fileobj', False)
    # Obtaining the member 'seek' of a type (line 232)
    seek_137662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), fileobj_137661, 'seek')
    # Calling seek(args, kwargs) (line 232)
    seek_call_result_137665 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), seek_137662, *[int_137663], **kwargs_137664)
    
    
    # Assigning a Call to a Name (line 233):
    
    # Call to read(...): (line 233)
    # Processing the call arguments (line 233)
    int_137668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 27), 'int')
    # Processing the call keyword arguments (line 233)
    kwargs_137669 = {}
    # Getting the type of 'fileobj' (line 233)
    fileobj_137666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 14), 'fileobj', False)
    # Obtaining the member 'read' of a type (line 233)
    read_137667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 14), fileobj_137666, 'read')
    # Calling read(args, kwargs) (line 233)
    read_call_result_137670 = invoke(stypy.reporting.localization.Localization(__file__, 233, 14), read_137667, *[int_137668], **kwargs_137669)
    
    # Assigning a type to the variable 'tst_str' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'tst_str', read_call_result_137670)
    
    # Call to seek(...): (line 234)
    # Processing the call arguments (line 234)
    int_137673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 17), 'int')
    # Processing the call keyword arguments (line 234)
    kwargs_137674 = {}
    # Getting the type of 'fileobj' (line 234)
    fileobj_137671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'fileobj', False)
    # Obtaining the member 'seek' of a type (line 234)
    seek_137672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 4), fileobj_137671, 'seek')
    # Calling seek(args, kwargs) (line 234)
    seek_call_result_137675 = invoke(stypy.reporting.localization.Localization(__file__, 234, 4), seek_137672, *[int_137673], **kwargs_137674)
    
    
    # Assigning a Call to a Name (line 235):
    
    # Call to int(...): (line 235)
    # Processing the call arguments (line 235)
    
    
    # Obtaining the type of the subscript
    int_137677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'int')
    # Getting the type of 'tst_str' (line 235)
    tst_str_137678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 18), 'tst_str', False)
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___137679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 18), tst_str_137678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_137680 = invoke(stypy.reporting.localization.Localization(__file__, 235, 18), getitem___137679, int_137677)
    
    
    # Obtaining the type of the subscript
    int_137681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 37), 'int')
    str_137682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 32), 'str', 'I')
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___137683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 32), str_137682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_137684 = invoke(stypy.reporting.localization.Localization(__file__, 235, 32), getitem___137683, int_137681)
    
    # Applying the binary operator '==' (line 235)
    result_eq_137685 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 18), '==', subscript_call_result_137680, subscript_call_result_137684)
    
    # Processing the call keyword arguments (line 235)
    kwargs_137686 = {}
    # Getting the type of 'int' (line 235)
    int_137676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 14), 'int', False)
    # Calling int(args, kwargs) (line 235)
    int_call_result_137687 = invoke(stypy.reporting.localization.Localization(__file__, 235, 14), int_137676, *[result_eq_137685], **kwargs_137686)
    
    # Assigning a type to the variable 'maj_ind' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'maj_ind', int_call_result_137687)
    
    # Assigning a Call to a Name (line 236):
    
    # Call to byteord(...): (line 236)
    # Processing the call arguments (line 236)
    
    # Obtaining the type of the subscript
    # Getting the type of 'maj_ind' (line 236)
    maj_ind_137689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'maj_ind', False)
    # Getting the type of 'tst_str' (line 236)
    tst_str_137690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'tst_str', False)
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___137691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 22), tst_str_137690, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_137692 = invoke(stypy.reporting.localization.Localization(__file__, 236, 22), getitem___137691, maj_ind_137689)
    
    # Processing the call keyword arguments (line 236)
    kwargs_137693 = {}
    # Getting the type of 'byteord' (line 236)
    byteord_137688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 'byteord', False)
    # Calling byteord(args, kwargs) (line 236)
    byteord_call_result_137694 = invoke(stypy.reporting.localization.Localization(__file__, 236, 14), byteord_137688, *[subscript_call_result_137692], **kwargs_137693)
    
    # Assigning a type to the variable 'maj_val' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'maj_val', byteord_call_result_137694)
    
    # Assigning a Call to a Name (line 237):
    
    # Call to byteord(...): (line 237)
    # Processing the call arguments (line 237)
    
    # Obtaining the type of the subscript
    int_137696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 30), 'int')
    # Getting the type of 'maj_ind' (line 237)
    maj_ind_137697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'maj_ind', False)
    # Applying the binary operator '-' (line 237)
    result_sub_137698 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 30), '-', int_137696, maj_ind_137697)
    
    # Getting the type of 'tst_str' (line 237)
    tst_str_137699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'tst_str', False)
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___137700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 22), tst_str_137699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_137701 = invoke(stypy.reporting.localization.Localization(__file__, 237, 22), getitem___137700, result_sub_137698)
    
    # Processing the call keyword arguments (line 237)
    kwargs_137702 = {}
    # Getting the type of 'byteord' (line 237)
    byteord_137695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'byteord', False)
    # Calling byteord(args, kwargs) (line 237)
    byteord_call_result_137703 = invoke(stypy.reporting.localization.Localization(__file__, 237, 14), byteord_137695, *[subscript_call_result_137701], **kwargs_137702)
    
    # Assigning a type to the variable 'min_val' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'min_val', byteord_call_result_137703)
    
    # Assigning a Tuple to a Name (line 238):
    
    # Obtaining an instance of the builtin type 'tuple' (line 238)
    tuple_137704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 238)
    # Adding element type (line 238)
    # Getting the type of 'maj_val' (line 238)
    maj_val_137705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'maj_val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_137704, maj_val_137705)
    # Adding element type (line 238)
    # Getting the type of 'min_val' (line 238)
    min_val_137706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'min_val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 11), tuple_137704, min_val_137706)
    
    # Assigning a type to the variable 'ret' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'ret', tuple_137704)
    
    
    # Getting the type of 'maj_val' (line 239)
    maj_val_137707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 7), 'maj_val')
    
    # Obtaining an instance of the builtin type 'tuple' (line 239)
    tuple_137708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 239)
    # Adding element type (line 239)
    int_137709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 19), tuple_137708, int_137709)
    # Adding element type (line 239)
    int_137710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 19), tuple_137708, int_137710)
    
    # Applying the binary operator 'in' (line 239)
    result_contains_137711 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 7), 'in', maj_val_137707, tuple_137708)
    
    # Testing the type of an if condition (line 239)
    if_condition_137712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 4), result_contains_137711)
    # Assigning a type to the variable 'if_condition_137712' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'if_condition_137712', if_condition_137712)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ret' (line 240)
    ret_137713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'stypy_return_type', ret_137713)
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 241)
    # Processing the call arguments (line 241)
    str_137715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 21), 'str', 'Unknown mat file type, version %s, %s')
    # Getting the type of 'ret' (line 241)
    ret_137716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 63), 'ret', False)
    # Applying the binary operator '%' (line 241)
    result_mod_137717 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 21), '%', str_137715, ret_137716)
    
    # Processing the call keyword arguments (line 241)
    kwargs_137718 = {}
    # Getting the type of 'ValueError' (line 241)
    ValueError_137714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 241)
    ValueError_call_result_137719 = invoke(stypy.reporting.localization.Localization(__file__, 241, 10), ValueError_137714, *[result_mod_137717], **kwargs_137718)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 4), ValueError_call_result_137719, 'raise parameter', BaseException)
    
    # ################# End of 'get_matfile_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_matfile_version' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_137720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137720)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_matfile_version'
    return stypy_return_type_137720

# Assigning a type to the variable 'get_matfile_version' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'get_matfile_version', get_matfile_version)

@norecursion
def matdims(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_137721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 25), 'str', 'column')
    defaults = [str_137721]
    # Create a new context for function 'matdims'
    module_type_store = module_type_store.open_function_context('matdims', 244, 0, False)
    
    # Passed parameters checking function
    matdims.stypy_localization = localization
    matdims.stypy_type_of_self = None
    matdims.stypy_type_store = module_type_store
    matdims.stypy_function_name = 'matdims'
    matdims.stypy_param_names_list = ['arr', 'oned_as']
    matdims.stypy_varargs_param_name = None
    matdims.stypy_kwargs_param_name = None
    matdims.stypy_call_defaults = defaults
    matdims.stypy_call_varargs = varargs
    matdims.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matdims', ['arr', 'oned_as'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matdims', localization, ['arr', 'oned_as'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matdims(...)' code ##################

    str_137722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, (-1)), 'str', '\n    Determine equivalent MATLAB dimensions for given array\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array\n    oned_as : {\'column\', \'row\'}, optional\n        Whether 1-D arrays are returned as MATLAB row or column matrices.\n        Default is \'column\'.\n\n    Returns\n    -------\n    dims : tuple\n        Shape tuple, in the form MATLAB expects it.\n\n    Notes\n    -----\n    We had to decide what shape a 1 dimensional array would be by\n    default.  ``np.atleast_2d`` thinks it is a row vector.  The\n    default for a vector in MATLAB (e.g. ``>> 1:12``) is a row vector.\n\n    Versions of scipy up to and including 0.11 resulted (accidentally)\n    in 1-D arrays being read as column vectors.  For the moment, we\n    maintain the same tradition here.\n\n    Examples\n    --------\n    >>> matdims(np.array(1)) # numpy scalar\n    (1, 1)\n    >>> matdims(np.array([1])) # 1d array, 1 element\n    (1, 1)\n    >>> matdims(np.array([1,2])) # 1d array, 2 elements\n    (2, 1)\n    >>> matdims(np.array([[2],[3]])) # 2d array, column vector\n    (2, 1)\n    >>> matdims(np.array([[2,3]])) # 2d array, row vector\n    (1, 2)\n    >>> matdims(np.array([[[2,3]]])) # 3d array, rowish vector\n    (1, 1, 2)\n    >>> matdims(np.array([])) # empty 1d array\n    (0, 0)\n    >>> matdims(np.array([[]])) # empty 2d\n    (0, 0)\n    >>> matdims(np.array([[[]]])) # empty 3d\n    (0, 0, 0)\n\n    Optional argument flips 1-D shape behavior.\n\n    >>> matdims(np.array([1,2]), \'row\') # 1d array, 2 elements\n    (1, 2)\n\n    The argument has to make sense though\n\n    >>> matdims(np.array([1,2]), \'bizarre\')\n    Traceback (most recent call last):\n       ...\n    ValueError: 1D option "bizarre" is strange\n\n    ')
    
    # Assigning a Attribute to a Name (line 305):
    # Getting the type of 'arr' (line 305)
    arr_137723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'arr')
    # Obtaining the member 'shape' of a type (line 305)
    shape_137724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), arr_137723, 'shape')
    # Assigning a type to the variable 'shape' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'shape', shape_137724)
    
    
    # Getting the type of 'shape' (line 306)
    shape_137725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 7), 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 306)
    tuple_137726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 306)
    
    # Applying the binary operator '==' (line 306)
    result_eq_137727 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 7), '==', shape_137725, tuple_137726)
    
    # Testing the type of an if condition (line 306)
    if_condition_137728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 4), result_eq_137727)
    # Assigning a type to the variable 'if_condition_137728' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'if_condition_137728', if_condition_137728)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 307)
    tuple_137729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 307)
    # Adding element type (line 307)
    int_137730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 16), tuple_137729, int_137730)
    # Adding element type (line 307)
    int_137731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 16), tuple_137729, int_137731)
    
    # Assigning a type to the variable 'stypy_return_type' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', tuple_137729)
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to reduce(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'operator' (line 308)
    operator_137733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 14), 'operator', False)
    # Obtaining the member 'mul' of a type (line 308)
    mul_137734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 14), operator_137733, 'mul')
    # Getting the type of 'shape' (line 308)
    shape_137735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'shape', False)
    # Processing the call keyword arguments (line 308)
    kwargs_137736 = {}
    # Getting the type of 'reduce' (line 308)
    reduce_137732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 7), 'reduce', False)
    # Calling reduce(args, kwargs) (line 308)
    reduce_call_result_137737 = invoke(stypy.reporting.localization.Localization(__file__, 308, 7), reduce_137732, *[mul_137734, shape_137735], **kwargs_137736)
    
    int_137738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 38), 'int')
    # Applying the binary operator '==' (line 308)
    result_eq_137739 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 7), '==', reduce_call_result_137737, int_137738)
    
    # Testing the type of an if condition (line 308)
    if_condition_137740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 4), result_eq_137739)
    # Assigning a type to the variable 'if_condition_137740' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'if_condition_137740', if_condition_137740)
    # SSA begins for if statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 309)
    tuple_137741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 309)
    # Adding element type (line 309)
    int_137742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 16), tuple_137741, int_137742)
    
    
    # Call to max(...): (line 309)
    # Processing the call arguments (line 309)
    
    # Obtaining an instance of the builtin type 'list' (line 309)
    list_137745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 309)
    # Adding element type (line 309)
    # Getting the type of 'arr' (line 309)
    arr_137746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 30), 'arr', False)
    # Obtaining the member 'ndim' of a type (line 309)
    ndim_137747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 30), arr_137746, 'ndim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 29), list_137745, ndim_137747)
    # Adding element type (line 309)
    int_137748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 29), list_137745, int_137748)
    
    # Processing the call keyword arguments (line 309)
    kwargs_137749 = {}
    # Getting the type of 'np' (line 309)
    np_137743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 22), 'np', False)
    # Obtaining the member 'max' of a type (line 309)
    max_137744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 22), np_137743, 'max')
    # Calling max(args, kwargs) (line 309)
    max_call_result_137750 = invoke(stypy.reporting.localization.Localization(__file__, 309, 22), max_137744, *[list_137745], **kwargs_137749)
    
    # Applying the binary operator '*' (line 309)
    result_mul_137751 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 15), '*', tuple_137741, max_call_result_137750)
    
    # Assigning a type to the variable 'stypy_return_type' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type', result_mul_137751)
    # SSA join for if statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'shape' (line 310)
    shape_137753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'shape', False)
    # Processing the call keyword arguments (line 310)
    kwargs_137754 = {}
    # Getting the type of 'len' (line 310)
    len_137752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 7), 'len', False)
    # Calling len(args, kwargs) (line 310)
    len_call_result_137755 = invoke(stypy.reporting.localization.Localization(__file__, 310, 7), len_137752, *[shape_137753], **kwargs_137754)
    
    int_137756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'int')
    # Applying the binary operator '==' (line 310)
    result_eq_137757 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 7), '==', len_call_result_137755, int_137756)
    
    # Testing the type of an if condition (line 310)
    if_condition_137758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 4), result_eq_137757)
    # Assigning a type to the variable 'if_condition_137758' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'if_condition_137758', if_condition_137758)
    # SSA begins for if statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'oned_as' (line 311)
    oned_as_137759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'oned_as')
    str_137760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 22), 'str', 'column')
    # Applying the binary operator '==' (line 311)
    result_eq_137761 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 11), '==', oned_as_137759, str_137760)
    
    # Testing the type of an if condition (line 311)
    if_condition_137762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), result_eq_137761)
    # Assigning a type to the variable 'if_condition_137762' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_137762', if_condition_137762)
    # SSA begins for if statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'shape' (line 312)
    shape_137763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 312)
    tuple_137764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 312)
    # Adding element type (line 312)
    int_137765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 28), tuple_137764, int_137765)
    
    # Applying the binary operator '+' (line 312)
    result_add_137766 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 19), '+', shape_137763, tuple_137764)
    
    # Assigning a type to the variable 'stypy_return_type' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'stypy_return_type', result_add_137766)
    # SSA branch for the else part of an if statement (line 311)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'oned_as' (line 313)
    oned_as_137767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'oned_as')
    str_137768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 24), 'str', 'row')
    # Applying the binary operator '==' (line 313)
    result_eq_137769 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 13), '==', oned_as_137767, str_137768)
    
    # Testing the type of an if condition (line 313)
    if_condition_137770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 13), result_eq_137769)
    # Assigning a type to the variable 'if_condition_137770' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'if_condition_137770', if_condition_137770)
    # SSA begins for if statement (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 314)
    tuple_137771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 314)
    # Adding element type (line 314)
    int_137772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 20), tuple_137771, int_137772)
    
    # Getting the type of 'shape' (line 314)
    shape_137773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'shape')
    # Applying the binary operator '+' (line 314)
    result_add_137774 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 19), '+', tuple_137771, shape_137773)
    
    # Assigning a type to the variable 'stypy_return_type' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'stypy_return_type', result_add_137774)
    # SSA branch for the else part of an if statement (line 313)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 316)
    # Processing the call arguments (line 316)
    str_137776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 29), 'str', '1D option "%s" is strange')
    # Getting the type of 'oned_as' (line 317)
    oned_as_137777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'oned_as', False)
    # Applying the binary operator '%' (line 316)
    result_mod_137778 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 29), '%', str_137776, oned_as_137777)
    
    # Processing the call keyword arguments (line 316)
    kwargs_137779 = {}
    # Getting the type of 'ValueError' (line 316)
    ValueError_137775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 316)
    ValueError_call_result_137780 = invoke(stypy.reporting.localization.Localization(__file__, 316, 18), ValueError_137775, *[result_mod_137778], **kwargs_137779)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 316, 12), ValueError_call_result_137780, 'raise parameter', BaseException)
    # SSA join for if statement (line 313)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 311)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'shape' (line 318)
    shape_137781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'shape')
    # Assigning a type to the variable 'stypy_return_type' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type', shape_137781)
    
    # ################# End of 'matdims(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matdims' in the type store
    # Getting the type of 'stypy_return_type' (line 244)
    stypy_return_type_137782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137782)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matdims'
    return stypy_return_type_137782

# Assigning a type to the variable 'matdims' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'matdims', matdims)
# Declaration of the 'MatVarReader' class

class MatVarReader(object, ):
    str_137783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 4), 'str', ' Abstract class defining required interface for var readers')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatVarReader.__init__', ['file_reader'], None, None, defaults, varargs, kwargs)

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

        pass
        
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
        module_type_store = module_type_store.open_function_context('read_header', 326, 4, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatVarReader.read_header.__dict__.__setitem__('stypy_localization', localization)
        MatVarReader.read_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatVarReader.read_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatVarReader.read_header.__dict__.__setitem__('stypy_function_name', 'MatVarReader.read_header')
        MatVarReader.read_header.__dict__.__setitem__('stypy_param_names_list', [])
        MatVarReader.read_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatVarReader.read_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatVarReader.read_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatVarReader.read_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatVarReader.read_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatVarReader.read_header.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatVarReader.read_header', [], None, None, defaults, varargs, kwargs)

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

        str_137784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'str', ' Returns header ')
        pass
        
        # ################# End of 'read_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_header' in the type store
        # Getting the type of 'stypy_return_type' (line 326)
        stypy_return_type_137785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137785)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_header'
        return stypy_return_type_137785


    @norecursion
    def array_from_header(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'array_from_header'
        module_type_store = module_type_store.open_function_context('array_from_header', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_localization', localization)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_function_name', 'MatVarReader.array_from_header')
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_param_names_list', ['header'])
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatVarReader.array_from_header.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatVarReader.array_from_header', ['header'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'array_from_header', localization, ['header'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'array_from_header(...)' code ##################

        str_137786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 8), 'str', ' Reads array given header ')
        pass
        
        # ################# End of 'array_from_header(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'array_from_header' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_137787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'array_from_header'
        return stypy_return_type_137787


# Assigning a type to the variable 'MatVarReader' (line 321)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'MatVarReader', MatVarReader)
# Declaration of the 'MatFileReader' class

class MatFileReader(object, ):
    str_137788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, (-1)), 'str', ' Base object for reading mat files\n\n    To make this class functional, you will need to override the\n    following methods:\n\n    matrix_getter_factory   - gives object to fetch next matrix from stream\n    guess_byte_order        - guesses file byte order from file\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 347)
        None_137789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 'None')
        # Getting the type of 'False' (line 348)
        False_137790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'False')
        # Getting the type of 'False' (line 349)
        False_137791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 28), 'False')
        # Getting the type of 'True' (line 350)
        True_137792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 34), 'True')
        # Getting the type of 'False' (line 351)
        False_137793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 35), 'False')
        # Getting the type of 'True' (line 352)
        True_137794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 34), 'True')
        # Getting the type of 'True' (line 353)
        True_137795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 50), 'True')
        defaults = [None_137789, False_137790, False_137791, True_137792, False_137793, True_137794, True_137795]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFileReader.__init__', ['mat_stream', 'byte_order', 'mat_dtype', 'squeeze_me', 'chars_as_strings', 'matlab_compatible', 'struct_as_record', 'verify_compressed_data_integrity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['mat_stream', 'byte_order', 'mat_dtype', 'squeeze_me', 'chars_as_strings', 'matlab_compatible', 'struct_as_record', 'verify_compressed_data_integrity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_137796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, (-1)), 'str', '\n        Initializer for mat file reader\n\n        mat_stream : file-like\n            object with file API, open for reading\n    %(load_args)s\n        ')
        
        # Assigning a Name to a Attribute (line 363):
        # Getting the type of 'mat_stream' (line 363)
        mat_stream_137797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'mat_stream')
        # Getting the type of 'self' (line 363)
        self_137798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self')
        # Setting the type of the member 'mat_stream' of a type (line 363)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), self_137798, 'mat_stream', mat_stream_137797)
        
        # Assigning a Dict to a Attribute (line 364):
        
        # Obtaining an instance of the builtin type 'dict' (line 364)
        dict_137799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 364)
        
        # Getting the type of 'self' (line 364)
        self_137800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'self')
        # Setting the type of the member 'dtypes' of a type (line 364)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 8), self_137800, 'dtypes', dict_137799)
        
        
        # Getting the type of 'byte_order' (line 365)
        byte_order_137801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'byte_order')
        # Applying the 'not' unary operator (line 365)
        result_not__137802 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 11), 'not', byte_order_137801)
        
        # Testing the type of an if condition (line 365)
        if_condition_137803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 8), result_not__137802)
        # Assigning a type to the variable 'if_condition_137803' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'if_condition_137803', if_condition_137803)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 366):
        
        # Call to guess_byte_order(...): (line 366)
        # Processing the call keyword arguments (line 366)
        kwargs_137806 = {}
        # Getting the type of 'self' (line 366)
        self_137804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 25), 'self', False)
        # Obtaining the member 'guess_byte_order' of a type (line 366)
        guess_byte_order_137805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 25), self_137804, 'guess_byte_order')
        # Calling guess_byte_order(args, kwargs) (line 366)
        guess_byte_order_call_result_137807 = invoke(stypy.reporting.localization.Localization(__file__, 366, 25), guess_byte_order_137805, *[], **kwargs_137806)
        
        # Assigning a type to the variable 'byte_order' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'byte_order', guess_byte_order_call_result_137807)
        # SSA branch for the else part of an if statement (line 365)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 368):
        
        # Call to to_numpy_code(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'byte_order' (line 368)
        byte_order_137810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 43), 'byte_order', False)
        # Processing the call keyword arguments (line 368)
        kwargs_137811 = {}
        # Getting the type of 'boc' (line 368)
        boc_137808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), 'boc', False)
        # Obtaining the member 'to_numpy_code' of a type (line 368)
        to_numpy_code_137809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 25), boc_137808, 'to_numpy_code')
        # Calling to_numpy_code(args, kwargs) (line 368)
        to_numpy_code_call_result_137812 = invoke(stypy.reporting.localization.Localization(__file__, 368, 25), to_numpy_code_137809, *[byte_order_137810], **kwargs_137811)
        
        # Assigning a type to the variable 'byte_order' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'byte_order', to_numpy_code_call_result_137812)
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 369):
        # Getting the type of 'byte_order' (line 369)
        byte_order_137813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 26), 'byte_order')
        # Getting the type of 'self' (line 369)
        self_137814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self')
        # Setting the type of the member 'byte_order' of a type (line 369)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_137814, 'byte_order', byte_order_137813)
        
        # Assigning a Name to a Attribute (line 370):
        # Getting the type of 'struct_as_record' (line 370)
        struct_as_record_137815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 'struct_as_record')
        # Getting the type of 'self' (line 370)
        self_137816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'self')
        # Setting the type of the member 'struct_as_record' of a type (line 370)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), self_137816, 'struct_as_record', struct_as_record_137815)
        
        # Getting the type of 'matlab_compatible' (line 371)
        matlab_compatible_137817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'matlab_compatible')
        # Testing the type of an if condition (line 371)
        if_condition_137818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 8), matlab_compatible_137817)
        # Assigning a type to the variable 'if_condition_137818' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'if_condition_137818', if_condition_137818)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_matlab_compatible(...): (line 372)
        # Processing the call keyword arguments (line 372)
        kwargs_137821 = {}
        # Getting the type of 'self' (line 372)
        self_137819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'self', False)
        # Obtaining the member 'set_matlab_compatible' of a type (line 372)
        set_matlab_compatible_137820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), self_137819, 'set_matlab_compatible')
        # Calling set_matlab_compatible(args, kwargs) (line 372)
        set_matlab_compatible_call_result_137822 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), set_matlab_compatible_137820, *[], **kwargs_137821)
        
        # SSA branch for the else part of an if statement (line 371)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 374):
        # Getting the type of 'squeeze_me' (line 374)
        squeeze_me_137823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'squeeze_me')
        # Getting the type of 'self' (line 374)
        self_137824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'self')
        # Setting the type of the member 'squeeze_me' of a type (line 374)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), self_137824, 'squeeze_me', squeeze_me_137823)
        
        # Assigning a Name to a Attribute (line 375):
        # Getting the type of 'chars_as_strings' (line 375)
        chars_as_strings_137825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 36), 'chars_as_strings')
        # Getting the type of 'self' (line 375)
        self_137826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'self')
        # Setting the type of the member 'chars_as_strings' of a type (line 375)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 12), self_137826, 'chars_as_strings', chars_as_strings_137825)
        
        # Assigning a Name to a Attribute (line 376):
        # Getting the type of 'mat_dtype' (line 376)
        mat_dtype_137827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 29), 'mat_dtype')
        # Getting the type of 'self' (line 376)
        self_137828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'self')
        # Setting the type of the member 'mat_dtype' of a type (line 376)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), self_137828, 'mat_dtype', mat_dtype_137827)
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 377):
        # Getting the type of 'verify_compressed_data_integrity' (line 377)
        verify_compressed_data_integrity_137829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 48), 'verify_compressed_data_integrity')
        # Getting the type of 'self' (line 377)
        self_137830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'self')
        # Setting the type of the member 'verify_compressed_data_integrity' of a type (line 377)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), self_137830, 'verify_compressed_data_integrity', verify_compressed_data_integrity_137829)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_matlab_compatible(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_matlab_compatible'
        module_type_store = module_type_store.open_function_context('set_matlab_compatible', 379, 4, False)
        # Assigning a type to the variable 'self' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_localization', localization)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_function_name', 'MatFileReader.set_matlab_compatible')
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_param_names_list', [])
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFileReader.set_matlab_compatible.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFileReader.set_matlab_compatible', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_matlab_compatible', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_matlab_compatible(...)' code ##################

        str_137831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 8), 'str', ' Sets options to return arrays as MATLAB loads them ')
        
        # Assigning a Name to a Attribute (line 381):
        # Getting the type of 'True' (line 381)
        True_137832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 25), 'True')
        # Getting the type of 'self' (line 381)
        self_137833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'self')
        # Setting the type of the member 'mat_dtype' of a type (line 381)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), self_137833, 'mat_dtype', True_137832)
        
        # Assigning a Name to a Attribute (line 382):
        # Getting the type of 'False' (line 382)
        False_137834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 26), 'False')
        # Getting the type of 'self' (line 382)
        self_137835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self')
        # Setting the type of the member 'squeeze_me' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_137835, 'squeeze_me', False_137834)
        
        # Assigning a Name to a Attribute (line 383):
        # Getting the type of 'False' (line 383)
        False_137836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 32), 'False')
        # Getting the type of 'self' (line 383)
        self_137837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'chars_as_strings' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_137837, 'chars_as_strings', False_137836)
        
        # ################# End of 'set_matlab_compatible(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_matlab_compatible' in the type store
        # Getting the type of 'stypy_return_type' (line 379)
        stypy_return_type_137838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137838)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_matlab_compatible'
        return stypy_return_type_137838


    @norecursion
    def guess_byte_order(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'guess_byte_order'
        module_type_store = module_type_store.open_function_context('guess_byte_order', 385, 4, False)
        # Assigning a type to the variable 'self' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_localization', localization)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_function_name', 'MatFileReader.guess_byte_order')
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_param_names_list', [])
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFileReader.guess_byte_order.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFileReader.guess_byte_order', [], None, None, defaults, varargs, kwargs)

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

        str_137839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 8), 'str', ' As we do not know what file type we have, assume native ')
        # Getting the type of 'boc' (line 387)
        boc_137840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 15), 'boc')
        # Obtaining the member 'native_code' of a type (line 387)
        native_code_137841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 15), boc_137840, 'native_code')
        # Assigning a type to the variable 'stypy_return_type' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'stypy_return_type', native_code_137841)
        
        # ################# End of 'guess_byte_order(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'guess_byte_order' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_137842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'guess_byte_order'
        return stypy_return_type_137842


    @norecursion
    def end_of_stream(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'end_of_stream'
        module_type_store = module_type_store.open_function_context('end_of_stream', 389, 4, False)
        # Assigning a type to the variable 'self' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_localization', localization)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_function_name', 'MatFileReader.end_of_stream')
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_param_names_list', [])
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatFileReader.end_of_stream.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatFileReader.end_of_stream', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'end_of_stream', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'end_of_stream(...)' code ##################

        
        # Assigning a Call to a Name (line 390):
        
        # Call to read(...): (line 390)
        # Processing the call arguments (line 390)
        int_137846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 33), 'int')
        # Processing the call keyword arguments (line 390)
        kwargs_137847 = {}
        # Getting the type of 'self' (line 390)
        self_137843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 390)
        mat_stream_137844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), self_137843, 'mat_stream')
        # Obtaining the member 'read' of a type (line 390)
        read_137845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), mat_stream_137844, 'read')
        # Calling read(args, kwargs) (line 390)
        read_call_result_137848 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), read_137845, *[int_137846], **kwargs_137847)
        
        # Assigning a type to the variable 'b' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'b', read_call_result_137848)
        
        # Assigning a Call to a Name (line 391):
        
        # Call to tell(...): (line 391)
        # Processing the call keyword arguments (line 391)
        kwargs_137852 = {}
        # Getting the type of 'self' (line 391)
        self_137849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 391)
        mat_stream_137850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 17), self_137849, 'mat_stream')
        # Obtaining the member 'tell' of a type (line 391)
        tell_137851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 17), mat_stream_137850, 'tell')
        # Calling tell(args, kwargs) (line 391)
        tell_call_result_137853 = invoke(stypy.reporting.localization.Localization(__file__, 391, 17), tell_137851, *[], **kwargs_137852)
        
        # Assigning a type to the variable 'curpos' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'curpos', tell_call_result_137853)
        
        # Call to seek(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'curpos' (line 392)
        curpos_137857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 29), 'curpos', False)
        int_137858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 36), 'int')
        # Applying the binary operator '-' (line 392)
        result_sub_137859 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 29), '-', curpos_137857, int_137858)
        
        # Processing the call keyword arguments (line 392)
        kwargs_137860 = {}
        # Getting the type of 'self' (line 392)
        self_137854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self', False)
        # Obtaining the member 'mat_stream' of a type (line 392)
        mat_stream_137855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_137854, 'mat_stream')
        # Obtaining the member 'seek' of a type (line 392)
        seek_137856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), mat_stream_137855, 'seek')
        # Calling seek(args, kwargs) (line 392)
        seek_call_result_137861 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), seek_137856, *[result_sub_137859], **kwargs_137860)
        
        
        
        # Call to len(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'b' (line 393)
        b_137863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 19), 'b', False)
        # Processing the call keyword arguments (line 393)
        kwargs_137864 = {}
        # Getting the type of 'len' (line 393)
        len_137862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'len', False)
        # Calling len(args, kwargs) (line 393)
        len_call_result_137865 = invoke(stypy.reporting.localization.Localization(__file__, 393, 15), len_137862, *[b_137863], **kwargs_137864)
        
        int_137866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 25), 'int')
        # Applying the binary operator '==' (line 393)
        result_eq_137867 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 15), '==', len_call_result_137865, int_137866)
        
        # Assigning a type to the variable 'stypy_return_type' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'stypy_return_type', result_eq_137867)
        
        # ################# End of 'end_of_stream(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'end_of_stream' in the type store
        # Getting the type of 'stypy_return_type' (line 389)
        stypy_return_type_137868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137868)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'end_of_stream'
        return stypy_return_type_137868


# Assigning a type to the variable 'MatFileReader' (line 335)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'MatFileReader', MatFileReader)

@norecursion
def arr_dtype_number(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arr_dtype_number'
    module_type_store = module_type_store.open_function_context('arr_dtype_number', 396, 0, False)
    
    # Passed parameters checking function
    arr_dtype_number.stypy_localization = localization
    arr_dtype_number.stypy_type_of_self = None
    arr_dtype_number.stypy_type_store = module_type_store
    arr_dtype_number.stypy_function_name = 'arr_dtype_number'
    arr_dtype_number.stypy_param_names_list = ['arr', 'num']
    arr_dtype_number.stypy_varargs_param_name = None
    arr_dtype_number.stypy_kwargs_param_name = None
    arr_dtype_number.stypy_call_defaults = defaults
    arr_dtype_number.stypy_call_varargs = varargs
    arr_dtype_number.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arr_dtype_number', ['arr', 'num'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arr_dtype_number', localization, ['arr', 'num'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arr_dtype_number(...)' code ##################

    str_137869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 4), 'str', ' Return dtype for given number of items per element')
    
    # Call to dtype(...): (line 398)
    # Processing the call arguments (line 398)
    
    # Obtaining the type of the subscript
    int_137872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 35), 'int')
    slice_137873 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 398, 20), None, int_137872, None)
    # Getting the type of 'arr' (line 398)
    arr_137874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 398)
    dtype_137875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 20), arr_137874, 'dtype')
    # Obtaining the member 'str' of a type (line 398)
    str_137876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 20), dtype_137875, 'str')
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___137877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 20), str_137876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_137878 = invoke(stypy.reporting.localization.Localization(__file__, 398, 20), getitem___137877, slice_137873)
    
    
    # Call to str(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'num' (line 398)
    num_137880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 44), 'num', False)
    # Processing the call keyword arguments (line 398)
    kwargs_137881 = {}
    # Getting the type of 'str' (line 398)
    str_137879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 40), 'str', False)
    # Calling str(args, kwargs) (line 398)
    str_call_result_137882 = invoke(stypy.reporting.localization.Localization(__file__, 398, 40), str_137879, *[num_137880], **kwargs_137881)
    
    # Applying the binary operator '+' (line 398)
    result_add_137883 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 20), '+', subscript_call_result_137878, str_call_result_137882)
    
    # Processing the call keyword arguments (line 398)
    kwargs_137884 = {}
    # Getting the type of 'np' (line 398)
    np_137870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 11), 'np', False)
    # Obtaining the member 'dtype' of a type (line 398)
    dtype_137871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 11), np_137870, 'dtype')
    # Calling dtype(args, kwargs) (line 398)
    dtype_call_result_137885 = invoke(stypy.reporting.localization.Localization(__file__, 398, 11), dtype_137871, *[result_add_137883], **kwargs_137884)
    
    # Assigning a type to the variable 'stypy_return_type' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'stypy_return_type', dtype_call_result_137885)
    
    # ################# End of 'arr_dtype_number(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arr_dtype_number' in the type store
    # Getting the type of 'stypy_return_type' (line 396)
    stypy_return_type_137886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arr_dtype_number'
    return stypy_return_type_137886

# Assigning a type to the variable 'arr_dtype_number' (line 396)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 0), 'arr_dtype_number', arr_dtype_number)

@norecursion
def arr_to_chars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arr_to_chars'
    module_type_store = module_type_store.open_function_context('arr_to_chars', 401, 0, False)
    
    # Passed parameters checking function
    arr_to_chars.stypy_localization = localization
    arr_to_chars.stypy_type_of_self = None
    arr_to_chars.stypy_type_store = module_type_store
    arr_to_chars.stypy_function_name = 'arr_to_chars'
    arr_to_chars.stypy_param_names_list = ['arr']
    arr_to_chars.stypy_varargs_param_name = None
    arr_to_chars.stypy_kwargs_param_name = None
    arr_to_chars.stypy_call_defaults = defaults
    arr_to_chars.stypy_call_varargs = varargs
    arr_to_chars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arr_to_chars', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arr_to_chars', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arr_to_chars(...)' code ##################

    str_137887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 4), 'str', ' Convert string array to char array ')
    
    # Assigning a Call to a Name (line 403):
    
    # Call to list(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'arr' (line 403)
    arr_137889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'arr', False)
    # Obtaining the member 'shape' of a type (line 403)
    shape_137890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), arr_137889, 'shape')
    # Processing the call keyword arguments (line 403)
    kwargs_137891 = {}
    # Getting the type of 'list' (line 403)
    list_137888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'list', False)
    # Calling list(args, kwargs) (line 403)
    list_call_result_137892 = invoke(stypy.reporting.localization.Localization(__file__, 403, 11), list_137888, *[shape_137890], **kwargs_137891)
    
    # Assigning a type to the variable 'dims' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'dims', list_call_result_137892)
    
    
    # Getting the type of 'dims' (line 404)
    dims_137893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 11), 'dims')
    # Applying the 'not' unary operator (line 404)
    result_not__137894 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 7), 'not', dims_137893)
    
    # Testing the type of an if condition (line 404)
    if_condition_137895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 4), result_not__137894)
    # Assigning a type to the variable 'if_condition_137895' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'if_condition_137895', if_condition_137895)
    # SSA begins for if statement (line 404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 405):
    
    # Obtaining an instance of the builtin type 'list' (line 405)
    list_137896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 405)
    # Adding element type (line 405)
    int_137897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 15), list_137896, int_137897)
    
    # Assigning a type to the variable 'dims' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'dims', list_137896)
    # SSA join for if statement (line 404)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 406)
    # Processing the call arguments (line 406)
    
    # Call to int(...): (line 406)
    # Processing the call arguments (line 406)
    
    # Obtaining the type of the subscript
    int_137901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 34), 'int')
    slice_137902 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 406, 20), int_137901, None, None)
    # Getting the type of 'arr' (line 406)
    arr_137903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 406)
    dtype_137904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), arr_137903, 'dtype')
    # Obtaining the member 'str' of a type (line 406)
    str_137905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), dtype_137904, 'str')
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___137906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 20), str_137905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_137907 = invoke(stypy.reporting.localization.Localization(__file__, 406, 20), getitem___137906, slice_137902)
    
    # Processing the call keyword arguments (line 406)
    kwargs_137908 = {}
    # Getting the type of 'int' (line 406)
    int_137900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'int', False)
    # Calling int(args, kwargs) (line 406)
    int_call_result_137909 = invoke(stypy.reporting.localization.Localization(__file__, 406, 16), int_137900, *[subscript_call_result_137907], **kwargs_137908)
    
    # Processing the call keyword arguments (line 406)
    kwargs_137910 = {}
    # Getting the type of 'dims' (line 406)
    dims_137898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'dims', False)
    # Obtaining the member 'append' of a type (line 406)
    append_137899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), dims_137898, 'append')
    # Calling append(args, kwargs) (line 406)
    append_call_result_137911 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), append_137899, *[int_call_result_137909], **kwargs_137910)
    
    
    # Assigning a Call to a Name (line 407):
    
    # Call to ndarray(...): (line 407)
    # Processing the call keyword arguments (line 407)
    # Getting the type of 'dims' (line 407)
    dims_137914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 27), 'dims', False)
    keyword_137915 = dims_137914
    
    # Call to arr_dtype_number(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'arr' (line 408)
    arr_137917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 44), 'arr', False)
    int_137918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 49), 'int')
    # Processing the call keyword arguments (line 408)
    kwargs_137919 = {}
    # Getting the type of 'arr_dtype_number' (line 408)
    arr_dtype_number_137916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 27), 'arr_dtype_number', False)
    # Calling arr_dtype_number(args, kwargs) (line 408)
    arr_dtype_number_call_result_137920 = invoke(stypy.reporting.localization.Localization(__file__, 408, 27), arr_dtype_number_137916, *[arr_137917, int_137918], **kwargs_137919)
    
    keyword_137921 = arr_dtype_number_call_result_137920
    # Getting the type of 'arr' (line 409)
    arr_137922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'arr', False)
    keyword_137923 = arr_137922
    kwargs_137924 = {'buffer': keyword_137923, 'dtype': keyword_137921, 'shape': keyword_137915}
    # Getting the type of 'np' (line 407)
    np_137912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 10), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 407)
    ndarray_137913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 10), np_137912, 'ndarray')
    # Calling ndarray(args, kwargs) (line 407)
    ndarray_call_result_137925 = invoke(stypy.reporting.localization.Localization(__file__, 407, 10), ndarray_137913, *[], **kwargs_137924)
    
    # Assigning a type to the variable 'arr' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'arr', ndarray_call_result_137925)
    
    # Assigning a List to a Name (line 410):
    
    # Obtaining an instance of the builtin type 'list' (line 410)
    list_137926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 410)
    # Adding element type (line 410)
    
    # Getting the type of 'arr' (line 410)
    arr_137927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'arr')
    str_137928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 22), 'str', '')
    # Applying the binary operator '==' (line 410)
    result_eq_137929 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 15), '==', arr_137927, str_137928)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 14), list_137926, result_eq_137929)
    
    # Assigning a type to the variable 'empties' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'empties', list_137926)
    
    
    
    # Call to any(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'empties' (line 411)
    empties_137932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 18), 'empties', False)
    # Processing the call keyword arguments (line 411)
    kwargs_137933 = {}
    # Getting the type of 'np' (line 411)
    np_137930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 411)
    any_137931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 11), np_137930, 'any')
    # Calling any(args, kwargs) (line 411)
    any_call_result_137934 = invoke(stypy.reporting.localization.Localization(__file__, 411, 11), any_137931, *[empties_137932], **kwargs_137933)
    
    # Applying the 'not' unary operator (line 411)
    result_not__137935 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 7), 'not', any_call_result_137934)
    
    # Testing the type of an if condition (line 411)
    if_condition_137936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), result_not__137935)
    # Assigning a type to the variable 'if_condition_137936' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'if_condition_137936', if_condition_137936)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 412)
    arr_137937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'stypy_return_type', arr_137937)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 413):
    
    # Call to copy(...): (line 413)
    # Processing the call keyword arguments (line 413)
    kwargs_137940 = {}
    # Getting the type of 'arr' (line 413)
    arr_137938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 10), 'arr', False)
    # Obtaining the member 'copy' of a type (line 413)
    copy_137939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 10), arr_137938, 'copy')
    # Calling copy(args, kwargs) (line 413)
    copy_call_result_137941 = invoke(stypy.reporting.localization.Localization(__file__, 413, 10), copy_137939, *[], **kwargs_137940)
    
    # Assigning a type to the variable 'arr' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'arr', copy_call_result_137941)
    
    # Assigning a Str to a Subscript (line 414):
    str_137942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 19), 'str', ' ')
    # Getting the type of 'arr' (line 414)
    arr_137943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'arr')
    # Getting the type of 'empties' (line 414)
    empties_137944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'empties')
    # Storing an element on a container (line 414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 4), arr_137943, (empties_137944, str_137942))
    # Getting the type of 'arr' (line 415)
    arr_137945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type', arr_137945)
    
    # ################# End of 'arr_to_chars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arr_to_chars' in the type store
    # Getting the type of 'stypy_return_type' (line 401)
    stypy_return_type_137946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137946)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arr_to_chars'
    return stypy_return_type_137946

# Assigning a type to the variable 'arr_to_chars' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'arr_to_chars', arr_to_chars)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
