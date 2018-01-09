
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import numpy as np
4: from .numeric import uint8, ndarray, dtype
5: from numpy.compat import long, basestring
6: 
7: __all__ = ['memmap']
8: 
9: dtypedescr = dtype
10: valid_filemodes = ["r", "c", "r+", "w+"]
11: writeable_filemodes = ["r+", "w+"]
12: 
13: mode_equivalents = {
14:     "readonly":"r",
15:     "copyonwrite":"c",
16:     "readwrite":"r+",
17:     "write":"w+"
18:     }
19: 
20: class memmap(ndarray):
21:     '''Create a memory-map to an array stored in a *binary* file on disk.
22: 
23:     Memory-mapped files are used for accessing small segments of large files
24:     on disk, without reading the entire file into memory.  Numpy's
25:     memmap's are array-like objects.  This differs from Python's ``mmap``
26:     module, which uses file-like objects.
27: 
28:     This subclass of ndarray has some unpleasant interactions with
29:     some operations, because it doesn't quite fit properly as a subclass.
30:     An alternative to using this subclass is to create the ``mmap``
31:     object yourself, then create an ndarray with ndarray.__new__ directly,
32:     passing the object created in its 'buffer=' parameter.
33: 
34:     This class may at some point be turned into a factory function
35:     which returns a view into an mmap buffer.
36: 
37:     Delete the memmap instance to close.
38: 
39: 
40:     Parameters
41:     ----------
42:     filename : str or file-like object
43:         The file name or file object to be used as the array data buffer.
44:     dtype : data-type, optional
45:         The data-type used to interpret the file contents.
46:         Default is `uint8`.
47:     mode : {'r+', 'r', 'w+', 'c'}, optional
48:         The file is opened in this mode:
49: 
50:         +------+-------------------------------------------------------------+
51:         | 'r'  | Open existing file for reading only.                        |
52:         +------+-------------------------------------------------------------+
53:         | 'r+' | Open existing file for reading and writing.                 |
54:         +------+-------------------------------------------------------------+
55:         | 'w+' | Create or overwrite existing file for reading and writing.  |
56:         +------+-------------------------------------------------------------+
57:         | 'c'  | Copy-on-write: assignments affect data in memory, but       |
58:         |      | changes are not saved to disk.  The file on disk is         |
59:         |      | read-only.                                                  |
60:         +------+-------------------------------------------------------------+
61: 
62:         Default is 'r+'.
63:     offset : int, optional
64:         In the file, array data starts at this offset. Since `offset` is
65:         measured in bytes, it should normally be a multiple of the byte-size
66:         of `dtype`. When ``mode != 'r'``, even positive offsets beyond end of
67:         file are valid; The file will be extended to accommodate the
68:         additional data. By default, ``memmap`` will start at the beginning of
69:         the file, even if ``filename`` is a file pointer ``fp`` and
70:         ``fp.tell() != 0``.
71:     shape : tuple, optional
72:         The desired shape of the array. If ``mode == 'r'`` and the number
73:         of remaining bytes after `offset` is not a multiple of the byte-size
74:         of `dtype`, you must specify `shape`. By default, the returned array
75:         will be 1-D with the number of elements determined by file size
76:         and data-type.
77:     order : {'C', 'F'}, optional
78:         Specify the order of the ndarray memory layout:
79:         :term:`row-major`, C-style or :term:`column-major`,
80:         Fortran-style.  This only has an effect if the shape is
81:         greater than 1-D.  The default order is 'C'.
82: 
83:     Attributes
84:     ----------
85:     filename : str
86:         Path to the mapped file.
87:     offset : int
88:         Offset position in the file.
89:     mode : str
90:         File mode.
91: 
92:     Methods
93:     -------
94:     flush
95:         Flush any changes in memory to file on disk.
96:         When you delete a memmap object, flush is called first to write
97:         changes to disk before removing the object.
98: 
99: 
100:     Notes
101:     -----
102:     The memmap object can be used anywhere an ndarray is accepted.
103:     Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns
104:     ``True``.
105: 
106:     Memory-mapped arrays use the Python memory-map object which
107:     (prior to Python 2.5) does not allow files to be larger than a
108:     certain size depending on the platform. This size is always < 2GB
109:     even on 64-bit systems.
110: 
111:     When a memmap causes a file to be created or extended beyond its
112:     current size in the filesystem, the contents of the new part are
113:     unspecified. On systems with POSIX filesystem semantics, the extended
114:     part will be filled with zero bytes.
115: 
116:     Examples
117:     --------
118:     >>> data = np.arange(12, dtype='float32')
119:     >>> data.resize((3,4))
120: 
121:     This example uses a temporary file so that doctest doesn't write
122:     files to your directory. You would use a 'normal' filename.
123: 
124:     >>> from tempfile import mkdtemp
125:     >>> import os.path as path
126:     >>> filename = path.join(mkdtemp(), 'newfile.dat')
127: 
128:     Create a memmap with dtype and shape that matches our data:
129: 
130:     >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
131:     >>> fp
132:     memmap([[ 0.,  0.,  0.,  0.],
133:             [ 0.,  0.,  0.,  0.],
134:             [ 0.,  0.,  0.,  0.]], dtype=float32)
135: 
136:     Write data to memmap array:
137: 
138:     >>> fp[:] = data[:]
139:     >>> fp
140:     memmap([[  0.,   1.,   2.,   3.],
141:             [  4.,   5.,   6.,   7.],
142:             [  8.,   9.,  10.,  11.]], dtype=float32)
143: 
144:     >>> fp.filename == path.abspath(filename)
145:     True
146: 
147:     Deletion flushes memory changes to disk before removing the object:
148: 
149:     >>> del fp
150: 
151:     Load the memmap and verify data was stored:
152: 
153:     >>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
154:     >>> newfp
155:     memmap([[  0.,   1.,   2.,   3.],
156:             [  4.,   5.,   6.,   7.],
157:             [  8.,   9.,  10.,  11.]], dtype=float32)
158: 
159:     Read-only memmap:
160: 
161:     >>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
162:     >>> fpr.flags.writeable
163:     False
164: 
165:     Copy-on-write memmap:
166: 
167:     >>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
168:     >>> fpc.flags.writeable
169:     True
170: 
171:     It's possible to assign to copy-on-write array, but values are only
172:     written into the memory copy of the array, and not written to disk:
173: 
174:     >>> fpc
175:     memmap([[  0.,   1.,   2.,   3.],
176:             [  4.,   5.,   6.,   7.],
177:             [  8.,   9.,  10.,  11.]], dtype=float32)
178:     >>> fpc[0,:] = 0
179:     >>> fpc
180:     memmap([[  0.,   0.,   0.,   0.],
181:             [  4.,   5.,   6.,   7.],
182:             [  8.,   9.,  10.,  11.]], dtype=float32)
183: 
184:     File on disk is unchanged:
185: 
186:     >>> fpr
187:     memmap([[  0.,   1.,   2.,   3.],
188:             [  4.,   5.,   6.,   7.],
189:             [  8.,   9.,  10.,  11.]], dtype=float32)
190: 
191:     Offset into a memmap:
192: 
193:     >>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
194:     >>> fpo
195:     memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)
196: 
197:     '''
198: 
199:     __array_priority__ = -100.0
200: 
201:     def __new__(subtype, filename, dtype=uint8, mode='r+', offset=0,
202:                 shape=None, order='C'):
203:         # Import here to minimize 'import numpy' overhead
204:         import mmap
205:         import os.path
206:         try:
207:             mode = mode_equivalents[mode]
208:         except KeyError:
209:             if mode not in valid_filemodes:
210:                 raise ValueError("mode must be one of %s" %
211:                                  (valid_filemodes + list(mode_equivalents.keys())))
212: 
213:         if hasattr(filename, 'read'):
214:             fid = filename
215:             own_file = False
216:         else:
217:             fid = open(filename, (mode == 'c' and 'r' or mode)+'b')
218:             own_file = True
219: 
220:         if (mode == 'w+') and shape is None:
221:             raise ValueError("shape must be given")
222: 
223:         fid.seek(0, 2)
224:         flen = fid.tell()
225:         descr = dtypedescr(dtype)
226:         _dbytes = descr.itemsize
227: 
228:         if shape is None:
229:             bytes = flen - offset
230:             if (bytes % _dbytes):
231:                 fid.close()
232:                 raise ValueError("Size of available data is not a "
233:                         "multiple of the data-type size.")
234:             size = bytes // _dbytes
235:             shape = (size,)
236:         else:
237:             if not isinstance(shape, tuple):
238:                 shape = (shape,)
239:             size = 1
240:             for k in shape:
241:                 size *= k
242: 
243:         bytes = long(offset + size*_dbytes)
244: 
245:         if mode == 'w+' or (mode == 'r+' and flen < bytes):
246:             fid.seek(bytes - 1, 0)
247:             fid.write(np.compat.asbytes('\0'))
248:             fid.flush()
249: 
250:         if mode == 'c':
251:             acc = mmap.ACCESS_COPY
252:         elif mode == 'r':
253:             acc = mmap.ACCESS_READ
254:         else:
255:             acc = mmap.ACCESS_WRITE
256: 
257:         start = offset - offset % mmap.ALLOCATIONGRANULARITY
258:         bytes -= start
259:         offset -= start
260:         mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
261: 
262:         self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
263:             offset=offset, order=order)
264:         self._mmap = mm
265:         self.offset = offset
266:         self.mode = mode
267: 
268:         if isinstance(filename, basestring):
269:             self.filename = os.path.abspath(filename)
270:         # py3 returns int for TemporaryFile().name
271:         elif (hasattr(filename, "name") and
272:               isinstance(filename.name, basestring)):
273:             self.filename = os.path.abspath(filename.name)
274:         # same as memmap copies (e.g. memmap + 1)
275:         else:
276:             self.filename = None
277: 
278:         if own_file:
279:             fid.close()
280: 
281:         return self
282: 
283:     def __array_finalize__(self, obj):
284:         if hasattr(obj, '_mmap') and np.may_share_memory(self, obj):
285:             self._mmap = obj._mmap
286:             self.filename = obj.filename
287:             self.offset = obj.offset
288:             self.mode = obj.mode
289:         else:
290:             self._mmap = None
291:             self.filename = None
292:             self.offset = None
293:             self.mode = None
294: 
295:     def flush(self):
296:         '''
297:         Write any changes in the array to the file on disk.
298: 
299:         For further information, see `memmap`.
300: 
301:         Parameters
302:         ----------
303:         None
304: 
305:         See Also
306:         --------
307:         memmap
308: 
309:         '''
310:         if self.base is not None and hasattr(self.base, 'flush'):
311:             self.base.flush()
312: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_7092 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_7092) is not StypyTypeError):

    if (import_7092 != 'pyd_module'):
        __import__(import_7092)
        sys_modules_7093 = sys.modules[import_7092]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_7093.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_7092)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.core.numeric import uint8, ndarray, dtype' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_7094 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.core.numeric')

if (type(import_7094) is not StypyTypeError):

    if (import_7094 != 'pyd_module'):
        __import__(import_7094)
        sys_modules_7095 = sys.modules[import_7094]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.core.numeric', sys_modules_7095.module_type_store, module_type_store, ['uint8', 'ndarray', 'dtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_7095, sys_modules_7095.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import uint8, ndarray, dtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.core.numeric', None, module_type_store, ['uint8', 'ndarray', 'dtype'], [uint8, ndarray, dtype])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.core.numeric', import_7094)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.compat import long, basestring' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_7096 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.compat')

if (type(import_7096) is not StypyTypeError):

    if (import_7096 != 'pyd_module'):
        __import__(import_7096)
        sys_modules_7097 = sys.modules[import_7096]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.compat', sys_modules_7097.module_type_store, module_type_store, ['long', 'basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_7097, sys_modules_7097.module_type_store, module_type_store)
    else:
        from numpy.compat import long, basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.compat', None, module_type_store, ['long', 'basestring'], [long, basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.compat', import_7096)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a List to a Name (line 7):
__all__ = ['memmap']
module_type_store.set_exportable_members(['memmap'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_7098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_7099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'memmap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_7098, str_7099)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_7098)

# Assigning a Name to a Name (line 9):
# Getting the type of 'dtype' (line 9)
dtype_7100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'dtype')
# Assigning a type to the variable 'dtypedescr' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'dtypedescr', dtype_7100)

# Assigning a List to a Name (line 10):

# Obtaining an instance of the builtin type 'list' (line 10)
list_7101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_7102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'str', 'r')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_7101, str_7102)
# Adding element type (line 10)
str_7103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_7101, str_7103)
# Adding element type (line 10)
str_7104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 29), 'str', 'r+')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_7101, str_7104)
# Adding element type (line 10)
str_7105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 35), 'str', 'w+')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_7101, str_7105)

# Assigning a type to the variable 'valid_filemodes' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'valid_filemodes', list_7101)

# Assigning a List to a Name (line 11):

# Obtaining an instance of the builtin type 'list' (line 11)
list_7106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_7107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'str', 'r+')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 22), list_7106, str_7107)
# Adding element type (line 11)
str_7108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'str', 'w+')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 22), list_7106, str_7108)

# Assigning a type to the variable 'writeable_filemodes' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'writeable_filemodes', list_7106)

# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_7109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
str_7110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'readonly')
str_7111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', 'r')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 19), dict_7109, (str_7110, str_7111))
# Adding element type (key, value) (line 13)
str_7112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'copyonwrite')
str_7113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'str', 'c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 19), dict_7109, (str_7112, str_7113))
# Adding element type (key, value) (line 13)
str_7114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'readwrite')
str_7115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'str', 'r+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 19), dict_7109, (str_7114, str_7115))
# Adding element type (key, value) (line 13)
str_7116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'write')
str_7117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'str', 'w+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 19), dict_7109, (str_7116, str_7117))

# Assigning a type to the variable 'mode_equivalents' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'mode_equivalents', dict_7109)
# Declaration of the 'memmap' class
# Getting the type of 'ndarray' (line 20)
ndarray_7118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), 'ndarray')

class memmap(ndarray_7118, ):
    str_7119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', "Create a memory-map to an array stored in a *binary* file on disk.\n\n    Memory-mapped files are used for accessing small segments of large files\n    on disk, without reading the entire file into memory.  Numpy's\n    memmap's are array-like objects.  This differs from Python's ``mmap``\n    module, which uses file-like objects.\n\n    This subclass of ndarray has some unpleasant interactions with\n    some operations, because it doesn't quite fit properly as a subclass.\n    An alternative to using this subclass is to create the ``mmap``\n    object yourself, then create an ndarray with ndarray.__new__ directly,\n    passing the object created in its 'buffer=' parameter.\n\n    This class may at some point be turned into a factory function\n    which returns a view into an mmap buffer.\n\n    Delete the memmap instance to close.\n\n\n    Parameters\n    ----------\n    filename : str or file-like object\n        The file name or file object to be used as the array data buffer.\n    dtype : data-type, optional\n        The data-type used to interpret the file contents.\n        Default is `uint8`.\n    mode : {'r+', 'r', 'w+', 'c'}, optional\n        The file is opened in this mode:\n\n        +------+-------------------------------------------------------------+\n        | 'r'  | Open existing file for reading only.                        |\n        +------+-------------------------------------------------------------+\n        | 'r+' | Open existing file for reading and writing.                 |\n        +------+-------------------------------------------------------------+\n        | 'w+' | Create or overwrite existing file for reading and writing.  |\n        +------+-------------------------------------------------------------+\n        | 'c'  | Copy-on-write: assignments affect data in memory, but       |\n        |      | changes are not saved to disk.  The file on disk is         |\n        |      | read-only.                                                  |\n        +------+-------------------------------------------------------------+\n\n        Default is 'r+'.\n    offset : int, optional\n        In the file, array data starts at this offset. Since `offset` is\n        measured in bytes, it should normally be a multiple of the byte-size\n        of `dtype`. When ``mode != 'r'``, even positive offsets beyond end of\n        file are valid; The file will be extended to accommodate the\n        additional data. By default, ``memmap`` will start at the beginning of\n        the file, even if ``filename`` is a file pointer ``fp`` and\n        ``fp.tell() != 0``.\n    shape : tuple, optional\n        The desired shape of the array. If ``mode == 'r'`` and the number\n        of remaining bytes after `offset` is not a multiple of the byte-size\n        of `dtype`, you must specify `shape`. By default, the returned array\n        will be 1-D with the number of elements determined by file size\n        and data-type.\n    order : {'C', 'F'}, optional\n        Specify the order of the ndarray memory layout:\n        :term:`row-major`, C-style or :term:`column-major`,\n        Fortran-style.  This only has an effect if the shape is\n        greater than 1-D.  The default order is 'C'.\n\n    Attributes\n    ----------\n    filename : str\n        Path to the mapped file.\n    offset : int\n        Offset position in the file.\n    mode : str\n        File mode.\n\n    Methods\n    -------\n    flush\n        Flush any changes in memory to file on disk.\n        When you delete a memmap object, flush is called first to write\n        changes to disk before removing the object.\n\n\n    Notes\n    -----\n    The memmap object can be used anywhere an ndarray is accepted.\n    Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns\n    ``True``.\n\n    Memory-mapped arrays use the Python memory-map object which\n    (prior to Python 2.5) does not allow files to be larger than a\n    certain size depending on the platform. This size is always < 2GB\n    even on 64-bit systems.\n\n    When a memmap causes a file to be created or extended beyond its\n    current size in the filesystem, the contents of the new part are\n    unspecified. On systems with POSIX filesystem semantics, the extended\n    part will be filled with zero bytes.\n\n    Examples\n    --------\n    >>> data = np.arange(12, dtype='float32')\n    >>> data.resize((3,4))\n\n    This example uses a temporary file so that doctest doesn't write\n    files to your directory. You would use a 'normal' filename.\n\n    >>> from tempfile import mkdtemp\n    >>> import os.path as path\n    >>> filename = path.join(mkdtemp(), 'newfile.dat')\n\n    Create a memmap with dtype and shape that matches our data:\n\n    >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))\n    >>> fp\n    memmap([[ 0.,  0.,  0.,  0.],\n            [ 0.,  0.,  0.,  0.],\n            [ 0.,  0.,  0.,  0.]], dtype=float32)\n\n    Write data to memmap array:\n\n    >>> fp[:] = data[:]\n    >>> fp\n    memmap([[  0.,   1.,   2.,   3.],\n            [  4.,   5.,   6.,   7.],\n            [  8.,   9.,  10.,  11.]], dtype=float32)\n\n    >>> fp.filename == path.abspath(filename)\n    True\n\n    Deletion flushes memory changes to disk before removing the object:\n\n    >>> del fp\n\n    Load the memmap and verify data was stored:\n\n    >>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))\n    >>> newfp\n    memmap([[  0.,   1.,   2.,   3.],\n            [  4.,   5.,   6.,   7.],\n            [  8.,   9.,  10.,  11.]], dtype=float32)\n\n    Read-only memmap:\n\n    >>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))\n    >>> fpr.flags.writeable\n    False\n\n    Copy-on-write memmap:\n\n    >>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))\n    >>> fpc.flags.writeable\n    True\n\n    It's possible to assign to copy-on-write array, but values are only\n    written into the memory copy of the array, and not written to disk:\n\n    >>> fpc\n    memmap([[  0.,   1.,   2.,   3.],\n            [  4.,   5.,   6.,   7.],\n            [  8.,   9.,  10.,  11.]], dtype=float32)\n    >>> fpc[0,:] = 0\n    >>> fpc\n    memmap([[  0.,   0.,   0.,   0.],\n            [  4.,   5.,   6.,   7.],\n            [  8.,   9.,  10.,  11.]], dtype=float32)\n\n    File on disk is unchanged:\n\n    >>> fpr\n    memmap([[  0.,   1.,   2.,   3.],\n            [  4.,   5.,   6.,   7.],\n            [  8.,   9.,  10.,  11.]], dtype=float32)\n\n    Offset into a memmap:\n\n    >>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)\n    >>> fpo\n    memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)\n\n    ")

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'uint8' (line 201)
        uint8_7120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 41), 'uint8')
        str_7121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 53), 'str', 'r+')
        int_7122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 66), 'int')
        # Getting the type of 'None' (line 202)
        None_7123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'None')
        str_7124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 34), 'str', 'C')
        defaults = [uint8_7120, str_7121, int_7122, None_7123, str_7124]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        memmap.__new__.__dict__.__setitem__('stypy_localization', localization)
        memmap.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        memmap.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        memmap.__new__.__dict__.__setitem__('stypy_function_name', 'memmap.__new__')
        memmap.__new__.__dict__.__setitem__('stypy_param_names_list', ['filename', 'dtype', 'mode', 'offset', 'shape', 'order'])
        memmap.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        memmap.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        memmap.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        memmap.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        memmap.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        memmap.__new__.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'memmap.__new__', ['filename', 'dtype', 'mode', 'offset', 'shape', 'order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['filename', 'dtype', 'mode', 'offset', 'shape', 'order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 204, 8))
        
        # 'import mmap' statement (line 204)
        import mmap

        import_module(stypy.reporting.localization.Localization(__file__, 204, 8), 'mmap', mmap, module_type_store)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 205, 8))
        
        # 'import os.path' statement (line 205)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
        import_7125 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 205, 8), 'os.path')

        if (type(import_7125) is not StypyTypeError):

            if (import_7125 != 'pyd_module'):
                __import__(import_7125)
                sys_modules_7126 = sys.modules[import_7125]
                import_module(stypy.reporting.localization.Localization(__file__, 205, 8), 'os.path', sys_modules_7126.module_type_store, module_type_store)
            else:
                import os.path

                import_module(stypy.reporting.localization.Localization(__file__, 205, 8), 'os.path', os.path, module_type_store)

        else:
            # Assigning a type to the variable 'os.path' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'os.path', import_7125)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
        
        
        
        # SSA begins for try-except statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mode' (line 207)
        mode_7127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 36), 'mode')
        # Getting the type of 'mode_equivalents' (line 207)
        mode_equivalents_7128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'mode_equivalents')
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___7129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 19), mode_equivalents_7128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_7130 = invoke(stypy.reporting.localization.Localization(__file__, 207, 19), getitem___7129, mode_7127)
        
        # Assigning a type to the variable 'mode' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'mode', subscript_call_result_7130)
        # SSA branch for the except part of a try statement (line 206)
        # SSA branch for the except 'KeyError' branch of a try statement (line 206)
        module_type_store.open_ssa_branch('except')
        
        
        # Getting the type of 'mode' (line 209)
        mode_7131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'mode')
        # Getting the type of 'valid_filemodes' (line 209)
        valid_filemodes_7132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'valid_filemodes')
        # Applying the binary operator 'notin' (line 209)
        result_contains_7133 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 15), 'notin', mode_7131, valid_filemodes_7132)
        
        # Testing the type of an if condition (line 209)
        if_condition_7134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 12), result_contains_7133)
        # Assigning a type to the variable 'if_condition_7134' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'if_condition_7134', if_condition_7134)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 210)
        # Processing the call arguments (line 210)
        str_7136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 33), 'str', 'mode must be one of %s')
        # Getting the type of 'valid_filemodes' (line 211)
        valid_filemodes_7137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 34), 'valid_filemodes', False)
        
        # Call to list(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Call to keys(...): (line 211)
        # Processing the call keyword arguments (line 211)
        kwargs_7141 = {}
        # Getting the type of 'mode_equivalents' (line 211)
        mode_equivalents_7139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 57), 'mode_equivalents', False)
        # Obtaining the member 'keys' of a type (line 211)
        keys_7140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 57), mode_equivalents_7139, 'keys')
        # Calling keys(args, kwargs) (line 211)
        keys_call_result_7142 = invoke(stypy.reporting.localization.Localization(__file__, 211, 57), keys_7140, *[], **kwargs_7141)
        
        # Processing the call keyword arguments (line 211)
        kwargs_7143 = {}
        # Getting the type of 'list' (line 211)
        list_7138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 52), 'list', False)
        # Calling list(args, kwargs) (line 211)
        list_call_result_7144 = invoke(stypy.reporting.localization.Localization(__file__, 211, 52), list_7138, *[keys_call_result_7142], **kwargs_7143)
        
        # Applying the binary operator '+' (line 211)
        result_add_7145 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 34), '+', valid_filemodes_7137, list_call_result_7144)
        
        # Applying the binary operator '%' (line 210)
        result_mod_7146 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 33), '%', str_7136, result_add_7145)
        
        # Processing the call keyword arguments (line 210)
        kwargs_7147 = {}
        # Getting the type of 'ValueError' (line 210)
        ValueError_7135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 210)
        ValueError_call_result_7148 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), ValueError_7135, *[result_mod_7146], **kwargs_7147)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 210, 16), ValueError_call_result_7148, 'raise parameter', BaseException)
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 213)
        str_7149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 29), 'str', 'read')
        # Getting the type of 'filename' (line 213)
        filename_7150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'filename')
        
        (may_be_7151, more_types_in_union_7152) = may_provide_member(str_7149, filename_7150)

        if may_be_7151:

            if more_types_in_union_7152:
                # Runtime conditional SSA (line 213)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'filename' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'filename', remove_not_member_provider_from_union(filename_7150, 'read'))
            
            # Assigning a Name to a Name (line 214):
            # Getting the type of 'filename' (line 214)
            filename_7153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'filename')
            # Assigning a type to the variable 'fid' (line 214)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'fid', filename_7153)
            
            # Assigning a Name to a Name (line 215):
            # Getting the type of 'False' (line 215)
            False_7154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'False')
            # Assigning a type to the variable 'own_file' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'own_file', False_7154)

            if more_types_in_union_7152:
                # Runtime conditional SSA for else branch (line 213)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_7151) or more_types_in_union_7152):
            # Assigning a type to the variable 'filename' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'filename', remove_member_provider_from_union(filename_7150, 'read'))
            
            # Assigning a Call to a Name (line 217):
            
            # Call to open(...): (line 217)
            # Processing the call arguments (line 217)
            # Getting the type of 'filename' (line 217)
            filename_7156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 23), 'filename', False)
            
            # Evaluating a boolean operation
            
            # Evaluating a boolean operation
            
            # Getting the type of 'mode' (line 217)
            mode_7157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'mode', False)
            str_7158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'str', 'c')
            # Applying the binary operator '==' (line 217)
            result_eq_7159 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 34), '==', mode_7157, str_7158)
            
            str_7160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 50), 'str', 'r')
            # Applying the binary operator 'and' (line 217)
            result_and_keyword_7161 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 34), 'and', result_eq_7159, str_7160)
            
            # Getting the type of 'mode' (line 217)
            mode_7162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 57), 'mode', False)
            # Applying the binary operator 'or' (line 217)
            result_or_keyword_7163 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 34), 'or', result_and_keyword_7161, mode_7162)
            
            str_7164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 63), 'str', 'b')
            # Applying the binary operator '+' (line 217)
            result_add_7165 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 33), '+', result_or_keyword_7163, str_7164)
            
            # Processing the call keyword arguments (line 217)
            kwargs_7166 = {}
            # Getting the type of 'open' (line 217)
            open_7155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'open', False)
            # Calling open(args, kwargs) (line 217)
            open_call_result_7167 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), open_7155, *[filename_7156, result_add_7165], **kwargs_7166)
            
            # Assigning a type to the variable 'fid' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'fid', open_call_result_7167)
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'True' (line 218)
            True_7168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'True')
            # Assigning a type to the variable 'own_file' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'own_file', True_7168)

            if (may_be_7151 and more_types_in_union_7152):
                # SSA join for if statement (line 213)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mode' (line 220)
        mode_7169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'mode')
        str_7170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'str', 'w+')
        # Applying the binary operator '==' (line 220)
        result_eq_7171 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 12), '==', mode_7169, str_7170)
        
        
        # Getting the type of 'shape' (line 220)
        shape_7172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'shape')
        # Getting the type of 'None' (line 220)
        None_7173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 39), 'None')
        # Applying the binary operator 'is' (line 220)
        result_is__7174 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 30), 'is', shape_7172, None_7173)
        
        # Applying the binary operator 'and' (line 220)
        result_and_keyword_7175 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 11), 'and', result_eq_7171, result_is__7174)
        
        # Testing the type of an if condition (line 220)
        if_condition_7176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), result_and_keyword_7175)
        # Assigning a type to the variable 'if_condition_7176' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_7176', if_condition_7176)
        # SSA begins for if statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 221)
        # Processing the call arguments (line 221)
        str_7178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 29), 'str', 'shape must be given')
        # Processing the call keyword arguments (line 221)
        kwargs_7179 = {}
        # Getting the type of 'ValueError' (line 221)
        ValueError_7177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 221)
        ValueError_call_result_7180 = invoke(stypy.reporting.localization.Localization(__file__, 221, 18), ValueError_7177, *[str_7178], **kwargs_7179)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 221, 12), ValueError_call_result_7180, 'raise parameter', BaseException)
        # SSA join for if statement (line 220)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to seek(...): (line 223)
        # Processing the call arguments (line 223)
        int_7183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 17), 'int')
        int_7184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 20), 'int')
        # Processing the call keyword arguments (line 223)
        kwargs_7185 = {}
        # Getting the type of 'fid' (line 223)
        fid_7181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'fid', False)
        # Obtaining the member 'seek' of a type (line 223)
        seek_7182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), fid_7181, 'seek')
        # Calling seek(args, kwargs) (line 223)
        seek_call_result_7186 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), seek_7182, *[int_7183, int_7184], **kwargs_7185)
        
        
        # Assigning a Call to a Name (line 224):
        
        # Call to tell(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_7189 = {}
        # Getting the type of 'fid' (line 224)
        fid_7187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'fid', False)
        # Obtaining the member 'tell' of a type (line 224)
        tell_7188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), fid_7187, 'tell')
        # Calling tell(args, kwargs) (line 224)
        tell_call_result_7190 = invoke(stypy.reporting.localization.Localization(__file__, 224, 15), tell_7188, *[], **kwargs_7189)
        
        # Assigning a type to the variable 'flen' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'flen', tell_call_result_7190)
        
        # Assigning a Call to a Name (line 225):
        
        # Call to dtypedescr(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'dtype' (line 225)
        dtype_7192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'dtype', False)
        # Processing the call keyword arguments (line 225)
        kwargs_7193 = {}
        # Getting the type of 'dtypedescr' (line 225)
        dtypedescr_7191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'dtypedescr', False)
        # Calling dtypedescr(args, kwargs) (line 225)
        dtypedescr_call_result_7194 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), dtypedescr_7191, *[dtype_7192], **kwargs_7193)
        
        # Assigning a type to the variable 'descr' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'descr', dtypedescr_call_result_7194)
        
        # Assigning a Attribute to a Name (line 226):
        # Getting the type of 'descr' (line 226)
        descr_7195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'descr')
        # Obtaining the member 'itemsize' of a type (line 226)
        itemsize_7196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 18), descr_7195, 'itemsize')
        # Assigning a type to the variable '_dbytes' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), '_dbytes', itemsize_7196)
        
        # Type idiom detected: calculating its left and rigth part (line 228)
        # Getting the type of 'shape' (line 228)
        shape_7197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'shape')
        # Getting the type of 'None' (line 228)
        None_7198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'None')
        
        (may_be_7199, more_types_in_union_7200) = may_be_none(shape_7197, None_7198)

        if may_be_7199:

            if more_types_in_union_7200:
                # Runtime conditional SSA (line 228)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 229):
            # Getting the type of 'flen' (line 229)
            flen_7201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'flen')
            # Getting the type of 'offset' (line 229)
            offset_7202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'offset')
            # Applying the binary operator '-' (line 229)
            result_sub_7203 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 20), '-', flen_7201, offset_7202)
            
            # Assigning a type to the variable 'bytes' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'bytes', result_sub_7203)
            
            # Getting the type of 'bytes' (line 230)
            bytes_7204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'bytes')
            # Getting the type of '_dbytes' (line 230)
            _dbytes_7205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), '_dbytes')
            # Applying the binary operator '%' (line 230)
            result_mod_7206 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 16), '%', bytes_7204, _dbytes_7205)
            
            # Testing the type of an if condition (line 230)
            if_condition_7207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 12), result_mod_7206)
            # Assigning a type to the variable 'if_condition_7207' (line 230)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'if_condition_7207', if_condition_7207)
            # SSA begins for if statement (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to close(...): (line 231)
            # Processing the call keyword arguments (line 231)
            kwargs_7210 = {}
            # Getting the type of 'fid' (line 231)
            fid_7208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'fid', False)
            # Obtaining the member 'close' of a type (line 231)
            close_7209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), fid_7208, 'close')
            # Calling close(args, kwargs) (line 231)
            close_call_result_7211 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), close_7209, *[], **kwargs_7210)
            
            
            # Call to ValueError(...): (line 232)
            # Processing the call arguments (line 232)
            str_7213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 33), 'str', 'Size of available data is not a multiple of the data-type size.')
            # Processing the call keyword arguments (line 232)
            kwargs_7214 = {}
            # Getting the type of 'ValueError' (line 232)
            ValueError_7212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 232)
            ValueError_call_result_7215 = invoke(stypy.reporting.localization.Localization(__file__, 232, 22), ValueError_7212, *[str_7213], **kwargs_7214)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 232, 16), ValueError_call_result_7215, 'raise parameter', BaseException)
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 234):
            # Getting the type of 'bytes' (line 234)
            bytes_7216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'bytes')
            # Getting the type of '_dbytes' (line 234)
            _dbytes_7217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 28), '_dbytes')
            # Applying the binary operator '//' (line 234)
            result_floordiv_7218 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 19), '//', bytes_7216, _dbytes_7217)
            
            # Assigning a type to the variable 'size' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'size', result_floordiv_7218)
            
            # Assigning a Tuple to a Name (line 235):
            
            # Obtaining an instance of the builtin type 'tuple' (line 235)
            tuple_7219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 235)
            # Adding element type (line 235)
            # Getting the type of 'size' (line 235)
            size_7220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 21), 'size')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 21), tuple_7219, size_7220)
            
            # Assigning a type to the variable 'shape' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'shape', tuple_7219)

            if more_types_in_union_7200:
                # Runtime conditional SSA for else branch (line 228)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_7199) or more_types_in_union_7200):
            
            # Type idiom detected: calculating its left and rigth part (line 237)
            # Getting the type of 'tuple' (line 237)
            tuple_7221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 37), 'tuple')
            # Getting the type of 'shape' (line 237)
            shape_7222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 30), 'shape')
            
            (may_be_7223, more_types_in_union_7224) = may_not_be_subtype(tuple_7221, shape_7222)

            if may_be_7223:

                if more_types_in_union_7224:
                    # Runtime conditional SSA (line 237)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'shape' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'shape', remove_subtype_from_union(shape_7222, tuple))
                
                # Assigning a Tuple to a Name (line 238):
                
                # Obtaining an instance of the builtin type 'tuple' (line 238)
                tuple_7225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 25), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 238)
                # Adding element type (line 238)
                # Getting the type of 'shape' (line 238)
                shape_7226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 25), 'shape')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 25), tuple_7225, shape_7226)
                
                # Assigning a type to the variable 'shape' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'shape', tuple_7225)

                if more_types_in_union_7224:
                    # SSA join for if statement (line 237)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Num to a Name (line 239):
            int_7227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'int')
            # Assigning a type to the variable 'size' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'size', int_7227)
            
            # Getting the type of 'shape' (line 240)
            shape_7228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'shape')
            # Testing the type of a for loop iterable (line 240)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 240, 12), shape_7228)
            # Getting the type of the for loop variable (line 240)
            for_loop_var_7229 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 240, 12), shape_7228)
            # Assigning a type to the variable 'k' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'k', for_loop_var_7229)
            # SSA begins for a for statement (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'size' (line 241)
            size_7230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'size')
            # Getting the type of 'k' (line 241)
            k_7231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'k')
            # Applying the binary operator '*=' (line 241)
            result_imul_7232 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 16), '*=', size_7230, k_7231)
            # Assigning a type to the variable 'size' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'size', result_imul_7232)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_7199 and more_types_in_union_7200):
                # SSA join for if statement (line 228)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 243):
        
        # Call to long(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'offset' (line 243)
        offset_7234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 21), 'offset', False)
        # Getting the type of 'size' (line 243)
        size_7235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'size', False)
        # Getting the type of '_dbytes' (line 243)
        _dbytes_7236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 35), '_dbytes', False)
        # Applying the binary operator '*' (line 243)
        result_mul_7237 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 30), '*', size_7235, _dbytes_7236)
        
        # Applying the binary operator '+' (line 243)
        result_add_7238 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 21), '+', offset_7234, result_mul_7237)
        
        # Processing the call keyword arguments (line 243)
        kwargs_7239 = {}
        # Getting the type of 'long' (line 243)
        long_7233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'long', False)
        # Calling long(args, kwargs) (line 243)
        long_call_result_7240 = invoke(stypy.reporting.localization.Localization(__file__, 243, 16), long_7233, *[result_add_7238], **kwargs_7239)
        
        # Assigning a type to the variable 'bytes' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'bytes', long_call_result_7240)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mode' (line 245)
        mode_7241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'mode')
        str_7242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'str', 'w+')
        # Applying the binary operator '==' (line 245)
        result_eq_7243 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 11), '==', mode_7241, str_7242)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mode' (line 245)
        mode_7244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'mode')
        str_7245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 36), 'str', 'r+')
        # Applying the binary operator '==' (line 245)
        result_eq_7246 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 28), '==', mode_7244, str_7245)
        
        
        # Getting the type of 'flen' (line 245)
        flen_7247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 45), 'flen')
        # Getting the type of 'bytes' (line 245)
        bytes_7248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 52), 'bytes')
        # Applying the binary operator '<' (line 245)
        result_lt_7249 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 45), '<', flen_7247, bytes_7248)
        
        # Applying the binary operator 'and' (line 245)
        result_and_keyword_7250 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 28), 'and', result_eq_7246, result_lt_7249)
        
        # Applying the binary operator 'or' (line 245)
        result_or_keyword_7251 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 11), 'or', result_eq_7243, result_and_keyword_7250)
        
        # Testing the type of an if condition (line 245)
        if_condition_7252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 8), result_or_keyword_7251)
        # Assigning a type to the variable 'if_condition_7252' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'if_condition_7252', if_condition_7252)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to seek(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'bytes' (line 246)
        bytes_7255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'bytes', False)
        int_7256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 29), 'int')
        # Applying the binary operator '-' (line 246)
        result_sub_7257 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 21), '-', bytes_7255, int_7256)
        
        int_7258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 32), 'int')
        # Processing the call keyword arguments (line 246)
        kwargs_7259 = {}
        # Getting the type of 'fid' (line 246)
        fid_7253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'fid', False)
        # Obtaining the member 'seek' of a type (line 246)
        seek_7254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), fid_7253, 'seek')
        # Calling seek(args, kwargs) (line 246)
        seek_call_result_7260 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), seek_7254, *[result_sub_7257, int_7258], **kwargs_7259)
        
        
        # Call to write(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Call to asbytes(...): (line 247)
        # Processing the call arguments (line 247)
        str_7266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 40), 'str', '\x00')
        # Processing the call keyword arguments (line 247)
        kwargs_7267 = {}
        # Getting the type of 'np' (line 247)
        np_7263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'np', False)
        # Obtaining the member 'compat' of a type (line 247)
        compat_7264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 22), np_7263, 'compat')
        # Obtaining the member 'asbytes' of a type (line 247)
        asbytes_7265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 22), compat_7264, 'asbytes')
        # Calling asbytes(args, kwargs) (line 247)
        asbytes_call_result_7268 = invoke(stypy.reporting.localization.Localization(__file__, 247, 22), asbytes_7265, *[str_7266], **kwargs_7267)
        
        # Processing the call keyword arguments (line 247)
        kwargs_7269 = {}
        # Getting the type of 'fid' (line 247)
        fid_7261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'fid', False)
        # Obtaining the member 'write' of a type (line 247)
        write_7262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 12), fid_7261, 'write')
        # Calling write(args, kwargs) (line 247)
        write_call_result_7270 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), write_7262, *[asbytes_call_result_7268], **kwargs_7269)
        
        
        # Call to flush(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_7273 = {}
        # Getting the type of 'fid' (line 248)
        fid_7271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'fid', False)
        # Obtaining the member 'flush' of a type (line 248)
        flush_7272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), fid_7271, 'flush')
        # Calling flush(args, kwargs) (line 248)
        flush_call_result_7274 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), flush_7272, *[], **kwargs_7273)
        
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'mode' (line 250)
        mode_7275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'mode')
        str_7276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 19), 'str', 'c')
        # Applying the binary operator '==' (line 250)
        result_eq_7277 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 11), '==', mode_7275, str_7276)
        
        # Testing the type of an if condition (line 250)
        if_condition_7278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 8), result_eq_7277)
        # Assigning a type to the variable 'if_condition_7278' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'if_condition_7278', if_condition_7278)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 251):
        # Getting the type of 'mmap' (line 251)
        mmap_7279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'mmap')
        # Obtaining the member 'ACCESS_COPY' of a type (line 251)
        ACCESS_COPY_7280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 18), mmap_7279, 'ACCESS_COPY')
        # Assigning a type to the variable 'acc' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'acc', ACCESS_COPY_7280)
        # SSA branch for the else part of an if statement (line 250)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'mode' (line 252)
        mode_7281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 13), 'mode')
        str_7282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 21), 'str', 'r')
        # Applying the binary operator '==' (line 252)
        result_eq_7283 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 13), '==', mode_7281, str_7282)
        
        # Testing the type of an if condition (line 252)
        if_condition_7284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 13), result_eq_7283)
        # Assigning a type to the variable 'if_condition_7284' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 13), 'if_condition_7284', if_condition_7284)
        # SSA begins for if statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 253):
        # Getting the type of 'mmap' (line 253)
        mmap_7285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 'mmap')
        # Obtaining the member 'ACCESS_READ' of a type (line 253)
        ACCESS_READ_7286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 18), mmap_7285, 'ACCESS_READ')
        # Assigning a type to the variable 'acc' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'acc', ACCESS_READ_7286)
        # SSA branch for the else part of an if statement (line 252)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 255):
        # Getting the type of 'mmap' (line 255)
        mmap_7287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 18), 'mmap')
        # Obtaining the member 'ACCESS_WRITE' of a type (line 255)
        ACCESS_WRITE_7288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 18), mmap_7287, 'ACCESS_WRITE')
        # Assigning a type to the variable 'acc' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'acc', ACCESS_WRITE_7288)
        # SSA join for if statement (line 252)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 257):
        # Getting the type of 'offset' (line 257)
        offset_7289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'offset')
        # Getting the type of 'offset' (line 257)
        offset_7290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'offset')
        # Getting the type of 'mmap' (line 257)
        mmap_7291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 'mmap')
        # Obtaining the member 'ALLOCATIONGRANULARITY' of a type (line 257)
        ALLOCATIONGRANULARITY_7292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 34), mmap_7291, 'ALLOCATIONGRANULARITY')
        # Applying the binary operator '%' (line 257)
        result_mod_7293 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 25), '%', offset_7290, ALLOCATIONGRANULARITY_7292)
        
        # Applying the binary operator '-' (line 257)
        result_sub_7294 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 16), '-', offset_7289, result_mod_7293)
        
        # Assigning a type to the variable 'start' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'start', result_sub_7294)
        
        # Getting the type of 'bytes' (line 258)
        bytes_7295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'bytes')
        # Getting the type of 'start' (line 258)
        start_7296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'start')
        # Applying the binary operator '-=' (line 258)
        result_isub_7297 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 8), '-=', bytes_7295, start_7296)
        # Assigning a type to the variable 'bytes' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'bytes', result_isub_7297)
        
        
        # Getting the type of 'offset' (line 259)
        offset_7298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'offset')
        # Getting the type of 'start' (line 259)
        start_7299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'start')
        # Applying the binary operator '-=' (line 259)
        result_isub_7300 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 8), '-=', offset_7298, start_7299)
        # Assigning a type to the variable 'offset' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'offset', result_isub_7300)
        
        
        # Assigning a Call to a Name (line 260):
        
        # Call to mmap(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Call to fileno(...): (line 260)
        # Processing the call keyword arguments (line 260)
        kwargs_7305 = {}
        # Getting the type of 'fid' (line 260)
        fid_7303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'fid', False)
        # Obtaining the member 'fileno' of a type (line 260)
        fileno_7304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 23), fid_7303, 'fileno')
        # Calling fileno(args, kwargs) (line 260)
        fileno_call_result_7306 = invoke(stypy.reporting.localization.Localization(__file__, 260, 23), fileno_7304, *[], **kwargs_7305)
        
        # Getting the type of 'bytes' (line 260)
        bytes_7307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'bytes', False)
        # Processing the call keyword arguments (line 260)
        # Getting the type of 'acc' (line 260)
        acc_7308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 51), 'acc', False)
        keyword_7309 = acc_7308
        # Getting the type of 'start' (line 260)
        start_7310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 63), 'start', False)
        keyword_7311 = start_7310
        kwargs_7312 = {'access': keyword_7309, 'offset': keyword_7311}
        # Getting the type of 'mmap' (line 260)
        mmap_7301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 13), 'mmap', False)
        # Obtaining the member 'mmap' of a type (line 260)
        mmap_7302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 13), mmap_7301, 'mmap')
        # Calling mmap(args, kwargs) (line 260)
        mmap_call_result_7313 = invoke(stypy.reporting.localization.Localization(__file__, 260, 13), mmap_7302, *[fileno_call_result_7306, bytes_7307], **kwargs_7312)
        
        # Assigning a type to the variable 'mm' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'mm', mmap_call_result_7313)
        
        # Assigning a Call to a Name (line 262):
        
        # Call to __new__(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'subtype' (line 262)
        subtype_7316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 31), 'subtype', False)
        # Getting the type of 'shape' (line 262)
        shape_7317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 40), 'shape', False)
        # Processing the call keyword arguments (line 262)
        # Getting the type of 'descr' (line 262)
        descr_7318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 53), 'descr', False)
        keyword_7319 = descr_7318
        # Getting the type of 'mm' (line 262)
        mm_7320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 67), 'mm', False)
        keyword_7321 = mm_7320
        # Getting the type of 'offset' (line 263)
        offset_7322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'offset', False)
        keyword_7323 = offset_7322
        # Getting the type of 'order' (line 263)
        order_7324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'order', False)
        keyword_7325 = order_7324
        kwargs_7326 = {'buffer': keyword_7321, 'dtype': keyword_7319, 'order': keyword_7325, 'offset': keyword_7323}
        # Getting the type of 'ndarray' (line 262)
        ndarray_7314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'ndarray', False)
        # Obtaining the member '__new__' of a type (line 262)
        new___7315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), ndarray_7314, '__new__')
        # Calling __new__(args, kwargs) (line 262)
        new___call_result_7327 = invoke(stypy.reporting.localization.Localization(__file__, 262, 15), new___7315, *[subtype_7316, shape_7317], **kwargs_7326)
        
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', new___call_result_7327)
        
        # Assigning a Name to a Attribute (line 264):
        # Getting the type of 'mm' (line 264)
        mm_7328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 21), 'mm')
        # Getting the type of 'self' (line 264)
        self_7329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
        # Setting the type of the member '_mmap' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_7329, '_mmap', mm_7328)
        
        # Assigning a Name to a Attribute (line 265):
        # Getting the type of 'offset' (line 265)
        offset_7330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'offset')
        # Getting the type of 'self' (line 265)
        self_7331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self')
        # Setting the type of the member 'offset' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_7331, 'offset', offset_7330)
        
        # Assigning a Name to a Attribute (line 266):
        # Getting the type of 'mode' (line 266)
        mode_7332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'mode')
        # Getting the type of 'self' (line 266)
        self_7333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self')
        # Setting the type of the member 'mode' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_7333, 'mode', mode_7332)
        
        # Type idiom detected: calculating its left and rigth part (line 268)
        # Getting the type of 'basestring' (line 268)
        basestring_7334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 32), 'basestring')
        # Getting the type of 'filename' (line 268)
        filename_7335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'filename')
        
        (may_be_7336, more_types_in_union_7337) = may_be_subtype(basestring_7334, filename_7335)

        if may_be_7336:

            if more_types_in_union_7337:
                # Runtime conditional SSA (line 268)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'filename' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'filename', remove_not_subtype_from_union(filename_7335, basestring))
            
            # Assigning a Call to a Attribute (line 269):
            
            # Call to abspath(...): (line 269)
            # Processing the call arguments (line 269)
            # Getting the type of 'filename' (line 269)
            filename_7341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 44), 'filename', False)
            # Processing the call keyword arguments (line 269)
            kwargs_7342 = {}
            # Getting the type of 'os' (line 269)
            os_7338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'os', False)
            # Obtaining the member 'path' of a type (line 269)
            path_7339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 28), os_7338, 'path')
            # Obtaining the member 'abspath' of a type (line 269)
            abspath_7340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 28), path_7339, 'abspath')
            # Calling abspath(args, kwargs) (line 269)
            abspath_call_result_7343 = invoke(stypy.reporting.localization.Localization(__file__, 269, 28), abspath_7340, *[filename_7341], **kwargs_7342)
            
            # Getting the type of 'self' (line 269)
            self_7344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'self')
            # Setting the type of the member 'filename' of a type (line 269)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), self_7344, 'filename', abspath_call_result_7343)

            if more_types_in_union_7337:
                # Runtime conditional SSA for else branch (line 268)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_7336) or more_types_in_union_7337):
            # Assigning a type to the variable 'filename' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'filename', remove_subtype_from_union(filename_7335, basestring))
            
            
            # Evaluating a boolean operation
            
            # Call to hasattr(...): (line 271)
            # Processing the call arguments (line 271)
            # Getting the type of 'filename' (line 271)
            filename_7346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'filename', False)
            str_7347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 32), 'str', 'name')
            # Processing the call keyword arguments (line 271)
            kwargs_7348 = {}
            # Getting the type of 'hasattr' (line 271)
            hasattr_7345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 271)
            hasattr_call_result_7349 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), hasattr_7345, *[filename_7346, str_7347], **kwargs_7348)
            
            
            # Call to isinstance(...): (line 272)
            # Processing the call arguments (line 272)
            # Getting the type of 'filename' (line 272)
            filename_7351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 25), 'filename', False)
            # Obtaining the member 'name' of a type (line 272)
            name_7352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 25), filename_7351, 'name')
            # Getting the type of 'basestring' (line 272)
            basestring_7353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 40), 'basestring', False)
            # Processing the call keyword arguments (line 272)
            kwargs_7354 = {}
            # Getting the type of 'isinstance' (line 272)
            isinstance_7350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 14), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 272)
            isinstance_call_result_7355 = invoke(stypy.reporting.localization.Localization(__file__, 272, 14), isinstance_7350, *[name_7352, basestring_7353], **kwargs_7354)
            
            # Applying the binary operator 'and' (line 271)
            result_and_keyword_7356 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 14), 'and', hasattr_call_result_7349, isinstance_call_result_7355)
            
            # Testing the type of an if condition (line 271)
            if_condition_7357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 13), result_and_keyword_7356)
            # Assigning a type to the variable 'if_condition_7357' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'if_condition_7357', if_condition_7357)
            # SSA begins for if statement (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 273):
            
            # Call to abspath(...): (line 273)
            # Processing the call arguments (line 273)
            # Getting the type of 'filename' (line 273)
            filename_7361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 44), 'filename', False)
            # Obtaining the member 'name' of a type (line 273)
            name_7362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 44), filename_7361, 'name')
            # Processing the call keyword arguments (line 273)
            kwargs_7363 = {}
            # Getting the type of 'os' (line 273)
            os_7358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 28), 'os', False)
            # Obtaining the member 'path' of a type (line 273)
            path_7359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 28), os_7358, 'path')
            # Obtaining the member 'abspath' of a type (line 273)
            abspath_7360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 28), path_7359, 'abspath')
            # Calling abspath(args, kwargs) (line 273)
            abspath_call_result_7364 = invoke(stypy.reporting.localization.Localization(__file__, 273, 28), abspath_7360, *[name_7362], **kwargs_7363)
            
            # Getting the type of 'self' (line 273)
            self_7365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'self')
            # Setting the type of the member 'filename' of a type (line 273)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), self_7365, 'filename', abspath_call_result_7364)
            # SSA branch for the else part of an if statement (line 271)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 276):
            # Getting the type of 'None' (line 276)
            None_7366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'None')
            # Getting the type of 'self' (line 276)
            self_7367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'self')
            # Setting the type of the member 'filename' of a type (line 276)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), self_7367, 'filename', None_7366)
            # SSA join for if statement (line 271)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_7336 and more_types_in_union_7337):
                # SSA join for if statement (line 268)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'own_file' (line 278)
        own_file_7368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'own_file')
        # Testing the type of an if condition (line 278)
        if_condition_7369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 8), own_file_7368)
        # Assigning a type to the variable 'if_condition_7369' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'if_condition_7369', if_condition_7369)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 279)
        # Processing the call keyword arguments (line 279)
        kwargs_7372 = {}
        # Getting the type of 'fid' (line 279)
        fid_7370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'fid', False)
        # Obtaining the member 'close' of a type (line 279)
        close_7371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), fid_7370, 'close')
        # Calling close(args, kwargs) (line 279)
        close_call_result_7373 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), close_7371, *[], **kwargs_7372)
        
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 281)
        self_7374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stypy_return_type', self_7374)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_7375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_7375


    @norecursion
    def __array_finalize__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_finalize__'
        module_type_store = module_type_store.open_function_context('__array_finalize__', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        memmap.__array_finalize__.__dict__.__setitem__('stypy_localization', localization)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_type_store', module_type_store)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_function_name', 'memmap.__array_finalize__')
        memmap.__array_finalize__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        memmap.__array_finalize__.__dict__.__setitem__('stypy_varargs_param_name', None)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_call_defaults', defaults)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_call_varargs', varargs)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        memmap.__array_finalize__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'memmap.__array_finalize__', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array_finalize__', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array_finalize__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'obj' (line 284)
        obj_7377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'obj', False)
        str_7378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'str', '_mmap')
        # Processing the call keyword arguments (line 284)
        kwargs_7379 = {}
        # Getting the type of 'hasattr' (line 284)
        hasattr_7376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 284)
        hasattr_call_result_7380 = invoke(stypy.reporting.localization.Localization(__file__, 284, 11), hasattr_7376, *[obj_7377, str_7378], **kwargs_7379)
        
        
        # Call to may_share_memory(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'self' (line 284)
        self_7383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 57), 'self', False)
        # Getting the type of 'obj' (line 284)
        obj_7384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 63), 'obj', False)
        # Processing the call keyword arguments (line 284)
        kwargs_7385 = {}
        # Getting the type of 'np' (line 284)
        np_7381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 37), 'np', False)
        # Obtaining the member 'may_share_memory' of a type (line 284)
        may_share_memory_7382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 37), np_7381, 'may_share_memory')
        # Calling may_share_memory(args, kwargs) (line 284)
        may_share_memory_call_result_7386 = invoke(stypy.reporting.localization.Localization(__file__, 284, 37), may_share_memory_7382, *[self_7383, obj_7384], **kwargs_7385)
        
        # Applying the binary operator 'and' (line 284)
        result_and_keyword_7387 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'and', hasattr_call_result_7380, may_share_memory_call_result_7386)
        
        # Testing the type of an if condition (line 284)
        if_condition_7388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_and_keyword_7387)
        # Assigning a type to the variable 'if_condition_7388' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_7388', if_condition_7388)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 285):
        # Getting the type of 'obj' (line 285)
        obj_7389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'obj')
        # Obtaining the member '_mmap' of a type (line 285)
        _mmap_7390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 25), obj_7389, '_mmap')
        # Getting the type of 'self' (line 285)
        self_7391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'self')
        # Setting the type of the member '_mmap' of a type (line 285)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), self_7391, '_mmap', _mmap_7390)
        
        # Assigning a Attribute to a Attribute (line 286):
        # Getting the type of 'obj' (line 286)
        obj_7392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 28), 'obj')
        # Obtaining the member 'filename' of a type (line 286)
        filename_7393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 28), obj_7392, 'filename')
        # Getting the type of 'self' (line 286)
        self_7394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'self')
        # Setting the type of the member 'filename' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), self_7394, 'filename', filename_7393)
        
        # Assigning a Attribute to a Attribute (line 287):
        # Getting the type of 'obj' (line 287)
        obj_7395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'obj')
        # Obtaining the member 'offset' of a type (line 287)
        offset_7396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 26), obj_7395, 'offset')
        # Getting the type of 'self' (line 287)
        self_7397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'self')
        # Setting the type of the member 'offset' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 12), self_7397, 'offset', offset_7396)
        
        # Assigning a Attribute to a Attribute (line 288):
        # Getting the type of 'obj' (line 288)
        obj_7398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'obj')
        # Obtaining the member 'mode' of a type (line 288)
        mode_7399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 24), obj_7398, 'mode')
        # Getting the type of 'self' (line 288)
        self_7400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'self')
        # Setting the type of the member 'mode' of a type (line 288)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), self_7400, 'mode', mode_7399)
        # SSA branch for the else part of an if statement (line 284)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 290):
        # Getting the type of 'None' (line 290)
        None_7401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 25), 'None')
        # Getting the type of 'self' (line 290)
        self_7402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'self')
        # Setting the type of the member '_mmap' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), self_7402, '_mmap', None_7401)
        
        # Assigning a Name to a Attribute (line 291):
        # Getting the type of 'None' (line 291)
        None_7403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'None')
        # Getting the type of 'self' (line 291)
        self_7404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'self')
        # Setting the type of the member 'filename' of a type (line 291)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), self_7404, 'filename', None_7403)
        
        # Assigning a Name to a Attribute (line 292):
        # Getting the type of 'None' (line 292)
        None_7405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'None')
        # Getting the type of 'self' (line 292)
        self_7406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'self')
        # Setting the type of the member 'offset' of a type (line 292)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), self_7406, 'offset', None_7405)
        
        # Assigning a Name to a Attribute (line 293):
        # Getting the type of 'None' (line 293)
        None_7407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'None')
        # Getting the type of 'self' (line 293)
        self_7408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'self')
        # Setting the type of the member 'mode' of a type (line 293)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), self_7408, 'mode', None_7407)
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__array_finalize__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_finalize__' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_7409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7409)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_finalize__'
        return stypy_return_type_7409


    @norecursion
    def flush(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flush'
        module_type_store = module_type_store.open_function_context('flush', 295, 4, False)
        # Assigning a type to the variable 'self' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        memmap.flush.__dict__.__setitem__('stypy_localization', localization)
        memmap.flush.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        memmap.flush.__dict__.__setitem__('stypy_type_store', module_type_store)
        memmap.flush.__dict__.__setitem__('stypy_function_name', 'memmap.flush')
        memmap.flush.__dict__.__setitem__('stypy_param_names_list', [])
        memmap.flush.__dict__.__setitem__('stypy_varargs_param_name', None)
        memmap.flush.__dict__.__setitem__('stypy_kwargs_param_name', None)
        memmap.flush.__dict__.__setitem__('stypy_call_defaults', defaults)
        memmap.flush.__dict__.__setitem__('stypy_call_varargs', varargs)
        memmap.flush.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        memmap.flush.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'memmap.flush', [], None, None, defaults, varargs, kwargs)

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

        str_7410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, (-1)), 'str', '\n        Write any changes in the array to the file on disk.\n\n        For further information, see `memmap`.\n\n        Parameters\n        ----------\n        None\n\n        See Also\n        --------\n        memmap\n\n        ')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 310)
        self_7411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'self')
        # Obtaining the member 'base' of a type (line 310)
        base_7412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 11), self_7411, 'base')
        # Getting the type of 'None' (line 310)
        None_7413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'None')
        # Applying the binary operator 'isnot' (line 310)
        result_is_not_7414 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 11), 'isnot', base_7412, None_7413)
        
        
        # Call to hasattr(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'self' (line 310)
        self_7416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 45), 'self', False)
        # Obtaining the member 'base' of a type (line 310)
        base_7417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 45), self_7416, 'base')
        str_7418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 56), 'str', 'flush')
        # Processing the call keyword arguments (line 310)
        kwargs_7419 = {}
        # Getting the type of 'hasattr' (line 310)
        hasattr_7415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 37), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 310)
        hasattr_call_result_7420 = invoke(stypy.reporting.localization.Localization(__file__, 310, 37), hasattr_7415, *[base_7417, str_7418], **kwargs_7419)
        
        # Applying the binary operator 'and' (line 310)
        result_and_keyword_7421 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 11), 'and', result_is_not_7414, hasattr_call_result_7420)
        
        # Testing the type of an if condition (line 310)
        if_condition_7422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 8), result_and_keyword_7421)
        # Assigning a type to the variable 'if_condition_7422' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'if_condition_7422', if_condition_7422)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to flush(...): (line 311)
        # Processing the call keyword arguments (line 311)
        kwargs_7426 = {}
        # Getting the type of 'self' (line 311)
        self_7423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'self', False)
        # Obtaining the member 'base' of a type (line 311)
        base_7424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), self_7423, 'base')
        # Obtaining the member 'flush' of a type (line 311)
        flush_7425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), base_7424, 'flush')
        # Calling flush(args, kwargs) (line 311)
        flush_call_result_7427 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), flush_7425, *[], **kwargs_7426)
        
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'flush(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flush' in the type store
        # Getting the type of 'stypy_return_type' (line 295)
        stypy_return_type_7428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7428)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flush'
        return stypy_return_type_7428


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'memmap.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'memmap' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'memmap', memmap)

# Assigning a Num to a Name (line 199):
float_7429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 25), 'float')
# Getting the type of 'memmap'
memmap_7430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'memmap')
# Setting the type of the member '__array_priority__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), memmap_7430, '__array_priority__', float_7429)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
