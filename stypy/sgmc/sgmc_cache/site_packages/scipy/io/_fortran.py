
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module to read / write Fortran unformatted sequential files.
3: 
4: This is in the spirit of code written by Neil Martinsen-Burrell and Joe Zuntz.
5: 
6: '''
7: from __future__ import division, print_function, absolute_import
8: 
9: import warnings
10: import numpy as np
11: 
12: __all__ = ['FortranFile']
13: 
14: 
15: class FortranFile(object):
16:     '''
17:     A file object for unformatted sequential files from Fortran code.
18: 
19:     Parameters
20:     ----------
21:     filename : file or str
22:         Open file object or filename.
23:     mode : {'r', 'w'}, optional
24:         Read-write mode, default is 'r'.
25:     header_dtype : dtype, optional
26:         Data type of the header. Size and endiness must match the input/output file.
27: 
28:     Notes
29:     -----
30:     These files are broken up into records of unspecified types. The size of
31:     each record is given at the start (although the size of this header is not
32:     standard) and the data is written onto disk without any formatting. Fortran
33:     compilers supporting the BACKSPACE statement will write a second copy of
34:     the size to facilitate backwards seeking.
35: 
36:     This class only supports files written with both sizes for the record.
37:     It also does not support the subrecords used in Intel and gfortran compilers
38:     for records which are greater than 2GB with a 4-byte header.
39: 
40:     An example of an unformatted sequential file in Fortran would be written as::
41: 
42:         OPEN(1, FILE=myfilename, FORM='unformatted')
43: 
44:         WRITE(1) myvariable
45: 
46:     Since this is a non-standard file format, whose contents depend on the
47:     compiler and the endianness of the machine, caution is advised. Files from
48:     gfortran 4.8.0 and gfortran 4.1.2 on x86_64 are known to work.
49: 
50:     Consider using Fortran direct-access files or files from the newer Stream
51:     I/O, which can be easily read by `numpy.fromfile`.
52: 
53:     Examples
54:     --------
55:     To create an unformatted sequential Fortran file:
56: 
57:     >>> from scipy.io import FortranFile
58:     >>> f = FortranFile('test.unf', 'w')
59:     >>> f.write_record(np.array([1,2,3,4,5], dtype=np.int32))
60:     >>> f.write_record(np.linspace(0,1,20).reshape((5,4)).T)
61:     >>> f.close()
62: 
63:     To read this file:
64: 
65:     >>> f = FortranFile('test.unf', 'r')
66:     >>> print(f.read_ints(np.int32))
67:     [1 2 3 4 5]
68:     >>> print(f.read_reals(float).reshape((5,4), order="F"))
69:     [[ 0.          0.05263158  0.10526316  0.15789474]
70:      [ 0.21052632  0.26315789  0.31578947  0.36842105]
71:      [ 0.42105263  0.47368421  0.52631579  0.57894737]
72:      [ 0.63157895  0.68421053  0.73684211  0.78947368]
73:      [ 0.84210526  0.89473684  0.94736842  1.        ]]
74:     >>> f.close()
75: 
76:     Or, in Fortran::
77: 
78:         integer :: a(5), i
79:         double precision :: b(5,4)
80:         open(1, file='test.unf', form='unformatted')
81:         read(1) a
82:         read(1) b
83:         close(1)
84:         write(*,*) a
85:         do i = 1, 5
86:             write(*,*) b(i,:)
87:         end do
88: 
89:     '''
90:     def __init__(self, filename, mode='r', header_dtype=np.uint32):
91:         if header_dtype is None:
92:             raise ValueError('Must specify dtype')
93: 
94:         header_dtype = np.dtype(header_dtype)
95:         if header_dtype.kind != 'u':
96:             warnings.warn("Given a dtype which is not unsigned.")
97: 
98:         if mode not in 'rw' or len(mode) != 1:
99:             raise ValueError('mode must be either r or w')
100: 
101:         if hasattr(filename, 'seek'):
102:             self._fp = filename
103:         else:
104:             self._fp = open(filename, '%sb' % mode)
105: 
106:         self._header_dtype = header_dtype
107: 
108:     def _read_size(self):
109:         return int(np.fromfile(self._fp, dtype=self._header_dtype, count=1))
110: 
111:     def write_record(self, *items):
112:         '''
113:         Write a record (including sizes) to the file.
114: 
115:         Parameters
116:         ----------
117:         *items : array_like
118:             The data arrays to write.
119: 
120:         Notes
121:         -----
122:         Writes data items to a file::
123: 
124:             write_record(a.T, b.T, c.T, ...)
125: 
126:             write(1) a, b, c, ...
127: 
128:         Note that data in multidimensional arrays is written in
129:         row-major order --- to make them read correctly by Fortran
130:         programs, you need to transpose the arrays yourself when
131:         writing them.
132: 
133:         '''
134:         items = tuple(np.asarray(item) for item in items)
135:         total_size = sum(item.nbytes for item in items)
136: 
137:         nb = np.array([total_size], dtype=self._header_dtype)
138: 
139:         nb.tofile(self._fp)
140:         for item in items:
141:             item.tofile(self._fp)
142:         nb.tofile(self._fp)
143: 
144:     def read_record(self, *dtypes, **kwargs):
145:         '''
146:         Reads a record of a given type from the file.
147: 
148:         Parameters
149:         ----------
150:         *dtypes : dtypes, optional
151:             Data type(s) specifying the size and endiness of the data.
152: 
153:         Returns
154:         -------
155:         data : ndarray
156:             A one-dimensional array object.
157: 
158:         Notes
159:         -----
160:         If the record contains a multi-dimensional array, you can specify
161:         the size in the dtype. For example::
162: 
163:             INTEGER var(5,4)
164: 
165:         can be read with::
166: 
167:             read_record('(4,5)i4').T
168: 
169:         Note that this function does **not** assume the file data is in Fortran
170:         column major order, so you need to (i) swap the order of dimensions
171:         when reading and (ii) transpose the resulting array.
172: 
173:         Alternatively, you can read the data as a 1D array and handle the
174:         ordering yourself. For example::
175: 
176:             read_record('i4').reshape(5, 4, order='F')
177: 
178:         For records that contain several variables or mixed types (as opposed
179:         to single scalar or array types), give them as separate arguments::
180: 
181:             double precision :: a
182:             integer :: b
183:             write(1) a, b
184: 
185:             record = f.read_record('<f4', '<i4')
186:             a = record[0]  # first number
187:             b = record[1]  # second number
188: 
189:         and if any of the variables are arrays, the shape can be specified as
190:         the third item in the relevant dtype::
191: 
192:             double precision :: a
193:             integer :: b(3,4)
194:             write(1) a, b
195: 
196:             record = f.read_record('<f4', np.dtype(('<i4', (4, 3))))
197:             a = record[0]
198:             b = record[1].T
199: 
200:         Numpy also supports a short syntax for this kind of type::
201: 
202:             record = f.read_record('<f4', '(3,3)<i4')
203: 
204:         See Also
205:         --------
206:         read_reals
207:         read_ints
208: 
209:         '''
210:         dtype = kwargs.pop('dtype', None)
211:         if kwargs:
212:             raise ValueError("Unknown keyword arguments {}".format(tuple(kwargs.keys())))
213: 
214:         if dtype is not None:
215:             dtypes = dtypes + (dtype,)
216:         elif not dtypes:
217:             raise ValueError('Must specify at least one dtype')
218: 
219:         first_size = self._read_size()
220: 
221:         dtypes = tuple(np.dtype(dtype) for dtype in dtypes)
222:         block_size = sum(dtype.itemsize for dtype in dtypes)
223: 
224:         num_blocks, remainder = divmod(first_size, block_size)
225:         if remainder != 0:
226:             raise ValueError('Size obtained ({0}) is not a multiple of the '
227:                              'dtypes given ({1}).'.format(first_size, block_size))
228: 
229:         if len(dtypes) != 1 and first_size != block_size:
230:             # Fortran does not write mixed type array items in interleaved order,
231:             # and it's not possible to guess the sizes of the arrays that were written.
232:             # The user must specify the exact sizes of each of the arrays.
233:             raise ValueError('Size obtained ({0}) does not match with the expected '
234:                              'size ({1}) of multi-item record'.format(first_size, block_size))
235: 
236:         data = []
237:         for dtype in dtypes:
238:             r = np.fromfile(self._fp, dtype=dtype, count=num_blocks)
239:             if dtype.shape != ():
240:                 # Squeeze outmost block dimension for array items
241:                 if num_blocks == 1:
242:                     assert r.shape == (1,) + dtype.shape
243:                     r = r[0]
244: 
245:             data.append(r)
246: 
247:         second_size = self._read_size()
248:         if first_size != second_size:
249:             raise IOError('Sizes do not agree in the header and footer for '
250:                           'this record - check header dtype')
251: 
252:         # Unpack result
253:         if len(dtypes) == 1:
254:             return data[0]
255:         else:
256:             return tuple(data)
257: 
258:     def read_ints(self, dtype='i4'):
259:         '''
260:         Reads a record of a given type from the file, defaulting to an integer
261:         type (``INTEGER*4`` in Fortran).
262: 
263:         Parameters
264:         ----------
265:         dtype : dtype, optional
266:             Data type specifying the size and endiness of the data.
267: 
268:         Returns
269:         -------
270:         data : ndarray
271:             A one-dimensional array object.
272: 
273:         See Also
274:         --------
275:         read_reals
276:         read_record
277: 
278:         '''
279:         return self.read_record(dtype)
280: 
281:     def read_reals(self, dtype='f8'):
282:         '''
283:         Reads a record of a given type from the file, defaulting to a floating
284:         point number (``real*8`` in Fortran).
285: 
286:         Parameters
287:         ----------
288:         dtype : dtype, optional
289:             Data type specifying the size and endiness of the data.
290: 
291:         Returns
292:         -------
293:         data : ndarray
294:             A one-dimensional array object.
295: 
296:         See Also
297:         --------
298:         read_ints
299:         read_record
300: 
301:         '''
302:         return self.read_record(dtype)
303: 
304:     def close(self):
305:         '''
306:         Closes the file. It is unsupported to call any other methods off this
307:         object after closing it. Note that this class supports the 'with'
308:         statement in modern versions of Python, to call this automatically
309: 
310:         '''
311:         self._fp.close()
312: 
313:     def __enter__(self):
314:         return self
315: 
316:     def __exit__(self, type, value, tb):
317:         self.close()
318: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_127875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nModule to read / write Fortran unformatted sequential files.\n\nThis is in the spirit of code written by Neil Martinsen-Burrell and Joe Zuntz.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import warnings' statement (line 9)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/')
import_127876 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_127876) is not StypyTypeError):

    if (import_127876 != 'pyd_module'):
        __import__(import_127876)
        sys_modules_127877 = sys.modules[import_127876]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_127877.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_127876)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/')


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['FortranFile']
module_type_store.set_exportable_members(['FortranFile'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_127878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_127879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'FortranFile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_127878, str_127879)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_127878)
# Declaration of the 'FortranFile' class

class FortranFile(object, ):
    str_127880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n    A file object for unformatted sequential files from Fortran code.\n\n    Parameters\n    ----------\n    filename : file or str\n        Open file object or filename.\n    mode : {\'r\', \'w\'}, optional\n        Read-write mode, default is \'r\'.\n    header_dtype : dtype, optional\n        Data type of the header. Size and endiness must match the input/output file.\n\n    Notes\n    -----\n    These files are broken up into records of unspecified types. The size of\n    each record is given at the start (although the size of this header is not\n    standard) and the data is written onto disk without any formatting. Fortran\n    compilers supporting the BACKSPACE statement will write a second copy of\n    the size to facilitate backwards seeking.\n\n    This class only supports files written with both sizes for the record.\n    It also does not support the subrecords used in Intel and gfortran compilers\n    for records which are greater than 2GB with a 4-byte header.\n\n    An example of an unformatted sequential file in Fortran would be written as::\n\n        OPEN(1, FILE=myfilename, FORM=\'unformatted\')\n\n        WRITE(1) myvariable\n\n    Since this is a non-standard file format, whose contents depend on the\n    compiler and the endianness of the machine, caution is advised. Files from\n    gfortran 4.8.0 and gfortran 4.1.2 on x86_64 are known to work.\n\n    Consider using Fortran direct-access files or files from the newer Stream\n    I/O, which can be easily read by `numpy.fromfile`.\n\n    Examples\n    --------\n    To create an unformatted sequential Fortran file:\n\n    >>> from scipy.io import FortranFile\n    >>> f = FortranFile(\'test.unf\', \'w\')\n    >>> f.write_record(np.array([1,2,3,4,5], dtype=np.int32))\n    >>> f.write_record(np.linspace(0,1,20).reshape((5,4)).T)\n    >>> f.close()\n\n    To read this file:\n\n    >>> f = FortranFile(\'test.unf\', \'r\')\n    >>> print(f.read_ints(np.int32))\n    [1 2 3 4 5]\n    >>> print(f.read_reals(float).reshape((5,4), order="F"))\n    [[ 0.          0.05263158  0.10526316  0.15789474]\n     [ 0.21052632  0.26315789  0.31578947  0.36842105]\n     [ 0.42105263  0.47368421  0.52631579  0.57894737]\n     [ 0.63157895  0.68421053  0.73684211  0.78947368]\n     [ 0.84210526  0.89473684  0.94736842  1.        ]]\n    >>> f.close()\n\n    Or, in Fortran::\n\n        integer :: a(5), i\n        double precision :: b(5,4)\n        open(1, file=\'test.unf\', form=\'unformatted\')\n        read(1) a\n        read(1) b\n        close(1)\n        write(*,*) a\n        do i = 1, 5\n            write(*,*) b(i,:)\n        end do\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_127881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 38), 'str', 'r')
        # Getting the type of 'np' (line 90)
        np_127882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 56), 'np')
        # Obtaining the member 'uint32' of a type (line 90)
        uint32_127883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 56), np_127882, 'uint32')
        defaults = [str_127881, uint32_127883]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.__init__', ['filename', 'mode', 'header_dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename', 'mode', 'header_dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 91)
        # Getting the type of 'header_dtype' (line 91)
        header_dtype_127884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'header_dtype')
        # Getting the type of 'None' (line 91)
        None_127885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'None')
        
        (may_be_127886, more_types_in_union_127887) = may_be_none(header_dtype_127884, None_127885)

        if may_be_127886:

            if more_types_in_union_127887:
                # Runtime conditional SSA (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 92)
            # Processing the call arguments (line 92)
            str_127889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'str', 'Must specify dtype')
            # Processing the call keyword arguments (line 92)
            kwargs_127890 = {}
            # Getting the type of 'ValueError' (line 92)
            ValueError_127888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 92)
            ValueError_call_result_127891 = invoke(stypy.reporting.localization.Localization(__file__, 92, 18), ValueError_127888, *[str_127889], **kwargs_127890)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 92, 12), ValueError_call_result_127891, 'raise parameter', BaseException)

            if more_types_in_union_127887:
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to dtype(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'header_dtype' (line 94)
        header_dtype_127894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'header_dtype', False)
        # Processing the call keyword arguments (line 94)
        kwargs_127895 = {}
        # Getting the type of 'np' (line 94)
        np_127892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'np', False)
        # Obtaining the member 'dtype' of a type (line 94)
        dtype_127893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 23), np_127892, 'dtype')
        # Calling dtype(args, kwargs) (line 94)
        dtype_call_result_127896 = invoke(stypy.reporting.localization.Localization(__file__, 94, 23), dtype_127893, *[header_dtype_127894], **kwargs_127895)
        
        # Assigning a type to the variable 'header_dtype' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'header_dtype', dtype_call_result_127896)
        
        
        # Getting the type of 'header_dtype' (line 95)
        header_dtype_127897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'header_dtype')
        # Obtaining the member 'kind' of a type (line 95)
        kind_127898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 11), header_dtype_127897, 'kind')
        str_127899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 32), 'str', 'u')
        # Applying the binary operator '!=' (line 95)
        result_ne_127900 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), '!=', kind_127898, str_127899)
        
        # Testing the type of an if condition (line 95)
        if_condition_127901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), result_ne_127900)
        # Assigning a type to the variable 'if_condition_127901' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_127901', if_condition_127901)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 96)
        # Processing the call arguments (line 96)
        str_127904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'str', 'Given a dtype which is not unsigned.')
        # Processing the call keyword arguments (line 96)
        kwargs_127905 = {}
        # Getting the type of 'warnings' (line 96)
        warnings_127902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 96)
        warn_127903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), warnings_127902, 'warn')
        # Calling warn(args, kwargs) (line 96)
        warn_call_result_127906 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), warn_127903, *[str_127904], **kwargs_127905)
        
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mode' (line 98)
        mode_127907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'mode')
        str_127908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'str', 'rw')
        # Applying the binary operator 'notin' (line 98)
        result_contains_127909 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), 'notin', mode_127907, str_127908)
        
        
        
        # Call to len(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'mode' (line 98)
        mode_127911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'mode', False)
        # Processing the call keyword arguments (line 98)
        kwargs_127912 = {}
        # Getting the type of 'len' (line 98)
        len_127910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'len', False)
        # Calling len(args, kwargs) (line 98)
        len_call_result_127913 = invoke(stypy.reporting.localization.Localization(__file__, 98, 31), len_127910, *[mode_127911], **kwargs_127912)
        
        int_127914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 44), 'int')
        # Applying the binary operator '!=' (line 98)
        result_ne_127915 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 31), '!=', len_call_result_127913, int_127914)
        
        # Applying the binary operator 'or' (line 98)
        result_or_keyword_127916 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), 'or', result_contains_127909, result_ne_127915)
        
        # Testing the type of an if condition (line 98)
        if_condition_127917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_or_keyword_127916)
        # Assigning a type to the variable 'if_condition_127917' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_127917', if_condition_127917)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 99)
        # Processing the call arguments (line 99)
        str_127919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'str', 'mode must be either r or w')
        # Processing the call keyword arguments (line 99)
        kwargs_127920 = {}
        # Getting the type of 'ValueError' (line 99)
        ValueError_127918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 99)
        ValueError_call_result_127921 = invoke(stypy.reporting.localization.Localization(__file__, 99, 18), ValueError_127918, *[str_127919], **kwargs_127920)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 12), ValueError_call_result_127921, 'raise parameter', BaseException)
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 101)
        str_127922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'str', 'seek')
        # Getting the type of 'filename' (line 101)
        filename_127923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'filename')
        
        (may_be_127924, more_types_in_union_127925) = may_provide_member(str_127922, filename_127923)

        if may_be_127924:

            if more_types_in_union_127925:
                # Runtime conditional SSA (line 101)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'filename' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'filename', remove_not_member_provider_from_union(filename_127923, 'seek'))
            
            # Assigning a Name to a Attribute (line 102):
            
            # Assigning a Name to a Attribute (line 102):
            # Getting the type of 'filename' (line 102)
            filename_127926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'filename')
            # Getting the type of 'self' (line 102)
            self_127927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self')
            # Setting the type of the member '_fp' of a type (line 102)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_127927, '_fp', filename_127926)

            if more_types_in_union_127925:
                # Runtime conditional SSA for else branch (line 101)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_127924) or more_types_in_union_127925):
            # Assigning a type to the variable 'filename' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'filename', remove_member_provider_from_union(filename_127923, 'seek'))
            
            # Assigning a Call to a Attribute (line 104):
            
            # Assigning a Call to a Attribute (line 104):
            
            # Call to open(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'filename' (line 104)
            filename_127929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'filename', False)
            str_127930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 38), 'str', '%sb')
            # Getting the type of 'mode' (line 104)
            mode_127931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 46), 'mode', False)
            # Applying the binary operator '%' (line 104)
            result_mod_127932 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 38), '%', str_127930, mode_127931)
            
            # Processing the call keyword arguments (line 104)
            kwargs_127933 = {}
            # Getting the type of 'open' (line 104)
            open_127928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'open', False)
            # Calling open(args, kwargs) (line 104)
            open_call_result_127934 = invoke(stypy.reporting.localization.Localization(__file__, 104, 23), open_127928, *[filename_127929, result_mod_127932], **kwargs_127933)
            
            # Getting the type of 'self' (line 104)
            self_127935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self')
            # Setting the type of the member '_fp' of a type (line 104)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_127935, '_fp', open_call_result_127934)

            if (may_be_127924 and more_types_in_union_127925):
                # SSA join for if statement (line 101)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 106):
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'header_dtype' (line 106)
        header_dtype_127936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'header_dtype')
        # Getting the type of 'self' (line 106)
        self_127937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member '_header_dtype' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_127937, '_header_dtype', header_dtype_127936)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _read_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_size'
        module_type_store = module_type_store.open_function_context('_read_size', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile._read_size.__dict__.__setitem__('stypy_localization', localization)
        FortranFile._read_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile._read_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile._read_size.__dict__.__setitem__('stypy_function_name', 'FortranFile._read_size')
        FortranFile._read_size.__dict__.__setitem__('stypy_param_names_list', [])
        FortranFile._read_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFile._read_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFile._read_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile._read_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile._read_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile._read_size.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile._read_size', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_size', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_size(...)' code ##################

        
        # Call to int(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to fromfile(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_127941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'self', False)
        # Obtaining the member '_fp' of a type (line 109)
        _fp_127942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 31), self_127941, '_fp')
        # Processing the call keyword arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_127943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 47), 'self', False)
        # Obtaining the member '_header_dtype' of a type (line 109)
        _header_dtype_127944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 47), self_127943, '_header_dtype')
        keyword_127945 = _header_dtype_127944
        int_127946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 73), 'int')
        keyword_127947 = int_127946
        kwargs_127948 = {'count': keyword_127947, 'dtype': keyword_127945}
        # Getting the type of 'np' (line 109)
        np_127939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'np', False)
        # Obtaining the member 'fromfile' of a type (line 109)
        fromfile_127940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), np_127939, 'fromfile')
        # Calling fromfile(args, kwargs) (line 109)
        fromfile_call_result_127949 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), fromfile_127940, *[_fp_127942], **kwargs_127948)
        
        # Processing the call keyword arguments (line 109)
        kwargs_127950 = {}
        # Getting the type of 'int' (line 109)
        int_127938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'int', False)
        # Calling int(args, kwargs) (line 109)
        int_call_result_127951 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), int_127938, *[fromfile_call_result_127949], **kwargs_127950)
        
        # Assigning a type to the variable 'stypy_return_type' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type', int_call_result_127951)
        
        # ################# End of '_read_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_size' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_127952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_127952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_size'
        return stypy_return_type_127952


    @norecursion
    def write_record(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_record'
        module_type_store = module_type_store.open_function_context('write_record', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile.write_record.__dict__.__setitem__('stypy_localization', localization)
        FortranFile.write_record.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile.write_record.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile.write_record.__dict__.__setitem__('stypy_function_name', 'FortranFile.write_record')
        FortranFile.write_record.__dict__.__setitem__('stypy_param_names_list', [])
        FortranFile.write_record.__dict__.__setitem__('stypy_varargs_param_name', 'items')
        FortranFile.write_record.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFile.write_record.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile.write_record.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile.write_record.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile.write_record.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.write_record', [], 'items', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_record', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_record(...)' code ##################

        str_127953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', '\n        Write a record (including sizes) to the file.\n\n        Parameters\n        ----------\n        *items : array_like\n            The data arrays to write.\n\n        Notes\n        -----\n        Writes data items to a file::\n\n            write_record(a.T, b.T, c.T, ...)\n\n            write(1) a, b, c, ...\n\n        Note that data in multidimensional arrays is written in\n        row-major order --- to make them read correctly by Fortran\n        programs, you need to transpose the arrays yourself when\n        writing them.\n\n        ')
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to tuple(...): (line 134)
        # Processing the call arguments (line 134)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 134, 22, True)
        # Calculating comprehension expression
        # Getting the type of 'items' (line 134)
        items_127960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 51), 'items', False)
        comprehension_127961 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), items_127960)
        # Assigning a type to the variable 'item' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'item', comprehension_127961)
        
        # Call to asarray(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'item' (line 134)
        item_127957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'item', False)
        # Processing the call keyword arguments (line 134)
        kwargs_127958 = {}
        # Getting the type of 'np' (line 134)
        np_127955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'np', False)
        # Obtaining the member 'asarray' of a type (line 134)
        asarray_127956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 22), np_127955, 'asarray')
        # Calling asarray(args, kwargs) (line 134)
        asarray_call_result_127959 = invoke(stypy.reporting.localization.Localization(__file__, 134, 22), asarray_127956, *[item_127957], **kwargs_127958)
        
        list_127962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 22), list_127962, asarray_call_result_127959)
        # Processing the call keyword arguments (line 134)
        kwargs_127963 = {}
        # Getting the type of 'tuple' (line 134)
        tuple_127954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 134)
        tuple_call_result_127964 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), tuple_127954, *[list_127962], **kwargs_127963)
        
        # Assigning a type to the variable 'items' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'items', tuple_call_result_127964)
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to sum(...): (line 135)
        # Processing the call arguments (line 135)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 135, 25, True)
        # Calculating comprehension expression
        # Getting the type of 'items' (line 135)
        items_127968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 49), 'items', False)
        comprehension_127969 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 25), items_127968)
        # Assigning a type to the variable 'item' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'item', comprehension_127969)
        # Getting the type of 'item' (line 135)
        item_127966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'item', False)
        # Obtaining the member 'nbytes' of a type (line 135)
        nbytes_127967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 25), item_127966, 'nbytes')
        list_127970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 25), list_127970, nbytes_127967)
        # Processing the call keyword arguments (line 135)
        kwargs_127971 = {}
        # Getting the type of 'sum' (line 135)
        sum_127965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'sum', False)
        # Calling sum(args, kwargs) (line 135)
        sum_call_result_127972 = invoke(stypy.reporting.localization.Localization(__file__, 135, 21), sum_127965, *[list_127970], **kwargs_127971)
        
        # Assigning a type to the variable 'total_size' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'total_size', sum_call_result_127972)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to array(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_127975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        # Getting the type of 'total_size' (line 137)
        total_size_127976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'total_size', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 22), list_127975, total_size_127976)
        
        # Processing the call keyword arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_127977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'self', False)
        # Obtaining the member '_header_dtype' of a type (line 137)
        _header_dtype_127978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 42), self_127977, '_header_dtype')
        keyword_127979 = _header_dtype_127978
        kwargs_127980 = {'dtype': keyword_127979}
        # Getting the type of 'np' (line 137)
        np_127973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 137)
        array_127974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 13), np_127973, 'array')
        # Calling array(args, kwargs) (line 137)
        array_call_result_127981 = invoke(stypy.reporting.localization.Localization(__file__, 137, 13), array_127974, *[list_127975], **kwargs_127980)
        
        # Assigning a type to the variable 'nb' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'nb', array_call_result_127981)
        
        # Call to tofile(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'self' (line 139)
        self_127984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'self', False)
        # Obtaining the member '_fp' of a type (line 139)
        _fp_127985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 18), self_127984, '_fp')
        # Processing the call keyword arguments (line 139)
        kwargs_127986 = {}
        # Getting the type of 'nb' (line 139)
        nb_127982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'nb', False)
        # Obtaining the member 'tofile' of a type (line 139)
        tofile_127983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), nb_127982, 'tofile')
        # Calling tofile(args, kwargs) (line 139)
        tofile_call_result_127987 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), tofile_127983, *[_fp_127985], **kwargs_127986)
        
        
        # Getting the type of 'items' (line 140)
        items_127988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'items')
        # Testing the type of a for loop iterable (line 140)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 8), items_127988)
        # Getting the type of the for loop variable (line 140)
        for_loop_var_127989 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 8), items_127988)
        # Assigning a type to the variable 'item' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'item', for_loop_var_127989)
        # SSA begins for a for statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to tofile(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_127992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'self', False)
        # Obtaining the member '_fp' of a type (line 141)
        _fp_127993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 24), self_127992, '_fp')
        # Processing the call keyword arguments (line 141)
        kwargs_127994 = {}
        # Getting the type of 'item' (line 141)
        item_127990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'item', False)
        # Obtaining the member 'tofile' of a type (line 141)
        tofile_127991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), item_127990, 'tofile')
        # Calling tofile(args, kwargs) (line 141)
        tofile_call_result_127995 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), tofile_127991, *[_fp_127993], **kwargs_127994)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to tofile(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_127998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'self', False)
        # Obtaining the member '_fp' of a type (line 142)
        _fp_127999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 18), self_127998, '_fp')
        # Processing the call keyword arguments (line 142)
        kwargs_128000 = {}
        # Getting the type of 'nb' (line 142)
        nb_127996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'nb', False)
        # Obtaining the member 'tofile' of a type (line 142)
        tofile_127997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), nb_127996, 'tofile')
        # Calling tofile(args, kwargs) (line 142)
        tofile_call_result_128001 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), tofile_127997, *[_fp_127999], **kwargs_128000)
        
        
        # ################# End of 'write_record(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_record' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_128002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128002)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_record'
        return stypy_return_type_128002


    @norecursion
    def read_record(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_record'
        module_type_store = module_type_store.open_function_context('read_record', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile.read_record.__dict__.__setitem__('stypy_localization', localization)
        FortranFile.read_record.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile.read_record.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile.read_record.__dict__.__setitem__('stypy_function_name', 'FortranFile.read_record')
        FortranFile.read_record.__dict__.__setitem__('stypy_param_names_list', [])
        FortranFile.read_record.__dict__.__setitem__('stypy_varargs_param_name', 'dtypes')
        FortranFile.read_record.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FortranFile.read_record.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile.read_record.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile.read_record.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile.read_record.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.read_record', [], 'dtypes', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_record', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_record(...)' code ##################

        str_128003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, (-1)), 'str', "\n        Reads a record of a given type from the file.\n\n        Parameters\n        ----------\n        *dtypes : dtypes, optional\n            Data type(s) specifying the size and endiness of the data.\n\n        Returns\n        -------\n        data : ndarray\n            A one-dimensional array object.\n\n        Notes\n        -----\n        If the record contains a multi-dimensional array, you can specify\n        the size in the dtype. For example::\n\n            INTEGER var(5,4)\n\n        can be read with::\n\n            read_record('(4,5)i4').T\n\n        Note that this function does **not** assume the file data is in Fortran\n        column major order, so you need to (i) swap the order of dimensions\n        when reading and (ii) transpose the resulting array.\n\n        Alternatively, you can read the data as a 1D array and handle the\n        ordering yourself. For example::\n\n            read_record('i4').reshape(5, 4, order='F')\n\n        For records that contain several variables or mixed types (as opposed\n        to single scalar or array types), give them as separate arguments::\n\n            double precision :: a\n            integer :: b\n            write(1) a, b\n\n            record = f.read_record('<f4', '<i4')\n            a = record[0]  # first number\n            b = record[1]  # second number\n\n        and if any of the variables are arrays, the shape can be specified as\n        the third item in the relevant dtype::\n\n            double precision :: a\n            integer :: b(3,4)\n            write(1) a, b\n\n            record = f.read_record('<f4', np.dtype(('<i4', (4, 3))))\n            a = record[0]\n            b = record[1].T\n\n        Numpy also supports a short syntax for this kind of type::\n\n            record = f.read_record('<f4', '(3,3)<i4')\n\n        See Also\n        --------\n        read_reals\n        read_ints\n\n        ")
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to pop(...): (line 210)
        # Processing the call arguments (line 210)
        str_128006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 27), 'str', 'dtype')
        # Getting the type of 'None' (line 210)
        None_128007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'None', False)
        # Processing the call keyword arguments (line 210)
        kwargs_128008 = {}
        # Getting the type of 'kwargs' (line 210)
        kwargs_128004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 210)
        pop_128005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), kwargs_128004, 'pop')
        # Calling pop(args, kwargs) (line 210)
        pop_call_result_128009 = invoke(stypy.reporting.localization.Localization(__file__, 210, 16), pop_128005, *[str_128006, None_128007], **kwargs_128008)
        
        # Assigning a type to the variable 'dtype' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'dtype', pop_call_result_128009)
        
        # Getting the type of 'kwargs' (line 211)
        kwargs_128010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'kwargs')
        # Testing the type of an if condition (line 211)
        if_condition_128011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), kwargs_128010)
        # Assigning a type to the variable 'if_condition_128011' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_128011', if_condition_128011)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Call to format(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Call to tuple(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Call to keys(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_128018 = {}
        # Getting the type of 'kwargs' (line 212)
        kwargs_128016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 73), 'kwargs', False)
        # Obtaining the member 'keys' of a type (line 212)
        keys_128017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 73), kwargs_128016, 'keys')
        # Calling keys(args, kwargs) (line 212)
        keys_call_result_128019 = invoke(stypy.reporting.localization.Localization(__file__, 212, 73), keys_128017, *[], **kwargs_128018)
        
        # Processing the call keyword arguments (line 212)
        kwargs_128020 = {}
        # Getting the type of 'tuple' (line 212)
        tuple_128015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 67), 'tuple', False)
        # Calling tuple(args, kwargs) (line 212)
        tuple_call_result_128021 = invoke(stypy.reporting.localization.Localization(__file__, 212, 67), tuple_128015, *[keys_call_result_128019], **kwargs_128020)
        
        # Processing the call keyword arguments (line 212)
        kwargs_128022 = {}
        str_128013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 29), 'str', 'Unknown keyword arguments {}')
        # Obtaining the member 'format' of a type (line 212)
        format_128014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 29), str_128013, 'format')
        # Calling format(args, kwargs) (line 212)
        format_call_result_128023 = invoke(stypy.reporting.localization.Localization(__file__, 212, 29), format_128014, *[tuple_call_result_128021], **kwargs_128022)
        
        # Processing the call keyword arguments (line 212)
        kwargs_128024 = {}
        # Getting the type of 'ValueError' (line 212)
        ValueError_128012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 212)
        ValueError_call_result_128025 = invoke(stypy.reporting.localization.Localization(__file__, 212, 18), ValueError_128012, *[format_call_result_128023], **kwargs_128024)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 212, 12), ValueError_call_result_128025, 'raise parameter', BaseException)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 214)
        # Getting the type of 'dtype' (line 214)
        dtype_128026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'dtype')
        # Getting the type of 'None' (line 214)
        None_128027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'None')
        
        (may_be_128028, more_types_in_union_128029) = may_not_be_none(dtype_128026, None_128027)

        if may_be_128028:

            if more_types_in_union_128029:
                # Runtime conditional SSA (line 214)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 215):
            
            # Assigning a BinOp to a Name (line 215):
            # Getting the type of 'dtypes' (line 215)
            dtypes_128030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'dtypes')
            
            # Obtaining an instance of the builtin type 'tuple' (line 215)
            tuple_128031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 215)
            # Adding element type (line 215)
            # Getting the type of 'dtype' (line 215)
            dtype_128032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'dtype')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 31), tuple_128031, dtype_128032)
            
            # Applying the binary operator '+' (line 215)
            result_add_128033 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 21), '+', dtypes_128030, tuple_128031)
            
            # Assigning a type to the variable 'dtypes' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'dtypes', result_add_128033)

            if more_types_in_union_128029:
                # Runtime conditional SSA for else branch (line 214)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_128028) or more_types_in_union_128029):
            
            
            # Getting the type of 'dtypes' (line 216)
            dtypes_128034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'dtypes')
            # Applying the 'not' unary operator (line 216)
            result_not__128035 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 13), 'not', dtypes_128034)
            
            # Testing the type of an if condition (line 216)
            if_condition_128036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 13), result_not__128035)
            # Assigning a type to the variable 'if_condition_128036' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'if_condition_128036', if_condition_128036)
            # SSA begins for if statement (line 216)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 217)
            # Processing the call arguments (line 217)
            str_128038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'str', 'Must specify at least one dtype')
            # Processing the call keyword arguments (line 217)
            kwargs_128039 = {}
            # Getting the type of 'ValueError' (line 217)
            ValueError_128037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 217)
            ValueError_call_result_128040 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), ValueError_128037, *[str_128038], **kwargs_128039)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 217, 12), ValueError_call_result_128040, 'raise parameter', BaseException)
            # SSA join for if statement (line 216)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_128028 and more_types_in_union_128029):
                # SSA join for if statement (line 214)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to _read_size(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_128043 = {}
        # Getting the type of 'self' (line 219)
        self_128041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'self', False)
        # Obtaining the member '_read_size' of a type (line 219)
        _read_size_128042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 21), self_128041, '_read_size')
        # Calling _read_size(args, kwargs) (line 219)
        _read_size_call_result_128044 = invoke(stypy.reporting.localization.Localization(__file__, 219, 21), _read_size_128042, *[], **kwargs_128043)
        
        # Assigning a type to the variable 'first_size' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'first_size', _read_size_call_result_128044)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to tuple(...): (line 221)
        # Processing the call arguments (line 221)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 221, 23, True)
        # Calculating comprehension expression
        # Getting the type of 'dtypes' (line 221)
        dtypes_128051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 52), 'dtypes', False)
        comprehension_128052 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 23), dtypes_128051)
        # Assigning a type to the variable 'dtype' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'dtype', comprehension_128052)
        
        # Call to dtype(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'dtype' (line 221)
        dtype_128048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'dtype', False)
        # Processing the call keyword arguments (line 221)
        kwargs_128049 = {}
        # Getting the type of 'np' (line 221)
        np_128046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'np', False)
        # Obtaining the member 'dtype' of a type (line 221)
        dtype_128047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 23), np_128046, 'dtype')
        # Calling dtype(args, kwargs) (line 221)
        dtype_call_result_128050 = invoke(stypy.reporting.localization.Localization(__file__, 221, 23), dtype_128047, *[dtype_128048], **kwargs_128049)
        
        list_128053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 23), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 23), list_128053, dtype_call_result_128050)
        # Processing the call keyword arguments (line 221)
        kwargs_128054 = {}
        # Getting the type of 'tuple' (line 221)
        tuple_128045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 17), 'tuple', False)
        # Calling tuple(args, kwargs) (line 221)
        tuple_call_result_128055 = invoke(stypy.reporting.localization.Localization(__file__, 221, 17), tuple_128045, *[list_128053], **kwargs_128054)
        
        # Assigning a type to the variable 'dtypes' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'dtypes', tuple_call_result_128055)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to sum(...): (line 222)
        # Processing the call arguments (line 222)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 222, 25, True)
        # Calculating comprehension expression
        # Getting the type of 'dtypes' (line 222)
        dtypes_128059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 53), 'dtypes', False)
        comprehension_128060 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), dtypes_128059)
        # Assigning a type to the variable 'dtype' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'dtype', comprehension_128060)
        # Getting the type of 'dtype' (line 222)
        dtype_128057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'dtype', False)
        # Obtaining the member 'itemsize' of a type (line 222)
        itemsize_128058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 25), dtype_128057, 'itemsize')
        list_128061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 25), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 25), list_128061, itemsize_128058)
        # Processing the call keyword arguments (line 222)
        kwargs_128062 = {}
        # Getting the type of 'sum' (line 222)
        sum_128056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'sum', False)
        # Calling sum(args, kwargs) (line 222)
        sum_call_result_128063 = invoke(stypy.reporting.localization.Localization(__file__, 222, 21), sum_128056, *[list_128061], **kwargs_128062)
        
        # Assigning a type to the variable 'block_size' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'block_size', sum_call_result_128063)
        
        # Assigning a Call to a Tuple (line 224):
        
        # Assigning a Subscript to a Name (line 224):
        
        # Obtaining the type of the subscript
        int_128064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
        
        # Call to divmod(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'first_size' (line 224)
        first_size_128066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 39), 'first_size', False)
        # Getting the type of 'block_size' (line 224)
        block_size_128067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 51), 'block_size', False)
        # Processing the call keyword arguments (line 224)
        kwargs_128068 = {}
        # Getting the type of 'divmod' (line 224)
        divmod_128065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 32), 'divmod', False)
        # Calling divmod(args, kwargs) (line 224)
        divmod_call_result_128069 = invoke(stypy.reporting.localization.Localization(__file__, 224, 32), divmod_128065, *[first_size_128066, block_size_128067], **kwargs_128068)
        
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___128070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), divmod_call_result_128069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_128071 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___128070, int_128064)
        
        # Assigning a type to the variable 'tuple_var_assignment_127873' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_127873', subscript_call_result_128071)
        
        # Assigning a Subscript to a Name (line 224):
        
        # Obtaining the type of the subscript
        int_128072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
        
        # Call to divmod(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'first_size' (line 224)
        first_size_128074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 39), 'first_size', False)
        # Getting the type of 'block_size' (line 224)
        block_size_128075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 51), 'block_size', False)
        # Processing the call keyword arguments (line 224)
        kwargs_128076 = {}
        # Getting the type of 'divmod' (line 224)
        divmod_128073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 32), 'divmod', False)
        # Calling divmod(args, kwargs) (line 224)
        divmod_call_result_128077 = invoke(stypy.reporting.localization.Localization(__file__, 224, 32), divmod_128073, *[first_size_128074, block_size_128075], **kwargs_128076)
        
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___128078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), divmod_call_result_128077, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_128079 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___128078, int_128072)
        
        # Assigning a type to the variable 'tuple_var_assignment_127874' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_127874', subscript_call_result_128079)
        
        # Assigning a Name to a Name (line 224):
        # Getting the type of 'tuple_var_assignment_127873' (line 224)
        tuple_var_assignment_127873_128080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_127873')
        # Assigning a type to the variable 'num_blocks' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'num_blocks', tuple_var_assignment_127873_128080)
        
        # Assigning a Name to a Name (line 224):
        # Getting the type of 'tuple_var_assignment_127874' (line 224)
        tuple_var_assignment_127874_128081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_127874')
        # Assigning a type to the variable 'remainder' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'remainder', tuple_var_assignment_127874_128081)
        
        
        # Getting the type of 'remainder' (line 225)
        remainder_128082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'remainder')
        int_128083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 24), 'int')
        # Applying the binary operator '!=' (line 225)
        result_ne_128084 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 11), '!=', remainder_128082, int_128083)
        
        # Testing the type of an if condition (line 225)
        if_condition_128085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), result_ne_128084)
        # Assigning a type to the variable 'if_condition_128085' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_128085', if_condition_128085)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to format(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'first_size' (line 227)
        first_size_128089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 58), 'first_size', False)
        # Getting the type of 'block_size' (line 227)
        block_size_128090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 70), 'block_size', False)
        # Processing the call keyword arguments (line 226)
        kwargs_128091 = {}
        str_128087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 29), 'str', 'Size obtained ({0}) is not a multiple of the dtypes given ({1}).')
        # Obtaining the member 'format' of a type (line 226)
        format_128088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 29), str_128087, 'format')
        # Calling format(args, kwargs) (line 226)
        format_call_result_128092 = invoke(stypy.reporting.localization.Localization(__file__, 226, 29), format_128088, *[first_size_128089, block_size_128090], **kwargs_128091)
        
        # Processing the call keyword arguments (line 226)
        kwargs_128093 = {}
        # Getting the type of 'ValueError' (line 226)
        ValueError_128086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 226)
        ValueError_call_result_128094 = invoke(stypy.reporting.localization.Localization(__file__, 226, 18), ValueError_128086, *[format_call_result_128092], **kwargs_128093)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 226, 12), ValueError_call_result_128094, 'raise parameter', BaseException)
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'dtypes' (line 229)
        dtypes_128096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'dtypes', False)
        # Processing the call keyword arguments (line 229)
        kwargs_128097 = {}
        # Getting the type of 'len' (line 229)
        len_128095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'len', False)
        # Calling len(args, kwargs) (line 229)
        len_call_result_128098 = invoke(stypy.reporting.localization.Localization(__file__, 229, 11), len_128095, *[dtypes_128096], **kwargs_128097)
        
        int_128099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 26), 'int')
        # Applying the binary operator '!=' (line 229)
        result_ne_128100 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), '!=', len_call_result_128098, int_128099)
        
        
        # Getting the type of 'first_size' (line 229)
        first_size_128101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 32), 'first_size')
        # Getting the type of 'block_size' (line 229)
        block_size_128102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 46), 'block_size')
        # Applying the binary operator '!=' (line 229)
        result_ne_128103 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 32), '!=', first_size_128101, block_size_128102)
        
        # Applying the binary operator 'and' (line 229)
        result_and_keyword_128104 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'and', result_ne_128100, result_ne_128103)
        
        # Testing the type of an if condition (line 229)
        if_condition_128105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), result_and_keyword_128104)
        # Assigning a type to the variable 'if_condition_128105' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_128105', if_condition_128105)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to format(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'first_size' (line 234)
        first_size_128109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 70), 'first_size', False)
        # Getting the type of 'block_size' (line 234)
        block_size_128110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 82), 'block_size', False)
        # Processing the call keyword arguments (line 233)
        kwargs_128111 = {}
        str_128107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'str', 'Size obtained ({0}) does not match with the expected size ({1}) of multi-item record')
        # Obtaining the member 'format' of a type (line 233)
        format_128108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 29), str_128107, 'format')
        # Calling format(args, kwargs) (line 233)
        format_call_result_128112 = invoke(stypy.reporting.localization.Localization(__file__, 233, 29), format_128108, *[first_size_128109, block_size_128110], **kwargs_128111)
        
        # Processing the call keyword arguments (line 233)
        kwargs_128113 = {}
        # Getting the type of 'ValueError' (line 233)
        ValueError_128106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 233)
        ValueError_call_result_128114 = invoke(stypy.reporting.localization.Localization(__file__, 233, 18), ValueError_128106, *[format_call_result_128112], **kwargs_128113)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 233, 12), ValueError_call_result_128114, 'raise parameter', BaseException)
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_128115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        
        # Assigning a type to the variable 'data' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'data', list_128115)
        
        # Getting the type of 'dtypes' (line 237)
        dtypes_128116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'dtypes')
        # Testing the type of a for loop iterable (line 237)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 8), dtypes_128116)
        # Getting the type of the for loop variable (line 237)
        for_loop_var_128117 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 8), dtypes_128116)
        # Assigning a type to the variable 'dtype' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'dtype', for_loop_var_128117)
        # SSA begins for a for statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to fromfile(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'self' (line 238)
        self_128120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'self', False)
        # Obtaining the member '_fp' of a type (line 238)
        _fp_128121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 28), self_128120, '_fp')
        # Processing the call keyword arguments (line 238)
        # Getting the type of 'dtype' (line 238)
        dtype_128122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 44), 'dtype', False)
        keyword_128123 = dtype_128122
        # Getting the type of 'num_blocks' (line 238)
        num_blocks_128124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 57), 'num_blocks', False)
        keyword_128125 = num_blocks_128124
        kwargs_128126 = {'count': keyword_128125, 'dtype': keyword_128123}
        # Getting the type of 'np' (line 238)
        np_128118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'np', False)
        # Obtaining the member 'fromfile' of a type (line 238)
        fromfile_128119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), np_128118, 'fromfile')
        # Calling fromfile(args, kwargs) (line 238)
        fromfile_call_result_128127 = invoke(stypy.reporting.localization.Localization(__file__, 238, 16), fromfile_128119, *[_fp_128121], **kwargs_128126)
        
        # Assigning a type to the variable 'r' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'r', fromfile_call_result_128127)
        
        
        # Getting the type of 'dtype' (line 239)
        dtype_128128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'dtype')
        # Obtaining the member 'shape' of a type (line 239)
        shape_128129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 15), dtype_128128, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 239)
        tuple_128130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 239)
        
        # Applying the binary operator '!=' (line 239)
        result_ne_128131 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 15), '!=', shape_128129, tuple_128130)
        
        # Testing the type of an if condition (line 239)
        if_condition_128132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 12), result_ne_128131)
        # Assigning a type to the variable 'if_condition_128132' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'if_condition_128132', if_condition_128132)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'num_blocks' (line 241)
        num_blocks_128133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 19), 'num_blocks')
        int_128134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 33), 'int')
        # Applying the binary operator '==' (line 241)
        result_eq_128135 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 19), '==', num_blocks_128133, int_128134)
        
        # Testing the type of an if condition (line 241)
        if_condition_128136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 16), result_eq_128135)
        # Assigning a type to the variable 'if_condition_128136' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'if_condition_128136', if_condition_128136)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Evaluating assert statement condition
        
        # Getting the type of 'r' (line 242)
        r_128137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 27), 'r')
        # Obtaining the member 'shape' of a type (line 242)
        shape_128138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 27), r_128137, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 242)
        tuple_128139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 242)
        # Adding element type (line 242)
        int_128140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 39), tuple_128139, int_128140)
        
        # Getting the type of 'dtype' (line 242)
        dtype_128141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 45), 'dtype')
        # Obtaining the member 'shape' of a type (line 242)
        shape_128142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 45), dtype_128141, 'shape')
        # Applying the binary operator '+' (line 242)
        result_add_128143 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 38), '+', tuple_128139, shape_128142)
        
        # Applying the binary operator '==' (line 242)
        result_eq_128144 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 27), '==', shape_128138, result_add_128143)
        
        
        # Assigning a Subscript to a Name (line 243):
        
        # Assigning a Subscript to a Name (line 243):
        
        # Obtaining the type of the subscript
        int_128145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 26), 'int')
        # Getting the type of 'r' (line 243)
        r_128146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'r')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___128147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 24), r_128146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_128148 = invoke(stypy.reporting.localization.Localization(__file__, 243, 24), getitem___128147, int_128145)
        
        # Assigning a type to the variable 'r' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'r', subscript_call_result_128148)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'r' (line 245)
        r_128151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'r', False)
        # Processing the call keyword arguments (line 245)
        kwargs_128152 = {}
        # Getting the type of 'data' (line 245)
        data_128149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'data', False)
        # Obtaining the member 'append' of a type (line 245)
        append_128150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), data_128149, 'append')
        # Calling append(args, kwargs) (line 245)
        append_call_result_128153 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), append_128150, *[r_128151], **kwargs_128152)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to _read_size(...): (line 247)
        # Processing the call keyword arguments (line 247)
        kwargs_128156 = {}
        # Getting the type of 'self' (line 247)
        self_128154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'self', False)
        # Obtaining the member '_read_size' of a type (line 247)
        _read_size_128155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 22), self_128154, '_read_size')
        # Calling _read_size(args, kwargs) (line 247)
        _read_size_call_result_128157 = invoke(stypy.reporting.localization.Localization(__file__, 247, 22), _read_size_128155, *[], **kwargs_128156)
        
        # Assigning a type to the variable 'second_size' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'second_size', _read_size_call_result_128157)
        
        
        # Getting the type of 'first_size' (line 248)
        first_size_128158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'first_size')
        # Getting the type of 'second_size' (line 248)
        second_size_128159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'second_size')
        # Applying the binary operator '!=' (line 248)
        result_ne_128160 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 11), '!=', first_size_128158, second_size_128159)
        
        # Testing the type of an if condition (line 248)
        if_condition_128161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 8), result_ne_128160)
        # Assigning a type to the variable 'if_condition_128161' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'if_condition_128161', if_condition_128161)
        # SSA begins for if statement (line 248)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IOError(...): (line 249)
        # Processing the call arguments (line 249)
        str_128163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 26), 'str', 'Sizes do not agree in the header and footer for this record - check header dtype')
        # Processing the call keyword arguments (line 249)
        kwargs_128164 = {}
        # Getting the type of 'IOError' (line 249)
        IOError_128162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'IOError', False)
        # Calling IOError(args, kwargs) (line 249)
        IOError_call_result_128165 = invoke(stypy.reporting.localization.Localization(__file__, 249, 18), IOError_128162, *[str_128163], **kwargs_128164)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 249, 12), IOError_call_result_128165, 'raise parameter', BaseException)
        # SSA join for if statement (line 248)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'dtypes' (line 253)
        dtypes_128167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'dtypes', False)
        # Processing the call keyword arguments (line 253)
        kwargs_128168 = {}
        # Getting the type of 'len' (line 253)
        len_128166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'len', False)
        # Calling len(args, kwargs) (line 253)
        len_call_result_128169 = invoke(stypy.reporting.localization.Localization(__file__, 253, 11), len_128166, *[dtypes_128167], **kwargs_128168)
        
        int_128170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 26), 'int')
        # Applying the binary operator '==' (line 253)
        result_eq_128171 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 11), '==', len_call_result_128169, int_128170)
        
        # Testing the type of an if condition (line 253)
        if_condition_128172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), result_eq_128171)
        # Assigning a type to the variable 'if_condition_128172' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_128172', if_condition_128172)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_128173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 24), 'int')
        # Getting the type of 'data' (line 254)
        data_128174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'data')
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___128175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), data_128174, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_128176 = invoke(stypy.reporting.localization.Localization(__file__, 254, 19), getitem___128175, int_128173)
        
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'stypy_return_type', subscript_call_result_128176)
        # SSA branch for the else part of an if statement (line 253)
        module_type_store.open_ssa_branch('else')
        
        # Call to tuple(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'data' (line 256)
        data_128178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 25), 'data', False)
        # Processing the call keyword arguments (line 256)
        kwargs_128179 = {}
        # Getting the type of 'tuple' (line 256)
        tuple_128177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'tuple', False)
        # Calling tuple(args, kwargs) (line 256)
        tuple_call_result_128180 = invoke(stypy.reporting.localization.Localization(__file__, 256, 19), tuple_128177, *[data_128178], **kwargs_128179)
        
        # Assigning a type to the variable 'stypy_return_type' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'stypy_return_type', tuple_call_result_128180)
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'read_record(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_record' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_128181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_record'
        return stypy_return_type_128181


    @norecursion
    def read_ints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_128182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'str', 'i4')
        defaults = [str_128182]
        # Create a new context for function 'read_ints'
        module_type_store = module_type_store.open_function_context('read_ints', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile.read_ints.__dict__.__setitem__('stypy_localization', localization)
        FortranFile.read_ints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile.read_ints.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile.read_ints.__dict__.__setitem__('stypy_function_name', 'FortranFile.read_ints')
        FortranFile.read_ints.__dict__.__setitem__('stypy_param_names_list', ['dtype'])
        FortranFile.read_ints.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFile.read_ints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFile.read_ints.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile.read_ints.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile.read_ints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile.read_ints.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.read_ints', ['dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_ints', localization, ['dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_ints(...)' code ##################

        str_128183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, (-1)), 'str', '\n        Reads a record of a given type from the file, defaulting to an integer\n        type (``INTEGER*4`` in Fortran).\n\n        Parameters\n        ----------\n        dtype : dtype, optional\n            Data type specifying the size and endiness of the data.\n\n        Returns\n        -------\n        data : ndarray\n            A one-dimensional array object.\n\n        See Also\n        --------\n        read_reals\n        read_record\n\n        ')
        
        # Call to read_record(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'dtype' (line 279)
        dtype_128186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 32), 'dtype', False)
        # Processing the call keyword arguments (line 279)
        kwargs_128187 = {}
        # Getting the type of 'self' (line 279)
        self_128184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'self', False)
        # Obtaining the member 'read_record' of a type (line 279)
        read_record_128185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 15), self_128184, 'read_record')
        # Calling read_record(args, kwargs) (line 279)
        read_record_call_result_128188 = invoke(stypy.reporting.localization.Localization(__file__, 279, 15), read_record_128185, *[dtype_128186], **kwargs_128187)
        
        # Assigning a type to the variable 'stypy_return_type' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'stypy_return_type', read_record_call_result_128188)
        
        # ################# End of 'read_ints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_ints' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_128189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_ints'
        return stypy_return_type_128189


    @norecursion
    def read_reals(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_128190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 31), 'str', 'f8')
        defaults = [str_128190]
        # Create a new context for function 'read_reals'
        module_type_store = module_type_store.open_function_context('read_reals', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile.read_reals.__dict__.__setitem__('stypy_localization', localization)
        FortranFile.read_reals.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile.read_reals.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile.read_reals.__dict__.__setitem__('stypy_function_name', 'FortranFile.read_reals')
        FortranFile.read_reals.__dict__.__setitem__('stypy_param_names_list', ['dtype'])
        FortranFile.read_reals.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFile.read_reals.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFile.read_reals.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile.read_reals.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile.read_reals.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile.read_reals.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.read_reals', ['dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_reals', localization, ['dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_reals(...)' code ##################

        str_128191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'str', '\n        Reads a record of a given type from the file, defaulting to a floating\n        point number (``real*8`` in Fortran).\n\n        Parameters\n        ----------\n        dtype : dtype, optional\n            Data type specifying the size and endiness of the data.\n\n        Returns\n        -------\n        data : ndarray\n            A one-dimensional array object.\n\n        See Also\n        --------\n        read_ints\n        read_record\n\n        ')
        
        # Call to read_record(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'dtype' (line 302)
        dtype_128194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 32), 'dtype', False)
        # Processing the call keyword arguments (line 302)
        kwargs_128195 = {}
        # Getting the type of 'self' (line 302)
        self_128192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'self', False)
        # Obtaining the member 'read_record' of a type (line 302)
        read_record_128193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), self_128192, 'read_record')
        # Calling read_record(args, kwargs) (line 302)
        read_record_call_result_128196 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), read_record_128193, *[dtype_128194], **kwargs_128195)
        
        # Assigning a type to the variable 'stypy_return_type' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'stypy_return_type', read_record_call_result_128196)
        
        # ################# End of 'read_reals(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_reals' in the type store
        # Getting the type of 'stypy_return_type' (line 281)
        stypy_return_type_128197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_reals'
        return stypy_return_type_128197


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile.close.__dict__.__setitem__('stypy_localization', localization)
        FortranFile.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile.close.__dict__.__setitem__('stypy_function_name', 'FortranFile.close')
        FortranFile.close.__dict__.__setitem__('stypy_param_names_list', [])
        FortranFile.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFile.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFile.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.close', [], None, None, defaults, varargs, kwargs)

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

        str_128198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, (-1)), 'str', "\n        Closes the file. It is unsupported to call any other methods off this\n        object after closing it. Note that this class supports the 'with'\n        statement in modern versions of Python, to call this automatically\n\n        ")
        
        # Call to close(...): (line 311)
        # Processing the call keyword arguments (line 311)
        kwargs_128202 = {}
        # Getting the type of 'self' (line 311)
        self_128199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self', False)
        # Obtaining the member '_fp' of a type (line 311)
        _fp_128200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_128199, '_fp')
        # Obtaining the member 'close' of a type (line 311)
        close_128201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), _fp_128200, 'close')
        # Calling close(args, kwargs) (line 311)
        close_call_result_128203 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), close_128201, *[], **kwargs_128202)
        
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_128204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128204)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_128204


    @norecursion
    def __enter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__enter__'
        module_type_store = module_type_store.open_function_context('__enter__', 313, 4, False)
        # Assigning a type to the variable 'self' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile.__enter__.__dict__.__setitem__('stypy_localization', localization)
        FortranFile.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile.__enter__.__dict__.__setitem__('stypy_function_name', 'FortranFile.__enter__')
        FortranFile.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
        FortranFile.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFile.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFile.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.__enter__', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 314)
        self_128205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'stypy_return_type', self_128205)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 313)
        stypy_return_type_128206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128206)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_128206


    @norecursion
    def __exit__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exit__'
        module_type_store = module_type_store.open_function_context('__exit__', 316, 4, False)
        # Assigning a type to the variable 'self' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FortranFile.__exit__.__dict__.__setitem__('stypy_localization', localization)
        FortranFile.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FortranFile.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FortranFile.__exit__.__dict__.__setitem__('stypy_function_name', 'FortranFile.__exit__')
        FortranFile.__exit__.__dict__.__setitem__('stypy_param_names_list', ['type', 'value', 'tb'])
        FortranFile.__exit__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FortranFile.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FortranFile.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FortranFile.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FortranFile.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FortranFile.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FortranFile.__exit__', ['type', 'value', 'tb'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exit__', localization, ['type', 'value', 'tb'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exit__(...)' code ##################

        
        # Call to close(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_128209 = {}
        # Getting the type of 'self' (line 317)
        self_128207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self', False)
        # Obtaining the member 'close' of a type (line 317)
        close_128208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_128207, 'close')
        # Calling close(args, kwargs) (line 317)
        close_call_result_128210 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), close_128208, *[], **kwargs_128209)
        
        
        # ################# End of '__exit__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exit__' in the type store
        # Getting the type of 'stypy_return_type' (line 316)
        stypy_return_type_128211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exit__'
        return stypy_return_type_128211


# Assigning a type to the variable 'FortranFile' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'FortranFile', FortranFile)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
