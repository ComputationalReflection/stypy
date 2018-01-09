
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module for reading and writing matlab (TM) .mat files
3: '''
4: # Authors: Travis Oliphant, Matthew Brett
5: 
6: from __future__ import division, print_function, absolute_import
7: 
8: import numpy as np
9: 
10: from scipy._lib.six import string_types
11: 
12: from .miobase import get_matfile_version, docfiller
13: from .mio4 import MatFile4Reader, MatFile4Writer
14: from .mio5 import MatFile5Reader, MatFile5Writer
15: 
16: __all__ = ['mat_reader_factory', 'loadmat', 'savemat', 'whosmat']
17: 
18: 
19: def _open_file(file_like, appendmat):
20:     '''
21:     Open `file_like` and return as file-like object. First, check if object is
22:     already file-like; if so, return it as-is. Otherwise, try to pass it
23:     to open(). If that fails, and `file_like` is a string, and `appendmat` is true,
24:     append '.mat' and try again.
25:     '''
26:     try:
27:         file_like.read(0)
28:         return file_like, False
29:     except AttributeError:
30:         pass
31: 
32:     try:
33:         return open(file_like, 'rb'), True
34:     except IOError:
35:         # Probably "not found"
36:         if isinstance(file_like, string_types):
37:             if appendmat and not file_like.endswith('.mat'):
38:                 file_like += '.mat'
39:                 return open(file_like, 'rb'), True
40:         else:
41:             raise IOError('Reader needs file name or open file-like object')
42: 
43: @docfiller
44: def mat_reader_factory(file_name, appendmat=True, **kwargs):
45:     '''
46:     Create reader for matlab .mat format files.
47: 
48:     Parameters
49:     ----------
50:     %(file_arg)s
51:     %(append_arg)s
52:     %(load_args)s
53:     %(struct_arg)s
54: 
55:     Returns
56:     -------
57:     matreader : MatFileReader object
58:        Initialized instance of MatFileReader class matching the mat file
59:        type detected in `filename`.
60:     file_opened : bool
61:        Whether the file was opened by this routine.
62: 
63:     '''
64:     byte_stream, file_opened = _open_file(file_name, appendmat)
65:     mjv, mnv = get_matfile_version(byte_stream)
66:     if mjv == 0:
67:         return MatFile4Reader(byte_stream, **kwargs), file_opened
68:     elif mjv == 1:
69:         return MatFile5Reader(byte_stream, **kwargs), file_opened
70:     elif mjv == 2:
71:         raise NotImplementedError('Please use HDF reader for matlab v7.3 files')
72:     else:
73:         raise TypeError('Did not recognize version %s' % mjv)
74: 
75: 
76: @docfiller
77: def loadmat(file_name, mdict=None, appendmat=True, **kwargs):
78:     '''
79:     Load MATLAB file.
80: 
81:     Parameters
82:     ----------
83:     file_name : str
84:        Name of the mat file (do not need .mat extension if
85:        appendmat==True). Can also pass open file-like object.
86:     mdict : dict, optional
87:         Dictionary in which to insert matfile variables.
88:     appendmat : bool, optional
89:        True to append the .mat extension to the end of the given
90:        filename, if not already present.
91:     byte_order : str or None, optional
92:        None by default, implying byte order guessed from mat
93:        file. Otherwise can be one of ('native', '=', 'little', '<',
94:        'BIG', '>').
95:     mat_dtype : bool, optional
96:        If True, return arrays in same dtype as would be loaded into
97:        MATLAB (instead of the dtype with which they are saved).
98:     squeeze_me : bool, optional
99:        Whether to squeeze unit matrix dimensions or not.
100:     chars_as_strings : bool, optional
101:        Whether to convert char arrays to string arrays.
102:     matlab_compatible : bool, optional
103:        Returns matrices as would be loaded by MATLAB (implies
104:        squeeze_me=False, chars_as_strings=False, mat_dtype=True,
105:        struct_as_record=True).
106:     struct_as_record : bool, optional
107:        Whether to load MATLAB structs as numpy record arrays, or as
108:        old-style numpy arrays with dtype=object.  Setting this flag to
109:        False replicates the behavior of scipy version 0.7.x (returning
110:        numpy object arrays).  The default setting is True, because it
111:        allows easier round-trip load and save of MATLAB files.
112:     verify_compressed_data_integrity : bool, optional
113:         Whether the length of compressed sequences in the MATLAB file
114:         should be checked, to ensure that they are not longer than we expect.
115:         It is advisable to enable this (the default) because overlong
116:         compressed sequences in MATLAB files generally indicate that the
117:         files have experienced some sort of corruption.
118:     variable_names : None or sequence
119:         If None (the default) - read all variables in file. Otherwise
120:         `variable_names` should be a sequence of strings, giving names of the
121:         matlab variables to read from the file.  The reader will skip any
122:         variable with a name not in this sequence, possibly saving some read
123:         processing.
124: 
125:     Returns
126:     -------
127:     mat_dict : dict
128:        dictionary with variable names as keys, and loaded matrices as
129:        values.
130: 
131:     Notes
132:     -----
133:     v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.
134: 
135:     You will need an HDF5 python library to read matlab 7.3 format mat
136:     files.  Because scipy does not supply one, we do not implement the
137:     HDF5 / 7.3 interface here.
138: 
139:     '''
140:     variable_names = kwargs.pop('variable_names', None)
141:     MR, file_opened = mat_reader_factory(file_name, appendmat, **kwargs)
142:     matfile_dict = MR.get_variables(variable_names)
143:     if mdict is not None:
144:         mdict.update(matfile_dict)
145:     else:
146:         mdict = matfile_dict
147:     if file_opened:
148:         MR.mat_stream.close()
149:     return mdict
150: 
151: 
152: @docfiller
153: def savemat(file_name, mdict,
154:             appendmat=True,
155:             format='5',
156:             long_field_names=False,
157:             do_compression=False,
158:             oned_as='row'):
159:     '''
160:     Save a dictionary of names and arrays into a MATLAB-style .mat file.
161: 
162:     This saves the array objects in the given dictionary to a MATLAB-
163:     style .mat file.
164: 
165:     Parameters
166:     ----------
167:     file_name : str or file-like object
168:         Name of the .mat file (.mat extension not needed if ``appendmat ==
169:         True``).
170:         Can also pass open file_like object.
171:     mdict : dict
172:         Dictionary from which to save matfile variables.
173:     appendmat : bool, optional
174:         True (the default) to append the .mat extension to the end of the
175:         given filename, if not already present.
176:     format : {'5', '4'}, string, optional
177:         '5' (the default) for MATLAB 5 and up (to 7.2),
178:         '4' for MATLAB 4 .mat files.
179:     long_field_names : bool, optional
180:         False (the default) - maximum field name length in a structure is
181:         31 characters which is the documented maximum length.
182:         True - maximum field name length in a structure is 63 characters
183:         which works for MATLAB 7.6+.
184:     do_compression : bool, optional
185:         Whether or not to compress matrices on write.  Default is False.
186:     oned_as : {'row', 'column'}, optional
187:         If 'column', write 1-D numpy arrays as column vectors.
188:         If 'row', write 1-D numpy arrays as row vectors.
189: 
190:     See also
191:     --------
192:     mio4.MatFile4Writer
193:     mio5.MatFile5Writer
194:     '''
195:     file_opened = False
196:     if hasattr(file_name, 'write'):
197:         # File-like object already; use as-is
198:         file_stream = file_name
199:     else:
200:         if isinstance(file_name, string_types):
201:             if appendmat and not file_name.endswith('.mat'):
202:                 file_name = file_name + ".mat"
203: 
204:         file_stream = open(file_name, 'wb')
205:         file_opened = True
206: 
207:     if format == '4':
208:         if long_field_names:
209:             raise ValueError("Long field names are not available for version 4 files")
210:         MW = MatFile4Writer(file_stream, oned_as)
211:     elif format == '5':
212:         MW = MatFile5Writer(file_stream,
213:                             do_compression=do_compression,
214:                             unicode_strings=True,
215:                             long_field_names=long_field_names,
216:                             oned_as=oned_as)
217:     else:
218:         raise ValueError("Format should be '4' or '5'")
219:     MW.put_variables(mdict)
220:     if file_opened:
221:         file_stream.close()
222: 
223: 
224: @docfiller
225: def whosmat(file_name, appendmat=True, **kwargs):
226:     '''
227:     List variables inside a MATLAB file.
228: 
229:     Parameters
230:     ----------
231:     %(file_arg)s
232:     %(append_arg)s
233:     %(load_args)s
234:     %(struct_arg)s
235: 
236:     Returns
237:     -------
238:     variables : list of tuples
239:         A list of tuples, where each tuple holds the matrix name (a string),
240:         its shape (tuple of ints), and its data class (a string).
241:         Possible data classes are: int8, uint8, int16, uint16, int32, uint32,
242:         int64, uint64, single, double, cell, struct, object, char, sparse,
243:         function, opaque, logical, unknown.
244: 
245:     Notes
246:     -----
247:     v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.
248: 
249:     You will need an HDF5 python library to read matlab 7.3 format mat
250:     files.  Because scipy does not supply one, we do not implement the
251:     HDF5 / 7.3 interface here.
252: 
253:     .. versionadded:: 0.12.0
254: 
255:     '''
256:     ML, file_opened = mat_reader_factory(file_name, **kwargs)
257:     variables = ML.list_variables()
258:     if file_opened:
259:         ML.mat_stream.close()
260:     return variables
261: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_133453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nModule for reading and writing matlab (TM) .mat files\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133454 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_133454) is not StypyTypeError):

    if (import_133454 != 'pyd_module'):
        __import__(import_133454)
        sys_modules_133455 = sys.modules[import_133454]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_133455.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_133454)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib.six import string_types' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133456 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six')

if (type(import_133456) is not StypyTypeError):

    if (import_133456 != 'pyd_module'):
        __import__(import_133456)
        sys_modules_133457 = sys.modules[import_133456]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', sys_modules_133457.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_133457, sys_modules_133457.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', import_133456)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.io.matlab.miobase import get_matfile_version, docfiller' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133458 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.miobase')

if (type(import_133458) is not StypyTypeError):

    if (import_133458 != 'pyd_module'):
        __import__(import_133458)
        sys_modules_133459 = sys.modules[import_133458]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.miobase', sys_modules_133459.module_type_store, module_type_store, ['get_matfile_version', 'docfiller'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_133459, sys_modules_133459.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.miobase import get_matfile_version, docfiller

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.miobase', None, module_type_store, ['get_matfile_version', 'docfiller'], [get_matfile_version, docfiller])

else:
    # Assigning a type to the variable 'scipy.io.matlab.miobase' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.miobase', import_133458)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.io.matlab.mio4 import MatFile4Reader, MatFile4Writer' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133460 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio4')

if (type(import_133460) is not StypyTypeError):

    if (import_133460 != 'pyd_module'):
        __import__(import_133460)
        sys_modules_133461 = sys.modules[import_133460]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio4', sys_modules_133461.module_type_store, module_type_store, ['MatFile4Reader', 'MatFile4Writer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_133461, sys_modules_133461.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio4 import MatFile4Reader, MatFile4Writer

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio4', None, module_type_store, ['MatFile4Reader', 'MatFile4Writer'], [MatFile4Reader, MatFile4Writer])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio4' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio4', import_133460)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.io.matlab.mio5 import MatFile5Reader, MatFile5Writer' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_133462 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab.mio5')

if (type(import_133462) is not StypyTypeError):

    if (import_133462 != 'pyd_module'):
        __import__(import_133462)
        sys_modules_133463 = sys.modules[import_133462]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab.mio5', sys_modules_133463.module_type_store, module_type_store, ['MatFile5Reader', 'MatFile5Writer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_133463, sys_modules_133463.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio5 import MatFile5Reader, MatFile5Writer

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab.mio5', None, module_type_store, ['MatFile5Reader', 'MatFile5Writer'], [MatFile5Reader, MatFile5Writer])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio5' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.io.matlab.mio5', import_133462)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')


# Assigning a List to a Name (line 16):

# Assigning a List to a Name (line 16):
__all__ = ['mat_reader_factory', 'loadmat', 'savemat', 'whosmat']
module_type_store.set_exportable_members(['mat_reader_factory', 'loadmat', 'savemat', 'whosmat'])

# Obtaining an instance of the builtin type 'list' (line 16)
list_133464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_133465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'mat_reader_factory')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_133464, str_133465)
# Adding element type (line 16)
str_133466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'str', 'loadmat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_133464, str_133466)
# Adding element type (line 16)
str_133467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 44), 'str', 'savemat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_133464, str_133467)
# Adding element type (line 16)
str_133468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 55), 'str', 'whosmat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_133464, str_133468)

# Assigning a type to the variable '__all__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__all__', list_133464)

@norecursion
def _open_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_open_file'
    module_type_store = module_type_store.open_function_context('_open_file', 19, 0, False)
    
    # Passed parameters checking function
    _open_file.stypy_localization = localization
    _open_file.stypy_type_of_self = None
    _open_file.stypy_type_store = module_type_store
    _open_file.stypy_function_name = '_open_file'
    _open_file.stypy_param_names_list = ['file_like', 'appendmat']
    _open_file.stypy_varargs_param_name = None
    _open_file.stypy_kwargs_param_name = None
    _open_file.stypy_call_defaults = defaults
    _open_file.stypy_call_varargs = varargs
    _open_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_open_file', ['file_like', 'appendmat'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_open_file', localization, ['file_like', 'appendmat'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_open_file(...)' code ##################

    str_133469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', "\n    Open `file_like` and return as file-like object. First, check if object is\n    already file-like; if so, return it as-is. Otherwise, try to pass it\n    to open(). If that fails, and `file_like` is a string, and `appendmat` is true,\n    append '.mat' and try again.\n    ")
    
    
    # SSA begins for try-except statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to read(...): (line 27)
    # Processing the call arguments (line 27)
    int_133472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'int')
    # Processing the call keyword arguments (line 27)
    kwargs_133473 = {}
    # Getting the type of 'file_like' (line 27)
    file_like_133470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'file_like', False)
    # Obtaining the member 'read' of a type (line 27)
    read_133471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), file_like_133470, 'read')
    # Calling read(args, kwargs) (line 27)
    read_call_result_133474 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), read_133471, *[int_133472], **kwargs_133473)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_133475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    # Getting the type of 'file_like' (line 28)
    file_like_133476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'file_like')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), tuple_133475, file_like_133476)
    # Adding element type (line 28)
    # Getting the type of 'False' (line 28)
    False_133477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), tuple_133475, False_133477)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', tuple_133475)
    # SSA branch for the except part of a try statement (line 26)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 26)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_133478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    
    # Call to open(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'file_like' (line 33)
    file_like_133480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'file_like', False)
    str_133481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'str', 'rb')
    # Processing the call keyword arguments (line 33)
    kwargs_133482 = {}
    # Getting the type of 'open' (line 33)
    open_133479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'open', False)
    # Calling open(args, kwargs) (line 33)
    open_call_result_133483 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), open_133479, *[file_like_133480, str_133481], **kwargs_133482)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 15), tuple_133478, open_call_result_133483)
    # Adding element type (line 33)
    # Getting the type of 'True' (line 33)
    True_133484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 15), tuple_133478, True_133484)
    
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', tuple_133478)
    # SSA branch for the except part of a try statement (line 32)
    # SSA branch for the except 'IOError' branch of a try statement (line 32)
    module_type_store.open_ssa_branch('except')
    
    
    # Call to isinstance(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'file_like' (line 36)
    file_like_133486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'file_like', False)
    # Getting the type of 'string_types' (line 36)
    string_types_133487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'string_types', False)
    # Processing the call keyword arguments (line 36)
    kwargs_133488 = {}
    # Getting the type of 'isinstance' (line 36)
    isinstance_133485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 36)
    isinstance_call_result_133489 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), isinstance_133485, *[file_like_133486, string_types_133487], **kwargs_133488)
    
    # Testing the type of an if condition (line 36)
    if_condition_133490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), isinstance_call_result_133489)
    # Assigning a type to the variable 'if_condition_133490' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_133490', if_condition_133490)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'appendmat' (line 37)
    appendmat_133491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'appendmat')
    
    
    # Call to endswith(...): (line 37)
    # Processing the call arguments (line 37)
    str_133494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 52), 'str', '.mat')
    # Processing the call keyword arguments (line 37)
    kwargs_133495 = {}
    # Getting the type of 'file_like' (line 37)
    file_like_133492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'file_like', False)
    # Obtaining the member 'endswith' of a type (line 37)
    endswith_133493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 33), file_like_133492, 'endswith')
    # Calling endswith(args, kwargs) (line 37)
    endswith_call_result_133496 = invoke(stypy.reporting.localization.Localization(__file__, 37, 33), endswith_133493, *[str_133494], **kwargs_133495)
    
    # Applying the 'not' unary operator (line 37)
    result_not__133497 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 29), 'not', endswith_call_result_133496)
    
    # Applying the binary operator 'and' (line 37)
    result_and_keyword_133498 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 15), 'and', appendmat_133491, result_not__133497)
    
    # Testing the type of an if condition (line 37)
    if_condition_133499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 12), result_and_keyword_133498)
    # Assigning a type to the variable 'if_condition_133499' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'if_condition_133499', if_condition_133499)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'file_like' (line 38)
    file_like_133500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'file_like')
    str_133501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 29), 'str', '.mat')
    # Applying the binary operator '+=' (line 38)
    result_iadd_133502 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 16), '+=', file_like_133500, str_133501)
    # Assigning a type to the variable 'file_like' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'file_like', result_iadd_133502)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 39)
    tuple_133503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 39)
    # Adding element type (line 39)
    
    # Call to open(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'file_like' (line 39)
    file_like_133505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'file_like', False)
    str_133506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'str', 'rb')
    # Processing the call keyword arguments (line 39)
    kwargs_133507 = {}
    # Getting the type of 'open' (line 39)
    open_133504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'open', False)
    # Calling open(args, kwargs) (line 39)
    open_call_result_133508 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), open_133504, *[file_like_133505, str_133506], **kwargs_133507)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), tuple_133503, open_call_result_133508)
    # Adding element type (line 39)
    # Getting the type of 'True' (line 39)
    True_133509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), tuple_133503, True_133509)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'stypy_return_type', tuple_133503)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 36)
    module_type_store.open_ssa_branch('else')
    
    # Call to IOError(...): (line 41)
    # Processing the call arguments (line 41)
    str_133511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'str', 'Reader needs file name or open file-like object')
    # Processing the call keyword arguments (line 41)
    kwargs_133512 = {}
    # Getting the type of 'IOError' (line 41)
    IOError_133510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'IOError', False)
    # Calling IOError(args, kwargs) (line 41)
    IOError_call_result_133513 = invoke(stypy.reporting.localization.Localization(__file__, 41, 18), IOError_133510, *[str_133511], **kwargs_133512)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 41, 12), IOError_call_result_133513, 'raise parameter', BaseException)
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_open_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_open_file' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_133514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133514)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_open_file'
    return stypy_return_type_133514

# Assigning a type to the variable '_open_file' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '_open_file', _open_file)

@norecursion
def mat_reader_factory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 44)
    True_133515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 44), 'True')
    defaults = [True_133515]
    # Create a new context for function 'mat_reader_factory'
    module_type_store = module_type_store.open_function_context('mat_reader_factory', 43, 0, False)
    
    # Passed parameters checking function
    mat_reader_factory.stypy_localization = localization
    mat_reader_factory.stypy_type_of_self = None
    mat_reader_factory.stypy_type_store = module_type_store
    mat_reader_factory.stypy_function_name = 'mat_reader_factory'
    mat_reader_factory.stypy_param_names_list = ['file_name', 'appendmat']
    mat_reader_factory.stypy_varargs_param_name = None
    mat_reader_factory.stypy_kwargs_param_name = 'kwargs'
    mat_reader_factory.stypy_call_defaults = defaults
    mat_reader_factory.stypy_call_varargs = varargs
    mat_reader_factory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mat_reader_factory', ['file_name', 'appendmat'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mat_reader_factory', localization, ['file_name', 'appendmat'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mat_reader_factory(...)' code ##################

    str_133516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', '\n    Create reader for matlab .mat format files.\n\n    Parameters\n    ----------\n    %(file_arg)s\n    %(append_arg)s\n    %(load_args)s\n    %(struct_arg)s\n\n    Returns\n    -------\n    matreader : MatFileReader object\n       Initialized instance of MatFileReader class matching the mat file\n       type detected in `filename`.\n    file_opened : bool\n       Whether the file was opened by this routine.\n\n    ')
    
    # Assigning a Call to a Tuple (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_133517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'int')
    
    # Call to _open_file(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'file_name' (line 64)
    file_name_133519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'file_name', False)
    # Getting the type of 'appendmat' (line 64)
    appendmat_133520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 53), 'appendmat', False)
    # Processing the call keyword arguments (line 64)
    kwargs_133521 = {}
    # Getting the type of '_open_file' (line 64)
    _open_file_133518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), '_open_file', False)
    # Calling _open_file(args, kwargs) (line 64)
    _open_file_call_result_133522 = invoke(stypy.reporting.localization.Localization(__file__, 64, 31), _open_file_133518, *[file_name_133519, appendmat_133520], **kwargs_133521)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___133523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), _open_file_call_result_133522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_133524 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), getitem___133523, int_133517)
    
    # Assigning a type to the variable 'tuple_var_assignment_133445' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_133445', subscript_call_result_133524)
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_133525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'int')
    
    # Call to _open_file(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'file_name' (line 64)
    file_name_133527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'file_name', False)
    # Getting the type of 'appendmat' (line 64)
    appendmat_133528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 53), 'appendmat', False)
    # Processing the call keyword arguments (line 64)
    kwargs_133529 = {}
    # Getting the type of '_open_file' (line 64)
    _open_file_133526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), '_open_file', False)
    # Calling _open_file(args, kwargs) (line 64)
    _open_file_call_result_133530 = invoke(stypy.reporting.localization.Localization(__file__, 64, 31), _open_file_133526, *[file_name_133527, appendmat_133528], **kwargs_133529)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___133531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), _open_file_call_result_133530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_133532 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), getitem___133531, int_133525)
    
    # Assigning a type to the variable 'tuple_var_assignment_133446' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_133446', subscript_call_result_133532)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_var_assignment_133445' (line 64)
    tuple_var_assignment_133445_133533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_133445')
    # Assigning a type to the variable 'byte_stream' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'byte_stream', tuple_var_assignment_133445_133533)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_var_assignment_133446' (line 64)
    tuple_var_assignment_133446_133534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_var_assignment_133446')
    # Assigning a type to the variable 'file_opened' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'file_opened', tuple_var_assignment_133446_133534)
    
    # Assigning a Call to a Tuple (line 65):
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_133535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to get_matfile_version(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'byte_stream' (line 65)
    byte_stream_133537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'byte_stream', False)
    # Processing the call keyword arguments (line 65)
    kwargs_133538 = {}
    # Getting the type of 'get_matfile_version' (line 65)
    get_matfile_version_133536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'get_matfile_version', False)
    # Calling get_matfile_version(args, kwargs) (line 65)
    get_matfile_version_call_result_133539 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), get_matfile_version_133536, *[byte_stream_133537], **kwargs_133538)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___133540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), get_matfile_version_call_result_133539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_133541 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___133540, int_133535)
    
    # Assigning a type to the variable 'tuple_var_assignment_133447' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_133447', subscript_call_result_133541)
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    int_133542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'int')
    
    # Call to get_matfile_version(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'byte_stream' (line 65)
    byte_stream_133544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'byte_stream', False)
    # Processing the call keyword arguments (line 65)
    kwargs_133545 = {}
    # Getting the type of 'get_matfile_version' (line 65)
    get_matfile_version_133543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'get_matfile_version', False)
    # Calling get_matfile_version(args, kwargs) (line 65)
    get_matfile_version_call_result_133546 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), get_matfile_version_133543, *[byte_stream_133544], **kwargs_133545)
    
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___133547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), get_matfile_version_call_result_133546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_133548 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), getitem___133547, int_133542)
    
    # Assigning a type to the variable 'tuple_var_assignment_133448' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_133448', subscript_call_result_133548)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_133447' (line 65)
    tuple_var_assignment_133447_133549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_133447')
    # Assigning a type to the variable 'mjv' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'mjv', tuple_var_assignment_133447_133549)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_var_assignment_133448' (line 65)
    tuple_var_assignment_133448_133550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_var_assignment_133448')
    # Assigning a type to the variable 'mnv' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 9), 'mnv', tuple_var_assignment_133448_133550)
    
    
    # Getting the type of 'mjv' (line 66)
    mjv_133551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'mjv')
    int_133552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'int')
    # Applying the binary operator '==' (line 66)
    result_eq_133553 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), '==', mjv_133551, int_133552)
    
    # Testing the type of an if condition (line 66)
    if_condition_133554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_eq_133553)
    # Assigning a type to the variable 'if_condition_133554' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_133554', if_condition_133554)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_133555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    
    # Call to MatFile4Reader(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'byte_stream' (line 67)
    byte_stream_133557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'byte_stream', False)
    # Processing the call keyword arguments (line 67)
    # Getting the type of 'kwargs' (line 67)
    kwargs_133558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 45), 'kwargs', False)
    kwargs_133559 = {'kwargs_133558': kwargs_133558}
    # Getting the type of 'MatFile4Reader' (line 67)
    MatFile4Reader_133556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'MatFile4Reader', False)
    # Calling MatFile4Reader(args, kwargs) (line 67)
    MatFile4Reader_call_result_133560 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), MatFile4Reader_133556, *[byte_stream_133557], **kwargs_133559)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 15), tuple_133555, MatFile4Reader_call_result_133560)
    # Adding element type (line 67)
    # Getting the type of 'file_opened' (line 67)
    file_opened_133561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 54), 'file_opened')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 15), tuple_133555, file_opened_133561)
    
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', tuple_133555)
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mjv' (line 68)
    mjv_133562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'mjv')
    int_133563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'int')
    # Applying the binary operator '==' (line 68)
    result_eq_133564 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 9), '==', mjv_133562, int_133563)
    
    # Testing the type of an if condition (line 68)
    if_condition_133565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 9), result_eq_133564)
    # Assigning a type to the variable 'if_condition_133565' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'if_condition_133565', if_condition_133565)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_133566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    
    # Call to MatFile5Reader(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'byte_stream' (line 69)
    byte_stream_133568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'byte_stream', False)
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'kwargs' (line 69)
    kwargs_133569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 45), 'kwargs', False)
    kwargs_133570 = {'kwargs_133569': kwargs_133569}
    # Getting the type of 'MatFile5Reader' (line 69)
    MatFile5Reader_133567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'MatFile5Reader', False)
    # Calling MatFile5Reader(args, kwargs) (line 69)
    MatFile5Reader_call_result_133571 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), MatFile5Reader_133567, *[byte_stream_133568], **kwargs_133570)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 15), tuple_133566, MatFile5Reader_call_result_133571)
    # Adding element type (line 69)
    # Getting the type of 'file_opened' (line 69)
    file_opened_133572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 54), 'file_opened')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 15), tuple_133566, file_opened_133572)
    
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', tuple_133566)
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mjv' (line 70)
    mjv_133573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 9), 'mjv')
    int_133574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 16), 'int')
    # Applying the binary operator '==' (line 70)
    result_eq_133575 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 9), '==', mjv_133573, int_133574)
    
    # Testing the type of an if condition (line 70)
    if_condition_133576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 9), result_eq_133575)
    # Assigning a type to the variable 'if_condition_133576' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 9), 'if_condition_133576', if_condition_133576)
    # SSA begins for if statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotImplementedError(...): (line 71)
    # Processing the call arguments (line 71)
    str_133578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'str', 'Please use HDF reader for matlab v7.3 files')
    # Processing the call keyword arguments (line 71)
    kwargs_133579 = {}
    # Getting the type of 'NotImplementedError' (line 71)
    NotImplementedError_133577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 71)
    NotImplementedError_call_result_133580 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), NotImplementedError_133577, *[str_133578], **kwargs_133579)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 8), NotImplementedError_call_result_133580, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 70)
    module_type_store.open_ssa_branch('else')
    
    # Call to TypeError(...): (line 73)
    # Processing the call arguments (line 73)
    str_133582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'str', 'Did not recognize version %s')
    # Getting the type of 'mjv' (line 73)
    mjv_133583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 57), 'mjv', False)
    # Applying the binary operator '%' (line 73)
    result_mod_133584 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 24), '%', str_133582, mjv_133583)
    
    # Processing the call keyword arguments (line 73)
    kwargs_133585 = {}
    # Getting the type of 'TypeError' (line 73)
    TypeError_133581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 73)
    TypeError_call_result_133586 = invoke(stypy.reporting.localization.Localization(__file__, 73, 14), TypeError_133581, *[result_mod_133584], **kwargs_133585)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 73, 8), TypeError_call_result_133586, 'raise parameter', BaseException)
    # SSA join for if statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'mat_reader_factory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mat_reader_factory' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_133587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133587)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mat_reader_factory'
    return stypy_return_type_133587

# Assigning a type to the variable 'mat_reader_factory' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'mat_reader_factory', mat_reader_factory)

@norecursion
def loadmat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 77)
    None_133588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'None')
    # Getting the type of 'True' (line 77)
    True_133589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'True')
    defaults = [None_133588, True_133589]
    # Create a new context for function 'loadmat'
    module_type_store = module_type_store.open_function_context('loadmat', 76, 0, False)
    
    # Passed parameters checking function
    loadmat.stypy_localization = localization
    loadmat.stypy_type_of_self = None
    loadmat.stypy_type_store = module_type_store
    loadmat.stypy_function_name = 'loadmat'
    loadmat.stypy_param_names_list = ['file_name', 'mdict', 'appendmat']
    loadmat.stypy_varargs_param_name = None
    loadmat.stypy_kwargs_param_name = 'kwargs'
    loadmat.stypy_call_defaults = defaults
    loadmat.stypy_call_varargs = varargs
    loadmat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'loadmat', ['file_name', 'mdict', 'appendmat'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'loadmat', localization, ['file_name', 'mdict', 'appendmat'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'loadmat(...)' code ##################

    str_133590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', "\n    Load MATLAB file.\n\n    Parameters\n    ----------\n    file_name : str\n       Name of the mat file (do not need .mat extension if\n       appendmat==True). Can also pass open file-like object.\n    mdict : dict, optional\n        Dictionary in which to insert matfile variables.\n    appendmat : bool, optional\n       True to append the .mat extension to the end of the given\n       filename, if not already present.\n    byte_order : str or None, optional\n       None by default, implying byte order guessed from mat\n       file. Otherwise can be one of ('native', '=', 'little', '<',\n       'BIG', '>').\n    mat_dtype : bool, optional\n       If True, return arrays in same dtype as would be loaded into\n       MATLAB (instead of the dtype with which they are saved).\n    squeeze_me : bool, optional\n       Whether to squeeze unit matrix dimensions or not.\n    chars_as_strings : bool, optional\n       Whether to convert char arrays to string arrays.\n    matlab_compatible : bool, optional\n       Returns matrices as would be loaded by MATLAB (implies\n       squeeze_me=False, chars_as_strings=False, mat_dtype=True,\n       struct_as_record=True).\n    struct_as_record : bool, optional\n       Whether to load MATLAB structs as numpy record arrays, or as\n       old-style numpy arrays with dtype=object.  Setting this flag to\n       False replicates the behavior of scipy version 0.7.x (returning\n       numpy object arrays).  The default setting is True, because it\n       allows easier round-trip load and save of MATLAB files.\n    verify_compressed_data_integrity : bool, optional\n        Whether the length of compressed sequences in the MATLAB file\n        should be checked, to ensure that they are not longer than we expect.\n        It is advisable to enable this (the default) because overlong\n        compressed sequences in MATLAB files generally indicate that the\n        files have experienced some sort of corruption.\n    variable_names : None or sequence\n        If None (the default) - read all variables in file. Otherwise\n        `variable_names` should be a sequence of strings, giving names of the\n        matlab variables to read from the file.  The reader will skip any\n        variable with a name not in this sequence, possibly saving some read\n        processing.\n\n    Returns\n    -------\n    mat_dict : dict\n       dictionary with variable names as keys, and loaded matrices as\n       values.\n\n    Notes\n    -----\n    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.\n\n    You will need an HDF5 python library to read matlab 7.3 format mat\n    files.  Because scipy does not supply one, we do not implement the\n    HDF5 / 7.3 interface here.\n\n    ")
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to pop(...): (line 140)
    # Processing the call arguments (line 140)
    str_133593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 32), 'str', 'variable_names')
    # Getting the type of 'None' (line 140)
    None_133594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 50), 'None', False)
    # Processing the call keyword arguments (line 140)
    kwargs_133595 = {}
    # Getting the type of 'kwargs' (line 140)
    kwargs_133591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 140)
    pop_133592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), kwargs_133591, 'pop')
    # Calling pop(args, kwargs) (line 140)
    pop_call_result_133596 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), pop_133592, *[str_133593, None_133594], **kwargs_133595)
    
    # Assigning a type to the variable 'variable_names' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'variable_names', pop_call_result_133596)
    
    # Assigning a Call to a Tuple (line 141):
    
    # Assigning a Subscript to a Name (line 141):
    
    # Obtaining the type of the subscript
    int_133597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'int')
    
    # Call to mat_reader_factory(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'file_name' (line 141)
    file_name_133599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'file_name', False)
    # Getting the type of 'appendmat' (line 141)
    appendmat_133600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 52), 'appendmat', False)
    # Processing the call keyword arguments (line 141)
    # Getting the type of 'kwargs' (line 141)
    kwargs_133601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 65), 'kwargs', False)
    kwargs_133602 = {'kwargs_133601': kwargs_133601}
    # Getting the type of 'mat_reader_factory' (line 141)
    mat_reader_factory_133598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'mat_reader_factory', False)
    # Calling mat_reader_factory(args, kwargs) (line 141)
    mat_reader_factory_call_result_133603 = invoke(stypy.reporting.localization.Localization(__file__, 141, 22), mat_reader_factory_133598, *[file_name_133599, appendmat_133600], **kwargs_133602)
    
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___133604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), mat_reader_factory_call_result_133603, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_133605 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), getitem___133604, int_133597)
    
    # Assigning a type to the variable 'tuple_var_assignment_133449' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_133449', subscript_call_result_133605)
    
    # Assigning a Subscript to a Name (line 141):
    
    # Obtaining the type of the subscript
    int_133606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'int')
    
    # Call to mat_reader_factory(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'file_name' (line 141)
    file_name_133608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'file_name', False)
    # Getting the type of 'appendmat' (line 141)
    appendmat_133609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 52), 'appendmat', False)
    # Processing the call keyword arguments (line 141)
    # Getting the type of 'kwargs' (line 141)
    kwargs_133610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 65), 'kwargs', False)
    kwargs_133611 = {'kwargs_133610': kwargs_133610}
    # Getting the type of 'mat_reader_factory' (line 141)
    mat_reader_factory_133607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'mat_reader_factory', False)
    # Calling mat_reader_factory(args, kwargs) (line 141)
    mat_reader_factory_call_result_133612 = invoke(stypy.reporting.localization.Localization(__file__, 141, 22), mat_reader_factory_133607, *[file_name_133608, appendmat_133609], **kwargs_133611)
    
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___133613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), mat_reader_factory_call_result_133612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_133614 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), getitem___133613, int_133606)
    
    # Assigning a type to the variable 'tuple_var_assignment_133450' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_133450', subscript_call_result_133614)
    
    # Assigning a Name to a Name (line 141):
    # Getting the type of 'tuple_var_assignment_133449' (line 141)
    tuple_var_assignment_133449_133615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_133449')
    # Assigning a type to the variable 'MR' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'MR', tuple_var_assignment_133449_133615)
    
    # Assigning a Name to a Name (line 141):
    # Getting the type of 'tuple_var_assignment_133450' (line 141)
    tuple_var_assignment_133450_133616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'tuple_var_assignment_133450')
    # Assigning a type to the variable 'file_opened' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'file_opened', tuple_var_assignment_133450_133616)
    
    # Assigning a Call to a Name (line 142):
    
    # Assigning a Call to a Name (line 142):
    
    # Call to get_variables(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'variable_names' (line 142)
    variable_names_133619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'variable_names', False)
    # Processing the call keyword arguments (line 142)
    kwargs_133620 = {}
    # Getting the type of 'MR' (line 142)
    MR_133617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'MR', False)
    # Obtaining the member 'get_variables' of a type (line 142)
    get_variables_133618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), MR_133617, 'get_variables')
    # Calling get_variables(args, kwargs) (line 142)
    get_variables_call_result_133621 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), get_variables_133618, *[variable_names_133619], **kwargs_133620)
    
    # Assigning a type to the variable 'matfile_dict' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'matfile_dict', get_variables_call_result_133621)
    
    # Type idiom detected: calculating its left and rigth part (line 143)
    # Getting the type of 'mdict' (line 143)
    mdict_133622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'mdict')
    # Getting the type of 'None' (line 143)
    None_133623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'None')
    
    (may_be_133624, more_types_in_union_133625) = may_not_be_none(mdict_133622, None_133623)

    if may_be_133624:

        if more_types_in_union_133625:
            # Runtime conditional SSA (line 143)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to update(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'matfile_dict' (line 144)
        matfile_dict_133628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'matfile_dict', False)
        # Processing the call keyword arguments (line 144)
        kwargs_133629 = {}
        # Getting the type of 'mdict' (line 144)
        mdict_133626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'mdict', False)
        # Obtaining the member 'update' of a type (line 144)
        update_133627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), mdict_133626, 'update')
        # Calling update(args, kwargs) (line 144)
        update_call_result_133630 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), update_133627, *[matfile_dict_133628], **kwargs_133629)
        

        if more_types_in_union_133625:
            # Runtime conditional SSA for else branch (line 143)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_133624) or more_types_in_union_133625):
        
        # Assigning a Name to a Name (line 146):
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'matfile_dict' (line 146)
        matfile_dict_133631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'matfile_dict')
        # Assigning a type to the variable 'mdict' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'mdict', matfile_dict_133631)

        if (may_be_133624 and more_types_in_union_133625):
            # SSA join for if statement (line 143)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'file_opened' (line 147)
    file_opened_133632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'file_opened')
    # Testing the type of an if condition (line 147)
    if_condition_133633 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), file_opened_133632)
    # Assigning a type to the variable 'if_condition_133633' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_133633', if_condition_133633)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_133637 = {}
    # Getting the type of 'MR' (line 148)
    MR_133634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'MR', False)
    # Obtaining the member 'mat_stream' of a type (line 148)
    mat_stream_133635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), MR_133634, 'mat_stream')
    # Obtaining the member 'close' of a type (line 148)
    close_133636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), mat_stream_133635, 'close')
    # Calling close(args, kwargs) (line 148)
    close_call_result_133638 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), close_133636, *[], **kwargs_133637)
    
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'mdict' (line 149)
    mdict_133639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'mdict')
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type', mdict_133639)
    
    # ################# End of 'loadmat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'loadmat' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_133640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'loadmat'
    return stypy_return_type_133640

# Assigning a type to the variable 'loadmat' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'loadmat', loadmat)

@norecursion
def savemat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 154)
    True_133641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'True')
    str_133642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 19), 'str', '5')
    # Getting the type of 'False' (line 156)
    False_133643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'False')
    # Getting the type of 'False' (line 157)
    False_133644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'False')
    str_133645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'str', 'row')
    defaults = [True_133641, str_133642, False_133643, False_133644, str_133645]
    # Create a new context for function 'savemat'
    module_type_store = module_type_store.open_function_context('savemat', 152, 0, False)
    
    # Passed parameters checking function
    savemat.stypy_localization = localization
    savemat.stypy_type_of_self = None
    savemat.stypy_type_store = module_type_store
    savemat.stypy_function_name = 'savemat'
    savemat.stypy_param_names_list = ['file_name', 'mdict', 'appendmat', 'format', 'long_field_names', 'do_compression', 'oned_as']
    savemat.stypy_varargs_param_name = None
    savemat.stypy_kwargs_param_name = None
    savemat.stypy_call_defaults = defaults
    savemat.stypy_call_varargs = varargs
    savemat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'savemat', ['file_name', 'mdict', 'appendmat', 'format', 'long_field_names', 'do_compression', 'oned_as'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'savemat', localization, ['file_name', 'mdict', 'appendmat', 'format', 'long_field_names', 'do_compression', 'oned_as'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'savemat(...)' code ##################

    str_133646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', "\n    Save a dictionary of names and arrays into a MATLAB-style .mat file.\n\n    This saves the array objects in the given dictionary to a MATLAB-\n    style .mat file.\n\n    Parameters\n    ----------\n    file_name : str or file-like object\n        Name of the .mat file (.mat extension not needed if ``appendmat ==\n        True``).\n        Can also pass open file_like object.\n    mdict : dict\n        Dictionary from which to save matfile variables.\n    appendmat : bool, optional\n        True (the default) to append the .mat extension to the end of the\n        given filename, if not already present.\n    format : {'5', '4'}, string, optional\n        '5' (the default) for MATLAB 5 and up (to 7.2),\n        '4' for MATLAB 4 .mat files.\n    long_field_names : bool, optional\n        False (the default) - maximum field name length in a structure is\n        31 characters which is the documented maximum length.\n        True - maximum field name length in a structure is 63 characters\n        which works for MATLAB 7.6+.\n    do_compression : bool, optional\n        Whether or not to compress matrices on write.  Default is False.\n    oned_as : {'row', 'column'}, optional\n        If 'column', write 1-D numpy arrays as column vectors.\n        If 'row', write 1-D numpy arrays as row vectors.\n\n    See also\n    --------\n    mio4.MatFile4Writer\n    mio5.MatFile5Writer\n    ")
    
    # Assigning a Name to a Name (line 195):
    
    # Assigning a Name to a Name (line 195):
    # Getting the type of 'False' (line 195)
    False_133647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'False')
    # Assigning a type to the variable 'file_opened' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'file_opened', False_133647)
    
    # Type idiom detected: calculating its left and rigth part (line 196)
    str_133648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 26), 'str', 'write')
    # Getting the type of 'file_name' (line 196)
    file_name_133649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'file_name')
    
    (may_be_133650, more_types_in_union_133651) = may_provide_member(str_133648, file_name_133649)

    if may_be_133650:

        if more_types_in_union_133651:
            # Runtime conditional SSA (line 196)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'file_name' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'file_name', remove_not_member_provider_from_union(file_name_133649, 'write'))
        
        # Assigning a Name to a Name (line 198):
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'file_name' (line 198)
        file_name_133652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 'file_name')
        # Assigning a type to the variable 'file_stream' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'file_stream', file_name_133652)

        if more_types_in_union_133651:
            # Runtime conditional SSA for else branch (line 196)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_133650) or more_types_in_union_133651):
        # Assigning a type to the variable 'file_name' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'file_name', remove_member_provider_from_union(file_name_133649, 'write'))
        
        
        # Call to isinstance(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'file_name' (line 200)
        file_name_133654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'file_name', False)
        # Getting the type of 'string_types' (line 200)
        string_types_133655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 33), 'string_types', False)
        # Processing the call keyword arguments (line 200)
        kwargs_133656 = {}
        # Getting the type of 'isinstance' (line 200)
        isinstance_133653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 200)
        isinstance_call_result_133657 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), isinstance_133653, *[file_name_133654, string_types_133655], **kwargs_133656)
        
        # Testing the type of an if condition (line 200)
        if_condition_133658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), isinstance_call_result_133657)
        # Assigning a type to the variable 'if_condition_133658' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_133658', if_condition_133658)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'appendmat' (line 201)
        appendmat_133659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'appendmat')
        
        
        # Call to endswith(...): (line 201)
        # Processing the call arguments (line 201)
        str_133662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 52), 'str', '.mat')
        # Processing the call keyword arguments (line 201)
        kwargs_133663 = {}
        # Getting the type of 'file_name' (line 201)
        file_name_133660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'file_name', False)
        # Obtaining the member 'endswith' of a type (line 201)
        endswith_133661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 33), file_name_133660, 'endswith')
        # Calling endswith(args, kwargs) (line 201)
        endswith_call_result_133664 = invoke(stypy.reporting.localization.Localization(__file__, 201, 33), endswith_133661, *[str_133662], **kwargs_133663)
        
        # Applying the 'not' unary operator (line 201)
        result_not__133665 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 29), 'not', endswith_call_result_133664)
        
        # Applying the binary operator 'and' (line 201)
        result_and_keyword_133666 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 15), 'and', appendmat_133659, result_not__133665)
        
        # Testing the type of an if condition (line 201)
        if_condition_133667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), result_and_keyword_133666)
        # Assigning a type to the variable 'if_condition_133667' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_133667', if_condition_133667)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 202):
        
        # Assigning a BinOp to a Name (line 202):
        # Getting the type of 'file_name' (line 202)
        file_name_133668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'file_name')
        str_133669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 40), 'str', '.mat')
        # Applying the binary operator '+' (line 202)
        result_add_133670 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 28), '+', file_name_133668, str_133669)
        
        # Assigning a type to the variable 'file_name' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'file_name', result_add_133670)
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to open(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'file_name' (line 204)
        file_name_133672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'file_name', False)
        str_133673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 38), 'str', 'wb')
        # Processing the call keyword arguments (line 204)
        kwargs_133674 = {}
        # Getting the type of 'open' (line 204)
        open_133671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'open', False)
        # Calling open(args, kwargs) (line 204)
        open_call_result_133675 = invoke(stypy.reporting.localization.Localization(__file__, 204, 22), open_133671, *[file_name_133672, str_133673], **kwargs_133674)
        
        # Assigning a type to the variable 'file_stream' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'file_stream', open_call_result_133675)
        
        # Assigning a Name to a Name (line 205):
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'True' (line 205)
        True_133676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), 'True')
        # Assigning a type to the variable 'file_opened' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'file_opened', True_133676)

        if (may_be_133650 and more_types_in_union_133651):
            # SSA join for if statement (line 196)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'format' (line 207)
    format_133677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 7), 'format')
    str_133678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 17), 'str', '4')
    # Applying the binary operator '==' (line 207)
    result_eq_133679 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 7), '==', format_133677, str_133678)
    
    # Testing the type of an if condition (line 207)
    if_condition_133680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 4), result_eq_133679)
    # Assigning a type to the variable 'if_condition_133680' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'if_condition_133680', if_condition_133680)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'long_field_names' (line 208)
    long_field_names_133681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'long_field_names')
    # Testing the type of an if condition (line 208)
    if_condition_133682 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), long_field_names_133681)
    # Assigning a type to the variable 'if_condition_133682' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_133682', if_condition_133682)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 209)
    # Processing the call arguments (line 209)
    str_133684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 29), 'str', 'Long field names are not available for version 4 files')
    # Processing the call keyword arguments (line 209)
    kwargs_133685 = {}
    # Getting the type of 'ValueError' (line 209)
    ValueError_133683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 209)
    ValueError_call_result_133686 = invoke(stypy.reporting.localization.Localization(__file__, 209, 18), ValueError_133683, *[str_133684], **kwargs_133685)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 12), ValueError_call_result_133686, 'raise parameter', BaseException)
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to MatFile4Writer(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'file_stream' (line 210)
    file_stream_133688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'file_stream', False)
    # Getting the type of 'oned_as' (line 210)
    oned_as_133689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'oned_as', False)
    # Processing the call keyword arguments (line 210)
    kwargs_133690 = {}
    # Getting the type of 'MatFile4Writer' (line 210)
    MatFile4Writer_133687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'MatFile4Writer', False)
    # Calling MatFile4Writer(args, kwargs) (line 210)
    MatFile4Writer_call_result_133691 = invoke(stypy.reporting.localization.Localization(__file__, 210, 13), MatFile4Writer_133687, *[file_stream_133688, oned_as_133689], **kwargs_133690)
    
    # Assigning a type to the variable 'MW' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'MW', MatFile4Writer_call_result_133691)
    # SSA branch for the else part of an if statement (line 207)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'format' (line 211)
    format_133692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 9), 'format')
    str_133693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'str', '5')
    # Applying the binary operator '==' (line 211)
    result_eq_133694 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 9), '==', format_133692, str_133693)
    
    # Testing the type of an if condition (line 211)
    if_condition_133695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 9), result_eq_133694)
    # Assigning a type to the variable 'if_condition_133695' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 9), 'if_condition_133695', if_condition_133695)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to MatFile5Writer(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'file_stream' (line 212)
    file_stream_133697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'file_stream', False)
    # Processing the call keyword arguments (line 212)
    # Getting the type of 'do_compression' (line 213)
    do_compression_133698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'do_compression', False)
    keyword_133699 = do_compression_133698
    # Getting the type of 'True' (line 214)
    True_133700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 44), 'True', False)
    keyword_133701 = True_133700
    # Getting the type of 'long_field_names' (line 215)
    long_field_names_133702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 45), 'long_field_names', False)
    keyword_133703 = long_field_names_133702
    # Getting the type of 'oned_as' (line 216)
    oned_as_133704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 36), 'oned_as', False)
    keyword_133705 = oned_as_133704
    kwargs_133706 = {'oned_as': keyword_133705, 'unicode_strings': keyword_133701, 'do_compression': keyword_133699, 'long_field_names': keyword_133703}
    # Getting the type of 'MatFile5Writer' (line 212)
    MatFile5Writer_133696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'MatFile5Writer', False)
    # Calling MatFile5Writer(args, kwargs) (line 212)
    MatFile5Writer_call_result_133707 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), MatFile5Writer_133696, *[file_stream_133697], **kwargs_133706)
    
    # Assigning a type to the variable 'MW' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'MW', MatFile5Writer_call_result_133707)
    # SSA branch for the else part of an if statement (line 211)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 218)
    # Processing the call arguments (line 218)
    str_133709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'str', "Format should be '4' or '5'")
    # Processing the call keyword arguments (line 218)
    kwargs_133710 = {}
    # Getting the type of 'ValueError' (line 218)
    ValueError_133708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 218)
    ValueError_call_result_133711 = invoke(stypy.reporting.localization.Localization(__file__, 218, 14), ValueError_133708, *[str_133709], **kwargs_133710)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 218, 8), ValueError_call_result_133711, 'raise parameter', BaseException)
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to put_variables(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'mdict' (line 219)
    mdict_133714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'mdict', False)
    # Processing the call keyword arguments (line 219)
    kwargs_133715 = {}
    # Getting the type of 'MW' (line 219)
    MW_133712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'MW', False)
    # Obtaining the member 'put_variables' of a type (line 219)
    put_variables_133713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 4), MW_133712, 'put_variables')
    # Calling put_variables(args, kwargs) (line 219)
    put_variables_call_result_133716 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), put_variables_133713, *[mdict_133714], **kwargs_133715)
    
    
    # Getting the type of 'file_opened' (line 220)
    file_opened_133717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 7), 'file_opened')
    # Testing the type of an if condition (line 220)
    if_condition_133718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 4), file_opened_133717)
    # Assigning a type to the variable 'if_condition_133718' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'if_condition_133718', if_condition_133718)
    # SSA begins for if statement (line 220)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 221)
    # Processing the call keyword arguments (line 221)
    kwargs_133721 = {}
    # Getting the type of 'file_stream' (line 221)
    file_stream_133719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'file_stream', False)
    # Obtaining the member 'close' of a type (line 221)
    close_133720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), file_stream_133719, 'close')
    # Calling close(args, kwargs) (line 221)
    close_call_result_133722 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), close_133720, *[], **kwargs_133721)
    
    # SSA join for if statement (line 220)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'savemat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'savemat' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_133723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133723)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'savemat'
    return stypy_return_type_133723

# Assigning a type to the variable 'savemat' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'savemat', savemat)

@norecursion
def whosmat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 225)
    True_133724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 33), 'True')
    defaults = [True_133724]
    # Create a new context for function 'whosmat'
    module_type_store = module_type_store.open_function_context('whosmat', 224, 0, False)
    
    # Passed parameters checking function
    whosmat.stypy_localization = localization
    whosmat.stypy_type_of_self = None
    whosmat.stypy_type_store = module_type_store
    whosmat.stypy_function_name = 'whosmat'
    whosmat.stypy_param_names_list = ['file_name', 'appendmat']
    whosmat.stypy_varargs_param_name = None
    whosmat.stypy_kwargs_param_name = 'kwargs'
    whosmat.stypy_call_defaults = defaults
    whosmat.stypy_call_varargs = varargs
    whosmat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'whosmat', ['file_name', 'appendmat'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'whosmat', localization, ['file_name', 'appendmat'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'whosmat(...)' code ##################

    str_133725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, (-1)), 'str', '\n    List variables inside a MATLAB file.\n\n    Parameters\n    ----------\n    %(file_arg)s\n    %(append_arg)s\n    %(load_args)s\n    %(struct_arg)s\n\n    Returns\n    -------\n    variables : list of tuples\n        A list of tuples, where each tuple holds the matrix name (a string),\n        its shape (tuple of ints), and its data class (a string).\n        Possible data classes are: int8, uint8, int16, uint16, int32, uint32,\n        int64, uint64, single, double, cell, struct, object, char, sparse,\n        function, opaque, logical, unknown.\n\n    Notes\n    -----\n    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.\n\n    You will need an HDF5 python library to read matlab 7.3 format mat\n    files.  Because scipy does not supply one, we do not implement the\n    HDF5 / 7.3 interface here.\n\n    .. versionadded:: 0.12.0\n\n    ')
    
    # Assigning a Call to a Tuple (line 256):
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_133726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to mat_reader_factory(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'file_name' (line 256)
    file_name_133728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 41), 'file_name', False)
    # Processing the call keyword arguments (line 256)
    # Getting the type of 'kwargs' (line 256)
    kwargs_133729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 54), 'kwargs', False)
    kwargs_133730 = {'kwargs_133729': kwargs_133729}
    # Getting the type of 'mat_reader_factory' (line 256)
    mat_reader_factory_133727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'mat_reader_factory', False)
    # Calling mat_reader_factory(args, kwargs) (line 256)
    mat_reader_factory_call_result_133731 = invoke(stypy.reporting.localization.Localization(__file__, 256, 22), mat_reader_factory_133727, *[file_name_133728], **kwargs_133730)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___133732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), mat_reader_factory_call_result_133731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_133733 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___133732, int_133726)
    
    # Assigning a type to the variable 'tuple_var_assignment_133451' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_133451', subscript_call_result_133733)
    
    # Assigning a Subscript to a Name (line 256):
    
    # Obtaining the type of the subscript
    int_133734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'int')
    
    # Call to mat_reader_factory(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'file_name' (line 256)
    file_name_133736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 41), 'file_name', False)
    # Processing the call keyword arguments (line 256)
    # Getting the type of 'kwargs' (line 256)
    kwargs_133737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 54), 'kwargs', False)
    kwargs_133738 = {'kwargs_133737': kwargs_133737}
    # Getting the type of 'mat_reader_factory' (line 256)
    mat_reader_factory_133735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'mat_reader_factory', False)
    # Calling mat_reader_factory(args, kwargs) (line 256)
    mat_reader_factory_call_result_133739 = invoke(stypy.reporting.localization.Localization(__file__, 256, 22), mat_reader_factory_133735, *[file_name_133736], **kwargs_133738)
    
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___133740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), mat_reader_factory_call_result_133739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_133741 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), getitem___133740, int_133734)
    
    # Assigning a type to the variable 'tuple_var_assignment_133452' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_133452', subscript_call_result_133741)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_133451' (line 256)
    tuple_var_assignment_133451_133742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_133451')
    # Assigning a type to the variable 'ML' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'ML', tuple_var_assignment_133451_133742)
    
    # Assigning a Name to a Name (line 256):
    # Getting the type of 'tuple_var_assignment_133452' (line 256)
    tuple_var_assignment_133452_133743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'tuple_var_assignment_133452')
    # Assigning a type to the variable 'file_opened' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'file_opened', tuple_var_assignment_133452_133743)
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to list_variables(...): (line 257)
    # Processing the call keyword arguments (line 257)
    kwargs_133746 = {}
    # Getting the type of 'ML' (line 257)
    ML_133744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'ML', False)
    # Obtaining the member 'list_variables' of a type (line 257)
    list_variables_133745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), ML_133744, 'list_variables')
    # Calling list_variables(args, kwargs) (line 257)
    list_variables_call_result_133747 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), list_variables_133745, *[], **kwargs_133746)
    
    # Assigning a type to the variable 'variables' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'variables', list_variables_call_result_133747)
    
    # Getting the type of 'file_opened' (line 258)
    file_opened_133748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 7), 'file_opened')
    # Testing the type of an if condition (line 258)
    if_condition_133749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 4), file_opened_133748)
    # Assigning a type to the variable 'if_condition_133749' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'if_condition_133749', if_condition_133749)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 259)
    # Processing the call keyword arguments (line 259)
    kwargs_133753 = {}
    # Getting the type of 'ML' (line 259)
    ML_133750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'ML', False)
    # Obtaining the member 'mat_stream' of a type (line 259)
    mat_stream_133751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), ML_133750, 'mat_stream')
    # Obtaining the member 'close' of a type (line 259)
    close_133752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), mat_stream_133751, 'close')
    # Calling close(args, kwargs) (line 259)
    close_call_result_133754 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), close_133752, *[], **kwargs_133753)
    
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'variables' (line 260)
    variables_133755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'variables')
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', variables_133755)
    
    # ################# End of 'whosmat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'whosmat' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_133756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133756)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'whosmat'
    return stypy_return_type_133756

# Assigning a type to the variable 'whosmat' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'whosmat', whosmat)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
