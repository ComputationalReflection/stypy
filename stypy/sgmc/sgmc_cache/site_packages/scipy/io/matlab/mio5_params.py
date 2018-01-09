
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Constants and classes for matlab 5 read and write
2: 
3: See also mio5_utils.pyx where these same constants arise as c enums.
4: 
5: If you make changes in this file, don't forget to change mio5_utils.pyx
6: '''
7: from __future__ import division, print_function, absolute_import
8: 
9: import numpy as np
10: 
11: from .miobase import convert_dtypes
12: 
13: miINT8 = 1
14: miUINT8 = 2
15: miINT16 = 3
16: miUINT16 = 4
17: miINT32 = 5
18: miUINT32 = 6
19: miSINGLE = 7
20: miDOUBLE = 9
21: miINT64 = 12
22: miUINT64 = 13
23: miMATRIX = 14
24: miCOMPRESSED = 15
25: miUTF8 = 16
26: miUTF16 = 17
27: miUTF32 = 18
28: 
29: mxCELL_CLASS = 1
30: mxSTRUCT_CLASS = 2
31: # The March 2008 edition of "Matlab 7 MAT-File Format" says that
32: # mxOBJECT_CLASS = 3, whereas matrix.h says that mxLOGICAL = 3.
33: # Matlab 2008a appears to save logicals as type 9, so we assume that
34: # the document is correct.  See type 18, below.
35: mxOBJECT_CLASS = 3
36: mxCHAR_CLASS = 4
37: mxSPARSE_CLASS = 5
38: mxDOUBLE_CLASS = 6
39: mxSINGLE_CLASS = 7
40: mxINT8_CLASS = 8
41: mxUINT8_CLASS = 9
42: mxINT16_CLASS = 10
43: mxUINT16_CLASS = 11
44: mxINT32_CLASS = 12
45: mxUINT32_CLASS = 13
46: # The following are not in the March 2008 edition of "Matlab 7
47: # MAT-File Format," but were guessed from matrix.h.
48: mxINT64_CLASS = 14
49: mxUINT64_CLASS = 15
50: mxFUNCTION_CLASS = 16
51: # Not doing anything with these at the moment.
52: mxOPAQUE_CLASS = 17  # This appears to be a function workspace
53: # Thread 'saveing/loading symbol table of annymous functions', octave-maintainers, April-May 2007
54: # https://lists.gnu.org/archive/html/octave-maintainers/2007-04/msg00031.html
55: # https://lists.gnu.org/archive/html/octave-maintainers/2007-05/msg00032.html
56: # (Was/Deprecated: https://www-old.cae.wisc.edu/pipermail/octave-maintainers/2007-May/002824.html)
57: mxOBJECT_CLASS_FROM_MATRIX_H = 18
58: 
59: mdtypes_template = {
60:     miINT8: 'i1',
61:     miUINT8: 'u1',
62:     miINT16: 'i2',
63:     miUINT16: 'u2',
64:     miINT32: 'i4',
65:     miUINT32: 'u4',
66:     miSINGLE: 'f4',
67:     miDOUBLE: 'f8',
68:     miINT64: 'i8',
69:     miUINT64: 'u8',
70:     miUTF8: 'u1',
71:     miUTF16: 'u2',
72:     miUTF32: 'u4',
73:     'file_header': [('description', 'S116'),
74:                     ('subsystem_offset', 'i8'),
75:                     ('version', 'u2'),
76:                     ('endian_test', 'S2')],
77:     'tag_full': [('mdtype', 'u4'), ('byte_count', 'u4')],
78:     'tag_smalldata':[('byte_count_mdtype', 'u4'), ('data', 'S4')],
79:     'array_flags': [('data_type', 'u4'),
80:                     ('byte_count', 'u4'),
81:                     ('flags_class','u4'),
82:                     ('nzmax', 'u4')],
83:     'U1': 'U1',
84:     }
85: 
86: mclass_dtypes_template = {
87:     mxINT8_CLASS: 'i1',
88:     mxUINT8_CLASS: 'u1',
89:     mxINT16_CLASS: 'i2',
90:     mxUINT16_CLASS: 'u2',
91:     mxINT32_CLASS: 'i4',
92:     mxUINT32_CLASS: 'u4',
93:     mxINT64_CLASS: 'i8',
94:     mxUINT64_CLASS: 'u8',
95:     mxSINGLE_CLASS: 'f4',
96:     mxDOUBLE_CLASS: 'f8',
97:     }
98: 
99: mclass_info = {
100:     mxINT8_CLASS: 'int8',
101:     mxUINT8_CLASS: 'uint8',
102:     mxINT16_CLASS: 'int16',
103:     mxUINT16_CLASS: 'uint16',
104:     mxINT32_CLASS: 'int32',
105:     mxUINT32_CLASS: 'uint32',
106:     mxINT64_CLASS: 'int64',
107:     mxUINT64_CLASS: 'uint64',
108:     mxSINGLE_CLASS: 'single',
109:     mxDOUBLE_CLASS: 'double',
110:     mxCELL_CLASS: 'cell',
111:     mxSTRUCT_CLASS: 'struct',
112:     mxOBJECT_CLASS: 'object',
113:     mxCHAR_CLASS: 'char',
114:     mxSPARSE_CLASS: 'sparse',
115:     mxFUNCTION_CLASS: 'function',
116:     mxOPAQUE_CLASS: 'opaque',
117:     }
118: 
119: NP_TO_MTYPES = {
120:     'f8': miDOUBLE,
121:     'c32': miDOUBLE,
122:     'c24': miDOUBLE,
123:     'c16': miDOUBLE,
124:     'f4': miSINGLE,
125:     'c8': miSINGLE,
126:     'i8': miINT64,
127:     'i4': miINT32,
128:     'i2': miINT16,
129:     'i1': miINT8,
130:     'u8': miUINT64,
131:     'u4': miUINT32,
132:     'u2': miUINT16,
133:     'u1': miUINT8,
134:     'S1': miUINT8,
135:     'U1': miUTF16,
136:     'b1': miUINT8,  # not standard but seems MATLAB uses this (gh-4022)
137:     }
138: 
139: 
140: NP_TO_MXTYPES = {
141:     'f8': mxDOUBLE_CLASS,
142:     'c32': mxDOUBLE_CLASS,
143:     'c24': mxDOUBLE_CLASS,
144:     'c16': mxDOUBLE_CLASS,
145:     'f4': mxSINGLE_CLASS,
146:     'c8': mxSINGLE_CLASS,
147:     'i8': mxINT64_CLASS,
148:     'i4': mxINT32_CLASS,
149:     'i2': mxINT16_CLASS,
150:     'i1': mxINT8_CLASS,
151:     'u8': mxUINT64_CLASS,
152:     'u4': mxUINT32_CLASS,
153:     'u2': mxUINT16_CLASS,
154:     'u1': mxUINT8_CLASS,
155:     'S1': mxUINT8_CLASS,
156:     'b1': mxUINT8_CLASS,  # not standard but seems MATLAB uses this
157:     }
158: 
159: ''' Before release v7.1 (release 14) matlab (TM) used the system
160: default character encoding scheme padded out to 16-bits. Release 14
161: and later use Unicode. When saving character data, R14 checks if it
162: can be encoded in 7-bit ascii, and saves in that format if so.'''
163: 
164: codecs_template = {
165:     miUTF8: {'codec': 'utf_8', 'width': 1},
166:     miUTF16: {'codec': 'utf_16', 'width': 2},
167:     miUTF32: {'codec': 'utf_32','width': 4},
168:     }
169: 
170: 
171: def _convert_codecs(template, byte_order):
172:     ''' Convert codec template mapping to byte order
173: 
174:     Set codecs not on this system to None
175: 
176:     Parameters
177:     ----------
178:     template : mapping
179:        key, value are respectively codec name, and root name for codec
180:        (without byte order suffix)
181:     byte_order : {'<', '>'}
182:        code for little or big endian
183: 
184:     Returns
185:     -------
186:     codecs : dict
187:        key, value are name, codec (as in .encode(codec))
188:     '''
189:     codecs = {}
190:     postfix = byte_order == '<' and '_le' or '_be'
191:     for k, v in template.items():
192:         codec = v['codec']
193:         try:
194:             " ".encode(codec)
195:         except LookupError:
196:             codecs[k] = None
197:             continue
198:         if v['width'] > 1:
199:             codec += postfix
200:         codecs[k] = codec
201:     return codecs.copy()
202: 
203: 
204: MDTYPES = {}
205: for _bytecode in '<>':
206:     _def = {'dtypes': convert_dtypes(mdtypes_template, _bytecode),
207:             'classes': convert_dtypes(mclass_dtypes_template, _bytecode),
208:             'codecs': _convert_codecs(codecs_template, _bytecode)}
209:     MDTYPES[_bytecode] = _def
210: 
211: 
212: class mat_struct(object):
213:     ''' Placeholder for holding read data from structs
214: 
215:     We use instances of this class when the user passes False as a value to the
216:     ``struct_as_record`` parameter of the :func:`scipy.io.matlab.loadmat`
217:     function.
218:     '''
219:     pass
220: 
221: 
222: class MatlabObject(np.ndarray):
223:     ''' ndarray Subclass to contain matlab object '''
224:     def __new__(cls, input_array, classname=None):
225:         # Input array is an already formed ndarray instance
226:         # We first cast to be our class type
227:         obj = np.asarray(input_array).view(cls)
228:         # add the new attribute to the created instance
229:         obj.classname = classname
230:         # Finally, we must return the newly created object:
231:         return obj
232: 
233:     def __array_finalize__(self,obj):
234:         # reset the attribute from passed original object
235:         self.classname = getattr(obj, 'classname', None)
236:         # We do not need to return anything
237: 
238: 
239: class MatlabFunction(np.ndarray):
240:     ''' Subclass to signal this is a matlab function '''
241:     def __new__(cls, input_array):
242:         obj = np.asarray(input_array).view(cls)
243:         return obj
244: 
245: 
246: class MatlabOpaque(np.ndarray):
247:     ''' Subclass to signal this is a matlab opaque matrix '''
248:     def __new__(cls, input_array):
249:         obj = np.asarray(input_array).view(cls)
250:         return obj
251: 
252: 
253: OPAQUE_DTYPE = np.dtype(
254:     [('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])
255: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_137129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', " Constants and classes for matlab 5 read and write\n\nSee also mio5_utils.pyx where these same constants arise as c enums.\n\nIf you make changes in this file, don't forget to change mio5_utils.pyx\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_137130 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_137130) is not StypyTypeError):

    if (import_137130 != 'pyd_module'):
        __import__(import_137130)
        sys_modules_137131 = sys.modules[import_137130]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_137131.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_137130)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.io.matlab.miobase import convert_dtypes' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/')
import_137132 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.io.matlab.miobase')

if (type(import_137132) is not StypyTypeError):

    if (import_137132 != 'pyd_module'):
        __import__(import_137132)
        sys_modules_137133 = sys.modules[import_137132]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.io.matlab.miobase', sys_modules_137133.module_type_store, module_type_store, ['convert_dtypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_137133, sys_modules_137133.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.miobase import convert_dtypes

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.io.matlab.miobase', None, module_type_store, ['convert_dtypes'], [convert_dtypes])

else:
    # Assigning a type to the variable 'scipy.io.matlab.miobase' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.io.matlab.miobase', import_137132)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/')


# Assigning a Num to a Name (line 13):
int_137134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'int')
# Assigning a type to the variable 'miINT8' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'miINT8', int_137134)

# Assigning a Num to a Name (line 14):
int_137135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'int')
# Assigning a type to the variable 'miUINT8' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'miUINT8', int_137135)

# Assigning a Num to a Name (line 15):
int_137136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'int')
# Assigning a type to the variable 'miINT16' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'miINT16', int_137136)

# Assigning a Num to a Name (line 16):
int_137137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'int')
# Assigning a type to the variable 'miUINT16' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'miUINT16', int_137137)

# Assigning a Num to a Name (line 17):
int_137138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'int')
# Assigning a type to the variable 'miINT32' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'miINT32', int_137138)

# Assigning a Num to a Name (line 18):
int_137139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'int')
# Assigning a type to the variable 'miUINT32' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'miUINT32', int_137139)

# Assigning a Num to a Name (line 19):
int_137140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'int')
# Assigning a type to the variable 'miSINGLE' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'miSINGLE', int_137140)

# Assigning a Num to a Name (line 20):
int_137141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'int')
# Assigning a type to the variable 'miDOUBLE' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'miDOUBLE', int_137141)

# Assigning a Num to a Name (line 21):
int_137142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'int')
# Assigning a type to the variable 'miINT64' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'miINT64', int_137142)

# Assigning a Num to a Name (line 22):
int_137143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
# Assigning a type to the variable 'miUINT64' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'miUINT64', int_137143)

# Assigning a Num to a Name (line 23):
int_137144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'int')
# Assigning a type to the variable 'miMATRIX' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'miMATRIX', int_137144)

# Assigning a Num to a Name (line 24):
int_137145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'int')
# Assigning a type to the variable 'miCOMPRESSED' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'miCOMPRESSED', int_137145)

# Assigning a Num to a Name (line 25):
int_137146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'int')
# Assigning a type to the variable 'miUTF8' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'miUTF8', int_137146)

# Assigning a Num to a Name (line 26):
int_137147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'int')
# Assigning a type to the variable 'miUTF16' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'miUTF16', int_137147)

# Assigning a Num to a Name (line 27):
int_137148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'int')
# Assigning a type to the variable 'miUTF32' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'miUTF32', int_137148)

# Assigning a Num to a Name (line 29):
int_137149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'int')
# Assigning a type to the variable 'mxCELL_CLASS' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'mxCELL_CLASS', int_137149)

# Assigning a Num to a Name (line 30):
int_137150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
# Assigning a type to the variable 'mxSTRUCT_CLASS' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'mxSTRUCT_CLASS', int_137150)

# Assigning a Num to a Name (line 35):
int_137151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'int')
# Assigning a type to the variable 'mxOBJECT_CLASS' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'mxOBJECT_CLASS', int_137151)

# Assigning a Num to a Name (line 36):
int_137152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'int')
# Assigning a type to the variable 'mxCHAR_CLASS' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'mxCHAR_CLASS', int_137152)

# Assigning a Num to a Name (line 37):
int_137153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'int')
# Assigning a type to the variable 'mxSPARSE_CLASS' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'mxSPARSE_CLASS', int_137153)

# Assigning a Num to a Name (line 38):
int_137154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'int')
# Assigning a type to the variable 'mxDOUBLE_CLASS' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'mxDOUBLE_CLASS', int_137154)

# Assigning a Num to a Name (line 39):
int_137155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'int')
# Assigning a type to the variable 'mxSINGLE_CLASS' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'mxSINGLE_CLASS', int_137155)

# Assigning a Num to a Name (line 40):
int_137156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'int')
# Assigning a type to the variable 'mxINT8_CLASS' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'mxINT8_CLASS', int_137156)

# Assigning a Num to a Name (line 41):
int_137157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'int')
# Assigning a type to the variable 'mxUINT8_CLASS' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'mxUINT8_CLASS', int_137157)

# Assigning a Num to a Name (line 42):
int_137158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'int')
# Assigning a type to the variable 'mxINT16_CLASS' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'mxINT16_CLASS', int_137158)

# Assigning a Num to a Name (line 43):
int_137159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'int')
# Assigning a type to the variable 'mxUINT16_CLASS' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'mxUINT16_CLASS', int_137159)

# Assigning a Num to a Name (line 44):
int_137160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
# Assigning a type to the variable 'mxINT32_CLASS' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'mxINT32_CLASS', int_137160)

# Assigning a Num to a Name (line 45):
int_137161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'int')
# Assigning a type to the variable 'mxUINT32_CLASS' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'mxUINT32_CLASS', int_137161)

# Assigning a Num to a Name (line 48):
int_137162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'int')
# Assigning a type to the variable 'mxINT64_CLASS' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'mxINT64_CLASS', int_137162)

# Assigning a Num to a Name (line 49):
int_137163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'int')
# Assigning a type to the variable 'mxUINT64_CLASS' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'mxUINT64_CLASS', int_137163)

# Assigning a Num to a Name (line 50):
int_137164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'int')
# Assigning a type to the variable 'mxFUNCTION_CLASS' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'mxFUNCTION_CLASS', int_137164)

# Assigning a Num to a Name (line 52):
int_137165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'int')
# Assigning a type to the variable 'mxOPAQUE_CLASS' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'mxOPAQUE_CLASS', int_137165)

# Assigning a Num to a Name (line 57):
int_137166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'int')
# Assigning a type to the variable 'mxOBJECT_CLASS_FROM_MATRIX_H' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'mxOBJECT_CLASS_FROM_MATRIX_H', int_137166)

# Assigning a Dict to a Name (line 59):

# Obtaining an instance of the builtin type 'dict' (line 59)
dict_137167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 59)
# Adding element type (key, value) (line 59)
# Getting the type of 'miINT8' (line 60)
miINT8_137168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'miINT8')
str_137169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'str', 'i1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miINT8_137168, str_137169))
# Adding element type (key, value) (line 59)
# Getting the type of 'miUINT8' (line 61)
miUINT8_137170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'miUINT8')
str_137171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'str', 'u1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miUINT8_137170, str_137171))
# Adding element type (key, value) (line 59)
# Getting the type of 'miINT16' (line 62)
miINT16_137172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'miINT16')
str_137173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 13), 'str', 'i2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miINT16_137172, str_137173))
# Adding element type (key, value) (line 59)
# Getting the type of 'miUINT16' (line 63)
miUINT16_137174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'miUINT16')
str_137175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'str', 'u2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miUINT16_137174, str_137175))
# Adding element type (key, value) (line 59)
# Getting the type of 'miINT32' (line 64)
miINT32_137176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'miINT32')
str_137177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'str', 'i4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miINT32_137176, str_137177))
# Adding element type (key, value) (line 59)
# Getting the type of 'miUINT32' (line 65)
miUINT32_137178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'miUINT32')
str_137179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 14), 'str', 'u4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miUINT32_137178, str_137179))
# Adding element type (key, value) (line 59)
# Getting the type of 'miSINGLE' (line 66)
miSINGLE_137180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'miSINGLE')
str_137181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'str', 'f4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miSINGLE_137180, str_137181))
# Adding element type (key, value) (line 59)
# Getting the type of 'miDOUBLE' (line 67)
miDOUBLE_137182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'miDOUBLE')
str_137183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'str', 'f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miDOUBLE_137182, str_137183))
# Adding element type (key, value) (line 59)
# Getting the type of 'miINT64' (line 68)
miINT64_137184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'miINT64')
str_137185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 13), 'str', 'i8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miINT64_137184, str_137185))
# Adding element type (key, value) (line 59)
# Getting the type of 'miUINT64' (line 69)
miUINT64_137186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'miUINT64')
str_137187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 14), 'str', 'u8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miUINT64_137186, str_137187))
# Adding element type (key, value) (line 59)
# Getting the type of 'miUTF8' (line 70)
miUTF8_137188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'miUTF8')
str_137189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'str', 'u1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miUTF8_137188, str_137189))
# Adding element type (key, value) (line 59)
# Getting the type of 'miUTF16' (line 71)
miUTF16_137190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'miUTF16')
str_137191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 13), 'str', 'u2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miUTF16_137190, str_137191))
# Adding element type (key, value) (line 59)
# Getting the type of 'miUTF32' (line 72)
miUTF32_137192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'miUTF32')
str_137193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 13), 'str', 'u4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (miUTF32_137192, str_137193))
# Adding element type (key, value) (line 59)
str_137194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'file_header')

# Obtaining an instance of the builtin type 'list' (line 73)
list_137195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 73)
# Adding element type (line 73)

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_137196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
str_137197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'str', 'description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 21), tuple_137196, str_137197)
# Adding element type (line 73)
str_137198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'str', 'S116')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 21), tuple_137196, str_137198)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 19), list_137195, tuple_137196)
# Adding element type (line 73)

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_137199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
str_137200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'str', 'subsystem_offset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), tuple_137199, str_137200)
# Adding element type (line 74)
str_137201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 41), 'str', 'i8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), tuple_137199, str_137201)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 19), list_137195, tuple_137199)
# Adding element type (line 73)

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_137202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)
str_137203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'str', 'version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), tuple_137202, str_137203)
# Adding element type (line 75)
str_137204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 32), 'str', 'u2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), tuple_137202, str_137204)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 19), list_137195, tuple_137202)
# Adding element type (line 73)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_137205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_137206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'str', 'endian_test')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 21), tuple_137205, str_137206)
# Adding element type (line 76)
str_137207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'str', 'S2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 21), tuple_137205, str_137207)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 19), list_137195, tuple_137205)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (str_137194, list_137195))
# Adding element type (key, value) (line 59)
str_137208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'tag_full')

# Obtaining an instance of the builtin type 'list' (line 77)
list_137209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 77)
# Adding element type (line 77)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_137210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
str_137211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 18), 'str', 'mdtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 18), tuple_137210, str_137211)
# Adding element type (line 77)
str_137212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 28), 'str', 'u4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 18), tuple_137210, str_137212)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 16), list_137209, tuple_137210)
# Adding element type (line 77)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_137213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
str_137214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'str', 'byte_count')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 36), tuple_137213, str_137214)
# Adding element type (line 77)
str_137215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 50), 'str', 'u4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 36), tuple_137213, str_137215)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 16), list_137209, tuple_137213)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (str_137208, list_137209))
# Adding element type (key, value) (line 59)
str_137216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'str', 'tag_smalldata')

# Obtaining an instance of the builtin type 'list' (line 78)
list_137217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 78)
# Adding element type (line 78)

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_137218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
str_137219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'str', 'byte_count_mdtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 22), tuple_137218, str_137219)
# Adding element type (line 78)
str_137220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 43), 'str', 'u4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 22), tuple_137218, str_137220)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 20), list_137217, tuple_137218)
# Adding element type (line 78)

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_137221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 51), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
str_137222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 51), 'str', 'data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 51), tuple_137221, str_137222)
# Adding element type (line 78)
str_137223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 59), 'str', 'S4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 51), tuple_137221, str_137223)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 20), list_137217, tuple_137221)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (str_137216, list_137217))
# Adding element type (key, value) (line 59)
str_137224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'str', 'array_flags')

# Obtaining an instance of the builtin type 'list' (line 79)
list_137225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 79)
# Adding element type (line 79)

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_137226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
str_137227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'str', 'data_type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 21), tuple_137226, str_137227)
# Adding element type (line 79)
str_137228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 34), 'str', 'u4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 21), tuple_137226, str_137228)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_137225, tuple_137226)
# Adding element type (line 79)

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_137229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
str_137230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'str', 'byte_count')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), tuple_137229, str_137230)
# Adding element type (line 80)
str_137231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 35), 'str', 'u4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), tuple_137229, str_137231)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_137225, tuple_137229)
# Adding element type (line 79)

# Obtaining an instance of the builtin type 'tuple' (line 81)
tuple_137232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 81)
# Adding element type (line 81)
str_137233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'str', 'flags_class')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), tuple_137232, str_137233)
# Adding element type (line 81)
str_137234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 35), 'str', 'u4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), tuple_137232, str_137234)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_137225, tuple_137232)
# Adding element type (line 79)

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_137235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
str_137236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'str', 'nzmax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 21), tuple_137235, str_137236)
# Adding element type (line 82)
str_137237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'str', 'u4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 21), tuple_137235, str_137237)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_137225, tuple_137235)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (str_137224, list_137225))
# Adding element type (key, value) (line 59)
str_137238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'str', 'U1')
str_137239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 10), 'str', 'U1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 19), dict_137167, (str_137238, str_137239))

# Assigning a type to the variable 'mdtypes_template' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'mdtypes_template', dict_137167)

# Assigning a Dict to a Name (line 86):

# Obtaining an instance of the builtin type 'dict' (line 86)
dict_137240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 86)
# Adding element type (key, value) (line 86)
# Getting the type of 'mxINT8_CLASS' (line 87)
mxINT8_CLASS_137241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'mxINT8_CLASS')
str_137242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'str', 'i1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxINT8_CLASS_137241, str_137242))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxUINT8_CLASS' (line 88)
mxUINT8_CLASS_137243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'mxUINT8_CLASS')
str_137244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'str', 'u1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxUINT8_CLASS_137243, str_137244))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxINT16_CLASS' (line 89)
mxINT16_CLASS_137245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'mxINT16_CLASS')
str_137246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'str', 'i2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxINT16_CLASS_137245, str_137246))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxUINT16_CLASS' (line 90)
mxUINT16_CLASS_137247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'mxUINT16_CLASS')
str_137248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'str', 'u2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxUINT16_CLASS_137247, str_137248))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxINT32_CLASS' (line 91)
mxINT32_CLASS_137249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'mxINT32_CLASS')
str_137250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'str', 'i4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxINT32_CLASS_137249, str_137250))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxUINT32_CLASS' (line 92)
mxUINT32_CLASS_137251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'mxUINT32_CLASS')
str_137252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'str', 'u4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxUINT32_CLASS_137251, str_137252))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxINT64_CLASS' (line 93)
mxINT64_CLASS_137253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'mxINT64_CLASS')
str_137254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'str', 'i8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxINT64_CLASS_137253, str_137254))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxUINT64_CLASS' (line 94)
mxUINT64_CLASS_137255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'mxUINT64_CLASS')
str_137256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'str', 'u8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxUINT64_CLASS_137255, str_137256))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxSINGLE_CLASS' (line 95)
mxSINGLE_CLASS_137257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'mxSINGLE_CLASS')
str_137258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'str', 'f4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxSINGLE_CLASS_137257, str_137258))
# Adding element type (key, value) (line 86)
# Getting the type of 'mxDOUBLE_CLASS' (line 96)
mxDOUBLE_CLASS_137259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'mxDOUBLE_CLASS')
str_137260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'str', 'f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 25), dict_137240, (mxDOUBLE_CLASS_137259, str_137260))

# Assigning a type to the variable 'mclass_dtypes_template' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'mclass_dtypes_template', dict_137240)

# Assigning a Dict to a Name (line 99):

# Obtaining an instance of the builtin type 'dict' (line 99)
dict_137261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 99)
# Adding element type (key, value) (line 99)
# Getting the type of 'mxINT8_CLASS' (line 100)
mxINT8_CLASS_137262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'mxINT8_CLASS')
str_137263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 18), 'str', 'int8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxINT8_CLASS_137262, str_137263))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxUINT8_CLASS' (line 101)
mxUINT8_CLASS_137264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'mxUINT8_CLASS')
str_137265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'str', 'uint8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxUINT8_CLASS_137264, str_137265))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxINT16_CLASS' (line 102)
mxINT16_CLASS_137266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'mxINT16_CLASS')
str_137267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'str', 'int16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxINT16_CLASS_137266, str_137267))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxUINT16_CLASS' (line 103)
mxUINT16_CLASS_137268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'mxUINT16_CLASS')
str_137269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'str', 'uint16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxUINT16_CLASS_137268, str_137269))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxINT32_CLASS' (line 104)
mxINT32_CLASS_137270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'mxINT32_CLASS')
str_137271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'str', 'int32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxINT32_CLASS_137270, str_137271))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxUINT32_CLASS' (line 105)
mxUINT32_CLASS_137272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'mxUINT32_CLASS')
str_137273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'str', 'uint32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxUINT32_CLASS_137272, str_137273))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxINT64_CLASS' (line 106)
mxINT64_CLASS_137274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'mxINT64_CLASS')
str_137275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'str', 'int64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxINT64_CLASS_137274, str_137275))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxUINT64_CLASS' (line 107)
mxUINT64_CLASS_137276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'mxUINT64_CLASS')
str_137277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 20), 'str', 'uint64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxUINT64_CLASS_137276, str_137277))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxSINGLE_CLASS' (line 108)
mxSINGLE_CLASS_137278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'mxSINGLE_CLASS')
str_137279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'str', 'single')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxSINGLE_CLASS_137278, str_137279))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxDOUBLE_CLASS' (line 109)
mxDOUBLE_CLASS_137280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'mxDOUBLE_CLASS')
str_137281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'str', 'double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxDOUBLE_CLASS_137280, str_137281))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxCELL_CLASS' (line 110)
mxCELL_CLASS_137282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'mxCELL_CLASS')
str_137283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 18), 'str', 'cell')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxCELL_CLASS_137282, str_137283))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxSTRUCT_CLASS' (line 111)
mxSTRUCT_CLASS_137284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'mxSTRUCT_CLASS')
str_137285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 20), 'str', 'struct')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxSTRUCT_CLASS_137284, str_137285))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxOBJECT_CLASS' (line 112)
mxOBJECT_CLASS_137286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'mxOBJECT_CLASS')
str_137287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'str', 'object')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxOBJECT_CLASS_137286, str_137287))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxCHAR_CLASS' (line 113)
mxCHAR_CLASS_137288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'mxCHAR_CLASS')
str_137289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 18), 'str', 'char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxCHAR_CLASS_137288, str_137289))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxSPARSE_CLASS' (line 114)
mxSPARSE_CLASS_137290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'mxSPARSE_CLASS')
str_137291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'str', 'sparse')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxSPARSE_CLASS_137290, str_137291))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxFUNCTION_CLASS' (line 115)
mxFUNCTION_CLASS_137292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'mxFUNCTION_CLASS')
str_137293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 22), 'str', 'function')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxFUNCTION_CLASS_137292, str_137293))
# Adding element type (key, value) (line 99)
# Getting the type of 'mxOPAQUE_CLASS' (line 116)
mxOPAQUE_CLASS_137294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'mxOPAQUE_CLASS')
str_137295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'str', 'opaque')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), dict_137261, (mxOPAQUE_CLASS_137294, str_137295))

# Assigning a type to the variable 'mclass_info' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'mclass_info', dict_137261)

# Assigning a Dict to a Name (line 119):

# Obtaining an instance of the builtin type 'dict' (line 119)
dict_137296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 119)
# Adding element type (key, value) (line 119)
str_137297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 4), 'str', 'f8')
# Getting the type of 'miDOUBLE' (line 120)
miDOUBLE_137298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 10), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137297, miDOUBLE_137298))
# Adding element type (key, value) (line 119)
str_137299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'str', 'c32')
# Getting the type of 'miDOUBLE' (line 121)
miDOUBLE_137300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137299, miDOUBLE_137300))
# Adding element type (key, value) (line 119)
str_137301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 4), 'str', 'c24')
# Getting the type of 'miDOUBLE' (line 122)
miDOUBLE_137302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137301, miDOUBLE_137302))
# Adding element type (key, value) (line 119)
str_137303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 4), 'str', 'c16')
# Getting the type of 'miDOUBLE' (line 123)
miDOUBLE_137304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'miDOUBLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137303, miDOUBLE_137304))
# Adding element type (key, value) (line 119)
str_137305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'str', 'f4')
# Getting the type of 'miSINGLE' (line 124)
miSINGLE_137306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'miSINGLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137305, miSINGLE_137306))
# Adding element type (key, value) (line 119)
str_137307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'str', 'c8')
# Getting the type of 'miSINGLE' (line 125)
miSINGLE_137308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 10), 'miSINGLE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137307, miSINGLE_137308))
# Adding element type (key, value) (line 119)
str_137309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'str', 'i8')
# Getting the type of 'miINT64' (line 126)
miINT64_137310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 10), 'miINT64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137309, miINT64_137310))
# Adding element type (key, value) (line 119)
str_137311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'str', 'i4')
# Getting the type of 'miINT32' (line 127)
miINT32_137312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 10), 'miINT32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137311, miINT32_137312))
# Adding element type (key, value) (line 119)
str_137313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'str', 'i2')
# Getting the type of 'miINT16' (line 128)
miINT16_137314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 10), 'miINT16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137313, miINT16_137314))
# Adding element type (key, value) (line 119)
str_137315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 4), 'str', 'i1')
# Getting the type of 'miINT8' (line 129)
miINT8_137316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 10), 'miINT8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137315, miINT8_137316))
# Adding element type (key, value) (line 119)
str_137317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'str', 'u8')
# Getting the type of 'miUINT64' (line 130)
miUINT64_137318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 10), 'miUINT64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137317, miUINT64_137318))
# Adding element type (key, value) (line 119)
str_137319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'str', 'u4')
# Getting the type of 'miUINT32' (line 131)
miUINT32_137320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 10), 'miUINT32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137319, miUINT32_137320))
# Adding element type (key, value) (line 119)
str_137321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'str', 'u2')
# Getting the type of 'miUINT16' (line 132)
miUINT16_137322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 10), 'miUINT16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137321, miUINT16_137322))
# Adding element type (key, value) (line 119)
str_137323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'str', 'u1')
# Getting the type of 'miUINT8' (line 133)
miUINT8_137324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 10), 'miUINT8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137323, miUINT8_137324))
# Adding element type (key, value) (line 119)
str_137325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 4), 'str', 'S1')
# Getting the type of 'miUINT8' (line 134)
miUINT8_137326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 10), 'miUINT8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137325, miUINT8_137326))
# Adding element type (key, value) (line 119)
str_137327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'str', 'U1')
# Getting the type of 'miUTF16' (line 135)
miUTF16_137328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 10), 'miUTF16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137327, miUTF16_137328))
# Adding element type (key, value) (line 119)
str_137329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'str', 'b1')
# Getting the type of 'miUINT8' (line 136)
miUINT8_137330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 10), 'miUINT8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), dict_137296, (str_137329, miUINT8_137330))

# Assigning a type to the variable 'NP_TO_MTYPES' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'NP_TO_MTYPES', dict_137296)

# Assigning a Dict to a Name (line 140):

# Obtaining an instance of the builtin type 'dict' (line 140)
dict_137331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 140)
# Adding element type (key, value) (line 140)
str_137332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'str', 'f8')
# Getting the type of 'mxDOUBLE_CLASS' (line 141)
mxDOUBLE_CLASS_137333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 10), 'mxDOUBLE_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137332, mxDOUBLE_CLASS_137333))
# Adding element type (key, value) (line 140)
str_137334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'str', 'c32')
# Getting the type of 'mxDOUBLE_CLASS' (line 142)
mxDOUBLE_CLASS_137335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'mxDOUBLE_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137334, mxDOUBLE_CLASS_137335))
# Adding element type (key, value) (line 140)
str_137336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'str', 'c24')
# Getting the type of 'mxDOUBLE_CLASS' (line 143)
mxDOUBLE_CLASS_137337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'mxDOUBLE_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137336, mxDOUBLE_CLASS_137337))
# Adding element type (key, value) (line 140)
str_137338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 4), 'str', 'c16')
# Getting the type of 'mxDOUBLE_CLASS' (line 144)
mxDOUBLE_CLASS_137339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'mxDOUBLE_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137338, mxDOUBLE_CLASS_137339))
# Adding element type (key, value) (line 140)
str_137340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 4), 'str', 'f4')
# Getting the type of 'mxSINGLE_CLASS' (line 145)
mxSINGLE_CLASS_137341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 10), 'mxSINGLE_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137340, mxSINGLE_CLASS_137341))
# Adding element type (key, value) (line 140)
str_137342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 4), 'str', 'c8')
# Getting the type of 'mxSINGLE_CLASS' (line 146)
mxSINGLE_CLASS_137343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 10), 'mxSINGLE_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137342, mxSINGLE_CLASS_137343))
# Adding element type (key, value) (line 140)
str_137344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 4), 'str', 'i8')
# Getting the type of 'mxINT64_CLASS' (line 147)
mxINT64_CLASS_137345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 10), 'mxINT64_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137344, mxINT64_CLASS_137345))
# Adding element type (key, value) (line 140)
str_137346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'str', 'i4')
# Getting the type of 'mxINT32_CLASS' (line 148)
mxINT32_CLASS_137347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 10), 'mxINT32_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137346, mxINT32_CLASS_137347))
# Adding element type (key, value) (line 140)
str_137348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 4), 'str', 'i2')
# Getting the type of 'mxINT16_CLASS' (line 149)
mxINT16_CLASS_137349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 10), 'mxINT16_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137348, mxINT16_CLASS_137349))
# Adding element type (key, value) (line 140)
str_137350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 4), 'str', 'i1')
# Getting the type of 'mxINT8_CLASS' (line 150)
mxINT8_CLASS_137351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 10), 'mxINT8_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137350, mxINT8_CLASS_137351))
# Adding element type (key, value) (line 140)
str_137352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'str', 'u8')
# Getting the type of 'mxUINT64_CLASS' (line 151)
mxUINT64_CLASS_137353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 10), 'mxUINT64_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137352, mxUINT64_CLASS_137353))
# Adding element type (key, value) (line 140)
str_137354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 4), 'str', 'u4')
# Getting the type of 'mxUINT32_CLASS' (line 152)
mxUINT32_CLASS_137355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 10), 'mxUINT32_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137354, mxUINT32_CLASS_137355))
# Adding element type (key, value) (line 140)
str_137356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'str', 'u2')
# Getting the type of 'mxUINT16_CLASS' (line 153)
mxUINT16_CLASS_137357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 10), 'mxUINT16_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137356, mxUINT16_CLASS_137357))
# Adding element type (key, value) (line 140)
str_137358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'str', 'u1')
# Getting the type of 'mxUINT8_CLASS' (line 154)
mxUINT8_CLASS_137359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 10), 'mxUINT8_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137358, mxUINT8_CLASS_137359))
# Adding element type (key, value) (line 140)
str_137360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'str', 'S1')
# Getting the type of 'mxUINT8_CLASS' (line 155)
mxUINT8_CLASS_137361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 10), 'mxUINT8_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137360, mxUINT8_CLASS_137361))
# Adding element type (key, value) (line 140)
str_137362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 4), 'str', 'b1')
# Getting the type of 'mxUINT8_CLASS' (line 156)
mxUINT8_CLASS_137363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 10), 'mxUINT8_CLASS')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 16), dict_137331, (str_137362, mxUINT8_CLASS_137363))

# Assigning a type to the variable 'NP_TO_MXTYPES' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'NP_TO_MXTYPES', dict_137331)
str_137364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', ' Before release v7.1 (release 14) matlab (TM) used the system\ndefault character encoding scheme padded out to 16-bits. Release 14\nand later use Unicode. When saving character data, R14 checks if it\ncan be encoded in 7-bit ascii, and saves in that format if so.')

# Assigning a Dict to a Name (line 164):

# Obtaining an instance of the builtin type 'dict' (line 164)
dict_137365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 164)
# Adding element type (key, value) (line 164)
# Getting the type of 'miUTF8' (line 165)
miUTF8_137366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'miUTF8')

# Obtaining an instance of the builtin type 'dict' (line 165)
dict_137367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 165)
# Adding element type (key, value) (line 165)
str_137368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 13), 'str', 'codec')
str_137369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 22), 'str', 'utf_8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 12), dict_137367, (str_137368, str_137369))
# Adding element type (key, value) (line 165)
str_137370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 31), 'str', 'width')
int_137371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 40), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 12), dict_137367, (str_137370, int_137371))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 18), dict_137365, (miUTF8_137366, dict_137367))
# Adding element type (key, value) (line 164)
# Getting the type of 'miUTF16' (line 166)
miUTF16_137372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'miUTF16')

# Obtaining an instance of the builtin type 'dict' (line 166)
dict_137373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 166)
# Adding element type (key, value) (line 166)
str_137374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 14), 'str', 'codec')
str_137375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 23), 'str', 'utf_16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), dict_137373, (str_137374, str_137375))
# Adding element type (key, value) (line 166)
str_137376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 33), 'str', 'width')
int_137377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), dict_137373, (str_137376, int_137377))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 18), dict_137365, (miUTF16_137372, dict_137373))
# Adding element type (key, value) (line 164)
# Getting the type of 'miUTF32' (line 167)
miUTF32_137378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'miUTF32')

# Obtaining an instance of the builtin type 'dict' (line 167)
dict_137379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 167)
# Adding element type (key, value) (line 167)
str_137380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 14), 'str', 'codec')
str_137381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'str', 'utf_32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 13), dict_137379, (str_137380, str_137381))
# Adding element type (key, value) (line 167)
str_137382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 32), 'str', 'width')
int_137383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 41), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 13), dict_137379, (str_137382, int_137383))

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 18), dict_137365, (miUTF32_137378, dict_137379))

# Assigning a type to the variable 'codecs_template' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'codecs_template', dict_137365)

@norecursion
def _convert_codecs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_convert_codecs'
    module_type_store = module_type_store.open_function_context('_convert_codecs', 171, 0, False)
    
    # Passed parameters checking function
    _convert_codecs.stypy_localization = localization
    _convert_codecs.stypy_type_of_self = None
    _convert_codecs.stypy_type_store = module_type_store
    _convert_codecs.stypy_function_name = '_convert_codecs'
    _convert_codecs.stypy_param_names_list = ['template', 'byte_order']
    _convert_codecs.stypy_varargs_param_name = None
    _convert_codecs.stypy_kwargs_param_name = None
    _convert_codecs.stypy_call_defaults = defaults
    _convert_codecs.stypy_call_varargs = varargs
    _convert_codecs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_convert_codecs', ['template', 'byte_order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_convert_codecs', localization, ['template', 'byte_order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_convert_codecs(...)' code ##################

    str_137384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, (-1)), 'str', " Convert codec template mapping to byte order\n\n    Set codecs not on this system to None\n\n    Parameters\n    ----------\n    template : mapping\n       key, value are respectively codec name, and root name for codec\n       (without byte order suffix)\n    byte_order : {'<', '>'}\n       code for little or big endian\n\n    Returns\n    -------\n    codecs : dict\n       key, value are name, codec (as in .encode(codec))\n    ")
    
    # Assigning a Dict to a Name (line 189):
    
    # Obtaining an instance of the builtin type 'dict' (line 189)
    dict_137385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 189)
    
    # Assigning a type to the variable 'codecs' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'codecs', dict_137385)
    
    # Assigning a BoolOp to a Name (line 190):
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'byte_order' (line 190)
    byte_order_137386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 14), 'byte_order')
    str_137387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 28), 'str', '<')
    # Applying the binary operator '==' (line 190)
    result_eq_137388 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 14), '==', byte_order_137386, str_137387)
    
    str_137389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 36), 'str', '_le')
    # Applying the binary operator 'and' (line 190)
    result_and_keyword_137390 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 14), 'and', result_eq_137388, str_137389)
    
    str_137391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 45), 'str', '_be')
    # Applying the binary operator 'or' (line 190)
    result_or_keyword_137392 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 14), 'or', result_and_keyword_137390, str_137391)
    
    # Assigning a type to the variable 'postfix' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'postfix', result_or_keyword_137392)
    
    
    # Call to items(...): (line 191)
    # Processing the call keyword arguments (line 191)
    kwargs_137395 = {}
    # Getting the type of 'template' (line 191)
    template_137393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'template', False)
    # Obtaining the member 'items' of a type (line 191)
    items_137394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 16), template_137393, 'items')
    # Calling items(args, kwargs) (line 191)
    items_call_result_137396 = invoke(stypy.reporting.localization.Localization(__file__, 191, 16), items_137394, *[], **kwargs_137395)
    
    # Testing the type of a for loop iterable (line 191)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 191, 4), items_call_result_137396)
    # Getting the type of the for loop variable (line 191)
    for_loop_var_137397 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 191, 4), items_call_result_137396)
    # Assigning a type to the variable 'k' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 4), for_loop_var_137397))
    # Assigning a type to the variable 'v' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 4), for_loop_var_137397))
    # SSA begins for a for statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 192):
    
    # Obtaining the type of the subscript
    str_137398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 18), 'str', 'codec')
    # Getting the type of 'v' (line 192)
    v_137399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'v')
    # Obtaining the member '__getitem__' of a type (line 192)
    getitem___137400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 16), v_137399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 192)
    subscript_call_result_137401 = invoke(stypy.reporting.localization.Localization(__file__, 192, 16), getitem___137400, str_137398)
    
    # Assigning a type to the variable 'codec' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'codec', subscript_call_result_137401)
    
    
    # SSA begins for try-except statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to encode(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'codec' (line 194)
    codec_137404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 23), 'codec', False)
    # Processing the call keyword arguments (line 194)
    kwargs_137405 = {}
    str_137402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 12), 'str', ' ')
    # Obtaining the member 'encode' of a type (line 194)
    encode_137403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), str_137402, 'encode')
    # Calling encode(args, kwargs) (line 194)
    encode_call_result_137406 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), encode_137403, *[codec_137404], **kwargs_137405)
    
    # SSA branch for the except part of a try statement (line 193)
    # SSA branch for the except 'LookupError' branch of a try statement (line 193)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Subscript (line 196):
    # Getting the type of 'None' (line 196)
    None_137407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'None')
    # Getting the type of 'codecs' (line 196)
    codecs_137408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'codecs')
    # Getting the type of 'k' (line 196)
    k_137409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'k')
    # Storing an element on a container (line 196)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 12), codecs_137408, (k_137409, None_137407))
    # SSA join for try-except statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    str_137410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 13), 'str', 'width')
    # Getting the type of 'v' (line 198)
    v_137411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'v')
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___137412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), v_137411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_137413 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), getitem___137412, str_137410)
    
    int_137414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 24), 'int')
    # Applying the binary operator '>' (line 198)
    result_gt_137415 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '>', subscript_call_result_137413, int_137414)
    
    # Testing the type of an if condition (line 198)
    if_condition_137416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_gt_137415)
    # Assigning a type to the variable 'if_condition_137416' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_137416', if_condition_137416)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'codec' (line 199)
    codec_137417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'codec')
    # Getting the type of 'postfix' (line 199)
    postfix_137418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'postfix')
    # Applying the binary operator '+=' (line 199)
    result_iadd_137419 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 12), '+=', codec_137417, postfix_137418)
    # Assigning a type to the variable 'codec' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'codec', result_iadd_137419)
    
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 200):
    # Getting the type of 'codec' (line 200)
    codec_137420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'codec')
    # Getting the type of 'codecs' (line 200)
    codecs_137421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'codecs')
    # Getting the type of 'k' (line 200)
    k_137422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'k')
    # Storing an element on a container (line 200)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 8), codecs_137421, (k_137422, codec_137420))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to copy(...): (line 201)
    # Processing the call keyword arguments (line 201)
    kwargs_137425 = {}
    # Getting the type of 'codecs' (line 201)
    codecs_137423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'codecs', False)
    # Obtaining the member 'copy' of a type (line 201)
    copy_137424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), codecs_137423, 'copy')
    # Calling copy(args, kwargs) (line 201)
    copy_call_result_137426 = invoke(stypy.reporting.localization.Localization(__file__, 201, 11), copy_137424, *[], **kwargs_137425)
    
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type', copy_call_result_137426)
    
    # ################# End of '_convert_codecs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_convert_codecs' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_137427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_convert_codecs'
    return stypy_return_type_137427

# Assigning a type to the variable '_convert_codecs' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), '_convert_codecs', _convert_codecs)

# Assigning a Dict to a Name (line 204):

# Obtaining an instance of the builtin type 'dict' (line 204)
dict_137428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 204)

# Assigning a type to the variable 'MDTYPES' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'MDTYPES', dict_137428)

str_137429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 17), 'str', '<>')
# Testing the type of a for loop iterable (line 205)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 0), str_137429)
# Getting the type of the for loop variable (line 205)
for_loop_var_137430 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 0), str_137429)
# Assigning a type to the variable '_bytecode' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), '_bytecode', for_loop_var_137430)
# SSA begins for a for statement (line 205)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Dict to a Name (line 206):

# Obtaining an instance of the builtin type 'dict' (line 206)
dict_137431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 206)
# Adding element type (key, value) (line 206)
str_137432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 12), 'str', 'dtypes')

# Call to convert_dtypes(...): (line 206)
# Processing the call arguments (line 206)
# Getting the type of 'mdtypes_template' (line 206)
mdtypes_template_137434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 37), 'mdtypes_template', False)
# Getting the type of '_bytecode' (line 206)
_bytecode_137435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 55), '_bytecode', False)
# Processing the call keyword arguments (line 206)
kwargs_137436 = {}
# Getting the type of 'convert_dtypes' (line 206)
convert_dtypes_137433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'convert_dtypes', False)
# Calling convert_dtypes(args, kwargs) (line 206)
convert_dtypes_call_result_137437 = invoke(stypy.reporting.localization.Localization(__file__, 206, 22), convert_dtypes_137433, *[mdtypes_template_137434, _bytecode_137435], **kwargs_137436)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 11), dict_137431, (str_137432, convert_dtypes_call_result_137437))
# Adding element type (key, value) (line 206)
str_137438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 12), 'str', 'classes')

# Call to convert_dtypes(...): (line 207)
# Processing the call arguments (line 207)
# Getting the type of 'mclass_dtypes_template' (line 207)
mclass_dtypes_template_137440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'mclass_dtypes_template', False)
# Getting the type of '_bytecode' (line 207)
_bytecode_137441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 62), '_bytecode', False)
# Processing the call keyword arguments (line 207)
kwargs_137442 = {}
# Getting the type of 'convert_dtypes' (line 207)
convert_dtypes_137439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'convert_dtypes', False)
# Calling convert_dtypes(args, kwargs) (line 207)
convert_dtypes_call_result_137443 = invoke(stypy.reporting.localization.Localization(__file__, 207, 23), convert_dtypes_137439, *[mclass_dtypes_template_137440, _bytecode_137441], **kwargs_137442)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 11), dict_137431, (str_137438, convert_dtypes_call_result_137443))
# Adding element type (key, value) (line 206)
str_137444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 12), 'str', 'codecs')

# Call to _convert_codecs(...): (line 208)
# Processing the call arguments (line 208)
# Getting the type of 'codecs_template' (line 208)
codecs_template_137446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 38), 'codecs_template', False)
# Getting the type of '_bytecode' (line 208)
_bytecode_137447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 55), '_bytecode', False)
# Processing the call keyword arguments (line 208)
kwargs_137448 = {}
# Getting the type of '_convert_codecs' (line 208)
_convert_codecs_137445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), '_convert_codecs', False)
# Calling _convert_codecs(args, kwargs) (line 208)
_convert_codecs_call_result_137449 = invoke(stypy.reporting.localization.Localization(__file__, 208, 22), _convert_codecs_137445, *[codecs_template_137446, _bytecode_137447], **kwargs_137448)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 11), dict_137431, (str_137444, _convert_codecs_call_result_137449))

# Assigning a type to the variable '_def' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), '_def', dict_137431)

# Assigning a Name to a Subscript (line 209):
# Getting the type of '_def' (line 209)
_def_137450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), '_def')
# Getting the type of 'MDTYPES' (line 209)
MDTYPES_137451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'MDTYPES')
# Getting the type of '_bytecode' (line 209)
_bytecode_137452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), '_bytecode')
# Storing an element on a container (line 209)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 4), MDTYPES_137451, (_bytecode_137452, _def_137450))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'mat_struct' class

class mat_struct(object, ):
    str_137453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, (-1)), 'str', ' Placeholder for holding read data from structs\n\n    We use instances of this class when the user passes False as a value to the\n    ``struct_as_record`` parameter of the :func:`scipy.io.matlab.loadmat`\n    function.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 212, 0, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'mat_struct.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'mat_struct' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'mat_struct', mat_struct)
# Declaration of the 'MatlabObject' class
# Getting the type of 'np' (line 222)
np_137454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'np')
# Obtaining the member 'ndarray' of a type (line 222)
ndarray_137455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), np_137454, 'ndarray')

class MatlabObject(ndarray_137455, ):
    str_137456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 4), 'str', ' ndarray Subclass to contain matlab object ')

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 224)
        None_137457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'None')
        defaults = [None_137457]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatlabObject.__new__.__dict__.__setitem__('stypy_localization', localization)
        MatlabObject.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatlabObject.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatlabObject.__new__.__dict__.__setitem__('stypy_function_name', 'MatlabObject.__new__')
        MatlabObject.__new__.__dict__.__setitem__('stypy_param_names_list', ['input_array', 'classname'])
        MatlabObject.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatlabObject.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatlabObject.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatlabObject.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatlabObject.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatlabObject.__new__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatlabObject.__new__', ['input_array', 'classname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['input_array', 'classname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Assigning a Call to a Name (line 227):
        
        # Call to view(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'cls' (line 227)
        cls_137464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 43), 'cls', False)
        # Processing the call keyword arguments (line 227)
        kwargs_137465 = {}
        
        # Call to asarray(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'input_array' (line 227)
        input_array_137460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'input_array', False)
        # Processing the call keyword arguments (line 227)
        kwargs_137461 = {}
        # Getting the type of 'np' (line 227)
        np_137458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 227)
        asarray_137459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 14), np_137458, 'asarray')
        # Calling asarray(args, kwargs) (line 227)
        asarray_call_result_137462 = invoke(stypy.reporting.localization.Localization(__file__, 227, 14), asarray_137459, *[input_array_137460], **kwargs_137461)
        
        # Obtaining the member 'view' of a type (line 227)
        view_137463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 14), asarray_call_result_137462, 'view')
        # Calling view(args, kwargs) (line 227)
        view_call_result_137466 = invoke(stypy.reporting.localization.Localization(__file__, 227, 14), view_137463, *[cls_137464], **kwargs_137465)
        
        # Assigning a type to the variable 'obj' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'obj', view_call_result_137466)
        
        # Assigning a Name to a Attribute (line 229):
        # Getting the type of 'classname' (line 229)
        classname_137467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 24), 'classname')
        # Getting the type of 'obj' (line 229)
        obj_137468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'obj')
        # Setting the type of the member 'classname' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), obj_137468, 'classname', classname_137467)
        # Getting the type of 'obj' (line 231)
        obj_137469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', obj_137469)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 224)
        stypy_return_type_137470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_137470


    @norecursion
    def __array_finalize__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_finalize__'
        module_type_store = module_type_store.open_function_context('__array_finalize__', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_localization', localization)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_function_name', 'MatlabObject.__array_finalize__')
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatlabObject.__array_finalize__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatlabObject.__array_finalize__', ['obj'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 235):
        
        # Call to getattr(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'obj' (line 235)
        obj_137472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'obj', False)
        str_137473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 38), 'str', 'classname')
        # Getting the type of 'None' (line 235)
        None_137474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 51), 'None', False)
        # Processing the call keyword arguments (line 235)
        kwargs_137475 = {}
        # Getting the type of 'getattr' (line 235)
        getattr_137471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'getattr', False)
        # Calling getattr(args, kwargs) (line 235)
        getattr_call_result_137476 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), getattr_137471, *[obj_137472, str_137473, None_137474], **kwargs_137475)
        
        # Getting the type of 'self' (line 235)
        self_137477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'classname' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_137477, 'classname', getattr_call_result_137476)
        
        # ################# End of '__array_finalize__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_finalize__' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_137478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_finalize__'
        return stypy_return_type_137478


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 222, 0, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatlabObject.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatlabObject' (line 222)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'MatlabObject', MatlabObject)
# Declaration of the 'MatlabFunction' class
# Getting the type of 'np' (line 239)
np_137479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'np')
# Obtaining the member 'ndarray' of a type (line 239)
ndarray_137480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 21), np_137479, 'ndarray')

class MatlabFunction(ndarray_137480, ):
    str_137481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'str', ' Subclass to signal this is a matlab function ')

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatlabFunction.__new__.__dict__.__setitem__('stypy_localization', localization)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_function_name', 'MatlabFunction.__new__')
        MatlabFunction.__new__.__dict__.__setitem__('stypy_param_names_list', ['input_array'])
        MatlabFunction.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatlabFunction.__new__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatlabFunction.__new__', ['input_array'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['input_array'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Assigning a Call to a Name (line 242):
        
        # Call to view(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'cls' (line 242)
        cls_137488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 43), 'cls', False)
        # Processing the call keyword arguments (line 242)
        kwargs_137489 = {}
        
        # Call to asarray(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'input_array' (line 242)
        input_array_137484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 25), 'input_array', False)
        # Processing the call keyword arguments (line 242)
        kwargs_137485 = {}
        # Getting the type of 'np' (line 242)
        np_137482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 242)
        asarray_137483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 14), np_137482, 'asarray')
        # Calling asarray(args, kwargs) (line 242)
        asarray_call_result_137486 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), asarray_137483, *[input_array_137484], **kwargs_137485)
        
        # Obtaining the member 'view' of a type (line 242)
        view_137487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 14), asarray_call_result_137486, 'view')
        # Calling view(args, kwargs) (line 242)
        view_call_result_137490 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), view_137487, *[cls_137488], **kwargs_137489)
        
        # Assigning a type to the variable 'obj' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'obj', view_call_result_137490)
        # Getting the type of 'obj' (line 243)
        obj_137491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'stypy_return_type', obj_137491)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_137492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137492)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_137492


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 239, 0, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatlabFunction.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatlabFunction' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'MatlabFunction', MatlabFunction)
# Declaration of the 'MatlabOpaque' class
# Getting the type of 'np' (line 246)
np_137493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'np')
# Obtaining the member 'ndarray' of a type (line 246)
ndarray_137494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), np_137493, 'ndarray')

class MatlabOpaque(ndarray_137494, ):
    str_137495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 4), 'str', ' Subclass to signal this is a matlab opaque matrix ')

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_localization', localization)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_function_name', 'MatlabOpaque.__new__')
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_param_names_list', ['input_array'])
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MatlabOpaque.__new__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatlabOpaque.__new__', ['input_array'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['input_array'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Assigning a Call to a Name (line 249):
        
        # Call to view(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'cls' (line 249)
        cls_137502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 43), 'cls', False)
        # Processing the call keyword arguments (line 249)
        kwargs_137503 = {}
        
        # Call to asarray(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'input_array' (line 249)
        input_array_137498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'input_array', False)
        # Processing the call keyword arguments (line 249)
        kwargs_137499 = {}
        # Getting the type of 'np' (line 249)
        np_137496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 249)
        asarray_137497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 14), np_137496, 'asarray')
        # Calling asarray(args, kwargs) (line 249)
        asarray_call_result_137500 = invoke(stypy.reporting.localization.Localization(__file__, 249, 14), asarray_137497, *[input_array_137498], **kwargs_137499)
        
        # Obtaining the member 'view' of a type (line 249)
        view_137501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 14), asarray_call_result_137500, 'view')
        # Calling view(args, kwargs) (line 249)
        view_call_result_137504 = invoke(stypy.reporting.localization.Localization(__file__, 249, 14), view_137501, *[cls_137502], **kwargs_137503)
        
        # Assigning a type to the variable 'obj' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'obj', view_call_result_137504)
        # Getting the type of 'obj' (line 250)
        obj_137505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type', obj_137505)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_137506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_137506)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_137506


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 246, 0, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MatlabOpaque.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MatlabOpaque' (line 246)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 0), 'MatlabOpaque', MatlabOpaque)

# Assigning a Call to a Name (line 253):

# Call to dtype(...): (line 253)
# Processing the call arguments (line 253)

# Obtaining an instance of the builtin type 'list' (line 254)
list_137509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 254)
# Adding element type (line 254)

# Obtaining an instance of the builtin type 'tuple' (line 254)
tuple_137510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 254)
# Adding element type (line 254)
str_137511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 6), 'str', 's0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 6), tuple_137510, str_137511)
# Adding element type (line 254)
str_137512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 12), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 6), tuple_137510, str_137512)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 4), list_137509, tuple_137510)
# Adding element type (line 254)

# Obtaining an instance of the builtin type 'tuple' (line 254)
tuple_137513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 254)
# Adding element type (line 254)
str_137514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 19), 'str', 's1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 19), tuple_137513, str_137514)
# Adding element type (line 254)
str_137515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 19), tuple_137513, str_137515)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 4), list_137509, tuple_137513)
# Adding element type (line 254)

# Obtaining an instance of the builtin type 'tuple' (line 254)
tuple_137516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 254)
# Adding element type (line 254)
str_137517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 32), 'str', 's2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 32), tuple_137516, str_137517)
# Adding element type (line 254)
str_137518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 32), tuple_137516, str_137518)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 4), list_137509, tuple_137516)
# Adding element type (line 254)

# Obtaining an instance of the builtin type 'tuple' (line 254)
tuple_137519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 45), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 254)
# Adding element type (line 254)
str_137520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 45), 'str', 'arr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 45), tuple_137519, str_137520)
# Adding element type (line 254)
str_137521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 52), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 45), tuple_137519, str_137521)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 4), list_137509, tuple_137519)

# Processing the call keyword arguments (line 253)
kwargs_137522 = {}
# Getting the type of 'np' (line 253)
np_137507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'np', False)
# Obtaining the member 'dtype' of a type (line 253)
dtype_137508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 15), np_137507, 'dtype')
# Calling dtype(args, kwargs) (line 253)
dtype_call_result_137523 = invoke(stypy.reporting.localization.Localization(__file__, 253, 15), dtype_137508, *[list_137509], **kwargs_137522)

# Assigning a type to the variable 'OPAQUE_DTYPE' (line 253)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'OPAQUE_DTYPE', dtype_call_result_137523)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
