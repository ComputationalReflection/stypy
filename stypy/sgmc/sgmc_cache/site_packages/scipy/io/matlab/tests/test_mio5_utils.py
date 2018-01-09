
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Testing mio5_utils Cython module
2: 
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import sys
7: 
8: from io import BytesIO
9: cStringIO = BytesIO
10: 
11: import numpy as np
12: 
13: from numpy.testing import assert_array_equal, assert_equal, assert_
14: from pytest import raises as assert_raises
15: 
16: from scipy._lib.six import u
17: 
18: import scipy.io.matlab.byteordercodes as boc
19: import scipy.io.matlab.streams as streams
20: import scipy.io.matlab.mio5_params as mio5p
21: import scipy.io.matlab.mio5_utils as m5u
22: 
23: 
24: def test_byteswap():
25:     for val in (
26:         1,
27:         0x100,
28:         0x10000):
29:         a = np.array(val, dtype=np.uint32)
30:         b = a.byteswap()
31:         c = m5u.byteswap_u4(a)
32:         assert_equal(b.item(), c)
33:         d = m5u.byteswap_u4(c)
34:         assert_equal(a.item(), d)
35: 
36: 
37: def _make_tag(base_dt, val, mdtype, sde=False):
38:     ''' Makes a simple matlab tag, full or sde '''
39:     base_dt = np.dtype(base_dt)
40:     bo = boc.to_numpy_code(base_dt.byteorder)
41:     byte_count = base_dt.itemsize
42:     if not sde:
43:         udt = bo + 'u4'
44:         padding = 8 - (byte_count % 8)
45:         all_dt = [('mdtype', udt),
46:                   ('byte_count', udt),
47:                   ('val', base_dt)]
48:         if padding:
49:             all_dt.append(('padding', 'u1', padding))
50:     else:  # is sde
51:         udt = bo + 'u2'
52:         padding = 4-byte_count
53:         if bo == '<':  # little endian
54:             all_dt = [('mdtype', udt),
55:                       ('byte_count', udt),
56:                       ('val', base_dt)]
57:         else:  # big endian
58:             all_dt = [('byte_count', udt),
59:                       ('mdtype', udt),
60:                       ('val', base_dt)]
61:         if padding:
62:             all_dt.append(('padding', 'u1', padding))
63:     tag = np.zeros((1,), dtype=all_dt)
64:     tag['mdtype'] = mdtype
65:     tag['byte_count'] = byte_count
66:     tag['val'] = val
67:     return tag
68: 
69: 
70: def _write_stream(stream, *strings):
71:     stream.truncate(0)
72:     stream.seek(0)
73:     for s in strings:
74:         stream.write(s)
75:     stream.seek(0)
76: 
77: 
78: def _make_readerlike(stream, byte_order=boc.native_code):
79:     class R(object):
80:         pass
81:     r = R()
82:     r.mat_stream = stream
83:     r.byte_order = byte_order
84:     r.struct_as_record = True
85:     r.uint16_codec = sys.getdefaultencoding()
86:     r.chars_as_strings = False
87:     r.mat_dtype = False
88:     r.squeeze_me = False
89:     return r
90: 
91: 
92: def test_read_tag():
93:     # mainly to test errors
94:     # make reader-like thing
95:     str_io = BytesIO()
96:     r = _make_readerlike(str_io)
97:     c_reader = m5u.VarReader5(r)
98:     # This works for StringIO but _not_ cStringIO
99:     assert_raises(IOError, c_reader.read_tag)
100:     # bad SDE
101:     tag = _make_tag('i4', 1, mio5p.miINT32, sde=True)
102:     tag['byte_count'] = 5
103:     _write_stream(str_io, tag.tostring())
104:     assert_raises(ValueError, c_reader.read_tag)
105: 
106: 
107: def test_read_stream():
108:     tag = _make_tag('i4', 1, mio5p.miINT32, sde=True)
109:     tag_str = tag.tostring()
110:     str_io = cStringIO(tag_str)
111:     st = streams.make_stream(str_io)
112:     s = streams._read_into(st, tag.itemsize)
113:     assert_equal(s, tag.tostring())
114: 
115: 
116: def test_read_numeric():
117:     # make reader-like thing
118:     str_io = cStringIO()
119:     r = _make_readerlike(str_io)
120:     # check simplest of tags
121:     for base_dt, val, mdtype in (('u2', 30, mio5p.miUINT16),
122:                                  ('i4', 1, mio5p.miINT32),
123:                                  ('i2', -1, mio5p.miINT16)):
124:         for byte_code in ('<', '>'):
125:             r.byte_order = byte_code
126:             c_reader = m5u.VarReader5(r)
127:             assert_equal(c_reader.little_endian, byte_code == '<')
128:             assert_equal(c_reader.is_swapped, byte_code != boc.native_code)
129:             for sde_f in (False, True):
130:                 dt = np.dtype(base_dt).newbyteorder(byte_code)
131:                 a = _make_tag(dt, val, mdtype, sde_f)
132:                 a_str = a.tostring()
133:                 _write_stream(str_io, a_str)
134:                 el = c_reader.read_numeric()
135:                 assert_equal(el, val)
136:                 # two sequential reads
137:                 _write_stream(str_io, a_str, a_str)
138:                 el = c_reader.read_numeric()
139:                 assert_equal(el, val)
140:                 el = c_reader.read_numeric()
141:                 assert_equal(el, val)
142: 
143: 
144: def test_read_numeric_writeable():
145:     # make reader-like thing
146:     str_io = cStringIO()
147:     r = _make_readerlike(str_io, '<')
148:     c_reader = m5u.VarReader5(r)
149:     dt = np.dtype('<u2')
150:     a = _make_tag(dt, 30, mio5p.miUINT16, 0)
151:     a_str = a.tostring()
152:     _write_stream(str_io, a_str)
153:     el = c_reader.read_numeric()
154:     assert_(el.flags.writeable is True)
155: 
156: 
157: def test_zero_byte_string():
158:     # Tests hack to allow chars of non-zero length, but 0 bytes
159:     # make reader-like thing
160:     str_io = cStringIO()
161:     r = _make_readerlike(str_io, boc.native_code)
162:     c_reader = m5u.VarReader5(r)
163:     tag_dt = np.dtype([('mdtype', 'u4'), ('byte_count', 'u4')])
164:     tag = np.zeros((1,), dtype=tag_dt)
165:     tag['mdtype'] = mio5p.miINT8
166:     tag['byte_count'] = 1
167:     hdr = m5u.VarHeader5()
168:     # Try when string is 1 length
169:     hdr.set_dims([1,])
170:     _write_stream(str_io, tag.tostring() + b'        ')
171:     str_io.seek(0)
172:     val = c_reader.read_char(hdr)
173:     assert_equal(val, u(' '))
174:     # Now when string has 0 bytes 1 length
175:     tag['byte_count'] = 0
176:     _write_stream(str_io, tag.tostring())
177:     str_io.seek(0)
178:     val = c_reader.read_char(hdr)
179:     assert_equal(val, u(' '))
180:     # Now when string has 0 bytes 4 length
181:     str_io.seek(0)
182:     hdr.set_dims([4,])
183:     val = c_reader.read_char(hdr)
184:     assert_array_equal(val, [u(' ')] * 4)
185: 
186: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_143392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' Testing mio5_utils Cython module\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from io import BytesIO' statement (line 8)
try:
    from io import BytesIO

except:
    BytesIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'io', None, module_type_store, ['BytesIO'], [BytesIO])


# Assigning a Name to a Name (line 9):
# Getting the type of 'BytesIO' (line 9)
BytesIO_143393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'BytesIO')
# Assigning a type to the variable 'cStringIO' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'cStringIO', BytesIO_143393)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143394 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_143394) is not StypyTypeError):

    if (import_143394 != 'pyd_module'):
        __import__(import_143394)
        sys_modules_143395 = sys.modules[import_143394]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_143395.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_143394)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.testing import assert_array_equal, assert_equal, assert_' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143396 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing')

if (type(import_143396) is not StypyTypeError):

    if (import_143396 != 'pyd_module'):
        __import__(import_143396)
        sys_modules_143397 = sys.modules[import_143396]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', sys_modules_143397.module_type_store, module_type_store, ['assert_array_equal', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_143397, sys_modules_143397.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal', 'assert_equal', 'assert_'], [assert_array_equal, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', import_143396)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from pytest import assert_raises' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143398 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'pytest')

if (type(import_143398) is not StypyTypeError):

    if (import_143398 != 'pyd_module'):
        __import__(import_143398)
        sys_modules_143399 = sys.modules[import_143398]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'pytest', sys_modules_143399.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_143399, sys_modules_143399.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'pytest', import_143398)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy._lib.six import u' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143400 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six')

if (type(import_143400) is not StypyTypeError):

    if (import_143400 != 'pyd_module'):
        __import__(import_143400)
        sys_modules_143401 = sys.modules[import_143400]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', sys_modules_143401.module_type_store, module_type_store, ['u'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_143401, sys_modules_143401.module_type_store, module_type_store)
    else:
        from scipy._lib.six import u

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', None, module_type_store, ['u'], [u])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy._lib.six', import_143400)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import scipy.io.matlab.byteordercodes' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143402 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.matlab.byteordercodes')

if (type(import_143402) is not StypyTypeError):

    if (import_143402 != 'pyd_module'):
        __import__(import_143402)
        sys_modules_143403 = sys.modules[import_143402]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'boc', sys_modules_143403.module_type_store, module_type_store)
    else:
        import scipy.io.matlab.byteordercodes as boc

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'boc', scipy.io.matlab.byteordercodes, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io.matlab.byteordercodes' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io.matlab.byteordercodes', import_143402)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import scipy.io.matlab.streams' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143404 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.matlab.streams')

if (type(import_143404) is not StypyTypeError):

    if (import_143404 != 'pyd_module'):
        __import__(import_143404)
        sys_modules_143405 = sys.modules[import_143404]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'streams', sys_modules_143405.module_type_store, module_type_store)
    else:
        import scipy.io.matlab.streams as streams

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'streams', scipy.io.matlab.streams, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io.matlab.streams' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.matlab.streams', import_143404)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import scipy.io.matlab.mio5_params' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143406 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.matlab.mio5_params')

if (type(import_143406) is not StypyTypeError):

    if (import_143406 != 'pyd_module'):
        __import__(import_143406)
        sys_modules_143407 = sys.modules[import_143406]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'mio5p', sys_modules_143407.module_type_store, module_type_store)
    else:
        import scipy.io.matlab.mio5_params as mio5p

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'mio5p', scipy.io.matlab.mio5_params, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio5_params' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.matlab.mio5_params', import_143406)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import scipy.io.matlab.mio5_utils' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_143408 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.io.matlab.mio5_utils')

if (type(import_143408) is not StypyTypeError):

    if (import_143408 != 'pyd_module'):
        __import__(import_143408)
        sys_modules_143409 = sys.modules[import_143408]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'm5u', sys_modules_143409.module_type_store, module_type_store)
    else:
        import scipy.io.matlab.mio5_utils as m5u

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'm5u', scipy.io.matlab.mio5_utils, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio5_utils' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.io.matlab.mio5_utils', import_143408)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')


@norecursion
def test_byteswap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_byteswap'
    module_type_store = module_type_store.open_function_context('test_byteswap', 24, 0, False)
    
    # Passed parameters checking function
    test_byteswap.stypy_localization = localization
    test_byteswap.stypy_type_of_self = None
    test_byteswap.stypy_type_store = module_type_store
    test_byteswap.stypy_function_name = 'test_byteswap'
    test_byteswap.stypy_param_names_list = []
    test_byteswap.stypy_varargs_param_name = None
    test_byteswap.stypy_kwargs_param_name = None
    test_byteswap.stypy_call_defaults = defaults
    test_byteswap.stypy_call_varargs = varargs
    test_byteswap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_byteswap', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_byteswap', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_byteswap(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_143410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    int_143411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 8), tuple_143410, int_143411)
    # Adding element type (line 26)
    int_143412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 8), tuple_143410, int_143412)
    # Adding element type (line 26)
    int_143413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 8), tuple_143410, int_143413)
    
    # Testing the type of a for loop iterable (line 25)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 4), tuple_143410)
    # Getting the type of the for loop variable (line 25)
    for_loop_var_143414 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 4), tuple_143410)
    # Assigning a type to the variable 'val' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'val', for_loop_var_143414)
    # SSA begins for a for statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 29):
    
    # Call to array(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'val' (line 29)
    val_143417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'val', False)
    # Processing the call keyword arguments (line 29)
    # Getting the type of 'np' (line 29)
    np_143418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 32), 'np', False)
    # Obtaining the member 'uint32' of a type (line 29)
    uint32_143419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 32), np_143418, 'uint32')
    keyword_143420 = uint32_143419
    kwargs_143421 = {'dtype': keyword_143420}
    # Getting the type of 'np' (line 29)
    np_143415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 29)
    array_143416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), np_143415, 'array')
    # Calling array(args, kwargs) (line 29)
    array_call_result_143422 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), array_143416, *[val_143417], **kwargs_143421)
    
    # Assigning a type to the variable 'a' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'a', array_call_result_143422)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to byteswap(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_143425 = {}
    # Getting the type of 'a' (line 30)
    a_143423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'a', False)
    # Obtaining the member 'byteswap' of a type (line 30)
    byteswap_143424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), a_143423, 'byteswap')
    # Calling byteswap(args, kwargs) (line 30)
    byteswap_call_result_143426 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), byteswap_143424, *[], **kwargs_143425)
    
    # Assigning a type to the variable 'b' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'b', byteswap_call_result_143426)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to byteswap_u4(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'a' (line 31)
    a_143429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'a', False)
    # Processing the call keyword arguments (line 31)
    kwargs_143430 = {}
    # Getting the type of 'm5u' (line 31)
    m5u_143427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'm5u', False)
    # Obtaining the member 'byteswap_u4' of a type (line 31)
    byteswap_u4_143428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), m5u_143427, 'byteswap_u4')
    # Calling byteswap_u4(args, kwargs) (line 31)
    byteswap_u4_call_result_143431 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), byteswap_u4_143428, *[a_143429], **kwargs_143430)
    
    # Assigning a type to the variable 'c' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'c', byteswap_u4_call_result_143431)
    
    # Call to assert_equal(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to item(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_143435 = {}
    # Getting the type of 'b' (line 32)
    b_143433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'b', False)
    # Obtaining the member 'item' of a type (line 32)
    item_143434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 21), b_143433, 'item')
    # Calling item(args, kwargs) (line 32)
    item_call_result_143436 = invoke(stypy.reporting.localization.Localization(__file__, 32, 21), item_143434, *[], **kwargs_143435)
    
    # Getting the type of 'c' (line 32)
    c_143437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'c', False)
    # Processing the call keyword arguments (line 32)
    kwargs_143438 = {}
    # Getting the type of 'assert_equal' (line 32)
    assert_equal_143432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 32)
    assert_equal_call_result_143439 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assert_equal_143432, *[item_call_result_143436, c_143437], **kwargs_143438)
    
    
    # Assigning a Call to a Name (line 33):
    
    # Call to byteswap_u4(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'c' (line 33)
    c_143442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 28), 'c', False)
    # Processing the call keyword arguments (line 33)
    kwargs_143443 = {}
    # Getting the type of 'm5u' (line 33)
    m5u_143440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'm5u', False)
    # Obtaining the member 'byteswap_u4' of a type (line 33)
    byteswap_u4_143441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), m5u_143440, 'byteswap_u4')
    # Calling byteswap_u4(args, kwargs) (line 33)
    byteswap_u4_call_result_143444 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), byteswap_u4_143441, *[c_143442], **kwargs_143443)
    
    # Assigning a type to the variable 'd' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'd', byteswap_u4_call_result_143444)
    
    # Call to assert_equal(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to item(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_143448 = {}
    # Getting the type of 'a' (line 34)
    a_143446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'a', False)
    # Obtaining the member 'item' of a type (line 34)
    item_143447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), a_143446, 'item')
    # Calling item(args, kwargs) (line 34)
    item_call_result_143449 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), item_143447, *[], **kwargs_143448)
    
    # Getting the type of 'd' (line 34)
    d_143450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'd', False)
    # Processing the call keyword arguments (line 34)
    kwargs_143451 = {}
    # Getting the type of 'assert_equal' (line 34)
    assert_equal_143445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 34)
    assert_equal_call_result_143452 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_equal_143445, *[item_call_result_143449, d_143450], **kwargs_143451)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_byteswap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_byteswap' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_143453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143453)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_byteswap'
    return stypy_return_type_143453

# Assigning a type to the variable 'test_byteswap' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_byteswap', test_byteswap)

@norecursion
def _make_tag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 37)
    False_143454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 40), 'False')
    defaults = [False_143454]
    # Create a new context for function '_make_tag'
    module_type_store = module_type_store.open_function_context('_make_tag', 37, 0, False)
    
    # Passed parameters checking function
    _make_tag.stypy_localization = localization
    _make_tag.stypy_type_of_self = None
    _make_tag.stypy_type_store = module_type_store
    _make_tag.stypy_function_name = '_make_tag'
    _make_tag.stypy_param_names_list = ['base_dt', 'val', 'mdtype', 'sde']
    _make_tag.stypy_varargs_param_name = None
    _make_tag.stypy_kwargs_param_name = None
    _make_tag.stypy_call_defaults = defaults
    _make_tag.stypy_call_varargs = varargs
    _make_tag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_tag', ['base_dt', 'val', 'mdtype', 'sde'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_tag', localization, ['base_dt', 'val', 'mdtype', 'sde'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_tag(...)' code ##################

    str_143455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', ' Makes a simple matlab tag, full or sde ')
    
    # Assigning a Call to a Name (line 39):
    
    # Call to dtype(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'base_dt' (line 39)
    base_dt_143458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'base_dt', False)
    # Processing the call keyword arguments (line 39)
    kwargs_143459 = {}
    # Getting the type of 'np' (line 39)
    np_143456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'np', False)
    # Obtaining the member 'dtype' of a type (line 39)
    dtype_143457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 14), np_143456, 'dtype')
    # Calling dtype(args, kwargs) (line 39)
    dtype_call_result_143460 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), dtype_143457, *[base_dt_143458], **kwargs_143459)
    
    # Assigning a type to the variable 'base_dt' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'base_dt', dtype_call_result_143460)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to to_numpy_code(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'base_dt' (line 40)
    base_dt_143463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'base_dt', False)
    # Obtaining the member 'byteorder' of a type (line 40)
    byteorder_143464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 27), base_dt_143463, 'byteorder')
    # Processing the call keyword arguments (line 40)
    kwargs_143465 = {}
    # Getting the type of 'boc' (line 40)
    boc_143461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'boc', False)
    # Obtaining the member 'to_numpy_code' of a type (line 40)
    to_numpy_code_143462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 9), boc_143461, 'to_numpy_code')
    # Calling to_numpy_code(args, kwargs) (line 40)
    to_numpy_code_call_result_143466 = invoke(stypy.reporting.localization.Localization(__file__, 40, 9), to_numpy_code_143462, *[byteorder_143464], **kwargs_143465)
    
    # Assigning a type to the variable 'bo' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'bo', to_numpy_code_call_result_143466)
    
    # Assigning a Attribute to a Name (line 41):
    # Getting the type of 'base_dt' (line 41)
    base_dt_143467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'base_dt')
    # Obtaining the member 'itemsize' of a type (line 41)
    itemsize_143468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), base_dt_143467, 'itemsize')
    # Assigning a type to the variable 'byte_count' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'byte_count', itemsize_143468)
    
    
    # Getting the type of 'sde' (line 42)
    sde_143469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'sde')
    # Applying the 'not' unary operator (line 42)
    result_not__143470 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 7), 'not', sde_143469)
    
    # Testing the type of an if condition (line 42)
    if_condition_143471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), result_not__143470)
    # Assigning a type to the variable 'if_condition_143471' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_143471', if_condition_143471)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 43):
    # Getting the type of 'bo' (line 43)
    bo_143472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'bo')
    str_143473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'str', 'u4')
    # Applying the binary operator '+' (line 43)
    result_add_143474 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 14), '+', bo_143472, str_143473)
    
    # Assigning a type to the variable 'udt' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'udt', result_add_143474)
    
    # Assigning a BinOp to a Name (line 44):
    int_143475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 18), 'int')
    # Getting the type of 'byte_count' (line 44)
    byte_count_143476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'byte_count')
    int_143477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'int')
    # Applying the binary operator '%' (line 44)
    result_mod_143478 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), '%', byte_count_143476, int_143477)
    
    # Applying the binary operator '-' (line 44)
    result_sub_143479 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 18), '-', int_143475, result_mod_143478)
    
    # Assigning a type to the variable 'padding' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'padding', result_sub_143479)
    
    # Assigning a List to a Name (line 45):
    
    # Obtaining an instance of the builtin type 'list' (line 45)
    list_143480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 45)
    # Adding element type (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_143481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    str_143482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'str', 'mdtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_143481, str_143482)
    # Adding element type (line 45)
    # Getting the type of 'udt' (line 45)
    udt_143483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'udt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_143481, udt_143483)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 17), list_143480, tuple_143481)
    # Adding element type (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_143484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    str_143485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'str', 'byte_count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), tuple_143484, str_143485)
    # Adding element type (line 46)
    # Getting the type of 'udt' (line 46)
    udt_143486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'udt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), tuple_143484, udt_143486)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 17), list_143480, tuple_143484)
    # Adding element type (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_143487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    str_143488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'str', 'val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_143487, str_143488)
    # Adding element type (line 47)
    # Getting the type of 'base_dt' (line 47)
    base_dt_143489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'base_dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), tuple_143487, base_dt_143489)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 17), list_143480, tuple_143487)
    
    # Assigning a type to the variable 'all_dt' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'all_dt', list_143480)
    
    # Getting the type of 'padding' (line 48)
    padding_143490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'padding')
    # Testing the type of an if condition (line 48)
    if_condition_143491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), padding_143490)
    # Assigning a type to the variable 'if_condition_143491' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_143491', if_condition_143491)
    # SSA begins for if statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Obtaining an instance of the builtin type 'tuple' (line 49)
    tuple_143494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 49)
    # Adding element type (line 49)
    str_143495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'str', 'padding')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 27), tuple_143494, str_143495)
    # Adding element type (line 49)
    str_143496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 38), 'str', 'u1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 27), tuple_143494, str_143496)
    # Adding element type (line 49)
    # Getting the type of 'padding' (line 49)
    padding_143497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 44), 'padding', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 27), tuple_143494, padding_143497)
    
    # Processing the call keyword arguments (line 49)
    kwargs_143498 = {}
    # Getting the type of 'all_dt' (line 49)
    all_dt_143492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'all_dt', False)
    # Obtaining the member 'append' of a type (line 49)
    append_143493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), all_dt_143492, 'append')
    # Calling append(args, kwargs) (line 49)
    append_call_result_143499 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), append_143493, *[tuple_143494], **kwargs_143498)
    
    # SSA join for if statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 42)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 51):
    # Getting the type of 'bo' (line 51)
    bo_143500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'bo')
    str_143501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'str', 'u2')
    # Applying the binary operator '+' (line 51)
    result_add_143502 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 14), '+', bo_143500, str_143501)
    
    # Assigning a type to the variable 'udt' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'udt', result_add_143502)
    
    # Assigning a BinOp to a Name (line 52):
    int_143503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'int')
    # Getting the type of 'byte_count' (line 52)
    byte_count_143504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'byte_count')
    # Applying the binary operator '-' (line 52)
    result_sub_143505 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 18), '-', int_143503, byte_count_143504)
    
    # Assigning a type to the variable 'padding' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'padding', result_sub_143505)
    
    
    # Getting the type of 'bo' (line 53)
    bo_143506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'bo')
    str_143507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'str', '<')
    # Applying the binary operator '==' (line 53)
    result_eq_143508 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 11), '==', bo_143506, str_143507)
    
    # Testing the type of an if condition (line 53)
    if_condition_143509 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 8), result_eq_143508)
    # Assigning a type to the variable 'if_condition_143509' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'if_condition_143509', if_condition_143509)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 54):
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_143510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_143511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    str_143512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 23), 'str', 'mdtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 23), tuple_143511, str_143512)
    # Adding element type (line 54)
    # Getting the type of 'udt' (line 54)
    udt_143513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'udt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 23), tuple_143511, udt_143513)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), list_143510, tuple_143511)
    # Adding element type (line 54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_143514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    str_143515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'str', 'byte_count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 23), tuple_143514, str_143515)
    # Adding element type (line 55)
    # Getting the type of 'udt' (line 55)
    udt_143516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 37), 'udt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 23), tuple_143514, udt_143516)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), list_143510, tuple_143514)
    # Adding element type (line 54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_143517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    str_143518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', 'val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 23), tuple_143517, str_143518)
    # Adding element type (line 56)
    # Getting the type of 'base_dt' (line 56)
    base_dt_143519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'base_dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 23), tuple_143517, base_dt_143519)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), list_143510, tuple_143517)
    
    # Assigning a type to the variable 'all_dt' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'all_dt', list_143510)
    # SSA branch for the else part of an if statement (line 53)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_143520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_143521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    str_143522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'str', 'byte_count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), tuple_143521, str_143522)
    # Adding element type (line 58)
    # Getting the type of 'udt' (line 58)
    udt_143523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'udt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), tuple_143521, udt_143523)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_143520, tuple_143521)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'tuple' (line 59)
    tuple_143524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 59)
    # Adding element type (line 59)
    str_143525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'str', 'mdtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 23), tuple_143524, str_143525)
    # Adding element type (line 59)
    # Getting the type of 'udt' (line 59)
    udt_143526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'udt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 23), tuple_143524, udt_143526)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_143520, tuple_143524)
    # Adding element type (line 58)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_143527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    str_143528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'str', 'val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_143527, str_143528)
    # Adding element type (line 60)
    # Getting the type of 'base_dt' (line 60)
    base_dt_143529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 30), 'base_dt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_143527, base_dt_143529)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), list_143520, tuple_143527)
    
    # Assigning a type to the variable 'all_dt' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'all_dt', list_143520)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'padding' (line 61)
    padding_143530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'padding')
    # Testing the type of an if condition (line 61)
    if_condition_143531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), padding_143530)
    # Assigning a type to the variable 'if_condition_143531' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'if_condition_143531', if_condition_143531)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_143534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    str_143535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'str', 'padding')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 27), tuple_143534, str_143535)
    # Adding element type (line 62)
    str_143536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 38), 'str', 'u1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 27), tuple_143534, str_143536)
    # Adding element type (line 62)
    # Getting the type of 'padding' (line 62)
    padding_143537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 44), 'padding', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 27), tuple_143534, padding_143537)
    
    # Processing the call keyword arguments (line 62)
    kwargs_143538 = {}
    # Getting the type of 'all_dt' (line 62)
    all_dt_143532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'all_dt', False)
    # Obtaining the member 'append' of a type (line 62)
    append_143533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), all_dt_143532, 'append')
    # Calling append(args, kwargs) (line 62)
    append_call_result_143539 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), append_143533, *[tuple_143534], **kwargs_143538)
    
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 63):
    
    # Call to zeros(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_143542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    int_143543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), tuple_143542, int_143543)
    
    # Processing the call keyword arguments (line 63)
    # Getting the type of 'all_dt' (line 63)
    all_dt_143544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'all_dt', False)
    keyword_143545 = all_dt_143544
    kwargs_143546 = {'dtype': keyword_143545}
    # Getting the type of 'np' (line 63)
    np_143540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 63)
    zeros_143541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 10), np_143540, 'zeros')
    # Calling zeros(args, kwargs) (line 63)
    zeros_call_result_143547 = invoke(stypy.reporting.localization.Localization(__file__, 63, 10), zeros_143541, *[tuple_143542], **kwargs_143546)
    
    # Assigning a type to the variable 'tag' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'tag', zeros_call_result_143547)
    
    # Assigning a Name to a Subscript (line 64):
    # Getting the type of 'mdtype' (line 64)
    mdtype_143548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'mdtype')
    # Getting the type of 'tag' (line 64)
    tag_143549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tag')
    str_143550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'str', 'mdtype')
    # Storing an element on a container (line 64)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 4), tag_143549, (str_143550, mdtype_143548))
    
    # Assigning a Name to a Subscript (line 65):
    # Getting the type of 'byte_count' (line 65)
    byte_count_143551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'byte_count')
    # Getting the type of 'tag' (line 65)
    tag_143552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tag')
    str_143553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'str', 'byte_count')
    # Storing an element on a container (line 65)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 4), tag_143552, (str_143553, byte_count_143551))
    
    # Assigning a Name to a Subscript (line 66):
    # Getting the type of 'val' (line 66)
    val_143554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'val')
    # Getting the type of 'tag' (line 66)
    tag_143555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tag')
    str_143556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'str', 'val')
    # Storing an element on a container (line 66)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 4), tag_143555, (str_143556, val_143554))
    # Getting the type of 'tag' (line 67)
    tag_143557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'tag')
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type', tag_143557)
    
    # ################# End of '_make_tag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_tag' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_143558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143558)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_tag'
    return stypy_return_type_143558

# Assigning a type to the variable '_make_tag' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '_make_tag', _make_tag)

@norecursion
def _write_stream(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_write_stream'
    module_type_store = module_type_store.open_function_context('_write_stream', 70, 0, False)
    
    # Passed parameters checking function
    _write_stream.stypy_localization = localization
    _write_stream.stypy_type_of_self = None
    _write_stream.stypy_type_store = module_type_store
    _write_stream.stypy_function_name = '_write_stream'
    _write_stream.stypy_param_names_list = ['stream']
    _write_stream.stypy_varargs_param_name = 'strings'
    _write_stream.stypy_kwargs_param_name = None
    _write_stream.stypy_call_defaults = defaults
    _write_stream.stypy_call_varargs = varargs
    _write_stream.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_write_stream', ['stream'], 'strings', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_write_stream', localization, ['stream'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_write_stream(...)' code ##################

    
    # Call to truncate(...): (line 71)
    # Processing the call arguments (line 71)
    int_143561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_143562 = {}
    # Getting the type of 'stream' (line 71)
    stream_143559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stream', False)
    # Obtaining the member 'truncate' of a type (line 71)
    truncate_143560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), stream_143559, 'truncate')
    # Calling truncate(args, kwargs) (line 71)
    truncate_call_result_143563 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), truncate_143560, *[int_143561], **kwargs_143562)
    
    
    # Call to seek(...): (line 72)
    # Processing the call arguments (line 72)
    int_143566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 16), 'int')
    # Processing the call keyword arguments (line 72)
    kwargs_143567 = {}
    # Getting the type of 'stream' (line 72)
    stream_143564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stream', False)
    # Obtaining the member 'seek' of a type (line 72)
    seek_143565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), stream_143564, 'seek')
    # Calling seek(args, kwargs) (line 72)
    seek_call_result_143568 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), seek_143565, *[int_143566], **kwargs_143567)
    
    
    # Getting the type of 'strings' (line 73)
    strings_143569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'strings')
    # Testing the type of a for loop iterable (line 73)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 4), strings_143569)
    # Getting the type of the for loop variable (line 73)
    for_loop_var_143570 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 4), strings_143569)
    # Assigning a type to the variable 's' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 's', for_loop_var_143570)
    # SSA begins for a for statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to write(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 's' (line 74)
    s_143573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 's', False)
    # Processing the call keyword arguments (line 74)
    kwargs_143574 = {}
    # Getting the type of 'stream' (line 74)
    stream_143571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stream', False)
    # Obtaining the member 'write' of a type (line 74)
    write_143572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), stream_143571, 'write')
    # Calling write(args, kwargs) (line 74)
    write_call_result_143575 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), write_143572, *[s_143573], **kwargs_143574)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to seek(...): (line 75)
    # Processing the call arguments (line 75)
    int_143578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 16), 'int')
    # Processing the call keyword arguments (line 75)
    kwargs_143579 = {}
    # Getting the type of 'stream' (line 75)
    stream_143576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stream', False)
    # Obtaining the member 'seek' of a type (line 75)
    seek_143577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), stream_143576, 'seek')
    # Calling seek(args, kwargs) (line 75)
    seek_call_result_143580 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), seek_143577, *[int_143578], **kwargs_143579)
    
    
    # ################# End of '_write_stream(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_write_stream' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_143581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143581)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_write_stream'
    return stypy_return_type_143581

# Assigning a type to the variable '_write_stream' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), '_write_stream', _write_stream)

@norecursion
def _make_readerlike(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'boc' (line 78)
    boc_143582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'boc')
    # Obtaining the member 'native_code' of a type (line 78)
    native_code_143583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 40), boc_143582, 'native_code')
    defaults = [native_code_143583]
    # Create a new context for function '_make_readerlike'
    module_type_store = module_type_store.open_function_context('_make_readerlike', 78, 0, False)
    
    # Passed parameters checking function
    _make_readerlike.stypy_localization = localization
    _make_readerlike.stypy_type_of_self = None
    _make_readerlike.stypy_type_store = module_type_store
    _make_readerlike.stypy_function_name = '_make_readerlike'
    _make_readerlike.stypy_param_names_list = ['stream', 'byte_order']
    _make_readerlike.stypy_varargs_param_name = None
    _make_readerlike.stypy_kwargs_param_name = None
    _make_readerlike.stypy_call_defaults = defaults
    _make_readerlike.stypy_call_varargs = varargs
    _make_readerlike.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_readerlike', ['stream', 'byte_order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_readerlike', localization, ['stream', 'byte_order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_readerlike(...)' code ##################

    # Declaration of the 'R' class

    class R(object, ):
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 79, 4, False)
            # Assigning a type to the variable 'self' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'R.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'R' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'R', R)
    
    # Assigning a Call to a Name (line 81):
    
    # Call to R(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_143585 = {}
    # Getting the type of 'R' (line 81)
    R_143584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'R', False)
    # Calling R(args, kwargs) (line 81)
    R_call_result_143586 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), R_143584, *[], **kwargs_143585)
    
    # Assigning a type to the variable 'r' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'r', R_call_result_143586)
    
    # Assigning a Name to a Attribute (line 82):
    # Getting the type of 'stream' (line 82)
    stream_143587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'stream')
    # Getting the type of 'r' (line 82)
    r_143588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'r')
    # Setting the type of the member 'mat_stream' of a type (line 82)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), r_143588, 'mat_stream', stream_143587)
    
    # Assigning a Name to a Attribute (line 83):
    # Getting the type of 'byte_order' (line 83)
    byte_order_143589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'byte_order')
    # Getting the type of 'r' (line 83)
    r_143590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'r')
    # Setting the type of the member 'byte_order' of a type (line 83)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 4), r_143590, 'byte_order', byte_order_143589)
    
    # Assigning a Name to a Attribute (line 84):
    # Getting the type of 'True' (line 84)
    True_143591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'True')
    # Getting the type of 'r' (line 84)
    r_143592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'r')
    # Setting the type of the member 'struct_as_record' of a type (line 84)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), r_143592, 'struct_as_record', True_143591)
    
    # Assigning a Call to a Attribute (line 85):
    
    # Call to getdefaultencoding(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_143595 = {}
    # Getting the type of 'sys' (line 85)
    sys_143593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'sys', False)
    # Obtaining the member 'getdefaultencoding' of a type (line 85)
    getdefaultencoding_143594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), sys_143593, 'getdefaultencoding')
    # Calling getdefaultencoding(args, kwargs) (line 85)
    getdefaultencoding_call_result_143596 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), getdefaultencoding_143594, *[], **kwargs_143595)
    
    # Getting the type of 'r' (line 85)
    r_143597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'r')
    # Setting the type of the member 'uint16_codec' of a type (line 85)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), r_143597, 'uint16_codec', getdefaultencoding_call_result_143596)
    
    # Assigning a Name to a Attribute (line 86):
    # Getting the type of 'False' (line 86)
    False_143598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'False')
    # Getting the type of 'r' (line 86)
    r_143599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'r')
    # Setting the type of the member 'chars_as_strings' of a type (line 86)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), r_143599, 'chars_as_strings', False_143598)
    
    # Assigning a Name to a Attribute (line 87):
    # Getting the type of 'False' (line 87)
    False_143600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'False')
    # Getting the type of 'r' (line 87)
    r_143601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'r')
    # Setting the type of the member 'mat_dtype' of a type (line 87)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 4), r_143601, 'mat_dtype', False_143600)
    
    # Assigning a Name to a Attribute (line 88):
    # Getting the type of 'False' (line 88)
    False_143602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'False')
    # Getting the type of 'r' (line 88)
    r_143603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'r')
    # Setting the type of the member 'squeeze_me' of a type (line 88)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), r_143603, 'squeeze_me', False_143602)
    # Getting the type of 'r' (line 89)
    r_143604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', r_143604)
    
    # ################# End of '_make_readerlike(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_readerlike' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_143605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143605)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_readerlike'
    return stypy_return_type_143605

# Assigning a type to the variable '_make_readerlike' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), '_make_readerlike', _make_readerlike)

@norecursion
def test_read_tag(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_tag'
    module_type_store = module_type_store.open_function_context('test_read_tag', 92, 0, False)
    
    # Passed parameters checking function
    test_read_tag.stypy_localization = localization
    test_read_tag.stypy_type_of_self = None
    test_read_tag.stypy_type_store = module_type_store
    test_read_tag.stypy_function_name = 'test_read_tag'
    test_read_tag.stypy_param_names_list = []
    test_read_tag.stypy_varargs_param_name = None
    test_read_tag.stypy_kwargs_param_name = None
    test_read_tag.stypy_call_defaults = defaults
    test_read_tag.stypy_call_varargs = varargs
    test_read_tag.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_tag', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_tag', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_tag(...)' code ##################

    
    # Assigning a Call to a Name (line 95):
    
    # Call to BytesIO(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_143607 = {}
    # Getting the type of 'BytesIO' (line 95)
    BytesIO_143606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 95)
    BytesIO_call_result_143608 = invoke(stypy.reporting.localization.Localization(__file__, 95, 13), BytesIO_143606, *[], **kwargs_143607)
    
    # Assigning a type to the variable 'str_io' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'str_io', BytesIO_call_result_143608)
    
    # Assigning a Call to a Name (line 96):
    
    # Call to _make_readerlike(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'str_io' (line 96)
    str_io_143610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'str_io', False)
    # Processing the call keyword arguments (line 96)
    kwargs_143611 = {}
    # Getting the type of '_make_readerlike' (line 96)
    _make_readerlike_143609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), '_make_readerlike', False)
    # Calling _make_readerlike(args, kwargs) (line 96)
    _make_readerlike_call_result_143612 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), _make_readerlike_143609, *[str_io_143610], **kwargs_143611)
    
    # Assigning a type to the variable 'r' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'r', _make_readerlike_call_result_143612)
    
    # Assigning a Call to a Name (line 97):
    
    # Call to VarReader5(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'r' (line 97)
    r_143615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'r', False)
    # Processing the call keyword arguments (line 97)
    kwargs_143616 = {}
    # Getting the type of 'm5u' (line 97)
    m5u_143613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'm5u', False)
    # Obtaining the member 'VarReader5' of a type (line 97)
    VarReader5_143614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), m5u_143613, 'VarReader5')
    # Calling VarReader5(args, kwargs) (line 97)
    VarReader5_call_result_143617 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), VarReader5_143614, *[r_143615], **kwargs_143616)
    
    # Assigning a type to the variable 'c_reader' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'c_reader', VarReader5_call_result_143617)
    
    # Call to assert_raises(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'IOError' (line 99)
    IOError_143619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'IOError', False)
    # Getting the type of 'c_reader' (line 99)
    c_reader_143620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'c_reader', False)
    # Obtaining the member 'read_tag' of a type (line 99)
    read_tag_143621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), c_reader_143620, 'read_tag')
    # Processing the call keyword arguments (line 99)
    kwargs_143622 = {}
    # Getting the type of 'assert_raises' (line 99)
    assert_raises_143618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 99)
    assert_raises_call_result_143623 = invoke(stypy.reporting.localization.Localization(__file__, 99, 4), assert_raises_143618, *[IOError_143619, read_tag_143621], **kwargs_143622)
    
    
    # Assigning a Call to a Name (line 101):
    
    # Call to _make_tag(...): (line 101)
    # Processing the call arguments (line 101)
    str_143625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'str', 'i4')
    int_143626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'int')
    # Getting the type of 'mio5p' (line 101)
    mio5p_143627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'mio5p', False)
    # Obtaining the member 'miINT32' of a type (line 101)
    miINT32_143628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 29), mio5p_143627, 'miINT32')
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'True' (line 101)
    True_143629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 48), 'True', False)
    keyword_143630 = True_143629
    kwargs_143631 = {'sde': keyword_143630}
    # Getting the type of '_make_tag' (line 101)
    _make_tag_143624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), '_make_tag', False)
    # Calling _make_tag(args, kwargs) (line 101)
    _make_tag_call_result_143632 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), _make_tag_143624, *[str_143625, int_143626, miINT32_143628], **kwargs_143631)
    
    # Assigning a type to the variable 'tag' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tag', _make_tag_call_result_143632)
    
    # Assigning a Num to a Subscript (line 102):
    int_143633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'int')
    # Getting the type of 'tag' (line 102)
    tag_143634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'tag')
    str_143635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'str', 'byte_count')
    # Storing an element on a container (line 102)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 4), tag_143634, (str_143635, int_143633))
    
    # Call to _write_stream(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'str_io' (line 103)
    str_io_143637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'str_io', False)
    
    # Call to tostring(...): (line 103)
    # Processing the call keyword arguments (line 103)
    kwargs_143640 = {}
    # Getting the type of 'tag' (line 103)
    tag_143638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'tag', False)
    # Obtaining the member 'tostring' of a type (line 103)
    tostring_143639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), tag_143638, 'tostring')
    # Calling tostring(args, kwargs) (line 103)
    tostring_call_result_143641 = invoke(stypy.reporting.localization.Localization(__file__, 103, 26), tostring_143639, *[], **kwargs_143640)
    
    # Processing the call keyword arguments (line 103)
    kwargs_143642 = {}
    # Getting the type of '_write_stream' (line 103)
    _write_stream_143636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), '_write_stream', False)
    # Calling _write_stream(args, kwargs) (line 103)
    _write_stream_call_result_143643 = invoke(stypy.reporting.localization.Localization(__file__, 103, 4), _write_stream_143636, *[str_io_143637, tostring_call_result_143641], **kwargs_143642)
    
    
    # Call to assert_raises(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'ValueError' (line 104)
    ValueError_143645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'ValueError', False)
    # Getting the type of 'c_reader' (line 104)
    c_reader_143646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'c_reader', False)
    # Obtaining the member 'read_tag' of a type (line 104)
    read_tag_143647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 30), c_reader_143646, 'read_tag')
    # Processing the call keyword arguments (line 104)
    kwargs_143648 = {}
    # Getting the type of 'assert_raises' (line 104)
    assert_raises_143644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 104)
    assert_raises_call_result_143649 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), assert_raises_143644, *[ValueError_143645, read_tag_143647], **kwargs_143648)
    
    
    # ################# End of 'test_read_tag(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_tag' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_143650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143650)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_tag'
    return stypy_return_type_143650

# Assigning a type to the variable 'test_read_tag' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'test_read_tag', test_read_tag)

@norecursion
def test_read_stream(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_stream'
    module_type_store = module_type_store.open_function_context('test_read_stream', 107, 0, False)
    
    # Passed parameters checking function
    test_read_stream.stypy_localization = localization
    test_read_stream.stypy_type_of_self = None
    test_read_stream.stypy_type_store = module_type_store
    test_read_stream.stypy_function_name = 'test_read_stream'
    test_read_stream.stypy_param_names_list = []
    test_read_stream.stypy_varargs_param_name = None
    test_read_stream.stypy_kwargs_param_name = None
    test_read_stream.stypy_call_defaults = defaults
    test_read_stream.stypy_call_varargs = varargs
    test_read_stream.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_stream', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_stream', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_stream(...)' code ##################

    
    # Assigning a Call to a Name (line 108):
    
    # Call to _make_tag(...): (line 108)
    # Processing the call arguments (line 108)
    str_143652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'str', 'i4')
    int_143653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'int')
    # Getting the type of 'mio5p' (line 108)
    mio5p_143654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'mio5p', False)
    # Obtaining the member 'miINT32' of a type (line 108)
    miINT32_143655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 29), mio5p_143654, 'miINT32')
    # Processing the call keyword arguments (line 108)
    # Getting the type of 'True' (line 108)
    True_143656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 48), 'True', False)
    keyword_143657 = True_143656
    kwargs_143658 = {'sde': keyword_143657}
    # Getting the type of '_make_tag' (line 108)
    _make_tag_143651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 10), '_make_tag', False)
    # Calling _make_tag(args, kwargs) (line 108)
    _make_tag_call_result_143659 = invoke(stypy.reporting.localization.Localization(__file__, 108, 10), _make_tag_143651, *[str_143652, int_143653, miINT32_143655], **kwargs_143658)
    
    # Assigning a type to the variable 'tag' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'tag', _make_tag_call_result_143659)
    
    # Assigning a Call to a Name (line 109):
    
    # Call to tostring(...): (line 109)
    # Processing the call keyword arguments (line 109)
    kwargs_143662 = {}
    # Getting the type of 'tag' (line 109)
    tag_143660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'tag', False)
    # Obtaining the member 'tostring' of a type (line 109)
    tostring_143661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 14), tag_143660, 'tostring')
    # Calling tostring(args, kwargs) (line 109)
    tostring_call_result_143663 = invoke(stypy.reporting.localization.Localization(__file__, 109, 14), tostring_143661, *[], **kwargs_143662)
    
    # Assigning a type to the variable 'tag_str' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'tag_str', tostring_call_result_143663)
    
    # Assigning a Call to a Name (line 110):
    
    # Call to cStringIO(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'tag_str' (line 110)
    tag_str_143665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'tag_str', False)
    # Processing the call keyword arguments (line 110)
    kwargs_143666 = {}
    # Getting the type of 'cStringIO' (line 110)
    cStringIO_143664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'cStringIO', False)
    # Calling cStringIO(args, kwargs) (line 110)
    cStringIO_call_result_143667 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), cStringIO_143664, *[tag_str_143665], **kwargs_143666)
    
    # Assigning a type to the variable 'str_io' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'str_io', cStringIO_call_result_143667)
    
    # Assigning a Call to a Name (line 111):
    
    # Call to make_stream(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'str_io' (line 111)
    str_io_143670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'str_io', False)
    # Processing the call keyword arguments (line 111)
    kwargs_143671 = {}
    # Getting the type of 'streams' (line 111)
    streams_143668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'streams', False)
    # Obtaining the member 'make_stream' of a type (line 111)
    make_stream_143669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 9), streams_143668, 'make_stream')
    # Calling make_stream(args, kwargs) (line 111)
    make_stream_call_result_143672 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), make_stream_143669, *[str_io_143670], **kwargs_143671)
    
    # Assigning a type to the variable 'st' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'st', make_stream_call_result_143672)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to _read_into(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'st' (line 112)
    st_143675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'st', False)
    # Getting the type of 'tag' (line 112)
    tag_143676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'tag', False)
    # Obtaining the member 'itemsize' of a type (line 112)
    itemsize_143677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), tag_143676, 'itemsize')
    # Processing the call keyword arguments (line 112)
    kwargs_143678 = {}
    # Getting the type of 'streams' (line 112)
    streams_143673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'streams', False)
    # Obtaining the member '_read_into' of a type (line 112)
    _read_into_143674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), streams_143673, '_read_into')
    # Calling _read_into(args, kwargs) (line 112)
    _read_into_call_result_143679 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), _read_into_143674, *[st_143675, itemsize_143677], **kwargs_143678)
    
    # Assigning a type to the variable 's' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 's', _read_into_call_result_143679)
    
    # Call to assert_equal(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 's' (line 113)
    s_143681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 's', False)
    
    # Call to tostring(...): (line 113)
    # Processing the call keyword arguments (line 113)
    kwargs_143684 = {}
    # Getting the type of 'tag' (line 113)
    tag_143682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'tag', False)
    # Obtaining the member 'tostring' of a type (line 113)
    tostring_143683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), tag_143682, 'tostring')
    # Calling tostring(args, kwargs) (line 113)
    tostring_call_result_143685 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), tostring_143683, *[], **kwargs_143684)
    
    # Processing the call keyword arguments (line 113)
    kwargs_143686 = {}
    # Getting the type of 'assert_equal' (line 113)
    assert_equal_143680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 113)
    assert_equal_call_result_143687 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), assert_equal_143680, *[s_143681, tostring_call_result_143685], **kwargs_143686)
    
    
    # ################# End of 'test_read_stream(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_stream' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_143688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143688)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_stream'
    return stypy_return_type_143688

# Assigning a type to the variable 'test_read_stream' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'test_read_stream', test_read_stream)

@norecursion
def test_read_numeric(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_numeric'
    module_type_store = module_type_store.open_function_context('test_read_numeric', 116, 0, False)
    
    # Passed parameters checking function
    test_read_numeric.stypy_localization = localization
    test_read_numeric.stypy_type_of_self = None
    test_read_numeric.stypy_type_store = module_type_store
    test_read_numeric.stypy_function_name = 'test_read_numeric'
    test_read_numeric.stypy_param_names_list = []
    test_read_numeric.stypy_varargs_param_name = None
    test_read_numeric.stypy_kwargs_param_name = None
    test_read_numeric.stypy_call_defaults = defaults
    test_read_numeric.stypy_call_varargs = varargs
    test_read_numeric.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_numeric', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_numeric', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_numeric(...)' code ##################

    
    # Assigning a Call to a Name (line 118):
    
    # Call to cStringIO(...): (line 118)
    # Processing the call keyword arguments (line 118)
    kwargs_143690 = {}
    # Getting the type of 'cStringIO' (line 118)
    cStringIO_143689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 13), 'cStringIO', False)
    # Calling cStringIO(args, kwargs) (line 118)
    cStringIO_call_result_143691 = invoke(stypy.reporting.localization.Localization(__file__, 118, 13), cStringIO_143689, *[], **kwargs_143690)
    
    # Assigning a type to the variable 'str_io' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'str_io', cStringIO_call_result_143691)
    
    # Assigning a Call to a Name (line 119):
    
    # Call to _make_readerlike(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'str_io' (line 119)
    str_io_143693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'str_io', False)
    # Processing the call keyword arguments (line 119)
    kwargs_143694 = {}
    # Getting the type of '_make_readerlike' (line 119)
    _make_readerlike_143692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), '_make_readerlike', False)
    # Calling _make_readerlike(args, kwargs) (line 119)
    _make_readerlike_call_result_143695 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), _make_readerlike_143692, *[str_io_143693], **kwargs_143694)
    
    # Assigning a type to the variable 'r' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'r', _make_readerlike_call_result_143695)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_143696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_143697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    str_143698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 34), 'str', 'u2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 34), tuple_143697, str_143698)
    # Adding element type (line 121)
    int_143699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 34), tuple_143697, int_143699)
    # Adding element type (line 121)
    # Getting the type of 'mio5p' (line 121)
    mio5p_143700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 44), 'mio5p')
    # Obtaining the member 'miUINT16' of a type (line 121)
    miUINT16_143701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 44), mio5p_143700, 'miUINT16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 34), tuple_143697, miUINT16_143701)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 33), tuple_143696, tuple_143697)
    # Adding element type (line 121)
    
    # Obtaining an instance of the builtin type 'tuple' (line 122)
    tuple_143702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 122)
    # Adding element type (line 122)
    str_143703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'str', 'i4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 34), tuple_143702, str_143703)
    # Adding element type (line 122)
    int_143704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 34), tuple_143702, int_143704)
    # Adding element type (line 122)
    # Getting the type of 'mio5p' (line 122)
    mio5p_143705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 43), 'mio5p')
    # Obtaining the member 'miINT32' of a type (line 122)
    miINT32_143706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 43), mio5p_143705, 'miINT32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 34), tuple_143702, miINT32_143706)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 33), tuple_143696, tuple_143702)
    # Adding element type (line 121)
    
    # Obtaining an instance of the builtin type 'tuple' (line 123)
    tuple_143707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 123)
    # Adding element type (line 123)
    str_143708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'str', 'i2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 34), tuple_143707, str_143708)
    # Adding element type (line 123)
    int_143709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 34), tuple_143707, int_143709)
    # Adding element type (line 123)
    # Getting the type of 'mio5p' (line 123)
    mio5p_143710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'mio5p')
    # Obtaining the member 'miINT16' of a type (line 123)
    miINT16_143711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 44), mio5p_143710, 'miINT16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 34), tuple_143707, miINT16_143711)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 33), tuple_143696, tuple_143707)
    
    # Testing the type of a for loop iterable (line 121)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 4), tuple_143696)
    # Getting the type of the for loop variable (line 121)
    for_loop_var_143712 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 4), tuple_143696)
    # Assigning a type to the variable 'base_dt' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'base_dt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 4), for_loop_var_143712))
    # Assigning a type to the variable 'val' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 4), for_loop_var_143712))
    # Assigning a type to the variable 'mdtype' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'mdtype', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 4), for_loop_var_143712))
    # SSA begins for a for statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 124)
    tuple_143713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 124)
    # Adding element type (line 124)
    str_143714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'str', '<')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 26), tuple_143713, str_143714)
    # Adding element type (line 124)
    str_143715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 31), 'str', '>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 26), tuple_143713, str_143715)
    
    # Testing the type of a for loop iterable (line 124)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 8), tuple_143713)
    # Getting the type of the for loop variable (line 124)
    for_loop_var_143716 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 8), tuple_143713)
    # Assigning a type to the variable 'byte_code' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'byte_code', for_loop_var_143716)
    # SSA begins for a for statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Attribute (line 125):
    # Getting the type of 'byte_code' (line 125)
    byte_code_143717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'byte_code')
    # Getting the type of 'r' (line 125)
    r_143718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'r')
    # Setting the type of the member 'byte_order' of a type (line 125)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), r_143718, 'byte_order', byte_code_143717)
    
    # Assigning a Call to a Name (line 126):
    
    # Call to VarReader5(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'r' (line 126)
    r_143721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 38), 'r', False)
    # Processing the call keyword arguments (line 126)
    kwargs_143722 = {}
    # Getting the type of 'm5u' (line 126)
    m5u_143719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'm5u', False)
    # Obtaining the member 'VarReader5' of a type (line 126)
    VarReader5_143720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), m5u_143719, 'VarReader5')
    # Calling VarReader5(args, kwargs) (line 126)
    VarReader5_call_result_143723 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), VarReader5_143720, *[r_143721], **kwargs_143722)
    
    # Assigning a type to the variable 'c_reader' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'c_reader', VarReader5_call_result_143723)
    
    # Call to assert_equal(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'c_reader' (line 127)
    c_reader_143725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'c_reader', False)
    # Obtaining the member 'little_endian' of a type (line 127)
    little_endian_143726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 25), c_reader_143725, 'little_endian')
    
    # Getting the type of 'byte_code' (line 127)
    byte_code_143727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 49), 'byte_code', False)
    str_143728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 62), 'str', '<')
    # Applying the binary operator '==' (line 127)
    result_eq_143729 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 49), '==', byte_code_143727, str_143728)
    
    # Processing the call keyword arguments (line 127)
    kwargs_143730 = {}
    # Getting the type of 'assert_equal' (line 127)
    assert_equal_143724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 127)
    assert_equal_call_result_143731 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), assert_equal_143724, *[little_endian_143726, result_eq_143729], **kwargs_143730)
    
    
    # Call to assert_equal(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'c_reader' (line 128)
    c_reader_143733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'c_reader', False)
    # Obtaining the member 'is_swapped' of a type (line 128)
    is_swapped_143734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), c_reader_143733, 'is_swapped')
    
    # Getting the type of 'byte_code' (line 128)
    byte_code_143735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'byte_code', False)
    # Getting the type of 'boc' (line 128)
    boc_143736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'boc', False)
    # Obtaining the member 'native_code' of a type (line 128)
    native_code_143737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 59), boc_143736, 'native_code')
    # Applying the binary operator '!=' (line 128)
    result_ne_143738 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 46), '!=', byte_code_143735, native_code_143737)
    
    # Processing the call keyword arguments (line 128)
    kwargs_143739 = {}
    # Getting the type of 'assert_equal' (line 128)
    assert_equal_143732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 128)
    assert_equal_call_result_143740 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), assert_equal_143732, *[is_swapped_143734, result_ne_143738], **kwargs_143739)
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 129)
    tuple_143741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 129)
    # Adding element type (line 129)
    # Getting the type of 'False' (line 129)
    False_143742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 26), tuple_143741, False_143742)
    # Adding element type (line 129)
    # Getting the type of 'True' (line 129)
    True_143743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 26), tuple_143741, True_143743)
    
    # Testing the type of a for loop iterable (line 129)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 12), tuple_143741)
    # Getting the type of the for loop variable (line 129)
    for_loop_var_143744 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 12), tuple_143741)
    # Assigning a type to the variable 'sde_f' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'sde_f', for_loop_var_143744)
    # SSA begins for a for statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 130):
    
    # Call to newbyteorder(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'byte_code' (line 130)
    byte_code_143751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 52), 'byte_code', False)
    # Processing the call keyword arguments (line 130)
    kwargs_143752 = {}
    
    # Call to dtype(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'base_dt' (line 130)
    base_dt_143747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'base_dt', False)
    # Processing the call keyword arguments (line 130)
    kwargs_143748 = {}
    # Getting the type of 'np' (line 130)
    np_143745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'np', False)
    # Obtaining the member 'dtype' of a type (line 130)
    dtype_143746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 21), np_143745, 'dtype')
    # Calling dtype(args, kwargs) (line 130)
    dtype_call_result_143749 = invoke(stypy.reporting.localization.Localization(__file__, 130, 21), dtype_143746, *[base_dt_143747], **kwargs_143748)
    
    # Obtaining the member 'newbyteorder' of a type (line 130)
    newbyteorder_143750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 21), dtype_call_result_143749, 'newbyteorder')
    # Calling newbyteorder(args, kwargs) (line 130)
    newbyteorder_call_result_143753 = invoke(stypy.reporting.localization.Localization(__file__, 130, 21), newbyteorder_143750, *[byte_code_143751], **kwargs_143752)
    
    # Assigning a type to the variable 'dt' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'dt', newbyteorder_call_result_143753)
    
    # Assigning a Call to a Name (line 131):
    
    # Call to _make_tag(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'dt' (line 131)
    dt_143755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'dt', False)
    # Getting the type of 'val' (line 131)
    val_143756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'val', False)
    # Getting the type of 'mdtype' (line 131)
    mdtype_143757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 39), 'mdtype', False)
    # Getting the type of 'sde_f' (line 131)
    sde_f_143758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 47), 'sde_f', False)
    # Processing the call keyword arguments (line 131)
    kwargs_143759 = {}
    # Getting the type of '_make_tag' (line 131)
    _make_tag_143754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), '_make_tag', False)
    # Calling _make_tag(args, kwargs) (line 131)
    _make_tag_call_result_143760 = invoke(stypy.reporting.localization.Localization(__file__, 131, 20), _make_tag_143754, *[dt_143755, val_143756, mdtype_143757, sde_f_143758], **kwargs_143759)
    
    # Assigning a type to the variable 'a' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'a', _make_tag_call_result_143760)
    
    # Assigning a Call to a Name (line 132):
    
    # Call to tostring(...): (line 132)
    # Processing the call keyword arguments (line 132)
    kwargs_143763 = {}
    # Getting the type of 'a' (line 132)
    a_143761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'a', False)
    # Obtaining the member 'tostring' of a type (line 132)
    tostring_143762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 24), a_143761, 'tostring')
    # Calling tostring(args, kwargs) (line 132)
    tostring_call_result_143764 = invoke(stypy.reporting.localization.Localization(__file__, 132, 24), tostring_143762, *[], **kwargs_143763)
    
    # Assigning a type to the variable 'a_str' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'a_str', tostring_call_result_143764)
    
    # Call to _write_stream(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'str_io' (line 133)
    str_io_143766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), 'str_io', False)
    # Getting the type of 'a_str' (line 133)
    a_str_143767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'a_str', False)
    # Processing the call keyword arguments (line 133)
    kwargs_143768 = {}
    # Getting the type of '_write_stream' (line 133)
    _write_stream_143765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), '_write_stream', False)
    # Calling _write_stream(args, kwargs) (line 133)
    _write_stream_call_result_143769 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), _write_stream_143765, *[str_io_143766, a_str_143767], **kwargs_143768)
    
    
    # Assigning a Call to a Name (line 134):
    
    # Call to read_numeric(...): (line 134)
    # Processing the call keyword arguments (line 134)
    kwargs_143772 = {}
    # Getting the type of 'c_reader' (line 134)
    c_reader_143770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'c_reader', False)
    # Obtaining the member 'read_numeric' of a type (line 134)
    read_numeric_143771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 21), c_reader_143770, 'read_numeric')
    # Calling read_numeric(args, kwargs) (line 134)
    read_numeric_call_result_143773 = invoke(stypy.reporting.localization.Localization(__file__, 134, 21), read_numeric_143771, *[], **kwargs_143772)
    
    # Assigning a type to the variable 'el' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'el', read_numeric_call_result_143773)
    
    # Call to assert_equal(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'el' (line 135)
    el_143775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'el', False)
    # Getting the type of 'val' (line 135)
    val_143776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'val', False)
    # Processing the call keyword arguments (line 135)
    kwargs_143777 = {}
    # Getting the type of 'assert_equal' (line 135)
    assert_equal_143774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 135)
    assert_equal_call_result_143778 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), assert_equal_143774, *[el_143775, val_143776], **kwargs_143777)
    
    
    # Call to _write_stream(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'str_io' (line 137)
    str_io_143780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'str_io', False)
    # Getting the type of 'a_str' (line 137)
    a_str_143781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'a_str', False)
    # Getting the type of 'a_str' (line 137)
    a_str_143782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 45), 'a_str', False)
    # Processing the call keyword arguments (line 137)
    kwargs_143783 = {}
    # Getting the type of '_write_stream' (line 137)
    _write_stream_143779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), '_write_stream', False)
    # Calling _write_stream(args, kwargs) (line 137)
    _write_stream_call_result_143784 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), _write_stream_143779, *[str_io_143780, a_str_143781, a_str_143782], **kwargs_143783)
    
    
    # Assigning a Call to a Name (line 138):
    
    # Call to read_numeric(...): (line 138)
    # Processing the call keyword arguments (line 138)
    kwargs_143787 = {}
    # Getting the type of 'c_reader' (line 138)
    c_reader_143785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'c_reader', False)
    # Obtaining the member 'read_numeric' of a type (line 138)
    read_numeric_143786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), c_reader_143785, 'read_numeric')
    # Calling read_numeric(args, kwargs) (line 138)
    read_numeric_call_result_143788 = invoke(stypy.reporting.localization.Localization(__file__, 138, 21), read_numeric_143786, *[], **kwargs_143787)
    
    # Assigning a type to the variable 'el' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'el', read_numeric_call_result_143788)
    
    # Call to assert_equal(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'el' (line 139)
    el_143790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'el', False)
    # Getting the type of 'val' (line 139)
    val_143791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'val', False)
    # Processing the call keyword arguments (line 139)
    kwargs_143792 = {}
    # Getting the type of 'assert_equal' (line 139)
    assert_equal_143789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 139)
    assert_equal_call_result_143793 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), assert_equal_143789, *[el_143790, val_143791], **kwargs_143792)
    
    
    # Assigning a Call to a Name (line 140):
    
    # Call to read_numeric(...): (line 140)
    # Processing the call keyword arguments (line 140)
    kwargs_143796 = {}
    # Getting the type of 'c_reader' (line 140)
    c_reader_143794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'c_reader', False)
    # Obtaining the member 'read_numeric' of a type (line 140)
    read_numeric_143795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), c_reader_143794, 'read_numeric')
    # Calling read_numeric(args, kwargs) (line 140)
    read_numeric_call_result_143797 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), read_numeric_143795, *[], **kwargs_143796)
    
    # Assigning a type to the variable 'el' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'el', read_numeric_call_result_143797)
    
    # Call to assert_equal(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'el' (line 141)
    el_143799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'el', False)
    # Getting the type of 'val' (line 141)
    val_143800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'val', False)
    # Processing the call keyword arguments (line 141)
    kwargs_143801 = {}
    # Getting the type of 'assert_equal' (line 141)
    assert_equal_143798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 141)
    assert_equal_call_result_143802 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), assert_equal_143798, *[el_143799, val_143800], **kwargs_143801)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_numeric(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_numeric' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_143803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143803)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_numeric'
    return stypy_return_type_143803

# Assigning a type to the variable 'test_read_numeric' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'test_read_numeric', test_read_numeric)

@norecursion
def test_read_numeric_writeable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_numeric_writeable'
    module_type_store = module_type_store.open_function_context('test_read_numeric_writeable', 144, 0, False)
    
    # Passed parameters checking function
    test_read_numeric_writeable.stypy_localization = localization
    test_read_numeric_writeable.stypy_type_of_self = None
    test_read_numeric_writeable.stypy_type_store = module_type_store
    test_read_numeric_writeable.stypy_function_name = 'test_read_numeric_writeable'
    test_read_numeric_writeable.stypy_param_names_list = []
    test_read_numeric_writeable.stypy_varargs_param_name = None
    test_read_numeric_writeable.stypy_kwargs_param_name = None
    test_read_numeric_writeable.stypy_call_defaults = defaults
    test_read_numeric_writeable.stypy_call_varargs = varargs
    test_read_numeric_writeable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_numeric_writeable', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_numeric_writeable', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_numeric_writeable(...)' code ##################

    
    # Assigning a Call to a Name (line 146):
    
    # Call to cStringIO(...): (line 146)
    # Processing the call keyword arguments (line 146)
    kwargs_143805 = {}
    # Getting the type of 'cStringIO' (line 146)
    cStringIO_143804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), 'cStringIO', False)
    # Calling cStringIO(args, kwargs) (line 146)
    cStringIO_call_result_143806 = invoke(stypy.reporting.localization.Localization(__file__, 146, 13), cStringIO_143804, *[], **kwargs_143805)
    
    # Assigning a type to the variable 'str_io' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'str_io', cStringIO_call_result_143806)
    
    # Assigning a Call to a Name (line 147):
    
    # Call to _make_readerlike(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'str_io' (line 147)
    str_io_143808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'str_io', False)
    str_143809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'str', '<')
    # Processing the call keyword arguments (line 147)
    kwargs_143810 = {}
    # Getting the type of '_make_readerlike' (line 147)
    _make_readerlike_143807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), '_make_readerlike', False)
    # Calling _make_readerlike(args, kwargs) (line 147)
    _make_readerlike_call_result_143811 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), _make_readerlike_143807, *[str_io_143808, str_143809], **kwargs_143810)
    
    # Assigning a type to the variable 'r' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'r', _make_readerlike_call_result_143811)
    
    # Assigning a Call to a Name (line 148):
    
    # Call to VarReader5(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'r' (line 148)
    r_143814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'r', False)
    # Processing the call keyword arguments (line 148)
    kwargs_143815 = {}
    # Getting the type of 'm5u' (line 148)
    m5u_143812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'm5u', False)
    # Obtaining the member 'VarReader5' of a type (line 148)
    VarReader5_143813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), m5u_143812, 'VarReader5')
    # Calling VarReader5(args, kwargs) (line 148)
    VarReader5_call_result_143816 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), VarReader5_143813, *[r_143814], **kwargs_143815)
    
    # Assigning a type to the variable 'c_reader' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'c_reader', VarReader5_call_result_143816)
    
    # Assigning a Call to a Name (line 149):
    
    # Call to dtype(...): (line 149)
    # Processing the call arguments (line 149)
    str_143819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 18), 'str', '<u2')
    # Processing the call keyword arguments (line 149)
    kwargs_143820 = {}
    # Getting the type of 'np' (line 149)
    np_143817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 9), 'np', False)
    # Obtaining the member 'dtype' of a type (line 149)
    dtype_143818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 9), np_143817, 'dtype')
    # Calling dtype(args, kwargs) (line 149)
    dtype_call_result_143821 = invoke(stypy.reporting.localization.Localization(__file__, 149, 9), dtype_143818, *[str_143819], **kwargs_143820)
    
    # Assigning a type to the variable 'dt' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'dt', dtype_call_result_143821)
    
    # Assigning a Call to a Name (line 150):
    
    # Call to _make_tag(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'dt' (line 150)
    dt_143823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'dt', False)
    int_143824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 22), 'int')
    # Getting the type of 'mio5p' (line 150)
    mio5p_143825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'mio5p', False)
    # Obtaining the member 'miUINT16' of a type (line 150)
    miUINT16_143826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 26), mio5p_143825, 'miUINT16')
    int_143827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 42), 'int')
    # Processing the call keyword arguments (line 150)
    kwargs_143828 = {}
    # Getting the type of '_make_tag' (line 150)
    _make_tag_143822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), '_make_tag', False)
    # Calling _make_tag(args, kwargs) (line 150)
    _make_tag_call_result_143829 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), _make_tag_143822, *[dt_143823, int_143824, miUINT16_143826, int_143827], **kwargs_143828)
    
    # Assigning a type to the variable 'a' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'a', _make_tag_call_result_143829)
    
    # Assigning a Call to a Name (line 151):
    
    # Call to tostring(...): (line 151)
    # Processing the call keyword arguments (line 151)
    kwargs_143832 = {}
    # Getting the type of 'a' (line 151)
    a_143830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'a', False)
    # Obtaining the member 'tostring' of a type (line 151)
    tostring_143831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), a_143830, 'tostring')
    # Calling tostring(args, kwargs) (line 151)
    tostring_call_result_143833 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), tostring_143831, *[], **kwargs_143832)
    
    # Assigning a type to the variable 'a_str' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'a_str', tostring_call_result_143833)
    
    # Call to _write_stream(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'str_io' (line 152)
    str_io_143835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'str_io', False)
    # Getting the type of 'a_str' (line 152)
    a_str_143836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 26), 'a_str', False)
    # Processing the call keyword arguments (line 152)
    kwargs_143837 = {}
    # Getting the type of '_write_stream' (line 152)
    _write_stream_143834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), '_write_stream', False)
    # Calling _write_stream(args, kwargs) (line 152)
    _write_stream_call_result_143838 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), _write_stream_143834, *[str_io_143835, a_str_143836], **kwargs_143837)
    
    
    # Assigning a Call to a Name (line 153):
    
    # Call to read_numeric(...): (line 153)
    # Processing the call keyword arguments (line 153)
    kwargs_143841 = {}
    # Getting the type of 'c_reader' (line 153)
    c_reader_143839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 9), 'c_reader', False)
    # Obtaining the member 'read_numeric' of a type (line 153)
    read_numeric_143840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 9), c_reader_143839, 'read_numeric')
    # Calling read_numeric(args, kwargs) (line 153)
    read_numeric_call_result_143842 = invoke(stypy.reporting.localization.Localization(__file__, 153, 9), read_numeric_143840, *[], **kwargs_143841)
    
    # Assigning a type to the variable 'el' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'el', read_numeric_call_result_143842)
    
    # Call to assert_(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Getting the type of 'el' (line 154)
    el_143844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'el', False)
    # Obtaining the member 'flags' of a type (line 154)
    flags_143845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), el_143844, 'flags')
    # Obtaining the member 'writeable' of a type (line 154)
    writeable_143846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), flags_143845, 'writeable')
    # Getting the type of 'True' (line 154)
    True_143847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'True', False)
    # Applying the binary operator 'is' (line 154)
    result_is__143848 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 12), 'is', writeable_143846, True_143847)
    
    # Processing the call keyword arguments (line 154)
    kwargs_143849 = {}
    # Getting the type of 'assert_' (line 154)
    assert__143843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 154)
    assert__call_result_143850 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), assert__143843, *[result_is__143848], **kwargs_143849)
    
    
    # ################# End of 'test_read_numeric_writeable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_numeric_writeable' in the type store
    # Getting the type of 'stypy_return_type' (line 144)
    stypy_return_type_143851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143851)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_numeric_writeable'
    return stypy_return_type_143851

# Assigning a type to the variable 'test_read_numeric_writeable' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'test_read_numeric_writeable', test_read_numeric_writeable)

@norecursion
def test_zero_byte_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_zero_byte_string'
    module_type_store = module_type_store.open_function_context('test_zero_byte_string', 157, 0, False)
    
    # Passed parameters checking function
    test_zero_byte_string.stypy_localization = localization
    test_zero_byte_string.stypy_type_of_self = None
    test_zero_byte_string.stypy_type_store = module_type_store
    test_zero_byte_string.stypy_function_name = 'test_zero_byte_string'
    test_zero_byte_string.stypy_param_names_list = []
    test_zero_byte_string.stypy_varargs_param_name = None
    test_zero_byte_string.stypy_kwargs_param_name = None
    test_zero_byte_string.stypy_call_defaults = defaults
    test_zero_byte_string.stypy_call_varargs = varargs
    test_zero_byte_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_zero_byte_string', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_zero_byte_string', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_zero_byte_string(...)' code ##################

    
    # Assigning a Call to a Name (line 160):
    
    # Call to cStringIO(...): (line 160)
    # Processing the call keyword arguments (line 160)
    kwargs_143853 = {}
    # Getting the type of 'cStringIO' (line 160)
    cStringIO_143852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'cStringIO', False)
    # Calling cStringIO(args, kwargs) (line 160)
    cStringIO_call_result_143854 = invoke(stypy.reporting.localization.Localization(__file__, 160, 13), cStringIO_143852, *[], **kwargs_143853)
    
    # Assigning a type to the variable 'str_io' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'str_io', cStringIO_call_result_143854)
    
    # Assigning a Call to a Name (line 161):
    
    # Call to _make_readerlike(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'str_io' (line 161)
    str_io_143856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'str_io', False)
    # Getting the type of 'boc' (line 161)
    boc_143857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 33), 'boc', False)
    # Obtaining the member 'native_code' of a type (line 161)
    native_code_143858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 33), boc_143857, 'native_code')
    # Processing the call keyword arguments (line 161)
    kwargs_143859 = {}
    # Getting the type of '_make_readerlike' (line 161)
    _make_readerlike_143855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), '_make_readerlike', False)
    # Calling _make_readerlike(args, kwargs) (line 161)
    _make_readerlike_call_result_143860 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), _make_readerlike_143855, *[str_io_143856, native_code_143858], **kwargs_143859)
    
    # Assigning a type to the variable 'r' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'r', _make_readerlike_call_result_143860)
    
    # Assigning a Call to a Name (line 162):
    
    # Call to VarReader5(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'r' (line 162)
    r_143863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'r', False)
    # Processing the call keyword arguments (line 162)
    kwargs_143864 = {}
    # Getting the type of 'm5u' (line 162)
    m5u_143861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'm5u', False)
    # Obtaining the member 'VarReader5' of a type (line 162)
    VarReader5_143862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 15), m5u_143861, 'VarReader5')
    # Calling VarReader5(args, kwargs) (line 162)
    VarReader5_call_result_143865 = invoke(stypy.reporting.localization.Localization(__file__, 162, 15), VarReader5_143862, *[r_143863], **kwargs_143864)
    
    # Assigning a type to the variable 'c_reader' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'c_reader', VarReader5_call_result_143865)
    
    # Assigning a Call to a Name (line 163):
    
    # Call to dtype(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Obtaining an instance of the builtin type 'list' (line 163)
    list_143868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 163)
    # Adding element type (line 163)
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_143869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    str_143870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 24), 'str', 'mdtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 24), tuple_143869, str_143870)
    # Adding element type (line 163)
    str_143871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 34), 'str', 'u4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 24), tuple_143869, str_143871)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 22), list_143868, tuple_143869)
    # Adding element type (line 163)
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_143872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    str_143873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 42), 'str', 'byte_count')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 42), tuple_143872, str_143873)
    # Adding element type (line 163)
    str_143874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 56), 'str', 'u4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 42), tuple_143872, str_143874)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 22), list_143868, tuple_143872)
    
    # Processing the call keyword arguments (line 163)
    kwargs_143875 = {}
    # Getting the type of 'np' (line 163)
    np_143866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 13), 'np', False)
    # Obtaining the member 'dtype' of a type (line 163)
    dtype_143867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 13), np_143866, 'dtype')
    # Calling dtype(args, kwargs) (line 163)
    dtype_call_result_143876 = invoke(stypy.reporting.localization.Localization(__file__, 163, 13), dtype_143867, *[list_143868], **kwargs_143875)
    
    # Assigning a type to the variable 'tag_dt' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'tag_dt', dtype_call_result_143876)
    
    # Assigning a Call to a Name (line 164):
    
    # Call to zeros(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_143879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    int_143880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 20), tuple_143879, int_143880)
    
    # Processing the call keyword arguments (line 164)
    # Getting the type of 'tag_dt' (line 164)
    tag_dt_143881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 31), 'tag_dt', False)
    keyword_143882 = tag_dt_143881
    kwargs_143883 = {'dtype': keyword_143882}
    # Getting the type of 'np' (line 164)
    np_143877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 164)
    zeros_143878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 10), np_143877, 'zeros')
    # Calling zeros(args, kwargs) (line 164)
    zeros_call_result_143884 = invoke(stypy.reporting.localization.Localization(__file__, 164, 10), zeros_143878, *[tuple_143879], **kwargs_143883)
    
    # Assigning a type to the variable 'tag' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'tag', zeros_call_result_143884)
    
    # Assigning a Attribute to a Subscript (line 165):
    # Getting the type of 'mio5p' (line 165)
    mio5p_143885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'mio5p')
    # Obtaining the member 'miINT8' of a type (line 165)
    miINT8_143886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 20), mio5p_143885, 'miINT8')
    # Getting the type of 'tag' (line 165)
    tag_143887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tag')
    str_143888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'str', 'mdtype')
    # Storing an element on a container (line 165)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 4), tag_143887, (str_143888, miINT8_143886))
    
    # Assigning a Num to a Subscript (line 166):
    int_143889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 24), 'int')
    # Getting the type of 'tag' (line 166)
    tag_143890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'tag')
    str_143891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'str', 'byte_count')
    # Storing an element on a container (line 166)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 4), tag_143890, (str_143891, int_143889))
    
    # Assigning a Call to a Name (line 167):
    
    # Call to VarHeader5(...): (line 167)
    # Processing the call keyword arguments (line 167)
    kwargs_143894 = {}
    # Getting the type of 'm5u' (line 167)
    m5u_143892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 10), 'm5u', False)
    # Obtaining the member 'VarHeader5' of a type (line 167)
    VarHeader5_143893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 10), m5u_143892, 'VarHeader5')
    # Calling VarHeader5(args, kwargs) (line 167)
    VarHeader5_call_result_143895 = invoke(stypy.reporting.localization.Localization(__file__, 167, 10), VarHeader5_143893, *[], **kwargs_143894)
    
    # Assigning a type to the variable 'hdr' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'hdr', VarHeader5_call_result_143895)
    
    # Call to set_dims(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Obtaining an instance of the builtin type 'list' (line 169)
    list_143898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 169)
    # Adding element type (line 169)
    int_143899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 17), list_143898, int_143899)
    
    # Processing the call keyword arguments (line 169)
    kwargs_143900 = {}
    # Getting the type of 'hdr' (line 169)
    hdr_143896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'hdr', False)
    # Obtaining the member 'set_dims' of a type (line 169)
    set_dims_143897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 4), hdr_143896, 'set_dims')
    # Calling set_dims(args, kwargs) (line 169)
    set_dims_call_result_143901 = invoke(stypy.reporting.localization.Localization(__file__, 169, 4), set_dims_143897, *[list_143898], **kwargs_143900)
    
    
    # Call to _write_stream(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'str_io' (line 170)
    str_io_143903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'str_io', False)
    
    # Call to tostring(...): (line 170)
    # Processing the call keyword arguments (line 170)
    kwargs_143906 = {}
    # Getting the type of 'tag' (line 170)
    tag_143904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'tag', False)
    # Obtaining the member 'tostring' of a type (line 170)
    tostring_143905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 26), tag_143904, 'tostring')
    # Calling tostring(args, kwargs) (line 170)
    tostring_call_result_143907 = invoke(stypy.reporting.localization.Localization(__file__, 170, 26), tostring_143905, *[], **kwargs_143906)
    
    str_143908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 43), 'str', '        ')
    # Applying the binary operator '+' (line 170)
    result_add_143909 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 26), '+', tostring_call_result_143907, str_143908)
    
    # Processing the call keyword arguments (line 170)
    kwargs_143910 = {}
    # Getting the type of '_write_stream' (line 170)
    _write_stream_143902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), '_write_stream', False)
    # Calling _write_stream(args, kwargs) (line 170)
    _write_stream_call_result_143911 = invoke(stypy.reporting.localization.Localization(__file__, 170, 4), _write_stream_143902, *[str_io_143903, result_add_143909], **kwargs_143910)
    
    
    # Call to seek(...): (line 171)
    # Processing the call arguments (line 171)
    int_143914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 16), 'int')
    # Processing the call keyword arguments (line 171)
    kwargs_143915 = {}
    # Getting the type of 'str_io' (line 171)
    str_io_143912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'str_io', False)
    # Obtaining the member 'seek' of a type (line 171)
    seek_143913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 4), str_io_143912, 'seek')
    # Calling seek(args, kwargs) (line 171)
    seek_call_result_143916 = invoke(stypy.reporting.localization.Localization(__file__, 171, 4), seek_143913, *[int_143914], **kwargs_143915)
    
    
    # Assigning a Call to a Name (line 172):
    
    # Call to read_char(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'hdr' (line 172)
    hdr_143919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 29), 'hdr', False)
    # Processing the call keyword arguments (line 172)
    kwargs_143920 = {}
    # Getting the type of 'c_reader' (line 172)
    c_reader_143917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 10), 'c_reader', False)
    # Obtaining the member 'read_char' of a type (line 172)
    read_char_143918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 10), c_reader_143917, 'read_char')
    # Calling read_char(args, kwargs) (line 172)
    read_char_call_result_143921 = invoke(stypy.reporting.localization.Localization(__file__, 172, 10), read_char_143918, *[hdr_143919], **kwargs_143920)
    
    # Assigning a type to the variable 'val' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'val', read_char_call_result_143921)
    
    # Call to assert_equal(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'val' (line 173)
    val_143923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'val', False)
    
    # Call to u(...): (line 173)
    # Processing the call arguments (line 173)
    str_143925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 24), 'str', ' ')
    # Processing the call keyword arguments (line 173)
    kwargs_143926 = {}
    # Getting the type of 'u' (line 173)
    u_143924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'u', False)
    # Calling u(args, kwargs) (line 173)
    u_call_result_143927 = invoke(stypy.reporting.localization.Localization(__file__, 173, 22), u_143924, *[str_143925], **kwargs_143926)
    
    # Processing the call keyword arguments (line 173)
    kwargs_143928 = {}
    # Getting the type of 'assert_equal' (line 173)
    assert_equal_143922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 173)
    assert_equal_call_result_143929 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), assert_equal_143922, *[val_143923, u_call_result_143927], **kwargs_143928)
    
    
    # Assigning a Num to a Subscript (line 175):
    int_143930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'int')
    # Getting the type of 'tag' (line 175)
    tag_143931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'tag')
    str_143932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 8), 'str', 'byte_count')
    # Storing an element on a container (line 175)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tag_143931, (str_143932, int_143930))
    
    # Call to _write_stream(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'str_io' (line 176)
    str_io_143934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'str_io', False)
    
    # Call to tostring(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_143937 = {}
    # Getting the type of 'tag' (line 176)
    tag_143935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'tag', False)
    # Obtaining the member 'tostring' of a type (line 176)
    tostring_143936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 26), tag_143935, 'tostring')
    # Calling tostring(args, kwargs) (line 176)
    tostring_call_result_143938 = invoke(stypy.reporting.localization.Localization(__file__, 176, 26), tostring_143936, *[], **kwargs_143937)
    
    # Processing the call keyword arguments (line 176)
    kwargs_143939 = {}
    # Getting the type of '_write_stream' (line 176)
    _write_stream_143933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), '_write_stream', False)
    # Calling _write_stream(args, kwargs) (line 176)
    _write_stream_call_result_143940 = invoke(stypy.reporting.localization.Localization(__file__, 176, 4), _write_stream_143933, *[str_io_143934, tostring_call_result_143938], **kwargs_143939)
    
    
    # Call to seek(...): (line 177)
    # Processing the call arguments (line 177)
    int_143943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 16), 'int')
    # Processing the call keyword arguments (line 177)
    kwargs_143944 = {}
    # Getting the type of 'str_io' (line 177)
    str_io_143941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'str_io', False)
    # Obtaining the member 'seek' of a type (line 177)
    seek_143942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), str_io_143941, 'seek')
    # Calling seek(args, kwargs) (line 177)
    seek_call_result_143945 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), seek_143942, *[int_143943], **kwargs_143944)
    
    
    # Assigning a Call to a Name (line 178):
    
    # Call to read_char(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'hdr' (line 178)
    hdr_143948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 29), 'hdr', False)
    # Processing the call keyword arguments (line 178)
    kwargs_143949 = {}
    # Getting the type of 'c_reader' (line 178)
    c_reader_143946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 10), 'c_reader', False)
    # Obtaining the member 'read_char' of a type (line 178)
    read_char_143947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 10), c_reader_143946, 'read_char')
    # Calling read_char(args, kwargs) (line 178)
    read_char_call_result_143950 = invoke(stypy.reporting.localization.Localization(__file__, 178, 10), read_char_143947, *[hdr_143948], **kwargs_143949)
    
    # Assigning a type to the variable 'val' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'val', read_char_call_result_143950)
    
    # Call to assert_equal(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'val' (line 179)
    val_143952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'val', False)
    
    # Call to u(...): (line 179)
    # Processing the call arguments (line 179)
    str_143954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 24), 'str', ' ')
    # Processing the call keyword arguments (line 179)
    kwargs_143955 = {}
    # Getting the type of 'u' (line 179)
    u_143953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'u', False)
    # Calling u(args, kwargs) (line 179)
    u_call_result_143956 = invoke(stypy.reporting.localization.Localization(__file__, 179, 22), u_143953, *[str_143954], **kwargs_143955)
    
    # Processing the call keyword arguments (line 179)
    kwargs_143957 = {}
    # Getting the type of 'assert_equal' (line 179)
    assert_equal_143951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 179)
    assert_equal_call_result_143958 = invoke(stypy.reporting.localization.Localization(__file__, 179, 4), assert_equal_143951, *[val_143952, u_call_result_143956], **kwargs_143957)
    
    
    # Call to seek(...): (line 181)
    # Processing the call arguments (line 181)
    int_143961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 16), 'int')
    # Processing the call keyword arguments (line 181)
    kwargs_143962 = {}
    # Getting the type of 'str_io' (line 181)
    str_io_143959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'str_io', False)
    # Obtaining the member 'seek' of a type (line 181)
    seek_143960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 4), str_io_143959, 'seek')
    # Calling seek(args, kwargs) (line 181)
    seek_call_result_143963 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), seek_143960, *[int_143961], **kwargs_143962)
    
    
    # Call to set_dims(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_143966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    int_143967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 17), list_143966, int_143967)
    
    # Processing the call keyword arguments (line 182)
    kwargs_143968 = {}
    # Getting the type of 'hdr' (line 182)
    hdr_143964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'hdr', False)
    # Obtaining the member 'set_dims' of a type (line 182)
    set_dims_143965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 4), hdr_143964, 'set_dims')
    # Calling set_dims(args, kwargs) (line 182)
    set_dims_call_result_143969 = invoke(stypy.reporting.localization.Localization(__file__, 182, 4), set_dims_143965, *[list_143966], **kwargs_143968)
    
    
    # Assigning a Call to a Name (line 183):
    
    # Call to read_char(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'hdr' (line 183)
    hdr_143972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 29), 'hdr', False)
    # Processing the call keyword arguments (line 183)
    kwargs_143973 = {}
    # Getting the type of 'c_reader' (line 183)
    c_reader_143970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 10), 'c_reader', False)
    # Obtaining the member 'read_char' of a type (line 183)
    read_char_143971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 10), c_reader_143970, 'read_char')
    # Calling read_char(args, kwargs) (line 183)
    read_char_call_result_143974 = invoke(stypy.reporting.localization.Localization(__file__, 183, 10), read_char_143971, *[hdr_143972], **kwargs_143973)
    
    # Assigning a type to the variable 'val' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'val', read_char_call_result_143974)
    
    # Call to assert_array_equal(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'val' (line 184)
    val_143976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'val', False)
    
    # Obtaining an instance of the builtin type 'list' (line 184)
    list_143977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 184)
    # Adding element type (line 184)
    
    # Call to u(...): (line 184)
    # Processing the call arguments (line 184)
    str_143979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 31), 'str', ' ')
    # Processing the call keyword arguments (line 184)
    kwargs_143980 = {}
    # Getting the type of 'u' (line 184)
    u_143978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 29), 'u', False)
    # Calling u(args, kwargs) (line 184)
    u_call_result_143981 = invoke(stypy.reporting.localization.Localization(__file__, 184, 29), u_143978, *[str_143979], **kwargs_143980)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), list_143977, u_call_result_143981)
    
    int_143982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 39), 'int')
    # Applying the binary operator '*' (line 184)
    result_mul_143983 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 28), '*', list_143977, int_143982)
    
    # Processing the call keyword arguments (line 184)
    kwargs_143984 = {}
    # Getting the type of 'assert_array_equal' (line 184)
    assert_array_equal_143975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 184)
    assert_array_equal_call_result_143985 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), assert_array_equal_143975, *[val_143976, result_mul_143983], **kwargs_143984)
    
    
    # ################# End of 'test_zero_byte_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_zero_byte_string' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_143986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_143986)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_zero_byte_string'
    return stypy_return_type_143986

# Assigning a type to the variable 'test_zero_byte_string' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'test_zero_byte_string', test_zero_byte_string)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
