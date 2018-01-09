
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Testing
2: 
3: '''
4: 
5: from __future__ import division, print_function, absolute_import
6: 
7: import os
8: import sys
9: import zlib
10: 
11: from io import BytesIO
12: 
13: if sys.version_info[0] >= 3:
14:     cStringIO = BytesIO
15: else:
16:     from cStringIO import StringIO as cStringIO
17: 
18: from tempfile import mkstemp
19: from contextlib import contextmanager
20: 
21: import numpy as np
22: 
23: from numpy.testing import assert_, assert_equal
24: from pytest import raises as assert_raises
25: 
26: from scipy.io.matlab.streams import (make_stream,
27:     GenericStream, cStringStream, FileStream, ZlibInputStream,
28:     _read_into, _read_string)
29: 
30: 
31: @contextmanager
32: def setup_test_file():
33:     val = b'a\x00string'
34:     fd, fname = mkstemp()
35: 
36:     with os.fdopen(fd, 'wb') as fs:
37:         fs.write(val)
38:     with open(fname, 'rb') as fs:
39:         gs = BytesIO(val)
40:         cs = cStringIO(val)
41:         yield fs, gs, cs
42:     os.unlink(fname)
43: 
44: 
45: def test_make_stream():
46:     with setup_test_file() as (fs, gs, cs):
47:         # test stream initialization
48:         assert_(isinstance(make_stream(gs), GenericStream))
49:         if sys.version_info[0] < 3:
50:             assert_(isinstance(make_stream(cs), cStringStream))
51:             assert_(isinstance(make_stream(fs), FileStream))
52: 
53: 
54: def test_tell_seek():
55:     with setup_test_file() as (fs, gs, cs):
56:         for s in (fs, gs, cs):
57:             st = make_stream(s)
58:             res = st.seek(0)
59:             assert_equal(res, 0)
60:             assert_equal(st.tell(), 0)
61:             res = st.seek(5)
62:             assert_equal(res, 0)
63:             assert_equal(st.tell(), 5)
64:             res = st.seek(2, 1)
65:             assert_equal(res, 0)
66:             assert_equal(st.tell(), 7)
67:             res = st.seek(-2, 2)
68:             assert_equal(res, 0)
69:             assert_equal(st.tell(), 6)
70: 
71: 
72: def test_read():
73:     with setup_test_file() as (fs, gs, cs):
74:         for s in (fs, gs, cs):
75:             st = make_stream(s)
76:             st.seek(0)
77:             res = st.read(-1)
78:             assert_equal(res, b'a\x00string')
79:             st.seek(0)
80:             res = st.read(4)
81:             assert_equal(res, b'a\x00st')
82:             # read into
83:             st.seek(0)
84:             res = _read_into(st, 4)
85:             assert_equal(res, b'a\x00st')
86:             res = _read_into(st, 4)
87:             assert_equal(res, b'ring')
88:             assert_raises(IOError, _read_into, st, 2)
89:             # read alloc
90:             st.seek(0)
91:             res = _read_string(st, 4)
92:             assert_equal(res, b'a\x00st')
93:             res = _read_string(st, 4)
94:             assert_equal(res, b'ring')
95:             assert_raises(IOError, _read_string, st, 2)
96: 
97: 
98: class TestZlibInputStream(object):
99:     def _get_data(self, size):
100:         data = np.random.randint(0, 256, size).astype(np.uint8).tostring()
101:         compressed_data = zlib.compress(data)
102:         stream = BytesIO(compressed_data)
103:         return stream, len(compressed_data), data
104: 
105:     def test_read(self):
106:         block_size = 131072
107: 
108:         SIZES = [0, 1, 10, block_size//2, block_size-1,
109:                  block_size, block_size+1, 2*block_size-1]
110: 
111:         READ_SIZES = [block_size//2, block_size-1,
112:                       block_size, block_size+1]
113: 
114:         def check(size, read_size):
115:             compressed_stream, compressed_data_len, data = self._get_data(size)
116:             stream = ZlibInputStream(compressed_stream, compressed_data_len)
117:             data2 = b''
118:             so_far = 0
119:             while True:
120:                 block = stream.read(min(read_size,
121:                                         size - so_far))
122:                 if not block:
123:                     break
124:                 so_far += len(block)
125:                 data2 += block
126:             assert_equal(data, data2)
127: 
128:         for size in SIZES:
129:             for read_size in READ_SIZES:
130:                 check(size, read_size)
131: 
132:     def test_read_max_length(self):
133:         size = 1234
134:         data = np.random.randint(0, 256, size).astype(np.uint8).tostring()
135:         compressed_data = zlib.compress(data)
136:         compressed_stream = BytesIO(compressed_data + b"abbacaca")
137:         stream = ZlibInputStream(compressed_stream, len(compressed_data))
138: 
139:         stream.read(len(data))
140:         assert_equal(compressed_stream.tell(), len(compressed_data))
141: 
142:         assert_raises(IOError, stream.read, 1)
143: 
144:     def test_seek(self):
145:         compressed_stream, compressed_data_len, data = self._get_data(1024)
146: 
147:         stream = ZlibInputStream(compressed_stream, compressed_data_len)
148: 
149:         stream.seek(123)
150:         p = 123
151:         assert_equal(stream.tell(), p)
152:         d1 = stream.read(11)
153:         assert_equal(d1, data[p:p+11])
154: 
155:         stream.seek(321, 1)
156:         p = 123+11+321
157:         assert_equal(stream.tell(), p)
158:         d2 = stream.read(21)
159:         assert_equal(d2, data[p:p+21])
160: 
161:         stream.seek(641, 0)
162:         p = 641
163:         assert_equal(stream.tell(), p)
164:         d3 = stream.read(11)
165:         assert_equal(d3, data[p:p+11])
166: 
167:         assert_raises(IOError, stream.seek, 10, 2)
168:         assert_raises(IOError, stream.seek, -1, 1)
169:         assert_raises(ValueError, stream.seek, 1, 123)
170: 
171:         stream.seek(10000, 1)
172:         assert_raises(IOError, stream.read, 12)
173: 
174:     def test_all_data_read(self):
175:         compressed_stream, compressed_data_len, data = self._get_data(1024)
176:         stream = ZlibInputStream(compressed_stream, compressed_data_len)
177:         assert_(not stream.all_data_read())
178:         stream.seek(512)
179:         assert_(not stream.all_data_read())
180:         stream.seek(1024)
181:         assert_(stream.all_data_read())
182: 
183: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' Testing\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import os' statement (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import zlib' statement (line 9)
import zlib

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'zlib', zlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from io import BytesIO' statement (line 11)
try:
    from io import BytesIO

except:
    BytesIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'io', None, module_type_store, ['BytesIO'], [BytesIO])




# Obtaining the type of the subscript
int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
# Getting the type of 'sys' (line 13)
sys_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 13)
version_info_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 3), sys_14, 'version_info')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 3), version_info_15, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 13, 3), getitem___16, int_13)

int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'int')
# Applying the binary operator '>=' (line 13)
result_ge_19 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 3), '>=', subscript_call_result_17, int_18)

# Testing the type of an if condition (line 13)
if_condition_20 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 0), result_ge_19)
# Assigning a type to the variable 'if_condition_20' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'if_condition_20', if_condition_20)
# SSA begins for if statement (line 13)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 14):

# Assigning a Name to a Name (line 14):
# Getting the type of 'BytesIO' (line 14)
BytesIO_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'BytesIO')
# Assigning a type to the variable 'cStringIO' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'cStringIO', BytesIO_21)
# SSA branch for the else part of an if statement (line 13)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 4))

# 'from cStringIO import cStringIO' statement (line 16)
try:
    from cStringIO import StringIO as cStringIO

except:
    cStringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'cStringIO', None, module_type_store, ['StringIO'], [cStringIO])
# Adding an alias
module_type_store.add_alias('cStringIO', 'StringIO')

# SSA join for if statement (line 13)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from tempfile import mkstemp' statement (line 18)
try:
    from tempfile import mkstemp

except:
    mkstemp = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'tempfile', None, module_type_store, ['mkstemp'], [mkstemp])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from contextlib import contextmanager' statement (line 19)
try:
    from contextlib import contextmanager

except:
    contextmanager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'contextlib', None, module_type_store, ['contextmanager'], [contextmanager])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import numpy' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_22 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_22) is not StypyTypeError):

    if (import_22 != 'pyd_module'):
        __import__(import_22)
        sys_modules_23 = sys.modules[import_22]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', sys_modules_23.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_22)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.testing import assert_, assert_equal' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_24 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing')

if (type(import_24) is not StypyTypeError):

    if (import_24 != 'pyd_module'):
        __import__(import_24)
        sys_modules_25 = sys.modules[import_24]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing', sys_modules_25.module_type_store, module_type_store, ['assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_25, sys_modules_25.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal'], [assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.testing', import_24)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from pytest import assert_raises' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_26 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'pytest')

if (type(import_26) is not StypyTypeError):

    if (import_26 != 'pyd_module'):
        __import__(import_26)
        sys_modules_27 = sys.modules[import_26]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'pytest', sys_modules_27.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_27, sys_modules_27.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'pytest', import_26)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.io.matlab.streams import make_stream, GenericStream, cStringStream, FileStream, ZlibInputStream, _read_into, _read_string' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_28 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.io.matlab.streams')

if (type(import_28) is not StypyTypeError):

    if (import_28 != 'pyd_module'):
        __import__(import_28)
        sys_modules_29 = sys.modules[import_28]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.io.matlab.streams', sys_modules_29.module_type_store, module_type_store, ['make_stream', 'GenericStream', 'cStringStream', 'FileStream', 'ZlibInputStream', '_read_into', '_read_string'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_29, sys_modules_29.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.streams import make_stream, GenericStream, cStringStream, FileStream, ZlibInputStream, _read_into, _read_string

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.io.matlab.streams', None, module_type_store, ['make_stream', 'GenericStream', 'cStringStream', 'FileStream', 'ZlibInputStream', '_read_into', '_read_string'], [make_stream, GenericStream, cStringStream, FileStream, ZlibInputStream, _read_into, _read_string])

else:
    # Assigning a type to the variable 'scipy.io.matlab.streams' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.io.matlab.streams', import_28)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')


@norecursion
def setup_test_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup_test_file'
    module_type_store = module_type_store.open_function_context('setup_test_file', 31, 0, False)
    
    # Passed parameters checking function
    setup_test_file.stypy_localization = localization
    setup_test_file.stypy_type_of_self = None
    setup_test_file.stypy_type_store = module_type_store
    setup_test_file.stypy_function_name = 'setup_test_file'
    setup_test_file.stypy_param_names_list = []
    setup_test_file.stypy_varargs_param_name = None
    setup_test_file.stypy_kwargs_param_name = None
    setup_test_file.stypy_call_defaults = defaults
    setup_test_file.stypy_call_varargs = varargs
    setup_test_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup_test_file', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup_test_file', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup_test_file(...)' code ##################

    
    # Assigning a Str to a Name (line 33):
    
    # Assigning a Str to a Name (line 33):
    str_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'str', 'a\x00string')
    # Assigning a type to the variable 'val' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'val', str_30)
    
    # Assigning a Call to a Tuple (line 34):
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'int')
    
    # Call to mkstemp(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_33 = {}
    # Getting the type of 'mkstemp' (line 34)
    mkstemp_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'mkstemp', False)
    # Calling mkstemp(args, kwargs) (line 34)
    mkstemp_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), mkstemp_32, *[], **kwargs_33)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), mkstemp_call_result_34, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), getitem___35, int_31)
    
    # Assigning a type to the variable 'tuple_var_assignment_1' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_1', subscript_call_result_36)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'int')
    
    # Call to mkstemp(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_39 = {}
    # Getting the type of 'mkstemp' (line 34)
    mkstemp_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'mkstemp', False)
    # Calling mkstemp(args, kwargs) (line 34)
    mkstemp_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), mkstemp_38, *[], **kwargs_39)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), mkstemp_call_result_40, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), getitem___41, int_37)
    
    # Assigning a type to the variable 'tuple_var_assignment_2' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_2', subscript_call_result_42)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_1' (line 34)
    tuple_var_assignment_1_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_1')
    # Assigning a type to the variable 'fd' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'fd', tuple_var_assignment_1_43)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_2' (line 34)
    tuple_var_assignment_2_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_var_assignment_2')
    # Assigning a type to the variable 'fname' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'fname', tuple_var_assignment_2_44)
    
    # Call to fdopen(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'fd' (line 36)
    fd_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'fd', False)
    str_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'wb')
    # Processing the call keyword arguments (line 36)
    kwargs_49 = {}
    # Getting the type of 'os' (line 36)
    os_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'os', False)
    # Obtaining the member 'fdopen' of a type (line 36)
    fdopen_46 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), os_45, 'fdopen')
    # Calling fdopen(args, kwargs) (line 36)
    fdopen_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), fdopen_46, *[fd_47, str_48], **kwargs_49)
    
    with_51 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 36, 9), fdopen_call_result_50, 'with parameter', '__enter__', '__exit__')

    if with_51:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 36)
        enter___52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), fdopen_call_result_50, '__enter__')
        with_enter_53 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), enter___52)
        # Assigning a type to the variable 'fs' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'fs', with_enter_53)
        
        # Call to write(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'val' (line 37)
        val_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'val', False)
        # Processing the call keyword arguments (line 37)
        kwargs_57 = {}
        # Getting the type of 'fs' (line 37)
        fs_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'fs', False)
        # Obtaining the member 'write' of a type (line 37)
        write_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), fs_54, 'write')
        # Calling write(args, kwargs) (line 37)
        write_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), write_55, *[val_56], **kwargs_57)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 36)
        exit___59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), fdopen_call_result_50, '__exit__')
        with_exit_60 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), exit___59, None, None, None)

    
    # Call to open(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'fname' (line 38)
    fname_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'fname', False)
    str_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'rb')
    # Processing the call keyword arguments (line 38)
    kwargs_64 = {}
    # Getting the type of 'open' (line 38)
    open_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'open', False)
    # Calling open(args, kwargs) (line 38)
    open_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), open_61, *[fname_62, str_63], **kwargs_64)
    
    with_66 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 38, 9), open_call_result_65, 'with parameter', '__enter__', '__exit__')

    if with_66:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 38)
        enter___67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), open_call_result_65, '__enter__')
        with_enter_68 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), enter___67)
        # Assigning a type to the variable 'fs' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'fs', with_enter_68)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to BytesIO(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'val' (line 39)
        val_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'val', False)
        # Processing the call keyword arguments (line 39)
        kwargs_71 = {}
        # Getting the type of 'BytesIO' (line 39)
        BytesIO_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'BytesIO', False)
        # Calling BytesIO(args, kwargs) (line 39)
        BytesIO_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 39, 13), BytesIO_69, *[val_70], **kwargs_71)
        
        # Assigning a type to the variable 'gs' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'gs', BytesIO_call_result_72)
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to cStringIO(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'val' (line 40)
        val_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'val', False)
        # Processing the call keyword arguments (line 40)
        kwargs_75 = {}
        # Getting the type of 'cStringIO' (line 40)
        cStringIO_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'cStringIO', False)
        # Calling cStringIO(args, kwargs) (line 40)
        cStringIO_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), cStringIO_73, *[val_74], **kwargs_75)
        
        # Assigning a type to the variable 'cs' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'cs', cStringIO_call_result_76)
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'fs' (line 41)
        fs_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'fs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_77, fs_78)
        # Adding element type (line 41)
        # Getting the type of 'gs' (line 41)
        gs_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'gs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_77, gs_79)
        # Adding element type (line 41)
        # Getting the type of 'cs' (line 41)
        cs_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_77, cs_80)
        
        GeneratorType_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 8), GeneratorType_81, tuple_77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', GeneratorType_81)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 38)
        exit___82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), open_call_result_65, '__exit__')
        with_exit_83 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), exit___82, None, None, None)

    
    # Call to unlink(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'fname' (line 42)
    fname_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'fname', False)
    # Processing the call keyword arguments (line 42)
    kwargs_87 = {}
    # Getting the type of 'os' (line 42)
    os_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'os', False)
    # Obtaining the member 'unlink' of a type (line 42)
    unlink_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), os_84, 'unlink')
    # Calling unlink(args, kwargs) (line 42)
    unlink_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), unlink_85, *[fname_86], **kwargs_87)
    
    
    # ################# End of 'setup_test_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup_test_file' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup_test_file'
    return stypy_return_type_89

# Assigning a type to the variable 'setup_test_file' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'setup_test_file', setup_test_file)

@norecursion
def test_make_stream(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_make_stream'
    module_type_store = module_type_store.open_function_context('test_make_stream', 45, 0, False)
    
    # Passed parameters checking function
    test_make_stream.stypy_localization = localization
    test_make_stream.stypy_type_of_self = None
    test_make_stream.stypy_type_store = module_type_store
    test_make_stream.stypy_function_name = 'test_make_stream'
    test_make_stream.stypy_param_names_list = []
    test_make_stream.stypy_varargs_param_name = None
    test_make_stream.stypy_kwargs_param_name = None
    test_make_stream.stypy_call_defaults = defaults
    test_make_stream.stypy_call_varargs = varargs
    test_make_stream.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_make_stream', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_make_stream', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_make_stream(...)' code ##################

    
    # Call to setup_test_file(...): (line 46)
    # Processing the call keyword arguments (line 46)
    kwargs_91 = {}
    # Getting the type of 'setup_test_file' (line 46)
    setup_test_file_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'setup_test_file', False)
    # Calling setup_test_file(args, kwargs) (line 46)
    setup_test_file_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 46, 9), setup_test_file_90, *[], **kwargs_91)
    
    with_93 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 46, 9), setup_test_file_call_result_92, 'with parameter', '__enter__', '__exit__')

    if with_93:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 46)
        enter___94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 9), setup_test_file_call_result_92, '__enter__')
        with_enter_95 = invoke(stypy.reporting.localization.Localization(__file__, 46, 9), enter___94)
        # Assigning a type to the variable 'fs' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'fs', with_enter_95)
        # Assigning a type to the variable 'gs' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'gs', with_enter_95)
        # Assigning a type to the variable 'cs' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'cs', with_enter_95)
        
        # Call to assert_(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to isinstance(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to make_stream(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'gs' (line 48)
        gs_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 'gs', False)
        # Processing the call keyword arguments (line 48)
        kwargs_100 = {}
        # Getting the type of 'make_stream' (line 48)
        make_stream_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'make_stream', False)
        # Calling make_stream(args, kwargs) (line 48)
        make_stream_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 48, 27), make_stream_98, *[gs_99], **kwargs_100)
        
        # Getting the type of 'GenericStream' (line 48)
        GenericStream_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 44), 'GenericStream', False)
        # Processing the call keyword arguments (line 48)
        kwargs_103 = {}
        # Getting the type of 'isinstance' (line 48)
        isinstance_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 48)
        isinstance_call_result_104 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), isinstance_97, *[make_stream_call_result_101, GenericStream_102], **kwargs_103)
        
        # Processing the call keyword arguments (line 48)
        kwargs_105 = {}
        # Getting the type of 'assert_' (line 48)
        assert__96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 48)
        assert__call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert__96, *[isinstance_call_result_104], **kwargs_105)
        
        
        
        
        # Obtaining the type of the subscript
        int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'int')
        # Getting the type of 'sys' (line 49)
        sys_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 49)
        version_info_109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), sys_108, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), version_info_109, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_111 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), getitem___110, int_107)
        
        int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 33), 'int')
        # Applying the binary operator '<' (line 49)
        result_lt_113 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), '<', subscript_call_result_111, int_112)
        
        # Testing the type of an if condition (line 49)
        if_condition_114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), result_lt_113)
        # Assigning a type to the variable 'if_condition_114' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'if_condition_114', if_condition_114)
        # SSA begins for if statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to isinstance(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to make_stream(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'cs' (line 50)
        cs_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 43), 'cs', False)
        # Processing the call keyword arguments (line 50)
        kwargs_119 = {}
        # Getting the type of 'make_stream' (line 50)
        make_stream_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'make_stream', False)
        # Calling make_stream(args, kwargs) (line 50)
        make_stream_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 50, 31), make_stream_117, *[cs_118], **kwargs_119)
        
        # Getting the type of 'cStringStream' (line 50)
        cStringStream_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 48), 'cStringStream', False)
        # Processing the call keyword arguments (line 50)
        kwargs_122 = {}
        # Getting the type of 'isinstance' (line 50)
        isinstance_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 50)
        isinstance_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 50, 20), isinstance_116, *[make_stream_call_result_120, cStringStream_121], **kwargs_122)
        
        # Processing the call keyword arguments (line 50)
        kwargs_124 = {}
        # Getting the type of 'assert_' (line 50)
        assert__115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 50)
        assert__call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), assert__115, *[isinstance_call_result_123], **kwargs_124)
        
        
        # Call to assert_(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to isinstance(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to make_stream(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'fs' (line 51)
        fs_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 43), 'fs', False)
        # Processing the call keyword arguments (line 51)
        kwargs_130 = {}
        # Getting the type of 'make_stream' (line 51)
        make_stream_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'make_stream', False)
        # Calling make_stream(args, kwargs) (line 51)
        make_stream_call_result_131 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), make_stream_128, *[fs_129], **kwargs_130)
        
        # Getting the type of 'FileStream' (line 51)
        FileStream_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'FileStream', False)
        # Processing the call keyword arguments (line 51)
        kwargs_133 = {}
        # Getting the type of 'isinstance' (line 51)
        isinstance_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 51)
        isinstance_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), isinstance_127, *[make_stream_call_result_131, FileStream_132], **kwargs_133)
        
        # Processing the call keyword arguments (line 51)
        kwargs_135 = {}
        # Getting the type of 'assert_' (line 51)
        assert__126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 51)
        assert__call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), assert__126, *[isinstance_call_result_134], **kwargs_135)
        
        # SSA join for if statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 46)
        exit___137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 9), setup_test_file_call_result_92, '__exit__')
        with_exit_138 = invoke(stypy.reporting.localization.Localization(__file__, 46, 9), exit___137, None, None, None)

    
    # ################# End of 'test_make_stream(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_make_stream' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_139)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_make_stream'
    return stypy_return_type_139

# Assigning a type to the variable 'test_make_stream' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'test_make_stream', test_make_stream)

@norecursion
def test_tell_seek(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_tell_seek'
    module_type_store = module_type_store.open_function_context('test_tell_seek', 54, 0, False)
    
    # Passed parameters checking function
    test_tell_seek.stypy_localization = localization
    test_tell_seek.stypy_type_of_self = None
    test_tell_seek.stypy_type_store = module_type_store
    test_tell_seek.stypy_function_name = 'test_tell_seek'
    test_tell_seek.stypy_param_names_list = []
    test_tell_seek.stypy_varargs_param_name = None
    test_tell_seek.stypy_kwargs_param_name = None
    test_tell_seek.stypy_call_defaults = defaults
    test_tell_seek.stypy_call_varargs = varargs
    test_tell_seek.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_tell_seek', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_tell_seek', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_tell_seek(...)' code ##################

    
    # Call to setup_test_file(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_141 = {}
    # Getting the type of 'setup_test_file' (line 55)
    setup_test_file_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'setup_test_file', False)
    # Calling setup_test_file(args, kwargs) (line 55)
    setup_test_file_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), setup_test_file_140, *[], **kwargs_141)
    
    with_143 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 55, 9), setup_test_file_call_result_142, 'with parameter', '__enter__', '__exit__')

    if with_143:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 55)
        enter___144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 9), setup_test_file_call_result_142, '__enter__')
        with_enter_145 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), enter___144)
        # Assigning a type to the variable 'fs' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'fs', with_enter_145)
        # Assigning a type to the variable 'gs' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'gs', with_enter_145)
        # Assigning a type to the variable 'cs' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'cs', with_enter_145)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        # Getting the type of 'fs' (line 56)
        fs_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'fs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), tuple_146, fs_147)
        # Adding element type (line 56)
        # Getting the type of 'gs' (line 56)
        gs_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'gs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), tuple_146, gs_148)
        # Adding element type (line 56)
        # Getting the type of 'cs' (line 56)
        cs_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), tuple_146, cs_149)
        
        # Testing the type of a for loop iterable (line 56)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 8), tuple_146)
        # Getting the type of the for loop variable (line 56)
        for_loop_var_150 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 8), tuple_146)
        # Assigning a type to the variable 's' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 's', for_loop_var_150)
        # SSA begins for a for statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to make_stream(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 's' (line 57)
        s_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 's', False)
        # Processing the call keyword arguments (line 57)
        kwargs_153 = {}
        # Getting the type of 'make_stream' (line 57)
        make_stream_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'make_stream', False)
        # Calling make_stream(args, kwargs) (line 57)
        make_stream_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), make_stream_151, *[s_152], **kwargs_153)
        
        # Assigning a type to the variable 'st' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'st', make_stream_call_result_154)
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to seek(...): (line 58)
        # Processing the call arguments (line 58)
        int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_158 = {}
        # Getting the type of 'st' (line 58)
        st_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'st', False)
        # Obtaining the member 'seek' of a type (line 58)
        seek_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 18), st_155, 'seek')
        # Calling seek(args, kwargs) (line 58)
        seek_call_result_159 = invoke(stypy.reporting.localization.Localization(__file__, 58, 18), seek_156, *[int_157], **kwargs_158)
        
        # Assigning a type to the variable 'res' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'res', seek_call_result_159)
        
        # Call to assert_equal(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'res' (line 59)
        res_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'res', False)
        int_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 30), 'int')
        # Processing the call keyword arguments (line 59)
        kwargs_163 = {}
        # Getting the type of 'assert_equal' (line 59)
        assert_equal_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 59)
        assert_equal_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), assert_equal_160, *[res_161, int_162], **kwargs_163)
        
        
        # Call to assert_equal(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to tell(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_168 = {}
        # Getting the type of 'st' (line 60)
        st_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'st', False)
        # Obtaining the member 'tell' of a type (line 60)
        tell_167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), st_166, 'tell')
        # Calling tell(args, kwargs) (line 60)
        tell_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), tell_167, *[], **kwargs_168)
        
        int_170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_171 = {}
        # Getting the type of 'assert_equal' (line 60)
        assert_equal_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 60)
        assert_equal_call_result_172 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), assert_equal_165, *[tell_call_result_169, int_170], **kwargs_171)
        
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to seek(...): (line 61)
        # Processing the call arguments (line 61)
        int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'int')
        # Processing the call keyword arguments (line 61)
        kwargs_176 = {}
        # Getting the type of 'st' (line 61)
        st_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'st', False)
        # Obtaining the member 'seek' of a type (line 61)
        seek_174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), st_173, 'seek')
        # Calling seek(args, kwargs) (line 61)
        seek_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), seek_174, *[int_175], **kwargs_176)
        
        # Assigning a type to the variable 'res' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'res', seek_call_result_177)
        
        # Call to assert_equal(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'res' (line 62)
        res_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'res', False)
        int_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'int')
        # Processing the call keyword arguments (line 62)
        kwargs_181 = {}
        # Getting the type of 'assert_equal' (line 62)
        assert_equal_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 62)
        assert_equal_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), assert_equal_178, *[res_179, int_180], **kwargs_181)
        
        
        # Call to assert_equal(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to tell(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_186 = {}
        # Getting the type of 'st' (line 63)
        st_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'st', False)
        # Obtaining the member 'tell' of a type (line 63)
        tell_185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), st_184, 'tell')
        # Calling tell(args, kwargs) (line 63)
        tell_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 63, 25), tell_185, *[], **kwargs_186)
        
        int_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 36), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_189 = {}
        # Getting the type of 'assert_equal' (line 63)
        assert_equal_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 63)
        assert_equal_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), assert_equal_183, *[tell_call_result_187, int_188], **kwargs_189)
        
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to seek(...): (line 64)
        # Processing the call arguments (line 64)
        int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'int')
        int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'int')
        # Processing the call keyword arguments (line 64)
        kwargs_195 = {}
        # Getting the type of 'st' (line 64)
        st_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'st', False)
        # Obtaining the member 'seek' of a type (line 64)
        seek_192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 18), st_191, 'seek')
        # Calling seek(args, kwargs) (line 64)
        seek_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 64, 18), seek_192, *[int_193, int_194], **kwargs_195)
        
        # Assigning a type to the variable 'res' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'res', seek_call_result_196)
        
        # Call to assert_equal(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'res' (line 65)
        res_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'res', False)
        int_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
        # Processing the call keyword arguments (line 65)
        kwargs_200 = {}
        # Getting the type of 'assert_equal' (line 65)
        assert_equal_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 65)
        assert_equal_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), assert_equal_197, *[res_198, int_199], **kwargs_200)
        
        
        # Call to assert_equal(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to tell(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_205 = {}
        # Getting the type of 'st' (line 66)
        st_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'st', False)
        # Obtaining the member 'tell' of a type (line 66)
        tell_204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), st_203, 'tell')
        # Calling tell(args, kwargs) (line 66)
        tell_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), tell_204, *[], **kwargs_205)
        
        int_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_208 = {}
        # Getting the type of 'assert_equal' (line 66)
        assert_equal_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 66)
        assert_equal_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), assert_equal_202, *[tell_call_result_206, int_207], **kwargs_208)
        
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to seek(...): (line 67)
        # Processing the call arguments (line 67)
        int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'int')
        int_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 30), 'int')
        # Processing the call keyword arguments (line 67)
        kwargs_214 = {}
        # Getting the type of 'st' (line 67)
        st_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'st', False)
        # Obtaining the member 'seek' of a type (line 67)
        seek_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), st_210, 'seek')
        # Calling seek(args, kwargs) (line 67)
        seek_call_result_215 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), seek_211, *[int_212, int_213], **kwargs_214)
        
        # Assigning a type to the variable 'res' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'res', seek_call_result_215)
        
        # Call to assert_equal(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'res' (line 68)
        res_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'res', False)
        int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 30), 'int')
        # Processing the call keyword arguments (line 68)
        kwargs_219 = {}
        # Getting the type of 'assert_equal' (line 68)
        assert_equal_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 68)
        assert_equal_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), assert_equal_216, *[res_217, int_218], **kwargs_219)
        
        
        # Call to assert_equal(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to tell(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_224 = {}
        # Getting the type of 'st' (line 69)
        st_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'st', False)
        # Obtaining the member 'tell' of a type (line 69)
        tell_223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), st_222, 'tell')
        # Calling tell(args, kwargs) (line 69)
        tell_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), tell_223, *[], **kwargs_224)
        
        int_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_227 = {}
        # Getting the type of 'assert_equal' (line 69)
        assert_equal_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 69)
        assert_equal_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), assert_equal_221, *[tell_call_result_225, int_226], **kwargs_227)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 55)
        exit___229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 9), setup_test_file_call_result_142, '__exit__')
        with_exit_230 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), exit___229, None, None, None)

    
    # ################# End of 'test_tell_seek(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_tell_seek' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_231)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_tell_seek'
    return stypy_return_type_231

# Assigning a type to the variable 'test_tell_seek' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'test_tell_seek', test_tell_seek)

@norecursion
def test_read(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read'
    module_type_store = module_type_store.open_function_context('test_read', 72, 0, False)
    
    # Passed parameters checking function
    test_read.stypy_localization = localization
    test_read.stypy_type_of_self = None
    test_read.stypy_type_store = module_type_store
    test_read.stypy_function_name = 'test_read'
    test_read.stypy_param_names_list = []
    test_read.stypy_varargs_param_name = None
    test_read.stypy_kwargs_param_name = None
    test_read.stypy_call_defaults = defaults
    test_read.stypy_call_varargs = varargs
    test_read.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read(...)' code ##################

    
    # Call to setup_test_file(...): (line 73)
    # Processing the call keyword arguments (line 73)
    kwargs_233 = {}
    # Getting the type of 'setup_test_file' (line 73)
    setup_test_file_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 9), 'setup_test_file', False)
    # Calling setup_test_file(args, kwargs) (line 73)
    setup_test_file_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 73, 9), setup_test_file_232, *[], **kwargs_233)
    
    with_235 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 73, 9), setup_test_file_call_result_234, 'with parameter', '__enter__', '__exit__')

    if with_235:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 73)
        enter___236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 9), setup_test_file_call_result_234, '__enter__')
        with_enter_237 = invoke(stypy.reporting.localization.Localization(__file__, 73, 9), enter___236)
        # Assigning a type to the variable 'fs' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 9), 'fs', with_enter_237)
        # Assigning a type to the variable 'gs' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 9), 'gs', with_enter_237)
        # Assigning a type to the variable 'cs' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 9), 'cs', with_enter_237)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        # Getting the type of 'fs' (line 74)
        fs_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'fs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 18), tuple_238, fs_239)
        # Adding element type (line 74)
        # Getting the type of 'gs' (line 74)
        gs_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'gs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 18), tuple_238, gs_240)
        # Adding element type (line 74)
        # Getting the type of 'cs' (line 74)
        cs_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'cs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 18), tuple_238, cs_241)
        
        # Testing the type of a for loop iterable (line 74)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 8), tuple_238)
        # Getting the type of the for loop variable (line 74)
        for_loop_var_242 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 8), tuple_238)
        # Assigning a type to the variable 's' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 's', for_loop_var_242)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to make_stream(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 's' (line 75)
        s_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 's', False)
        # Processing the call keyword arguments (line 75)
        kwargs_245 = {}
        # Getting the type of 'make_stream' (line 75)
        make_stream_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'make_stream', False)
        # Calling make_stream(args, kwargs) (line 75)
        make_stream_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), make_stream_243, *[s_244], **kwargs_245)
        
        # Assigning a type to the variable 'st' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'st', make_stream_call_result_246)
        
        # Call to seek(...): (line 76)
        # Processing the call arguments (line 76)
        int_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'int')
        # Processing the call keyword arguments (line 76)
        kwargs_250 = {}
        # Getting the type of 'st' (line 76)
        st_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'st', False)
        # Obtaining the member 'seek' of a type (line 76)
        seek_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), st_247, 'seek')
        # Calling seek(args, kwargs) (line 76)
        seek_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), seek_248, *[int_249], **kwargs_250)
        
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to read(...): (line 77)
        # Processing the call arguments (line 77)
        int_254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_255 = {}
        # Getting the type of 'st' (line 77)
        st_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'st', False)
        # Obtaining the member 'read' of a type (line 77)
        read_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 18), st_252, 'read')
        # Calling read(args, kwargs) (line 77)
        read_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), read_253, *[int_254], **kwargs_255)
        
        # Assigning a type to the variable 'res' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'res', read_call_result_256)
        
        # Call to assert_equal(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'res' (line 78)
        res_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'res', False)
        str_259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'str', 'a\x00string')
        # Processing the call keyword arguments (line 78)
        kwargs_260 = {}
        # Getting the type of 'assert_equal' (line 78)
        assert_equal_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 78)
        assert_equal_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), assert_equal_257, *[res_258, str_259], **kwargs_260)
        
        
        # Call to seek(...): (line 79)
        # Processing the call arguments (line 79)
        int_264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'int')
        # Processing the call keyword arguments (line 79)
        kwargs_265 = {}
        # Getting the type of 'st' (line 79)
        st_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'st', False)
        # Obtaining the member 'seek' of a type (line 79)
        seek_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), st_262, 'seek')
        # Calling seek(args, kwargs) (line 79)
        seek_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), seek_263, *[int_264], **kwargs_265)
        
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to read(...): (line 80)
        # Processing the call arguments (line 80)
        int_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 26), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_270 = {}
        # Getting the type of 'st' (line 80)
        st_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'st', False)
        # Obtaining the member 'read' of a type (line 80)
        read_268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), st_267, 'read')
        # Calling read(args, kwargs) (line 80)
        read_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 80, 18), read_268, *[int_269], **kwargs_270)
        
        # Assigning a type to the variable 'res' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'res', read_call_result_271)
        
        # Call to assert_equal(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'res' (line 81)
        res_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'res', False)
        str_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 30), 'str', 'a\x00st')
        # Processing the call keyword arguments (line 81)
        kwargs_275 = {}
        # Getting the type of 'assert_equal' (line 81)
        assert_equal_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 81)
        assert_equal_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), assert_equal_272, *[res_273, str_274], **kwargs_275)
        
        
        # Call to seek(...): (line 83)
        # Processing the call arguments (line 83)
        int_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_280 = {}
        # Getting the type of 'st' (line 83)
        st_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'st', False)
        # Obtaining the member 'seek' of a type (line 83)
        seek_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), st_277, 'seek')
        # Calling seek(args, kwargs) (line 83)
        seek_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), seek_278, *[int_279], **kwargs_280)
        
        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to _read_into(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'st' (line 84)
        st_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 29), 'st', False)
        int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 33), 'int')
        # Processing the call keyword arguments (line 84)
        kwargs_285 = {}
        # Getting the type of '_read_into' (line 84)
        _read_into_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), '_read_into', False)
        # Calling _read_into(args, kwargs) (line 84)
        _read_into_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 84, 18), _read_into_282, *[st_283, int_284], **kwargs_285)
        
        # Assigning a type to the variable 'res' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'res', _read_into_call_result_286)
        
        # Call to assert_equal(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'res' (line 85)
        res_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'res', False)
        str_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'str', 'a\x00st')
        # Processing the call keyword arguments (line 85)
        kwargs_290 = {}
        # Getting the type of 'assert_equal' (line 85)
        assert_equal_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 85)
        assert_equal_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), assert_equal_287, *[res_288, str_289], **kwargs_290)
        
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to _read_into(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'st' (line 86)
        st_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'st', False)
        int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 33), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_295 = {}
        # Getting the type of '_read_into' (line 86)
        _read_into_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), '_read_into', False)
        # Calling _read_into(args, kwargs) (line 86)
        _read_into_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), _read_into_292, *[st_293, int_294], **kwargs_295)
        
        # Assigning a type to the variable 'res' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'res', _read_into_call_result_296)
        
        # Call to assert_equal(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'res' (line 87)
        res_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'res', False)
        str_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'str', 'ring')
        # Processing the call keyword arguments (line 87)
        kwargs_300 = {}
        # Getting the type of 'assert_equal' (line 87)
        assert_equal_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 87)
        assert_equal_call_result_301 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), assert_equal_297, *[res_298, str_299], **kwargs_300)
        
        
        # Call to assert_raises(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'IOError' (line 88)
        IOError_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'IOError', False)
        # Getting the type of '_read_into' (line 88)
        _read_into_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 35), '_read_into', False)
        # Getting the type of 'st' (line 88)
        st_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 47), 'st', False)
        int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 51), 'int')
        # Processing the call keyword arguments (line 88)
        kwargs_307 = {}
        # Getting the type of 'assert_raises' (line 88)
        assert_raises_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 88)
        assert_raises_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), assert_raises_302, *[IOError_303, _read_into_304, st_305, int_306], **kwargs_307)
        
        
        # Call to seek(...): (line 90)
        # Processing the call arguments (line 90)
        int_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'int')
        # Processing the call keyword arguments (line 90)
        kwargs_312 = {}
        # Getting the type of 'st' (line 90)
        st_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'st', False)
        # Obtaining the member 'seek' of a type (line 90)
        seek_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), st_309, 'seek')
        # Calling seek(args, kwargs) (line 90)
        seek_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), seek_310, *[int_311], **kwargs_312)
        
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to _read_string(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'st' (line 91)
        st_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'st', False)
        int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 35), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_317 = {}
        # Getting the type of '_read_string' (line 91)
        _read_string_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), '_read_string', False)
        # Calling _read_string(args, kwargs) (line 91)
        _read_string_call_result_318 = invoke(stypy.reporting.localization.Localization(__file__, 91, 18), _read_string_314, *[st_315, int_316], **kwargs_317)
        
        # Assigning a type to the variable 'res' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'res', _read_string_call_result_318)
        
        # Call to assert_equal(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'res' (line 92)
        res_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'res', False)
        str_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 30), 'str', 'a\x00st')
        # Processing the call keyword arguments (line 92)
        kwargs_322 = {}
        # Getting the type of 'assert_equal' (line 92)
        assert_equal_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 92)
        assert_equal_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), assert_equal_319, *[res_320, str_321], **kwargs_322)
        
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to _read_string(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'st' (line 93)
        st_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'st', False)
        int_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 35), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_327 = {}
        # Getting the type of '_read_string' (line 93)
        _read_string_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), '_read_string', False)
        # Calling _read_string(args, kwargs) (line 93)
        _read_string_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), _read_string_324, *[st_325, int_326], **kwargs_327)
        
        # Assigning a type to the variable 'res' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'res', _read_string_call_result_328)
        
        # Call to assert_equal(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'res' (line 94)
        res_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'res', False)
        str_331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'str', 'ring')
        # Processing the call keyword arguments (line 94)
        kwargs_332 = {}
        # Getting the type of 'assert_equal' (line 94)
        assert_equal_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 94)
        assert_equal_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), assert_equal_329, *[res_330, str_331], **kwargs_332)
        
        
        # Call to assert_raises(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'IOError' (line 95)
        IOError_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'IOError', False)
        # Getting the type of '_read_string' (line 95)
        _read_string_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), '_read_string', False)
        # Getting the type of 'st' (line 95)
        st_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 49), 'st', False)
        int_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 53), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_339 = {}
        # Getting the type of 'assert_raises' (line 95)
        assert_raises_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 95)
        assert_raises_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), assert_raises_334, *[IOError_335, _read_string_336, st_337, int_338], **kwargs_339)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 73)
        exit___341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 9), setup_test_file_call_result_234, '__exit__')
        with_exit_342 = invoke(stypy.reporting.localization.Localization(__file__, 73, 9), exit___341, None, None, None)

    
    # ################# End of 'test_read(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_343)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read'
    return stypy_return_type_343

# Assigning a type to the variable 'test_read' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'test_read', test_read)
# Declaration of the 'TestZlibInputStream' class

class TestZlibInputStream(object, ):

    @norecursion
    def _get_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_data'
        module_type_store = module_type_store.open_function_context('_get_data', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_localization', localization)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_function_name', 'TestZlibInputStream._get_data')
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_param_names_list', ['size'])
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestZlibInputStream._get_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZlibInputStream._get_data', ['size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_data', localization, ['size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_data(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to tostring(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_358 = {}
        
        # Call to astype(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'np' (line 100)
        np_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 54), 'np', False)
        # Obtaining the member 'uint8' of a type (line 100)
        uint8_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 54), np_353, 'uint8')
        # Processing the call keyword arguments (line 100)
        kwargs_355 = {}
        
        # Call to randint(...): (line 100)
        # Processing the call arguments (line 100)
        int_347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 33), 'int')
        int_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 36), 'int')
        # Getting the type of 'size' (line 100)
        size_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 41), 'size', False)
        # Processing the call keyword arguments (line 100)
        kwargs_350 = {}
        # Getting the type of 'np' (line 100)
        np_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 100)
        random_345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), np_344, 'random')
        # Obtaining the member 'randint' of a type (line 100)
        randint_346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), random_345, 'randint')
        # Calling randint(args, kwargs) (line 100)
        randint_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), randint_346, *[int_347, int_348, size_349], **kwargs_350)
        
        # Obtaining the member 'astype' of a type (line 100)
        astype_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), randint_call_result_351, 'astype')
        # Calling astype(args, kwargs) (line 100)
        astype_call_result_356 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), astype_352, *[uint8_354], **kwargs_355)
        
        # Obtaining the member 'tostring' of a type (line 100)
        tostring_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), astype_call_result_356, 'tostring')
        # Calling tostring(args, kwargs) (line 100)
        tostring_call_result_359 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), tostring_357, *[], **kwargs_358)
        
        # Assigning a type to the variable 'data' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'data', tostring_call_result_359)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to compress(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'data' (line 101)
        data_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 40), 'data', False)
        # Processing the call keyword arguments (line 101)
        kwargs_363 = {}
        # Getting the type of 'zlib' (line 101)
        zlib_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'zlib', False)
        # Obtaining the member 'compress' of a type (line 101)
        compress_361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 26), zlib_360, 'compress')
        # Calling compress(args, kwargs) (line 101)
        compress_call_result_364 = invoke(stypy.reporting.localization.Localization(__file__, 101, 26), compress_361, *[data_362], **kwargs_363)
        
        # Assigning a type to the variable 'compressed_data' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'compressed_data', compress_call_result_364)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to BytesIO(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'compressed_data' (line 102)
        compressed_data_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'compressed_data', False)
        # Processing the call keyword arguments (line 102)
        kwargs_367 = {}
        # Getting the type of 'BytesIO' (line 102)
        BytesIO_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'BytesIO', False)
        # Calling BytesIO(args, kwargs) (line 102)
        BytesIO_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), BytesIO_365, *[compressed_data_366], **kwargs_367)
        
        # Assigning a type to the variable 'stream' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stream', BytesIO_call_result_368)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'stream' (line 103)
        stream_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'stream')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), tuple_369, stream_370)
        # Adding element type (line 103)
        
        # Call to len(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'compressed_data' (line 103)
        compressed_data_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'compressed_data', False)
        # Processing the call keyword arguments (line 103)
        kwargs_373 = {}
        # Getting the type of 'len' (line 103)
        len_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'len', False)
        # Calling len(args, kwargs) (line 103)
        len_call_result_374 = invoke(stypy.reporting.localization.Localization(__file__, 103, 23), len_371, *[compressed_data_372], **kwargs_373)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), tuple_369, len_call_result_374)
        # Adding element type (line 103)
        # Getting the type of 'data' (line 103)
        data_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 45), 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), tuple_369, data_375)
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', tuple_369)
        
        # ################# End of '_get_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_data' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_data'
        return stypy_return_type_376


    @norecursion
    def test_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read'
        module_type_store = module_type_store.open_function_context('test_read', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_localization', localization)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_function_name', 'TestZlibInputStream.test_read')
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_param_names_list', [])
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestZlibInputStream.test_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZlibInputStream.test_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read(...)' code ##################

        
        # Assigning a Num to a Name (line 106):
        
        # Assigning a Num to a Name (line 106):
        int_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'int')
        # Assigning a type to the variable 'block_size' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'block_size', int_377)
        
        # Assigning a List to a Name (line 108):
        
        # Assigning a List to a Name (line 108):
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, int_379)
        # Adding element type (line 108)
        int_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, int_380)
        # Adding element type (line 108)
        int_381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, int_381)
        # Adding element type (line 108)
        # Getting the type of 'block_size' (line 108)
        block_size_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'block_size')
        int_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'int')
        # Applying the binary operator '//' (line 108)
        result_floordiv_384 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 27), '//', block_size_382, int_383)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, result_floordiv_384)
        # Adding element type (line 108)
        # Getting the type of 'block_size' (line 108)
        block_size_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 42), 'block_size')
        int_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 53), 'int')
        # Applying the binary operator '-' (line 108)
        result_sub_387 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 42), '-', block_size_385, int_386)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, result_sub_387)
        # Adding element type (line 108)
        # Getting the type of 'block_size' (line 109)
        block_size_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'block_size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, block_size_388)
        # Adding element type (line 108)
        # Getting the type of 'block_size' (line 109)
        block_size_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 29), 'block_size')
        int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 40), 'int')
        # Applying the binary operator '+' (line 109)
        result_add_391 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 29), '+', block_size_389, int_390)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, result_add_391)
        # Adding element type (line 108)
        int_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 43), 'int')
        # Getting the type of 'block_size' (line 109)
        block_size_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 45), 'block_size')
        # Applying the binary operator '*' (line 109)
        result_mul_394 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 43), '*', int_392, block_size_393)
        
        int_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 56), 'int')
        # Applying the binary operator '-' (line 109)
        result_sub_396 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 43), '-', result_mul_394, int_395)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 16), list_378, result_sub_396)
        
        # Assigning a type to the variable 'SIZES' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'SIZES', list_378)
        
        # Assigning a List to a Name (line 111):
        
        # Assigning a List to a Name (line 111):
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 'block_size' (line 111)
        block_size_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'block_size')
        int_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 34), 'int')
        # Applying the binary operator '//' (line 111)
        result_floordiv_400 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 22), '//', block_size_398, int_399)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_397, result_floordiv_400)
        # Adding element type (line 111)
        # Getting the type of 'block_size' (line 111)
        block_size_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 37), 'block_size')
        int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 48), 'int')
        # Applying the binary operator '-' (line 111)
        result_sub_403 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 37), '-', block_size_401, int_402)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_397, result_sub_403)
        # Adding element type (line 111)
        # Getting the type of 'block_size' (line 112)
        block_size_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'block_size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_397, block_size_404)
        # Adding element type (line 111)
        # Getting the type of 'block_size' (line 112)
        block_size_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 34), 'block_size')
        int_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 45), 'int')
        # Applying the binary operator '+' (line 112)
        result_add_407 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 34), '+', block_size_405, int_406)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_397, result_add_407)
        
        # Assigning a type to the variable 'READ_SIZES' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'READ_SIZES', list_397)

        @norecursion
        def check(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'check'
            module_type_store = module_type_store.open_function_context('check', 114, 8, False)
            
            # Passed parameters checking function
            check.stypy_localization = localization
            check.stypy_type_of_self = None
            check.stypy_type_store = module_type_store
            check.stypy_function_name = 'check'
            check.stypy_param_names_list = ['size', 'read_size']
            check.stypy_varargs_param_name = None
            check.stypy_kwargs_param_name = None
            check.stypy_call_defaults = defaults
            check.stypy_call_varargs = varargs
            check.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'check', ['size', 'read_size'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'check', localization, ['size', 'read_size'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'check(...)' code ##################

            
            # Assigning a Call to a Tuple (line 115):
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Call to _get_data(...): (line 115)
            # Processing the call arguments (line 115)
            # Getting the type of 'size' (line 115)
            size_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 74), 'size', False)
            # Processing the call keyword arguments (line 115)
            kwargs_412 = {}
            # Getting the type of 'self' (line 115)
            self_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 59), 'self', False)
            # Obtaining the member '_get_data' of a type (line 115)
            _get_data_410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 59), self_409, '_get_data')
            # Calling _get_data(args, kwargs) (line 115)
            _get_data_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 115, 59), _get_data_410, *[size_411], **kwargs_412)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), _get_data_call_result_413, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___414, int_408)
            
            # Assigning a type to the variable 'tuple_var_assignment_3' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3', subscript_call_result_415)
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Call to _get_data(...): (line 115)
            # Processing the call arguments (line 115)
            # Getting the type of 'size' (line 115)
            size_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 74), 'size', False)
            # Processing the call keyword arguments (line 115)
            kwargs_420 = {}
            # Getting the type of 'self' (line 115)
            self_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 59), 'self', False)
            # Obtaining the member '_get_data' of a type (line 115)
            _get_data_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 59), self_417, '_get_data')
            # Calling _get_data(args, kwargs) (line 115)
            _get_data_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 115, 59), _get_data_418, *[size_419], **kwargs_420)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), _get_data_call_result_421, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_423 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___422, int_416)
            
            # Assigning a type to the variable 'tuple_var_assignment_4' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_4', subscript_call_result_423)
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Call to _get_data(...): (line 115)
            # Processing the call arguments (line 115)
            # Getting the type of 'size' (line 115)
            size_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 74), 'size', False)
            # Processing the call keyword arguments (line 115)
            kwargs_428 = {}
            # Getting the type of 'self' (line 115)
            self_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 59), 'self', False)
            # Obtaining the member '_get_data' of a type (line 115)
            _get_data_426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 59), self_425, '_get_data')
            # Calling _get_data(args, kwargs) (line 115)
            _get_data_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 115, 59), _get_data_426, *[size_427], **kwargs_428)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), _get_data_call_result_429, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___430, int_424)
            
            # Assigning a type to the variable 'tuple_var_assignment_5' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_5', subscript_call_result_431)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_3' (line 115)
            tuple_var_assignment_3_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3')
            # Assigning a type to the variable 'compressed_stream' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'compressed_stream', tuple_var_assignment_3_432)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_4' (line 115)
            tuple_var_assignment_4_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_4')
            # Assigning a type to the variable 'compressed_data_len' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), 'compressed_data_len', tuple_var_assignment_4_433)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_5' (line 115)
            tuple_var_assignment_5_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_5')
            # Assigning a type to the variable 'data' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 52), 'data', tuple_var_assignment_5_434)
            
            # Assigning a Call to a Name (line 116):
            
            # Assigning a Call to a Name (line 116):
            
            # Call to ZlibInputStream(...): (line 116)
            # Processing the call arguments (line 116)
            # Getting the type of 'compressed_stream' (line 116)
            compressed_stream_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'compressed_stream', False)
            # Getting the type of 'compressed_data_len' (line 116)
            compressed_data_len_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 56), 'compressed_data_len', False)
            # Processing the call keyword arguments (line 116)
            kwargs_438 = {}
            # Getting the type of 'ZlibInputStream' (line 116)
            ZlibInputStream_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'ZlibInputStream', False)
            # Calling ZlibInputStream(args, kwargs) (line 116)
            ZlibInputStream_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 116, 21), ZlibInputStream_435, *[compressed_stream_436, compressed_data_len_437], **kwargs_438)
            
            # Assigning a type to the variable 'stream' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'stream', ZlibInputStream_call_result_439)
            
            # Assigning a Str to a Name (line 117):
            
            # Assigning a Str to a Name (line 117):
            str_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'str', '')
            # Assigning a type to the variable 'data2' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'data2', str_440)
            
            # Assigning a Num to a Name (line 118):
            
            # Assigning a Num to a Name (line 118):
            int_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 21), 'int')
            # Assigning a type to the variable 'so_far' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'so_far', int_441)
            
            # Getting the type of 'True' (line 119)
            True_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'True')
            # Testing the type of an if condition (line 119)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), True_442)
            # SSA begins for while statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 120):
            
            # Assigning a Call to a Name (line 120):
            
            # Call to read(...): (line 120)
            # Processing the call arguments (line 120)
            
            # Call to min(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'read_size' (line 120)
            read_size_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'read_size', False)
            # Getting the type of 'size' (line 121)
            size_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'size', False)
            # Getting the type of 'so_far' (line 121)
            so_far_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 47), 'so_far', False)
            # Applying the binary operator '-' (line 121)
            result_sub_449 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 40), '-', size_447, so_far_448)
            
            # Processing the call keyword arguments (line 120)
            kwargs_450 = {}
            # Getting the type of 'min' (line 120)
            min_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 36), 'min', False)
            # Calling min(args, kwargs) (line 120)
            min_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 120, 36), min_445, *[read_size_446, result_sub_449], **kwargs_450)
            
            # Processing the call keyword arguments (line 120)
            kwargs_452 = {}
            # Getting the type of 'stream' (line 120)
            stream_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'stream', False)
            # Obtaining the member 'read' of a type (line 120)
            read_444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 24), stream_443, 'read')
            # Calling read(args, kwargs) (line 120)
            read_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 120, 24), read_444, *[min_call_result_451], **kwargs_452)
            
            # Assigning a type to the variable 'block' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'block', read_call_result_453)
            
            
            # Getting the type of 'block' (line 122)
            block_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'block')
            # Applying the 'not' unary operator (line 122)
            result_not__455 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 19), 'not', block_454)
            
            # Testing the type of an if condition (line 122)
            if_condition_456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 16), result_not__455)
            # Assigning a type to the variable 'if_condition_456' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'if_condition_456', if_condition_456)
            # SSA begins for if statement (line 122)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 122)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'so_far' (line 124)
            so_far_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'so_far')
            
            # Call to len(...): (line 124)
            # Processing the call arguments (line 124)
            # Getting the type of 'block' (line 124)
            block_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'block', False)
            # Processing the call keyword arguments (line 124)
            kwargs_460 = {}
            # Getting the type of 'len' (line 124)
            len_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'len', False)
            # Calling len(args, kwargs) (line 124)
            len_call_result_461 = invoke(stypy.reporting.localization.Localization(__file__, 124, 26), len_458, *[block_459], **kwargs_460)
            
            # Applying the binary operator '+=' (line 124)
            result_iadd_462 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), '+=', so_far_457, len_call_result_461)
            # Assigning a type to the variable 'so_far' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'so_far', result_iadd_462)
            
            
            # Getting the type of 'data2' (line 125)
            data2_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'data2')
            # Getting the type of 'block' (line 125)
            block_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'block')
            # Applying the binary operator '+=' (line 125)
            result_iadd_465 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 16), '+=', data2_463, block_464)
            # Assigning a type to the variable 'data2' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'data2', result_iadd_465)
            
            # SSA join for while statement (line 119)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to assert_equal(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'data' (line 126)
            data_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'data', False)
            # Getting the type of 'data2' (line 126)
            data2_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 'data2', False)
            # Processing the call keyword arguments (line 126)
            kwargs_469 = {}
            # Getting the type of 'assert_equal' (line 126)
            assert_equal_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 126)
            assert_equal_call_result_470 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), assert_equal_466, *[data_467, data2_468], **kwargs_469)
            
            
            # ################# End of 'check(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'check' in the type store
            # Getting the type of 'stypy_return_type' (line 114)
            stypy_return_type_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_471)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'check'
            return stypy_return_type_471

        # Assigning a type to the variable 'check' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'check', check)
        
        # Getting the type of 'SIZES' (line 128)
        SIZES_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'SIZES')
        # Testing the type of a for loop iterable (line 128)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 128, 8), SIZES_472)
        # Getting the type of the for loop variable (line 128)
        for_loop_var_473 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 128, 8), SIZES_472)
        # Assigning a type to the variable 'size' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'size', for_loop_var_473)
        # SSA begins for a for statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'READ_SIZES' (line 129)
        READ_SIZES_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 29), 'READ_SIZES')
        # Testing the type of a for loop iterable (line 129)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 12), READ_SIZES_474)
        # Getting the type of the for loop variable (line 129)
        for_loop_var_475 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 12), READ_SIZES_474)
        # Assigning a type to the variable 'read_size' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'read_size', for_loop_var_475)
        # SSA begins for a for statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to check(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'size' (line 130)
        size_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'size', False)
        # Getting the type of 'read_size' (line 130)
        read_size_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'read_size', False)
        # Processing the call keyword arguments (line 130)
        kwargs_479 = {}
        # Getting the type of 'check' (line 130)
        check_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'check', False)
        # Calling check(args, kwargs) (line 130)
        check_call_result_480 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), check_476, *[size_477, read_size_478], **kwargs_479)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_481)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read'
        return stypy_return_type_481


    @norecursion
    def test_read_max_length(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_max_length'
        module_type_store = module_type_store.open_function_context('test_read_max_length', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_localization', localization)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_function_name', 'TestZlibInputStream.test_read_max_length')
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_param_names_list', [])
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestZlibInputStream.test_read_max_length.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZlibInputStream.test_read_max_length', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_max_length', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_max_length(...)' code ##################

        
        # Assigning a Num to a Name (line 133):
        
        # Assigning a Num to a Name (line 133):
        int_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 15), 'int')
        # Assigning a type to the variable 'size' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'size', int_482)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to tostring(...): (line 134)
        # Processing the call keyword arguments (line 134)
        kwargs_497 = {}
        
        # Call to astype(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'np' (line 134)
        np_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 54), 'np', False)
        # Obtaining the member 'uint8' of a type (line 134)
        uint8_493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 54), np_492, 'uint8')
        # Processing the call keyword arguments (line 134)
        kwargs_494 = {}
        
        # Call to randint(...): (line 134)
        # Processing the call arguments (line 134)
        int_486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 33), 'int')
        int_487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'int')
        # Getting the type of 'size' (line 134)
        size_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 41), 'size', False)
        # Processing the call keyword arguments (line 134)
        kwargs_489 = {}
        # Getting the type of 'np' (line 134)
        np_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'np', False)
        # Obtaining the member 'random' of a type (line 134)
        random_484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), np_483, 'random')
        # Obtaining the member 'randint' of a type (line 134)
        randint_485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), random_484, 'randint')
        # Calling randint(args, kwargs) (line 134)
        randint_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), randint_485, *[int_486, int_487, size_488], **kwargs_489)
        
        # Obtaining the member 'astype' of a type (line 134)
        astype_491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), randint_call_result_490, 'astype')
        # Calling astype(args, kwargs) (line 134)
        astype_call_result_495 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), astype_491, *[uint8_493], **kwargs_494)
        
        # Obtaining the member 'tostring' of a type (line 134)
        tostring_496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), astype_call_result_495, 'tostring')
        # Calling tostring(args, kwargs) (line 134)
        tostring_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), tostring_496, *[], **kwargs_497)
        
        # Assigning a type to the variable 'data' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'data', tostring_call_result_498)
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to compress(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'data' (line 135)
        data_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), 'data', False)
        # Processing the call keyword arguments (line 135)
        kwargs_502 = {}
        # Getting the type of 'zlib' (line 135)
        zlib_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 26), 'zlib', False)
        # Obtaining the member 'compress' of a type (line 135)
        compress_500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 26), zlib_499, 'compress')
        # Calling compress(args, kwargs) (line 135)
        compress_call_result_503 = invoke(stypy.reporting.localization.Localization(__file__, 135, 26), compress_500, *[data_501], **kwargs_502)
        
        # Assigning a type to the variable 'compressed_data' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'compressed_data', compress_call_result_503)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to BytesIO(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'compressed_data' (line 136)
        compressed_data_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 36), 'compressed_data', False)
        str_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 54), 'str', 'abbacaca')
        # Applying the binary operator '+' (line 136)
        result_add_507 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 36), '+', compressed_data_505, str_506)
        
        # Processing the call keyword arguments (line 136)
        kwargs_508 = {}
        # Getting the type of 'BytesIO' (line 136)
        BytesIO_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'BytesIO', False)
        # Calling BytesIO(args, kwargs) (line 136)
        BytesIO_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 136, 28), BytesIO_504, *[result_add_507], **kwargs_508)
        
        # Assigning a type to the variable 'compressed_stream' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'compressed_stream', BytesIO_call_result_509)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to ZlibInputStream(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'compressed_stream' (line 137)
        compressed_stream_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'compressed_stream', False)
        
        # Call to len(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'compressed_data' (line 137)
        compressed_data_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 56), 'compressed_data', False)
        # Processing the call keyword arguments (line 137)
        kwargs_514 = {}
        # Getting the type of 'len' (line 137)
        len_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 52), 'len', False)
        # Calling len(args, kwargs) (line 137)
        len_call_result_515 = invoke(stypy.reporting.localization.Localization(__file__, 137, 52), len_512, *[compressed_data_513], **kwargs_514)
        
        # Processing the call keyword arguments (line 137)
        kwargs_516 = {}
        # Getting the type of 'ZlibInputStream' (line 137)
        ZlibInputStream_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'ZlibInputStream', False)
        # Calling ZlibInputStream(args, kwargs) (line 137)
        ZlibInputStream_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 137, 17), ZlibInputStream_510, *[compressed_stream_511, len_call_result_515], **kwargs_516)
        
        # Assigning a type to the variable 'stream' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stream', ZlibInputStream_call_result_517)
        
        # Call to read(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to len(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'data' (line 139)
        data_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'data', False)
        # Processing the call keyword arguments (line 139)
        kwargs_522 = {}
        # Getting the type of 'len' (line 139)
        len_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'len', False)
        # Calling len(args, kwargs) (line 139)
        len_call_result_523 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), len_520, *[data_521], **kwargs_522)
        
        # Processing the call keyword arguments (line 139)
        kwargs_524 = {}
        # Getting the type of 'stream' (line 139)
        stream_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stream', False)
        # Obtaining the member 'read' of a type (line 139)
        read_519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), stream_518, 'read')
        # Calling read(args, kwargs) (line 139)
        read_call_result_525 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), read_519, *[len_call_result_523], **kwargs_524)
        
        
        # Call to assert_equal(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to tell(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_529 = {}
        # Getting the type of 'compressed_stream' (line 140)
        compressed_stream_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'compressed_stream', False)
        # Obtaining the member 'tell' of a type (line 140)
        tell_528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), compressed_stream_527, 'tell')
        # Calling tell(args, kwargs) (line 140)
        tell_call_result_530 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), tell_528, *[], **kwargs_529)
        
        
        # Call to len(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'compressed_data' (line 140)
        compressed_data_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 51), 'compressed_data', False)
        # Processing the call keyword arguments (line 140)
        kwargs_533 = {}
        # Getting the type of 'len' (line 140)
        len_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'len', False)
        # Calling len(args, kwargs) (line 140)
        len_call_result_534 = invoke(stypy.reporting.localization.Localization(__file__, 140, 47), len_531, *[compressed_data_532], **kwargs_533)
        
        # Processing the call keyword arguments (line 140)
        kwargs_535 = {}
        # Getting the type of 'assert_equal' (line 140)
        assert_equal_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 140)
        assert_equal_call_result_536 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), assert_equal_526, *[tell_call_result_530, len_call_result_534], **kwargs_535)
        
        
        # Call to assert_raises(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'IOError' (line 142)
        IOError_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'IOError', False)
        # Getting the type of 'stream' (line 142)
        stream_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'stream', False)
        # Obtaining the member 'read' of a type (line 142)
        read_540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 31), stream_539, 'read')
        int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 44), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_542 = {}
        # Getting the type of 'assert_raises' (line 142)
        assert_raises_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 142)
        assert_raises_call_result_543 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert_raises_537, *[IOError_538, read_540, int_541], **kwargs_542)
        
        
        # ################# End of 'test_read_max_length(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_max_length' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_max_length'
        return stypy_return_type_544


    @norecursion
    def test_seek(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_seek'
        module_type_store = module_type_store.open_function_context('test_seek', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_localization', localization)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_function_name', 'TestZlibInputStream.test_seek')
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_param_names_list', [])
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestZlibInputStream.test_seek.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZlibInputStream.test_seek', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_seek', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_seek(...)' code ##################

        
        # Assigning a Call to a Tuple (line 145):
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        int_545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'int')
        
        # Call to _get_data(...): (line 145)
        # Processing the call arguments (line 145)
        int_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 70), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_549 = {}
        # Getting the type of 'self' (line 145)
        self_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 55), 'self', False)
        # Obtaining the member '_get_data' of a type (line 145)
        _get_data_547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 55), self_546, '_get_data')
        # Calling _get_data(args, kwargs) (line 145)
        _get_data_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 145, 55), _get_data_547, *[int_548], **kwargs_549)
        
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), _get_data_call_result_550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_552 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), getitem___551, int_545)
        
        # Assigning a type to the variable 'tuple_var_assignment_6' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'tuple_var_assignment_6', subscript_call_result_552)
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        int_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'int')
        
        # Call to _get_data(...): (line 145)
        # Processing the call arguments (line 145)
        int_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 70), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_557 = {}
        # Getting the type of 'self' (line 145)
        self_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 55), 'self', False)
        # Obtaining the member '_get_data' of a type (line 145)
        _get_data_555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 55), self_554, '_get_data')
        # Calling _get_data(args, kwargs) (line 145)
        _get_data_call_result_558 = invoke(stypy.reporting.localization.Localization(__file__, 145, 55), _get_data_555, *[int_556], **kwargs_557)
        
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), _get_data_call_result_558, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), getitem___559, int_553)
        
        # Assigning a type to the variable 'tuple_var_assignment_7' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'tuple_var_assignment_7', subscript_call_result_560)
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        int_561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'int')
        
        # Call to _get_data(...): (line 145)
        # Processing the call arguments (line 145)
        int_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 70), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_565 = {}
        # Getting the type of 'self' (line 145)
        self_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 55), 'self', False)
        # Obtaining the member '_get_data' of a type (line 145)
        _get_data_563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 55), self_562, '_get_data')
        # Calling _get_data(args, kwargs) (line 145)
        _get_data_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 145, 55), _get_data_563, *[int_564], **kwargs_565)
        
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), _get_data_call_result_566, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_568 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), getitem___567, int_561)
        
        # Assigning a type to the variable 'tuple_var_assignment_8' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'tuple_var_assignment_8', subscript_call_result_568)
        
        # Assigning a Name to a Name (line 145):
        # Getting the type of 'tuple_var_assignment_6' (line 145)
        tuple_var_assignment_6_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'tuple_var_assignment_6')
        # Assigning a type to the variable 'compressed_stream' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'compressed_stream', tuple_var_assignment_6_569)
        
        # Assigning a Name to a Name (line 145):
        # Getting the type of 'tuple_var_assignment_7' (line 145)
        tuple_var_assignment_7_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'tuple_var_assignment_7')
        # Assigning a type to the variable 'compressed_data_len' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 27), 'compressed_data_len', tuple_var_assignment_7_570)
        
        # Assigning a Name to a Name (line 145):
        # Getting the type of 'tuple_var_assignment_8' (line 145)
        tuple_var_assignment_8_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'tuple_var_assignment_8')
        # Assigning a type to the variable 'data' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 48), 'data', tuple_var_assignment_8_571)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to ZlibInputStream(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'compressed_stream' (line 147)
        compressed_stream_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'compressed_stream', False)
        # Getting the type of 'compressed_data_len' (line 147)
        compressed_data_len_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 52), 'compressed_data_len', False)
        # Processing the call keyword arguments (line 147)
        kwargs_575 = {}
        # Getting the type of 'ZlibInputStream' (line 147)
        ZlibInputStream_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'ZlibInputStream', False)
        # Calling ZlibInputStream(args, kwargs) (line 147)
        ZlibInputStream_call_result_576 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), ZlibInputStream_572, *[compressed_stream_573, compressed_data_len_574], **kwargs_575)
        
        # Assigning a type to the variable 'stream' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stream', ZlibInputStream_call_result_576)
        
        # Call to seek(...): (line 149)
        # Processing the call arguments (line 149)
        int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'int')
        # Processing the call keyword arguments (line 149)
        kwargs_580 = {}
        # Getting the type of 'stream' (line 149)
        stream_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stream', False)
        # Obtaining the member 'seek' of a type (line 149)
        seek_578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), stream_577, 'seek')
        # Calling seek(args, kwargs) (line 149)
        seek_call_result_581 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), seek_578, *[int_579], **kwargs_580)
        
        
        # Assigning a Num to a Name (line 150):
        
        # Assigning a Num to a Name (line 150):
        int_582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 12), 'int')
        # Assigning a type to the variable 'p' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'p', int_582)
        
        # Call to assert_equal(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Call to tell(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_586 = {}
        # Getting the type of 'stream' (line 151)
        stream_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'stream', False)
        # Obtaining the member 'tell' of a type (line 151)
        tell_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 21), stream_584, 'tell')
        # Calling tell(args, kwargs) (line 151)
        tell_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), tell_585, *[], **kwargs_586)
        
        # Getting the type of 'p' (line 151)
        p_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 36), 'p', False)
        # Processing the call keyword arguments (line 151)
        kwargs_589 = {}
        # Getting the type of 'assert_equal' (line 151)
        assert_equal_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 151)
        assert_equal_call_result_590 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), assert_equal_583, *[tell_call_result_587, p_588], **kwargs_589)
        
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to read(...): (line 152)
        # Processing the call arguments (line 152)
        int_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'int')
        # Processing the call keyword arguments (line 152)
        kwargs_594 = {}
        # Getting the type of 'stream' (line 152)
        stream_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'stream', False)
        # Obtaining the member 'read' of a type (line 152)
        read_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), stream_591, 'read')
        # Calling read(args, kwargs) (line 152)
        read_call_result_595 = invoke(stypy.reporting.localization.Localization(__file__, 152, 13), read_592, *[int_593], **kwargs_594)
        
        # Assigning a type to the variable 'd1' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'd1', read_call_result_595)
        
        # Call to assert_equal(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'd1' (line 153)
        d1_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'd1', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'p' (line 153)
        p_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'p', False)
        # Getting the type of 'p' (line 153)
        p_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 32), 'p', False)
        int_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 34), 'int')
        # Applying the binary operator '+' (line 153)
        result_add_601 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 32), '+', p_599, int_600)
        
        slice_602 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 153, 25), p_598, result_add_601, None)
        # Getting the type of 'data' (line 153)
        data_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 25), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 25), data_603, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_605 = invoke(stypy.reporting.localization.Localization(__file__, 153, 25), getitem___604, slice_602)
        
        # Processing the call keyword arguments (line 153)
        kwargs_606 = {}
        # Getting the type of 'assert_equal' (line 153)
        assert_equal_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 153)
        assert_equal_call_result_607 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_equal_596, *[d1_597, subscript_call_result_605], **kwargs_606)
        
        
        # Call to seek(...): (line 155)
        # Processing the call arguments (line 155)
        int_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 20), 'int')
        int_611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 25), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_612 = {}
        # Getting the type of 'stream' (line 155)
        stream_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'stream', False)
        # Obtaining the member 'seek' of a type (line 155)
        seek_609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), stream_608, 'seek')
        # Calling seek(args, kwargs) (line 155)
        seek_call_result_613 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), seek_609, *[int_610, int_611], **kwargs_612)
        
        
        # Assigning a BinOp to a Name (line 156):
        
        # Assigning a BinOp to a Name (line 156):
        int_614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 12), 'int')
        int_615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 16), 'int')
        # Applying the binary operator '+' (line 156)
        result_add_616 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 12), '+', int_614, int_615)
        
        int_617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 19), 'int')
        # Applying the binary operator '+' (line 156)
        result_add_618 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 18), '+', result_add_616, int_617)
        
        # Assigning a type to the variable 'p' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'p', result_add_618)
        
        # Call to assert_equal(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Call to tell(...): (line 157)
        # Processing the call keyword arguments (line 157)
        kwargs_622 = {}
        # Getting the type of 'stream' (line 157)
        stream_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'stream', False)
        # Obtaining the member 'tell' of a type (line 157)
        tell_621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 21), stream_620, 'tell')
        # Calling tell(args, kwargs) (line 157)
        tell_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), tell_621, *[], **kwargs_622)
        
        # Getting the type of 'p' (line 157)
        p_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'p', False)
        # Processing the call keyword arguments (line 157)
        kwargs_625 = {}
        # Getting the type of 'assert_equal' (line 157)
        assert_equal_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 157)
        assert_equal_call_result_626 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), assert_equal_619, *[tell_call_result_623, p_624], **kwargs_625)
        
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to read(...): (line 158)
        # Processing the call arguments (line 158)
        int_629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'int')
        # Processing the call keyword arguments (line 158)
        kwargs_630 = {}
        # Getting the type of 'stream' (line 158)
        stream_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), 'stream', False)
        # Obtaining the member 'read' of a type (line 158)
        read_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 13), stream_627, 'read')
        # Calling read(args, kwargs) (line 158)
        read_call_result_631 = invoke(stypy.reporting.localization.Localization(__file__, 158, 13), read_628, *[int_629], **kwargs_630)
        
        # Assigning a type to the variable 'd2' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'd2', read_call_result_631)
        
        # Call to assert_equal(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'd2' (line 159)
        d2_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'd2', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'p' (line 159)
        p_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'p', False)
        # Getting the type of 'p' (line 159)
        p_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'p', False)
        int_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 34), 'int')
        # Applying the binary operator '+' (line 159)
        result_add_637 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 32), '+', p_635, int_636)
        
        slice_638 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 25), p_634, result_add_637, None)
        # Getting the type of 'data' (line 159)
        data_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 25), data_639, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), getitem___640, slice_638)
        
        # Processing the call keyword arguments (line 159)
        kwargs_642 = {}
        # Getting the type of 'assert_equal' (line 159)
        assert_equal_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 159)
        assert_equal_call_result_643 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), assert_equal_632, *[d2_633, subscript_call_result_641], **kwargs_642)
        
        
        # Call to seek(...): (line 161)
        # Processing the call arguments (line 161)
        int_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'int')
        int_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'int')
        # Processing the call keyword arguments (line 161)
        kwargs_648 = {}
        # Getting the type of 'stream' (line 161)
        stream_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'stream', False)
        # Obtaining the member 'seek' of a type (line 161)
        seek_645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), stream_644, 'seek')
        # Calling seek(args, kwargs) (line 161)
        seek_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), seek_645, *[int_646, int_647], **kwargs_648)
        
        
        # Assigning a Num to a Name (line 162):
        
        # Assigning a Num to a Name (line 162):
        int_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 12), 'int')
        # Assigning a type to the variable 'p' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'p', int_650)
        
        # Call to assert_equal(...): (line 163)
        # Processing the call arguments (line 163)
        
        # Call to tell(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_654 = {}
        # Getting the type of 'stream' (line 163)
        stream_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'stream', False)
        # Obtaining the member 'tell' of a type (line 163)
        tell_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), stream_652, 'tell')
        # Calling tell(args, kwargs) (line 163)
        tell_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 163, 21), tell_653, *[], **kwargs_654)
        
        # Getting the type of 'p' (line 163)
        p_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), 'p', False)
        # Processing the call keyword arguments (line 163)
        kwargs_657 = {}
        # Getting the type of 'assert_equal' (line 163)
        assert_equal_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 163)
        assert_equal_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), assert_equal_651, *[tell_call_result_655, p_656], **kwargs_657)
        
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to read(...): (line 164)
        # Processing the call arguments (line 164)
        int_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'int')
        # Processing the call keyword arguments (line 164)
        kwargs_662 = {}
        # Getting the type of 'stream' (line 164)
        stream_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'stream', False)
        # Obtaining the member 'read' of a type (line 164)
        read_660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), stream_659, 'read')
        # Calling read(args, kwargs) (line 164)
        read_call_result_663 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), read_660, *[int_661], **kwargs_662)
        
        # Assigning a type to the variable 'd3' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'd3', read_call_result_663)
        
        # Call to assert_equal(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'd3' (line 165)
        d3_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'd3', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'p' (line 165)
        p_666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'p', False)
        # Getting the type of 'p' (line 165)
        p_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'p', False)
        int_668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 34), 'int')
        # Applying the binary operator '+' (line 165)
        result_add_669 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 32), '+', p_667, int_668)
        
        slice_670 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 165, 25), p_666, result_add_669, None)
        # Getting the type of 'data' (line 165)
        data_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), data_671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_673 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), getitem___672, slice_670)
        
        # Processing the call keyword arguments (line 165)
        kwargs_674 = {}
        # Getting the type of 'assert_equal' (line 165)
        assert_equal_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 165)
        assert_equal_call_result_675 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert_equal_664, *[d3_665, subscript_call_result_673], **kwargs_674)
        
        
        # Call to assert_raises(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'IOError' (line 167)
        IOError_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'IOError', False)
        # Getting the type of 'stream' (line 167)
        stream_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'stream', False)
        # Obtaining the member 'seek' of a type (line 167)
        seek_679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 31), stream_678, 'seek')
        int_680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 44), 'int')
        int_681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 48), 'int')
        # Processing the call keyword arguments (line 167)
        kwargs_682 = {}
        # Getting the type of 'assert_raises' (line 167)
        assert_raises_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 167)
        assert_raises_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert_raises_676, *[IOError_677, seek_679, int_680, int_681], **kwargs_682)
        
        
        # Call to assert_raises(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'IOError' (line 168)
        IOError_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'IOError', False)
        # Getting the type of 'stream' (line 168)
        stream_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'stream', False)
        # Obtaining the member 'seek' of a type (line 168)
        seek_687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 31), stream_686, 'seek')
        int_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 44), 'int')
        int_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 48), 'int')
        # Processing the call keyword arguments (line 168)
        kwargs_690 = {}
        # Getting the type of 'assert_raises' (line 168)
        assert_raises_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 168)
        assert_raises_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), assert_raises_684, *[IOError_685, seek_687, int_688, int_689], **kwargs_690)
        
        
        # Call to assert_raises(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'ValueError' (line 169)
        ValueError_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'ValueError', False)
        # Getting the type of 'stream' (line 169)
        stream_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'stream', False)
        # Obtaining the member 'seek' of a type (line 169)
        seek_695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), stream_694, 'seek')
        int_696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 47), 'int')
        int_697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 50), 'int')
        # Processing the call keyword arguments (line 169)
        kwargs_698 = {}
        # Getting the type of 'assert_raises' (line 169)
        assert_raises_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 169)
        assert_raises_call_result_699 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), assert_raises_692, *[ValueError_693, seek_695, int_696, int_697], **kwargs_698)
        
        
        # Call to seek(...): (line 171)
        # Processing the call arguments (line 171)
        int_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'int')
        int_703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'int')
        # Processing the call keyword arguments (line 171)
        kwargs_704 = {}
        # Getting the type of 'stream' (line 171)
        stream_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stream', False)
        # Obtaining the member 'seek' of a type (line 171)
        seek_701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), stream_700, 'seek')
        # Calling seek(args, kwargs) (line 171)
        seek_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), seek_701, *[int_702, int_703], **kwargs_704)
        
        
        # Call to assert_raises(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'IOError' (line 172)
        IOError_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'IOError', False)
        # Getting the type of 'stream' (line 172)
        stream_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 31), 'stream', False)
        # Obtaining the member 'read' of a type (line 172)
        read_709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 31), stream_708, 'read')
        int_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 44), 'int')
        # Processing the call keyword arguments (line 172)
        kwargs_711 = {}
        # Getting the type of 'assert_raises' (line 172)
        assert_raises_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 172)
        assert_raises_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assert_raises_706, *[IOError_707, read_709, int_710], **kwargs_711)
        
        
        # ################# End of 'test_seek(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_seek' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_seek'
        return stypy_return_type_713


    @norecursion
    def test_all_data_read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_all_data_read'
        module_type_store = module_type_store.open_function_context('test_all_data_read', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_localization', localization)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_function_name', 'TestZlibInputStream.test_all_data_read')
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_param_names_list', [])
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestZlibInputStream.test_all_data_read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZlibInputStream.test_all_data_read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_all_data_read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_all_data_read(...)' code ##################

        
        # Assigning a Call to a Tuple (line 175):
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        int_714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 8), 'int')
        
        # Call to _get_data(...): (line 175)
        # Processing the call arguments (line 175)
        int_717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 70), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_718 = {}
        # Getting the type of 'self' (line 175)
        self_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 55), 'self', False)
        # Obtaining the member '_get_data' of a type (line 175)
        _get_data_716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 55), self_715, '_get_data')
        # Calling _get_data(args, kwargs) (line 175)
        _get_data_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 175, 55), _get_data_716, *[int_717], **kwargs_718)
        
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), _get_data_call_result_719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_721 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), getitem___720, int_714)
        
        # Assigning a type to the variable 'tuple_var_assignment_9' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_9', subscript_call_result_721)
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        int_722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 8), 'int')
        
        # Call to _get_data(...): (line 175)
        # Processing the call arguments (line 175)
        int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 70), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_726 = {}
        # Getting the type of 'self' (line 175)
        self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 55), 'self', False)
        # Obtaining the member '_get_data' of a type (line 175)
        _get_data_724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 55), self_723, '_get_data')
        # Calling _get_data(args, kwargs) (line 175)
        _get_data_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 175, 55), _get_data_724, *[int_725], **kwargs_726)
        
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), _get_data_call_result_727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_729 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), getitem___728, int_722)
        
        # Assigning a type to the variable 'tuple_var_assignment_10' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_10', subscript_call_result_729)
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        int_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 8), 'int')
        
        # Call to _get_data(...): (line 175)
        # Processing the call arguments (line 175)
        int_733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 70), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_734 = {}
        # Getting the type of 'self' (line 175)
        self_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 55), 'self', False)
        # Obtaining the member '_get_data' of a type (line 175)
        _get_data_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 55), self_731, '_get_data')
        # Calling _get_data(args, kwargs) (line 175)
        _get_data_call_result_735 = invoke(stypy.reporting.localization.Localization(__file__, 175, 55), _get_data_732, *[int_733], **kwargs_734)
        
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), _get_data_call_result_735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_737 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), getitem___736, int_730)
        
        # Assigning a type to the variable 'tuple_var_assignment_11' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_11', subscript_call_result_737)
        
        # Assigning a Name to a Name (line 175):
        # Getting the type of 'tuple_var_assignment_9' (line 175)
        tuple_var_assignment_9_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_9')
        # Assigning a type to the variable 'compressed_stream' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'compressed_stream', tuple_var_assignment_9_738)
        
        # Assigning a Name to a Name (line 175):
        # Getting the type of 'tuple_var_assignment_10' (line 175)
        tuple_var_assignment_10_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_10')
        # Assigning a type to the variable 'compressed_data_len' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 'compressed_data_len', tuple_var_assignment_10_739)
        
        # Assigning a Name to a Name (line 175):
        # Getting the type of 'tuple_var_assignment_11' (line 175)
        tuple_var_assignment_11_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_11')
        # Assigning a type to the variable 'data' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 48), 'data', tuple_var_assignment_11_740)
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to ZlibInputStream(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'compressed_stream' (line 176)
        compressed_stream_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'compressed_stream', False)
        # Getting the type of 'compressed_data_len' (line 176)
        compressed_data_len_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'compressed_data_len', False)
        # Processing the call keyword arguments (line 176)
        kwargs_744 = {}
        # Getting the type of 'ZlibInputStream' (line 176)
        ZlibInputStream_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'ZlibInputStream', False)
        # Calling ZlibInputStream(args, kwargs) (line 176)
        ZlibInputStream_call_result_745 = invoke(stypy.reporting.localization.Localization(__file__, 176, 17), ZlibInputStream_741, *[compressed_stream_742, compressed_data_len_743], **kwargs_744)
        
        # Assigning a type to the variable 'stream' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stream', ZlibInputStream_call_result_745)
        
        # Call to assert_(...): (line 177)
        # Processing the call arguments (line 177)
        
        
        # Call to all_data_read(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_749 = {}
        # Getting the type of 'stream' (line 177)
        stream_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'stream', False)
        # Obtaining the member 'all_data_read' of a type (line 177)
        all_data_read_748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), stream_747, 'all_data_read')
        # Calling all_data_read(args, kwargs) (line 177)
        all_data_read_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 177, 20), all_data_read_748, *[], **kwargs_749)
        
        # Applying the 'not' unary operator (line 177)
        result_not__751 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 16), 'not', all_data_read_call_result_750)
        
        # Processing the call keyword arguments (line 177)
        kwargs_752 = {}
        # Getting the type of 'assert_' (line 177)
        assert__746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 177)
        assert__call_result_753 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assert__746, *[result_not__751], **kwargs_752)
        
        
        # Call to seek(...): (line 178)
        # Processing the call arguments (line 178)
        int_756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 20), 'int')
        # Processing the call keyword arguments (line 178)
        kwargs_757 = {}
        # Getting the type of 'stream' (line 178)
        stream_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stream', False)
        # Obtaining the member 'seek' of a type (line 178)
        seek_755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), stream_754, 'seek')
        # Calling seek(args, kwargs) (line 178)
        seek_call_result_758 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), seek_755, *[int_756], **kwargs_757)
        
        
        # Call to assert_(...): (line 179)
        # Processing the call arguments (line 179)
        
        
        # Call to all_data_read(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_762 = {}
        # Getting the type of 'stream' (line 179)
        stream_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'stream', False)
        # Obtaining the member 'all_data_read' of a type (line 179)
        all_data_read_761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), stream_760, 'all_data_read')
        # Calling all_data_read(args, kwargs) (line 179)
        all_data_read_call_result_763 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), all_data_read_761, *[], **kwargs_762)
        
        # Applying the 'not' unary operator (line 179)
        result_not__764 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 16), 'not', all_data_read_call_result_763)
        
        # Processing the call keyword arguments (line 179)
        kwargs_765 = {}
        # Getting the type of 'assert_' (line 179)
        assert__759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 179)
        assert__call_result_766 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), assert__759, *[result_not__764], **kwargs_765)
        
        
        # Call to seek(...): (line 180)
        # Processing the call arguments (line 180)
        int_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 20), 'int')
        # Processing the call keyword arguments (line 180)
        kwargs_770 = {}
        # Getting the type of 'stream' (line 180)
        stream_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stream', False)
        # Obtaining the member 'seek' of a type (line 180)
        seek_768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), stream_767, 'seek')
        # Calling seek(args, kwargs) (line 180)
        seek_call_result_771 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), seek_768, *[int_769], **kwargs_770)
        
        
        # Call to assert_(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to all_data_read(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_775 = {}
        # Getting the type of 'stream' (line 181)
        stream_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'stream', False)
        # Obtaining the member 'all_data_read' of a type (line 181)
        all_data_read_774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), stream_773, 'all_data_read')
        # Calling all_data_read(args, kwargs) (line 181)
        all_data_read_call_result_776 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), all_data_read_774, *[], **kwargs_775)
        
        # Processing the call keyword arguments (line 181)
        kwargs_777 = {}
        # Getting the type of 'assert_' (line 181)
        assert__772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 181)
        assert__call_result_778 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assert__772, *[all_data_read_call_result_776], **kwargs_777)
        
        
        # ################# End of 'test_all_data_read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_all_data_read' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_779)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_all_data_read'
        return stypy_return_type_779


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 98, 0, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZlibInputStream.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZlibInputStream' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'TestZlibInputStream', TestZlibInputStream)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
