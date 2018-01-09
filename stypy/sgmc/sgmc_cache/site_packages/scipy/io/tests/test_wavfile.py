
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: import sys
5: import tempfile
6: from io import BytesIO
7: 
8: import numpy as np
9: from numpy.testing import assert_equal, assert_, assert_array_equal
10: from pytest import raises as assert_raises
11: from scipy._lib._numpy_compat import suppress_warnings
12: 
13: from scipy.io import wavfile
14: 
15: 
16: def datafile(fn):
17:     return os.path.join(os.path.dirname(__file__), 'data', fn)
18: 
19: 
20: def test_read_1():
21:     for mmap in [False, True]:
22:         rate, data = wavfile.read(datafile('test-44100Hz-le-1ch-4bytes.wav'),
23:                                   mmap=mmap)
24: 
25:         assert_equal(rate, 44100)
26:         assert_(np.issubdtype(data.dtype, np.int32))
27:         assert_equal(data.shape, (4410,))
28: 
29:         del data
30: 
31: 
32: def test_read_2():
33:     for mmap in [False, True]:
34:         rate, data = wavfile.read(datafile('test-8000Hz-le-2ch-1byteu.wav'),
35:                                   mmap=mmap)
36:         assert_equal(rate, 8000)
37:         assert_(np.issubdtype(data.dtype, np.uint8))
38:         assert_equal(data.shape, (800, 2))
39: 
40:         del data
41: 
42: def test_read_3():
43:     for mmap in [False, True]:
44:         rate, data = wavfile.read(datafile('test-44100Hz-2ch-32bit-float-le.wav'),
45:                                   mmap=mmap)
46:         assert_equal(rate, 44100)
47:         assert_(np.issubdtype(data.dtype, np.float32))
48:         assert_equal(data.shape, (441, 2))
49: 
50:         del data
51: 
52: def test_read_4():
53:     for mmap in [False, True]:
54:         with suppress_warnings() as sup:
55:             sup.filter(wavfile.WavFileWarning,
56:                        "Chunk .non-data. not understood, skipping it")
57:             rate, data = wavfile.read(datafile('test-48000Hz-2ch-64bit-float-le-wavex.wav'),
58:                                       mmap=mmap)
59: 
60:         assert_equal(rate, 48000)
61:         assert_(np.issubdtype(data.dtype, np.float64))
62:         assert_equal(data.shape, (480, 2))
63: 
64:         del data
65: 
66: 
67: def test_read_5():
68:     for mmap in [False, True]:
69:         rate, data = wavfile.read(datafile('test-44100Hz-2ch-32bit-float-be.wav'),
70:                                   mmap=mmap)
71:         assert_equal(rate, 44100)
72:         assert_(np.issubdtype(data.dtype, np.float32))
73:         assert_(data.dtype.byteorder == '>' or (sys.byteorder == 'big' and
74:                                                 data.dtype.byteorder == '='))
75:         assert_equal(data.shape, (441, 2))
76: 
77:         del data
78: 
79: 
80: def test_read_fail():
81:     for mmap in [False, True]:
82:         fp = open(datafile('example_1.nc'), 'rb')
83:         assert_raises(ValueError, wavfile.read, fp, mmap=mmap)
84:         fp.close()
85: 
86: 
87: def test_read_early_eof():
88:     for mmap in [False, True]:
89:         fp = open(datafile('test-44100Hz-le-1ch-4bytes-early-eof.wav'), 'rb')
90:         assert_raises(ValueError, wavfile.read, fp, mmap=mmap)
91:         fp.close()
92: 
93: 
94: def test_read_incomplete_chunk():
95:     for mmap in [False, True]:
96:         fp = open(datafile('test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav'), 'rb')
97:         assert_raises(ValueError, wavfile.read, fp, mmap=mmap)
98:         fp.close()
99: 
100: 
101: def _check_roundtrip(realfile, rate, dtype, channels):
102:     if realfile:
103:         fd, tmpfile = tempfile.mkstemp(suffix='.wav')
104:         os.close(fd)
105:     else:
106:         tmpfile = BytesIO()
107:     try:
108:         data = np.random.rand(100, channels)
109:         if channels == 1:
110:             data = data[:,0]
111:         if dtype.kind == 'f':
112:             # The range of the float type should be in [-1, 1]
113:             data = data.astype(dtype)
114:         else:
115:             data = (data*128).astype(dtype)
116: 
117:         wavfile.write(tmpfile, rate, data)
118: 
119:         for mmap in [False, True]:
120:             rate2, data2 = wavfile.read(tmpfile, mmap=mmap)
121: 
122:             assert_equal(rate, rate2)
123:             assert_(data2.dtype.byteorder in ('<', '=', '|'), msg=data2.dtype)
124:             assert_array_equal(data, data2)
125: 
126:             del data2
127:     finally:
128:         if realfile:
129:             os.unlink(tmpfile)
130: 
131: 
132: def test_write_roundtrip():
133:     for realfile in (False, True):
134:         for dtypechar in ('i', 'u', 'f', 'g', 'q'):
135:             for size in (1, 2, 4, 8):
136:                 if size == 1 and dtypechar == 'i':
137:                     # signed 8-bit integer PCM is not allowed
138:                     continue
139:                 if size > 1 and dtypechar == 'u':
140:                     # unsigned > 8-bit integer PCM is not allowed
141:                     continue
142:                 if (size == 1 or size == 2) and dtypechar == 'f':
143:                     # 8- or 16-bit float PCM is not expected
144:                     continue
145:                 if dtypechar in 'gq':
146:                     # no size allowed for these types
147:                     if size == 1:
148:                         size = ''
149:                     else:
150:                         continue
151: 
152:                 for endianness in ('>', '<'):
153:                     if size == 1 and endianness == '<':
154:                         continue
155:                     for rate in (8000, 32000):
156:                         for channels in (1, 2, 5):
157:                             dt = np.dtype('%s%s%s' % (endianness, dtypechar, size))
158:                             _check_roundtrip(realfile, rate, dt, channels)
159: 
160: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import tempfile' statement (line 5)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from io import BytesIO' statement (line 6)
try:
    from io import BytesIO

except:
    BytesIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'io', None, module_type_store, ['BytesIO'], [BytesIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9383 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_9383) is not StypyTypeError):

    if (import_9383 != 'pyd_module'):
        __import__(import_9383)
        sys_modules_9384 = sys.modules[import_9383]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_9384.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_9383)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_equal, assert_, assert_array_equal' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9385 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_9385) is not StypyTypeError):

    if (import_9385 != 'pyd_module'):
        __import__(import_9385)
        sys_modules_9386 = sys.modules[import_9385]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_9386.module_type_store, module_type_store, ['assert_equal', 'assert_', 'assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_9386, sys_modules_9386.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_, assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_', 'assert_array_equal'], [assert_equal, assert_, assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_9385)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from pytest import assert_raises' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9387 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_9387) is not StypyTypeError):

    if (import_9387 != 'pyd_module'):
        __import__(import_9387)
        sys_modules_9388 = sys.modules[import_9387]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_9388.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_9388, sys_modules_9388.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_9387)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9389 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat')

if (type(import_9389) is not StypyTypeError):

    if (import_9389 != 'pyd_module'):
        __import__(import_9389)
        sys_modules_9390 = sys.modules[import_9389]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', sys_modules_9390.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_9390, sys_modules_9390.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', import_9389)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.io import wavfile' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_9391 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io')

if (type(import_9391) is not StypyTypeError):

    if (import_9391 != 'pyd_module'):
        __import__(import_9391)
        sys_modules_9392 = sys.modules[import_9391]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io', sys_modules_9392.module_type_store, module_type_store, ['wavfile'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_9392, sys_modules_9392.module_type_store, module_type_store)
    else:
        from scipy.io import wavfile

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io', None, module_type_store, ['wavfile'], [wavfile])

else:
    # Assigning a type to the variable 'scipy.io' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io', import_9391)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')


@norecursion
def datafile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'datafile'
    module_type_store = module_type_store.open_function_context('datafile', 16, 0, False)
    
    # Passed parameters checking function
    datafile.stypy_localization = localization
    datafile.stypy_type_of_self = None
    datafile.stypy_type_store = module_type_store
    datafile.stypy_function_name = 'datafile'
    datafile.stypy_param_names_list = ['fn']
    datafile.stypy_varargs_param_name = None
    datafile.stypy_kwargs_param_name = None
    datafile.stypy_call_defaults = defaults
    datafile.stypy_call_varargs = varargs
    datafile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'datafile', ['fn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'datafile', localization, ['fn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'datafile(...)' code ##################

    
    # Call to join(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to dirname(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of '__file__' (line 17)
    file___9399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 40), '__file__', False)
    # Processing the call keyword arguments (line 17)
    kwargs_9400 = {}
    # Getting the type of 'os' (line 17)
    os_9396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_9397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), os_9396, 'path')
    # Obtaining the member 'dirname' of a type (line 17)
    dirname_9398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), path_9397, 'dirname')
    # Calling dirname(args, kwargs) (line 17)
    dirname_call_result_9401 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), dirname_9398, *[file___9399], **kwargs_9400)
    
    str_9402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 51), 'str', 'data')
    # Getting the type of 'fn' (line 17)
    fn_9403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 59), 'fn', False)
    # Processing the call keyword arguments (line 17)
    kwargs_9404 = {}
    # Getting the type of 'os' (line 17)
    os_9393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_9394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 11), os_9393, 'path')
    # Obtaining the member 'join' of a type (line 17)
    join_9395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 11), path_9394, 'join')
    # Calling join(args, kwargs) (line 17)
    join_call_result_9405 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), join_9395, *[dirname_call_result_9401, str_9402, fn_9403], **kwargs_9404)
    
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', join_call_result_9405)
    
    # ################# End of 'datafile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'datafile' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_9406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9406)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'datafile'
    return stypy_return_type_9406

# Assigning a type to the variable 'datafile' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'datafile', datafile)

@norecursion
def test_read_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_1'
    module_type_store = module_type_store.open_function_context('test_read_1', 20, 0, False)
    
    # Passed parameters checking function
    test_read_1.stypy_localization = localization
    test_read_1.stypy_type_of_self = None
    test_read_1.stypy_type_store = module_type_store
    test_read_1.stypy_function_name = 'test_read_1'
    test_read_1.stypy_param_names_list = []
    test_read_1.stypy_varargs_param_name = None
    test_read_1.stypy_kwargs_param_name = None
    test_read_1.stypy_call_defaults = defaults
    test_read_1.stypy_call_varargs = varargs
    test_read_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_1', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_1', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_1(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_9407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    # Getting the type of 'False' (line 21)
    False_9408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 16), list_9407, False_9408)
    # Adding element type (line 21)
    # Getting the type of 'True' (line 21)
    True_9409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 16), list_9407, True_9409)
    
    # Testing the type of a for loop iterable (line 21)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 4), list_9407)
    # Getting the type of the for loop variable (line 21)
    for_loop_var_9410 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 4), list_9407)
    # Assigning a type to the variable 'mmap' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'mmap', for_loop_var_9410)
    # SSA begins for a for statement (line 21)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 22):
    
    # Assigning a Subscript to a Name (line 22):
    
    # Obtaining the type of the subscript
    int_9411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'int')
    
    # Call to read(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to datafile(...): (line 22)
    # Processing the call arguments (line 22)
    str_9415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'str', 'test-44100Hz-le-1ch-4bytes.wav')
    # Processing the call keyword arguments (line 22)
    kwargs_9416 = {}
    # Getting the type of 'datafile' (line 22)
    datafile_9414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 22)
    datafile_call_result_9417 = invoke(stypy.reporting.localization.Localization(__file__, 22, 34), datafile_9414, *[str_9415], **kwargs_9416)
    
    # Processing the call keyword arguments (line 22)
    # Getting the type of 'mmap' (line 23)
    mmap_9418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), 'mmap', False)
    keyword_9419 = mmap_9418
    kwargs_9420 = {'mmap': keyword_9419}
    # Getting the type of 'wavfile' (line 22)
    wavfile_9412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 22)
    read_9413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 21), wavfile_9412, 'read')
    # Calling read(args, kwargs) (line 22)
    read_call_result_9421 = invoke(stypy.reporting.localization.Localization(__file__, 22, 21), read_9413, *[datafile_call_result_9417], **kwargs_9420)
    
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___9422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), read_call_result_9421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_9423 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), getitem___9422, int_9411)
    
    # Assigning a type to the variable 'tuple_var_assignment_9369' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_var_assignment_9369', subscript_call_result_9423)
    
    # Assigning a Subscript to a Name (line 22):
    
    # Obtaining the type of the subscript
    int_9424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'int')
    
    # Call to read(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to datafile(...): (line 22)
    # Processing the call arguments (line 22)
    str_9428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'str', 'test-44100Hz-le-1ch-4bytes.wav')
    # Processing the call keyword arguments (line 22)
    kwargs_9429 = {}
    # Getting the type of 'datafile' (line 22)
    datafile_9427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 22)
    datafile_call_result_9430 = invoke(stypy.reporting.localization.Localization(__file__, 22, 34), datafile_9427, *[str_9428], **kwargs_9429)
    
    # Processing the call keyword arguments (line 22)
    # Getting the type of 'mmap' (line 23)
    mmap_9431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), 'mmap', False)
    keyword_9432 = mmap_9431
    kwargs_9433 = {'mmap': keyword_9432}
    # Getting the type of 'wavfile' (line 22)
    wavfile_9425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 22)
    read_9426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 21), wavfile_9425, 'read')
    # Calling read(args, kwargs) (line 22)
    read_call_result_9434 = invoke(stypy.reporting.localization.Localization(__file__, 22, 21), read_9426, *[datafile_call_result_9430], **kwargs_9433)
    
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___9435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), read_call_result_9434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_9436 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), getitem___9435, int_9424)
    
    # Assigning a type to the variable 'tuple_var_assignment_9370' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_var_assignment_9370', subscript_call_result_9436)
    
    # Assigning a Name to a Name (line 22):
    # Getting the type of 'tuple_var_assignment_9369' (line 22)
    tuple_var_assignment_9369_9437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_var_assignment_9369')
    # Assigning a type to the variable 'rate' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'rate', tuple_var_assignment_9369_9437)
    
    # Assigning a Name to a Name (line 22):
    # Getting the type of 'tuple_var_assignment_9370' (line 22)
    tuple_var_assignment_9370_9438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_var_assignment_9370')
    # Assigning a type to the variable 'data' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'data', tuple_var_assignment_9370_9438)
    
    # Call to assert_equal(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'rate' (line 25)
    rate_9440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'rate', False)
    int_9441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_9442 = {}
    # Getting the type of 'assert_equal' (line 25)
    assert_equal_9439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 25)
    assert_equal_call_result_9443 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assert_equal_9439, *[rate_9440, int_9441], **kwargs_9442)
    
    
    # Call to assert_(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to issubdtype(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'data' (line 26)
    data_9447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'data', False)
    # Obtaining the member 'dtype' of a type (line 26)
    dtype_9448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 30), data_9447, 'dtype')
    # Getting the type of 'np' (line 26)
    np_9449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 42), 'np', False)
    # Obtaining the member 'int32' of a type (line 26)
    int32_9450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 42), np_9449, 'int32')
    # Processing the call keyword arguments (line 26)
    kwargs_9451 = {}
    # Getting the type of 'np' (line 26)
    np_9445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 26)
    issubdtype_9446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), np_9445, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 26)
    issubdtype_call_result_9452 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), issubdtype_9446, *[dtype_9448, int32_9450], **kwargs_9451)
    
    # Processing the call keyword arguments (line 26)
    kwargs_9453 = {}
    # Getting the type of 'assert_' (line 26)
    assert__9444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 26)
    assert__call_result_9454 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assert__9444, *[issubdtype_call_result_9452], **kwargs_9453)
    
    
    # Call to assert_equal(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'data' (line 27)
    data_9456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'data', False)
    # Obtaining the member 'shape' of a type (line 27)
    shape_9457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), data_9456, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_9458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_9459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 34), tuple_9458, int_9459)
    
    # Processing the call keyword arguments (line 27)
    kwargs_9460 = {}
    # Getting the type of 'assert_equal' (line 27)
    assert_equal_9455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 27)
    assert_equal_call_result_9461 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_equal_9455, *[shape_9457, tuple_9458], **kwargs_9460)
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 29, 8), module_type_store, 'data')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_1' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_9462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9462)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_1'
    return stypy_return_type_9462

# Assigning a type to the variable 'test_read_1' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'test_read_1', test_read_1)

@norecursion
def test_read_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_2'
    module_type_store = module_type_store.open_function_context('test_read_2', 32, 0, False)
    
    # Passed parameters checking function
    test_read_2.stypy_localization = localization
    test_read_2.stypy_type_of_self = None
    test_read_2.stypy_type_store = module_type_store
    test_read_2.stypy_function_name = 'test_read_2'
    test_read_2.stypy_param_names_list = []
    test_read_2.stypy_varargs_param_name = None
    test_read_2.stypy_kwargs_param_name = None
    test_read_2.stypy_call_defaults = defaults
    test_read_2.stypy_call_varargs = varargs
    test_read_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_2(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_9463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    # Getting the type of 'False' (line 33)
    False_9464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 16), list_9463, False_9464)
    # Adding element type (line 33)
    # Getting the type of 'True' (line 33)
    True_9465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 16), list_9463, True_9465)
    
    # Testing the type of a for loop iterable (line 33)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 33, 4), list_9463)
    # Getting the type of the for loop variable (line 33)
    for_loop_var_9466 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 33, 4), list_9463)
    # Assigning a type to the variable 'mmap' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'mmap', for_loop_var_9466)
    # SSA begins for a for statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 34):
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_9467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'int')
    
    # Call to read(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to datafile(...): (line 34)
    # Processing the call arguments (line 34)
    str_9471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'str', 'test-8000Hz-le-2ch-1byteu.wav')
    # Processing the call keyword arguments (line 34)
    kwargs_9472 = {}
    # Getting the type of 'datafile' (line 34)
    datafile_9470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 34)
    datafile_call_result_9473 = invoke(stypy.reporting.localization.Localization(__file__, 34, 34), datafile_9470, *[str_9471], **kwargs_9472)
    
    # Processing the call keyword arguments (line 34)
    # Getting the type of 'mmap' (line 35)
    mmap_9474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'mmap', False)
    keyword_9475 = mmap_9474
    kwargs_9476 = {'mmap': keyword_9475}
    # Getting the type of 'wavfile' (line 34)
    wavfile_9468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 34)
    read_9469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), wavfile_9468, 'read')
    # Calling read(args, kwargs) (line 34)
    read_call_result_9477 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), read_9469, *[datafile_call_result_9473], **kwargs_9476)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___9478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), read_call_result_9477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_9479 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), getitem___9478, int_9467)
    
    # Assigning a type to the variable 'tuple_var_assignment_9371' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_9371', subscript_call_result_9479)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_9480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'int')
    
    # Call to read(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to datafile(...): (line 34)
    # Processing the call arguments (line 34)
    str_9484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'str', 'test-8000Hz-le-2ch-1byteu.wav')
    # Processing the call keyword arguments (line 34)
    kwargs_9485 = {}
    # Getting the type of 'datafile' (line 34)
    datafile_9483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 34)
    datafile_call_result_9486 = invoke(stypy.reporting.localization.Localization(__file__, 34, 34), datafile_9483, *[str_9484], **kwargs_9485)
    
    # Processing the call keyword arguments (line 34)
    # Getting the type of 'mmap' (line 35)
    mmap_9487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'mmap', False)
    keyword_9488 = mmap_9487
    kwargs_9489 = {'mmap': keyword_9488}
    # Getting the type of 'wavfile' (line 34)
    wavfile_9481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 34)
    read_9482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), wavfile_9481, 'read')
    # Calling read(args, kwargs) (line 34)
    read_call_result_9490 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), read_9482, *[datafile_call_result_9486], **kwargs_9489)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___9491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), read_call_result_9490, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_9492 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), getitem___9491, int_9480)
    
    # Assigning a type to the variable 'tuple_var_assignment_9372' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_9372', subscript_call_result_9492)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_9371' (line 34)
    tuple_var_assignment_9371_9493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_9371')
    # Assigning a type to the variable 'rate' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'rate', tuple_var_assignment_9371_9493)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_9372' (line 34)
    tuple_var_assignment_9372_9494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_9372')
    # Assigning a type to the variable 'data' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'data', tuple_var_assignment_9372_9494)
    
    # Call to assert_equal(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'rate' (line 36)
    rate_9496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'rate', False)
    int_9497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_9498 = {}
    # Getting the type of 'assert_equal' (line 36)
    assert_equal_9495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 36)
    assert_equal_call_result_9499 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_equal_9495, *[rate_9496, int_9497], **kwargs_9498)
    
    
    # Call to assert_(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to issubdtype(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'data' (line 37)
    data_9503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'data', False)
    # Obtaining the member 'dtype' of a type (line 37)
    dtype_9504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 30), data_9503, 'dtype')
    # Getting the type of 'np' (line 37)
    np_9505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 42), 'np', False)
    # Obtaining the member 'uint8' of a type (line 37)
    uint8_9506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 42), np_9505, 'uint8')
    # Processing the call keyword arguments (line 37)
    kwargs_9507 = {}
    # Getting the type of 'np' (line 37)
    np_9501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 37)
    issubdtype_9502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), np_9501, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 37)
    issubdtype_call_result_9508 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), issubdtype_9502, *[dtype_9504, uint8_9506], **kwargs_9507)
    
    # Processing the call keyword arguments (line 37)
    kwargs_9509 = {}
    # Getting the type of 'assert_' (line 37)
    assert__9500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 37)
    assert__call_result_9510 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assert__9500, *[issubdtype_call_result_9508], **kwargs_9509)
    
    
    # Call to assert_equal(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'data' (line 38)
    data_9512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'data', False)
    # Obtaining the member 'shape' of a type (line 38)
    shape_9513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), data_9512, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_9514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    int_9515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 34), tuple_9514, int_9515)
    # Adding element type (line 38)
    int_9516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 34), tuple_9514, int_9516)
    
    # Processing the call keyword arguments (line 38)
    kwargs_9517 = {}
    # Getting the type of 'assert_equal' (line 38)
    assert_equal_9511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 38)
    assert_equal_call_result_9518 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assert_equal_9511, *[shape_9513, tuple_9514], **kwargs_9517)
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 40, 8), module_type_store, 'data')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_2' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_9519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9519)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_2'
    return stypy_return_type_9519

# Assigning a type to the variable 'test_read_2' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'test_read_2', test_read_2)

@norecursion
def test_read_3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_3'
    module_type_store = module_type_store.open_function_context('test_read_3', 42, 0, False)
    
    # Passed parameters checking function
    test_read_3.stypy_localization = localization
    test_read_3.stypy_type_of_self = None
    test_read_3.stypy_type_store = module_type_store
    test_read_3.stypy_function_name = 'test_read_3'
    test_read_3.stypy_param_names_list = []
    test_read_3.stypy_varargs_param_name = None
    test_read_3.stypy_kwargs_param_name = None
    test_read_3.stypy_call_defaults = defaults
    test_read_3.stypy_call_varargs = varargs
    test_read_3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_3', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_3', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_3(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_9520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    # Getting the type of 'False' (line 43)
    False_9521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 16), list_9520, False_9521)
    # Adding element type (line 43)
    # Getting the type of 'True' (line 43)
    True_9522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 16), list_9520, True_9522)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 4), list_9520)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_9523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 4), list_9520)
    # Assigning a type to the variable 'mmap' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'mmap', for_loop_var_9523)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 44):
    
    # Assigning a Subscript to a Name (line 44):
    
    # Obtaining the type of the subscript
    int_9524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'int')
    
    # Call to read(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to datafile(...): (line 44)
    # Processing the call arguments (line 44)
    str_9528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 43), 'str', 'test-44100Hz-2ch-32bit-float-le.wav')
    # Processing the call keyword arguments (line 44)
    kwargs_9529 = {}
    # Getting the type of 'datafile' (line 44)
    datafile_9527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 44)
    datafile_call_result_9530 = invoke(stypy.reporting.localization.Localization(__file__, 44, 34), datafile_9527, *[str_9528], **kwargs_9529)
    
    # Processing the call keyword arguments (line 44)
    # Getting the type of 'mmap' (line 45)
    mmap_9531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'mmap', False)
    keyword_9532 = mmap_9531
    kwargs_9533 = {'mmap': keyword_9532}
    # Getting the type of 'wavfile' (line 44)
    wavfile_9525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 44)
    read_9526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 21), wavfile_9525, 'read')
    # Calling read(args, kwargs) (line 44)
    read_call_result_9534 = invoke(stypy.reporting.localization.Localization(__file__, 44, 21), read_9526, *[datafile_call_result_9530], **kwargs_9533)
    
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___9535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), read_call_result_9534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_9536 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), getitem___9535, int_9524)
    
    # Assigning a type to the variable 'tuple_var_assignment_9373' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_9373', subscript_call_result_9536)
    
    # Assigning a Subscript to a Name (line 44):
    
    # Obtaining the type of the subscript
    int_9537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 8), 'int')
    
    # Call to read(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to datafile(...): (line 44)
    # Processing the call arguments (line 44)
    str_9541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 43), 'str', 'test-44100Hz-2ch-32bit-float-le.wav')
    # Processing the call keyword arguments (line 44)
    kwargs_9542 = {}
    # Getting the type of 'datafile' (line 44)
    datafile_9540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 44)
    datafile_call_result_9543 = invoke(stypy.reporting.localization.Localization(__file__, 44, 34), datafile_9540, *[str_9541], **kwargs_9542)
    
    # Processing the call keyword arguments (line 44)
    # Getting the type of 'mmap' (line 45)
    mmap_9544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'mmap', False)
    keyword_9545 = mmap_9544
    kwargs_9546 = {'mmap': keyword_9545}
    # Getting the type of 'wavfile' (line 44)
    wavfile_9538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 44)
    read_9539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 21), wavfile_9538, 'read')
    # Calling read(args, kwargs) (line 44)
    read_call_result_9547 = invoke(stypy.reporting.localization.Localization(__file__, 44, 21), read_9539, *[datafile_call_result_9543], **kwargs_9546)
    
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___9548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), read_call_result_9547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_9549 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), getitem___9548, int_9537)
    
    # Assigning a type to the variable 'tuple_var_assignment_9374' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_9374', subscript_call_result_9549)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_var_assignment_9373' (line 44)
    tuple_var_assignment_9373_9550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_9373')
    # Assigning a type to the variable 'rate' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'rate', tuple_var_assignment_9373_9550)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_var_assignment_9374' (line 44)
    tuple_var_assignment_9374_9551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'tuple_var_assignment_9374')
    # Assigning a type to the variable 'data' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'data', tuple_var_assignment_9374_9551)
    
    # Call to assert_equal(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'rate' (line 46)
    rate_9553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'rate', False)
    int_9554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'int')
    # Processing the call keyword arguments (line 46)
    kwargs_9555 = {}
    # Getting the type of 'assert_equal' (line 46)
    assert_equal_9552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 46)
    assert_equal_call_result_9556 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assert_equal_9552, *[rate_9553, int_9554], **kwargs_9555)
    
    
    # Call to assert_(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to issubdtype(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'data' (line 47)
    data_9560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'data', False)
    # Obtaining the member 'dtype' of a type (line 47)
    dtype_9561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 30), data_9560, 'dtype')
    # Getting the type of 'np' (line 47)
    np_9562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'np', False)
    # Obtaining the member 'float32' of a type (line 47)
    float32_9563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 42), np_9562, 'float32')
    # Processing the call keyword arguments (line 47)
    kwargs_9564 = {}
    # Getting the type of 'np' (line 47)
    np_9558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 47)
    issubdtype_9559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), np_9558, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 47)
    issubdtype_call_result_9565 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), issubdtype_9559, *[dtype_9561, float32_9563], **kwargs_9564)
    
    # Processing the call keyword arguments (line 47)
    kwargs_9566 = {}
    # Getting the type of 'assert_' (line 47)
    assert__9557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 47)
    assert__call_result_9567 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert__9557, *[issubdtype_call_result_9565], **kwargs_9566)
    
    
    # Call to assert_equal(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'data' (line 48)
    data_9569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'data', False)
    # Obtaining the member 'shape' of a type (line 48)
    shape_9570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 21), data_9569, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_9571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    int_9572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 34), tuple_9571, int_9572)
    # Adding element type (line 48)
    int_9573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 34), tuple_9571, int_9573)
    
    # Processing the call keyword arguments (line 48)
    kwargs_9574 = {}
    # Getting the type of 'assert_equal' (line 48)
    assert_equal_9568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 48)
    assert_equal_call_result_9575 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_equal_9568, *[shape_9570, tuple_9571], **kwargs_9574)
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 50, 8), module_type_store, 'data')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_3' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_9576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9576)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_3'
    return stypy_return_type_9576

# Assigning a type to the variable 'test_read_3' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'test_read_3', test_read_3)

@norecursion
def test_read_4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_4'
    module_type_store = module_type_store.open_function_context('test_read_4', 52, 0, False)
    
    # Passed parameters checking function
    test_read_4.stypy_localization = localization
    test_read_4.stypy_type_of_self = None
    test_read_4.stypy_type_store = module_type_store
    test_read_4.stypy_function_name = 'test_read_4'
    test_read_4.stypy_param_names_list = []
    test_read_4.stypy_varargs_param_name = None
    test_read_4.stypy_kwargs_param_name = None
    test_read_4.stypy_call_defaults = defaults
    test_read_4.stypy_call_varargs = varargs
    test_read_4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_4', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_4', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_4(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_9577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    # Getting the type of 'False' (line 53)
    False_9578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), list_9577, False_9578)
    # Adding element type (line 53)
    # Getting the type of 'True' (line 53)
    True_9579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), list_9577, True_9579)
    
    # Testing the type of a for loop iterable (line 53)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 4), list_9577)
    # Getting the type of the for loop variable (line 53)
    for_loop_var_9580 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 4), list_9577)
    # Assigning a type to the variable 'mmap' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'mmap', for_loop_var_9580)
    # SSA begins for a for statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to suppress_warnings(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_9582 = {}
    # Getting the type of 'suppress_warnings' (line 54)
    suppress_warnings_9581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 54)
    suppress_warnings_call_result_9583 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), suppress_warnings_9581, *[], **kwargs_9582)
    
    with_9584 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 54, 13), suppress_warnings_call_result_9583, 'with parameter', '__enter__', '__exit__')

    if with_9584:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 54)
        enter___9585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), suppress_warnings_call_result_9583, '__enter__')
        with_enter_9586 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), enter___9585)
        # Assigning a type to the variable 'sup' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'sup', with_enter_9586)
        
        # Call to filter(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'wavfile' (line 55)
        wavfile_9589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'wavfile', False)
        # Obtaining the member 'WavFileWarning' of a type (line 55)
        WavFileWarning_9590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 23), wavfile_9589, 'WavFileWarning')
        str_9591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', 'Chunk .non-data. not understood, skipping it')
        # Processing the call keyword arguments (line 55)
        kwargs_9592 = {}
        # Getting the type of 'sup' (line 55)
        sup_9587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'sup', False)
        # Obtaining the member 'filter' of a type (line 55)
        filter_9588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), sup_9587, 'filter')
        # Calling filter(args, kwargs) (line 55)
        filter_call_result_9593 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), filter_9588, *[WavFileWarning_9590, str_9591], **kwargs_9592)
        
        
        # Assigning a Call to a Tuple (line 57):
        
        # Assigning a Subscript to a Name (line 57):
        
        # Obtaining the type of the subscript
        int_9594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'int')
        
        # Call to read(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to datafile(...): (line 57)
        # Processing the call arguments (line 57)
        str_9598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 47), 'str', 'test-48000Hz-2ch-64bit-float-le-wavex.wav')
        # Processing the call keyword arguments (line 57)
        kwargs_9599 = {}
        # Getting the type of 'datafile' (line 57)
        datafile_9597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 38), 'datafile', False)
        # Calling datafile(args, kwargs) (line 57)
        datafile_call_result_9600 = invoke(stypy.reporting.localization.Localization(__file__, 57, 38), datafile_9597, *[str_9598], **kwargs_9599)
        
        # Processing the call keyword arguments (line 57)
        # Getting the type of 'mmap' (line 58)
        mmap_9601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'mmap', False)
        keyword_9602 = mmap_9601
        kwargs_9603 = {'mmap': keyword_9602}
        # Getting the type of 'wavfile' (line 57)
        wavfile_9595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'wavfile', False)
        # Obtaining the member 'read' of a type (line 57)
        read_9596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), wavfile_9595, 'read')
        # Calling read(args, kwargs) (line 57)
        read_call_result_9604 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), read_9596, *[datafile_call_result_9600], **kwargs_9603)
        
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___9605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), read_call_result_9604, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_9606 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), getitem___9605, int_9594)
        
        # Assigning a type to the variable 'tuple_var_assignment_9375' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_9375', subscript_call_result_9606)
        
        # Assigning a Subscript to a Name (line 57):
        
        # Obtaining the type of the subscript
        int_9607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'int')
        
        # Call to read(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to datafile(...): (line 57)
        # Processing the call arguments (line 57)
        str_9611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 47), 'str', 'test-48000Hz-2ch-64bit-float-le-wavex.wav')
        # Processing the call keyword arguments (line 57)
        kwargs_9612 = {}
        # Getting the type of 'datafile' (line 57)
        datafile_9610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 38), 'datafile', False)
        # Calling datafile(args, kwargs) (line 57)
        datafile_call_result_9613 = invoke(stypy.reporting.localization.Localization(__file__, 57, 38), datafile_9610, *[str_9611], **kwargs_9612)
        
        # Processing the call keyword arguments (line 57)
        # Getting the type of 'mmap' (line 58)
        mmap_9614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'mmap', False)
        keyword_9615 = mmap_9614
        kwargs_9616 = {'mmap': keyword_9615}
        # Getting the type of 'wavfile' (line 57)
        wavfile_9608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'wavfile', False)
        # Obtaining the member 'read' of a type (line 57)
        read_9609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), wavfile_9608, 'read')
        # Calling read(args, kwargs) (line 57)
        read_call_result_9617 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), read_9609, *[datafile_call_result_9613], **kwargs_9616)
        
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___9618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), read_call_result_9617, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_9619 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), getitem___9618, int_9607)
        
        # Assigning a type to the variable 'tuple_var_assignment_9376' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_9376', subscript_call_result_9619)
        
        # Assigning a Name to a Name (line 57):
        # Getting the type of 'tuple_var_assignment_9375' (line 57)
        tuple_var_assignment_9375_9620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_9375')
        # Assigning a type to the variable 'rate' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'rate', tuple_var_assignment_9375_9620)
        
        # Assigning a Name to a Name (line 57):
        # Getting the type of 'tuple_var_assignment_9376' (line 57)
        tuple_var_assignment_9376_9621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_9376')
        # Assigning a type to the variable 'data' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'data', tuple_var_assignment_9376_9621)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 54)
        exit___9622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), suppress_warnings_call_result_9583, '__exit__')
        with_exit_9623 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), exit___9622, None, None, None)

    
    # Call to assert_equal(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'rate' (line 60)
    rate_9625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'rate', False)
    int_9626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'int')
    # Processing the call keyword arguments (line 60)
    kwargs_9627 = {}
    # Getting the type of 'assert_equal' (line 60)
    assert_equal_9624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 60)
    assert_equal_call_result_9628 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_equal_9624, *[rate_9625, int_9626], **kwargs_9627)
    
    
    # Call to assert_(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to issubdtype(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'data' (line 61)
    data_9632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'data', False)
    # Obtaining the member 'dtype' of a type (line 61)
    dtype_9633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), data_9632, 'dtype')
    # Getting the type of 'np' (line 61)
    np_9634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'np', False)
    # Obtaining the member 'float64' of a type (line 61)
    float64_9635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 42), np_9634, 'float64')
    # Processing the call keyword arguments (line 61)
    kwargs_9636 = {}
    # Getting the type of 'np' (line 61)
    np_9630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 61)
    issubdtype_9631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), np_9630, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 61)
    issubdtype_call_result_9637 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), issubdtype_9631, *[dtype_9633, float64_9635], **kwargs_9636)
    
    # Processing the call keyword arguments (line 61)
    kwargs_9638 = {}
    # Getting the type of 'assert_' (line 61)
    assert__9629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 61)
    assert__call_result_9639 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assert__9629, *[issubdtype_call_result_9637], **kwargs_9638)
    
    
    # Call to assert_equal(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'data' (line 62)
    data_9641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'data', False)
    # Obtaining the member 'shape' of a type (line 62)
    shape_9642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 21), data_9641, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_9643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    int_9644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 34), tuple_9643, int_9644)
    # Adding element type (line 62)
    int_9645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 34), tuple_9643, int_9645)
    
    # Processing the call keyword arguments (line 62)
    kwargs_9646 = {}
    # Getting the type of 'assert_equal' (line 62)
    assert_equal_9640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 62)
    assert_equal_call_result_9647 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_equal_9640, *[shape_9642, tuple_9643], **kwargs_9646)
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 64, 8), module_type_store, 'data')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_4' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_9648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9648)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_4'
    return stypy_return_type_9648

# Assigning a type to the variable 'test_read_4' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'test_read_4', test_read_4)

@norecursion
def test_read_5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_5'
    module_type_store = module_type_store.open_function_context('test_read_5', 67, 0, False)
    
    # Passed parameters checking function
    test_read_5.stypy_localization = localization
    test_read_5.stypy_type_of_self = None
    test_read_5.stypy_type_store = module_type_store
    test_read_5.stypy_function_name = 'test_read_5'
    test_read_5.stypy_param_names_list = []
    test_read_5.stypy_varargs_param_name = None
    test_read_5.stypy_kwargs_param_name = None
    test_read_5.stypy_call_defaults = defaults
    test_read_5.stypy_call_varargs = varargs
    test_read_5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_5', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_5', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_5(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_9649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'False' (line 68)
    False_9650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), list_9649, False_9650)
    # Adding element type (line 68)
    # Getting the type of 'True' (line 68)
    True_9651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), list_9649, True_9651)
    
    # Testing the type of a for loop iterable (line 68)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 4), list_9649)
    # Getting the type of the for loop variable (line 68)
    for_loop_var_9652 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 4), list_9649)
    # Assigning a type to the variable 'mmap' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'mmap', for_loop_var_9652)
    # SSA begins for a for statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 69):
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    int_9653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
    
    # Call to read(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to datafile(...): (line 69)
    # Processing the call arguments (line 69)
    str_9657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 43), 'str', 'test-44100Hz-2ch-32bit-float-be.wav')
    # Processing the call keyword arguments (line 69)
    kwargs_9658 = {}
    # Getting the type of 'datafile' (line 69)
    datafile_9656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 69)
    datafile_call_result_9659 = invoke(stypy.reporting.localization.Localization(__file__, 69, 34), datafile_9656, *[str_9657], **kwargs_9658)
    
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'mmap' (line 70)
    mmap_9660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 39), 'mmap', False)
    keyword_9661 = mmap_9660
    kwargs_9662 = {'mmap': keyword_9661}
    # Getting the type of 'wavfile' (line 69)
    wavfile_9654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 69)
    read_9655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 21), wavfile_9654, 'read')
    # Calling read(args, kwargs) (line 69)
    read_call_result_9663 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), read_9655, *[datafile_call_result_9659], **kwargs_9662)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___9664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), read_call_result_9663, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_9665 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), getitem___9664, int_9653)
    
    # Assigning a type to the variable 'tuple_var_assignment_9377' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_9377', subscript_call_result_9665)
    
    # Assigning a Subscript to a Name (line 69):
    
    # Obtaining the type of the subscript
    int_9666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
    
    # Call to read(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to datafile(...): (line 69)
    # Processing the call arguments (line 69)
    str_9670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 43), 'str', 'test-44100Hz-2ch-32bit-float-be.wav')
    # Processing the call keyword arguments (line 69)
    kwargs_9671 = {}
    # Getting the type of 'datafile' (line 69)
    datafile_9669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'datafile', False)
    # Calling datafile(args, kwargs) (line 69)
    datafile_call_result_9672 = invoke(stypy.reporting.localization.Localization(__file__, 69, 34), datafile_9669, *[str_9670], **kwargs_9671)
    
    # Processing the call keyword arguments (line 69)
    # Getting the type of 'mmap' (line 70)
    mmap_9673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 39), 'mmap', False)
    keyword_9674 = mmap_9673
    kwargs_9675 = {'mmap': keyword_9674}
    # Getting the type of 'wavfile' (line 69)
    wavfile_9667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 69)
    read_9668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 21), wavfile_9667, 'read')
    # Calling read(args, kwargs) (line 69)
    read_call_result_9676 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), read_9668, *[datafile_call_result_9672], **kwargs_9675)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___9677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), read_call_result_9676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_9678 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), getitem___9677, int_9666)
    
    # Assigning a type to the variable 'tuple_var_assignment_9378' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_9378', subscript_call_result_9678)
    
    # Assigning a Name to a Name (line 69):
    # Getting the type of 'tuple_var_assignment_9377' (line 69)
    tuple_var_assignment_9377_9679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_9377')
    # Assigning a type to the variable 'rate' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'rate', tuple_var_assignment_9377_9679)
    
    # Assigning a Name to a Name (line 69):
    # Getting the type of 'tuple_var_assignment_9378' (line 69)
    tuple_var_assignment_9378_9680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_var_assignment_9378')
    # Assigning a type to the variable 'data' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'data', tuple_var_assignment_9378_9680)
    
    # Call to assert_equal(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'rate' (line 71)
    rate_9682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'rate', False)
    int_9683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_9684 = {}
    # Getting the type of 'assert_equal' (line 71)
    assert_equal_9681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 71)
    assert_equal_call_result_9685 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert_equal_9681, *[rate_9682, int_9683], **kwargs_9684)
    
    
    # Call to assert_(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to issubdtype(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'data' (line 72)
    data_9689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'data', False)
    # Obtaining the member 'dtype' of a type (line 72)
    dtype_9690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 30), data_9689, 'dtype')
    # Getting the type of 'np' (line 72)
    np_9691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'np', False)
    # Obtaining the member 'float32' of a type (line 72)
    float32_9692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 42), np_9691, 'float32')
    # Processing the call keyword arguments (line 72)
    kwargs_9693 = {}
    # Getting the type of 'np' (line 72)
    np_9687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 72)
    issubdtype_9688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), np_9687, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 72)
    issubdtype_call_result_9694 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), issubdtype_9688, *[dtype_9690, float32_9692], **kwargs_9693)
    
    # Processing the call keyword arguments (line 72)
    kwargs_9695 = {}
    # Getting the type of 'assert_' (line 72)
    assert__9686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 72)
    assert__call_result_9696 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert__9686, *[issubdtype_call_result_9694], **kwargs_9695)
    
    
    # Call to assert_(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'data' (line 73)
    data_9698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'data', False)
    # Obtaining the member 'dtype' of a type (line 73)
    dtype_9699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), data_9698, 'dtype')
    # Obtaining the member 'byteorder' of a type (line 73)
    byteorder_9700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), dtype_9699, 'byteorder')
    str_9701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 40), 'str', '>')
    # Applying the binary operator '==' (line 73)
    result_eq_9702 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 16), '==', byteorder_9700, str_9701)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sys' (line 73)
    sys_9703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 48), 'sys', False)
    # Obtaining the member 'byteorder' of a type (line 73)
    byteorder_9704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 48), sys_9703, 'byteorder')
    str_9705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 65), 'str', 'big')
    # Applying the binary operator '==' (line 73)
    result_eq_9706 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 48), '==', byteorder_9704, str_9705)
    
    
    # Getting the type of 'data' (line 74)
    data_9707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 48), 'data', False)
    # Obtaining the member 'dtype' of a type (line 74)
    dtype_9708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 48), data_9707, 'dtype')
    # Obtaining the member 'byteorder' of a type (line 74)
    byteorder_9709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 48), dtype_9708, 'byteorder')
    str_9710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 72), 'str', '=')
    # Applying the binary operator '==' (line 74)
    result_eq_9711 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 48), '==', byteorder_9709, str_9710)
    
    # Applying the binary operator 'and' (line 73)
    result_and_keyword_9712 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 48), 'and', result_eq_9706, result_eq_9711)
    
    # Applying the binary operator 'or' (line 73)
    result_or_keyword_9713 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 16), 'or', result_eq_9702, result_and_keyword_9712)
    
    # Processing the call keyword arguments (line 73)
    kwargs_9714 = {}
    # Getting the type of 'assert_' (line 73)
    assert__9697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 73)
    assert__call_result_9715 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assert__9697, *[result_or_keyword_9713], **kwargs_9714)
    
    
    # Call to assert_equal(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'data' (line 75)
    data_9717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'data', False)
    # Obtaining the member 'shape' of a type (line 75)
    shape_9718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), data_9717, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_9719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    int_9720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 34), tuple_9719, int_9720)
    # Adding element type (line 75)
    int_9721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 34), tuple_9719, int_9721)
    
    # Processing the call keyword arguments (line 75)
    kwargs_9722 = {}
    # Getting the type of 'assert_equal' (line 75)
    assert_equal_9716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 75)
    assert_equal_call_result_9723 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assert_equal_9716, *[shape_9718, tuple_9719], **kwargs_9722)
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 77, 8), module_type_store, 'data')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_5' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_9724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9724)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_5'
    return stypy_return_type_9724

# Assigning a type to the variable 'test_read_5' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'test_read_5', test_read_5)

@norecursion
def test_read_fail(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_fail'
    module_type_store = module_type_store.open_function_context('test_read_fail', 80, 0, False)
    
    # Passed parameters checking function
    test_read_fail.stypy_localization = localization
    test_read_fail.stypy_type_of_self = None
    test_read_fail.stypy_type_store = module_type_store
    test_read_fail.stypy_function_name = 'test_read_fail'
    test_read_fail.stypy_param_names_list = []
    test_read_fail.stypy_varargs_param_name = None
    test_read_fail.stypy_kwargs_param_name = None
    test_read_fail.stypy_call_defaults = defaults
    test_read_fail.stypy_call_varargs = varargs
    test_read_fail.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_fail', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_fail', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_fail(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_9725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    # Getting the type of 'False' (line 81)
    False_9726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 16), list_9725, False_9726)
    # Adding element type (line 81)
    # Getting the type of 'True' (line 81)
    True_9727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 16), list_9725, True_9727)
    
    # Testing the type of a for loop iterable (line 81)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 4), list_9725)
    # Getting the type of the for loop variable (line 81)
    for_loop_var_9728 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 4), list_9725)
    # Assigning a type to the variable 'mmap' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'mmap', for_loop_var_9728)
    # SSA begins for a for statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to open(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Call to datafile(...): (line 82)
    # Processing the call arguments (line 82)
    str_9731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'str', 'example_1.nc')
    # Processing the call keyword arguments (line 82)
    kwargs_9732 = {}
    # Getting the type of 'datafile' (line 82)
    datafile_9730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'datafile', False)
    # Calling datafile(args, kwargs) (line 82)
    datafile_call_result_9733 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), datafile_9730, *[str_9731], **kwargs_9732)
    
    str_9734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 44), 'str', 'rb')
    # Processing the call keyword arguments (line 82)
    kwargs_9735 = {}
    # Getting the type of 'open' (line 82)
    open_9729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'open', False)
    # Calling open(args, kwargs) (line 82)
    open_call_result_9736 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), open_9729, *[datafile_call_result_9733, str_9734], **kwargs_9735)
    
    # Assigning a type to the variable 'fp' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'fp', open_call_result_9736)
    
    # Call to assert_raises(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'ValueError' (line 83)
    ValueError_9738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'ValueError', False)
    # Getting the type of 'wavfile' (line 83)
    wavfile_9739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 83)
    read_9740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 34), wavfile_9739, 'read')
    # Getting the type of 'fp' (line 83)
    fp_9741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 48), 'fp', False)
    # Processing the call keyword arguments (line 83)
    # Getting the type of 'mmap' (line 83)
    mmap_9742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 57), 'mmap', False)
    keyword_9743 = mmap_9742
    kwargs_9744 = {'mmap': keyword_9743}
    # Getting the type of 'assert_raises' (line 83)
    assert_raises_9737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 83)
    assert_raises_call_result_9745 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert_raises_9737, *[ValueError_9738, read_9740, fp_9741], **kwargs_9744)
    
    
    # Call to close(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_9748 = {}
    # Getting the type of 'fp' (line 84)
    fp_9746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'fp', False)
    # Obtaining the member 'close' of a type (line 84)
    close_9747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), fp_9746, 'close')
    # Calling close(args, kwargs) (line 84)
    close_call_result_9749 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), close_9747, *[], **kwargs_9748)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_fail(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_fail' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_9750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_fail'
    return stypy_return_type_9750

# Assigning a type to the variable 'test_read_fail' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'test_read_fail', test_read_fail)

@norecursion
def test_read_early_eof(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_early_eof'
    module_type_store = module_type_store.open_function_context('test_read_early_eof', 87, 0, False)
    
    # Passed parameters checking function
    test_read_early_eof.stypy_localization = localization
    test_read_early_eof.stypy_type_of_self = None
    test_read_early_eof.stypy_type_store = module_type_store
    test_read_early_eof.stypy_function_name = 'test_read_early_eof'
    test_read_early_eof.stypy_param_names_list = []
    test_read_early_eof.stypy_varargs_param_name = None
    test_read_early_eof.stypy_kwargs_param_name = None
    test_read_early_eof.stypy_call_defaults = defaults
    test_read_early_eof.stypy_call_varargs = varargs
    test_read_early_eof.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_early_eof', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_early_eof', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_early_eof(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_9751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    # Adding element type (line 88)
    # Getting the type of 'False' (line 88)
    False_9752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 16), list_9751, False_9752)
    # Adding element type (line 88)
    # Getting the type of 'True' (line 88)
    True_9753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 16), list_9751, True_9753)
    
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), list_9751)
    # Getting the type of the for loop variable (line 88)
    for_loop_var_9754 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), list_9751)
    # Assigning a type to the variable 'mmap' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'mmap', for_loop_var_9754)
    # SSA begins for a for statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to open(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Call to datafile(...): (line 89)
    # Processing the call arguments (line 89)
    str_9757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 27), 'str', 'test-44100Hz-le-1ch-4bytes-early-eof.wav')
    # Processing the call keyword arguments (line 89)
    kwargs_9758 = {}
    # Getting the type of 'datafile' (line 89)
    datafile_9756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'datafile', False)
    # Calling datafile(args, kwargs) (line 89)
    datafile_call_result_9759 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), datafile_9756, *[str_9757], **kwargs_9758)
    
    str_9760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 72), 'str', 'rb')
    # Processing the call keyword arguments (line 89)
    kwargs_9761 = {}
    # Getting the type of 'open' (line 89)
    open_9755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'open', False)
    # Calling open(args, kwargs) (line 89)
    open_call_result_9762 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), open_9755, *[datafile_call_result_9759, str_9760], **kwargs_9761)
    
    # Assigning a type to the variable 'fp' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'fp', open_call_result_9762)
    
    # Call to assert_raises(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'ValueError' (line 90)
    ValueError_9764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'ValueError', False)
    # Getting the type of 'wavfile' (line 90)
    wavfile_9765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 90)
    read_9766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 34), wavfile_9765, 'read')
    # Getting the type of 'fp' (line 90)
    fp_9767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 48), 'fp', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'mmap' (line 90)
    mmap_9768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 57), 'mmap', False)
    keyword_9769 = mmap_9768
    kwargs_9770 = {'mmap': keyword_9769}
    # Getting the type of 'assert_raises' (line 90)
    assert_raises_9763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 90)
    assert_raises_call_result_9771 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_raises_9763, *[ValueError_9764, read_9766, fp_9767], **kwargs_9770)
    
    
    # Call to close(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_9774 = {}
    # Getting the type of 'fp' (line 91)
    fp_9772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'fp', False)
    # Obtaining the member 'close' of a type (line 91)
    close_9773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), fp_9772, 'close')
    # Calling close(args, kwargs) (line 91)
    close_call_result_9775 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), close_9773, *[], **kwargs_9774)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_early_eof(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_early_eof' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_9776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9776)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_early_eof'
    return stypy_return_type_9776

# Assigning a type to the variable 'test_read_early_eof' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'test_read_early_eof', test_read_early_eof)

@norecursion
def test_read_incomplete_chunk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_read_incomplete_chunk'
    module_type_store = module_type_store.open_function_context('test_read_incomplete_chunk', 94, 0, False)
    
    # Passed parameters checking function
    test_read_incomplete_chunk.stypy_localization = localization
    test_read_incomplete_chunk.stypy_type_of_self = None
    test_read_incomplete_chunk.stypy_type_store = module_type_store
    test_read_incomplete_chunk.stypy_function_name = 'test_read_incomplete_chunk'
    test_read_incomplete_chunk.stypy_param_names_list = []
    test_read_incomplete_chunk.stypy_varargs_param_name = None
    test_read_incomplete_chunk.stypy_kwargs_param_name = None
    test_read_incomplete_chunk.stypy_call_defaults = defaults
    test_read_incomplete_chunk.stypy_call_varargs = varargs
    test_read_incomplete_chunk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_read_incomplete_chunk', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_read_incomplete_chunk', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_read_incomplete_chunk(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 95)
    list_9777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 95)
    # Adding element type (line 95)
    # Getting the type of 'False' (line 95)
    False_9778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 16), list_9777, False_9778)
    # Adding element type (line 95)
    # Getting the type of 'True' (line 95)
    True_9779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 16), list_9777, True_9779)
    
    # Testing the type of a for loop iterable (line 95)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 4), list_9777)
    # Getting the type of the for loop variable (line 95)
    for_loop_var_9780 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 4), list_9777)
    # Assigning a type to the variable 'mmap' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'mmap', for_loop_var_9780)
    # SSA begins for a for statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to open(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Call to datafile(...): (line 96)
    # Processing the call arguments (line 96)
    str_9783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 27), 'str', 'test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav')
    # Processing the call keyword arguments (line 96)
    kwargs_9784 = {}
    # Getting the type of 'datafile' (line 96)
    datafile_9782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'datafile', False)
    # Calling datafile(args, kwargs) (line 96)
    datafile_call_result_9785 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), datafile_9782, *[str_9783], **kwargs_9784)
    
    str_9786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 79), 'str', 'rb')
    # Processing the call keyword arguments (line 96)
    kwargs_9787 = {}
    # Getting the type of 'open' (line 96)
    open_9781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'open', False)
    # Calling open(args, kwargs) (line 96)
    open_call_result_9788 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), open_9781, *[datafile_call_result_9785, str_9786], **kwargs_9787)
    
    # Assigning a type to the variable 'fp' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'fp', open_call_result_9788)
    
    # Call to assert_raises(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'ValueError' (line 97)
    ValueError_9790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'ValueError', False)
    # Getting the type of 'wavfile' (line 97)
    wavfile_9791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 34), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 97)
    read_9792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 34), wavfile_9791, 'read')
    # Getting the type of 'fp' (line 97)
    fp_9793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 48), 'fp', False)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'mmap' (line 97)
    mmap_9794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 57), 'mmap', False)
    keyword_9795 = mmap_9794
    kwargs_9796 = {'mmap': keyword_9795}
    # Getting the type of 'assert_raises' (line 97)
    assert_raises_9789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 97)
    assert_raises_call_result_9797 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_raises_9789, *[ValueError_9790, read_9792, fp_9793], **kwargs_9796)
    
    
    # Call to close(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_9800 = {}
    # Getting the type of 'fp' (line 98)
    fp_9798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'fp', False)
    # Obtaining the member 'close' of a type (line 98)
    close_9799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), fp_9798, 'close')
    # Calling close(args, kwargs) (line 98)
    close_call_result_9801 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), close_9799, *[], **kwargs_9800)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_read_incomplete_chunk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_read_incomplete_chunk' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_9802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9802)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_read_incomplete_chunk'
    return stypy_return_type_9802

# Assigning a type to the variable 'test_read_incomplete_chunk' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'test_read_incomplete_chunk', test_read_incomplete_chunk)

@norecursion
def _check_roundtrip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_roundtrip'
    module_type_store = module_type_store.open_function_context('_check_roundtrip', 101, 0, False)
    
    # Passed parameters checking function
    _check_roundtrip.stypy_localization = localization
    _check_roundtrip.stypy_type_of_self = None
    _check_roundtrip.stypy_type_store = module_type_store
    _check_roundtrip.stypy_function_name = '_check_roundtrip'
    _check_roundtrip.stypy_param_names_list = ['realfile', 'rate', 'dtype', 'channels']
    _check_roundtrip.stypy_varargs_param_name = None
    _check_roundtrip.stypy_kwargs_param_name = None
    _check_roundtrip.stypy_call_defaults = defaults
    _check_roundtrip.stypy_call_varargs = varargs
    _check_roundtrip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_roundtrip', ['realfile', 'rate', 'dtype', 'channels'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_roundtrip', localization, ['realfile', 'rate', 'dtype', 'channels'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_roundtrip(...)' code ##################

    
    # Getting the type of 'realfile' (line 102)
    realfile_9803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'realfile')
    # Testing the type of an if condition (line 102)
    if_condition_9804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), realfile_9803)
    # Assigning a type to the variable 'if_condition_9804' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_9804', if_condition_9804)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 103):
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    int_9805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
    
    # Call to mkstemp(...): (line 103)
    # Processing the call keyword arguments (line 103)
    str_9808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 46), 'str', '.wav')
    keyword_9809 = str_9808
    kwargs_9810 = {'suffix': keyword_9809}
    # Getting the type of 'tempfile' (line 103)
    tempfile_9806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 103)
    mkstemp_9807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 22), tempfile_9806, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 103)
    mkstemp_call_result_9811 = invoke(stypy.reporting.localization.Localization(__file__, 103, 22), mkstemp_9807, *[], **kwargs_9810)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___9812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), mkstemp_call_result_9811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_9813 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), getitem___9812, int_9805)
    
    # Assigning a type to the variable 'tuple_var_assignment_9379' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_9379', subscript_call_result_9813)
    
    # Assigning a Subscript to a Name (line 103):
    
    # Obtaining the type of the subscript
    int_9814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
    
    # Call to mkstemp(...): (line 103)
    # Processing the call keyword arguments (line 103)
    str_9817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 46), 'str', '.wav')
    keyword_9818 = str_9817
    kwargs_9819 = {'suffix': keyword_9818}
    # Getting the type of 'tempfile' (line 103)
    tempfile_9815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 103)
    mkstemp_9816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 22), tempfile_9815, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 103)
    mkstemp_call_result_9820 = invoke(stypy.reporting.localization.Localization(__file__, 103, 22), mkstemp_9816, *[], **kwargs_9819)
    
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___9821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), mkstemp_call_result_9820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_9822 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), getitem___9821, int_9814)
    
    # Assigning a type to the variable 'tuple_var_assignment_9380' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_9380', subscript_call_result_9822)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_var_assignment_9379' (line 103)
    tuple_var_assignment_9379_9823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_9379')
    # Assigning a type to the variable 'fd' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'fd', tuple_var_assignment_9379_9823)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_var_assignment_9380' (line 103)
    tuple_var_assignment_9380_9824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_9380')
    # Assigning a type to the variable 'tmpfile' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'tmpfile', tuple_var_assignment_9380_9824)
    
    # Call to close(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'fd' (line 104)
    fd_9827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'fd', False)
    # Processing the call keyword arguments (line 104)
    kwargs_9828 = {}
    # Getting the type of 'os' (line 104)
    os_9825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'os', False)
    # Obtaining the member 'close' of a type (line 104)
    close_9826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), os_9825, 'close')
    # Calling close(args, kwargs) (line 104)
    close_call_result_9829 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), close_9826, *[fd_9827], **kwargs_9828)
    
    # SSA branch for the else part of an if statement (line 102)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to BytesIO(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_9831 = {}
    # Getting the type of 'BytesIO' (line 106)
    BytesIO_9830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'BytesIO', False)
    # Calling BytesIO(args, kwargs) (line 106)
    BytesIO_call_result_9832 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), BytesIO_9830, *[], **kwargs_9831)
    
    # Assigning a type to the variable 'tmpfile' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tmpfile', BytesIO_call_result_9832)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Try-finally block (line 107)
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to rand(...): (line 108)
    # Processing the call arguments (line 108)
    int_9836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 30), 'int')
    # Getting the type of 'channels' (line 108)
    channels_9837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'channels', False)
    # Processing the call keyword arguments (line 108)
    kwargs_9838 = {}
    # Getting the type of 'np' (line 108)
    np_9833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'np', False)
    # Obtaining the member 'random' of a type (line 108)
    random_9834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), np_9833, 'random')
    # Obtaining the member 'rand' of a type (line 108)
    rand_9835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), random_9834, 'rand')
    # Calling rand(args, kwargs) (line 108)
    rand_call_result_9839 = invoke(stypy.reporting.localization.Localization(__file__, 108, 15), rand_9835, *[int_9836, channels_9837], **kwargs_9838)
    
    # Assigning a type to the variable 'data' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'data', rand_call_result_9839)
    
    
    # Getting the type of 'channels' (line 109)
    channels_9840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'channels')
    int_9841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
    # Applying the binary operator '==' (line 109)
    result_eq_9842 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '==', channels_9840, int_9841)
    
    # Testing the type of an if condition (line 109)
    if_condition_9843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_9842)
    # Assigning a type to the variable 'if_condition_9843' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_9843', if_condition_9843)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 110):
    
    # Assigning a Subscript to a Name (line 110):
    
    # Obtaining the type of the subscript
    slice_9844 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 110, 19), None, None, None)
    int_9845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 26), 'int')
    # Getting the type of 'data' (line 110)
    data_9846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'data')
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___9847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 19), data_9846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_9848 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), getitem___9847, (slice_9844, int_9845))
    
    # Assigning a type to the variable 'data' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'data', subscript_call_result_9848)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'dtype' (line 111)
    dtype_9849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'dtype')
    # Obtaining the member 'kind' of a type (line 111)
    kind_9850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), dtype_9849, 'kind')
    str_9851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 25), 'str', 'f')
    # Applying the binary operator '==' (line 111)
    result_eq_9852 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 11), '==', kind_9850, str_9851)
    
    # Testing the type of an if condition (line 111)
    if_condition_9853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), result_eq_9852)
    # Assigning a type to the variable 'if_condition_9853' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_9853', if_condition_9853)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to astype(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'dtype' (line 113)
    dtype_9856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'dtype', False)
    # Processing the call keyword arguments (line 113)
    kwargs_9857 = {}
    # Getting the type of 'data' (line 113)
    data_9854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'data', False)
    # Obtaining the member 'astype' of a type (line 113)
    astype_9855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), data_9854, 'astype')
    # Calling astype(args, kwargs) (line 113)
    astype_call_result_9858 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), astype_9855, *[dtype_9856], **kwargs_9857)
    
    # Assigning a type to the variable 'data' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'data', astype_call_result_9858)
    # SSA branch for the else part of an if statement (line 111)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to astype(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'dtype' (line 115)
    dtype_9863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'dtype', False)
    # Processing the call keyword arguments (line 115)
    kwargs_9864 = {}
    # Getting the type of 'data' (line 115)
    data_9859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'data', False)
    int_9860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'int')
    # Applying the binary operator '*' (line 115)
    result_mul_9861 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 20), '*', data_9859, int_9860)
    
    # Obtaining the member 'astype' of a type (line 115)
    astype_9862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), result_mul_9861, 'astype')
    # Calling astype(args, kwargs) (line 115)
    astype_call_result_9865 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), astype_9862, *[dtype_9863], **kwargs_9864)
    
    # Assigning a type to the variable 'data' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'data', astype_call_result_9865)
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'tmpfile' (line 117)
    tmpfile_9868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'tmpfile', False)
    # Getting the type of 'rate' (line 117)
    rate_9869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 31), 'rate', False)
    # Getting the type of 'data' (line 117)
    data_9870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'data', False)
    # Processing the call keyword arguments (line 117)
    kwargs_9871 = {}
    # Getting the type of 'wavfile' (line 117)
    wavfile_9866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'wavfile', False)
    # Obtaining the member 'write' of a type (line 117)
    write_9867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), wavfile_9866, 'write')
    # Calling write(args, kwargs) (line 117)
    write_call_result_9872 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), write_9867, *[tmpfile_9868, rate_9869, data_9870], **kwargs_9871)
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_9873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    # Adding element type (line 119)
    # Getting the type of 'False' (line 119)
    False_9874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), list_9873, False_9874)
    # Adding element type (line 119)
    # Getting the type of 'True' (line 119)
    True_9875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 20), list_9873, True_9875)
    
    # Testing the type of a for loop iterable (line 119)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 8), list_9873)
    # Getting the type of the for loop variable (line 119)
    for_loop_var_9876 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 8), list_9873)
    # Assigning a type to the variable 'mmap' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'mmap', for_loop_var_9876)
    # SSA begins for a for statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 120):
    
    # Assigning a Subscript to a Name (line 120):
    
    # Obtaining the type of the subscript
    int_9877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
    
    # Call to read(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'tmpfile' (line 120)
    tmpfile_9880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'tmpfile', False)
    # Processing the call keyword arguments (line 120)
    # Getting the type of 'mmap' (line 120)
    mmap_9881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 54), 'mmap', False)
    keyword_9882 = mmap_9881
    kwargs_9883 = {'mmap': keyword_9882}
    # Getting the type of 'wavfile' (line 120)
    wavfile_9878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 120)
    read_9879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), wavfile_9878, 'read')
    # Calling read(args, kwargs) (line 120)
    read_call_result_9884 = invoke(stypy.reporting.localization.Localization(__file__, 120, 27), read_9879, *[tmpfile_9880], **kwargs_9883)
    
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___9885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), read_call_result_9884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_9886 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), getitem___9885, int_9877)
    
    # Assigning a type to the variable 'tuple_var_assignment_9381' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_9381', subscript_call_result_9886)
    
    # Assigning a Subscript to a Name (line 120):
    
    # Obtaining the type of the subscript
    int_9887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
    
    # Call to read(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'tmpfile' (line 120)
    tmpfile_9890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'tmpfile', False)
    # Processing the call keyword arguments (line 120)
    # Getting the type of 'mmap' (line 120)
    mmap_9891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 54), 'mmap', False)
    keyword_9892 = mmap_9891
    kwargs_9893 = {'mmap': keyword_9892}
    # Getting the type of 'wavfile' (line 120)
    wavfile_9888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'wavfile', False)
    # Obtaining the member 'read' of a type (line 120)
    read_9889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), wavfile_9888, 'read')
    # Calling read(args, kwargs) (line 120)
    read_call_result_9894 = invoke(stypy.reporting.localization.Localization(__file__, 120, 27), read_9889, *[tmpfile_9890], **kwargs_9893)
    
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___9895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), read_call_result_9894, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_9896 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), getitem___9895, int_9887)
    
    # Assigning a type to the variable 'tuple_var_assignment_9382' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_9382', subscript_call_result_9896)
    
    # Assigning a Name to a Name (line 120):
    # Getting the type of 'tuple_var_assignment_9381' (line 120)
    tuple_var_assignment_9381_9897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_9381')
    # Assigning a type to the variable 'rate2' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'rate2', tuple_var_assignment_9381_9897)
    
    # Assigning a Name to a Name (line 120):
    # Getting the type of 'tuple_var_assignment_9382' (line 120)
    tuple_var_assignment_9382_9898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'tuple_var_assignment_9382')
    # Assigning a type to the variable 'data2' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'data2', tuple_var_assignment_9382_9898)
    
    # Call to assert_equal(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'rate' (line 122)
    rate_9900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'rate', False)
    # Getting the type of 'rate2' (line 122)
    rate2_9901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'rate2', False)
    # Processing the call keyword arguments (line 122)
    kwargs_9902 = {}
    # Getting the type of 'assert_equal' (line 122)
    assert_equal_9899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 122)
    assert_equal_call_result_9903 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), assert_equal_9899, *[rate_9900, rate2_9901], **kwargs_9902)
    
    
    # Call to assert_(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Getting the type of 'data2' (line 123)
    data2_9905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'data2', False)
    # Obtaining the member 'dtype' of a type (line 123)
    dtype_9906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 20), data2_9905, 'dtype')
    # Obtaining the member 'byteorder' of a type (line 123)
    byteorder_9907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 20), dtype_9906, 'byteorder')
    
    # Obtaining an instance of the builtin type 'tuple' (line 123)
    tuple_9908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 123)
    # Adding element type (line 123)
    str_9909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 46), 'str', '<')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 46), tuple_9908, str_9909)
    # Adding element type (line 123)
    str_9910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 51), 'str', '=')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 46), tuple_9908, str_9910)
    # Adding element type (line 123)
    str_9911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 56), 'str', '|')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 46), tuple_9908, str_9911)
    
    # Applying the binary operator 'in' (line 123)
    result_contains_9912 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 20), 'in', byteorder_9907, tuple_9908)
    
    # Processing the call keyword arguments (line 123)
    # Getting the type of 'data2' (line 123)
    data2_9913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 66), 'data2', False)
    # Obtaining the member 'dtype' of a type (line 123)
    dtype_9914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 66), data2_9913, 'dtype')
    keyword_9915 = dtype_9914
    kwargs_9916 = {'msg': keyword_9915}
    # Getting the type of 'assert_' (line 123)
    assert__9904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 123)
    assert__call_result_9917 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), assert__9904, *[result_contains_9912], **kwargs_9916)
    
    
    # Call to assert_array_equal(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'data' (line 124)
    data_9919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'data', False)
    # Getting the type of 'data2' (line 124)
    data2_9920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'data2', False)
    # Processing the call keyword arguments (line 124)
    kwargs_9921 = {}
    # Getting the type of 'assert_array_equal' (line 124)
    assert_array_equal_9918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 124)
    assert_array_equal_call_result_9922 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), assert_array_equal_9918, *[data_9919, data2_9920], **kwargs_9921)
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 126, 12), module_type_store, 'data2')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 107)
    
    # Getting the type of 'realfile' (line 128)
    realfile_9923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'realfile')
    # Testing the type of an if condition (line 128)
    if_condition_9924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 8), realfile_9923)
    # Assigning a type to the variable 'if_condition_9924' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'if_condition_9924', if_condition_9924)
    # SSA begins for if statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to unlink(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'tmpfile' (line 129)
    tmpfile_9927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 22), 'tmpfile', False)
    # Processing the call keyword arguments (line 129)
    kwargs_9928 = {}
    # Getting the type of 'os' (line 129)
    os_9925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'os', False)
    # Obtaining the member 'unlink' of a type (line 129)
    unlink_9926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), os_9925, 'unlink')
    # Calling unlink(args, kwargs) (line 129)
    unlink_call_result_9929 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), unlink_9926, *[tmpfile_9927], **kwargs_9928)
    
    # SSA join for if statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # ################# End of '_check_roundtrip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_roundtrip' in the type store
    # Getting the type of 'stypy_return_type' (line 101)
    stypy_return_type_9930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9930)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_roundtrip'
    return stypy_return_type_9930

# Assigning a type to the variable '_check_roundtrip' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '_check_roundtrip', _check_roundtrip)

@norecursion
def test_write_roundtrip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_write_roundtrip'
    module_type_store = module_type_store.open_function_context('test_write_roundtrip', 132, 0, False)
    
    # Passed parameters checking function
    test_write_roundtrip.stypy_localization = localization
    test_write_roundtrip.stypy_type_of_self = None
    test_write_roundtrip.stypy_type_store = module_type_store
    test_write_roundtrip.stypy_function_name = 'test_write_roundtrip'
    test_write_roundtrip.stypy_param_names_list = []
    test_write_roundtrip.stypy_varargs_param_name = None
    test_write_roundtrip.stypy_kwargs_param_name = None
    test_write_roundtrip.stypy_call_defaults = defaults
    test_write_roundtrip.stypy_call_varargs = varargs
    test_write_roundtrip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_write_roundtrip', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_write_roundtrip', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_write_roundtrip(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_9931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    # Getting the type of 'False' (line 133)
    False_9932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), tuple_9931, False_9932)
    # Adding element type (line 133)
    # Getting the type of 'True' (line 133)
    True_9933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), tuple_9931, True_9933)
    
    # Testing the type of a for loop iterable (line 133)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_9931)
    # Getting the type of the for loop variable (line 133)
    for_loop_var_9934 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 133, 4), tuple_9931)
    # Assigning a type to the variable 'realfile' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'realfile', for_loop_var_9934)
    # SSA begins for a for statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 134)
    tuple_9935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 134)
    # Adding element type (line 134)
    str_9936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'str', 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_9935, str_9936)
    # Adding element type (line 134)
    str_9937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 31), 'str', 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_9935, str_9937)
    # Adding element type (line 134)
    str_9938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_9935, str_9938)
    # Adding element type (line 134)
    str_9939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 41), 'str', 'g')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_9935, str_9939)
    # Adding element type (line 134)
    str_9940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 46), 'str', 'q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 26), tuple_9935, str_9940)
    
    # Testing the type of a for loop iterable (line 134)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 8), tuple_9935)
    # Getting the type of the for loop variable (line 134)
    for_loop_var_9941 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 8), tuple_9935)
    # Assigning a type to the variable 'dtypechar' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'dtypechar', for_loop_var_9941)
    # SSA begins for a for statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 135)
    tuple_9942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 135)
    # Adding element type (line 135)
    int_9943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 25), tuple_9942, int_9943)
    # Adding element type (line 135)
    int_9944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 25), tuple_9942, int_9944)
    # Adding element type (line 135)
    int_9945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 25), tuple_9942, int_9945)
    # Adding element type (line 135)
    int_9946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 25), tuple_9942, int_9946)
    
    # Testing the type of a for loop iterable (line 135)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 12), tuple_9942)
    # Getting the type of the for loop variable (line 135)
    for_loop_var_9947 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 12), tuple_9942)
    # Assigning a type to the variable 'size' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'size', for_loop_var_9947)
    # SSA begins for a for statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 136)
    size_9948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'size')
    int_9949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 27), 'int')
    # Applying the binary operator '==' (line 136)
    result_eq_9950 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 19), '==', size_9948, int_9949)
    
    
    # Getting the type of 'dtypechar' (line 136)
    dtypechar_9951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 33), 'dtypechar')
    str_9952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 46), 'str', 'i')
    # Applying the binary operator '==' (line 136)
    result_eq_9953 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 33), '==', dtypechar_9951, str_9952)
    
    # Applying the binary operator 'and' (line 136)
    result_and_keyword_9954 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 19), 'and', result_eq_9950, result_eq_9953)
    
    # Testing the type of an if condition (line 136)
    if_condition_9955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 16), result_and_keyword_9954)
    # Assigning a type to the variable 'if_condition_9955' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'if_condition_9955', if_condition_9955)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 139)
    size_9956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'size')
    int_9957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 26), 'int')
    # Applying the binary operator '>' (line 139)
    result_gt_9958 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 19), '>', size_9956, int_9957)
    
    
    # Getting the type of 'dtypechar' (line 139)
    dtypechar_9959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 32), 'dtypechar')
    str_9960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 45), 'str', 'u')
    # Applying the binary operator '==' (line 139)
    result_eq_9961 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 32), '==', dtypechar_9959, str_9960)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_9962 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 19), 'and', result_gt_9958, result_eq_9961)
    
    # Testing the type of an if condition (line 139)
    if_condition_9963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 16), result_and_keyword_9962)
    # Assigning a type to the variable 'if_condition_9963' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'if_condition_9963', if_condition_9963)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 142)
    size_9964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'size')
    int_9965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 28), 'int')
    # Applying the binary operator '==' (line 142)
    result_eq_9966 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 20), '==', size_9964, int_9965)
    
    
    # Getting the type of 'size' (line 142)
    size_9967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 33), 'size')
    int_9968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 41), 'int')
    # Applying the binary operator '==' (line 142)
    result_eq_9969 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 33), '==', size_9967, int_9968)
    
    # Applying the binary operator 'or' (line 142)
    result_or_keyword_9970 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 20), 'or', result_eq_9966, result_eq_9969)
    
    
    # Getting the type of 'dtypechar' (line 142)
    dtypechar_9971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'dtypechar')
    str_9972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 61), 'str', 'f')
    # Applying the binary operator '==' (line 142)
    result_eq_9973 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 48), '==', dtypechar_9971, str_9972)
    
    # Applying the binary operator 'and' (line 142)
    result_and_keyword_9974 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 19), 'and', result_or_keyword_9970, result_eq_9973)
    
    # Testing the type of an if condition (line 142)
    if_condition_9975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 16), result_and_keyword_9974)
    # Assigning a type to the variable 'if_condition_9975' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'if_condition_9975', if_condition_9975)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'dtypechar' (line 145)
    dtypechar_9976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'dtypechar')
    str_9977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 32), 'str', 'gq')
    # Applying the binary operator 'in' (line 145)
    result_contains_9978 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 19), 'in', dtypechar_9976, str_9977)
    
    # Testing the type of an if condition (line 145)
    if_condition_9979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 16), result_contains_9978)
    # Assigning a type to the variable 'if_condition_9979' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'if_condition_9979', if_condition_9979)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'size' (line 147)
    size_9980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'size')
    int_9981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 31), 'int')
    # Applying the binary operator '==' (line 147)
    result_eq_9982 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 23), '==', size_9980, int_9981)
    
    # Testing the type of an if condition (line 147)
    if_condition_9983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 20), result_eq_9982)
    # Assigning a type to the variable 'if_condition_9983' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'if_condition_9983', if_condition_9983)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 148):
    
    # Assigning a Str to a Name (line 148):
    str_9984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 31), 'str', '')
    # Assigning a type to the variable 'size' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'size', str_9984)
    # SSA branch for the else part of an if statement (line 147)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_9985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    str_9986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 35), 'str', '>')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 35), tuple_9985, str_9986)
    # Adding element type (line 152)
    str_9987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 40), 'str', '<')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 35), tuple_9985, str_9987)
    
    # Testing the type of a for loop iterable (line 152)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 16), tuple_9985)
    # Getting the type of the for loop variable (line 152)
    for_loop_var_9988 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 16), tuple_9985)
    # Assigning a type to the variable 'endianness' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'endianness', for_loop_var_9988)
    # SSA begins for a for statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 153)
    size_9989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'size')
    int_9990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 31), 'int')
    # Applying the binary operator '==' (line 153)
    result_eq_9991 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 23), '==', size_9989, int_9990)
    
    
    # Getting the type of 'endianness' (line 153)
    endianness_9992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'endianness')
    str_9993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 51), 'str', '<')
    # Applying the binary operator '==' (line 153)
    result_eq_9994 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 37), '==', endianness_9992, str_9993)
    
    # Applying the binary operator 'and' (line 153)
    result_and_keyword_9995 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 23), 'and', result_eq_9991, result_eq_9994)
    
    # Testing the type of an if condition (line 153)
    if_condition_9996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 20), result_and_keyword_9995)
    # Assigning a type to the variable 'if_condition_9996' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'if_condition_9996', if_condition_9996)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 155)
    tuple_9997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 155)
    # Adding element type (line 155)
    int_9998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 33), tuple_9997, int_9998)
    # Adding element type (line 155)
    int_9999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 33), tuple_9997, int_9999)
    
    # Testing the type of a for loop iterable (line 155)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 20), tuple_9997)
    # Getting the type of the for loop variable (line 155)
    for_loop_var_10000 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 20), tuple_9997)
    # Assigning a type to the variable 'rate' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'rate', for_loop_var_10000)
    # SSA begins for a for statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_10001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    int_10002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 41), tuple_10001, int_10002)
    # Adding element type (line 156)
    int_10003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 41), tuple_10001, int_10003)
    # Adding element type (line 156)
    int_10004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 41), tuple_10001, int_10004)
    
    # Testing the type of a for loop iterable (line 156)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 24), tuple_10001)
    # Getting the type of the for loop variable (line 156)
    for_loop_var_10005 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 24), tuple_10001)
    # Assigning a type to the variable 'channels' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'channels', for_loop_var_10005)
    # SSA begins for a for statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to dtype(...): (line 157)
    # Processing the call arguments (line 157)
    str_10008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 42), 'str', '%s%s%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 157)
    tuple_10009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 157)
    # Adding element type (line 157)
    # Getting the type of 'endianness' (line 157)
    endianness_10010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 54), 'endianness', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 54), tuple_10009, endianness_10010)
    # Adding element type (line 157)
    # Getting the type of 'dtypechar' (line 157)
    dtypechar_10011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 66), 'dtypechar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 54), tuple_10009, dtypechar_10011)
    # Adding element type (line 157)
    # Getting the type of 'size' (line 157)
    size_10012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 77), 'size', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 54), tuple_10009, size_10012)
    
    # Applying the binary operator '%' (line 157)
    result_mod_10013 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 42), '%', str_10008, tuple_10009)
    
    # Processing the call keyword arguments (line 157)
    kwargs_10014 = {}
    # Getting the type of 'np' (line 157)
    np_10006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'np', False)
    # Obtaining the member 'dtype' of a type (line 157)
    dtype_10007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 33), np_10006, 'dtype')
    # Calling dtype(args, kwargs) (line 157)
    dtype_call_result_10015 = invoke(stypy.reporting.localization.Localization(__file__, 157, 33), dtype_10007, *[result_mod_10013], **kwargs_10014)
    
    # Assigning a type to the variable 'dt' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'dt', dtype_call_result_10015)
    
    # Call to _check_roundtrip(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'realfile' (line 158)
    realfile_10017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 45), 'realfile', False)
    # Getting the type of 'rate' (line 158)
    rate_10018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'rate', False)
    # Getting the type of 'dt' (line 158)
    dt_10019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 61), 'dt', False)
    # Getting the type of 'channels' (line 158)
    channels_10020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 65), 'channels', False)
    # Processing the call keyword arguments (line 158)
    kwargs_10021 = {}
    # Getting the type of '_check_roundtrip' (line 158)
    _check_roundtrip_10016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), '_check_roundtrip', False)
    # Calling _check_roundtrip(args, kwargs) (line 158)
    _check_roundtrip_call_result_10022 = invoke(stypy.reporting.localization.Localization(__file__, 158, 28), _check_roundtrip_10016, *[realfile_10017, rate_10018, dt_10019, channels_10020], **kwargs_10021)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_write_roundtrip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_write_roundtrip' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_10023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10023)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_write_roundtrip'
    return stypy_return_type_10023

# Assigning a type to the variable 'test_write_roundtrip' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'test_write_roundtrip', test_write_roundtrip)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
