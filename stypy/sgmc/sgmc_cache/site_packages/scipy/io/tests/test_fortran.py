
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Tests for fortran sequential files '''
2: 
3: import tempfile
4: import shutil
5: from os import path, unlink
6: from glob import iglob
7: import re
8: 
9: from numpy.testing import assert_equal, assert_allclose
10: import numpy as np
11: 
12: from scipy.io import FortranFile, _test_fortran
13: 
14: 
15: DATA_PATH = path.join(path.dirname(__file__), 'data')
16: 
17: 
18: def test_fortranfiles_read():
19:     for filename in iglob(path.join(DATA_PATH, "fortran-*-*x*x*.dat")):
20:         m = re.search(r'fortran-([^-]+)-(\d+)x(\d+)x(\d+).dat', filename, re.I)
21:         if not m:
22:             raise RuntimeError("Couldn't match %s filename to regex" % filename)
23: 
24:         dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
25: 
26:         dtype = m.group(1).replace('s', '<')
27: 
28:         f = FortranFile(filename, 'r', '<u4')
29:         data = f.read_record(dtype=dtype).reshape(dims, order='F')
30:         f.close()
31: 
32:         expected = np.arange(np.prod(dims)).reshape(dims).astype(dtype)
33:         assert_equal(data, expected)
34: 
35: 
36: def test_fortranfiles_mixed_record():
37:     filename = path.join(DATA_PATH, "fortran-mixed.dat")
38:     with FortranFile(filename, 'r', '<u4') as f:
39:         record = f.read_record('<i4,<f4,<i8,(2)<f8')
40: 
41:     assert_equal(record['f0'][0], 1)
42:     assert_allclose(record['f1'][0], 2.3)
43:     assert_equal(record['f2'][0], 4)
44:     assert_allclose(record['f3'][0], [5.6, 7.8])
45: 
46: 
47: def test_fortranfiles_write():
48:     for filename in iglob(path.join(DATA_PATH, "fortran-*-*x*x*.dat")):
49:         m = re.search(r'fortran-([^-]+)-(\d+)x(\d+)x(\d+).dat', filename, re.I)
50:         if not m:
51:             raise RuntimeError("Couldn't match %s filename to regex" % filename)
52:         dims = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
53: 
54:         dtype = m.group(1).replace('s', '<')
55:         data = np.arange(np.prod(dims)).reshape(dims).astype(dtype)
56: 
57:         tmpdir = tempfile.mkdtemp()
58:         try:
59:             testFile = path.join(tmpdir,path.basename(filename))
60:             f = FortranFile(testFile, 'w','<u4')
61:             f.write_record(data.T)
62:             f.close()
63:             originalfile = open(filename, 'rb')
64:             newfile = open(testFile, 'rb')
65:             assert_equal(originalfile.read(), newfile.read(),
66:                          err_msg=filename)
67:             originalfile.close()
68:             newfile.close()
69:         finally:
70:             shutil.rmtree(tmpdir)
71: 
72: 
73: def test_fortranfile_read_mixed_record():
74:     # The data file fortran-3x3d-2i.dat contains the program that
75:     # produced it at the end.
76:     #
77:     # double precision :: a(3,3)
78:     # integer :: b(2)
79:     # ...
80:     # open(1, file='fortran-3x3d-2i.dat', form='unformatted')
81:     # write(1) a, b
82:     # close(1)
83:     #
84: 
85:     filename = path.join(DATA_PATH, "fortran-3x3d-2i.dat")
86:     with FortranFile(filename, 'r', '<u4') as f:
87:         record = f.read_record('(3,3)f8', '2i4')
88: 
89:     ax = np.arange(3*3).reshape(3, 3).astype(np.double)
90:     bx = np.array([-1, -2], dtype=np.int32)
91: 
92:     assert_equal(record[0], ax.T)
93:     assert_equal(record[1], bx.T)
94: 
95: 
96: def test_fortranfile_write_mixed_record(tmpdir):
97:     tf = path.join(str(tmpdir), 'test.dat')
98: 
99:     records = [
100:         (('f4', 'f4', 'i4'), (np.float32(2), np.float32(3), np.int32(100))),
101:         (('4f4', '(3,3)f4', '8i4'), (np.random.randint(255, size=[4]).astype(np.float32),
102:                                      np.random.randint(255, size=[3, 3]).astype(np.float32),
103:                                      np.random.randint(255, size=[8]).astype(np.int32)))
104:     ]
105: 
106:     for dtype, a in records:
107:         with FortranFile(tf, 'w') as f:
108:             f.write_record(*a)
109: 
110:         with FortranFile(tf, 'r') as f:
111:             b = f.read_record(*dtype)
112: 
113:         assert_equal(len(a), len(b))
114: 
115:         for aa, bb in zip(a, b):
116:             assert_equal(bb, aa)
117: 
118: 
119: def test_fortran_roundtrip(tmpdir):
120:     filename = path.join(str(tmpdir), 'test.dat')
121: 
122:     np.random.seed(1)
123: 
124:     # double precision
125:     m, n, k = 5, 3, 2
126:     a = np.random.randn(m, n, k)
127:     with FortranFile(filename, 'w') as f:
128:         f.write_record(a.T)
129:     a2 = _test_fortran.read_unformatted_double(m, n, k, filename)
130:     with FortranFile(filename, 'r') as f:
131:         a3 = f.read_record('(2,3,5)f8').T
132:     assert_equal(a2, a)
133:     assert_equal(a3, a)
134: 
135:     # integer
136:     m, n, k = 5, 3, 2
137:     a = np.random.randn(m, n, k).astype(np.int32)
138:     with FortranFile(filename, 'w') as f:
139:         f.write_record(a.T)
140:     a2 = _test_fortran.read_unformatted_int(m, n, k, filename)
141:     with FortranFile(filename, 'r') as f:
142:         a3 = f.read_record('(2,3,5)i4').T
143:     assert_equal(a2, a)
144:     assert_equal(a3, a)
145: 
146:     # mixed
147:     m, n, k = 5, 3, 2
148:     a = np.random.randn(m, n)
149:     b = np.random.randn(k).astype(np.intc)
150:     with FortranFile(filename, 'w') as f:
151:         f.write_record(a.T, b.T)
152:     a2, b2 = _test_fortran.read_unformatted_mixed(m, n, k, filename)
153:     with FortranFile(filename, 'r') as f:
154:         a3, b3 = f.read_record('(3,5)f8', '2i4')
155:         a3 = a3.T
156:     assert_equal(a2, a)
157:     assert_equal(a3, a)
158:     assert_equal(b2, b)
159:     assert_equal(b3, b)
160: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Tests for fortran sequential files ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import tempfile' statement (line 3)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import shutil' statement (line 4)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from os import path, unlink' statement (line 5)
try:
    from os import path, unlink

except:
    path = UndefinedType
    unlink = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', None, module_type_store, ['path', 'unlink'], [path, unlink])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from glob import iglob' statement (line 6)
try:
    from glob import iglob

except:
    iglob = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'glob', None, module_type_store, ['iglob'], [iglob])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import re' statement (line 7)
import re

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_794 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_794) is not StypyTypeError):

    if (import_794 != 'pyd_module'):
        __import__(import_794)
        sys_modules_795 = sys.modules[import_794]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_795.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_795, sys_modules_795.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_794)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_796 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_796) is not StypyTypeError):

    if (import_796 != 'pyd_module'):
        __import__(import_796)
        sys_modules_797 = sys.modules[import_796]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_797.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_796)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.io import FortranFile, _test_fortran' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_798 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io')

if (type(import_798) is not StypyTypeError):

    if (import_798 != 'pyd_module'):
        __import__(import_798)
        sys_modules_799 = sys.modules[import_798]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io', sys_modules_799.module_type_store, module_type_store, ['FortranFile', '_test_fortran'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_799, sys_modules_799.module_type_store, module_type_store)
    else:
        from scipy.io import FortranFile, _test_fortran

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io', None, module_type_store, ['FortranFile', '_test_fortran'], [FortranFile, _test_fortran])

else:
    # Assigning a type to the variable 'scipy.io' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io', import_798)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')


# Assigning a Call to a Name (line 15):

# Assigning a Call to a Name (line 15):

# Call to join(...): (line 15)
# Processing the call arguments (line 15)

# Call to dirname(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of '__file__' (line 15)
file___804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 35), '__file__', False)
# Processing the call keyword arguments (line 15)
kwargs_805 = {}
# Getting the type of 'path' (line 15)
path_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'path', False)
# Obtaining the member 'dirname' of a type (line 15)
dirname_803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 22), path_802, 'dirname')
# Calling dirname(args, kwargs) (line 15)
dirname_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 15, 22), dirname_803, *[file___804], **kwargs_805)

str_807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 46), 'str', 'data')
# Processing the call keyword arguments (line 15)
kwargs_808 = {}
# Getting the type of 'path' (line 15)
path_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'path', False)
# Obtaining the member 'join' of a type (line 15)
join_801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 12), path_800, 'join')
# Calling join(args, kwargs) (line 15)
join_call_result_809 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), join_801, *[dirname_call_result_806, str_807], **kwargs_808)

# Assigning a type to the variable 'DATA_PATH' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'DATA_PATH', join_call_result_809)

@norecursion
def test_fortranfiles_read(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fortranfiles_read'
    module_type_store = module_type_store.open_function_context('test_fortranfiles_read', 18, 0, False)
    
    # Passed parameters checking function
    test_fortranfiles_read.stypy_localization = localization
    test_fortranfiles_read.stypy_type_of_self = None
    test_fortranfiles_read.stypy_type_store = module_type_store
    test_fortranfiles_read.stypy_function_name = 'test_fortranfiles_read'
    test_fortranfiles_read.stypy_param_names_list = []
    test_fortranfiles_read.stypy_varargs_param_name = None
    test_fortranfiles_read.stypy_kwargs_param_name = None
    test_fortranfiles_read.stypy_call_defaults = defaults
    test_fortranfiles_read.stypy_call_varargs = varargs
    test_fortranfiles_read.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fortranfiles_read', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fortranfiles_read', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fortranfiles_read(...)' code ##################

    
    
    # Call to iglob(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to join(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'DATA_PATH' (line 19)
    DATA_PATH_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'DATA_PATH', False)
    str_814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 47), 'str', 'fortran-*-*x*x*.dat')
    # Processing the call keyword arguments (line 19)
    kwargs_815 = {}
    # Getting the type of 'path' (line 19)
    path_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'path', False)
    # Obtaining the member 'join' of a type (line 19)
    join_812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 26), path_811, 'join')
    # Calling join(args, kwargs) (line 19)
    join_call_result_816 = invoke(stypy.reporting.localization.Localization(__file__, 19, 26), join_812, *[DATA_PATH_813, str_814], **kwargs_815)
    
    # Processing the call keyword arguments (line 19)
    kwargs_817 = {}
    # Getting the type of 'iglob' (line 19)
    iglob_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'iglob', False)
    # Calling iglob(args, kwargs) (line 19)
    iglob_call_result_818 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), iglob_810, *[join_call_result_816], **kwargs_817)
    
    # Testing the type of a for loop iterable (line 19)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 4), iglob_call_result_818)
    # Getting the type of the for loop variable (line 19)
    for_loop_var_819 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 4), iglob_call_result_818)
    # Assigning a type to the variable 'filename' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'filename', for_loop_var_819)
    # SSA begins for a for statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to search(...): (line 20)
    # Processing the call arguments (line 20)
    str_822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'str', 'fortran-([^-]+)-(\\d+)x(\\d+)x(\\d+).dat')
    # Getting the type of 'filename' (line 20)
    filename_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 64), 'filename', False)
    # Getting the type of 're' (line 20)
    re_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 74), 're', False)
    # Obtaining the member 'I' of a type (line 20)
    I_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 74), re_824, 'I')
    # Processing the call keyword arguments (line 20)
    kwargs_826 = {}
    # Getting the type of 're' (line 20)
    re_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 're', False)
    # Obtaining the member 'search' of a type (line 20)
    search_821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), re_820, 'search')
    # Calling search(args, kwargs) (line 20)
    search_call_result_827 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), search_821, *[str_822, filename_823, I_825], **kwargs_826)
    
    # Assigning a type to the variable 'm' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'm', search_call_result_827)
    
    
    # Getting the type of 'm' (line 21)
    m_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'm')
    # Applying the 'not' unary operator (line 21)
    result_not__829 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), 'not', m_828)
    
    # Testing the type of an if condition (line 21)
    if_condition_830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 8), result_not__829)
    # Assigning a type to the variable 'if_condition_830' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'if_condition_830', if_condition_830)
    # SSA begins for if statement (line 21)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 22)
    # Processing the call arguments (line 22)
    str_832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'str', "Couldn't match %s filename to regex")
    # Getting the type of 'filename' (line 22)
    filename_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 71), 'filename', False)
    # Applying the binary operator '%' (line 22)
    result_mod_834 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 31), '%', str_832, filename_833)
    
    # Processing the call keyword arguments (line 22)
    kwargs_835 = {}
    # Getting the type of 'RuntimeError' (line 22)
    RuntimeError_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 22)
    RuntimeError_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 22, 18), RuntimeError_831, *[result_mod_834], **kwargs_835)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 22, 12), RuntimeError_call_result_836, 'raise parameter', BaseException)
    # SSA join for if statement (line 21)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 24):
    
    # Assigning a Tuple to a Name (line 24):
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    
    # Call to int(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to group(...): (line 24)
    # Processing the call arguments (line 24)
    int_841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_842 = {}
    # Getting the type of 'm' (line 24)
    m_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'm', False)
    # Obtaining the member 'group' of a type (line 24)
    group_840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), m_839, 'group')
    # Calling group(args, kwargs) (line 24)
    group_call_result_843 = invoke(stypy.reporting.localization.Localization(__file__, 24, 20), group_840, *[int_841], **kwargs_842)
    
    # Processing the call keyword arguments (line 24)
    kwargs_844 = {}
    # Getting the type of 'int' (line 24)
    int_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'int', False)
    # Calling int(args, kwargs) (line 24)
    int_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), int_838, *[group_call_result_843], **kwargs_844)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), tuple_837, int_call_result_845)
    # Adding element type (line 24)
    
    # Call to int(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to group(...): (line 24)
    # Processing the call arguments (line 24)
    int_849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 45), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_850 = {}
    # Getting the type of 'm' (line 24)
    m_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'm', False)
    # Obtaining the member 'group' of a type (line 24)
    group_848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 37), m_847, 'group')
    # Calling group(args, kwargs) (line 24)
    group_call_result_851 = invoke(stypy.reporting.localization.Localization(__file__, 24, 37), group_848, *[int_849], **kwargs_850)
    
    # Processing the call keyword arguments (line 24)
    kwargs_852 = {}
    # Getting the type of 'int' (line 24)
    int_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 33), 'int', False)
    # Calling int(args, kwargs) (line 24)
    int_call_result_853 = invoke(stypy.reporting.localization.Localization(__file__, 24, 33), int_846, *[group_call_result_851], **kwargs_852)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), tuple_837, int_call_result_853)
    # Adding element type (line 24)
    
    # Call to int(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to group(...): (line 24)
    # Processing the call arguments (line 24)
    int_857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 62), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_858 = {}
    # Getting the type of 'm' (line 24)
    m_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 54), 'm', False)
    # Obtaining the member 'group' of a type (line 24)
    group_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 54), m_855, 'group')
    # Calling group(args, kwargs) (line 24)
    group_call_result_859 = invoke(stypy.reporting.localization.Localization(__file__, 24, 54), group_856, *[int_857], **kwargs_858)
    
    # Processing the call keyword arguments (line 24)
    kwargs_860 = {}
    # Getting the type of 'int' (line 24)
    int_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 50), 'int', False)
    # Calling int(args, kwargs) (line 24)
    int_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 24, 50), int_854, *[group_call_result_859], **kwargs_860)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), tuple_837, int_call_result_861)
    
    # Assigning a type to the variable 'dims' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'dims', tuple_837)
    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to replace(...): (line 26)
    # Processing the call arguments (line 26)
    str_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'str', 's')
    str_869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'str', '<')
    # Processing the call keyword arguments (line 26)
    kwargs_870 = {}
    
    # Call to group(...): (line 26)
    # Processing the call arguments (line 26)
    int_864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 24), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_865 = {}
    # Getting the type of 'm' (line 26)
    m_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'm', False)
    # Obtaining the member 'group' of a type (line 26)
    group_863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), m_862, 'group')
    # Calling group(args, kwargs) (line 26)
    group_call_result_866 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), group_863, *[int_864], **kwargs_865)
    
    # Obtaining the member 'replace' of a type (line 26)
    replace_867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), group_call_result_866, 'replace')
    # Calling replace(args, kwargs) (line 26)
    replace_call_result_871 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), replace_867, *[str_868, str_869], **kwargs_870)
    
    # Assigning a type to the variable 'dtype' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'dtype', replace_call_result_871)
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to FortranFile(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'filename' (line 28)
    filename_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'filename', False)
    str_874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'str', 'r')
    str_875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'str', '<u4')
    # Processing the call keyword arguments (line 28)
    kwargs_876 = {}
    # Getting the type of 'FortranFile' (line 28)
    FortranFile_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 28)
    FortranFile_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), FortranFile_872, *[filename_873, str_874, str_875], **kwargs_876)
    
    # Assigning a type to the variable 'f' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'f', FortranFile_call_result_877)
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to reshape(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'dims' (line 29)
    dims_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 50), 'dims', False)
    # Processing the call keyword arguments (line 29)
    str_886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 62), 'str', 'F')
    keyword_887 = str_886
    kwargs_888 = {'order': keyword_887}
    
    # Call to read_record(...): (line 29)
    # Processing the call keyword arguments (line 29)
    # Getting the type of 'dtype' (line 29)
    dtype_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'dtype', False)
    keyword_881 = dtype_880
    kwargs_882 = {'dtype': keyword_881}
    # Getting the type of 'f' (line 29)
    f_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'f', False)
    # Obtaining the member 'read_record' of a type (line 29)
    read_record_879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), f_878, 'read_record')
    # Calling read_record(args, kwargs) (line 29)
    read_record_call_result_883 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), read_record_879, *[], **kwargs_882)
    
    # Obtaining the member 'reshape' of a type (line 29)
    reshape_884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 15), read_record_call_result_883, 'reshape')
    # Calling reshape(args, kwargs) (line 29)
    reshape_call_result_889 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), reshape_884, *[dims_885], **kwargs_888)
    
    # Assigning a type to the variable 'data' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'data', reshape_call_result_889)
    
    # Call to close(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_892 = {}
    # Getting the type of 'f' (line 30)
    f_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'f', False)
    # Obtaining the member 'close' of a type (line 30)
    close_891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), f_890, 'close')
    # Calling close(args, kwargs) (line 30)
    close_call_result_893 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), close_891, *[], **kwargs_892)
    
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to astype(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'dtype' (line 32)
    dtype_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 65), 'dtype', False)
    # Processing the call keyword arguments (line 32)
    kwargs_909 = {}
    
    # Call to reshape(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'dims' (line 32)
    dims_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 52), 'dims', False)
    # Processing the call keyword arguments (line 32)
    kwargs_905 = {}
    
    # Call to arange(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to prod(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'dims' (line 32)
    dims_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 37), 'dims', False)
    # Processing the call keyword arguments (line 32)
    kwargs_899 = {}
    # Getting the type of 'np' (line 32)
    np_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'np', False)
    # Obtaining the member 'prod' of a type (line 32)
    prod_897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 29), np_896, 'prod')
    # Calling prod(args, kwargs) (line 32)
    prod_call_result_900 = invoke(stypy.reporting.localization.Localization(__file__, 32, 29), prod_897, *[dims_898], **kwargs_899)
    
    # Processing the call keyword arguments (line 32)
    kwargs_901 = {}
    # Getting the type of 'np' (line 32)
    np_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'np', False)
    # Obtaining the member 'arange' of a type (line 32)
    arange_895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 19), np_894, 'arange')
    # Calling arange(args, kwargs) (line 32)
    arange_call_result_902 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), arange_895, *[prod_call_result_900], **kwargs_901)
    
    # Obtaining the member 'reshape' of a type (line 32)
    reshape_903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 19), arange_call_result_902, 'reshape')
    # Calling reshape(args, kwargs) (line 32)
    reshape_call_result_906 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), reshape_903, *[dims_904], **kwargs_905)
    
    # Obtaining the member 'astype' of a type (line 32)
    astype_907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 19), reshape_call_result_906, 'astype')
    # Calling astype(args, kwargs) (line 32)
    astype_call_result_910 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), astype_907, *[dtype_908], **kwargs_909)
    
    # Assigning a type to the variable 'expected' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'expected', astype_call_result_910)
    
    # Call to assert_equal(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'data' (line 33)
    data_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'data', False)
    # Getting the type of 'expected' (line 33)
    expected_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'expected', False)
    # Processing the call keyword arguments (line 33)
    kwargs_914 = {}
    # Getting the type of 'assert_equal' (line 33)
    assert_equal_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 33)
    assert_equal_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_equal_911, *[data_912, expected_913], **kwargs_914)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_fortranfiles_read(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fortranfiles_read' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_916)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fortranfiles_read'
    return stypy_return_type_916

# Assigning a type to the variable 'test_fortranfiles_read' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_fortranfiles_read', test_fortranfiles_read)

@norecursion
def test_fortranfiles_mixed_record(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fortranfiles_mixed_record'
    module_type_store = module_type_store.open_function_context('test_fortranfiles_mixed_record', 36, 0, False)
    
    # Passed parameters checking function
    test_fortranfiles_mixed_record.stypy_localization = localization
    test_fortranfiles_mixed_record.stypy_type_of_self = None
    test_fortranfiles_mixed_record.stypy_type_store = module_type_store
    test_fortranfiles_mixed_record.stypy_function_name = 'test_fortranfiles_mixed_record'
    test_fortranfiles_mixed_record.stypy_param_names_list = []
    test_fortranfiles_mixed_record.stypy_varargs_param_name = None
    test_fortranfiles_mixed_record.stypy_kwargs_param_name = None
    test_fortranfiles_mixed_record.stypy_call_defaults = defaults
    test_fortranfiles_mixed_record.stypy_call_varargs = varargs
    test_fortranfiles_mixed_record.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fortranfiles_mixed_record', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fortranfiles_mixed_record', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fortranfiles_mixed_record(...)' code ##################

    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to join(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'DATA_PATH' (line 37)
    DATA_PATH_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'DATA_PATH', False)
    str_920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'str', 'fortran-mixed.dat')
    # Processing the call keyword arguments (line 37)
    kwargs_921 = {}
    # Getting the type of 'path' (line 37)
    path_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'path', False)
    # Obtaining the member 'join' of a type (line 37)
    join_918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), path_917, 'join')
    # Calling join(args, kwargs) (line 37)
    join_call_result_922 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), join_918, *[DATA_PATH_919, str_920], **kwargs_921)
    
    # Assigning a type to the variable 'filename' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'filename', join_call_result_922)
    
    # Call to FortranFile(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'filename' (line 38)
    filename_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'filename', False)
    str_925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'str', 'r')
    str_926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'str', '<u4')
    # Processing the call keyword arguments (line 38)
    kwargs_927 = {}
    # Getting the type of 'FortranFile' (line 38)
    FortranFile_923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 38)
    FortranFile_call_result_928 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), FortranFile_923, *[filename_924, str_925, str_926], **kwargs_927)
    
    with_929 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 38, 9), FortranFile_call_result_928, 'with parameter', '__enter__', '__exit__')

    if with_929:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 38)
        enter___930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), FortranFile_call_result_928, '__enter__')
        with_enter_931 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), enter___930)
        # Assigning a type to the variable 'f' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'f', with_enter_931)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to read_record(...): (line 39)
        # Processing the call arguments (line 39)
        str_934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'str', '<i4,<f4,<i8,(2)<f8')
        # Processing the call keyword arguments (line 39)
        kwargs_935 = {}
        # Getting the type of 'f' (line 39)
        f_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'f', False)
        # Obtaining the member 'read_record' of a type (line 39)
        read_record_933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), f_932, 'read_record')
        # Calling read_record(args, kwargs) (line 39)
        read_record_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), read_record_933, *[str_934], **kwargs_935)
        
        # Assigning a type to the variable 'record' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'record', read_record_call_result_936)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 38)
        exit___937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 9), FortranFile_call_result_928, '__exit__')
        with_exit_938 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), exit___937, None, None, None)

    
    # Call to assert_equal(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining the type of the subscript
    int_940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
    
    # Obtaining the type of the subscript
    str_941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'str', 'f0')
    # Getting the type of 'record' (line 41)
    record_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), record_942, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_944 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), getitem___943, str_941)
    
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 17), subscript_call_result_944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_946 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), getitem___945, int_940)
    
    int_947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_948 = {}
    # Getting the type of 'assert_equal' (line 41)
    assert_equal_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 41)
    assert_equal_call_result_949 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), assert_equal_939, *[subscript_call_result_946, int_947], **kwargs_948)
    
    
    # Call to assert_allclose(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining the type of the subscript
    int_951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'int')
    
    # Obtaining the type of the subscript
    str_952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 27), 'str', 'f1')
    # Getting the type of 'record' (line 42)
    record_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), record_953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_955 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), getitem___954, str_952)
    
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), subscript_call_result_955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_957 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), getitem___956, int_951)
    
    float_958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'float')
    # Processing the call keyword arguments (line 42)
    kwargs_959 = {}
    # Getting the type of 'assert_allclose' (line 42)
    assert_allclose_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 42)
    assert_allclose_call_result_960 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), assert_allclose_950, *[subscript_call_result_957, float_958], **kwargs_959)
    
    
    # Call to assert_equal(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining the type of the subscript
    int_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'int')
    
    # Obtaining the type of the subscript
    str_963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'str', 'f2')
    # Getting the type of 'record' (line 43)
    record_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 17), record_964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_966 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), getitem___965, str_963)
    
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 17), subscript_call_result_966, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_968 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), getitem___967, int_962)
    
    int_969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_970 = {}
    # Getting the type of 'assert_equal' (line 43)
    assert_equal_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 43)
    assert_equal_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), assert_equal_961, *[subscript_call_result_968, int_969], **kwargs_970)
    
    
    # Call to assert_allclose(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining the type of the subscript
    int_973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 33), 'int')
    
    # Obtaining the type of the subscript
    str_974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'str', 'f3')
    # Getting the type of 'record' (line 44)
    record_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), record_975, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), getitem___976, str_974)
    
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), subscript_call_result_977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_979 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), getitem___978, int_973)
    
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    float_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 37), list_980, float_981)
    # Adding element type (line 44)
    float_982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 37), list_980, float_982)
    
    # Processing the call keyword arguments (line 44)
    kwargs_983 = {}
    # Getting the type of 'assert_allclose' (line 44)
    assert_allclose_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 44)
    assert_allclose_call_result_984 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), assert_allclose_972, *[subscript_call_result_979, list_980], **kwargs_983)
    
    
    # ################# End of 'test_fortranfiles_mixed_record(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fortranfiles_mixed_record' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_985)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fortranfiles_mixed_record'
    return stypy_return_type_985

# Assigning a type to the variable 'test_fortranfiles_mixed_record' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'test_fortranfiles_mixed_record', test_fortranfiles_mixed_record)

@norecursion
def test_fortranfiles_write(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fortranfiles_write'
    module_type_store = module_type_store.open_function_context('test_fortranfiles_write', 47, 0, False)
    
    # Passed parameters checking function
    test_fortranfiles_write.stypy_localization = localization
    test_fortranfiles_write.stypy_type_of_self = None
    test_fortranfiles_write.stypy_type_store = module_type_store
    test_fortranfiles_write.stypy_function_name = 'test_fortranfiles_write'
    test_fortranfiles_write.stypy_param_names_list = []
    test_fortranfiles_write.stypy_varargs_param_name = None
    test_fortranfiles_write.stypy_kwargs_param_name = None
    test_fortranfiles_write.stypy_call_defaults = defaults
    test_fortranfiles_write.stypy_call_varargs = varargs
    test_fortranfiles_write.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fortranfiles_write', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fortranfiles_write', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fortranfiles_write(...)' code ##################

    
    
    # Call to iglob(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Call to join(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'DATA_PATH' (line 48)
    DATA_PATH_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'DATA_PATH', False)
    str_990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 47), 'str', 'fortran-*-*x*x*.dat')
    # Processing the call keyword arguments (line 48)
    kwargs_991 = {}
    # Getting the type of 'path' (line 48)
    path_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'path', False)
    # Obtaining the member 'join' of a type (line 48)
    join_988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 26), path_987, 'join')
    # Calling join(args, kwargs) (line 48)
    join_call_result_992 = invoke(stypy.reporting.localization.Localization(__file__, 48, 26), join_988, *[DATA_PATH_989, str_990], **kwargs_991)
    
    # Processing the call keyword arguments (line 48)
    kwargs_993 = {}
    # Getting the type of 'iglob' (line 48)
    iglob_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'iglob', False)
    # Calling iglob(args, kwargs) (line 48)
    iglob_call_result_994 = invoke(stypy.reporting.localization.Localization(__file__, 48, 20), iglob_986, *[join_call_result_992], **kwargs_993)
    
    # Testing the type of a for loop iterable (line 48)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 4), iglob_call_result_994)
    # Getting the type of the for loop variable (line 48)
    for_loop_var_995 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 4), iglob_call_result_994)
    # Assigning a type to the variable 'filename' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'filename', for_loop_var_995)
    # SSA begins for a for statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to search(...): (line 49)
    # Processing the call arguments (line 49)
    str_998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'str', 'fortran-([^-]+)-(\\d+)x(\\d+)x(\\d+).dat')
    # Getting the type of 'filename' (line 49)
    filename_999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 64), 'filename', False)
    # Getting the type of 're' (line 49)
    re_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 74), 're', False)
    # Obtaining the member 'I' of a type (line 49)
    I_1001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 74), re_1000, 'I')
    # Processing the call keyword arguments (line 49)
    kwargs_1002 = {}
    # Getting the type of 're' (line 49)
    re_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 're', False)
    # Obtaining the member 'search' of a type (line 49)
    search_997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), re_996, 'search')
    # Calling search(args, kwargs) (line 49)
    search_call_result_1003 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), search_997, *[str_998, filename_999, I_1001], **kwargs_1002)
    
    # Assigning a type to the variable 'm' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'm', search_call_result_1003)
    
    
    # Getting the type of 'm' (line 50)
    m_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'm')
    # Applying the 'not' unary operator (line 50)
    result_not__1005 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'not', m_1004)
    
    # Testing the type of an if condition (line 50)
    if_condition_1006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_not__1005)
    # Assigning a type to the variable 'if_condition_1006' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_1006', if_condition_1006)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 51)
    # Processing the call arguments (line 51)
    str_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'str', "Couldn't match %s filename to regex")
    # Getting the type of 'filename' (line 51)
    filename_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 71), 'filename', False)
    # Applying the binary operator '%' (line 51)
    result_mod_1010 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 31), '%', str_1008, filename_1009)
    
    # Processing the call keyword arguments (line 51)
    kwargs_1011 = {}
    # Getting the type of 'RuntimeError' (line 51)
    RuntimeError_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 51)
    RuntimeError_call_result_1012 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), RuntimeError_1007, *[result_mod_1010], **kwargs_1011)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 51, 12), RuntimeError_call_result_1012, 'raise parameter', BaseException)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 52):
    
    # Assigning a Tuple to a Name (line 52):
    
    # Obtaining an instance of the builtin type 'tuple' (line 52)
    tuple_1013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 52)
    # Adding element type (line 52)
    
    # Call to int(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Call to group(...): (line 52)
    # Processing the call arguments (line 52)
    int_1017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_1018 = {}
    # Getting the type of 'm' (line 52)
    m_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'm', False)
    # Obtaining the member 'group' of a type (line 52)
    group_1016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 20), m_1015, 'group')
    # Calling group(args, kwargs) (line 52)
    group_call_result_1019 = invoke(stypy.reporting.localization.Localization(__file__, 52, 20), group_1016, *[int_1017], **kwargs_1018)
    
    # Processing the call keyword arguments (line 52)
    kwargs_1020 = {}
    # Getting the type of 'int' (line 52)
    int_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'int', False)
    # Calling int(args, kwargs) (line 52)
    int_call_result_1021 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), int_1014, *[group_call_result_1019], **kwargs_1020)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 16), tuple_1013, int_call_result_1021)
    # Adding element type (line 52)
    
    # Call to int(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Call to group(...): (line 52)
    # Processing the call arguments (line 52)
    int_1025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 45), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_1026 = {}
    # Getting the type of 'm' (line 52)
    m_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 37), 'm', False)
    # Obtaining the member 'group' of a type (line 52)
    group_1024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 37), m_1023, 'group')
    # Calling group(args, kwargs) (line 52)
    group_call_result_1027 = invoke(stypy.reporting.localization.Localization(__file__, 52, 37), group_1024, *[int_1025], **kwargs_1026)
    
    # Processing the call keyword arguments (line 52)
    kwargs_1028 = {}
    # Getting the type of 'int' (line 52)
    int_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'int', False)
    # Calling int(args, kwargs) (line 52)
    int_call_result_1029 = invoke(stypy.reporting.localization.Localization(__file__, 52, 33), int_1022, *[group_call_result_1027], **kwargs_1028)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 16), tuple_1013, int_call_result_1029)
    # Adding element type (line 52)
    
    # Call to int(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Call to group(...): (line 52)
    # Processing the call arguments (line 52)
    int_1033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 62), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_1034 = {}
    # Getting the type of 'm' (line 52)
    m_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 54), 'm', False)
    # Obtaining the member 'group' of a type (line 52)
    group_1032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 54), m_1031, 'group')
    # Calling group(args, kwargs) (line 52)
    group_call_result_1035 = invoke(stypy.reporting.localization.Localization(__file__, 52, 54), group_1032, *[int_1033], **kwargs_1034)
    
    # Processing the call keyword arguments (line 52)
    kwargs_1036 = {}
    # Getting the type of 'int' (line 52)
    int_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 50), 'int', False)
    # Calling int(args, kwargs) (line 52)
    int_call_result_1037 = invoke(stypy.reporting.localization.Localization(__file__, 52, 50), int_1030, *[group_call_result_1035], **kwargs_1036)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 16), tuple_1013, int_call_result_1037)
    
    # Assigning a type to the variable 'dims' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'dims', tuple_1013)
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to replace(...): (line 54)
    # Processing the call arguments (line 54)
    str_1044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'str', 's')
    str_1045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 40), 'str', '<')
    # Processing the call keyword arguments (line 54)
    kwargs_1046 = {}
    
    # Call to group(...): (line 54)
    # Processing the call arguments (line 54)
    int_1040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 24), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_1041 = {}
    # Getting the type of 'm' (line 54)
    m_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'm', False)
    # Obtaining the member 'group' of a type (line 54)
    group_1039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), m_1038, 'group')
    # Calling group(args, kwargs) (line 54)
    group_call_result_1042 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), group_1039, *[int_1040], **kwargs_1041)
    
    # Obtaining the member 'replace' of a type (line 54)
    replace_1043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), group_call_result_1042, 'replace')
    # Calling replace(args, kwargs) (line 54)
    replace_call_result_1047 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), replace_1043, *[str_1044, str_1045], **kwargs_1046)
    
    # Assigning a type to the variable 'dtype' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'dtype', replace_call_result_1047)
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to astype(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'dtype' (line 55)
    dtype_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 61), 'dtype', False)
    # Processing the call keyword arguments (line 55)
    kwargs_1063 = {}
    
    # Call to reshape(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'dims' (line 55)
    dims_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'dims', False)
    # Processing the call keyword arguments (line 55)
    kwargs_1059 = {}
    
    # Call to arange(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to prod(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'dims' (line 55)
    dims_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'dims', False)
    # Processing the call keyword arguments (line 55)
    kwargs_1053 = {}
    # Getting the type of 'np' (line 55)
    np_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'np', False)
    # Obtaining the member 'prod' of a type (line 55)
    prod_1051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 25), np_1050, 'prod')
    # Calling prod(args, kwargs) (line 55)
    prod_call_result_1054 = invoke(stypy.reporting.localization.Localization(__file__, 55, 25), prod_1051, *[dims_1052], **kwargs_1053)
    
    # Processing the call keyword arguments (line 55)
    kwargs_1055 = {}
    # Getting the type of 'np' (line 55)
    np_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'np', False)
    # Obtaining the member 'arange' of a type (line 55)
    arange_1049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), np_1048, 'arange')
    # Calling arange(args, kwargs) (line 55)
    arange_call_result_1056 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), arange_1049, *[prod_call_result_1054], **kwargs_1055)
    
    # Obtaining the member 'reshape' of a type (line 55)
    reshape_1057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), arange_call_result_1056, 'reshape')
    # Calling reshape(args, kwargs) (line 55)
    reshape_call_result_1060 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), reshape_1057, *[dims_1058], **kwargs_1059)
    
    # Obtaining the member 'astype' of a type (line 55)
    astype_1061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), reshape_call_result_1060, 'astype')
    # Calling astype(args, kwargs) (line 55)
    astype_call_result_1064 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), astype_1061, *[dtype_1062], **kwargs_1063)
    
    # Assigning a type to the variable 'data' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'data', astype_call_result_1064)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to mkdtemp(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_1067 = {}
    # Getting the type of 'tempfile' (line 57)
    tempfile_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'tempfile', False)
    # Obtaining the member 'mkdtemp' of a type (line 57)
    mkdtemp_1066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 17), tempfile_1065, 'mkdtemp')
    # Calling mkdtemp(args, kwargs) (line 57)
    mkdtemp_call_result_1068 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), mkdtemp_1066, *[], **kwargs_1067)
    
    # Assigning a type to the variable 'tmpdir' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'tmpdir', mkdtemp_call_result_1068)
    
    # Try-finally block (line 58)
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to join(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'tmpdir' (line 59)
    tmpdir_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'tmpdir', False)
    
    # Call to basename(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'filename' (line 59)
    filename_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 54), 'filename', False)
    # Processing the call keyword arguments (line 59)
    kwargs_1075 = {}
    # Getting the type of 'path' (line 59)
    path_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'path', False)
    # Obtaining the member 'basename' of a type (line 59)
    basename_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 40), path_1072, 'basename')
    # Calling basename(args, kwargs) (line 59)
    basename_call_result_1076 = invoke(stypy.reporting.localization.Localization(__file__, 59, 40), basename_1073, *[filename_1074], **kwargs_1075)
    
    # Processing the call keyword arguments (line 59)
    kwargs_1077 = {}
    # Getting the type of 'path' (line 59)
    path_1069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'path', False)
    # Obtaining the member 'join' of a type (line 59)
    join_1070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), path_1069, 'join')
    # Calling join(args, kwargs) (line 59)
    join_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 59, 23), join_1070, *[tmpdir_1071, basename_call_result_1076], **kwargs_1077)
    
    # Assigning a type to the variable 'testFile' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'testFile', join_call_result_1078)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to FortranFile(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'testFile' (line 60)
    testFile_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 28), 'testFile', False)
    str_1081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 38), 'str', 'w')
    str_1082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 42), 'str', '<u4')
    # Processing the call keyword arguments (line 60)
    kwargs_1083 = {}
    # Getting the type of 'FortranFile' (line 60)
    FortranFile_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 60)
    FortranFile_call_result_1084 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), FortranFile_1079, *[testFile_1080, str_1081, str_1082], **kwargs_1083)
    
    # Assigning a type to the variable 'f' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'f', FortranFile_call_result_1084)
    
    # Call to write_record(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'data' (line 61)
    data_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'data', False)
    # Obtaining the member 'T' of a type (line 61)
    T_1088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 27), data_1087, 'T')
    # Processing the call keyword arguments (line 61)
    kwargs_1089 = {}
    # Getting the type of 'f' (line 61)
    f_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'f', False)
    # Obtaining the member 'write_record' of a type (line 61)
    write_record_1086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), f_1085, 'write_record')
    # Calling write_record(args, kwargs) (line 61)
    write_record_call_result_1090 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), write_record_1086, *[T_1088], **kwargs_1089)
    
    
    # Call to close(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_1093 = {}
    # Getting the type of 'f' (line 62)
    f_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'f', False)
    # Obtaining the member 'close' of a type (line 62)
    close_1092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), f_1091, 'close')
    # Calling close(args, kwargs) (line 62)
    close_call_result_1094 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), close_1092, *[], **kwargs_1093)
    
    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to open(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'filename' (line 63)
    filename_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'filename', False)
    str_1097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'str', 'rb')
    # Processing the call keyword arguments (line 63)
    kwargs_1098 = {}
    # Getting the type of 'open' (line 63)
    open_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'open', False)
    # Calling open(args, kwargs) (line 63)
    open_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 63, 27), open_1095, *[filename_1096, str_1097], **kwargs_1098)
    
    # Assigning a type to the variable 'originalfile' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'originalfile', open_call_result_1099)
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to open(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'testFile' (line 64)
    testFile_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'testFile', False)
    str_1102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'str', 'rb')
    # Processing the call keyword arguments (line 64)
    kwargs_1103 = {}
    # Getting the type of 'open' (line 64)
    open_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'open', False)
    # Calling open(args, kwargs) (line 64)
    open_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 64, 22), open_1100, *[testFile_1101, str_1102], **kwargs_1103)
    
    # Assigning a type to the variable 'newfile' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'newfile', open_call_result_1104)
    
    # Call to assert_equal(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Call to read(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_1108 = {}
    # Getting the type of 'originalfile' (line 65)
    originalfile_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'originalfile', False)
    # Obtaining the member 'read' of a type (line 65)
    read_1107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 25), originalfile_1106, 'read')
    # Calling read(args, kwargs) (line 65)
    read_call_result_1109 = invoke(stypy.reporting.localization.Localization(__file__, 65, 25), read_1107, *[], **kwargs_1108)
    
    
    # Call to read(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_1112 = {}
    # Getting the type of 'newfile' (line 65)
    newfile_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 46), 'newfile', False)
    # Obtaining the member 'read' of a type (line 65)
    read_1111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 46), newfile_1110, 'read')
    # Calling read(args, kwargs) (line 65)
    read_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 65, 46), read_1111, *[], **kwargs_1112)
    
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'filename' (line 66)
    filename_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'filename', False)
    keyword_1115 = filename_1114
    kwargs_1116 = {'err_msg': keyword_1115}
    # Getting the type of 'assert_equal' (line 65)
    assert_equal_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 65)
    assert_equal_call_result_1117 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), assert_equal_1105, *[read_call_result_1109, read_call_result_1113], **kwargs_1116)
    
    
    # Call to close(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_1120 = {}
    # Getting the type of 'originalfile' (line 67)
    originalfile_1118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'originalfile', False)
    # Obtaining the member 'close' of a type (line 67)
    close_1119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), originalfile_1118, 'close')
    # Calling close(args, kwargs) (line 67)
    close_call_result_1121 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), close_1119, *[], **kwargs_1120)
    
    
    # Call to close(...): (line 68)
    # Processing the call keyword arguments (line 68)
    kwargs_1124 = {}
    # Getting the type of 'newfile' (line 68)
    newfile_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'newfile', False)
    # Obtaining the member 'close' of a type (line 68)
    close_1123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), newfile_1122, 'close')
    # Calling close(args, kwargs) (line 68)
    close_call_result_1125 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), close_1123, *[], **kwargs_1124)
    
    
    # finally branch of the try-finally block (line 58)
    
    # Call to rmtree(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'tmpdir' (line 70)
    tmpdir_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'tmpdir', False)
    # Processing the call keyword arguments (line 70)
    kwargs_1129 = {}
    # Getting the type of 'shutil' (line 70)
    shutil_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'shutil', False)
    # Obtaining the member 'rmtree' of a type (line 70)
    rmtree_1127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), shutil_1126, 'rmtree')
    # Calling rmtree(args, kwargs) (line 70)
    rmtree_call_result_1130 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), rmtree_1127, *[tmpdir_1128], **kwargs_1129)
    
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_fortranfiles_write(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fortranfiles_write' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fortranfiles_write'
    return stypy_return_type_1131

# Assigning a type to the variable 'test_fortranfiles_write' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'test_fortranfiles_write', test_fortranfiles_write)

@norecursion
def test_fortranfile_read_mixed_record(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fortranfile_read_mixed_record'
    module_type_store = module_type_store.open_function_context('test_fortranfile_read_mixed_record', 73, 0, False)
    
    # Passed parameters checking function
    test_fortranfile_read_mixed_record.stypy_localization = localization
    test_fortranfile_read_mixed_record.stypy_type_of_self = None
    test_fortranfile_read_mixed_record.stypy_type_store = module_type_store
    test_fortranfile_read_mixed_record.stypy_function_name = 'test_fortranfile_read_mixed_record'
    test_fortranfile_read_mixed_record.stypy_param_names_list = []
    test_fortranfile_read_mixed_record.stypy_varargs_param_name = None
    test_fortranfile_read_mixed_record.stypy_kwargs_param_name = None
    test_fortranfile_read_mixed_record.stypy_call_defaults = defaults
    test_fortranfile_read_mixed_record.stypy_call_varargs = varargs
    test_fortranfile_read_mixed_record.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fortranfile_read_mixed_record', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fortranfile_read_mixed_record', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fortranfile_read_mixed_record(...)' code ##################

    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to join(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'DATA_PATH' (line 85)
    DATA_PATH_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'DATA_PATH', False)
    str_1135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 36), 'str', 'fortran-3x3d-2i.dat')
    # Processing the call keyword arguments (line 85)
    kwargs_1136 = {}
    # Getting the type of 'path' (line 85)
    path_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'path', False)
    # Obtaining the member 'join' of a type (line 85)
    join_1133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), path_1132, 'join')
    # Calling join(args, kwargs) (line 85)
    join_call_result_1137 = invoke(stypy.reporting.localization.Localization(__file__, 85, 15), join_1133, *[DATA_PATH_1134, str_1135], **kwargs_1136)
    
    # Assigning a type to the variable 'filename' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'filename', join_call_result_1137)
    
    # Call to FortranFile(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'filename' (line 86)
    filename_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'filename', False)
    str_1140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'str', 'r')
    str_1141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 36), 'str', '<u4')
    # Processing the call keyword arguments (line 86)
    kwargs_1142 = {}
    # Getting the type of 'FortranFile' (line 86)
    FortranFile_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 86)
    FortranFile_call_result_1143 = invoke(stypy.reporting.localization.Localization(__file__, 86, 9), FortranFile_1138, *[filename_1139, str_1140, str_1141], **kwargs_1142)
    
    with_1144 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 86, 9), FortranFile_call_result_1143, 'with parameter', '__enter__', '__exit__')

    if with_1144:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 86)
        enter___1145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 9), FortranFile_call_result_1143, '__enter__')
        with_enter_1146 = invoke(stypy.reporting.localization.Localization(__file__, 86, 9), enter___1145)
        # Assigning a type to the variable 'f' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'f', with_enter_1146)
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to read_record(...): (line 87)
        # Processing the call arguments (line 87)
        str_1149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 31), 'str', '(3,3)f8')
        str_1150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 42), 'str', '2i4')
        # Processing the call keyword arguments (line 87)
        kwargs_1151 = {}
        # Getting the type of 'f' (line 87)
        f_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'f', False)
        # Obtaining the member 'read_record' of a type (line 87)
        read_record_1148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 17), f_1147, 'read_record')
        # Calling read_record(args, kwargs) (line 87)
        read_record_call_result_1152 = invoke(stypy.reporting.localization.Localization(__file__, 87, 17), read_record_1148, *[str_1149, str_1150], **kwargs_1151)
        
        # Assigning a type to the variable 'record' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'record', read_record_call_result_1152)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 86)
        exit___1153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 9), FortranFile_call_result_1143, '__exit__')
        with_exit_1154 = invoke(stypy.reporting.localization.Localization(__file__, 86, 9), exit___1153, None, None, None)

    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to astype(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'np' (line 89)
    np_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 45), 'np', False)
    # Obtaining the member 'double' of a type (line 89)
    double_1169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 45), np_1168, 'double')
    # Processing the call keyword arguments (line 89)
    kwargs_1170 = {}
    
    # Call to reshape(...): (line 89)
    # Processing the call arguments (line 89)
    int_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'int')
    int_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 35), 'int')
    # Processing the call keyword arguments (line 89)
    kwargs_1165 = {}
    
    # Call to arange(...): (line 89)
    # Processing the call arguments (line 89)
    int_1157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'int')
    int_1158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'int')
    # Applying the binary operator '*' (line 89)
    result_mul_1159 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 19), '*', int_1157, int_1158)
    
    # Processing the call keyword arguments (line 89)
    kwargs_1160 = {}
    # Getting the type of 'np' (line 89)
    np_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'np', False)
    # Obtaining the member 'arange' of a type (line 89)
    arange_1156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 9), np_1155, 'arange')
    # Calling arange(args, kwargs) (line 89)
    arange_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 89, 9), arange_1156, *[result_mul_1159], **kwargs_1160)
    
    # Obtaining the member 'reshape' of a type (line 89)
    reshape_1162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 9), arange_call_result_1161, 'reshape')
    # Calling reshape(args, kwargs) (line 89)
    reshape_call_result_1166 = invoke(stypy.reporting.localization.Localization(__file__, 89, 9), reshape_1162, *[int_1163, int_1164], **kwargs_1165)
    
    # Obtaining the member 'astype' of a type (line 89)
    astype_1167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 9), reshape_call_result_1166, 'astype')
    # Calling astype(args, kwargs) (line 89)
    astype_call_result_1171 = invoke(stypy.reporting.localization.Localization(__file__, 89, 9), astype_1167, *[double_1169], **kwargs_1170)
    
    # Assigning a type to the variable 'ax' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'ax', astype_call_result_1171)
    
    # Assigning a Call to a Name (line 90):
    
    # Assigning a Call to a Name (line 90):
    
    # Call to array(...): (line 90)
    # Processing the call arguments (line 90)
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_1174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    int_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), list_1174, int_1175)
    # Adding element type (line 90)
    int_1176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), list_1174, int_1176)
    
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'np' (line 90)
    np_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'np', False)
    # Obtaining the member 'int32' of a type (line 90)
    int32_1178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 34), np_1177, 'int32')
    keyword_1179 = int32_1178
    kwargs_1180 = {'dtype': keyword_1179}
    # Getting the type of 'np' (line 90)
    np_1172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 90)
    array_1173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 9), np_1172, 'array')
    # Calling array(args, kwargs) (line 90)
    array_call_result_1181 = invoke(stypy.reporting.localization.Localization(__file__, 90, 9), array_1173, *[list_1174], **kwargs_1180)
    
    # Assigning a type to the variable 'bx' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'bx', array_call_result_1181)
    
    # Call to assert_equal(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Obtaining the type of the subscript
    int_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'int')
    # Getting the type of 'record' (line 92)
    record_1184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___1185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), record_1184, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_1186 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), getitem___1185, int_1183)
    
    # Getting the type of 'ax' (line 92)
    ax_1187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'ax', False)
    # Obtaining the member 'T' of a type (line 92)
    T_1188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), ax_1187, 'T')
    # Processing the call keyword arguments (line 92)
    kwargs_1189 = {}
    # Getting the type of 'assert_equal' (line 92)
    assert_equal_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 92)
    assert_equal_call_result_1190 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), assert_equal_1182, *[subscript_call_result_1186, T_1188], **kwargs_1189)
    
    
    # Call to assert_equal(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Obtaining the type of the subscript
    int_1192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 24), 'int')
    # Getting the type of 'record' (line 93)
    record_1193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'record', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___1194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), record_1193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_1195 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), getitem___1194, int_1192)
    
    # Getting the type of 'bx' (line 93)
    bx_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'bx', False)
    # Obtaining the member 'T' of a type (line 93)
    T_1197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), bx_1196, 'T')
    # Processing the call keyword arguments (line 93)
    kwargs_1198 = {}
    # Getting the type of 'assert_equal' (line 93)
    assert_equal_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 93)
    assert_equal_call_result_1199 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), assert_equal_1191, *[subscript_call_result_1195, T_1197], **kwargs_1198)
    
    
    # ################# End of 'test_fortranfile_read_mixed_record(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fortranfile_read_mixed_record' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1200)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fortranfile_read_mixed_record'
    return stypy_return_type_1200

# Assigning a type to the variable 'test_fortranfile_read_mixed_record' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'test_fortranfile_read_mixed_record', test_fortranfile_read_mixed_record)

@norecursion
def test_fortranfile_write_mixed_record(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fortranfile_write_mixed_record'
    module_type_store = module_type_store.open_function_context('test_fortranfile_write_mixed_record', 96, 0, False)
    
    # Passed parameters checking function
    test_fortranfile_write_mixed_record.stypy_localization = localization
    test_fortranfile_write_mixed_record.stypy_type_of_self = None
    test_fortranfile_write_mixed_record.stypy_type_store = module_type_store
    test_fortranfile_write_mixed_record.stypy_function_name = 'test_fortranfile_write_mixed_record'
    test_fortranfile_write_mixed_record.stypy_param_names_list = ['tmpdir']
    test_fortranfile_write_mixed_record.stypy_varargs_param_name = None
    test_fortranfile_write_mixed_record.stypy_kwargs_param_name = None
    test_fortranfile_write_mixed_record.stypy_call_defaults = defaults
    test_fortranfile_write_mixed_record.stypy_call_varargs = varargs
    test_fortranfile_write_mixed_record.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fortranfile_write_mixed_record', ['tmpdir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fortranfile_write_mixed_record', localization, ['tmpdir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fortranfile_write_mixed_record(...)' code ##################

    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to join(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Call to str(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'tmpdir' (line 97)
    tmpdir_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'tmpdir', False)
    # Processing the call keyword arguments (line 97)
    kwargs_1205 = {}
    # Getting the type of 'str' (line 97)
    str_1203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'str', False)
    # Calling str(args, kwargs) (line 97)
    str_call_result_1206 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), str_1203, *[tmpdir_1204], **kwargs_1205)
    
    str_1207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'str', 'test.dat')
    # Processing the call keyword arguments (line 97)
    kwargs_1208 = {}
    # Getting the type of 'path' (line 97)
    path_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 9), 'path', False)
    # Obtaining the member 'join' of a type (line 97)
    join_1202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 9), path_1201, 'join')
    # Calling join(args, kwargs) (line 97)
    join_call_result_1209 = invoke(stypy.reporting.localization.Localization(__file__, 97, 9), join_1202, *[str_call_result_1206, str_1207], **kwargs_1208)
    
    # Assigning a type to the variable 'tf' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'tf', join_call_result_1209)
    
    # Assigning a List to a Name (line 99):
    
    # Assigning a List to a Name (line 99):
    
    # Obtaining an instance of the builtin type 'list' (line 99)
    list_1210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 99)
    # Adding element type (line 99)
    
    # Obtaining an instance of the builtin type 'tuple' (line 100)
    tuple_1211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 100)
    # Adding element type (line 100)
    
    # Obtaining an instance of the builtin type 'tuple' (line 100)
    tuple_1212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 100)
    # Adding element type (line 100)
    str_1213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 10), 'str', 'f4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 10), tuple_1212, str_1213)
    # Adding element type (line 100)
    str_1214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'str', 'f4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 10), tuple_1212, str_1214)
    # Adding element type (line 100)
    str_1215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'str', 'i4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 10), tuple_1212, str_1215)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_1211, tuple_1212)
    # Adding element type (line 100)
    
    # Obtaining an instance of the builtin type 'tuple' (line 100)
    tuple_1216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 100)
    # Adding element type (line 100)
    
    # Call to float32(...): (line 100)
    # Processing the call arguments (line 100)
    int_1219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 41), 'int')
    # Processing the call keyword arguments (line 100)
    kwargs_1220 = {}
    # Getting the type of 'np' (line 100)
    np_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'np', False)
    # Obtaining the member 'float32' of a type (line 100)
    float32_1218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 30), np_1217, 'float32')
    # Calling float32(args, kwargs) (line 100)
    float32_call_result_1221 = invoke(stypy.reporting.localization.Localization(__file__, 100, 30), float32_1218, *[int_1219], **kwargs_1220)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 30), tuple_1216, float32_call_result_1221)
    # Adding element type (line 100)
    
    # Call to float32(...): (line 100)
    # Processing the call arguments (line 100)
    int_1224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 56), 'int')
    # Processing the call keyword arguments (line 100)
    kwargs_1225 = {}
    # Getting the type of 'np' (line 100)
    np_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 45), 'np', False)
    # Obtaining the member 'float32' of a type (line 100)
    float32_1223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 45), np_1222, 'float32')
    # Calling float32(args, kwargs) (line 100)
    float32_call_result_1226 = invoke(stypy.reporting.localization.Localization(__file__, 100, 45), float32_1223, *[int_1224], **kwargs_1225)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 30), tuple_1216, float32_call_result_1226)
    # Adding element type (line 100)
    
    # Call to int32(...): (line 100)
    # Processing the call arguments (line 100)
    int_1229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 69), 'int')
    # Processing the call keyword arguments (line 100)
    kwargs_1230 = {}
    # Getting the type of 'np' (line 100)
    np_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 60), 'np', False)
    # Obtaining the member 'int32' of a type (line 100)
    int32_1228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 60), np_1227, 'int32')
    # Calling int32(args, kwargs) (line 100)
    int32_call_result_1231 = invoke(stypy.reporting.localization.Localization(__file__, 100, 60), int32_1228, *[int_1229], **kwargs_1230)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 30), tuple_1216, int32_call_result_1231)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_1211, tuple_1216)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), list_1210, tuple_1211)
    # Adding element type (line 99)
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_1232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_1233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    str_1234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 10), 'str', '4f4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 10), tuple_1233, str_1234)
    # Adding element type (line 101)
    str_1235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'str', '(3,3)f4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 10), tuple_1233, str_1235)
    # Adding element type (line 101)
    str_1236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 28), 'str', '8i4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 10), tuple_1233, str_1236)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 9), tuple_1232, tuple_1233)
    # Adding element type (line 101)
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_1237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    
    # Call to astype(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'np' (line 101)
    np_1248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 77), 'np', False)
    # Obtaining the member 'float32' of a type (line 101)
    float32_1249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 77), np_1248, 'float32')
    # Processing the call keyword arguments (line 101)
    kwargs_1250 = {}
    
    # Call to randint(...): (line 101)
    # Processing the call arguments (line 101)
    int_1241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 55), 'int')
    # Processing the call keyword arguments (line 101)
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_1242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 65), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    # Adding element type (line 101)
    int_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 65), list_1242, int_1243)
    
    keyword_1244 = list_1242
    kwargs_1245 = {'size': keyword_1244}
    # Getting the type of 'np' (line 101)
    np_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 37), 'np', False)
    # Obtaining the member 'random' of a type (line 101)
    random_1239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 37), np_1238, 'random')
    # Obtaining the member 'randint' of a type (line 101)
    randint_1240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 37), random_1239, 'randint')
    # Calling randint(args, kwargs) (line 101)
    randint_call_result_1246 = invoke(stypy.reporting.localization.Localization(__file__, 101, 37), randint_1240, *[int_1241], **kwargs_1245)
    
    # Obtaining the member 'astype' of a type (line 101)
    astype_1247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 37), randint_call_result_1246, 'astype')
    # Calling astype(args, kwargs) (line 101)
    astype_call_result_1251 = invoke(stypy.reporting.localization.Localization(__file__, 101, 37), astype_1247, *[float32_1249], **kwargs_1250)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 37), tuple_1237, astype_call_result_1251)
    # Adding element type (line 101)
    
    # Call to astype(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'np' (line 102)
    np_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 80), 'np', False)
    # Obtaining the member 'float32' of a type (line 102)
    float32_1264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 80), np_1263, 'float32')
    # Processing the call keyword arguments (line 102)
    kwargs_1265 = {}
    
    # Call to randint(...): (line 102)
    # Processing the call arguments (line 102)
    int_1255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 55), 'int')
    # Processing the call keyword arguments (line 102)
    
    # Obtaining an instance of the builtin type 'list' (line 102)
    list_1256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 65), 'list')
    # Adding type elements to the builtin type 'list' instance (line 102)
    # Adding element type (line 102)
    int_1257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 65), list_1256, int_1257)
    # Adding element type (line 102)
    int_1258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 69), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 65), list_1256, int_1258)
    
    keyword_1259 = list_1256
    kwargs_1260 = {'size': keyword_1259}
    # Getting the type of 'np' (line 102)
    np_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 37), 'np', False)
    # Obtaining the member 'random' of a type (line 102)
    random_1253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 37), np_1252, 'random')
    # Obtaining the member 'randint' of a type (line 102)
    randint_1254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 37), random_1253, 'randint')
    # Calling randint(args, kwargs) (line 102)
    randint_call_result_1261 = invoke(stypy.reporting.localization.Localization(__file__, 102, 37), randint_1254, *[int_1255], **kwargs_1260)
    
    # Obtaining the member 'astype' of a type (line 102)
    astype_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 37), randint_call_result_1261, 'astype')
    # Calling astype(args, kwargs) (line 102)
    astype_call_result_1266 = invoke(stypy.reporting.localization.Localization(__file__, 102, 37), astype_1262, *[float32_1264], **kwargs_1265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 37), tuple_1237, astype_call_result_1266)
    # Adding element type (line 101)
    
    # Call to astype(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'np' (line 103)
    np_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 77), 'np', False)
    # Obtaining the member 'int32' of a type (line 103)
    int32_1278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 77), np_1277, 'int32')
    # Processing the call keyword arguments (line 103)
    kwargs_1279 = {}
    
    # Call to randint(...): (line 103)
    # Processing the call arguments (line 103)
    int_1270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 55), 'int')
    # Processing the call keyword arguments (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_1271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 65), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    int_1272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 66), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 65), list_1271, int_1272)
    
    keyword_1273 = list_1271
    kwargs_1274 = {'size': keyword_1273}
    # Getting the type of 'np' (line 103)
    np_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'np', False)
    # Obtaining the member 'random' of a type (line 103)
    random_1268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 37), np_1267, 'random')
    # Obtaining the member 'randint' of a type (line 103)
    randint_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 37), random_1268, 'randint')
    # Calling randint(args, kwargs) (line 103)
    randint_call_result_1275 = invoke(stypy.reporting.localization.Localization(__file__, 103, 37), randint_1269, *[int_1270], **kwargs_1274)
    
    # Obtaining the member 'astype' of a type (line 103)
    astype_1276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 37), randint_call_result_1275, 'astype')
    # Calling astype(args, kwargs) (line 103)
    astype_call_result_1280 = invoke(stypy.reporting.localization.Localization(__file__, 103, 37), astype_1276, *[int32_1278], **kwargs_1279)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 37), tuple_1237, astype_call_result_1280)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 9), tuple_1232, tuple_1237)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 14), list_1210, tuple_1232)
    
    # Assigning a type to the variable 'records' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'records', list_1210)
    
    # Getting the type of 'records' (line 106)
    records_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'records')
    # Testing the type of a for loop iterable (line 106)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 4), records_1281)
    # Getting the type of the for loop variable (line 106)
    for_loop_var_1282 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 4), records_1281)
    # Assigning a type to the variable 'dtype' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'dtype', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 4), for_loop_var_1282))
    # Assigning a type to the variable 'a' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 4), for_loop_var_1282))
    # SSA begins for a for statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to FortranFile(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'tf' (line 107)
    tf_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'tf', False)
    str_1285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 29), 'str', 'w')
    # Processing the call keyword arguments (line 107)
    kwargs_1286 = {}
    # Getting the type of 'FortranFile' (line 107)
    FortranFile_1283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 107)
    FortranFile_call_result_1287 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), FortranFile_1283, *[tf_1284, str_1285], **kwargs_1286)
    
    with_1288 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 107, 13), FortranFile_call_result_1287, 'with parameter', '__enter__', '__exit__')

    if with_1288:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 107)
        enter___1289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), FortranFile_call_result_1287, '__enter__')
        with_enter_1290 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), enter___1289)
        # Assigning a type to the variable 'f' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'f', with_enter_1290)
        
        # Call to write_record(...): (line 108)
        # Getting the type of 'a' (line 108)
        a_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'a', False)
        # Processing the call keyword arguments (line 108)
        kwargs_1294 = {}
        # Getting the type of 'f' (line 108)
        f_1291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'f', False)
        # Obtaining the member 'write_record' of a type (line 108)
        write_record_1292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), f_1291, 'write_record')
        # Calling write_record(args, kwargs) (line 108)
        write_record_call_result_1295 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), write_record_1292, *[a_1293], **kwargs_1294)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 107)
        exit___1296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), FortranFile_call_result_1287, '__exit__')
        with_exit_1297 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), exit___1296, None, None, None)

    
    # Call to FortranFile(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'tf' (line 110)
    tf_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'tf', False)
    str_1300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'str', 'r')
    # Processing the call keyword arguments (line 110)
    kwargs_1301 = {}
    # Getting the type of 'FortranFile' (line 110)
    FortranFile_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 110)
    FortranFile_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), FortranFile_1298, *[tf_1299, str_1300], **kwargs_1301)
    
    with_1303 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 110, 13), FortranFile_call_result_1302, 'with parameter', '__enter__', '__exit__')

    if with_1303:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 110)
        enter___1304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), FortranFile_call_result_1302, '__enter__')
        with_enter_1305 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), enter___1304)
        # Assigning a type to the variable 'f' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'f', with_enter_1305)
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to read_record(...): (line 111)
        # Getting the type of 'dtype' (line 111)
        dtype_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 31), 'dtype', False)
        # Processing the call keyword arguments (line 111)
        kwargs_1309 = {}
        # Getting the type of 'f' (line 111)
        f_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'f', False)
        # Obtaining the member 'read_record' of a type (line 111)
        read_record_1307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), f_1306, 'read_record')
        # Calling read_record(args, kwargs) (line 111)
        read_record_call_result_1310 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), read_record_1307, *[dtype_1308], **kwargs_1309)
        
        # Assigning a type to the variable 'b' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'b', read_record_call_result_1310)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 110)
        exit___1311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), FortranFile_call_result_1302, '__exit__')
        with_exit_1312 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), exit___1311, None, None, None)

    
    # Call to assert_equal(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to len(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'a' (line 113)
    a_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'a', False)
    # Processing the call keyword arguments (line 113)
    kwargs_1316 = {}
    # Getting the type of 'len' (line 113)
    len_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'len', False)
    # Calling len(args, kwargs) (line 113)
    len_call_result_1317 = invoke(stypy.reporting.localization.Localization(__file__, 113, 21), len_1314, *[a_1315], **kwargs_1316)
    
    
    # Call to len(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'b' (line 113)
    b_1319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'b', False)
    # Processing the call keyword arguments (line 113)
    kwargs_1320 = {}
    # Getting the type of 'len' (line 113)
    len_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'len', False)
    # Calling len(args, kwargs) (line 113)
    len_call_result_1321 = invoke(stypy.reporting.localization.Localization(__file__, 113, 29), len_1318, *[b_1319], **kwargs_1320)
    
    # Processing the call keyword arguments (line 113)
    kwargs_1322 = {}
    # Getting the type of 'assert_equal' (line 113)
    assert_equal_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 113)
    assert_equal_call_result_1323 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_equal_1313, *[len_call_result_1317, len_call_result_1321], **kwargs_1322)
    
    
    
    # Call to zip(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'a' (line 115)
    a_1325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'a', False)
    # Getting the type of 'b' (line 115)
    b_1326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'b', False)
    # Processing the call keyword arguments (line 115)
    kwargs_1327 = {}
    # Getting the type of 'zip' (line 115)
    zip_1324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'zip', False)
    # Calling zip(args, kwargs) (line 115)
    zip_call_result_1328 = invoke(stypy.reporting.localization.Localization(__file__, 115, 22), zip_1324, *[a_1325, b_1326], **kwargs_1327)
    
    # Testing the type of a for loop iterable (line 115)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), zip_call_result_1328)
    # Getting the type of the for loop variable (line 115)
    for_loop_var_1329 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), zip_call_result_1328)
    # Assigning a type to the variable 'aa' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'aa', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), for_loop_var_1329))
    # Assigning a type to the variable 'bb' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'bb', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), for_loop_var_1329))
    # SSA begins for a for statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_equal(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'bb' (line 116)
    bb_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'bb', False)
    # Getting the type of 'aa' (line 116)
    aa_1332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'aa', False)
    # Processing the call keyword arguments (line 116)
    kwargs_1333 = {}
    # Getting the type of 'assert_equal' (line 116)
    assert_equal_1330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 116)
    assert_equal_call_result_1334 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), assert_equal_1330, *[bb_1331, aa_1332], **kwargs_1333)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_fortranfile_write_mixed_record(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fortranfile_write_mixed_record' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_1335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1335)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fortranfile_write_mixed_record'
    return stypy_return_type_1335

# Assigning a type to the variable 'test_fortranfile_write_mixed_record' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'test_fortranfile_write_mixed_record', test_fortranfile_write_mixed_record)

@norecursion
def test_fortran_roundtrip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fortran_roundtrip'
    module_type_store = module_type_store.open_function_context('test_fortran_roundtrip', 119, 0, False)
    
    # Passed parameters checking function
    test_fortran_roundtrip.stypy_localization = localization
    test_fortran_roundtrip.stypy_type_of_self = None
    test_fortran_roundtrip.stypy_type_store = module_type_store
    test_fortran_roundtrip.stypy_function_name = 'test_fortran_roundtrip'
    test_fortran_roundtrip.stypy_param_names_list = ['tmpdir']
    test_fortran_roundtrip.stypy_varargs_param_name = None
    test_fortran_roundtrip.stypy_kwargs_param_name = None
    test_fortran_roundtrip.stypy_call_defaults = defaults
    test_fortran_roundtrip.stypy_call_varargs = varargs
    test_fortran_roundtrip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fortran_roundtrip', ['tmpdir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fortran_roundtrip', localization, ['tmpdir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fortran_roundtrip(...)' code ##################

    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to join(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Call to str(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'tmpdir' (line 120)
    tmpdir_1339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'tmpdir', False)
    # Processing the call keyword arguments (line 120)
    kwargs_1340 = {}
    # Getting the type of 'str' (line 120)
    str_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'str', False)
    # Calling str(args, kwargs) (line 120)
    str_call_result_1341 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), str_1338, *[tmpdir_1339], **kwargs_1340)
    
    str_1342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'str', 'test.dat')
    # Processing the call keyword arguments (line 120)
    kwargs_1343 = {}
    # Getting the type of 'path' (line 120)
    path_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'path', False)
    # Obtaining the member 'join' of a type (line 120)
    join_1337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), path_1336, 'join')
    # Calling join(args, kwargs) (line 120)
    join_call_result_1344 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), join_1337, *[str_call_result_1341, str_1342], **kwargs_1343)
    
    # Assigning a type to the variable 'filename' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'filename', join_call_result_1344)
    
    # Call to seed(...): (line 122)
    # Processing the call arguments (line 122)
    int_1348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 19), 'int')
    # Processing the call keyword arguments (line 122)
    kwargs_1349 = {}
    # Getting the type of 'np' (line 122)
    np_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 122)
    random_1346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 4), np_1345, 'random')
    # Obtaining the member 'seed' of a type (line 122)
    seed_1347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 4), random_1346, 'seed')
    # Calling seed(args, kwargs) (line 122)
    seed_call_result_1350 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), seed_1347, *[int_1348], **kwargs_1349)
    
    
    # Assigning a Tuple to a Tuple (line 125):
    
    # Assigning a Num to a Name (line 125):
    int_1351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_780' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_assignment_780', int_1351)
    
    # Assigning a Num to a Name (line 125):
    int_1352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_781' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_assignment_781', int_1352)
    
    # Assigning a Num to a Name (line 125):
    int_1353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_782' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_assignment_782', int_1353)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_assignment_780' (line 125)
    tuple_assignment_780_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_assignment_780')
    # Assigning a type to the variable 'm' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'm', tuple_assignment_780_1354)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_assignment_781' (line 125)
    tuple_assignment_781_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_assignment_781')
    # Assigning a type to the variable 'n' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 7), 'n', tuple_assignment_781_1355)
    
    # Assigning a Name to a Name (line 125):
    # Getting the type of 'tuple_assignment_782' (line 125)
    tuple_assignment_782_1356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'tuple_assignment_782')
    # Assigning a type to the variable 'k' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 10), 'k', tuple_assignment_782_1356)
    
    # Assigning a Call to a Name (line 126):
    
    # Assigning a Call to a Name (line 126):
    
    # Call to randn(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'm' (line 126)
    m_1360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'm', False)
    # Getting the type of 'n' (line 126)
    n_1361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'n', False)
    # Getting the type of 'k' (line 126)
    k_1362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'k', False)
    # Processing the call keyword arguments (line 126)
    kwargs_1363 = {}
    # Getting the type of 'np' (line 126)
    np_1357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 126)
    random_1358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), np_1357, 'random')
    # Obtaining the member 'randn' of a type (line 126)
    randn_1359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), random_1358, 'randn')
    # Calling randn(args, kwargs) (line 126)
    randn_call_result_1364 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), randn_1359, *[m_1360, n_1361, k_1362], **kwargs_1363)
    
    # Assigning a type to the variable 'a' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'a', randn_call_result_1364)
    
    # Call to FortranFile(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'filename' (line 127)
    filename_1366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), 'filename', False)
    str_1367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 31), 'str', 'w')
    # Processing the call keyword arguments (line 127)
    kwargs_1368 = {}
    # Getting the type of 'FortranFile' (line 127)
    FortranFile_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 127)
    FortranFile_call_result_1369 = invoke(stypy.reporting.localization.Localization(__file__, 127, 9), FortranFile_1365, *[filename_1366, str_1367], **kwargs_1368)
    
    with_1370 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 127, 9), FortranFile_call_result_1369, 'with parameter', '__enter__', '__exit__')

    if with_1370:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 127)
        enter___1371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 9), FortranFile_call_result_1369, '__enter__')
        with_enter_1372 = invoke(stypy.reporting.localization.Localization(__file__, 127, 9), enter___1371)
        # Assigning a type to the variable 'f' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 9), 'f', with_enter_1372)
        
        # Call to write_record(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'a' (line 128)
        a_1375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'a', False)
        # Obtaining the member 'T' of a type (line 128)
        T_1376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 23), a_1375, 'T')
        # Processing the call keyword arguments (line 128)
        kwargs_1377 = {}
        # Getting the type of 'f' (line 128)
        f_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'f', False)
        # Obtaining the member 'write_record' of a type (line 128)
        write_record_1374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), f_1373, 'write_record')
        # Calling write_record(args, kwargs) (line 128)
        write_record_call_result_1378 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), write_record_1374, *[T_1376], **kwargs_1377)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 127)
        exit___1379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 9), FortranFile_call_result_1369, '__exit__')
        with_exit_1380 = invoke(stypy.reporting.localization.Localization(__file__, 127, 9), exit___1379, None, None, None)

    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to read_unformatted_double(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'm' (line 129)
    m_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 47), 'm', False)
    # Getting the type of 'n' (line 129)
    n_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'n', False)
    # Getting the type of 'k' (line 129)
    k_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 53), 'k', False)
    # Getting the type of 'filename' (line 129)
    filename_1386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'filename', False)
    # Processing the call keyword arguments (line 129)
    kwargs_1387 = {}
    # Getting the type of '_test_fortran' (line 129)
    _test_fortran_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), '_test_fortran', False)
    # Obtaining the member 'read_unformatted_double' of a type (line 129)
    read_unformatted_double_1382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 9), _test_fortran_1381, 'read_unformatted_double')
    # Calling read_unformatted_double(args, kwargs) (line 129)
    read_unformatted_double_call_result_1388 = invoke(stypy.reporting.localization.Localization(__file__, 129, 9), read_unformatted_double_1382, *[m_1383, n_1384, k_1385, filename_1386], **kwargs_1387)
    
    # Assigning a type to the variable 'a2' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'a2', read_unformatted_double_call_result_1388)
    
    # Call to FortranFile(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'filename' (line 130)
    filename_1390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'filename', False)
    str_1391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 31), 'str', 'r')
    # Processing the call keyword arguments (line 130)
    kwargs_1392 = {}
    # Getting the type of 'FortranFile' (line 130)
    FortranFile_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 130)
    FortranFile_call_result_1393 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), FortranFile_1389, *[filename_1390, str_1391], **kwargs_1392)
    
    with_1394 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 130, 9), FortranFile_call_result_1393, 'with parameter', '__enter__', '__exit__')

    if with_1394:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 130)
        enter___1395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), FortranFile_call_result_1393, '__enter__')
        with_enter_1396 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), enter___1395)
        # Assigning a type to the variable 'f' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'f', with_enter_1396)
        
        # Assigning a Attribute to a Name (line 131):
        
        # Assigning a Attribute to a Name (line 131):
        
        # Call to read_record(...): (line 131)
        # Processing the call arguments (line 131)
        str_1399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 27), 'str', '(2,3,5)f8')
        # Processing the call keyword arguments (line 131)
        kwargs_1400 = {}
        # Getting the type of 'f' (line 131)
        f_1397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'f', False)
        # Obtaining the member 'read_record' of a type (line 131)
        read_record_1398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 13), f_1397, 'read_record')
        # Calling read_record(args, kwargs) (line 131)
        read_record_call_result_1401 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), read_record_1398, *[str_1399], **kwargs_1400)
        
        # Obtaining the member 'T' of a type (line 131)
        T_1402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 13), read_record_call_result_1401, 'T')
        # Assigning a type to the variable 'a3' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'a3', T_1402)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 130)
        exit___1403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), FortranFile_call_result_1393, '__exit__')
        with_exit_1404 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), exit___1403, None, None, None)

    
    # Call to assert_equal(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'a2' (line 132)
    a2_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'a2', False)
    # Getting the type of 'a' (line 132)
    a_1407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 21), 'a', False)
    # Processing the call keyword arguments (line 132)
    kwargs_1408 = {}
    # Getting the type of 'assert_equal' (line 132)
    assert_equal_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 132)
    assert_equal_call_result_1409 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), assert_equal_1405, *[a2_1406, a_1407], **kwargs_1408)
    
    
    # Call to assert_equal(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'a3' (line 133)
    a3_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 17), 'a3', False)
    # Getting the type of 'a' (line 133)
    a_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 21), 'a', False)
    # Processing the call keyword arguments (line 133)
    kwargs_1413 = {}
    # Getting the type of 'assert_equal' (line 133)
    assert_equal_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 133)
    assert_equal_call_result_1414 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), assert_equal_1410, *[a3_1411, a_1412], **kwargs_1413)
    
    
    # Assigning a Tuple to a Tuple (line 136):
    
    # Assigning a Num to a Name (line 136):
    int_1415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_783' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_assignment_783', int_1415)
    
    # Assigning a Num to a Name (line 136):
    int_1416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_784' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_assignment_784', int_1416)
    
    # Assigning a Num to a Name (line 136):
    int_1417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_785' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_assignment_785', int_1417)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_assignment_783' (line 136)
    tuple_assignment_783_1418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_assignment_783')
    # Assigning a type to the variable 'm' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'm', tuple_assignment_783_1418)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_assignment_784' (line 136)
    tuple_assignment_784_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_assignment_784')
    # Assigning a type to the variable 'n' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 7), 'n', tuple_assignment_784_1419)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_assignment_785' (line 136)
    tuple_assignment_785_1420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'tuple_assignment_785')
    # Assigning a type to the variable 'k' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 10), 'k', tuple_assignment_785_1420)
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to astype(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'np' (line 137)
    np_1430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'np', False)
    # Obtaining the member 'int32' of a type (line 137)
    int32_1431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 40), np_1430, 'int32')
    # Processing the call keyword arguments (line 137)
    kwargs_1432 = {}
    
    # Call to randn(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'm' (line 137)
    m_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'm', False)
    # Getting the type of 'n' (line 137)
    n_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'n', False)
    # Getting the type of 'k' (line 137)
    k_1426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'k', False)
    # Processing the call keyword arguments (line 137)
    kwargs_1427 = {}
    # Getting the type of 'np' (line 137)
    np_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 137)
    random_1422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), np_1421, 'random')
    # Obtaining the member 'randn' of a type (line 137)
    randn_1423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), random_1422, 'randn')
    # Calling randn(args, kwargs) (line 137)
    randn_call_result_1428 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), randn_1423, *[m_1424, n_1425, k_1426], **kwargs_1427)
    
    # Obtaining the member 'astype' of a type (line 137)
    astype_1429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), randn_call_result_1428, 'astype')
    # Calling astype(args, kwargs) (line 137)
    astype_call_result_1433 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), astype_1429, *[int32_1431], **kwargs_1432)
    
    # Assigning a type to the variable 'a' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'a', astype_call_result_1433)
    
    # Call to FortranFile(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'filename' (line 138)
    filename_1435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'filename', False)
    str_1436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'str', 'w')
    # Processing the call keyword arguments (line 138)
    kwargs_1437 = {}
    # Getting the type of 'FortranFile' (line 138)
    FortranFile_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 138)
    FortranFile_call_result_1438 = invoke(stypy.reporting.localization.Localization(__file__, 138, 9), FortranFile_1434, *[filename_1435, str_1436], **kwargs_1437)
    
    with_1439 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 138, 9), FortranFile_call_result_1438, 'with parameter', '__enter__', '__exit__')

    if with_1439:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 138)
        enter___1440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 9), FortranFile_call_result_1438, '__enter__')
        with_enter_1441 = invoke(stypy.reporting.localization.Localization(__file__, 138, 9), enter___1440)
        # Assigning a type to the variable 'f' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'f', with_enter_1441)
        
        # Call to write_record(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'a' (line 139)
        a_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'a', False)
        # Obtaining the member 'T' of a type (line 139)
        T_1445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 23), a_1444, 'T')
        # Processing the call keyword arguments (line 139)
        kwargs_1446 = {}
        # Getting the type of 'f' (line 139)
        f_1442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'f', False)
        # Obtaining the member 'write_record' of a type (line 139)
        write_record_1443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), f_1442, 'write_record')
        # Calling write_record(args, kwargs) (line 139)
        write_record_call_result_1447 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), write_record_1443, *[T_1445], **kwargs_1446)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 138)
        exit___1448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 9), FortranFile_call_result_1438, '__exit__')
        with_exit_1449 = invoke(stypy.reporting.localization.Localization(__file__, 138, 9), exit___1448, None, None, None)

    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to read_unformatted_int(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'm' (line 140)
    m_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'm', False)
    # Getting the type of 'n' (line 140)
    n_1453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'n', False)
    # Getting the type of 'k' (line 140)
    k_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 50), 'k', False)
    # Getting the type of 'filename' (line 140)
    filename_1455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 53), 'filename', False)
    # Processing the call keyword arguments (line 140)
    kwargs_1456 = {}
    # Getting the type of '_test_fortran' (line 140)
    _test_fortran_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), '_test_fortran', False)
    # Obtaining the member 'read_unformatted_int' of a type (line 140)
    read_unformatted_int_1451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 9), _test_fortran_1450, 'read_unformatted_int')
    # Calling read_unformatted_int(args, kwargs) (line 140)
    read_unformatted_int_call_result_1457 = invoke(stypy.reporting.localization.Localization(__file__, 140, 9), read_unformatted_int_1451, *[m_1452, n_1453, k_1454, filename_1455], **kwargs_1456)
    
    # Assigning a type to the variable 'a2' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'a2', read_unformatted_int_call_result_1457)
    
    # Call to FortranFile(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'filename' (line 141)
    filename_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'filename', False)
    str_1460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 31), 'str', 'r')
    # Processing the call keyword arguments (line 141)
    kwargs_1461 = {}
    # Getting the type of 'FortranFile' (line 141)
    FortranFile_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 141)
    FortranFile_call_result_1462 = invoke(stypy.reporting.localization.Localization(__file__, 141, 9), FortranFile_1458, *[filename_1459, str_1460], **kwargs_1461)
    
    with_1463 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 141, 9), FortranFile_call_result_1462, 'with parameter', '__enter__', '__exit__')

    if with_1463:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 141)
        enter___1464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 9), FortranFile_call_result_1462, '__enter__')
        with_enter_1465 = invoke(stypy.reporting.localization.Localization(__file__, 141, 9), enter___1464)
        # Assigning a type to the variable 'f' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'f', with_enter_1465)
        
        # Assigning a Attribute to a Name (line 142):
        
        # Assigning a Attribute to a Name (line 142):
        
        # Call to read_record(...): (line 142)
        # Processing the call arguments (line 142)
        str_1468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 27), 'str', '(2,3,5)i4')
        # Processing the call keyword arguments (line 142)
        kwargs_1469 = {}
        # Getting the type of 'f' (line 142)
        f_1466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'f', False)
        # Obtaining the member 'read_record' of a type (line 142)
        read_record_1467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 13), f_1466, 'read_record')
        # Calling read_record(args, kwargs) (line 142)
        read_record_call_result_1470 = invoke(stypy.reporting.localization.Localization(__file__, 142, 13), read_record_1467, *[str_1468], **kwargs_1469)
        
        # Obtaining the member 'T' of a type (line 142)
        T_1471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 13), read_record_call_result_1470, 'T')
        # Assigning a type to the variable 'a3' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'a3', T_1471)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 141)
        exit___1472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 9), FortranFile_call_result_1462, '__exit__')
        with_exit_1473 = invoke(stypy.reporting.localization.Localization(__file__, 141, 9), exit___1472, None, None, None)

    
    # Call to assert_equal(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'a2' (line 143)
    a2_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 17), 'a2', False)
    # Getting the type of 'a' (line 143)
    a_1476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'a', False)
    # Processing the call keyword arguments (line 143)
    kwargs_1477 = {}
    # Getting the type of 'assert_equal' (line 143)
    assert_equal_1474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 143)
    assert_equal_call_result_1478 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), assert_equal_1474, *[a2_1475, a_1476], **kwargs_1477)
    
    
    # Call to assert_equal(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'a3' (line 144)
    a3_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 17), 'a3', False)
    # Getting the type of 'a' (line 144)
    a_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'a', False)
    # Processing the call keyword arguments (line 144)
    kwargs_1482 = {}
    # Getting the type of 'assert_equal' (line 144)
    assert_equal_1479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 144)
    assert_equal_call_result_1483 = invoke(stypy.reporting.localization.Localization(__file__, 144, 4), assert_equal_1479, *[a3_1480, a_1481], **kwargs_1482)
    
    
    # Assigning a Tuple to a Tuple (line 147):
    
    # Assigning a Num to a Name (line 147):
    int_1484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 14), 'int')
    # Assigning a type to the variable 'tuple_assignment_786' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_assignment_786', int_1484)
    
    # Assigning a Num to a Name (line 147):
    int_1485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_787' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_assignment_787', int_1485)
    
    # Assigning a Num to a Name (line 147):
    int_1486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'int')
    # Assigning a type to the variable 'tuple_assignment_788' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_assignment_788', int_1486)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_assignment_786' (line 147)
    tuple_assignment_786_1487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_assignment_786')
    # Assigning a type to the variable 'm' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'm', tuple_assignment_786_1487)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_assignment_787' (line 147)
    tuple_assignment_787_1488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_assignment_787')
    # Assigning a type to the variable 'n' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'n', tuple_assignment_787_1488)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_assignment_788' (line 147)
    tuple_assignment_788_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'tuple_assignment_788')
    # Assigning a type to the variable 'k' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 10), 'k', tuple_assignment_788_1489)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to randn(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'm' (line 148)
    m_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'm', False)
    # Getting the type of 'n' (line 148)
    n_1494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'n', False)
    # Processing the call keyword arguments (line 148)
    kwargs_1495 = {}
    # Getting the type of 'np' (line 148)
    np_1490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 148)
    random_1491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), np_1490, 'random')
    # Obtaining the member 'randn' of a type (line 148)
    randn_1492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), random_1491, 'randn')
    # Calling randn(args, kwargs) (line 148)
    randn_call_result_1496 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), randn_1492, *[m_1493, n_1494], **kwargs_1495)
    
    # Assigning a type to the variable 'a' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'a', randn_call_result_1496)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to astype(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'np' (line 149)
    np_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 34), 'np', False)
    # Obtaining the member 'intc' of a type (line 149)
    intc_1505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 34), np_1504, 'intc')
    # Processing the call keyword arguments (line 149)
    kwargs_1506 = {}
    
    # Call to randn(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'k' (line 149)
    k_1500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'k', False)
    # Processing the call keyword arguments (line 149)
    kwargs_1501 = {}
    # Getting the type of 'np' (line 149)
    np_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 149)
    random_1498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), np_1497, 'random')
    # Obtaining the member 'randn' of a type (line 149)
    randn_1499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), random_1498, 'randn')
    # Calling randn(args, kwargs) (line 149)
    randn_call_result_1502 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), randn_1499, *[k_1500], **kwargs_1501)
    
    # Obtaining the member 'astype' of a type (line 149)
    astype_1503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), randn_call_result_1502, 'astype')
    # Calling astype(args, kwargs) (line 149)
    astype_call_result_1507 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), astype_1503, *[intc_1505], **kwargs_1506)
    
    # Assigning a type to the variable 'b' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'b', astype_call_result_1507)
    
    # Call to FortranFile(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'filename' (line 150)
    filename_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'filename', False)
    str_1510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 31), 'str', 'w')
    # Processing the call keyword arguments (line 150)
    kwargs_1511 = {}
    # Getting the type of 'FortranFile' (line 150)
    FortranFile_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 150)
    FortranFile_call_result_1512 = invoke(stypy.reporting.localization.Localization(__file__, 150, 9), FortranFile_1508, *[filename_1509, str_1510], **kwargs_1511)
    
    with_1513 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 150, 9), FortranFile_call_result_1512, 'with parameter', '__enter__', '__exit__')

    if with_1513:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 150)
        enter___1514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 9), FortranFile_call_result_1512, '__enter__')
        with_enter_1515 = invoke(stypy.reporting.localization.Localization(__file__, 150, 9), enter___1514)
        # Assigning a type to the variable 'f' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 9), 'f', with_enter_1515)
        
        # Call to write_record(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'a' (line 151)
        a_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'a', False)
        # Obtaining the member 'T' of a type (line 151)
        T_1519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 23), a_1518, 'T')
        # Getting the type of 'b' (line 151)
        b_1520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'b', False)
        # Obtaining the member 'T' of a type (line 151)
        T_1521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 28), b_1520, 'T')
        # Processing the call keyword arguments (line 151)
        kwargs_1522 = {}
        # Getting the type of 'f' (line 151)
        f_1516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'f', False)
        # Obtaining the member 'write_record' of a type (line 151)
        write_record_1517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), f_1516, 'write_record')
        # Calling write_record(args, kwargs) (line 151)
        write_record_call_result_1523 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), write_record_1517, *[T_1519, T_1521], **kwargs_1522)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 150)
        exit___1524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 9), FortranFile_call_result_1512, '__exit__')
        with_exit_1525 = invoke(stypy.reporting.localization.Localization(__file__, 150, 9), exit___1524, None, None, None)

    
    # Assigning a Call to a Tuple (line 152):
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_1526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 4), 'int')
    
    # Call to read_unformatted_mixed(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'm' (line 152)
    m_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 50), 'm', False)
    # Getting the type of 'n' (line 152)
    n_1530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 53), 'n', False)
    # Getting the type of 'k' (line 152)
    k_1531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 56), 'k', False)
    # Getting the type of 'filename' (line 152)
    filename_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 59), 'filename', False)
    # Processing the call keyword arguments (line 152)
    kwargs_1533 = {}
    # Getting the type of '_test_fortran' (line 152)
    _test_fortran_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), '_test_fortran', False)
    # Obtaining the member 'read_unformatted_mixed' of a type (line 152)
    read_unformatted_mixed_1528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), _test_fortran_1527, 'read_unformatted_mixed')
    # Calling read_unformatted_mixed(args, kwargs) (line 152)
    read_unformatted_mixed_call_result_1534 = invoke(stypy.reporting.localization.Localization(__file__, 152, 13), read_unformatted_mixed_1528, *[m_1529, n_1530, k_1531, filename_1532], **kwargs_1533)
    
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___1535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), read_unformatted_mixed_call_result_1534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_1536 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), getitem___1535, int_1526)
    
    # Assigning a type to the variable 'tuple_var_assignment_789' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_789', subscript_call_result_1536)
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_1537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 4), 'int')
    
    # Call to read_unformatted_mixed(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'm' (line 152)
    m_1540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 50), 'm', False)
    # Getting the type of 'n' (line 152)
    n_1541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 53), 'n', False)
    # Getting the type of 'k' (line 152)
    k_1542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 56), 'k', False)
    # Getting the type of 'filename' (line 152)
    filename_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 59), 'filename', False)
    # Processing the call keyword arguments (line 152)
    kwargs_1544 = {}
    # Getting the type of '_test_fortran' (line 152)
    _test_fortran_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), '_test_fortran', False)
    # Obtaining the member 'read_unformatted_mixed' of a type (line 152)
    read_unformatted_mixed_1539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), _test_fortran_1538, 'read_unformatted_mixed')
    # Calling read_unformatted_mixed(args, kwargs) (line 152)
    read_unformatted_mixed_call_result_1545 = invoke(stypy.reporting.localization.Localization(__file__, 152, 13), read_unformatted_mixed_1539, *[m_1540, n_1541, k_1542, filename_1543], **kwargs_1544)
    
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___1546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), read_unformatted_mixed_call_result_1545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_1547 = invoke(stypy.reporting.localization.Localization(__file__, 152, 4), getitem___1546, int_1537)
    
    # Assigning a type to the variable 'tuple_var_assignment_790' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_790', subscript_call_result_1547)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_789' (line 152)
    tuple_var_assignment_789_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_789')
    # Assigning a type to the variable 'a2' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'a2', tuple_var_assignment_789_1548)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_790' (line 152)
    tuple_var_assignment_790_1549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tuple_var_assignment_790')
    # Assigning a type to the variable 'b2' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'b2', tuple_var_assignment_790_1549)
    
    # Call to FortranFile(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'filename' (line 153)
    filename_1551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'filename', False)
    str_1552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 31), 'str', 'r')
    # Processing the call keyword arguments (line 153)
    kwargs_1553 = {}
    # Getting the type of 'FortranFile' (line 153)
    FortranFile_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 9), 'FortranFile', False)
    # Calling FortranFile(args, kwargs) (line 153)
    FortranFile_call_result_1554 = invoke(stypy.reporting.localization.Localization(__file__, 153, 9), FortranFile_1550, *[filename_1551, str_1552], **kwargs_1553)
    
    with_1555 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 153, 9), FortranFile_call_result_1554, 'with parameter', '__enter__', '__exit__')

    if with_1555:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 153)
        enter___1556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 9), FortranFile_call_result_1554, '__enter__')
        with_enter_1557 = invoke(stypy.reporting.localization.Localization(__file__, 153, 9), enter___1556)
        # Assigning a type to the variable 'f' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 9), 'f', with_enter_1557)
        
        # Assigning a Call to a Tuple (line 154):
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_1558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        
        # Call to read_record(...): (line 154)
        # Processing the call arguments (line 154)
        str_1561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'str', '(3,5)f8')
        str_1562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 42), 'str', '2i4')
        # Processing the call keyword arguments (line 154)
        kwargs_1563 = {}
        # Getting the type of 'f' (line 154)
        f_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'f', False)
        # Obtaining the member 'read_record' of a type (line 154)
        read_record_1560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 17), f_1559, 'read_record')
        # Calling read_record(args, kwargs) (line 154)
        read_record_call_result_1564 = invoke(stypy.reporting.localization.Localization(__file__, 154, 17), read_record_1560, *[str_1561, str_1562], **kwargs_1563)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___1565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), read_record_call_result_1564, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_1566 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), getitem___1565, int_1558)
        
        # Assigning a type to the variable 'tuple_var_assignment_791' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_791', subscript_call_result_1566)
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_1567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'int')
        
        # Call to read_record(...): (line 154)
        # Processing the call arguments (line 154)
        str_1570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'str', '(3,5)f8')
        str_1571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 42), 'str', '2i4')
        # Processing the call keyword arguments (line 154)
        kwargs_1572 = {}
        # Getting the type of 'f' (line 154)
        f_1568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 17), 'f', False)
        # Obtaining the member 'read_record' of a type (line 154)
        read_record_1569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 17), f_1568, 'read_record')
        # Calling read_record(args, kwargs) (line 154)
        read_record_call_result_1573 = invoke(stypy.reporting.localization.Localization(__file__, 154, 17), read_record_1569, *[str_1570, str_1571], **kwargs_1572)
        
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___1574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), read_record_call_result_1573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_1575 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), getitem___1574, int_1567)
        
        # Assigning a type to the variable 'tuple_var_assignment_792' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_792', subscript_call_result_1575)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'tuple_var_assignment_791' (line 154)
        tuple_var_assignment_791_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_791')
        # Assigning a type to the variable 'a3' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'a3', tuple_var_assignment_791_1576)
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'tuple_var_assignment_792' (line 154)
        tuple_var_assignment_792_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'tuple_var_assignment_792')
        # Assigning a type to the variable 'b3' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'b3', tuple_var_assignment_792_1577)
        
        # Assigning a Attribute to a Name (line 155):
        
        # Assigning a Attribute to a Name (line 155):
        # Getting the type of 'a3' (line 155)
        a3_1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 13), 'a3')
        # Obtaining the member 'T' of a type (line 155)
        T_1579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 13), a3_1578, 'T')
        # Assigning a type to the variable 'a3' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'a3', T_1579)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 153)
        exit___1580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 9), FortranFile_call_result_1554, '__exit__')
        with_exit_1581 = invoke(stypy.reporting.localization.Localization(__file__, 153, 9), exit___1580, None, None, None)

    
    # Call to assert_equal(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'a2' (line 156)
    a2_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 17), 'a2', False)
    # Getting the type of 'a' (line 156)
    a_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'a', False)
    # Processing the call keyword arguments (line 156)
    kwargs_1585 = {}
    # Getting the type of 'assert_equal' (line 156)
    assert_equal_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 156)
    assert_equal_call_result_1586 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), assert_equal_1582, *[a2_1583, a_1584], **kwargs_1585)
    
    
    # Call to assert_equal(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'a3' (line 157)
    a3_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'a3', False)
    # Getting the type of 'a' (line 157)
    a_1589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'a', False)
    # Processing the call keyword arguments (line 157)
    kwargs_1590 = {}
    # Getting the type of 'assert_equal' (line 157)
    assert_equal_1587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 157)
    assert_equal_call_result_1591 = invoke(stypy.reporting.localization.Localization(__file__, 157, 4), assert_equal_1587, *[a3_1588, a_1589], **kwargs_1590)
    
    
    # Call to assert_equal(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'b2' (line 158)
    b2_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'b2', False)
    # Getting the type of 'b' (line 158)
    b_1594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'b', False)
    # Processing the call keyword arguments (line 158)
    kwargs_1595 = {}
    # Getting the type of 'assert_equal' (line 158)
    assert_equal_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 158)
    assert_equal_call_result_1596 = invoke(stypy.reporting.localization.Localization(__file__, 158, 4), assert_equal_1592, *[b2_1593, b_1594], **kwargs_1595)
    
    
    # Call to assert_equal(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'b3' (line 159)
    b3_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'b3', False)
    # Getting the type of 'b' (line 159)
    b_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 21), 'b', False)
    # Processing the call keyword arguments (line 159)
    kwargs_1600 = {}
    # Getting the type of 'assert_equal' (line 159)
    assert_equal_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 159)
    assert_equal_call_result_1601 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), assert_equal_1597, *[b3_1598, b_1599], **kwargs_1600)
    
    
    # ################# End of 'test_fortran_roundtrip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fortran_roundtrip' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_1602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1602)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fortran_roundtrip'
    return stypy_return_type_1602

# Assigning a type to the variable 'test_fortran_roundtrip' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'test_fortran_roundtrip', test_fortran_roundtrip)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
