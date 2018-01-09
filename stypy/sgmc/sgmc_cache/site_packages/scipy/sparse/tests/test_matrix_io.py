
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: import os
5: import numpy as np
6: import tempfile
7: 
8: import pytest
9: from pytest import raises as assert_raises
10: from numpy.testing import assert_equal, assert_
11: from scipy._lib._version import NumpyVersion
12: 
13: from scipy.sparse import (csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
14:                           coo_matrix, save_npz, load_npz)
15: 
16: 
17: DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
18: 
19: 
20: def _save_and_load(matrix):
21:     fd, tmpfile = tempfile.mkstemp(suffix='.npz')
22:     os.close(fd)
23:     try:
24:         save_npz(tmpfile, matrix)
25:         loaded_matrix = load_npz(tmpfile)
26:     finally:
27:         os.remove(tmpfile)
28:     return loaded_matrix
29: 
30: def _check_save_and_load(dense_matrix):
31:     for matrix_class in [csc_matrix, csr_matrix, bsr_matrix, dia_matrix, coo_matrix]:
32:         matrix = matrix_class(dense_matrix)
33:         loaded_matrix = _save_and_load(matrix)
34:         assert_(type(loaded_matrix) is matrix_class)
35:         assert_(loaded_matrix.shape == dense_matrix.shape)
36:         assert_(loaded_matrix.dtype == dense_matrix.dtype)
37:         assert_equal(loaded_matrix.toarray(), dense_matrix)
38: 
39: def test_save_and_load_random():
40:     N = 10
41:     np.random.seed(0)
42:     dense_matrix = np.random.random((N, N))
43:     dense_matrix[dense_matrix > 0.7] = 0
44:     _check_save_and_load(dense_matrix)
45: 
46: def test_save_and_load_empty():
47:     dense_matrix = np.zeros((4,6))
48:     _check_save_and_load(dense_matrix)
49: 
50: def test_save_and_load_one_entry():
51:     dense_matrix = np.zeros((4,6))
52:     dense_matrix[1,2] = 1
53:     _check_save_and_load(dense_matrix)
54: 
55: 
56: @pytest.mark.skipif(NumpyVersion(np.__version__) < '1.10.0',
57:                     reason='disabling unpickling requires numpy >= 1.10.0')
58: def test_malicious_load():
59:     class Executor(object):
60:         def __reduce__(self):
61:             return (assert_, (False, 'unexpected code execution'))
62: 
63:     fd, tmpfile = tempfile.mkstemp(suffix='.npz')
64:     os.close(fd)
65:     try:
66:         np.savez(tmpfile, format=Executor())
67: 
68:         # Should raise a ValueError, not execute code
69:         assert_raises(ValueError, load_npz, tmpfile)
70:     finally:
71:         os.remove(tmpfile)
72: 
73: 
74: def test_py23_compatibility():
75:     # Try loading files saved on Python 2 and Python 3.  They are not
76:     # the same, since files saved with Scipy versions < 1.0.0 may
77:     # contain unicode.
78: 
79:     a = load_npz(os.path.join(DATA_DIR, 'csc_py2.npz'))
80:     b = load_npz(os.path.join(DATA_DIR, 'csc_py3.npz'))
81:     c = csc_matrix([[0]])
82: 
83:     assert_equal(a.toarray(), c.toarray())
84:     assert_equal(b.toarray(), c.toarray())
85: 
86: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460154 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_460154) is not StypyTypeError):

    if (import_460154 != 'pyd_module'):
        __import__(import_460154)
        sys_modules_460155 = sys.modules[import_460154]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_460155.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_460154)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import tempfile' statement (line 6)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import pytest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460156 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_460156) is not StypyTypeError):

    if (import_460156 != 'pyd_module'):
        __import__(import_460156)
        sys_modules_460157 = sys.modules[import_460156]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_460157.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_460156)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from pytest import assert_raises' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460158 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_460158) is not StypyTypeError):

    if (import_460158 != 'pyd_module'):
        __import__(import_460158)
        sys_modules_460159 = sys.modules[import_460158]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_460159.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_460159, sys_modules_460159.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_460158)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.testing import assert_equal, assert_' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460160 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing')

if (type(import_460160) is not StypyTypeError):

    if (import_460160 != 'pyd_module'):
        __import__(import_460160)
        sys_modules_460161 = sys.modules[import_460160]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', sys_modules_460161.module_type_store, module_type_store, ['assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_460161, sys_modules_460161.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_'], [assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', import_460160)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._version import NumpyVersion' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460162 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version')

if (type(import_460162) is not StypyTypeError):

    if (import_460162 != 'pyd_module'):
        __import__(import_460162)
        sys_modules_460163 = sys.modules[import_460162]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version', sys_modules_460163.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_460163, sys_modules_460163.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._version', import_460162)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse import csc_matrix, csr_matrix, bsr_matrix, dia_matrix, coo_matrix, save_npz, load_npz' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/tests/')
import_460164 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse')

if (type(import_460164) is not StypyTypeError):

    if (import_460164 != 'pyd_module'):
        __import__(import_460164)
        sys_modules_460165 = sys.modules[import_460164]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', sys_modules_460165.module_type_store, module_type_store, ['csc_matrix', 'csr_matrix', 'bsr_matrix', 'dia_matrix', 'coo_matrix', 'save_npz', 'load_npz'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_460165, sys_modules_460165.module_type_store, module_type_store)
    else:
        from scipy.sparse import csc_matrix, csr_matrix, bsr_matrix, dia_matrix, coo_matrix, save_npz, load_npz

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', None, module_type_store, ['csc_matrix', 'csr_matrix', 'bsr_matrix', 'dia_matrix', 'coo_matrix', 'save_npz', 'load_npz'], [csc_matrix, csr_matrix, bsr_matrix, dia_matrix, coo_matrix, save_npz, load_npz])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', import_460164)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/tests/')


# Assigning a Call to a Name (line 17):

# Assigning a Call to a Name (line 17):

# Call to join(...): (line 17)
# Processing the call arguments (line 17)

# Call to dirname(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of '__file__' (line 17)
file___460172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 40), '__file__', False)
# Processing the call keyword arguments (line 17)
kwargs_460173 = {}
# Getting the type of 'os' (line 17)
os_460169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'os', False)
# Obtaining the member 'path' of a type (line 17)
path_460170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), os_460169, 'path')
# Obtaining the member 'dirname' of a type (line 17)
dirname_460171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), path_460170, 'dirname')
# Calling dirname(args, kwargs) (line 17)
dirname_call_result_460174 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), dirname_460171, *[file___460172], **kwargs_460173)

str_460175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 51), 'str', 'data')
# Processing the call keyword arguments (line 17)
kwargs_460176 = {}
# Getting the type of 'os' (line 17)
os_460166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'os', False)
# Obtaining the member 'path' of a type (line 17)
path_460167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 11), os_460166, 'path')
# Obtaining the member 'join' of a type (line 17)
join_460168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 11), path_460167, 'join')
# Calling join(args, kwargs) (line 17)
join_call_result_460177 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), join_460168, *[dirname_call_result_460174, str_460175], **kwargs_460176)

# Assigning a type to the variable 'DATA_DIR' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'DATA_DIR', join_call_result_460177)

@norecursion
def _save_and_load(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_save_and_load'
    module_type_store = module_type_store.open_function_context('_save_and_load', 20, 0, False)
    
    # Passed parameters checking function
    _save_and_load.stypy_localization = localization
    _save_and_load.stypy_type_of_self = None
    _save_and_load.stypy_type_store = module_type_store
    _save_and_load.stypy_function_name = '_save_and_load'
    _save_and_load.stypy_param_names_list = ['matrix']
    _save_and_load.stypy_varargs_param_name = None
    _save_and_load.stypy_kwargs_param_name = None
    _save_and_load.stypy_call_defaults = defaults
    _save_and_load.stypy_call_varargs = varargs
    _save_and_load.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_save_and_load', ['matrix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_save_and_load', localization, ['matrix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_save_and_load(...)' code ##################

    
    # Assigning a Call to a Tuple (line 21):
    
    # Assigning a Subscript to a Name (line 21):
    
    # Obtaining the type of the subscript
    int_460178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'int')
    
    # Call to mkstemp(...): (line 21)
    # Processing the call keyword arguments (line 21)
    str_460181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'str', '.npz')
    keyword_460182 = str_460181
    kwargs_460183 = {'suffix': keyword_460182}
    # Getting the type of 'tempfile' (line 21)
    tempfile_460179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 21)
    mkstemp_460180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 18), tempfile_460179, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 21)
    mkstemp_call_result_460184 = invoke(stypy.reporting.localization.Localization(__file__, 21, 18), mkstemp_460180, *[], **kwargs_460183)
    
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___460185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), mkstemp_call_result_460184, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_460186 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), getitem___460185, int_460178)
    
    # Assigning a type to the variable 'tuple_var_assignment_460150' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_460150', subscript_call_result_460186)
    
    # Assigning a Subscript to a Name (line 21):
    
    # Obtaining the type of the subscript
    int_460187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'int')
    
    # Call to mkstemp(...): (line 21)
    # Processing the call keyword arguments (line 21)
    str_460190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'str', '.npz')
    keyword_460191 = str_460190
    kwargs_460192 = {'suffix': keyword_460191}
    # Getting the type of 'tempfile' (line 21)
    tempfile_460188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 21)
    mkstemp_460189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 18), tempfile_460188, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 21)
    mkstemp_call_result_460193 = invoke(stypy.reporting.localization.Localization(__file__, 21, 18), mkstemp_460189, *[], **kwargs_460192)
    
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___460194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), mkstemp_call_result_460193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_460195 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), getitem___460194, int_460187)
    
    # Assigning a type to the variable 'tuple_var_assignment_460151' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_460151', subscript_call_result_460195)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'tuple_var_assignment_460150' (line 21)
    tuple_var_assignment_460150_460196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_460150')
    # Assigning a type to the variable 'fd' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'fd', tuple_var_assignment_460150_460196)
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'tuple_var_assignment_460151' (line 21)
    tuple_var_assignment_460151_460197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'tuple_var_assignment_460151')
    # Assigning a type to the variable 'tmpfile' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'tmpfile', tuple_var_assignment_460151_460197)
    
    # Call to close(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'fd' (line 22)
    fd_460200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'fd', False)
    # Processing the call keyword arguments (line 22)
    kwargs_460201 = {}
    # Getting the type of 'os' (line 22)
    os_460198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'os', False)
    # Obtaining the member 'close' of a type (line 22)
    close_460199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), os_460198, 'close')
    # Calling close(args, kwargs) (line 22)
    close_call_result_460202 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), close_460199, *[fd_460200], **kwargs_460201)
    
    
    # Try-finally block (line 23)
    
    # Call to save_npz(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'tmpfile' (line 24)
    tmpfile_460204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'tmpfile', False)
    # Getting the type of 'matrix' (line 24)
    matrix_460205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'matrix', False)
    # Processing the call keyword arguments (line 24)
    kwargs_460206 = {}
    # Getting the type of 'save_npz' (line 24)
    save_npz_460203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'save_npz', False)
    # Calling save_npz(args, kwargs) (line 24)
    save_npz_call_result_460207 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), save_npz_460203, *[tmpfile_460204, matrix_460205], **kwargs_460206)
    
    
    # Assigning a Call to a Name (line 25):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to load_npz(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'tmpfile' (line 25)
    tmpfile_460209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 33), 'tmpfile', False)
    # Processing the call keyword arguments (line 25)
    kwargs_460210 = {}
    # Getting the type of 'load_npz' (line 25)
    load_npz_460208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'load_npz', False)
    # Calling load_npz(args, kwargs) (line 25)
    load_npz_call_result_460211 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), load_npz_460208, *[tmpfile_460209], **kwargs_460210)
    
    # Assigning a type to the variable 'loaded_matrix' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'loaded_matrix', load_npz_call_result_460211)
    
    # finally branch of the try-finally block (line 23)
    
    # Call to remove(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'tmpfile' (line 27)
    tmpfile_460214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'tmpfile', False)
    # Processing the call keyword arguments (line 27)
    kwargs_460215 = {}
    # Getting the type of 'os' (line 27)
    os_460212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'os', False)
    # Obtaining the member 'remove' of a type (line 27)
    remove_460213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), os_460212, 'remove')
    # Calling remove(args, kwargs) (line 27)
    remove_call_result_460216 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), remove_460213, *[tmpfile_460214], **kwargs_460215)
    
    
    # Getting the type of 'loaded_matrix' (line 28)
    loaded_matrix_460217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'loaded_matrix')
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', loaded_matrix_460217)
    
    # ################# End of '_save_and_load(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_save_and_load' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_460218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460218)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_save_and_load'
    return stypy_return_type_460218

# Assigning a type to the variable '_save_and_load' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '_save_and_load', _save_and_load)

@norecursion
def _check_save_and_load(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_save_and_load'
    module_type_store = module_type_store.open_function_context('_check_save_and_load', 30, 0, False)
    
    # Passed parameters checking function
    _check_save_and_load.stypy_localization = localization
    _check_save_and_load.stypy_type_of_self = None
    _check_save_and_load.stypy_type_store = module_type_store
    _check_save_and_load.stypy_function_name = '_check_save_and_load'
    _check_save_and_load.stypy_param_names_list = ['dense_matrix']
    _check_save_and_load.stypy_varargs_param_name = None
    _check_save_and_load.stypy_kwargs_param_name = None
    _check_save_and_load.stypy_call_defaults = defaults
    _check_save_and_load.stypy_call_varargs = varargs
    _check_save_and_load.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_save_and_load', ['dense_matrix'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_save_and_load', localization, ['dense_matrix'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_save_and_load(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_460219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    # Getting the type of 'csc_matrix' (line 31)
    csc_matrix_460220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'csc_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_460219, csc_matrix_460220)
    # Adding element type (line 31)
    # Getting the type of 'csr_matrix' (line 31)
    csr_matrix_460221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'csr_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_460219, csr_matrix_460221)
    # Adding element type (line 31)
    # Getting the type of 'bsr_matrix' (line 31)
    bsr_matrix_460222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 49), 'bsr_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_460219, bsr_matrix_460222)
    # Adding element type (line 31)
    # Getting the type of 'dia_matrix' (line 31)
    dia_matrix_460223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 61), 'dia_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_460219, dia_matrix_460223)
    # Adding element type (line 31)
    # Getting the type of 'coo_matrix' (line 31)
    coo_matrix_460224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 73), 'coo_matrix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_460219, coo_matrix_460224)
    
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), list_460219)
    # Getting the type of the for loop variable (line 31)
    for_loop_var_460225 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), list_460219)
    # Assigning a type to the variable 'matrix_class' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'matrix_class', for_loop_var_460225)
    # SSA begins for a for statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to matrix_class(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'dense_matrix' (line 32)
    dense_matrix_460227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'dense_matrix', False)
    # Processing the call keyword arguments (line 32)
    kwargs_460228 = {}
    # Getting the type of 'matrix_class' (line 32)
    matrix_class_460226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'matrix_class', False)
    # Calling matrix_class(args, kwargs) (line 32)
    matrix_class_call_result_460229 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), matrix_class_460226, *[dense_matrix_460227], **kwargs_460228)
    
    # Assigning a type to the variable 'matrix' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'matrix', matrix_class_call_result_460229)
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to _save_and_load(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'matrix' (line 33)
    matrix_460231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 39), 'matrix', False)
    # Processing the call keyword arguments (line 33)
    kwargs_460232 = {}
    # Getting the type of '_save_and_load' (line 33)
    _save_and_load_460230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), '_save_and_load', False)
    # Calling _save_and_load(args, kwargs) (line 33)
    _save_and_load_call_result_460233 = invoke(stypy.reporting.localization.Localization(__file__, 33, 24), _save_and_load_460230, *[matrix_460231], **kwargs_460232)
    
    # Assigning a type to the variable 'loaded_matrix' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'loaded_matrix', _save_and_load_call_result_460233)
    
    # Call to assert_(...): (line 34)
    # Processing the call arguments (line 34)
    
    
    # Call to type(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'loaded_matrix' (line 34)
    loaded_matrix_460236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'loaded_matrix', False)
    # Processing the call keyword arguments (line 34)
    kwargs_460237 = {}
    # Getting the type of 'type' (line 34)
    type_460235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'type', False)
    # Calling type(args, kwargs) (line 34)
    type_call_result_460238 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), type_460235, *[loaded_matrix_460236], **kwargs_460237)
    
    # Getting the type of 'matrix_class' (line 34)
    matrix_class_460239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), 'matrix_class', False)
    # Applying the binary operator 'is' (line 34)
    result_is__460240 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 16), 'is', type_call_result_460238, matrix_class_460239)
    
    # Processing the call keyword arguments (line 34)
    kwargs_460241 = {}
    # Getting the type of 'assert_' (line 34)
    assert__460234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 34)
    assert__call_result_460242 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert__460234, *[result_is__460240], **kwargs_460241)
    
    
    # Call to assert_(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Getting the type of 'loaded_matrix' (line 35)
    loaded_matrix_460244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'loaded_matrix', False)
    # Obtaining the member 'shape' of a type (line 35)
    shape_460245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), loaded_matrix_460244, 'shape')
    # Getting the type of 'dense_matrix' (line 35)
    dense_matrix_460246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'dense_matrix', False)
    # Obtaining the member 'shape' of a type (line 35)
    shape_460247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 39), dense_matrix_460246, 'shape')
    # Applying the binary operator '==' (line 35)
    result_eq_460248 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 16), '==', shape_460245, shape_460247)
    
    # Processing the call keyword arguments (line 35)
    kwargs_460249 = {}
    # Getting the type of 'assert_' (line 35)
    assert__460243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 35)
    assert__call_result_460250 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert__460243, *[result_eq_460248], **kwargs_460249)
    
    
    # Call to assert_(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Getting the type of 'loaded_matrix' (line 36)
    loaded_matrix_460252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'loaded_matrix', False)
    # Obtaining the member 'dtype' of a type (line 36)
    dtype_460253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), loaded_matrix_460252, 'dtype')
    # Getting the type of 'dense_matrix' (line 36)
    dense_matrix_460254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'dense_matrix', False)
    # Obtaining the member 'dtype' of a type (line 36)
    dtype_460255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 39), dense_matrix_460254, 'dtype')
    # Applying the binary operator '==' (line 36)
    result_eq_460256 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 16), '==', dtype_460253, dtype_460255)
    
    # Processing the call keyword arguments (line 36)
    kwargs_460257 = {}
    # Getting the type of 'assert_' (line 36)
    assert__460251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 36)
    assert__call_result_460258 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert__460251, *[result_eq_460256], **kwargs_460257)
    
    
    # Call to assert_equal(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to toarray(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_460262 = {}
    # Getting the type of 'loaded_matrix' (line 37)
    loaded_matrix_460260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'loaded_matrix', False)
    # Obtaining the member 'toarray' of a type (line 37)
    toarray_460261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 21), loaded_matrix_460260, 'toarray')
    # Calling toarray(args, kwargs) (line 37)
    toarray_call_result_460263 = invoke(stypy.reporting.localization.Localization(__file__, 37, 21), toarray_460261, *[], **kwargs_460262)
    
    # Getting the type of 'dense_matrix' (line 37)
    dense_matrix_460264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 46), 'dense_matrix', False)
    # Processing the call keyword arguments (line 37)
    kwargs_460265 = {}
    # Getting the type of 'assert_equal' (line 37)
    assert_equal_460259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 37)
    assert_equal_call_result_460266 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assert_equal_460259, *[toarray_call_result_460263, dense_matrix_460264], **kwargs_460265)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_save_and_load(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_save_and_load' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_460267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460267)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_save_and_load'
    return stypy_return_type_460267

# Assigning a type to the variable '_check_save_and_load' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_check_save_and_load', _check_save_and_load)

@norecursion
def test_save_and_load_random(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_save_and_load_random'
    module_type_store = module_type_store.open_function_context('test_save_and_load_random', 39, 0, False)
    
    # Passed parameters checking function
    test_save_and_load_random.stypy_localization = localization
    test_save_and_load_random.stypy_type_of_self = None
    test_save_and_load_random.stypy_type_store = module_type_store
    test_save_and_load_random.stypy_function_name = 'test_save_and_load_random'
    test_save_and_load_random.stypy_param_names_list = []
    test_save_and_load_random.stypy_varargs_param_name = None
    test_save_and_load_random.stypy_kwargs_param_name = None
    test_save_and_load_random.stypy_call_defaults = defaults
    test_save_and_load_random.stypy_call_varargs = varargs
    test_save_and_load_random.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_save_and_load_random', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_save_and_load_random', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_save_and_load_random(...)' code ##################

    
    # Assigning a Num to a Name (line 40):
    
    # Assigning a Num to a Name (line 40):
    int_460268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
    # Assigning a type to the variable 'N' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'N', int_460268)
    
    # Call to seed(...): (line 41)
    # Processing the call arguments (line 41)
    int_460272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_460273 = {}
    # Getting the type of 'np' (line 41)
    np_460269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 41)
    random_460270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), np_460269, 'random')
    # Obtaining the member 'seed' of a type (line 41)
    seed_460271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), random_460270, 'seed')
    # Calling seed(args, kwargs) (line 41)
    seed_call_result_460274 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), seed_460271, *[int_460272], **kwargs_460273)
    
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to random(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_460278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'N' (line 42)
    N_460279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 37), tuple_460278, N_460279)
    # Adding element type (line 42)
    # Getting the type of 'N' (line 42)
    N_460280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 37), tuple_460278, N_460280)
    
    # Processing the call keyword arguments (line 42)
    kwargs_460281 = {}
    # Getting the type of 'np' (line 42)
    np_460275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'np', False)
    # Obtaining the member 'random' of a type (line 42)
    random_460276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 19), np_460275, 'random')
    # Obtaining the member 'random' of a type (line 42)
    random_460277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 19), random_460276, 'random')
    # Calling random(args, kwargs) (line 42)
    random_call_result_460282 = invoke(stypy.reporting.localization.Localization(__file__, 42, 19), random_460277, *[tuple_460278], **kwargs_460281)
    
    # Assigning a type to the variable 'dense_matrix' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'dense_matrix', random_call_result_460282)
    
    # Assigning a Num to a Subscript (line 43):
    
    # Assigning a Num to a Subscript (line 43):
    int_460283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'int')
    # Getting the type of 'dense_matrix' (line 43)
    dense_matrix_460284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'dense_matrix')
    
    # Getting the type of 'dense_matrix' (line 43)
    dense_matrix_460285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'dense_matrix')
    float_460286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 32), 'float')
    # Applying the binary operator '>' (line 43)
    result_gt_460287 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 17), '>', dense_matrix_460285, float_460286)
    
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), dense_matrix_460284, (result_gt_460287, int_460283))
    
    # Call to _check_save_and_load(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'dense_matrix' (line 44)
    dense_matrix_460289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'dense_matrix', False)
    # Processing the call keyword arguments (line 44)
    kwargs_460290 = {}
    # Getting the type of '_check_save_and_load' (line 44)
    _check_save_and_load_460288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), '_check_save_and_load', False)
    # Calling _check_save_and_load(args, kwargs) (line 44)
    _check_save_and_load_call_result_460291 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), _check_save_and_load_460288, *[dense_matrix_460289], **kwargs_460290)
    
    
    # ################# End of 'test_save_and_load_random(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_save_and_load_random' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_460292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460292)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_save_and_load_random'
    return stypy_return_type_460292

# Assigning a type to the variable 'test_save_and_load_random' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'test_save_and_load_random', test_save_and_load_random)

@norecursion
def test_save_and_load_empty(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_save_and_load_empty'
    module_type_store = module_type_store.open_function_context('test_save_and_load_empty', 46, 0, False)
    
    # Passed parameters checking function
    test_save_and_load_empty.stypy_localization = localization
    test_save_and_load_empty.stypy_type_of_self = None
    test_save_and_load_empty.stypy_type_store = module_type_store
    test_save_and_load_empty.stypy_function_name = 'test_save_and_load_empty'
    test_save_and_load_empty.stypy_param_names_list = []
    test_save_and_load_empty.stypy_varargs_param_name = None
    test_save_and_load_empty.stypy_kwargs_param_name = None
    test_save_and_load_empty.stypy_call_defaults = defaults
    test_save_and_load_empty.stypy_call_varargs = varargs
    test_save_and_load_empty.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_save_and_load_empty', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_save_and_load_empty', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_save_and_load_empty(...)' code ##################

    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to zeros(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_460295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    int_460296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 29), tuple_460295, int_460296)
    # Adding element type (line 47)
    int_460297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 29), tuple_460295, int_460297)
    
    # Processing the call keyword arguments (line 47)
    kwargs_460298 = {}
    # Getting the type of 'np' (line 47)
    np_460293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 47)
    zeros_460294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), np_460293, 'zeros')
    # Calling zeros(args, kwargs) (line 47)
    zeros_call_result_460299 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), zeros_460294, *[tuple_460295], **kwargs_460298)
    
    # Assigning a type to the variable 'dense_matrix' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'dense_matrix', zeros_call_result_460299)
    
    # Call to _check_save_and_load(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'dense_matrix' (line 48)
    dense_matrix_460301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'dense_matrix', False)
    # Processing the call keyword arguments (line 48)
    kwargs_460302 = {}
    # Getting the type of '_check_save_and_load' (line 48)
    _check_save_and_load_460300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), '_check_save_and_load', False)
    # Calling _check_save_and_load(args, kwargs) (line 48)
    _check_save_and_load_call_result_460303 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), _check_save_and_load_460300, *[dense_matrix_460301], **kwargs_460302)
    
    
    # ################# End of 'test_save_and_load_empty(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_save_and_load_empty' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_460304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460304)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_save_and_load_empty'
    return stypy_return_type_460304

# Assigning a type to the variable 'test_save_and_load_empty' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'test_save_and_load_empty', test_save_and_load_empty)

@norecursion
def test_save_and_load_one_entry(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_save_and_load_one_entry'
    module_type_store = module_type_store.open_function_context('test_save_and_load_one_entry', 50, 0, False)
    
    # Passed parameters checking function
    test_save_and_load_one_entry.stypy_localization = localization
    test_save_and_load_one_entry.stypy_type_of_self = None
    test_save_and_load_one_entry.stypy_type_store = module_type_store
    test_save_and_load_one_entry.stypy_function_name = 'test_save_and_load_one_entry'
    test_save_and_load_one_entry.stypy_param_names_list = []
    test_save_and_load_one_entry.stypy_varargs_param_name = None
    test_save_and_load_one_entry.stypy_kwargs_param_name = None
    test_save_and_load_one_entry.stypy_call_defaults = defaults
    test_save_and_load_one_entry.stypy_call_varargs = varargs
    test_save_and_load_one_entry.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_save_and_load_one_entry', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_save_and_load_one_entry', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_save_and_load_one_entry(...)' code ##################

    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to zeros(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Obtaining an instance of the builtin type 'tuple' (line 51)
    tuple_460307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 51)
    # Adding element type (line 51)
    int_460308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 29), tuple_460307, int_460308)
    # Adding element type (line 51)
    int_460309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 29), tuple_460307, int_460309)
    
    # Processing the call keyword arguments (line 51)
    kwargs_460310 = {}
    # Getting the type of 'np' (line 51)
    np_460305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 51)
    zeros_460306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 19), np_460305, 'zeros')
    # Calling zeros(args, kwargs) (line 51)
    zeros_call_result_460311 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), zeros_460306, *[tuple_460307], **kwargs_460310)
    
    # Assigning a type to the variable 'dense_matrix' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'dense_matrix', zeros_call_result_460311)
    
    # Assigning a Num to a Subscript (line 52):
    
    # Assigning a Num to a Subscript (line 52):
    int_460312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 24), 'int')
    # Getting the type of 'dense_matrix' (line 52)
    dense_matrix_460313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'dense_matrix')
    
    # Obtaining an instance of the builtin type 'tuple' (line 52)
    tuple_460314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 52)
    # Adding element type (line 52)
    int_460315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), tuple_460314, int_460315)
    # Adding element type (line 52)
    int_460316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 17), tuple_460314, int_460316)
    
    # Storing an element on a container (line 52)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), dense_matrix_460313, (tuple_460314, int_460312))
    
    # Call to _check_save_and_load(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'dense_matrix' (line 53)
    dense_matrix_460318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'dense_matrix', False)
    # Processing the call keyword arguments (line 53)
    kwargs_460319 = {}
    # Getting the type of '_check_save_and_load' (line 53)
    _check_save_and_load_460317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), '_check_save_and_load', False)
    # Calling _check_save_and_load(args, kwargs) (line 53)
    _check_save_and_load_call_result_460320 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), _check_save_and_load_460317, *[dense_matrix_460318], **kwargs_460319)
    
    
    # ################# End of 'test_save_and_load_one_entry(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_save_and_load_one_entry' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_460321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460321)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_save_and_load_one_entry'
    return stypy_return_type_460321

# Assigning a type to the variable 'test_save_and_load_one_entry' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'test_save_and_load_one_entry', test_save_and_load_one_entry)

@norecursion
def test_malicious_load(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_malicious_load'
    module_type_store = module_type_store.open_function_context('test_malicious_load', 56, 0, False)
    
    # Passed parameters checking function
    test_malicious_load.stypy_localization = localization
    test_malicious_load.stypy_type_of_self = None
    test_malicious_load.stypy_type_store = module_type_store
    test_malicious_load.stypy_function_name = 'test_malicious_load'
    test_malicious_load.stypy_param_names_list = []
    test_malicious_load.stypy_varargs_param_name = None
    test_malicious_load.stypy_kwargs_param_name = None
    test_malicious_load.stypy_call_defaults = defaults
    test_malicious_load.stypy_call_varargs = varargs
    test_malicious_load.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_malicious_load', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_malicious_load', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_malicious_load(...)' code ##################

    # Declaration of the 'Executor' class

    class Executor(object, ):

        @norecursion
        def __reduce__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__reduce__'
            module_type_store = module_type_store.open_function_context('__reduce__', 60, 8, False)
            # Assigning a type to the variable 'self' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Executor.__reduce__.__dict__.__setitem__('stypy_localization', localization)
            Executor.__reduce__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Executor.__reduce__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Executor.__reduce__.__dict__.__setitem__('stypy_function_name', 'Executor.__reduce__')
            Executor.__reduce__.__dict__.__setitem__('stypy_param_names_list', [])
            Executor.__reduce__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Executor.__reduce__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Executor.__reduce__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Executor.__reduce__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Executor.__reduce__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Executor.__reduce__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Executor.__reduce__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__reduce__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__reduce__(...)' code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 61)
            tuple_460322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 61)
            # Adding element type (line 61)
            # Getting the type of 'assert_' (line 61)
            assert__460323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'assert_')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 20), tuple_460322, assert__460323)
            # Adding element type (line 61)
            
            # Obtaining an instance of the builtin type 'tuple' (line 61)
            tuple_460324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 61)
            # Adding element type (line 61)
            # Getting the type of 'False' (line 61)
            False_460325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 30), tuple_460324, False_460325)
            # Adding element type (line 61)
            str_460326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'str', 'unexpected code execution')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 30), tuple_460324, str_460326)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 20), tuple_460322, tuple_460324)
            
            # Assigning a type to the variable 'stypy_return_type' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'stypy_return_type', tuple_460322)
            
            # ################# End of '__reduce__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__reduce__' in the type store
            # Getting the type of 'stypy_return_type' (line 60)
            stypy_return_type_460327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_460327)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__reduce__'
            return stypy_return_type_460327


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 59, 4, False)
            # Assigning a type to the variable 'self' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Executor.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Executor' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'Executor', Executor)
    
    # Assigning a Call to a Tuple (line 63):
    
    # Assigning a Subscript to a Name (line 63):
    
    # Obtaining the type of the subscript
    int_460328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'int')
    
    # Call to mkstemp(...): (line 63)
    # Processing the call keyword arguments (line 63)
    str_460331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'str', '.npz')
    keyword_460332 = str_460331
    kwargs_460333 = {'suffix': keyword_460332}
    # Getting the type of 'tempfile' (line 63)
    tempfile_460329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 63)
    mkstemp_460330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 18), tempfile_460329, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 63)
    mkstemp_call_result_460334 = invoke(stypy.reporting.localization.Localization(__file__, 63, 18), mkstemp_460330, *[], **kwargs_460333)
    
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___460335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), mkstemp_call_result_460334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_460336 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), getitem___460335, int_460328)
    
    # Assigning a type to the variable 'tuple_var_assignment_460152' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'tuple_var_assignment_460152', subscript_call_result_460336)
    
    # Assigning a Subscript to a Name (line 63):
    
    # Obtaining the type of the subscript
    int_460337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'int')
    
    # Call to mkstemp(...): (line 63)
    # Processing the call keyword arguments (line 63)
    str_460340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'str', '.npz')
    keyword_460341 = str_460340
    kwargs_460342 = {'suffix': keyword_460341}
    # Getting the type of 'tempfile' (line 63)
    tempfile_460338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'tempfile', False)
    # Obtaining the member 'mkstemp' of a type (line 63)
    mkstemp_460339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 18), tempfile_460338, 'mkstemp')
    # Calling mkstemp(args, kwargs) (line 63)
    mkstemp_call_result_460343 = invoke(stypy.reporting.localization.Localization(__file__, 63, 18), mkstemp_460339, *[], **kwargs_460342)
    
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___460344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), mkstemp_call_result_460343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_460345 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), getitem___460344, int_460337)
    
    # Assigning a type to the variable 'tuple_var_assignment_460153' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'tuple_var_assignment_460153', subscript_call_result_460345)
    
    # Assigning a Name to a Name (line 63):
    # Getting the type of 'tuple_var_assignment_460152' (line 63)
    tuple_var_assignment_460152_460346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'tuple_var_assignment_460152')
    # Assigning a type to the variable 'fd' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'fd', tuple_var_assignment_460152_460346)
    
    # Assigning a Name to a Name (line 63):
    # Getting the type of 'tuple_var_assignment_460153' (line 63)
    tuple_var_assignment_460153_460347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'tuple_var_assignment_460153')
    # Assigning a type to the variable 'tmpfile' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tmpfile', tuple_var_assignment_460153_460347)
    
    # Call to close(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'fd' (line 64)
    fd_460350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'fd', False)
    # Processing the call keyword arguments (line 64)
    kwargs_460351 = {}
    # Getting the type of 'os' (line 64)
    os_460348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'os', False)
    # Obtaining the member 'close' of a type (line 64)
    close_460349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), os_460348, 'close')
    # Calling close(args, kwargs) (line 64)
    close_call_result_460352 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), close_460349, *[fd_460350], **kwargs_460351)
    
    
    # Try-finally block (line 65)
    
    # Call to savez(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'tmpfile' (line 66)
    tmpfile_460355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'tmpfile', False)
    # Processing the call keyword arguments (line 66)
    
    # Call to Executor(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_460357 = {}
    # Getting the type of 'Executor' (line 66)
    Executor_460356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'Executor', False)
    # Calling Executor(args, kwargs) (line 66)
    Executor_call_result_460358 = invoke(stypy.reporting.localization.Localization(__file__, 66, 33), Executor_460356, *[], **kwargs_460357)
    
    keyword_460359 = Executor_call_result_460358
    kwargs_460360 = {'format': keyword_460359}
    # Getting the type of 'np' (line 66)
    np_460353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'np', False)
    # Obtaining the member 'savez' of a type (line 66)
    savez_460354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), np_460353, 'savez')
    # Calling savez(args, kwargs) (line 66)
    savez_call_result_460361 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), savez_460354, *[tmpfile_460355], **kwargs_460360)
    
    
    # Call to assert_raises(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'ValueError' (line 69)
    ValueError_460363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'ValueError', False)
    # Getting the type of 'load_npz' (line 69)
    load_npz_460364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'load_npz', False)
    # Getting the type of 'tmpfile' (line 69)
    tmpfile_460365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 44), 'tmpfile', False)
    # Processing the call keyword arguments (line 69)
    kwargs_460366 = {}
    # Getting the type of 'assert_raises' (line 69)
    assert_raises_460362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 69)
    assert_raises_call_result_460367 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assert_raises_460362, *[ValueError_460363, load_npz_460364, tmpfile_460365], **kwargs_460366)
    
    
    # finally branch of the try-finally block (line 65)
    
    # Call to remove(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'tmpfile' (line 71)
    tmpfile_460370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'tmpfile', False)
    # Processing the call keyword arguments (line 71)
    kwargs_460371 = {}
    # Getting the type of 'os' (line 71)
    os_460368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'os', False)
    # Obtaining the member 'remove' of a type (line 71)
    remove_460369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), os_460368, 'remove')
    # Calling remove(args, kwargs) (line 71)
    remove_call_result_460372 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), remove_460369, *[tmpfile_460370], **kwargs_460371)
    
    
    
    # ################# End of 'test_malicious_load(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_malicious_load' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_460373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460373)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_malicious_load'
    return stypy_return_type_460373

# Assigning a type to the variable 'test_malicious_load' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'test_malicious_load', test_malicious_load)

@norecursion
def test_py23_compatibility(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_py23_compatibility'
    module_type_store = module_type_store.open_function_context('test_py23_compatibility', 74, 0, False)
    
    # Passed parameters checking function
    test_py23_compatibility.stypy_localization = localization
    test_py23_compatibility.stypy_type_of_self = None
    test_py23_compatibility.stypy_type_store = module_type_store
    test_py23_compatibility.stypy_function_name = 'test_py23_compatibility'
    test_py23_compatibility.stypy_param_names_list = []
    test_py23_compatibility.stypy_varargs_param_name = None
    test_py23_compatibility.stypy_kwargs_param_name = None
    test_py23_compatibility.stypy_call_defaults = defaults
    test_py23_compatibility.stypy_call_varargs = varargs
    test_py23_compatibility.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_py23_compatibility', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_py23_compatibility', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_py23_compatibility(...)' code ##################

    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to load_npz(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Call to join(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'DATA_DIR' (line 79)
    DATA_DIR_460378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'DATA_DIR', False)
    str_460379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 40), 'str', 'csc_py2.npz')
    # Processing the call keyword arguments (line 79)
    kwargs_460380 = {}
    # Getting the type of 'os' (line 79)
    os_460375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 79)
    path_460376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), os_460375, 'path')
    # Obtaining the member 'join' of a type (line 79)
    join_460377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), path_460376, 'join')
    # Calling join(args, kwargs) (line 79)
    join_call_result_460381 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), join_460377, *[DATA_DIR_460378, str_460379], **kwargs_460380)
    
    # Processing the call keyword arguments (line 79)
    kwargs_460382 = {}
    # Getting the type of 'load_npz' (line 79)
    load_npz_460374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'load_npz', False)
    # Calling load_npz(args, kwargs) (line 79)
    load_npz_call_result_460383 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), load_npz_460374, *[join_call_result_460381], **kwargs_460382)
    
    # Assigning a type to the variable 'a' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'a', load_npz_call_result_460383)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to load_npz(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Call to join(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'DATA_DIR' (line 80)
    DATA_DIR_460388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'DATA_DIR', False)
    str_460389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 40), 'str', 'csc_py3.npz')
    # Processing the call keyword arguments (line 80)
    kwargs_460390 = {}
    # Getting the type of 'os' (line 80)
    os_460385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 80)
    path_460386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), os_460385, 'path')
    # Obtaining the member 'join' of a type (line 80)
    join_460387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), path_460386, 'join')
    # Calling join(args, kwargs) (line 80)
    join_call_result_460391 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), join_460387, *[DATA_DIR_460388, str_460389], **kwargs_460390)
    
    # Processing the call keyword arguments (line 80)
    kwargs_460392 = {}
    # Getting the type of 'load_npz' (line 80)
    load_npz_460384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'load_npz', False)
    # Calling load_npz(args, kwargs) (line 80)
    load_npz_call_result_460393 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), load_npz_460384, *[join_call_result_460391], **kwargs_460392)
    
    # Assigning a type to the variable 'b' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'b', load_npz_call_result_460393)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to csc_matrix(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_460395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_460396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    int_460397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), list_460396, int_460397)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 19), list_460395, list_460396)
    
    # Processing the call keyword arguments (line 81)
    kwargs_460398 = {}
    # Getting the type of 'csc_matrix' (line 81)
    csc_matrix_460394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 81)
    csc_matrix_call_result_460399 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), csc_matrix_460394, *[list_460395], **kwargs_460398)
    
    # Assigning a type to the variable 'c' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'c', csc_matrix_call_result_460399)
    
    # Call to assert_equal(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Call to toarray(...): (line 83)
    # Processing the call keyword arguments (line 83)
    kwargs_460403 = {}
    # Getting the type of 'a' (line 83)
    a_460401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'a', False)
    # Obtaining the member 'toarray' of a type (line 83)
    toarray_460402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), a_460401, 'toarray')
    # Calling toarray(args, kwargs) (line 83)
    toarray_call_result_460404 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), toarray_460402, *[], **kwargs_460403)
    
    
    # Call to toarray(...): (line 83)
    # Processing the call keyword arguments (line 83)
    kwargs_460407 = {}
    # Getting the type of 'c' (line 83)
    c_460405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'c', False)
    # Obtaining the member 'toarray' of a type (line 83)
    toarray_460406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), c_460405, 'toarray')
    # Calling toarray(args, kwargs) (line 83)
    toarray_call_result_460408 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), toarray_460406, *[], **kwargs_460407)
    
    # Processing the call keyword arguments (line 83)
    kwargs_460409 = {}
    # Getting the type of 'assert_equal' (line 83)
    assert_equal_460400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 83)
    assert_equal_call_result_460410 = invoke(stypy.reporting.localization.Localization(__file__, 83, 4), assert_equal_460400, *[toarray_call_result_460404, toarray_call_result_460408], **kwargs_460409)
    
    
    # Call to assert_equal(...): (line 84)
    # Processing the call arguments (line 84)
    
    # Call to toarray(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_460414 = {}
    # Getting the type of 'b' (line 84)
    b_460412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'b', False)
    # Obtaining the member 'toarray' of a type (line 84)
    toarray_460413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 17), b_460412, 'toarray')
    # Calling toarray(args, kwargs) (line 84)
    toarray_call_result_460415 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), toarray_460413, *[], **kwargs_460414)
    
    
    # Call to toarray(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_460418 = {}
    # Getting the type of 'c' (line 84)
    c_460416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'c', False)
    # Obtaining the member 'toarray' of a type (line 84)
    toarray_460417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), c_460416, 'toarray')
    # Calling toarray(args, kwargs) (line 84)
    toarray_call_result_460419 = invoke(stypy.reporting.localization.Localization(__file__, 84, 30), toarray_460417, *[], **kwargs_460418)
    
    # Processing the call keyword arguments (line 84)
    kwargs_460420 = {}
    # Getting the type of 'assert_equal' (line 84)
    assert_equal_460411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 84)
    assert_equal_call_result_460421 = invoke(stypy.reporting.localization.Localization(__file__, 84, 4), assert_equal_460411, *[toarray_call_result_460415, toarray_call_result_460419], **kwargs_460420)
    
    
    # ################# End of 'test_py23_compatibility(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_py23_compatibility' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_460422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460422)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_py23_compatibility'
    return stypy_return_type_460422

# Assigning a type to the variable 'test_py23_compatibility' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_py23_compatibility', test_py23_compatibility)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
