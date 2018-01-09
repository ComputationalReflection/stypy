
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import sys
4: import numpy as np
5: import scipy.sparse
6: 
7: from scipy._lib._version import NumpyVersion
8: 
9: __all__ = ['save_npz', 'load_npz']
10: 
11: 
12: if NumpyVersion(np.__version__) >= '1.10.0':
13:     # Make loading safe vs. malicious input
14:     PICKLE_KWARGS = dict(allow_pickle=False)
15: else:
16:     PICKLE_KWARGS = dict()
17: 
18: 
19: def save_npz(file, matrix, compressed=True):
20:     ''' Save a sparse matrix to a file using ``.npz`` format.
21: 
22:     Parameters
23:     ----------
24:     file : str or file-like object
25:         Either the file name (string) or an open file (file-like object)
26:         where the data will be saved. If file is a string, the ``.npz``
27:         extension will be appended to the file name if it is not already
28:         there.
29:     matrix: spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)
30:         The sparse matrix to save.
31:     compressed : bool, optional
32:         Allow compressing the file. Default: True
33: 
34:     See Also
35:     --------
36:     scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.
37:     numpy.savez: Save several arrays into a ``.npz`` archive.
38:     numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.
39: 
40:     Examples
41:     --------
42:     Store sparse matrix to disk, and load it again:
43: 
44:     >>> import scipy.sparse
45:     >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
46:     >>> sparse_matrix
47:     <2x3 sparse matrix of type '<class 'numpy.int64'>'
48:        with 2 stored elements in Compressed Sparse Column format>
49:     >>> sparse_matrix.todense()
50:     matrix([[0, 0, 3],
51:             [4, 0, 0]], dtype=int64)
52: 
53:     >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
54:     >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
55: 
56:     >>> sparse_matrix
57:     <2x3 sparse matrix of type '<class 'numpy.int64'>'
58:        with 2 stored elements in Compressed Sparse Column format>
59:     >>> sparse_matrix.todense()
60:     matrix([[0, 0, 3],
61:             [4, 0, 0]], dtype=int64)
62:     '''
63: 
64:     arrays_dict = dict(format=matrix.format.encode('ascii'),
65:                        shape=matrix.shape,
66:                        data=matrix.data)
67:     if matrix.format in ('csc', 'csr', 'bsr'):
68:         arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
69:     elif matrix.format == 'dia':
70:         arrays_dict.update(offsets=matrix.offsets)
71:     elif matrix.format == 'coo':
72:         arrays_dict.update(row=matrix.row, col=matrix.col)
73:     else:
74:         raise NotImplementedError('Save is not implemented for sparse matrix of format {}.'.format(matrix.format))
75: 
76:     if compressed:
77:         np.savez_compressed(file, **arrays_dict)
78:     else:
79:         np.savez(file, **arrays_dict)
80: 
81: 
82: def load_npz(file):
83:     ''' Load a sparse matrix from a file using ``.npz`` format.
84: 
85:     Parameters
86:     ----------
87:     file : str or file-like object
88:         Either the file name (string) or an open file (file-like object)
89:         where the data will be loaded.
90: 
91:     Returns
92:     -------
93:     result : csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix
94:         A sparse matrix containing the loaded data.
95: 
96:     Raises
97:     ------
98:     IOError
99:         If the input file does not exist or cannot be read.
100: 
101:     See Also
102:     --------
103:     scipy.sparse.save_npz: Save a sparse matrix to a file using ``.npz`` format.
104:     numpy.load: Load several arrays from a ``.npz`` archive.
105: 
106:     Examples
107:     --------
108:     Store sparse matrix to disk, and load it again:
109: 
110:     >>> import scipy.sparse
111:     >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
112:     >>> sparse_matrix
113:     <2x3 sparse matrix of type '<class 'numpy.int64'>'
114:        with 2 stored elements in Compressed Sparse Column format>
115:     >>> sparse_matrix.todense()
116:     matrix([[0, 0, 3],
117:             [4, 0, 0]], dtype=int64)
118: 
119:     >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
120:     >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
121: 
122:     >>> sparse_matrix
123:     <2x3 sparse matrix of type '<class 'numpy.int64'>'
124:         with 2 stored elements in Compressed Sparse Column format>
125:     >>> sparse_matrix.todense()
126:     matrix([[0, 0, 3],
127:             [4, 0, 0]], dtype=int64)
128:     '''
129: 
130:     with np.load(file, **PICKLE_KWARGS) as loaded:
131:         try:
132:             matrix_format = loaded['format']
133:         except KeyError:
134:             raise ValueError('The file {} does not contain a sparse matrix.'.format(file))
135: 
136:         matrix_format = matrix_format.item()
137: 
138:         if sys.version_info[0] >= 3 and not isinstance(matrix_format, str):
139:             # Play safe with Python 2 vs 3 backward compatibility;
140:             # files saved with Scipy < 1.0.0 may contain unicode or bytes.
141:             matrix_format = matrix_format.decode('ascii')
142: 
143:         try:
144:             cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
145:         except AttributeError:
146:             raise ValueError('Unknown matrix format "{}"'.format(matrix_format))
147: 
148:         if matrix_format in ('csc', 'csr', 'bsr'):
149:             return cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
150:         elif matrix_format == 'dia':
151:             return cls((loaded['data'], loaded['offsets']), shape=loaded['shape'])
152:         elif matrix_format == 'coo':
153:             return cls((loaded['data'], (loaded['row'], loaded['col'])), shape=loaded['shape'])
154:         else:
155:             raise NotImplementedError('Load is not implemented for '
156:                                       'sparse matrix of format {}.'.format(matrix_format))
157: 

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

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_380806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_380806) is not StypyTypeError):

    if (import_380806 != 'pyd_module'):
        __import__(import_380806)
        sys_modules_380807 = sys.modules[import_380806]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_380807.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_380806)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.sparse' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_380808 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse')

if (type(import_380808) is not StypyTypeError):

    if (import_380808 != 'pyd_module'):
        __import__(import_380808)
        sys_modules_380809 = sys.modules[import_380808]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', sys_modules_380809.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', import_380808)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._version import NumpyVersion' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_380810 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._version')

if (type(import_380810) is not StypyTypeError):

    if (import_380810 != 'pyd_module'):
        __import__(import_380810)
        sys_modules_380811 = sys.modules[import_380810]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._version', sys_modules_380811.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_380811, sys_modules_380811.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._version', import_380810)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# Assigning a List to a Name (line 9):
__all__ = ['save_npz', 'load_npz']
module_type_store.set_exportable_members(['save_npz', 'load_npz'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_380812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_380813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'save_npz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_380812, str_380813)
# Adding element type (line 9)
str_380814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'str', 'load_npz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_380812, str_380814)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_380812)



# Call to NumpyVersion(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'np' (line 12)
np_380816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'np', False)
# Obtaining the member '__version__' of a type (line 12)
version___380817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 16), np_380816, '__version__')
# Processing the call keyword arguments (line 12)
kwargs_380818 = {}
# Getting the type of 'NumpyVersion' (line 12)
NumpyVersion_380815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 3), 'NumpyVersion', False)
# Calling NumpyVersion(args, kwargs) (line 12)
NumpyVersion_call_result_380819 = invoke(stypy.reporting.localization.Localization(__file__, 12, 3), NumpyVersion_380815, *[version___380817], **kwargs_380818)

str_380820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'str', '1.10.0')
# Applying the binary operator '>=' (line 12)
result_ge_380821 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 3), '>=', NumpyVersion_call_result_380819, str_380820)

# Testing the type of an if condition (line 12)
if_condition_380822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 0), result_ge_380821)
# Assigning a type to the variable 'if_condition_380822' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'if_condition_380822', if_condition_380822)
# SSA begins for if statement (line 12)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 14):

# Call to dict(...): (line 14)
# Processing the call keyword arguments (line 14)
# Getting the type of 'False' (line 14)
False_380824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'False', False)
keyword_380825 = False_380824
kwargs_380826 = {'allow_pickle': keyword_380825}
# Getting the type of 'dict' (line 14)
dict_380823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'dict', False)
# Calling dict(args, kwargs) (line 14)
dict_call_result_380827 = invoke(stypy.reporting.localization.Localization(__file__, 14, 20), dict_380823, *[], **kwargs_380826)

# Assigning a type to the variable 'PICKLE_KWARGS' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'PICKLE_KWARGS', dict_call_result_380827)
# SSA branch for the else part of an if statement (line 12)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 16):

# Call to dict(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_380829 = {}
# Getting the type of 'dict' (line 16)
dict_380828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'dict', False)
# Calling dict(args, kwargs) (line 16)
dict_call_result_380830 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), dict_380828, *[], **kwargs_380829)

# Assigning a type to the variable 'PICKLE_KWARGS' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'PICKLE_KWARGS', dict_call_result_380830)
# SSA join for if statement (line 12)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def save_npz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 19)
    True_380831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'True')
    defaults = [True_380831]
    # Create a new context for function 'save_npz'
    module_type_store = module_type_store.open_function_context('save_npz', 19, 0, False)
    
    # Passed parameters checking function
    save_npz.stypy_localization = localization
    save_npz.stypy_type_of_self = None
    save_npz.stypy_type_store = module_type_store
    save_npz.stypy_function_name = 'save_npz'
    save_npz.stypy_param_names_list = ['file', 'matrix', 'compressed']
    save_npz.stypy_varargs_param_name = None
    save_npz.stypy_kwargs_param_name = None
    save_npz.stypy_call_defaults = defaults
    save_npz.stypy_call_varargs = varargs
    save_npz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'save_npz', ['file', 'matrix', 'compressed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'save_npz', localization, ['file', 'matrix', 'compressed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'save_npz(...)' code ##################

    str_380832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', " Save a sparse matrix to a file using ``.npz`` format.\n\n    Parameters\n    ----------\n    file : str or file-like object\n        Either the file name (string) or an open file (file-like object)\n        where the data will be saved. If file is a string, the ``.npz``\n        extension will be appended to the file name if it is not already\n        there.\n    matrix: spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)\n        The sparse matrix to save.\n    compressed : bool, optional\n        Allow compressing the file. Default: True\n\n    See Also\n    --------\n    scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.\n    numpy.savez: Save several arrays into a ``.npz`` archive.\n    numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.\n\n    Examples\n    --------\n    Store sparse matrix to disk, and load it again:\n\n    >>> import scipy.sparse\n    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n       with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.todense()\n    matrix([[0, 0, 3],\n            [4, 0, 0]], dtype=int64)\n\n    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)\n    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')\n\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n       with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.todense()\n    matrix([[0, 0, 3],\n            [4, 0, 0]], dtype=int64)\n    ")
    
    # Assigning a Call to a Name (line 64):
    
    # Call to dict(...): (line 64)
    # Processing the call keyword arguments (line 64)
    
    # Call to encode(...): (line 64)
    # Processing the call arguments (line 64)
    str_380837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 51), 'str', 'ascii')
    # Processing the call keyword arguments (line 64)
    kwargs_380838 = {}
    # Getting the type of 'matrix' (line 64)
    matrix_380834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'matrix', False)
    # Obtaining the member 'format' of a type (line 64)
    format_380835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), matrix_380834, 'format')
    # Obtaining the member 'encode' of a type (line 64)
    encode_380836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), format_380835, 'encode')
    # Calling encode(args, kwargs) (line 64)
    encode_call_result_380839 = invoke(stypy.reporting.localization.Localization(__file__, 64, 30), encode_380836, *[str_380837], **kwargs_380838)
    
    keyword_380840 = encode_call_result_380839
    # Getting the type of 'matrix' (line 65)
    matrix_380841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'matrix', False)
    # Obtaining the member 'shape' of a type (line 65)
    shape_380842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 29), matrix_380841, 'shape')
    keyword_380843 = shape_380842
    # Getting the type of 'matrix' (line 66)
    matrix_380844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'matrix', False)
    # Obtaining the member 'data' of a type (line 66)
    data_380845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 28), matrix_380844, 'data')
    keyword_380846 = data_380845
    kwargs_380847 = {'shape': keyword_380843, 'data': keyword_380846, 'format': keyword_380840}
    # Getting the type of 'dict' (line 64)
    dict_380833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'dict', False)
    # Calling dict(args, kwargs) (line 64)
    dict_call_result_380848 = invoke(stypy.reporting.localization.Localization(__file__, 64, 18), dict_380833, *[], **kwargs_380847)
    
    # Assigning a type to the variable 'arrays_dict' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'arrays_dict', dict_call_result_380848)
    
    
    # Getting the type of 'matrix' (line 67)
    matrix_380849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), 'matrix')
    # Obtaining the member 'format' of a type (line 67)
    format_380850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 7), matrix_380849, 'format')
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_380851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    str_380852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'str', 'csc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), tuple_380851, str_380852)
    # Adding element type (line 67)
    str_380853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 32), 'str', 'csr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), tuple_380851, str_380853)
    # Adding element type (line 67)
    str_380854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 39), 'str', 'bsr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), tuple_380851, str_380854)
    
    # Applying the binary operator 'in' (line 67)
    result_contains_380855 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 7), 'in', format_380850, tuple_380851)
    
    # Testing the type of an if condition (line 67)
    if_condition_380856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), result_contains_380855)
    # Assigning a type to the variable 'if_condition_380856' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_380856', if_condition_380856)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to update(...): (line 68)
    # Processing the call keyword arguments (line 68)
    # Getting the type of 'matrix' (line 68)
    matrix_380859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'matrix', False)
    # Obtaining the member 'indices' of a type (line 68)
    indices_380860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 35), matrix_380859, 'indices')
    keyword_380861 = indices_380860
    # Getting the type of 'matrix' (line 68)
    matrix_380862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 58), 'matrix', False)
    # Obtaining the member 'indptr' of a type (line 68)
    indptr_380863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 58), matrix_380862, 'indptr')
    keyword_380864 = indptr_380863
    kwargs_380865 = {'indices': keyword_380861, 'indptr': keyword_380864}
    # Getting the type of 'arrays_dict' (line 68)
    arrays_dict_380857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'arrays_dict', False)
    # Obtaining the member 'update' of a type (line 68)
    update_380858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), arrays_dict_380857, 'update')
    # Calling update(args, kwargs) (line 68)
    update_call_result_380866 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), update_380858, *[], **kwargs_380865)
    
    # SSA branch for the else part of an if statement (line 67)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'matrix' (line 69)
    matrix_380867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'matrix')
    # Obtaining the member 'format' of a type (line 69)
    format_380868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 9), matrix_380867, 'format')
    str_380869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'str', 'dia')
    # Applying the binary operator '==' (line 69)
    result_eq_380870 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 9), '==', format_380868, str_380869)
    
    # Testing the type of an if condition (line 69)
    if_condition_380871 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 9), result_eq_380870)
    # Assigning a type to the variable 'if_condition_380871' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'if_condition_380871', if_condition_380871)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to update(...): (line 70)
    # Processing the call keyword arguments (line 70)
    # Getting the type of 'matrix' (line 70)
    matrix_380874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'matrix', False)
    # Obtaining the member 'offsets' of a type (line 70)
    offsets_380875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 35), matrix_380874, 'offsets')
    keyword_380876 = offsets_380875
    kwargs_380877 = {'offsets': keyword_380876}
    # Getting the type of 'arrays_dict' (line 70)
    arrays_dict_380872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'arrays_dict', False)
    # Obtaining the member 'update' of a type (line 70)
    update_380873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), arrays_dict_380872, 'update')
    # Calling update(args, kwargs) (line 70)
    update_call_result_380878 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), update_380873, *[], **kwargs_380877)
    
    # SSA branch for the else part of an if statement (line 69)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'matrix' (line 71)
    matrix_380879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 9), 'matrix')
    # Obtaining the member 'format' of a type (line 71)
    format_380880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 9), matrix_380879, 'format')
    str_380881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'str', 'coo')
    # Applying the binary operator '==' (line 71)
    result_eq_380882 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), '==', format_380880, str_380881)
    
    # Testing the type of an if condition (line 71)
    if_condition_380883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 9), result_eq_380882)
    # Assigning a type to the variable 'if_condition_380883' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 9), 'if_condition_380883', if_condition_380883)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to update(...): (line 72)
    # Processing the call keyword arguments (line 72)
    # Getting the type of 'matrix' (line 72)
    matrix_380886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'matrix', False)
    # Obtaining the member 'row' of a type (line 72)
    row_380887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 31), matrix_380886, 'row')
    keyword_380888 = row_380887
    # Getting the type of 'matrix' (line 72)
    matrix_380889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'matrix', False)
    # Obtaining the member 'col' of a type (line 72)
    col_380890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 47), matrix_380889, 'col')
    keyword_380891 = col_380890
    kwargs_380892 = {'col': keyword_380891, 'row': keyword_380888}
    # Getting the type of 'arrays_dict' (line 72)
    arrays_dict_380884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'arrays_dict', False)
    # Obtaining the member 'update' of a type (line 72)
    update_380885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), arrays_dict_380884, 'update')
    # Calling update(args, kwargs) (line 72)
    update_call_result_380893 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), update_380885, *[], **kwargs_380892)
    
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Call to NotImplementedError(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Call to format(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'matrix' (line 74)
    matrix_380897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 99), 'matrix', False)
    # Obtaining the member 'format' of a type (line 74)
    format_380898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 99), matrix_380897, 'format')
    # Processing the call keyword arguments (line 74)
    kwargs_380899 = {}
    str_380895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 34), 'str', 'Save is not implemented for sparse matrix of format {}.')
    # Obtaining the member 'format' of a type (line 74)
    format_380896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 34), str_380895, 'format')
    # Calling format(args, kwargs) (line 74)
    format_call_result_380900 = invoke(stypy.reporting.localization.Localization(__file__, 74, 34), format_380896, *[format_380898], **kwargs_380899)
    
    # Processing the call keyword arguments (line 74)
    kwargs_380901 = {}
    # Getting the type of 'NotImplementedError' (line 74)
    NotImplementedError_380894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 74)
    NotImplementedError_call_result_380902 = invoke(stypy.reporting.localization.Localization(__file__, 74, 14), NotImplementedError_380894, *[format_call_result_380900], **kwargs_380901)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 74, 8), NotImplementedError_call_result_380902, 'raise parameter', BaseException)
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'compressed' (line 76)
    compressed_380903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'compressed')
    # Testing the type of an if condition (line 76)
    if_condition_380904 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), compressed_380903)
    # Assigning a type to the variable 'if_condition_380904' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_380904', if_condition_380904)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to savez_compressed(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'file' (line 77)
    file_380907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'file', False)
    # Processing the call keyword arguments (line 77)
    # Getting the type of 'arrays_dict' (line 77)
    arrays_dict_380908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'arrays_dict', False)
    kwargs_380909 = {'arrays_dict_380908': arrays_dict_380908}
    # Getting the type of 'np' (line 77)
    np_380905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'np', False)
    # Obtaining the member 'savez_compressed' of a type (line 77)
    savez_compressed_380906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), np_380905, 'savez_compressed')
    # Calling savez_compressed(args, kwargs) (line 77)
    savez_compressed_call_result_380910 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), savez_compressed_380906, *[file_380907], **kwargs_380909)
    
    # SSA branch for the else part of an if statement (line 76)
    module_type_store.open_ssa_branch('else')
    
    # Call to savez(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'file' (line 79)
    file_380913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'file', False)
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'arrays_dict' (line 79)
    arrays_dict_380914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'arrays_dict', False)
    kwargs_380915 = {'arrays_dict_380914': arrays_dict_380914}
    # Getting the type of 'np' (line 79)
    np_380911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'np', False)
    # Obtaining the member 'savez' of a type (line 79)
    savez_380912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), np_380911, 'savez')
    # Calling savez(args, kwargs) (line 79)
    savez_call_result_380916 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), savez_380912, *[file_380913], **kwargs_380915)
    
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'save_npz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save_npz' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_380917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380917)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save_npz'
    return stypy_return_type_380917

# Assigning a type to the variable 'save_npz' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'save_npz', save_npz)

@norecursion
def load_npz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'load_npz'
    module_type_store = module_type_store.open_function_context('load_npz', 82, 0, False)
    
    # Passed parameters checking function
    load_npz.stypy_localization = localization
    load_npz.stypy_type_of_self = None
    load_npz.stypy_type_store = module_type_store
    load_npz.stypy_function_name = 'load_npz'
    load_npz.stypy_param_names_list = ['file']
    load_npz.stypy_varargs_param_name = None
    load_npz.stypy_kwargs_param_name = None
    load_npz.stypy_call_defaults = defaults
    load_npz.stypy_call_varargs = varargs
    load_npz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'load_npz', ['file'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'load_npz', localization, ['file'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'load_npz(...)' code ##################

    str_380918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', " Load a sparse matrix from a file using ``.npz`` format.\n\n    Parameters\n    ----------\n    file : str or file-like object\n        Either the file name (string) or an open file (file-like object)\n        where the data will be loaded.\n\n    Returns\n    -------\n    result : csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix\n        A sparse matrix containing the loaded data.\n\n    Raises\n    ------\n    IOError\n        If the input file does not exist or cannot be read.\n\n    See Also\n    --------\n    scipy.sparse.save_npz: Save a sparse matrix to a file using ``.npz`` format.\n    numpy.load: Load several arrays from a ``.npz`` archive.\n\n    Examples\n    --------\n    Store sparse matrix to disk, and load it again:\n\n    >>> import scipy.sparse\n    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n       with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.todense()\n    matrix([[0, 0, 3],\n            [4, 0, 0]], dtype=int64)\n\n    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)\n    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')\n\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n        with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.todense()\n    matrix([[0, 0, 3],\n            [4, 0, 0]], dtype=int64)\n    ")
    
    # Call to load(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'file' (line 130)
    file_380921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 17), 'file', False)
    # Processing the call keyword arguments (line 130)
    # Getting the type of 'PICKLE_KWARGS' (line 130)
    PICKLE_KWARGS_380922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'PICKLE_KWARGS', False)
    kwargs_380923 = {'PICKLE_KWARGS_380922': PICKLE_KWARGS_380922}
    # Getting the type of 'np' (line 130)
    np_380919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'np', False)
    # Obtaining the member 'load' of a type (line 130)
    load_380920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), np_380919, 'load')
    # Calling load(args, kwargs) (line 130)
    load_call_result_380924 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), load_380920, *[file_380921], **kwargs_380923)
    
    with_380925 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 130, 9), load_call_result_380924, 'with parameter', '__enter__', '__exit__')

    if with_380925:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 130)
        enter___380926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), load_call_result_380924, '__enter__')
        with_enter_380927 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), enter___380926)
        # Assigning a type to the variable 'loaded' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'loaded', with_enter_380927)
        
        
        # SSA begins for try-except statement (line 131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        str_380928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'str', 'format')
        # Getting the type of 'loaded' (line 132)
        loaded_380929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'loaded')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___380930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), loaded_380929, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_380931 = invoke(stypy.reporting.localization.Localization(__file__, 132, 28), getitem___380930, str_380928)
        
        # Assigning a type to the variable 'matrix_format' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'matrix_format', subscript_call_result_380931)
        # SSA branch for the except part of a try statement (line 131)
        # SSA branch for the except 'KeyError' branch of a try statement (line 131)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Call to format(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'file' (line 134)
        file_380935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 84), 'file', False)
        # Processing the call keyword arguments (line 134)
        kwargs_380936 = {}
        str_380933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'str', 'The file {} does not contain a sparse matrix.')
        # Obtaining the member 'format' of a type (line 134)
        format_380934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 29), str_380933, 'format')
        # Calling format(args, kwargs) (line 134)
        format_call_result_380937 = invoke(stypy.reporting.localization.Localization(__file__, 134, 29), format_380934, *[file_380935], **kwargs_380936)
        
        # Processing the call keyword arguments (line 134)
        kwargs_380938 = {}
        # Getting the type of 'ValueError' (line 134)
        ValueError_380932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 134)
        ValueError_call_result_380939 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), ValueError_380932, *[format_call_result_380937], **kwargs_380938)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 134, 12), ValueError_call_result_380939, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 131)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 136):
        
        # Call to item(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_380942 = {}
        # Getting the type of 'matrix_format' (line 136)
        matrix_format_380940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'matrix_format', False)
        # Obtaining the member 'item' of a type (line 136)
        item_380941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 24), matrix_format_380940, 'item')
        # Calling item(args, kwargs) (line 136)
        item_call_result_380943 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), item_380941, *[], **kwargs_380942)
        
        # Assigning a type to the variable 'matrix_format' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'matrix_format', item_call_result_380943)
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_380944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 28), 'int')
        # Getting the type of 'sys' (line 138)
        sys_380945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 138)
        version_info_380946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), sys_380945, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___380947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 11), version_info_380946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_380948 = invoke(stypy.reporting.localization.Localization(__file__, 138, 11), getitem___380947, int_380944)
        
        int_380949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 34), 'int')
        # Applying the binary operator '>=' (line 138)
        result_ge_380950 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), '>=', subscript_call_result_380948, int_380949)
        
        
        
        # Call to isinstance(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'matrix_format' (line 138)
        matrix_format_380952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 55), 'matrix_format', False)
        # Getting the type of 'str' (line 138)
        str_380953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 70), 'str', False)
        # Processing the call keyword arguments (line 138)
        kwargs_380954 = {}
        # Getting the type of 'isinstance' (line 138)
        isinstance_380951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 44), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 138)
        isinstance_call_result_380955 = invoke(stypy.reporting.localization.Localization(__file__, 138, 44), isinstance_380951, *[matrix_format_380952, str_380953], **kwargs_380954)
        
        # Applying the 'not' unary operator (line 138)
        result_not__380956 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 40), 'not', isinstance_call_result_380955)
        
        # Applying the binary operator 'and' (line 138)
        result_and_keyword_380957 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), 'and', result_ge_380950, result_not__380956)
        
        # Testing the type of an if condition (line 138)
        if_condition_380958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), result_and_keyword_380957)
        # Assigning a type to the variable 'if_condition_380958' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_380958', if_condition_380958)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 141):
        
        # Call to decode(...): (line 141)
        # Processing the call arguments (line 141)
        str_380961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 49), 'str', 'ascii')
        # Processing the call keyword arguments (line 141)
        kwargs_380962 = {}
        # Getting the type of 'matrix_format' (line 141)
        matrix_format_380959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 28), 'matrix_format', False)
        # Obtaining the member 'decode' of a type (line 141)
        decode_380960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 28), matrix_format_380959, 'decode')
        # Calling decode(args, kwargs) (line 141)
        decode_call_result_380963 = invoke(stypy.reporting.localization.Localization(__file__, 141, 28), decode_380960, *[str_380961], **kwargs_380962)
        
        # Assigning a type to the variable 'matrix_format' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'matrix_format', decode_call_result_380963)
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 144):
        
        # Call to getattr(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'scipy' (line 144)
        scipy_380965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 144)
        sparse_380966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 26), scipy_380965, 'sparse')
        
        # Call to format(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'matrix_format' (line 144)
        matrix_format_380969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 59), 'matrix_format', False)
        # Processing the call keyword arguments (line 144)
        kwargs_380970 = {}
        str_380967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 40), 'str', '{}_matrix')
        # Obtaining the member 'format' of a type (line 144)
        format_380968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 40), str_380967, 'format')
        # Calling format(args, kwargs) (line 144)
        format_call_result_380971 = invoke(stypy.reporting.localization.Localization(__file__, 144, 40), format_380968, *[matrix_format_380969], **kwargs_380970)
        
        # Processing the call keyword arguments (line 144)
        kwargs_380972 = {}
        # Getting the type of 'getattr' (line 144)
        getattr_380964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 144)
        getattr_call_result_380973 = invoke(stypy.reporting.localization.Localization(__file__, 144, 18), getattr_380964, *[sparse_380966, format_call_result_380971], **kwargs_380972)
        
        # Assigning a type to the variable 'cls' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'cls', getattr_call_result_380973)
        # SSA branch for the except part of a try statement (line 143)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 143)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to format(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'matrix_format' (line 146)
        matrix_format_380977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 65), 'matrix_format', False)
        # Processing the call keyword arguments (line 146)
        kwargs_380978 = {}
        str_380975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 29), 'str', 'Unknown matrix format "{}"')
        # Obtaining the member 'format' of a type (line 146)
        format_380976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 29), str_380975, 'format')
        # Calling format(args, kwargs) (line 146)
        format_call_result_380979 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), format_380976, *[matrix_format_380977], **kwargs_380978)
        
        # Processing the call keyword arguments (line 146)
        kwargs_380980 = {}
        # Getting the type of 'ValueError' (line 146)
        ValueError_380974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 146)
        ValueError_call_result_380981 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), ValueError_380974, *[format_call_result_380979], **kwargs_380980)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 12), ValueError_call_result_380981, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'matrix_format' (line 148)
        matrix_format_380982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'matrix_format')
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_380983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        str_380984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 29), 'str', 'csc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 29), tuple_380983, str_380984)
        # Adding element type (line 148)
        str_380985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 36), 'str', 'csr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 29), tuple_380983, str_380985)
        # Adding element type (line 148)
        str_380986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 43), 'str', 'bsr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 29), tuple_380983, str_380986)
        
        # Applying the binary operator 'in' (line 148)
        result_contains_380987 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), 'in', matrix_format_380982, tuple_380983)
        
        # Testing the type of an if condition (line 148)
        if_condition_380988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 8), result_contains_380987)
        # Assigning a type to the variable 'if_condition_380988' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'if_condition_380988', if_condition_380988)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to cls(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_380990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        
        # Obtaining the type of the subscript
        str_380991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'str', 'data')
        # Getting the type of 'loaded' (line 149)
        loaded_380992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___380993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), loaded_380992, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_380994 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), getitem___380993, str_380991)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 24), tuple_380990, subscript_call_result_380994)
        # Adding element type (line 149)
        
        # Obtaining the type of the subscript
        str_380995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 47), 'str', 'indices')
        # Getting the type of 'loaded' (line 149)
        loaded_380996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___380997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 40), loaded_380996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_380998 = invoke(stypy.reporting.localization.Localization(__file__, 149, 40), getitem___380997, str_380995)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 24), tuple_380990, subscript_call_result_380998)
        # Adding element type (line 149)
        
        # Obtaining the type of the subscript
        str_380999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 66), 'str', 'indptr')
        # Getting the type of 'loaded' (line 149)
        loaded_381000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 59), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___381001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 59), loaded_381000, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_381002 = invoke(stypy.reporting.localization.Localization(__file__, 149, 59), getitem___381001, str_380999)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 24), tuple_380990, subscript_call_result_381002)
        
        # Processing the call keyword arguments (line 149)
        
        # Obtaining the type of the subscript
        str_381003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 91), 'str', 'shape')
        # Getting the type of 'loaded' (line 149)
        loaded_381004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 84), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___381005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 84), loaded_381004, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_381006 = invoke(stypy.reporting.localization.Localization(__file__, 149, 84), getitem___381005, str_381003)
        
        keyword_381007 = subscript_call_result_381006
        kwargs_381008 = {'shape': keyword_381007}
        # Getting the type of 'cls' (line 149)
        cls_380989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'cls', False)
        # Calling cls(args, kwargs) (line 149)
        cls_call_result_381009 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), cls_380989, *[tuple_380990], **kwargs_381008)
        
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'stypy_return_type', cls_call_result_381009)
        # SSA branch for the else part of an if statement (line 148)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'matrix_format' (line 150)
        matrix_format_381010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'matrix_format')
        str_381011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 30), 'str', 'dia')
        # Applying the binary operator '==' (line 150)
        result_eq_381012 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 13), '==', matrix_format_381010, str_381011)
        
        # Testing the type of an if condition (line 150)
        if_condition_381013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 13), result_eq_381012)
        # Assigning a type to the variable 'if_condition_381013' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'if_condition_381013', if_condition_381013)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to cls(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining an instance of the builtin type 'tuple' (line 151)
        tuple_381015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 151)
        # Adding element type (line 151)
        
        # Obtaining the type of the subscript
        str_381016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 31), 'str', 'data')
        # Getting the type of 'loaded' (line 151)
        loaded_381017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___381018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), loaded_381017, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_381019 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), getitem___381018, str_381016)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 24), tuple_381015, subscript_call_result_381019)
        # Adding element type (line 151)
        
        # Obtaining the type of the subscript
        str_381020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 47), 'str', 'offsets')
        # Getting the type of 'loaded' (line 151)
        loaded_381021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 40), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___381022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 40), loaded_381021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_381023 = invoke(stypy.reporting.localization.Localization(__file__, 151, 40), getitem___381022, str_381020)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 24), tuple_381015, subscript_call_result_381023)
        
        # Processing the call keyword arguments (line 151)
        
        # Obtaining the type of the subscript
        str_381024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 73), 'str', 'shape')
        # Getting the type of 'loaded' (line 151)
        loaded_381025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 66), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___381026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 66), loaded_381025, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_381027 = invoke(stypy.reporting.localization.Localization(__file__, 151, 66), getitem___381026, str_381024)
        
        keyword_381028 = subscript_call_result_381027
        kwargs_381029 = {'shape': keyword_381028}
        # Getting the type of 'cls' (line 151)
        cls_381014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'cls', False)
        # Calling cls(args, kwargs) (line 151)
        cls_call_result_381030 = invoke(stypy.reporting.localization.Localization(__file__, 151, 19), cls_381014, *[tuple_381015], **kwargs_381029)
        
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'stypy_return_type', cls_call_result_381030)
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'matrix_format' (line 152)
        matrix_format_381031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'matrix_format')
        str_381032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 30), 'str', 'coo')
        # Applying the binary operator '==' (line 152)
        result_eq_381033 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 13), '==', matrix_format_381031, str_381032)
        
        # Testing the type of an if condition (line 152)
        if_condition_381034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 13), result_eq_381033)
        # Assigning a type to the variable 'if_condition_381034' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'if_condition_381034', if_condition_381034)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to cls(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_381036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        
        # Obtaining the type of the subscript
        str_381037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 31), 'str', 'data')
        # Getting the type of 'loaded' (line 153)
        loaded_381038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___381039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), loaded_381038, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_381040 = invoke(stypy.reporting.localization.Localization(__file__, 153, 24), getitem___381039, str_381037)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 24), tuple_381036, subscript_call_result_381040)
        # Adding element type (line 153)
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_381041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        
        # Obtaining the type of the subscript
        str_381042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 48), 'str', 'row')
        # Getting the type of 'loaded' (line 153)
        loaded_381043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 41), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___381044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 41), loaded_381043, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_381045 = invoke(stypy.reporting.localization.Localization(__file__, 153, 41), getitem___381044, str_381042)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 41), tuple_381041, subscript_call_result_381045)
        # Adding element type (line 153)
        
        # Obtaining the type of the subscript
        str_381046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 63), 'str', 'col')
        # Getting the type of 'loaded' (line 153)
        loaded_381047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 56), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___381048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 56), loaded_381047, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_381049 = invoke(stypy.reporting.localization.Localization(__file__, 153, 56), getitem___381048, str_381046)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 41), tuple_381041, subscript_call_result_381049)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 24), tuple_381036, tuple_381041)
        
        # Processing the call keyword arguments (line 153)
        
        # Obtaining the type of the subscript
        str_381050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 86), 'str', 'shape')
        # Getting the type of 'loaded' (line 153)
        loaded_381051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 79), 'loaded', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___381052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 79), loaded_381051, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_381053 = invoke(stypy.reporting.localization.Localization(__file__, 153, 79), getitem___381052, str_381050)
        
        keyword_381054 = subscript_call_result_381053
        kwargs_381055 = {'shape': keyword_381054}
        # Getting the type of 'cls' (line 153)
        cls_381035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'cls', False)
        # Calling cls(args, kwargs) (line 153)
        cls_call_result_381056 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), cls_381035, *[tuple_381036], **kwargs_381055)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'stypy_return_type', cls_call_result_381056)
        # SSA branch for the else part of an if statement (line 152)
        module_type_store.open_ssa_branch('else')
        
        # Call to NotImplementedError(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to format(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'matrix_format' (line 156)
        matrix_format_381060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 75), 'matrix_format', False)
        # Processing the call keyword arguments (line 155)
        kwargs_381061 = {}
        str_381058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 38), 'str', 'Load is not implemented for sparse matrix of format {}.')
        # Obtaining the member 'format' of a type (line 155)
        format_381059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 38), str_381058, 'format')
        # Calling format(args, kwargs) (line 155)
        format_call_result_381062 = invoke(stypy.reporting.localization.Localization(__file__, 155, 38), format_381059, *[matrix_format_381060], **kwargs_381061)
        
        # Processing the call keyword arguments (line 155)
        kwargs_381063 = {}
        # Getting the type of 'NotImplementedError' (line 155)
        NotImplementedError_381057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 155)
        NotImplementedError_call_result_381064 = invoke(stypy.reporting.localization.Localization(__file__, 155, 18), NotImplementedError_381057, *[format_call_result_381062], **kwargs_381063)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 12), NotImplementedError_call_result_381064, 'raise parameter', BaseException)
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 130)
        exit___381065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), load_call_result_380924, '__exit__')
        with_exit_381066 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), exit___381065, None, None, None)

    
    # ################# End of 'load_npz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_npz' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_381067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381067)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_npz'
    return stypy_return_type_381067

# Assigning a type to the variable 'load_npz' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'load_npz', load_npz)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
