
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Functions to extract parts of sparse matrices
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: __docformat__ = "restructuredtext en"
7: 
8: __all__ = ['find', 'tril', 'triu']
9: 
10: 
11: from .coo import coo_matrix
12: 
13: 
14: def find(A):
15:     '''Return the indices and values of the nonzero elements of a matrix
16: 
17:     Parameters
18:     ----------
19:     A : dense or sparse matrix
20:         Matrix whose nonzero elements are desired.
21: 
22:     Returns
23:     -------
24:     (I,J,V) : tuple of arrays
25:         I,J, and V contain the row indices, column indices, and values
26:         of the nonzero matrix entries.
27: 
28: 
29:     Examples
30:     --------
31:     >>> from scipy.sparse import csr_matrix, find
32:     >>> A = csr_matrix([[7.0, 8.0, 0],[0, 0, 9.0]])
33:     >>> find(A)
34:     (array([0, 0, 1], dtype=int32), array([0, 1, 2], dtype=int32), array([ 7.,  8.,  9.]))
35: 
36:     '''
37: 
38:     A = coo_matrix(A, copy=True)
39:     A.sum_duplicates()
40:     # remove explicit zeros
41:     nz_mask = A.data != 0
42:     return A.row[nz_mask], A.col[nz_mask], A.data[nz_mask]
43: 
44: 
45: def tril(A, k=0, format=None):
46:     '''Return the lower triangular portion of a matrix in sparse format
47: 
48:     Returns the elements on or below the k-th diagonal of the matrix A.
49:         - k = 0 corresponds to the main diagonal
50:         - k > 0 is above the main diagonal
51:         - k < 0 is below the main diagonal
52: 
53:     Parameters
54:     ----------
55:     A : dense or sparse matrix
56:         Matrix whose lower trianglar portion is desired.
57:     k : integer : optional
58:         The top-most diagonal of the lower triangle.
59:     format : string
60:         Sparse format of the result, e.g. format="csr", etc.
61: 
62:     Returns
63:     -------
64:     L : sparse matrix
65:         Lower triangular portion of A in sparse format.
66: 
67:     See Also
68:     --------
69:     triu : upper triangle in sparse format
70: 
71:     Examples
72:     --------
73:     >>> from scipy.sparse import csr_matrix, tril
74:     >>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
75:     ...                dtype='int32')
76:     >>> A.toarray()
77:     array([[1, 2, 0, 0, 3],
78:            [4, 5, 0, 6, 7],
79:            [0, 0, 8, 9, 0]])
80:     >>> tril(A).toarray()
81:     array([[1, 0, 0, 0, 0],
82:            [4, 5, 0, 0, 0],
83:            [0, 0, 8, 0, 0]])
84:     >>> tril(A).nnz
85:     4
86:     >>> tril(A, k=1).toarray()
87:     array([[1, 2, 0, 0, 0],
88:            [4, 5, 0, 0, 0],
89:            [0, 0, 8, 9, 0]])
90:     >>> tril(A, k=-1).toarray()
91:     array([[0, 0, 0, 0, 0],
92:            [4, 0, 0, 0, 0],
93:            [0, 0, 0, 0, 0]])
94:     >>> tril(A, format='csc')
95:     <3x5 sparse matrix of type '<class 'numpy.int32'>'
96:             with 4 stored elements in Compressed Sparse Column format>
97: 
98:     '''
99: 
100:     # convert to COOrdinate format where things are easy
101:     A = coo_matrix(A, copy=False)
102:     mask = A.row + k >= A.col
103:     return _masked_coo(A, mask).asformat(format)
104: 
105: 
106: def triu(A, k=0, format=None):
107:     '''Return the upper triangular portion of a matrix in sparse format
108: 
109:     Returns the elements on or above the k-th diagonal of the matrix A.
110:         - k = 0 corresponds to the main diagonal
111:         - k > 0 is above the main diagonal
112:         - k < 0 is below the main diagonal
113: 
114:     Parameters
115:     ----------
116:     A : dense or sparse matrix
117:         Matrix whose upper trianglar portion is desired.
118:     k : integer : optional
119:         The bottom-most diagonal of the upper triangle.
120:     format : string
121:         Sparse format of the result, e.g. format="csr", etc.
122: 
123:     Returns
124:     -------
125:     L : sparse matrix
126:         Upper triangular portion of A in sparse format.
127: 
128:     See Also
129:     --------
130:     tril : lower triangle in sparse format
131: 
132:     Examples
133:     --------
134:     >>> from scipy.sparse import csr_matrix, triu
135:     >>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
136:     ...                dtype='int32')
137:     >>> A.toarray()
138:     array([[1, 2, 0, 0, 3],
139:            [4, 5, 0, 6, 7],
140:            [0, 0, 8, 9, 0]])
141:     >>> triu(A).toarray()
142:     array([[1, 2, 0, 0, 3],
143:            [0, 5, 0, 6, 7],
144:            [0, 0, 8, 9, 0]])
145:     >>> triu(A).nnz
146:     8
147:     >>> triu(A, k=1).toarray()
148:     array([[0, 2, 0, 0, 3],
149:            [0, 0, 0, 6, 7],
150:            [0, 0, 0, 9, 0]])
151:     >>> triu(A, k=-1).toarray()
152:     array([[1, 2, 0, 0, 3],
153:            [4, 5, 0, 6, 7],
154:            [0, 0, 8, 9, 0]])
155:     >>> triu(A, format='csc')
156:     <3x5 sparse matrix of type '<class 'numpy.int32'>'
157:             with 8 stored elements in Compressed Sparse Column format>
158: 
159:     '''
160: 
161:     # convert to COOrdinate format where things are easy
162:     A = coo_matrix(A, copy=False)
163:     mask = A.row + k <= A.col
164:     return _masked_coo(A, mask).asformat(format)
165: 
166: 
167: def _masked_coo(A, mask):
168:     row = A.row[mask]
169:     col = A.col[mask]
170:     data = A.data[mask]
171:     return coo_matrix((data, (row, col)), shape=A.shape, dtype=A.dtype)
172: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_376594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Functions to extract parts of sparse matrices\n')

# Assigning a Str to a Name (line 6):
str_376595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__docformat__', str_376595)

# Assigning a List to a Name (line 8):
__all__ = ['find', 'tril', 'triu']
module_type_store.set_exportable_members(['find', 'tril', 'triu'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_376596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_376597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'find')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_376596, str_376597)
# Adding element type (line 8)
str_376598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', 'tril')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_376596, str_376598)
# Adding element type (line 8)
str_376599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'triu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_376596, str_376599)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_376596)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.coo import coo_matrix' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_376600 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.coo')

if (type(import_376600) is not StypyTypeError):

    if (import_376600 != 'pyd_module'):
        __import__(import_376600)
        sys_modules_376601 = sys.modules[import_376600]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.coo', sys_modules_376601.module_type_store, module_type_store, ['coo_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_376601, sys_modules_376601.module_type_store, module_type_store)
    else:
        from scipy.sparse.coo import coo_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.coo' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.coo', import_376600)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


@norecursion
def find(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find'
    module_type_store = module_type_store.open_function_context('find', 14, 0, False)
    
    # Passed parameters checking function
    find.stypy_localization = localization
    find.stypy_type_of_self = None
    find.stypy_type_store = module_type_store
    find.stypy_function_name = 'find'
    find.stypy_param_names_list = ['A']
    find.stypy_varargs_param_name = None
    find.stypy_kwargs_param_name = None
    find.stypy_call_defaults = defaults
    find.stypy_call_varargs = varargs
    find.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find(...)' code ##################

    str_376602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', 'Return the indices and values of the nonzero elements of a matrix\n\n    Parameters\n    ----------\n    A : dense or sparse matrix\n        Matrix whose nonzero elements are desired.\n\n    Returns\n    -------\n    (I,J,V) : tuple of arrays\n        I,J, and V contain the row indices, column indices, and values\n        of the nonzero matrix entries.\n\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, find\n    >>> A = csr_matrix([[7.0, 8.0, 0],[0, 0, 9.0]])\n    >>> find(A)\n    (array([0, 0, 1], dtype=int32), array([0, 1, 2], dtype=int32), array([ 7.,  8.,  9.]))\n\n    ')
    
    # Assigning a Call to a Name (line 38):
    
    # Call to coo_matrix(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'A' (line 38)
    A_376604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'A', False)
    # Processing the call keyword arguments (line 38)
    # Getting the type of 'True' (line 38)
    True_376605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'True', False)
    keyword_376606 = True_376605
    kwargs_376607 = {'copy': keyword_376606}
    # Getting the type of 'coo_matrix' (line 38)
    coo_matrix_376603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 38)
    coo_matrix_call_result_376608 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), coo_matrix_376603, *[A_376604], **kwargs_376607)
    
    # Assigning a type to the variable 'A' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'A', coo_matrix_call_result_376608)
    
    # Call to sum_duplicates(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_376611 = {}
    # Getting the type of 'A' (line 39)
    A_376609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'A', False)
    # Obtaining the member 'sum_duplicates' of a type (line 39)
    sum_duplicates_376610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 4), A_376609, 'sum_duplicates')
    # Calling sum_duplicates(args, kwargs) (line 39)
    sum_duplicates_call_result_376612 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), sum_duplicates_376610, *[], **kwargs_376611)
    
    
    # Assigning a Compare to a Name (line 41):
    
    # Getting the type of 'A' (line 41)
    A_376613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'A')
    # Obtaining the member 'data' of a type (line 41)
    data_376614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 14), A_376613, 'data')
    int_376615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'int')
    # Applying the binary operator '!=' (line 41)
    result_ne_376616 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 14), '!=', data_376614, int_376615)
    
    # Assigning a type to the variable 'nz_mask' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'nz_mask', result_ne_376616)
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_376617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nz_mask' (line 42)
    nz_mask_376618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'nz_mask')
    # Getting the type of 'A' (line 42)
    A_376619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'A')
    # Obtaining the member 'row' of a type (line 42)
    row_376620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), A_376619, 'row')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___376621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), row_376620, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_376622 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), getitem___376621, nz_mask_376618)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 11), tuple_376617, subscript_call_result_376622)
    # Adding element type (line 42)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nz_mask' (line 42)
    nz_mask_376623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'nz_mask')
    # Getting the type of 'A' (line 42)
    A_376624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'A')
    # Obtaining the member 'col' of a type (line 42)
    col_376625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), A_376624, 'col')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___376626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), col_376625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_376627 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), getitem___376626, nz_mask_376623)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 11), tuple_376617, subscript_call_result_376627)
    # Adding element type (line 42)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nz_mask' (line 42)
    nz_mask_376628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 50), 'nz_mask')
    # Getting the type of 'A' (line 42)
    A_376629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 43), 'A')
    # Obtaining the member 'data' of a type (line 42)
    data_376630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 43), A_376629, 'data')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___376631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 43), data_376630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_376632 = invoke(stypy.reporting.localization.Localization(__file__, 42, 43), getitem___376631, nz_mask_376628)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 11), tuple_376617, subscript_call_result_376632)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', tuple_376617)
    
    # ################# End of 'find(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_376633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_376633)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find'
    return stypy_return_type_376633

# Assigning a type to the variable 'find' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'find', find)

@norecursion
def tril(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_376634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'int')
    # Getting the type of 'None' (line 45)
    None_376635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'None')
    defaults = [int_376634, None_376635]
    # Create a new context for function 'tril'
    module_type_store = module_type_store.open_function_context('tril', 45, 0, False)
    
    # Passed parameters checking function
    tril.stypy_localization = localization
    tril.stypy_type_of_self = None
    tril.stypy_type_store = module_type_store
    tril.stypy_function_name = 'tril'
    tril.stypy_param_names_list = ['A', 'k', 'format']
    tril.stypy_varargs_param_name = None
    tril.stypy_kwargs_param_name = None
    tril.stypy_call_defaults = defaults
    tril.stypy_call_varargs = varargs
    tril.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tril', ['A', 'k', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tril', localization, ['A', 'k', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tril(...)' code ##################

    str_376636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', 'Return the lower triangular portion of a matrix in sparse format\n\n    Returns the elements on or below the k-th diagonal of the matrix A.\n        - k = 0 corresponds to the main diagonal\n        - k > 0 is above the main diagonal\n        - k < 0 is below the main diagonal\n\n    Parameters\n    ----------\n    A : dense or sparse matrix\n        Matrix whose lower trianglar portion is desired.\n    k : integer : optional\n        The top-most diagonal of the lower triangle.\n    format : string\n        Sparse format of the result, e.g. format="csr", etc.\n\n    Returns\n    -------\n    L : sparse matrix\n        Lower triangular portion of A in sparse format.\n\n    See Also\n    --------\n    triu : upper triangle in sparse format\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, tril\n    >>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],\n    ...                dtype=\'int32\')\n    >>> A.toarray()\n    array([[1, 2, 0, 0, 3],\n           [4, 5, 0, 6, 7],\n           [0, 0, 8, 9, 0]])\n    >>> tril(A).toarray()\n    array([[1, 0, 0, 0, 0],\n           [4, 5, 0, 0, 0],\n           [0, 0, 8, 0, 0]])\n    >>> tril(A).nnz\n    4\n    >>> tril(A, k=1).toarray()\n    array([[1, 2, 0, 0, 0],\n           [4, 5, 0, 0, 0],\n           [0, 0, 8, 9, 0]])\n    >>> tril(A, k=-1).toarray()\n    array([[0, 0, 0, 0, 0],\n           [4, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0]])\n    >>> tril(A, format=\'csc\')\n    <3x5 sparse matrix of type \'<class \'numpy.int32\'>\'\n            with 4 stored elements in Compressed Sparse Column format>\n\n    ')
    
    # Assigning a Call to a Name (line 101):
    
    # Call to coo_matrix(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'A' (line 101)
    A_376638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'A', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'False' (line 101)
    False_376639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'False', False)
    keyword_376640 = False_376639
    kwargs_376641 = {'copy': keyword_376640}
    # Getting the type of 'coo_matrix' (line 101)
    coo_matrix_376637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 101)
    coo_matrix_call_result_376642 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), coo_matrix_376637, *[A_376638], **kwargs_376641)
    
    # Assigning a type to the variable 'A' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'A', coo_matrix_call_result_376642)
    
    # Assigning a Compare to a Name (line 102):
    
    # Getting the type of 'A' (line 102)
    A_376643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'A')
    # Obtaining the member 'row' of a type (line 102)
    row_376644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), A_376643, 'row')
    # Getting the type of 'k' (line 102)
    k_376645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'k')
    # Applying the binary operator '+' (line 102)
    result_add_376646 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '+', row_376644, k_376645)
    
    # Getting the type of 'A' (line 102)
    A_376647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'A')
    # Obtaining the member 'col' of a type (line 102)
    col_376648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), A_376647, 'col')
    # Applying the binary operator '>=' (line 102)
    result_ge_376649 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '>=', result_add_376646, col_376648)
    
    # Assigning a type to the variable 'mask' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'mask', result_ge_376649)
    
    # Call to asformat(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'format' (line 103)
    format_376656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 41), 'format', False)
    # Processing the call keyword arguments (line 103)
    kwargs_376657 = {}
    
    # Call to _masked_coo(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'A' (line 103)
    A_376651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'A', False)
    # Getting the type of 'mask' (line 103)
    mask_376652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'mask', False)
    # Processing the call keyword arguments (line 103)
    kwargs_376653 = {}
    # Getting the type of '_masked_coo' (line 103)
    _masked_coo_376650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), '_masked_coo', False)
    # Calling _masked_coo(args, kwargs) (line 103)
    _masked_coo_call_result_376654 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), _masked_coo_376650, *[A_376651, mask_376652], **kwargs_376653)
    
    # Obtaining the member 'asformat' of a type (line 103)
    asformat_376655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 11), _masked_coo_call_result_376654, 'asformat')
    # Calling asformat(args, kwargs) (line 103)
    asformat_call_result_376658 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), asformat_376655, *[format_376656], **kwargs_376657)
    
    # Assigning a type to the variable 'stypy_return_type' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type', asformat_call_result_376658)
    
    # ################# End of 'tril(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tril' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_376659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_376659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tril'
    return stypy_return_type_376659

# Assigning a type to the variable 'tril' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'tril', tril)

@norecursion
def triu(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_376660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 14), 'int')
    # Getting the type of 'None' (line 106)
    None_376661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'None')
    defaults = [int_376660, None_376661]
    # Create a new context for function 'triu'
    module_type_store = module_type_store.open_function_context('triu', 106, 0, False)
    
    # Passed parameters checking function
    triu.stypy_localization = localization
    triu.stypy_type_of_self = None
    triu.stypy_type_store = module_type_store
    triu.stypy_function_name = 'triu'
    triu.stypy_param_names_list = ['A', 'k', 'format']
    triu.stypy_varargs_param_name = None
    triu.stypy_kwargs_param_name = None
    triu.stypy_call_defaults = defaults
    triu.stypy_call_varargs = varargs
    triu.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'triu', ['A', 'k', 'format'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'triu', localization, ['A', 'k', 'format'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'triu(...)' code ##################

    str_376662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, (-1)), 'str', 'Return the upper triangular portion of a matrix in sparse format\n\n    Returns the elements on or above the k-th diagonal of the matrix A.\n        - k = 0 corresponds to the main diagonal\n        - k > 0 is above the main diagonal\n        - k < 0 is below the main diagonal\n\n    Parameters\n    ----------\n    A : dense or sparse matrix\n        Matrix whose upper trianglar portion is desired.\n    k : integer : optional\n        The bottom-most diagonal of the upper triangle.\n    format : string\n        Sparse format of the result, e.g. format="csr", etc.\n\n    Returns\n    -------\n    L : sparse matrix\n        Upper triangular portion of A in sparse format.\n\n    See Also\n    --------\n    tril : lower triangle in sparse format\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, triu\n    >>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],\n    ...                dtype=\'int32\')\n    >>> A.toarray()\n    array([[1, 2, 0, 0, 3],\n           [4, 5, 0, 6, 7],\n           [0, 0, 8, 9, 0]])\n    >>> triu(A).toarray()\n    array([[1, 2, 0, 0, 3],\n           [0, 5, 0, 6, 7],\n           [0, 0, 8, 9, 0]])\n    >>> triu(A).nnz\n    8\n    >>> triu(A, k=1).toarray()\n    array([[0, 2, 0, 0, 3],\n           [0, 0, 0, 6, 7],\n           [0, 0, 0, 9, 0]])\n    >>> triu(A, k=-1).toarray()\n    array([[1, 2, 0, 0, 3],\n           [4, 5, 0, 6, 7],\n           [0, 0, 8, 9, 0]])\n    >>> triu(A, format=\'csc\')\n    <3x5 sparse matrix of type \'<class \'numpy.int32\'>\'\n            with 8 stored elements in Compressed Sparse Column format>\n\n    ')
    
    # Assigning a Call to a Name (line 162):
    
    # Call to coo_matrix(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'A' (line 162)
    A_376664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'A', False)
    # Processing the call keyword arguments (line 162)
    # Getting the type of 'False' (line 162)
    False_376665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'False', False)
    keyword_376666 = False_376665
    kwargs_376667 = {'copy': keyword_376666}
    # Getting the type of 'coo_matrix' (line 162)
    coo_matrix_376663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 162)
    coo_matrix_call_result_376668 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), coo_matrix_376663, *[A_376664], **kwargs_376667)
    
    # Assigning a type to the variable 'A' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'A', coo_matrix_call_result_376668)
    
    # Assigning a Compare to a Name (line 163):
    
    # Getting the type of 'A' (line 163)
    A_376669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'A')
    # Obtaining the member 'row' of a type (line 163)
    row_376670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), A_376669, 'row')
    # Getting the type of 'k' (line 163)
    k_376671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'k')
    # Applying the binary operator '+' (line 163)
    result_add_376672 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '+', row_376670, k_376671)
    
    # Getting the type of 'A' (line 163)
    A_376673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'A')
    # Obtaining the member 'col' of a type (line 163)
    col_376674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 24), A_376673, 'col')
    # Applying the binary operator '<=' (line 163)
    result_le_376675 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '<=', result_add_376672, col_376674)
    
    # Assigning a type to the variable 'mask' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'mask', result_le_376675)
    
    # Call to asformat(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'format' (line 164)
    format_376682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 41), 'format', False)
    # Processing the call keyword arguments (line 164)
    kwargs_376683 = {}
    
    # Call to _masked_coo(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'A' (line 164)
    A_376677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'A', False)
    # Getting the type of 'mask' (line 164)
    mask_376678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'mask', False)
    # Processing the call keyword arguments (line 164)
    kwargs_376679 = {}
    # Getting the type of '_masked_coo' (line 164)
    _masked_coo_376676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), '_masked_coo', False)
    # Calling _masked_coo(args, kwargs) (line 164)
    _masked_coo_call_result_376680 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), _masked_coo_376676, *[A_376677, mask_376678], **kwargs_376679)
    
    # Obtaining the member 'asformat' of a type (line 164)
    asformat_376681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), _masked_coo_call_result_376680, 'asformat')
    # Calling asformat(args, kwargs) (line 164)
    asformat_call_result_376684 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), asformat_376681, *[format_376682], **kwargs_376683)
    
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type', asformat_call_result_376684)
    
    # ################# End of 'triu(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'triu' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_376685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_376685)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'triu'
    return stypy_return_type_376685

# Assigning a type to the variable 'triu' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'triu', triu)

@norecursion
def _masked_coo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_masked_coo'
    module_type_store = module_type_store.open_function_context('_masked_coo', 167, 0, False)
    
    # Passed parameters checking function
    _masked_coo.stypy_localization = localization
    _masked_coo.stypy_type_of_self = None
    _masked_coo.stypy_type_store = module_type_store
    _masked_coo.stypy_function_name = '_masked_coo'
    _masked_coo.stypy_param_names_list = ['A', 'mask']
    _masked_coo.stypy_varargs_param_name = None
    _masked_coo.stypy_kwargs_param_name = None
    _masked_coo.stypy_call_defaults = defaults
    _masked_coo.stypy_call_varargs = varargs
    _masked_coo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_masked_coo', ['A', 'mask'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_masked_coo', localization, ['A', 'mask'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_masked_coo(...)' code ##################

    
    # Assigning a Subscript to a Name (line 168):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 168)
    mask_376686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'mask')
    # Getting the type of 'A' (line 168)
    A_376687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 10), 'A')
    # Obtaining the member 'row' of a type (line 168)
    row_376688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 10), A_376687, 'row')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___376689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 10), row_376688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_376690 = invoke(stypy.reporting.localization.Localization(__file__, 168, 10), getitem___376689, mask_376686)
    
    # Assigning a type to the variable 'row' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'row', subscript_call_result_376690)
    
    # Assigning a Subscript to a Name (line 169):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 169)
    mask_376691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'mask')
    # Getting the type of 'A' (line 169)
    A_376692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 10), 'A')
    # Obtaining the member 'col' of a type (line 169)
    col_376693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 10), A_376692, 'col')
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___376694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 10), col_376693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_376695 = invoke(stypy.reporting.localization.Localization(__file__, 169, 10), getitem___376694, mask_376691)
    
    # Assigning a type to the variable 'col' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'col', subscript_call_result_376695)
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 170)
    mask_376696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'mask')
    # Getting the type of 'A' (line 170)
    A_376697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'A')
    # Obtaining the member 'data' of a type (line 170)
    data_376698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), A_376697, 'data')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___376699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), data_376698, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_376700 = invoke(stypy.reporting.localization.Localization(__file__, 170, 11), getitem___376699, mask_376696)
    
    # Assigning a type to the variable 'data' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'data', subscript_call_result_376700)
    
    # Call to coo_matrix(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Obtaining an instance of the builtin type 'tuple' (line 171)
    tuple_376702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 171)
    # Adding element type (line 171)
    # Getting the type of 'data' (line 171)
    data_376703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 23), tuple_376702, data_376703)
    # Adding element type (line 171)
    
    # Obtaining an instance of the builtin type 'tuple' (line 171)
    tuple_376704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 171)
    # Adding element type (line 171)
    # Getting the type of 'row' (line 171)
    row_376705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'row', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), tuple_376704, row_376705)
    # Adding element type (line 171)
    # Getting the type of 'col' (line 171)
    col_376706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'col', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), tuple_376704, col_376706)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 23), tuple_376702, tuple_376704)
    
    # Processing the call keyword arguments (line 171)
    # Getting the type of 'A' (line 171)
    A_376707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'A', False)
    # Obtaining the member 'shape' of a type (line 171)
    shape_376708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 48), A_376707, 'shape')
    keyword_376709 = shape_376708
    # Getting the type of 'A' (line 171)
    A_376710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 63), 'A', False)
    # Obtaining the member 'dtype' of a type (line 171)
    dtype_376711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 63), A_376710, 'dtype')
    keyword_376712 = dtype_376711
    kwargs_376713 = {'dtype': keyword_376712, 'shape': keyword_376709}
    # Getting the type of 'coo_matrix' (line 171)
    coo_matrix_376701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 171)
    coo_matrix_call_result_376714 = invoke(stypy.reporting.localization.Localization(__file__, 171, 11), coo_matrix_376701, *[tuple_376702], **kwargs_376713)
    
    # Assigning a type to the variable 'stypy_return_type' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type', coo_matrix_call_result_376714)
    
    # ################# End of '_masked_coo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_masked_coo' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_376715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_376715)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_masked_coo'
    return stypy_return_type_376715

# Assigning a type to the variable '_masked_coo' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), '_masked_coo', _masked_coo)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
