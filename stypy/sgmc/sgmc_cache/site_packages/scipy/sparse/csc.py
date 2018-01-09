
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Compressed Sparse Column matrix format'''
2: from __future__ import division, print_function, absolute_import
3: 
4: __docformat__ = "restructuredtext en"
5: 
6: __all__ = ['csc_matrix', 'isspmatrix_csc']
7: 
8: 
9: import numpy as np
10: 
11: from .base import spmatrix
12: from ._sparsetools import csc_tocsr
13: from . import _sparsetools
14: from .sputils import upcast, isintlike, IndexMixin, get_index_dtype
15: 
16: from .compressed import _cs_matrix
17: 
18: 
19: class csc_matrix(_cs_matrix, IndexMixin):
20:     '''
21:     Compressed Sparse Column matrix
22: 
23:     This can be instantiated in several ways:
24: 
25:         csc_matrix(D)
26:             with a dense matrix or rank-2 ndarray D
27: 
28:         csc_matrix(S)
29:             with another sparse matrix S (equivalent to S.tocsc())
30: 
31:         csc_matrix((M, N), [dtype])
32:             to construct an empty matrix with shape (M, N)
33:             dtype is optional, defaulting to dtype='d'.
34: 
35:         csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
36:             where ``data``, ``row_ind`` and ``col_ind`` satisfy the
37:             relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
38: 
39:         csc_matrix((data, indices, indptr), [shape=(M, N)])
40:             is the standard CSC representation where the row indices for
41:             column i are stored in ``indices[indptr[i]:indptr[i+1]]``
42:             and their corresponding values are stored in
43:             ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
44:             not supplied, the matrix dimensions are inferred from
45:             the index arrays.
46: 
47:     Attributes
48:     ----------
49:     dtype : dtype
50:         Data type of the matrix
51:     shape : 2-tuple
52:         Shape of the matrix
53:     ndim : int
54:         Number of dimensions (this is always 2)
55:     nnz
56:         Number of nonzero elements
57:     data
58:         Data array of the matrix
59:     indices
60:         CSC format index array
61:     indptr
62:         CSC format index pointer array
63:     has_sorted_indices
64:         Whether indices are sorted
65: 
66:     Notes
67:     -----
68: 
69:     Sparse matrices can be used in arithmetic operations: they support
70:     addition, subtraction, multiplication, division, and matrix power.
71: 
72:     Advantages of the CSC format
73:         - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
74:         - efficient column slicing
75:         - fast matrix vector products (CSR, BSR may be faster)
76: 
77:     Disadvantages of the CSC format
78:       - slow row slicing operations (consider CSR)
79:       - changes to the sparsity structure are expensive (consider LIL or DOK)
80: 
81: 
82:     Examples
83:     --------
84: 
85:     >>> import numpy as np
86:     >>> from scipy.sparse import csc_matrix
87:     >>> csc_matrix((3, 4), dtype=np.int8).toarray()
88:     array([[0, 0, 0, 0],
89:            [0, 0, 0, 0],
90:            [0, 0, 0, 0]], dtype=int8)
91: 
92:     >>> row = np.array([0, 2, 2, 0, 1, 2])
93:     >>> col = np.array([0, 0, 1, 2, 2, 2])
94:     >>> data = np.array([1, 2, 3, 4, 5, 6])
95:     >>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
96:     array([[1, 0, 4],
97:            [0, 0, 5],
98:            [2, 3, 6]])
99: 
100:     >>> indptr = np.array([0, 2, 3, 6])
101:     >>> indices = np.array([0, 2, 2, 0, 1, 2])
102:     >>> data = np.array([1, 2, 3, 4, 5, 6])
103:     >>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
104:     array([[1, 0, 4],
105:            [0, 0, 5],
106:            [2, 3, 6]])
107: 
108:     '''
109:     format = 'csc'
110: 
111:     def transpose(self, axes=None, copy=False):
112:         if axes is not None:
113:             raise ValueError(("Sparse matrices do not support "
114:                               "an 'axes' parameter because swapping "
115:                               "dimensions is the only logical permutation."))
116: 
117:         M, N = self.shape
118: 
119:         from .csr import csr_matrix
120:         return csr_matrix((self.data, self.indices,
121:                            self.indptr), (N, M), copy=copy)
122: 
123:     transpose.__doc__ = spmatrix.transpose.__doc__
124: 
125:     def __iter__(self):
126:         for r in self.tocsr():
127:             yield r
128: 
129:     def tocsc(self, copy=False):
130:         if copy:
131:             return self.copy()
132:         else:
133:             return self
134: 
135:     tocsc.__doc__ = spmatrix.tocsc.__doc__
136: 
137:     def tocsr(self, copy=False):
138:         M,N = self.shape
139:         idx_dtype = get_index_dtype((self.indptr, self.indices),
140:                                     maxval=max(self.nnz, N))
141:         indptr = np.empty(M + 1, dtype=idx_dtype)
142:         indices = np.empty(self.nnz, dtype=idx_dtype)
143:         data = np.empty(self.nnz, dtype=upcast(self.dtype))
144: 
145:         csc_tocsr(M, N,
146:                   self.indptr.astype(idx_dtype),
147:                   self.indices.astype(idx_dtype),
148:                   self.data,
149:                   indptr,
150:                   indices,
151:                   data)
152: 
153:         from .csr import csr_matrix
154:         A = csr_matrix((data, indices, indptr), shape=self.shape, copy=False)
155:         A.has_sorted_indices = True
156:         return A
157: 
158:     tocsr.__doc__ = spmatrix.tocsr.__doc__
159: 
160:     def __getitem__(self, key):
161:         # Use CSR to implement fancy indexing.
162: 
163:         row, col = self._unpack_index(key)
164:         # Things that return submatrices. row or col is a int or slice.
165:         if (isinstance(row, slice) or isinstance(col, slice) or
166:                 isintlike(row) or isintlike(col)):
167:             return self.T[col, row].T
168:         # Things that return a sequence of values.
169:         else:
170:             return self.T[col, row]
171: 
172:     def nonzero(self):
173:         # CSC can't use _cs_matrix's .nonzero method because it
174:         # returns the indices sorted for self transposed.
175: 
176:         # Get row and col indices, from _cs_matrix.tocoo
177:         major_dim, minor_dim = self._swap(self.shape)
178:         minor_indices = self.indices
179:         major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
180:         _sparsetools.expandptr(major_dim, self.indptr, major_indices)
181:         row, col = self._swap((major_indices, minor_indices))
182: 
183:         # Remove explicit zeros
184:         nz_mask = self.data != 0
185:         row = row[nz_mask]
186:         col = col[nz_mask]
187: 
188:         # Sort them to be in C-style order
189:         ind = np.argsort(row, kind='mergesort')
190:         row = row[ind]
191:         col = col[ind]
192: 
193:         return row, col
194: 
195:     nonzero.__doc__ = _cs_matrix.nonzero.__doc__
196: 
197:     def getrow(self, i):
198:         '''Returns a copy of row i of the matrix, as a (1 x n)
199:         CSR matrix (row vector).
200:         '''
201:         # we convert to CSR to maintain compatibility with old impl.
202:         # in spmatrix.getrow()
203:         return self._get_submatrix(i, slice(None)).tocsr()
204: 
205:     def getcol(self, i):
206:         '''Returns a copy of column i of the matrix, as a (m x 1)
207:         CSC matrix (column vector).
208:         '''
209:         M, N = self.shape
210:         i = int(i)
211:         if i < 0:
212:             i += N
213:         if i < 0 or i >= N:
214:             raise IndexError('index (%d) out of range' % i)
215:         idx = slice(*self.indptr[i:i+2])
216:         data = self.data[idx].copy()
217:         indices = self.indices[idx].copy()
218:         indptr = np.array([0, len(indices)], dtype=self.indptr.dtype)
219:         return csc_matrix((data, indices, indptr), shape=(M, 1),
220:                           dtype=self.dtype, copy=False)
221: 
222:     # these functions are used by the parent class (_cs_matrix)
223:     # to remove redudancy between csc_matrix and csr_matrix
224:     def _swap(self, x):
225:         '''swap the members of x if this is a column-oriented matrix
226:         '''
227:         return x[1], x[0]
228: 
229: 
230: def isspmatrix_csc(x):
231:     '''Is x of csc_matrix type?
232: 
233:     Parameters
234:     ----------
235:     x
236:         object to check for being a csc matrix
237: 
238:     Returns
239:     -------
240:     bool
241:         True if x is a csc matrix, False otherwise
242: 
243:     Examples
244:     --------
245:     >>> from scipy.sparse import csc_matrix, isspmatrix_csc
246:     >>> isspmatrix_csc(csc_matrix([[5]]))
247:     True
248: 
249:     >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
250:     >>> isspmatrix_csc(csr_matrix([[5]]))
251:     False
252:     '''
253:     return isinstance(x, csc_matrix)
254: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_369842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Compressed Sparse Column matrix format')

# Assigning a Str to a Name (line 4):

# Assigning a Str to a Name (line 4):
str_369843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__docformat__', str_369843)

# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):
__all__ = ['csc_matrix', 'isspmatrix_csc']
module_type_store.set_exportable_members(['csc_matrix', 'isspmatrix_csc'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_369844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_369845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'csc_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_369844, str_369845)
# Adding element type (line 6)
str_369846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'str', 'isspmatrix_csc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_369844, str_369846)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_369844)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_369847 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_369847) is not StypyTypeError):

    if (import_369847 != 'pyd_module'):
        __import__(import_369847)
        sys_modules_369848 = sys.modules[import_369847]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_369848.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_369847)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.base import spmatrix' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_369849 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base')

if (type(import_369849) is not StypyTypeError):

    if (import_369849 != 'pyd_module'):
        __import__(import_369849)
        sys_modules_369850 = sys.modules[import_369849]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base', sys_modules_369850.module_type_store, module_type_store, ['spmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_369850, sys_modules_369850.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import spmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base', None, module_type_store, ['spmatrix'], [spmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base', import_369849)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse._sparsetools import csc_tocsr' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_369851 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse._sparsetools')

if (type(import_369851) is not StypyTypeError):

    if (import_369851 != 'pyd_module'):
        __import__(import_369851)
        sys_modules_369852 = sys.modules[import_369851]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse._sparsetools', sys_modules_369852.module_type_store, module_type_store, ['csc_tocsr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_369852, sys_modules_369852.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import csc_tocsr

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['csc_tocsr'], [csc_tocsr])

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse._sparsetools', import_369851)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse import _sparsetools' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_369853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse')

if (type(import_369853) is not StypyTypeError):

    if (import_369853 != 'pyd_module'):
        __import__(import_369853)
        sys_modules_369854 = sys.modules[import_369853]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', sys_modules_369854.module_type_store, module_type_store, ['_sparsetools'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_369854, sys_modules_369854.module_type_store, module_type_store)
    else:
        from scipy.sparse import _sparsetools

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', None, module_type_store, ['_sparsetools'], [_sparsetools])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse', import_369853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.sparse.sputils import upcast, isintlike, IndexMixin, get_index_dtype' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_369855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils')

if (type(import_369855) is not StypyTypeError):

    if (import_369855 != 'pyd_module'):
        __import__(import_369855)
        sys_modules_369856 = sys.modules[import_369855]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', sys_modules_369856.module_type_store, module_type_store, ['upcast', 'isintlike', 'IndexMixin', 'get_index_dtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_369856, sys_modules_369856.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import upcast, isintlike, IndexMixin, get_index_dtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', None, module_type_store, ['upcast', 'isintlike', 'IndexMixin', 'get_index_dtype'], [upcast, isintlike, IndexMixin, get_index_dtype])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', import_369855)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse.compressed import _cs_matrix' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_369857 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.compressed')

if (type(import_369857) is not StypyTypeError):

    if (import_369857 != 'pyd_module'):
        __import__(import_369857)
        sys_modules_369858 = sys.modules[import_369857]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.compressed', sys_modules_369858.module_type_store, module_type_store, ['_cs_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_369858, sys_modules_369858.module_type_store, module_type_store)
    else:
        from scipy.sparse.compressed import _cs_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.compressed', None, module_type_store, ['_cs_matrix'], [_cs_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.compressed' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.compressed', import_369857)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

# Declaration of the 'csc_matrix' class
# Getting the type of '_cs_matrix' (line 19)
_cs_matrix_369859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), '_cs_matrix')
# Getting the type of 'IndexMixin' (line 19)
IndexMixin_369860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'IndexMixin')

class csc_matrix(_cs_matrix_369859, IndexMixin_369860, ):
    str_369861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', "\n    Compressed Sparse Column matrix\n\n    This can be instantiated in several ways:\n\n        csc_matrix(D)\n            with a dense matrix or rank-2 ndarray D\n\n        csc_matrix(S)\n            with another sparse matrix S (equivalent to S.tocsc())\n\n        csc_matrix((M, N), [dtype])\n            to construct an empty matrix with shape (M, N)\n            dtype is optional, defaulting to dtype='d'.\n\n        csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])\n            where ``data``, ``row_ind`` and ``col_ind`` satisfy the\n            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.\n\n        csc_matrix((data, indices, indptr), [shape=(M, N)])\n            is the standard CSC representation where the row indices for\n            column i are stored in ``indices[indptr[i]:indptr[i+1]]``\n            and their corresponding values are stored in\n            ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is\n            not supplied, the matrix dimensions are inferred from\n            the index arrays.\n\n    Attributes\n    ----------\n    dtype : dtype\n        Data type of the matrix\n    shape : 2-tuple\n        Shape of the matrix\n    ndim : int\n        Number of dimensions (this is always 2)\n    nnz\n        Number of nonzero elements\n    data\n        Data array of the matrix\n    indices\n        CSC format index array\n    indptr\n        CSC format index pointer array\n    has_sorted_indices\n        Whether indices are sorted\n\n    Notes\n    -----\n\n    Sparse matrices can be used in arithmetic operations: they support\n    addition, subtraction, multiplication, division, and matrix power.\n\n    Advantages of the CSC format\n        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.\n        - efficient column slicing\n        - fast matrix vector products (CSR, BSR may be faster)\n\n    Disadvantages of the CSC format\n      - slow row slicing operations (consider CSR)\n      - changes to the sparsity structure are expensive (consider LIL or DOK)\n\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from scipy.sparse import csc_matrix\n    >>> csc_matrix((3, 4), dtype=np.int8).toarray()\n    array([[0, 0, 0, 0],\n           [0, 0, 0, 0],\n           [0, 0, 0, 0]], dtype=int8)\n\n    >>> row = np.array([0, 2, 2, 0, 1, 2])\n    >>> col = np.array([0, 0, 1, 2, 2, 2])\n    >>> data = np.array([1, 2, 3, 4, 5, 6])\n    >>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()\n    array([[1, 0, 4],\n           [0, 0, 5],\n           [2, 3, 6]])\n\n    >>> indptr = np.array([0, 2, 3, 6])\n    >>> indices = np.array([0, 2, 2, 0, 1, 2])\n    >>> data = np.array([1, 2, 3, 4, 5, 6])\n    >>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()\n    array([[1, 0, 4],\n           [0, 0, 5],\n           [2, 3, 6]])\n\n    ")
    
    # Assigning a Str to a Name (line 109):

    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 111)
        None_369862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'None')
        # Getting the type of 'False' (line 111)
        False_369863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), 'False')
        defaults = [None_369862, False_369863]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.transpose.__dict__.__setitem__('stypy_function_name', 'csc_matrix.transpose')
        csc_matrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        csc_matrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transpose', localization, ['axes', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transpose(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 112)
        # Getting the type of 'axes' (line 112)
        axes_369864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'axes')
        # Getting the type of 'None' (line 112)
        None_369865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'None')
        
        (may_be_369866, more_types_in_union_369867) = may_not_be_none(axes_369864, None_369865)

        if may_be_369866:

            if more_types_in_union_369867:
                # Runtime conditional SSA (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 113)
            # Processing the call arguments (line 113)
            str_369869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 30), 'str', "Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
            # Processing the call keyword arguments (line 113)
            kwargs_369870 = {}
            # Getting the type of 'ValueError' (line 113)
            ValueError_369868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 113)
            ValueError_call_result_369871 = invoke(stypy.reporting.localization.Localization(__file__, 113, 18), ValueError_369868, *[str_369869], **kwargs_369870)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 113, 12), ValueError_call_result_369871, 'raise parameter', BaseException)

            if more_types_in_union_369867:
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 117):
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_369872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Getting the type of 'self' (line 117)
        self_369873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'self')
        # Obtaining the member 'shape' of a type (line 117)
        shape_369874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), self_369873, 'shape')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___369875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), shape_369874, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_369876 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___369875, int_369872)
        
        # Assigning a type to the variable 'tuple_var_assignment_369830' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_369830', subscript_call_result_369876)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_369877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Getting the type of 'self' (line 117)
        self_369878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'self')
        # Obtaining the member 'shape' of a type (line 117)
        shape_369879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), self_369878, 'shape')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___369880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), shape_369879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_369881 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___369880, int_369877)
        
        # Assigning a type to the variable 'tuple_var_assignment_369831' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_369831', subscript_call_result_369881)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_369830' (line 117)
        tuple_var_assignment_369830_369882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_369830')
        # Assigning a type to the variable 'M' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'M', tuple_var_assignment_369830_369882)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_369831' (line 117)
        tuple_var_assignment_369831_369883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_369831')
        # Assigning a type to the variable 'N' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'N', tuple_var_assignment_369831_369883)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 119, 8))
        
        # 'from scipy.sparse.csr import csr_matrix' statement (line 119)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_369884 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 119, 8), 'scipy.sparse.csr')

        if (type(import_369884) is not StypyTypeError):

            if (import_369884 != 'pyd_module'):
                __import__(import_369884)
                sys_modules_369885 = sys.modules[import_369884]
                import_from_module(stypy.reporting.localization.Localization(__file__, 119, 8), 'scipy.sparse.csr', sys_modules_369885.module_type_store, module_type_store, ['csr_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 119, 8), __file__, sys_modules_369885, sys_modules_369885.module_type_store, module_type_store)
            else:
                from scipy.sparse.csr import csr_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 119, 8), 'scipy.sparse.csr', None, module_type_store, ['csr_matrix'], [csr_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csr' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'scipy.sparse.csr', import_369884)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Call to csr_matrix(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_369887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        # Getting the type of 'self' (line 120)
        self_369888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 120)
        data_369889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), self_369888, 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 27), tuple_369887, data_369889)
        # Adding element type (line 120)
        # Getting the type of 'self' (line 120)
        self_369890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 38), 'self', False)
        # Obtaining the member 'indices' of a type (line 120)
        indices_369891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 38), self_369890, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 27), tuple_369887, indices_369891)
        # Adding element type (line 120)
        # Getting the type of 'self' (line 121)
        self_369892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'self', False)
        # Obtaining the member 'indptr' of a type (line 121)
        indptr_369893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 27), self_369892, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 27), tuple_369887, indptr_369893)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_369894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        # Getting the type of 'N' (line 121)
        N_369895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 42), tuple_369894, N_369895)
        # Adding element type (line 121)
        # Getting the type of 'M' (line 121)
        M_369896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 45), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 42), tuple_369894, M_369896)
        
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'copy' (line 121)
        copy_369897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 54), 'copy', False)
        keyword_369898 = copy_369897
        kwargs_369899 = {'copy': keyword_369898}
        # Getting the type of 'csr_matrix' (line 120)
        csr_matrix_369886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 120)
        csr_matrix_call_result_369900 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), csr_matrix_369886, *[tuple_369887, tuple_369894], **kwargs_369899)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', csr_matrix_call_result_369900)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_369901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369901)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_369901

    
    # Assigning a Attribute to a Attribute (line 123):

    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.__iter__.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_function_name', 'csc_matrix.__iter__')
        csc_matrix.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        csc_matrix.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        
        # Call to tocsr(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_369904 = {}
        # Getting the type of 'self' (line 126)
        self_369902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 126)
        tocsr_369903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), self_369902, 'tocsr')
        # Calling tocsr(args, kwargs) (line 126)
        tocsr_call_result_369905 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), tocsr_369903, *[], **kwargs_369904)
        
        # Testing the type of a for loop iterable (line 126)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 8), tocsr_call_result_369905)
        # Getting the type of the for loop variable (line 126)
        for_loop_var_369906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 8), tocsr_call_result_369905)
        # Assigning a type to the variable 'r' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'r', for_loop_var_369906)
        # SSA begins for a for statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        # Getting the type of 'r' (line 127)
        r_369907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'r')
        GeneratorType_369908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 12), GeneratorType_369908, r_369907)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'stypy_return_type', GeneratorType_369908)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_369909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_369909


    @norecursion
    def tocsc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 129)
        False_369910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'False')
        defaults = [False_369910]
        # Create a new context for function 'tocsc'
        module_type_store = module_type_store.open_function_context('tocsc', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.tocsc.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_function_name', 'csc_matrix.tocsc')
        csc_matrix.tocsc.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        csc_matrix.tocsc.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.tocsc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.tocsc', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocsc', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocsc(...)' code ##################

        
        # Getting the type of 'copy' (line 130)
        copy_369911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'copy')
        # Testing the type of an if condition (line 130)
        if_condition_369912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), copy_369911)
        # Assigning a type to the variable 'if_condition_369912' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_369912', if_condition_369912)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_369915 = {}
        # Getting the type of 'self' (line 131)
        self_369913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 131)
        copy_369914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 19), self_369913, 'copy')
        # Calling copy(args, kwargs) (line 131)
        copy_call_result_369916 = invoke(stypy.reporting.localization.Localization(__file__, 131, 19), copy_369914, *[], **kwargs_369915)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', copy_call_result_369916)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 133)
        self_369917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'stypy_return_type', self_369917)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tocsc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsc' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_369918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsc'
        return stypy_return_type_369918

    
    # Assigning a Attribute to a Attribute (line 135):

    @norecursion
    def tocsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 137)
        False_369919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'False')
        defaults = [False_369919]
        # Create a new context for function 'tocsr'
        module_type_store = module_type_store.open_function_context('tocsr', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.tocsr.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_function_name', 'csc_matrix.tocsr')
        csc_matrix.tocsr.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        csc_matrix.tocsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.tocsr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.tocsr', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tocsr', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tocsr(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 138):
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_369920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 8), 'int')
        # Getting the type of 'self' (line 138)
        self_369921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 14), 'self')
        # Obtaining the member 'shape' of a type (line 138)
        shape_369922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 14), self_369921, 'shape')
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___369923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), shape_369922, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_369924 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), getitem___369923, int_369920)
        
        # Assigning a type to the variable 'tuple_var_assignment_369832' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'tuple_var_assignment_369832', subscript_call_result_369924)
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_369925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 8), 'int')
        # Getting the type of 'self' (line 138)
        self_369926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 14), 'self')
        # Obtaining the member 'shape' of a type (line 138)
        shape_369927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 14), self_369926, 'shape')
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___369928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), shape_369927, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_369929 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), getitem___369928, int_369925)
        
        # Assigning a type to the variable 'tuple_var_assignment_369833' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'tuple_var_assignment_369833', subscript_call_result_369929)
        
        # Assigning a Name to a Name (line 138):
        # Getting the type of 'tuple_var_assignment_369832' (line 138)
        tuple_var_assignment_369832_369930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'tuple_var_assignment_369832')
        # Assigning a type to the variable 'M' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'M', tuple_var_assignment_369832_369930)
        
        # Assigning a Name to a Name (line 138):
        # Getting the type of 'tuple_var_assignment_369833' (line 138)
        tuple_var_assignment_369833_369931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'tuple_var_assignment_369833')
        # Assigning a type to the variable 'N' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 10), 'N', tuple_var_assignment_369833_369931)
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to get_index_dtype(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Obtaining an instance of the builtin type 'tuple' (line 139)
        tuple_369933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 139)
        # Adding element type (line 139)
        # Getting the type of 'self' (line 139)
        self_369934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 37), 'self', False)
        # Obtaining the member 'indptr' of a type (line 139)
        indptr_369935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 37), self_369934, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 37), tuple_369933, indptr_369935)
        # Adding element type (line 139)
        # Getting the type of 'self' (line 139)
        self_369936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'self', False)
        # Obtaining the member 'indices' of a type (line 139)
        indices_369937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 50), self_369936, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 37), tuple_369933, indices_369937)
        
        # Processing the call keyword arguments (line 139)
        
        # Call to max(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_369939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'self', False)
        # Obtaining the member 'nnz' of a type (line 140)
        nnz_369940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 47), self_369939, 'nnz')
        # Getting the type of 'N' (line 140)
        N_369941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 57), 'N', False)
        # Processing the call keyword arguments (line 140)
        kwargs_369942 = {}
        # Getting the type of 'max' (line 140)
        max_369938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'max', False)
        # Calling max(args, kwargs) (line 140)
        max_call_result_369943 = invoke(stypy.reporting.localization.Localization(__file__, 140, 43), max_369938, *[nnz_369940, N_369941], **kwargs_369942)
        
        keyword_369944 = max_call_result_369943
        kwargs_369945 = {'maxval': keyword_369944}
        # Getting the type of 'get_index_dtype' (line 139)
        get_index_dtype_369932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 139)
        get_index_dtype_call_result_369946 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), get_index_dtype_369932, *[tuple_369933], **kwargs_369945)
        
        # Assigning a type to the variable 'idx_dtype' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'idx_dtype', get_index_dtype_call_result_369946)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to empty(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'M' (line 141)
        M_369949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'M', False)
        int_369950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'int')
        # Applying the binary operator '+' (line 141)
        result_add_369951 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 26), '+', M_369949, int_369950)
        
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'idx_dtype' (line 141)
        idx_dtype_369952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 39), 'idx_dtype', False)
        keyword_369953 = idx_dtype_369952
        kwargs_369954 = {'dtype': keyword_369953}
        # Getting the type of 'np' (line 141)
        np_369947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 141)
        empty_369948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 17), np_369947, 'empty')
        # Calling empty(args, kwargs) (line 141)
        empty_call_result_369955 = invoke(stypy.reporting.localization.Localization(__file__, 141, 17), empty_369948, *[result_add_369951], **kwargs_369954)
        
        # Assigning a type to the variable 'indptr' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'indptr', empty_call_result_369955)
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to empty(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_369958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'self', False)
        # Obtaining the member 'nnz' of a type (line 142)
        nnz_369959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 27), self_369958, 'nnz')
        # Processing the call keyword arguments (line 142)
        # Getting the type of 'idx_dtype' (line 142)
        idx_dtype_369960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 43), 'idx_dtype', False)
        keyword_369961 = idx_dtype_369960
        kwargs_369962 = {'dtype': keyword_369961}
        # Getting the type of 'np' (line 142)
        np_369956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'np', False)
        # Obtaining the member 'empty' of a type (line 142)
        empty_369957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 18), np_369956, 'empty')
        # Calling empty(args, kwargs) (line 142)
        empty_call_result_369963 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), empty_369957, *[nnz_369959], **kwargs_369962)
        
        # Assigning a type to the variable 'indices' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'indices', empty_call_result_369963)
        
        # Assigning a Call to a Name (line 143):
        
        # Assigning a Call to a Name (line 143):
        
        # Call to empty(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'self' (line 143)
        self_369966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'self', False)
        # Obtaining the member 'nnz' of a type (line 143)
        nnz_369967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 24), self_369966, 'nnz')
        # Processing the call keyword arguments (line 143)
        
        # Call to upcast(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'self' (line 143)
        self_369969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 143)
        dtype_369970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 47), self_369969, 'dtype')
        # Processing the call keyword arguments (line 143)
        kwargs_369971 = {}
        # Getting the type of 'upcast' (line 143)
        upcast_369968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 40), 'upcast', False)
        # Calling upcast(args, kwargs) (line 143)
        upcast_call_result_369972 = invoke(stypy.reporting.localization.Localization(__file__, 143, 40), upcast_369968, *[dtype_369970], **kwargs_369971)
        
        keyword_369973 = upcast_call_result_369972
        kwargs_369974 = {'dtype': keyword_369973}
        # Getting the type of 'np' (line 143)
        np_369964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'np', False)
        # Obtaining the member 'empty' of a type (line 143)
        empty_369965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), np_369964, 'empty')
        # Calling empty(args, kwargs) (line 143)
        empty_call_result_369975 = invoke(stypy.reporting.localization.Localization(__file__, 143, 15), empty_369965, *[nnz_369967], **kwargs_369974)
        
        # Assigning a type to the variable 'data' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'data', empty_call_result_369975)
        
        # Call to csc_tocsr(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'M' (line 145)
        M_369977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'M', False)
        # Getting the type of 'N' (line 145)
        N_369978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 'N', False)
        
        # Call to astype(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'idx_dtype' (line 146)
        idx_dtype_369982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'idx_dtype', False)
        # Processing the call keyword arguments (line 146)
        kwargs_369983 = {}
        # Getting the type of 'self' (line 146)
        self_369979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'self', False)
        # Obtaining the member 'indptr' of a type (line 146)
        indptr_369980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 18), self_369979, 'indptr')
        # Obtaining the member 'astype' of a type (line 146)
        astype_369981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 18), indptr_369980, 'astype')
        # Calling astype(args, kwargs) (line 146)
        astype_call_result_369984 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), astype_369981, *[idx_dtype_369982], **kwargs_369983)
        
        
        # Call to astype(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'idx_dtype' (line 147)
        idx_dtype_369988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'idx_dtype', False)
        # Processing the call keyword arguments (line 147)
        kwargs_369989 = {}
        # Getting the type of 'self' (line 147)
        self_369985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'self', False)
        # Obtaining the member 'indices' of a type (line 147)
        indices_369986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 18), self_369985, 'indices')
        # Obtaining the member 'astype' of a type (line 147)
        astype_369987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 18), indices_369986, 'astype')
        # Calling astype(args, kwargs) (line 147)
        astype_call_result_369990 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), astype_369987, *[idx_dtype_369988], **kwargs_369989)
        
        # Getting the type of 'self' (line 148)
        self_369991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 18), 'self', False)
        # Obtaining the member 'data' of a type (line 148)
        data_369992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 18), self_369991, 'data')
        # Getting the type of 'indptr' (line 149)
        indptr_369993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'indptr', False)
        # Getting the type of 'indices' (line 150)
        indices_369994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'indices', False)
        # Getting the type of 'data' (line 151)
        data_369995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'data', False)
        # Processing the call keyword arguments (line 145)
        kwargs_369996 = {}
        # Getting the type of 'csc_tocsr' (line 145)
        csc_tocsr_369976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'csc_tocsr', False)
        # Calling csc_tocsr(args, kwargs) (line 145)
        csc_tocsr_call_result_369997 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), csc_tocsr_369976, *[M_369977, N_369978, astype_call_result_369984, astype_call_result_369990, data_369992, indptr_369993, indices_369994, data_369995], **kwargs_369996)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 153, 8))
        
        # 'from scipy.sparse.csr import csr_matrix' statement (line 153)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_369998 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 153, 8), 'scipy.sparse.csr')

        if (type(import_369998) is not StypyTypeError):

            if (import_369998 != 'pyd_module'):
                __import__(import_369998)
                sys_modules_369999 = sys.modules[import_369998]
                import_from_module(stypy.reporting.localization.Localization(__file__, 153, 8), 'scipy.sparse.csr', sys_modules_369999.module_type_store, module_type_store, ['csr_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 153, 8), __file__, sys_modules_369999, sys_modules_369999.module_type_store, module_type_store)
            else:
                from scipy.sparse.csr import csr_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 153, 8), 'scipy.sparse.csr', None, module_type_store, ['csr_matrix'], [csr_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csr' (line 153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'scipy.sparse.csr', import_369998)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to csr_matrix(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_370001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        # Getting the type of 'data' (line 154)
        data_370002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), tuple_370001, data_370002)
        # Adding element type (line 154)
        # Getting the type of 'indices' (line 154)
        indices_370003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), tuple_370001, indices_370003)
        # Adding element type (line 154)
        # Getting the type of 'indptr' (line 154)
        indptr_370004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), tuple_370001, indptr_370004)
        
        # Processing the call keyword arguments (line 154)
        # Getting the type of 'self' (line 154)
        self_370005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 154)
        shape_370006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 54), self_370005, 'shape')
        keyword_370007 = shape_370006
        # Getting the type of 'False' (line 154)
        False_370008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 71), 'False', False)
        keyword_370009 = False_370008
        kwargs_370010 = {'shape': keyword_370007, 'copy': keyword_370009}
        # Getting the type of 'csr_matrix' (line 154)
        csr_matrix_370000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 154)
        csr_matrix_call_result_370011 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), csr_matrix_370000, *[tuple_370001], **kwargs_370010)
        
        # Assigning a type to the variable 'A' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'A', csr_matrix_call_result_370011)
        
        # Assigning a Name to a Attribute (line 155):
        
        # Assigning a Name to a Attribute (line 155):
        # Getting the type of 'True' (line 155)
        True_370012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 31), 'True')
        # Getting the type of 'A' (line 155)
        A_370013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'A')
        # Setting the type of the member 'has_sorted_indices' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), A_370013, 'has_sorted_indices', True_370012)
        # Getting the type of 'A' (line 156)
        A_370014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'A')
        # Assigning a type to the variable 'stypy_return_type' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', A_370014)
        
        # ################# End of 'tocsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsr' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_370015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370015)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsr'
        return stypy_return_type_370015

    
    # Assigning a Attribute to a Attribute (line 158):

    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_function_name', 'csc_matrix.__getitem__')
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Assigning a Call to a Tuple (line 163):
        
        # Assigning a Subscript to a Name (line 163):
        
        # Obtaining the type of the subscript
        int_370016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 8), 'int')
        
        # Call to _unpack_index(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'key' (line 163)
        key_370019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 38), 'key', False)
        # Processing the call keyword arguments (line 163)
        kwargs_370020 = {}
        # Getting the type of 'self' (line 163)
        self_370017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 163)
        _unpack_index_370018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), self_370017, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 163)
        _unpack_index_call_result_370021 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), _unpack_index_370018, *[key_370019], **kwargs_370020)
        
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___370022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), _unpack_index_call_result_370021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_370023 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), getitem___370022, int_370016)
        
        # Assigning a type to the variable 'tuple_var_assignment_369834' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_369834', subscript_call_result_370023)
        
        # Assigning a Subscript to a Name (line 163):
        
        # Obtaining the type of the subscript
        int_370024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 8), 'int')
        
        # Call to _unpack_index(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'key' (line 163)
        key_370027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 38), 'key', False)
        # Processing the call keyword arguments (line 163)
        kwargs_370028 = {}
        # Getting the type of 'self' (line 163)
        self_370025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 163)
        _unpack_index_370026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), self_370025, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 163)
        _unpack_index_call_result_370029 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), _unpack_index_370026, *[key_370027], **kwargs_370028)
        
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___370030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), _unpack_index_call_result_370029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_370031 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), getitem___370030, int_370024)
        
        # Assigning a type to the variable 'tuple_var_assignment_369835' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_369835', subscript_call_result_370031)
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'tuple_var_assignment_369834' (line 163)
        tuple_var_assignment_369834_370032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_369834')
        # Assigning a type to the variable 'row' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'row', tuple_var_assignment_369834_370032)
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'tuple_var_assignment_369835' (line 163)
        tuple_var_assignment_369835_370033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'tuple_var_assignment_369835')
        # Assigning a type to the variable 'col' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 13), 'col', tuple_var_assignment_369835_370033)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'row' (line 165)
        row_370035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'row', False)
        # Getting the type of 'slice' (line 165)
        slice_370036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'slice', False)
        # Processing the call keyword arguments (line 165)
        kwargs_370037 = {}
        # Getting the type of 'isinstance' (line 165)
        isinstance_370034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 165)
        isinstance_call_result_370038 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), isinstance_370034, *[row_370035, slice_370036], **kwargs_370037)
        
        
        # Call to isinstance(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'col' (line 165)
        col_370040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 49), 'col', False)
        # Getting the type of 'slice' (line 165)
        slice_370041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 54), 'slice', False)
        # Processing the call keyword arguments (line 165)
        kwargs_370042 = {}
        # Getting the type of 'isinstance' (line 165)
        isinstance_370039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 38), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 165)
        isinstance_call_result_370043 = invoke(stypy.reporting.localization.Localization(__file__, 165, 38), isinstance_370039, *[col_370040, slice_370041], **kwargs_370042)
        
        # Applying the binary operator 'or' (line 165)
        result_or_keyword_370044 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 12), 'or', isinstance_call_result_370038, isinstance_call_result_370043)
        
        # Call to isintlike(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'row' (line 166)
        row_370046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 26), 'row', False)
        # Processing the call keyword arguments (line 166)
        kwargs_370047 = {}
        # Getting the type of 'isintlike' (line 166)
        isintlike_370045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 166)
        isintlike_call_result_370048 = invoke(stypy.reporting.localization.Localization(__file__, 166, 16), isintlike_370045, *[row_370046], **kwargs_370047)
        
        # Applying the binary operator 'or' (line 165)
        result_or_keyword_370049 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 12), 'or', result_or_keyword_370044, isintlike_call_result_370048)
        
        # Call to isintlike(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'col' (line 166)
        col_370051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 44), 'col', False)
        # Processing the call keyword arguments (line 166)
        kwargs_370052 = {}
        # Getting the type of 'isintlike' (line 166)
        isintlike_370050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 166)
        isintlike_call_result_370053 = invoke(stypy.reporting.localization.Localization(__file__, 166, 34), isintlike_370050, *[col_370051], **kwargs_370052)
        
        # Applying the binary operator 'or' (line 165)
        result_or_keyword_370054 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 12), 'or', result_or_keyword_370049, isintlike_call_result_370053)
        
        # Testing the type of an if condition (line 165)
        if_condition_370055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), result_or_keyword_370054)
        # Assigning a type to the variable 'if_condition_370055' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_370055', if_condition_370055)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_370056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        # Getting the type of 'col' (line 167)
        col_370057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 26), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 26), tuple_370056, col_370057)
        # Adding element type (line 167)
        # Getting the type of 'row' (line 167)
        row_370058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 26), tuple_370056, row_370058)
        
        # Getting the type of 'self' (line 167)
        self_370059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'self')
        # Obtaining the member 'T' of a type (line 167)
        T_370060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 19), self_370059, 'T')
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___370061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 19), T_370060, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_370062 = invoke(stypy.reporting.localization.Localization(__file__, 167, 19), getitem___370061, tuple_370056)
        
        # Obtaining the member 'T' of a type (line 167)
        T_370063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 19), subscript_call_result_370062, 'T')
        # Assigning a type to the variable 'stypy_return_type' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'stypy_return_type', T_370063)
        # SSA branch for the else part of an if statement (line 165)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 170)
        tuple_370064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 170)
        # Adding element type (line 170)
        # Getting the type of 'col' (line 170)
        col_370065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 26), tuple_370064, col_370065)
        # Adding element type (line 170)
        # Getting the type of 'row' (line 170)
        row_370066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 26), tuple_370064, row_370066)
        
        # Getting the type of 'self' (line 170)
        self_370067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'self')
        # Obtaining the member 'T' of a type (line 170)
        T_370068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 19), self_370067, 'T')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___370069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 19), T_370068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_370070 = invoke(stypy.reporting.localization.Localization(__file__, 170, 19), getitem___370069, tuple_370064)
        
        # Assigning a type to the variable 'stypy_return_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'stypy_return_type', subscript_call_result_370070)
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_370071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370071)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_370071


    @norecursion
    def nonzero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'nonzero'
        module_type_store = module_type_store.open_function_context('nonzero', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.nonzero.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_function_name', 'csc_matrix.nonzero')
        csc_matrix.nonzero.__dict__.__setitem__('stypy_param_names_list', [])
        csc_matrix.nonzero.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.nonzero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.nonzero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'nonzero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'nonzero(...)' code ##################

        
        # Assigning a Call to a Tuple (line 177):
        
        # Assigning a Subscript to a Name (line 177):
        
        # Obtaining the type of the subscript
        int_370072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        
        # Call to _swap(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'self' (line 177)
        self_370075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 42), 'self', False)
        # Obtaining the member 'shape' of a type (line 177)
        shape_370076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 42), self_370075, 'shape')
        # Processing the call keyword arguments (line 177)
        kwargs_370077 = {}
        # Getting the type of 'self' (line 177)
        self_370073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'self', False)
        # Obtaining the member '_swap' of a type (line 177)
        _swap_370074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 31), self_370073, '_swap')
        # Calling _swap(args, kwargs) (line 177)
        _swap_call_result_370078 = invoke(stypy.reporting.localization.Localization(__file__, 177, 31), _swap_370074, *[shape_370076], **kwargs_370077)
        
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___370079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), _swap_call_result_370078, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_370080 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), getitem___370079, int_370072)
        
        # Assigning a type to the variable 'tuple_var_assignment_369836' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_369836', subscript_call_result_370080)
        
        # Assigning a Subscript to a Name (line 177):
        
        # Obtaining the type of the subscript
        int_370081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 8), 'int')
        
        # Call to _swap(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'self' (line 177)
        self_370084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 42), 'self', False)
        # Obtaining the member 'shape' of a type (line 177)
        shape_370085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 42), self_370084, 'shape')
        # Processing the call keyword arguments (line 177)
        kwargs_370086 = {}
        # Getting the type of 'self' (line 177)
        self_370082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'self', False)
        # Obtaining the member '_swap' of a type (line 177)
        _swap_370083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 31), self_370082, '_swap')
        # Calling _swap(args, kwargs) (line 177)
        _swap_call_result_370087 = invoke(stypy.reporting.localization.Localization(__file__, 177, 31), _swap_370083, *[shape_370085], **kwargs_370086)
        
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___370088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), _swap_call_result_370087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_370089 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), getitem___370088, int_370081)
        
        # Assigning a type to the variable 'tuple_var_assignment_369837' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_369837', subscript_call_result_370089)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'tuple_var_assignment_369836' (line 177)
        tuple_var_assignment_369836_370090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_369836')
        # Assigning a type to the variable 'major_dim' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'major_dim', tuple_var_assignment_369836_370090)
        
        # Assigning a Name to a Name (line 177):
        # Getting the type of 'tuple_var_assignment_369837' (line 177)
        tuple_var_assignment_369837_370091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'tuple_var_assignment_369837')
        # Assigning a type to the variable 'minor_dim' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'minor_dim', tuple_var_assignment_369837_370091)
        
        # Assigning a Attribute to a Name (line 178):
        
        # Assigning a Attribute to a Name (line 178):
        # Getting the type of 'self' (line 178)
        self_370092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'self')
        # Obtaining the member 'indices' of a type (line 178)
        indices_370093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 24), self_370092, 'indices')
        # Assigning a type to the variable 'minor_indices' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'minor_indices', indices_370093)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to empty(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Call to len(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'minor_indices' (line 179)
        minor_indices_370097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 37), 'minor_indices', False)
        # Processing the call keyword arguments (line 179)
        kwargs_370098 = {}
        # Getting the type of 'len' (line 179)
        len_370096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 33), 'len', False)
        # Calling len(args, kwargs) (line 179)
        len_call_result_370099 = invoke(stypy.reporting.localization.Localization(__file__, 179, 33), len_370096, *[minor_indices_370097], **kwargs_370098)
        
        # Processing the call keyword arguments (line 179)
        # Getting the type of 'self' (line 179)
        self_370100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 59), 'self', False)
        # Obtaining the member 'indices' of a type (line 179)
        indices_370101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 59), self_370100, 'indices')
        # Obtaining the member 'dtype' of a type (line 179)
        dtype_370102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 59), indices_370101, 'dtype')
        keyword_370103 = dtype_370102
        kwargs_370104 = {'dtype': keyword_370103}
        # Getting the type of 'np' (line 179)
        np_370094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'np', False)
        # Obtaining the member 'empty' of a type (line 179)
        empty_370095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 24), np_370094, 'empty')
        # Calling empty(args, kwargs) (line 179)
        empty_call_result_370105 = invoke(stypy.reporting.localization.Localization(__file__, 179, 24), empty_370095, *[len_call_result_370099], **kwargs_370104)
        
        # Assigning a type to the variable 'major_indices' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'major_indices', empty_call_result_370105)
        
        # Call to expandptr(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'major_dim' (line 180)
        major_dim_370108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'major_dim', False)
        # Getting the type of 'self' (line 180)
        self_370109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 42), 'self', False)
        # Obtaining the member 'indptr' of a type (line 180)
        indptr_370110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 42), self_370109, 'indptr')
        # Getting the type of 'major_indices' (line 180)
        major_indices_370111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 55), 'major_indices', False)
        # Processing the call keyword arguments (line 180)
        kwargs_370112 = {}
        # Getting the type of '_sparsetools' (line 180)
        _sparsetools_370106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), '_sparsetools', False)
        # Obtaining the member 'expandptr' of a type (line 180)
        expandptr_370107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), _sparsetools_370106, 'expandptr')
        # Calling expandptr(args, kwargs) (line 180)
        expandptr_call_result_370113 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), expandptr_370107, *[major_dim_370108, indptr_370110, major_indices_370111], **kwargs_370112)
        
        
        # Assigning a Call to a Tuple (line 181):
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_370114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'int')
        
        # Call to _swap(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_370117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        # Getting the type of 'major_indices' (line 181)
        major_indices_370118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'major_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 31), tuple_370117, major_indices_370118)
        # Adding element type (line 181)
        # Getting the type of 'minor_indices' (line 181)
        minor_indices_370119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'minor_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 31), tuple_370117, minor_indices_370119)
        
        # Processing the call keyword arguments (line 181)
        kwargs_370120 = {}
        # Getting the type of 'self' (line 181)
        self_370115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'self', False)
        # Obtaining the member '_swap' of a type (line 181)
        _swap_370116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 19), self_370115, '_swap')
        # Calling _swap(args, kwargs) (line 181)
        _swap_call_result_370121 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), _swap_370116, *[tuple_370117], **kwargs_370120)
        
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___370122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), _swap_call_result_370121, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_370123 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), getitem___370122, int_370114)
        
        # Assigning a type to the variable 'tuple_var_assignment_369838' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_369838', subscript_call_result_370123)
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_370124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'int')
        
        # Call to _swap(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_370127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        # Getting the type of 'major_indices' (line 181)
        major_indices_370128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'major_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 31), tuple_370127, major_indices_370128)
        # Adding element type (line 181)
        # Getting the type of 'minor_indices' (line 181)
        minor_indices_370129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'minor_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 31), tuple_370127, minor_indices_370129)
        
        # Processing the call keyword arguments (line 181)
        kwargs_370130 = {}
        # Getting the type of 'self' (line 181)
        self_370125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'self', False)
        # Obtaining the member '_swap' of a type (line 181)
        _swap_370126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 19), self_370125, '_swap')
        # Calling _swap(args, kwargs) (line 181)
        _swap_call_result_370131 = invoke(stypy.reporting.localization.Localization(__file__, 181, 19), _swap_370126, *[tuple_370127], **kwargs_370130)
        
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___370132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), _swap_call_result_370131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_370133 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), getitem___370132, int_370124)
        
        # Assigning a type to the variable 'tuple_var_assignment_369839' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_369839', subscript_call_result_370133)
        
        # Assigning a Name to a Name (line 181):
        # Getting the type of 'tuple_var_assignment_369838' (line 181)
        tuple_var_assignment_369838_370134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_369838')
        # Assigning a type to the variable 'row' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'row', tuple_var_assignment_369838_370134)
        
        # Assigning a Name to a Name (line 181):
        # Getting the type of 'tuple_var_assignment_369839' (line 181)
        tuple_var_assignment_369839_370135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_369839')
        # Assigning a type to the variable 'col' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'col', tuple_var_assignment_369839_370135)
        
        # Assigning a Compare to a Name (line 184):
        
        # Assigning a Compare to a Name (line 184):
        
        # Getting the type of 'self' (line 184)
        self_370136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'self')
        # Obtaining the member 'data' of a type (line 184)
        data_370137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 18), self_370136, 'data')
        int_370138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 31), 'int')
        # Applying the binary operator '!=' (line 184)
        result_ne_370139 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 18), '!=', data_370137, int_370138)
        
        # Assigning a type to the variable 'nz_mask' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'nz_mask', result_ne_370139)
        
        # Assigning a Subscript to a Name (line 185):
        
        # Assigning a Subscript to a Name (line 185):
        
        # Obtaining the type of the subscript
        # Getting the type of 'nz_mask' (line 185)
        nz_mask_370140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'nz_mask')
        # Getting the type of 'row' (line 185)
        row_370141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), 'row')
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___370142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 14), row_370141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_370143 = invoke(stypy.reporting.localization.Localization(__file__, 185, 14), getitem___370142, nz_mask_370140)
        
        # Assigning a type to the variable 'row' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'row', subscript_call_result_370143)
        
        # Assigning a Subscript to a Name (line 186):
        
        # Assigning a Subscript to a Name (line 186):
        
        # Obtaining the type of the subscript
        # Getting the type of 'nz_mask' (line 186)
        nz_mask_370144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'nz_mask')
        # Getting the type of 'col' (line 186)
        col_370145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 14), 'col')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___370146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 14), col_370145, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_370147 = invoke(stypy.reporting.localization.Localization(__file__, 186, 14), getitem___370146, nz_mask_370144)
        
        # Assigning a type to the variable 'col' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'col', subscript_call_result_370147)
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to argsort(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'row' (line 189)
        row_370150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'row', False)
        # Processing the call keyword arguments (line 189)
        str_370151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 35), 'str', 'mergesort')
        keyword_370152 = str_370151
        kwargs_370153 = {'kind': keyword_370152}
        # Getting the type of 'np' (line 189)
        np_370148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 14), 'np', False)
        # Obtaining the member 'argsort' of a type (line 189)
        argsort_370149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 14), np_370148, 'argsort')
        # Calling argsort(args, kwargs) (line 189)
        argsort_call_result_370154 = invoke(stypy.reporting.localization.Localization(__file__, 189, 14), argsort_370149, *[row_370150], **kwargs_370153)
        
        # Assigning a type to the variable 'ind' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'ind', argsort_call_result_370154)
        
        # Assigning a Subscript to a Name (line 190):
        
        # Assigning a Subscript to a Name (line 190):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 190)
        ind_370155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 18), 'ind')
        # Getting the type of 'row' (line 190)
        row_370156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 14), 'row')
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___370157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 14), row_370156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_370158 = invoke(stypy.reporting.localization.Localization(__file__, 190, 14), getitem___370157, ind_370155)
        
        # Assigning a type to the variable 'row' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'row', subscript_call_result_370158)
        
        # Assigning a Subscript to a Name (line 191):
        
        # Assigning a Subscript to a Name (line 191):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 191)
        ind_370159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 18), 'ind')
        # Getting the type of 'col' (line 191)
        col_370160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'col')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___370161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 14), col_370160, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_370162 = invoke(stypy.reporting.localization.Localization(__file__, 191, 14), getitem___370161, ind_370159)
        
        # Assigning a type to the variable 'col' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'col', subscript_call_result_370162)
        
        # Obtaining an instance of the builtin type 'tuple' (line 193)
        tuple_370163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 193)
        # Adding element type (line 193)
        # Getting the type of 'row' (line 193)
        row_370164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 15), tuple_370163, row_370164)
        # Adding element type (line 193)
        # Getting the type of 'col' (line 193)
        col_370165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 15), tuple_370163, col_370165)
        
        # Assigning a type to the variable 'stypy_return_type' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', tuple_370163)
        
        # ################# End of 'nonzero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'nonzero' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_370166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370166)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'nonzero'
        return stypy_return_type_370166

    
    # Assigning a Attribute to a Attribute (line 195):

    @norecursion
    def getrow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getrow'
        module_type_store = module_type_store.open_function_context('getrow', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.getrow.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.getrow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.getrow.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.getrow.__dict__.__setitem__('stypy_function_name', 'csc_matrix.getrow')
        csc_matrix.getrow.__dict__.__setitem__('stypy_param_names_list', ['i'])
        csc_matrix.getrow.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.getrow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.getrow.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.getrow.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.getrow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.getrow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.getrow', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getrow', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getrow(...)' code ##################

        str_370167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'str', 'Returns a copy of row i of the matrix, as a (1 x n)\n        CSR matrix (row vector).\n        ')
        
        # Call to tocsr(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_370178 = {}
        
        # Call to _get_submatrix(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'i' (line 203)
        i_370170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 35), 'i', False)
        
        # Call to slice(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'None' (line 203)
        None_370172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 44), 'None', False)
        # Processing the call keyword arguments (line 203)
        kwargs_370173 = {}
        # Getting the type of 'slice' (line 203)
        slice_370171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'slice', False)
        # Calling slice(args, kwargs) (line 203)
        slice_call_result_370174 = invoke(stypy.reporting.localization.Localization(__file__, 203, 38), slice_370171, *[None_370172], **kwargs_370173)
        
        # Processing the call keyword arguments (line 203)
        kwargs_370175 = {}
        # Getting the type of 'self' (line 203)
        self_370168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'self', False)
        # Obtaining the member '_get_submatrix' of a type (line 203)
        _get_submatrix_370169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), self_370168, '_get_submatrix')
        # Calling _get_submatrix(args, kwargs) (line 203)
        _get_submatrix_call_result_370176 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), _get_submatrix_370169, *[i_370170, slice_call_result_370174], **kwargs_370175)
        
        # Obtaining the member 'tocsr' of a type (line 203)
        tocsr_370177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), _get_submatrix_call_result_370176, 'tocsr')
        # Calling tocsr(args, kwargs) (line 203)
        tocsr_call_result_370179 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), tocsr_370177, *[], **kwargs_370178)
        
        # Assigning a type to the variable 'stypy_return_type' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', tocsr_call_result_370179)
        
        # ################# End of 'getrow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getrow' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_370180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370180)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getrow'
        return stypy_return_type_370180


    @norecursion
    def getcol(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getcol'
        module_type_store = module_type_store.open_function_context('getcol', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix.getcol.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix.getcol.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix.getcol.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix.getcol.__dict__.__setitem__('stypy_function_name', 'csc_matrix.getcol')
        csc_matrix.getcol.__dict__.__setitem__('stypy_param_names_list', ['i'])
        csc_matrix.getcol.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix.getcol.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix.getcol.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix.getcol.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix.getcol.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix.getcol.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.getcol', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getcol', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getcol(...)' code ##################

        str_370181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, (-1)), 'str', 'Returns a copy of column i of the matrix, as a (m x 1)\n        CSC matrix (column vector).\n        ')
        
        # Assigning a Attribute to a Tuple (line 209):
        
        # Assigning a Subscript to a Name (line 209):
        
        # Obtaining the type of the subscript
        int_370182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 8), 'int')
        # Getting the type of 'self' (line 209)
        self_370183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'self')
        # Obtaining the member 'shape' of a type (line 209)
        shape_370184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), self_370183, 'shape')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___370185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), shape_370184, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_370186 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), getitem___370185, int_370182)
        
        # Assigning a type to the variable 'tuple_var_assignment_369840' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_369840', subscript_call_result_370186)
        
        # Assigning a Subscript to a Name (line 209):
        
        # Obtaining the type of the subscript
        int_370187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 8), 'int')
        # Getting the type of 'self' (line 209)
        self_370188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'self')
        # Obtaining the member 'shape' of a type (line 209)
        shape_370189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), self_370188, 'shape')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___370190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), shape_370189, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_370191 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), getitem___370190, int_370187)
        
        # Assigning a type to the variable 'tuple_var_assignment_369841' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_369841', subscript_call_result_370191)
        
        # Assigning a Name to a Name (line 209):
        # Getting the type of 'tuple_var_assignment_369840' (line 209)
        tuple_var_assignment_369840_370192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_369840')
        # Assigning a type to the variable 'M' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'M', tuple_var_assignment_369840_370192)
        
        # Assigning a Name to a Name (line 209):
        # Getting the type of 'tuple_var_assignment_369841' (line 209)
        tuple_var_assignment_369841_370193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'tuple_var_assignment_369841')
        # Assigning a type to the variable 'N' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'N', tuple_var_assignment_369841_370193)
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to int(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'i' (line 210)
        i_370195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'i', False)
        # Processing the call keyword arguments (line 210)
        kwargs_370196 = {}
        # Getting the type of 'int' (line 210)
        int_370194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'int', False)
        # Calling int(args, kwargs) (line 210)
        int_call_result_370197 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), int_370194, *[i_370195], **kwargs_370196)
        
        # Assigning a type to the variable 'i' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'i', int_call_result_370197)
        
        
        # Getting the type of 'i' (line 211)
        i_370198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'i')
        int_370199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'int')
        # Applying the binary operator '<' (line 211)
        result_lt_370200 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), '<', i_370198, int_370199)
        
        # Testing the type of an if condition (line 211)
        if_condition_370201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), result_lt_370200)
        # Assigning a type to the variable 'if_condition_370201' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_370201', if_condition_370201)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'i' (line 212)
        i_370202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'i')
        # Getting the type of 'N' (line 212)
        N_370203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'N')
        # Applying the binary operator '+=' (line 212)
        result_iadd_370204 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), '+=', i_370202, N_370203)
        # Assigning a type to the variable 'i' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'i', result_iadd_370204)
        
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 213)
        i_370205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'i')
        int_370206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 15), 'int')
        # Applying the binary operator '<' (line 213)
        result_lt_370207 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 11), '<', i_370205, int_370206)
        
        
        # Getting the type of 'i' (line 213)
        i_370208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'i')
        # Getting the type of 'N' (line 213)
        N_370209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'N')
        # Applying the binary operator '>=' (line 213)
        result_ge_370210 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 20), '>=', i_370208, N_370209)
        
        # Applying the binary operator 'or' (line 213)
        result_or_keyword_370211 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 11), 'or', result_lt_370207, result_ge_370210)
        
        # Testing the type of an if condition (line 213)
        if_condition_370212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), result_or_keyword_370211)
        # Assigning a type to the variable 'if_condition_370212' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_370212', if_condition_370212)
        # SSA begins for if statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 214)
        # Processing the call arguments (line 214)
        str_370214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 29), 'str', 'index (%d) out of range')
        # Getting the type of 'i' (line 214)
        i_370215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 57), 'i', False)
        # Applying the binary operator '%' (line 214)
        result_mod_370216 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 29), '%', str_370214, i_370215)
        
        # Processing the call keyword arguments (line 214)
        kwargs_370217 = {}
        # Getting the type of 'IndexError' (line 214)
        IndexError_370213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 214)
        IndexError_call_result_370218 = invoke(stypy.reporting.localization.Localization(__file__, 214, 18), IndexError_370213, *[result_mod_370216], **kwargs_370217)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 214, 12), IndexError_call_result_370218, 'raise parameter', BaseException)
        # SSA join for if statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 215):
        
        # Assigning a Call to a Name (line 215):
        
        # Call to slice(...): (line 215)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 215)
        i_370220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 33), 'i', False)
        # Getting the type of 'i' (line 215)
        i_370221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 35), 'i', False)
        int_370222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 37), 'int')
        # Applying the binary operator '+' (line 215)
        result_add_370223 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 35), '+', i_370221, int_370222)
        
        slice_370224 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 21), i_370220, result_add_370223, None)
        # Getting the type of 'self' (line 215)
        self_370225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'self', False)
        # Obtaining the member 'indptr' of a type (line 215)
        indptr_370226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), self_370225, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___370227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), indptr_370226, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_370228 = invoke(stypy.reporting.localization.Localization(__file__, 215, 21), getitem___370227, slice_370224)
        
        # Processing the call keyword arguments (line 215)
        kwargs_370229 = {}
        # Getting the type of 'slice' (line 215)
        slice_370219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 14), 'slice', False)
        # Calling slice(args, kwargs) (line 215)
        slice_call_result_370230 = invoke(stypy.reporting.localization.Localization(__file__, 215, 14), slice_370219, *[subscript_call_result_370228], **kwargs_370229)
        
        # Assigning a type to the variable 'idx' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'idx', slice_call_result_370230)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to copy(...): (line 216)
        # Processing the call keyword arguments (line 216)
        kwargs_370237 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 216)
        idx_370231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'idx', False)
        # Getting the type of 'self' (line 216)
        self_370232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'self', False)
        # Obtaining the member 'data' of a type (line 216)
        data_370233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), self_370232, 'data')
        # Obtaining the member '__getitem__' of a type (line 216)
        getitem___370234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), data_370233, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 216)
        subscript_call_result_370235 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), getitem___370234, idx_370231)
        
        # Obtaining the member 'copy' of a type (line 216)
        copy_370236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), subscript_call_result_370235, 'copy')
        # Calling copy(args, kwargs) (line 216)
        copy_call_result_370238 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), copy_370236, *[], **kwargs_370237)
        
        # Assigning a type to the variable 'data' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'data', copy_call_result_370238)
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to copy(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_370245 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 217)
        idx_370239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'idx', False)
        # Getting the type of 'self' (line 217)
        self_370240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'self', False)
        # Obtaining the member 'indices' of a type (line 217)
        indices_370241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 18), self_370240, 'indices')
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___370242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 18), indices_370241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_370243 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), getitem___370242, idx_370239)
        
        # Obtaining the member 'copy' of a type (line 217)
        copy_370244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 18), subscript_call_result_370243, 'copy')
        # Calling copy(args, kwargs) (line 217)
        copy_call_result_370246 = invoke(stypy.reporting.localization.Localization(__file__, 217, 18), copy_370244, *[], **kwargs_370245)
        
        # Assigning a type to the variable 'indices' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'indices', copy_call_result_370246)
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to array(...): (line 218)
        # Processing the call arguments (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_370249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        int_370250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 26), list_370249, int_370250)
        # Adding element type (line 218)
        
        # Call to len(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'indices' (line 218)
        indices_370252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'indices', False)
        # Processing the call keyword arguments (line 218)
        kwargs_370253 = {}
        # Getting the type of 'len' (line 218)
        len_370251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 30), 'len', False)
        # Calling len(args, kwargs) (line 218)
        len_call_result_370254 = invoke(stypy.reporting.localization.Localization(__file__, 218, 30), len_370251, *[indices_370252], **kwargs_370253)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 26), list_370249, len_call_result_370254)
        
        # Processing the call keyword arguments (line 218)
        # Getting the type of 'self' (line 218)
        self_370255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 51), 'self', False)
        # Obtaining the member 'indptr' of a type (line 218)
        indptr_370256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 51), self_370255, 'indptr')
        # Obtaining the member 'dtype' of a type (line 218)
        dtype_370257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 51), indptr_370256, 'dtype')
        keyword_370258 = dtype_370257
        kwargs_370259 = {'dtype': keyword_370258}
        # Getting the type of 'np' (line 218)
        np_370247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 218)
        array_370248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 17), np_370247, 'array')
        # Calling array(args, kwargs) (line 218)
        array_call_result_370260 = invoke(stypy.reporting.localization.Localization(__file__, 218, 17), array_370248, *[list_370249], **kwargs_370259)
        
        # Assigning a type to the variable 'indptr' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'indptr', array_call_result_370260)
        
        # Call to csc_matrix(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Obtaining an instance of the builtin type 'tuple' (line 219)
        tuple_370262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 219)
        # Adding element type (line 219)
        # Getting the type of 'data' (line 219)
        data_370263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 27), tuple_370262, data_370263)
        # Adding element type (line 219)
        # Getting the type of 'indices' (line 219)
        indices_370264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 27), tuple_370262, indices_370264)
        # Adding element type (line 219)
        # Getting the type of 'indptr' (line 219)
        indptr_370265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 42), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 27), tuple_370262, indptr_370265)
        
        # Processing the call keyword arguments (line 219)
        
        # Obtaining an instance of the builtin type 'tuple' (line 219)
        tuple_370266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 219)
        # Adding element type (line 219)
        # Getting the type of 'M' (line 219)
        M_370267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 58), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 58), tuple_370266, M_370267)
        # Adding element type (line 219)
        int_370268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 58), tuple_370266, int_370268)
        
        keyword_370269 = tuple_370266
        # Getting the type of 'self' (line 220)
        self_370270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 220)
        dtype_370271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 32), self_370270, 'dtype')
        keyword_370272 = dtype_370271
        # Getting the type of 'False' (line 220)
        False_370273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 49), 'False', False)
        keyword_370274 = False_370273
        kwargs_370275 = {'dtype': keyword_370272, 'shape': keyword_370269, 'copy': keyword_370274}
        # Getting the type of 'csc_matrix' (line 219)
        csc_matrix_370261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 219)
        csc_matrix_call_result_370276 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), csc_matrix_370261, *[tuple_370262], **kwargs_370275)
        
        # Assigning a type to the variable 'stypy_return_type' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type', csc_matrix_call_result_370276)
        
        # ################# End of 'getcol(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getcol' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_370277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getcol'
        return stypy_return_type_370277


    @norecursion
    def _swap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_swap'
        module_type_store = module_type_store.open_function_context('_swap', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csc_matrix._swap.__dict__.__setitem__('stypy_localization', localization)
        csc_matrix._swap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csc_matrix._swap.__dict__.__setitem__('stypy_type_store', module_type_store)
        csc_matrix._swap.__dict__.__setitem__('stypy_function_name', 'csc_matrix._swap')
        csc_matrix._swap.__dict__.__setitem__('stypy_param_names_list', ['x'])
        csc_matrix._swap.__dict__.__setitem__('stypy_varargs_param_name', None)
        csc_matrix._swap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csc_matrix._swap.__dict__.__setitem__('stypy_call_defaults', defaults)
        csc_matrix._swap.__dict__.__setitem__('stypy_call_varargs', varargs)
        csc_matrix._swap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csc_matrix._swap.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix._swap', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_swap', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_swap(...)' code ##################

        str_370278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, (-1)), 'str', 'swap the members of x if this is a column-oriented matrix\n        ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 227)
        tuple_370279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 227)
        # Adding element type (line 227)
        
        # Obtaining the type of the subscript
        int_370280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 17), 'int')
        # Getting the type of 'x' (line 227)
        x_370281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'x')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___370282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 15), x_370281, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_370283 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), getitem___370282, int_370280)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), tuple_370279, subscript_call_result_370283)
        # Adding element type (line 227)
        
        # Obtaining the type of the subscript
        int_370284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 23), 'int')
        # Getting the type of 'x' (line 227)
        x_370285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'x')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___370286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 21), x_370285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_370287 = invoke(stypy.reporting.localization.Localization(__file__, 227, 21), getitem___370286, int_370284)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), tuple_370279, subscript_call_result_370287)
        
        # Assigning a type to the variable 'stypy_return_type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'stypy_return_type', tuple_370279)
        
        # ################# End of '_swap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_swap' in the type store
        # Getting the type of 'stypy_return_type' (line 224)
        stypy_return_type_370288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370288)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_swap'
        return stypy_return_type_370288


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csc_matrix.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'csc_matrix' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'csc_matrix', csc_matrix)

# Assigning a Str to a Name (line 109):
str_370289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 13), 'str', 'csc')
# Getting the type of 'csc_matrix'
csc_matrix_370290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csc_matrix')
# Setting the type of the member 'format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csc_matrix_370290, 'format', str_370289)

# Assigning a Attribute to a Attribute (line 123):
# Getting the type of 'spmatrix' (line 123)
spmatrix_370291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'spmatrix')
# Obtaining the member 'transpose' of a type (line 123)
transpose_370292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), spmatrix_370291, 'transpose')
# Obtaining the member '__doc__' of a type (line 123)
doc___370293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), transpose_370292, '__doc__')
# Getting the type of 'csc_matrix'
csc_matrix_370294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csc_matrix')
# Obtaining the member 'transpose' of a type
transpose_370295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csc_matrix_370294, 'transpose')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), transpose_370295, '__doc__', doc___370293)

# Assigning a Attribute to a Attribute (line 135):
# Getting the type of 'spmatrix' (line 135)
spmatrix_370296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 20), 'spmatrix')
# Obtaining the member 'tocsc' of a type (line 135)
tocsc_370297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 20), spmatrix_370296, 'tocsc')
# Obtaining the member '__doc__' of a type (line 135)
doc___370298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 20), tocsc_370297, '__doc__')
# Getting the type of 'csc_matrix'
csc_matrix_370299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csc_matrix')
# Obtaining the member 'tocsc' of a type
tocsc_370300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csc_matrix_370299, 'tocsc')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsc_370300, '__doc__', doc___370298)

# Assigning a Attribute to a Attribute (line 158):
# Getting the type of 'spmatrix' (line 158)
spmatrix_370301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'spmatrix')
# Obtaining the member 'tocsr' of a type (line 158)
tocsr_370302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 20), spmatrix_370301, 'tocsr')
# Obtaining the member '__doc__' of a type (line 158)
doc___370303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 20), tocsr_370302, '__doc__')
# Getting the type of 'csc_matrix'
csc_matrix_370304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csc_matrix')
# Obtaining the member 'tocsr' of a type
tocsr_370305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csc_matrix_370304, 'tocsr')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsr_370305, '__doc__', doc___370303)

# Assigning a Attribute to a Attribute (line 195):
# Getting the type of '_cs_matrix' (line 195)
_cs_matrix_370306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), '_cs_matrix')
# Obtaining the member 'nonzero' of a type (line 195)
nonzero_370307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 22), _cs_matrix_370306, 'nonzero')
# Obtaining the member '__doc__' of a type (line 195)
doc___370308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 22), nonzero_370307, '__doc__')
# Getting the type of 'csc_matrix'
csc_matrix_370309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csc_matrix')
# Obtaining the member 'nonzero' of a type
nonzero_370310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csc_matrix_370309, 'nonzero')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), nonzero_370310, '__doc__', doc___370308)

@norecursion
def isspmatrix_csc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix_csc'
    module_type_store = module_type_store.open_function_context('isspmatrix_csc', 230, 0, False)
    
    # Passed parameters checking function
    isspmatrix_csc.stypy_localization = localization
    isspmatrix_csc.stypy_type_of_self = None
    isspmatrix_csc.stypy_type_store = module_type_store
    isspmatrix_csc.stypy_function_name = 'isspmatrix_csc'
    isspmatrix_csc.stypy_param_names_list = ['x']
    isspmatrix_csc.stypy_varargs_param_name = None
    isspmatrix_csc.stypy_kwargs_param_name = None
    isspmatrix_csc.stypy_call_defaults = defaults
    isspmatrix_csc.stypy_call_varargs = varargs
    isspmatrix_csc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix_csc', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix_csc', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix_csc(...)' code ##################

    str_370311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'str', 'Is x of csc_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a csc matrix\n\n    Returns\n    -------\n    bool\n        True if x is a csc matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import csc_matrix, isspmatrix_csc\n    >>> isspmatrix_csc(csc_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc\n    >>> isspmatrix_csc(csr_matrix([[5]]))\n    False\n    ')
    
    # Call to isinstance(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'x' (line 253)
    x_370313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'x', False)
    # Getting the type of 'csc_matrix' (line 253)
    csc_matrix_370314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 25), 'csc_matrix', False)
    # Processing the call keyword arguments (line 253)
    kwargs_370315 = {}
    # Getting the type of 'isinstance' (line 253)
    isinstance_370312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 253)
    isinstance_call_result_370316 = invoke(stypy.reporting.localization.Localization(__file__, 253, 11), isinstance_370312, *[x_370313, csc_matrix_370314], **kwargs_370315)
    
    # Assigning a type to the variable 'stypy_return_type' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type', isinstance_call_result_370316)
    
    # ################# End of 'isspmatrix_csc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix_csc' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_370317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_370317)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix_csc'
    return stypy_return_type_370317

# Assigning a type to the variable 'isspmatrix_csc' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'isspmatrix_csc', isspmatrix_csc)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
