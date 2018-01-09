
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =====================================
3: Sparse matrices (:mod:`scipy.sparse`)
4: =====================================
5: 
6: .. currentmodule:: scipy.sparse
7: 
8: SciPy 2-D sparse matrix package for numeric data.
9: 
10: Contents
11: ========
12: 
13: Sparse matrix classes
14: ---------------------
15: 
16: .. autosummary::
17:    :toctree: generated/
18: 
19:    bsr_matrix - Block Sparse Row matrix
20:    coo_matrix - A sparse matrix in COOrdinate format
21:    csc_matrix - Compressed Sparse Column matrix
22:    csr_matrix - Compressed Sparse Row matrix
23:    dia_matrix - Sparse matrix with DIAgonal storage
24:    dok_matrix - Dictionary Of Keys based sparse matrix
25:    lil_matrix - Row-based linked list sparse matrix
26:    spmatrix - Sparse matrix base class
27: 
28: Functions
29: ---------
30: 
31: Building sparse matrices:
32: 
33: .. autosummary::
34:    :toctree: generated/
35: 
36:    eye - Sparse MxN matrix whose k-th diagonal is all ones
37:    identity - Identity matrix in sparse format
38:    kron - kronecker product of two sparse matrices
39:    kronsum - kronecker sum of sparse matrices
40:    diags - Return a sparse matrix from diagonals
41:    spdiags - Return a sparse matrix from diagonals
42:    block_diag - Build a block diagonal sparse matrix
43:    tril - Lower triangular portion of a matrix in sparse format
44:    triu - Upper triangular portion of a matrix in sparse format
45:    bmat - Build a sparse matrix from sparse sub-blocks
46:    hstack - Stack sparse matrices horizontally (column wise)
47:    vstack - Stack sparse matrices vertically (row wise)
48:    rand - Random values in a given shape
49:    random - Random values in a given shape
50: 
51: Save and load sparse matrices:
52: 
53: .. autosummary::
54:    :toctree: generated/
55: 
56:    save_npz - Save a sparse matrix to a file using ``.npz`` format.
57:    load_npz - Load a sparse matrix from a file using ``.npz`` format.
58: 
59: Sparse matrix tools:
60: 
61: .. autosummary::
62:    :toctree: generated/
63: 
64:    find
65: 
66: Identifying sparse matrices:
67: 
68: .. autosummary::
69:    :toctree: generated/
70: 
71:    issparse
72:    isspmatrix
73:    isspmatrix_csc
74:    isspmatrix_csr
75:    isspmatrix_bsr
76:    isspmatrix_lil
77:    isspmatrix_dok
78:    isspmatrix_coo
79:    isspmatrix_dia
80: 
81: Submodules
82: ----------
83: 
84: .. autosummary::
85:    :toctree: generated/
86: 
87:    csgraph - Compressed sparse graph routines
88:    linalg - sparse linear algebra routines
89: 
90: Exceptions
91: ----------
92: 
93: .. autosummary::
94:    :toctree: generated/
95: 
96:    SparseEfficiencyWarning
97:    SparseWarning
98: 
99: 
100: Usage information
101: =================
102: 
103: There are seven available sparse matrix types:
104: 
105:     1. csc_matrix: Compressed Sparse Column format
106:     2. csr_matrix: Compressed Sparse Row format
107:     3. bsr_matrix: Block Sparse Row format
108:     4. lil_matrix: List of Lists format
109:     5. dok_matrix: Dictionary of Keys format
110:     6. coo_matrix: COOrdinate format (aka IJV, triplet format)
111:     7. dia_matrix: DIAgonal format
112: 
113: To construct a matrix efficiently, use either dok_matrix or lil_matrix.
114: The lil_matrix class supports basic slicing and fancy indexing with a
115: similar syntax to NumPy arrays. As illustrated below, the COO format
116: may also be used to efficiently construct matrices. Despite their
117: similarity to NumPy arrays, it is **strongly discouraged** to use NumPy
118: functions directly on these matrices because NumPy may not properly convert
119: them for computations, leading to unexpected (and incorrect) results. If you
120: do want to apply a NumPy function to these matrices, first check if SciPy has
121: its own implementation for the given sparse matrix class, or **convert the
122: sparse matrix to a NumPy array** (e.g. using the `toarray()` method of the
123: class) first before applying the method.
124: 
125: To perform manipulations such as multiplication or inversion, first
126: convert the matrix to either CSC or CSR format. The lil_matrix format is
127: row-based, so conversion to CSR is efficient, whereas conversion to CSC
128: is less so.
129: 
130: All conversions among the CSR, CSC, and COO formats are efficient,
131: linear-time operations.
132: 
133: Matrix vector product
134: ---------------------
135: To do a vector product between a sparse matrix and a vector simply use
136: the matrix `dot` method, as described in its docstring:
137: 
138: >>> import numpy as np
139: >>> from scipy.sparse import csr_matrix
140: >>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
141: >>> v = np.array([1, 0, -1])
142: >>> A.dot(v)
143: array([ 1, -3, -1], dtype=int64)
144: 
145: .. warning:: As of NumPy 1.7, `np.dot` is not aware of sparse matrices,
146:   therefore using it will result on unexpected results or errors.
147:   The corresponding dense array should be obtained first instead:
148: 
149:   >>> np.dot(A.toarray(), v)
150:   array([ 1, -3, -1], dtype=int64)
151: 
152:   but then all the performance advantages would be lost.
153: 
154: The CSR format is specially suitable for fast matrix vector products.
155: 
156: Example 1
157: ---------
158: Construct a 1000x1000 lil_matrix and add some values to it:
159: 
160: >>> from scipy.sparse import lil_matrix
161: >>> from scipy.sparse.linalg import spsolve
162: >>> from numpy.linalg import solve, norm
163: >>> from numpy.random import rand
164: 
165: >>> A = lil_matrix((1000, 1000))
166: >>> A[0, :100] = rand(100)
167: >>> A[1, 100:200] = A[0, :100]
168: >>> A.setdiag(rand(1000))
169: 
170: Now convert it to CSR format and solve A x = b for x:
171: 
172: >>> A = A.tocsr()
173: >>> b = rand(1000)
174: >>> x = spsolve(A, b)
175: 
176: Convert it to a dense matrix and solve, and check that the result
177: is the same:
178: 
179: >>> x_ = solve(A.toarray(), b)
180: 
181: Now we can compute norm of the error with:
182: 
183: >>> err = norm(x-x_)
184: >>> err < 1e-10
185: True
186: 
187: It should be small :)
188: 
189: 
190: Example 2
191: ---------
192: 
193: Construct a matrix in COO format:
194: 
195: >>> from scipy import sparse
196: >>> from numpy import array
197: >>> I = array([0,3,1,0])
198: >>> J = array([0,3,1,2])
199: >>> V = array([4,5,7,9])
200: >>> A = sparse.coo_matrix((V,(I,J)),shape=(4,4))
201: 
202: Notice that the indices do not need to be sorted.
203: 
204: Duplicate (i,j) entries are summed when converting to CSR or CSC.
205: 
206: >>> I = array([0,0,1,3,1,0,0])
207: >>> J = array([0,2,1,3,1,0,0])
208: >>> V = array([1,1,1,1,1,1,1])
209: >>> B = sparse.coo_matrix((V,(I,J)),shape=(4,4)).tocsr()
210: 
211: This is useful for constructing finite-element stiffness and mass matrices.
212: 
213: Further Details
214: ---------------
215: 
216: CSR column indices are not necessarily sorted.  Likewise for CSC row
217: indices.  Use the .sorted_indices() and .sort_indices() methods when
218: sorted indices are required (e.g. when passing data to other libraries).
219: 
220: '''
221: 
222: from __future__ import division, print_function, absolute_import
223: 
224: # Original code by Travis Oliphant.
225: # Modified and extended by Ed Schofield, Robert Cimrman,
226: # Nathan Bell, and Jake Vanderplas.
227: 
228: from .base import *
229: from .csr import *
230: from .csc import *
231: from .lil import *
232: from .dok import *
233: from .coo import *
234: from .dia import *
235: from .bsr import *
236: from .construct import *
237: from .extract import *
238: from ._matrix_io import *
239: 
240: __all__ = [s for s in dir() if not s.startswith('_')]
241: 
242: from scipy._lib._testutils import PytestTester
243: test = PytestTester(__name__)
244: del PytestTester
245: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_381068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'str', '\n=====================================\nSparse matrices (:mod:`scipy.sparse`)\n=====================================\n\n.. currentmodule:: scipy.sparse\n\nSciPy 2-D sparse matrix package for numeric data.\n\nContents\n========\n\nSparse matrix classes\n---------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   bsr_matrix - Block Sparse Row matrix\n   coo_matrix - A sparse matrix in COOrdinate format\n   csc_matrix - Compressed Sparse Column matrix\n   csr_matrix - Compressed Sparse Row matrix\n   dia_matrix - Sparse matrix with DIAgonal storage\n   dok_matrix - Dictionary Of Keys based sparse matrix\n   lil_matrix - Row-based linked list sparse matrix\n   spmatrix - Sparse matrix base class\n\nFunctions\n---------\n\nBuilding sparse matrices:\n\n.. autosummary::\n   :toctree: generated/\n\n   eye - Sparse MxN matrix whose k-th diagonal is all ones\n   identity - Identity matrix in sparse format\n   kron - kronecker product of two sparse matrices\n   kronsum - kronecker sum of sparse matrices\n   diags - Return a sparse matrix from diagonals\n   spdiags - Return a sparse matrix from diagonals\n   block_diag - Build a block diagonal sparse matrix\n   tril - Lower triangular portion of a matrix in sparse format\n   triu - Upper triangular portion of a matrix in sparse format\n   bmat - Build a sparse matrix from sparse sub-blocks\n   hstack - Stack sparse matrices horizontally (column wise)\n   vstack - Stack sparse matrices vertically (row wise)\n   rand - Random values in a given shape\n   random - Random values in a given shape\n\nSave and load sparse matrices:\n\n.. autosummary::\n   :toctree: generated/\n\n   save_npz - Save a sparse matrix to a file using ``.npz`` format.\n   load_npz - Load a sparse matrix from a file using ``.npz`` format.\n\nSparse matrix tools:\n\n.. autosummary::\n   :toctree: generated/\n\n   find\n\nIdentifying sparse matrices:\n\n.. autosummary::\n   :toctree: generated/\n\n   issparse\n   isspmatrix\n   isspmatrix_csc\n   isspmatrix_csr\n   isspmatrix_bsr\n   isspmatrix_lil\n   isspmatrix_dok\n   isspmatrix_coo\n   isspmatrix_dia\n\nSubmodules\n----------\n\n.. autosummary::\n   :toctree: generated/\n\n   csgraph - Compressed sparse graph routines\n   linalg - sparse linear algebra routines\n\nExceptions\n----------\n\n.. autosummary::\n   :toctree: generated/\n\n   SparseEfficiencyWarning\n   SparseWarning\n\n\nUsage information\n=================\n\nThere are seven available sparse matrix types:\n\n    1. csc_matrix: Compressed Sparse Column format\n    2. csr_matrix: Compressed Sparse Row format\n    3. bsr_matrix: Block Sparse Row format\n    4. lil_matrix: List of Lists format\n    5. dok_matrix: Dictionary of Keys format\n    6. coo_matrix: COOrdinate format (aka IJV, triplet format)\n    7. dia_matrix: DIAgonal format\n\nTo construct a matrix efficiently, use either dok_matrix or lil_matrix.\nThe lil_matrix class supports basic slicing and fancy indexing with a\nsimilar syntax to NumPy arrays. As illustrated below, the COO format\nmay also be used to efficiently construct matrices. Despite their\nsimilarity to NumPy arrays, it is **strongly discouraged** to use NumPy\nfunctions directly on these matrices because NumPy may not properly convert\nthem for computations, leading to unexpected (and incorrect) results. If you\ndo want to apply a NumPy function to these matrices, first check if SciPy has\nits own implementation for the given sparse matrix class, or **convert the\nsparse matrix to a NumPy array** (e.g. using the `toarray()` method of the\nclass) first before applying the method.\n\nTo perform manipulations such as multiplication or inversion, first\nconvert the matrix to either CSC or CSR format. The lil_matrix format is\nrow-based, so conversion to CSR is efficient, whereas conversion to CSC\nis less so.\n\nAll conversions among the CSR, CSC, and COO formats are efficient,\nlinear-time operations.\n\nMatrix vector product\n---------------------\nTo do a vector product between a sparse matrix and a vector simply use\nthe matrix `dot` method, as described in its docstring:\n\n>>> import numpy as np\n>>> from scipy.sparse import csr_matrix\n>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])\n>>> v = np.array([1, 0, -1])\n>>> A.dot(v)\narray([ 1, -3, -1], dtype=int64)\n\n.. warning:: As of NumPy 1.7, `np.dot` is not aware of sparse matrices,\n  therefore using it will result on unexpected results or errors.\n  The corresponding dense array should be obtained first instead:\n\n  >>> np.dot(A.toarray(), v)\n  array([ 1, -3, -1], dtype=int64)\n\n  but then all the performance advantages would be lost.\n\nThe CSR format is specially suitable for fast matrix vector products.\n\nExample 1\n---------\nConstruct a 1000x1000 lil_matrix and add some values to it:\n\n>>> from scipy.sparse import lil_matrix\n>>> from scipy.sparse.linalg import spsolve\n>>> from numpy.linalg import solve, norm\n>>> from numpy.random import rand\n\n>>> A = lil_matrix((1000, 1000))\n>>> A[0, :100] = rand(100)\n>>> A[1, 100:200] = A[0, :100]\n>>> A.setdiag(rand(1000))\n\nNow convert it to CSR format and solve A x = b for x:\n\n>>> A = A.tocsr()\n>>> b = rand(1000)\n>>> x = spsolve(A, b)\n\nConvert it to a dense matrix and solve, and check that the result\nis the same:\n\n>>> x_ = solve(A.toarray(), b)\n\nNow we can compute norm of the error with:\n\n>>> err = norm(x-x_)\n>>> err < 1e-10\nTrue\n\nIt should be small :)\n\n\nExample 2\n---------\n\nConstruct a matrix in COO format:\n\n>>> from scipy import sparse\n>>> from numpy import array\n>>> I = array([0,3,1,0])\n>>> J = array([0,3,1,2])\n>>> V = array([4,5,7,9])\n>>> A = sparse.coo_matrix((V,(I,J)),shape=(4,4))\n\nNotice that the indices do not need to be sorted.\n\nDuplicate (i,j) entries are summed when converting to CSR or CSC.\n\n>>> I = array([0,0,1,3,1,0,0])\n>>> J = array([0,2,1,3,1,0,0])\n>>> V = array([1,1,1,1,1,1,1])\n>>> B = sparse.coo_matrix((V,(I,J)),shape=(4,4)).tocsr()\n\nThis is useful for constructing finite-element stiffness and mass matrices.\n\nFurther Details\n---------------\n\nCSR column indices are not necessarily sorted.  Likewise for CSC row\nindices.  Use the .sorted_indices() and .sort_indices() methods when\nsorted indices are required (e.g. when passing data to other libraries).\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 228, 0))

# 'from scipy.sparse.base import ' statement (line 228)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381069 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 228, 0), 'scipy.sparse.base')

if (type(import_381069) is not StypyTypeError):

    if (import_381069 != 'pyd_module'):
        __import__(import_381069)
        sys_modules_381070 = sys.modules[import_381069]
        import_from_module(stypy.reporting.localization.Localization(__file__, 228, 0), 'scipy.sparse.base', sys_modules_381070.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 228, 0), __file__, sys_modules_381070, sys_modules_381070.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 228, 0), 'scipy.sparse.base', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'scipy.sparse.base', import_381069)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 229, 0))

# 'from scipy.sparse.csr import ' statement (line 229)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381071 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 229, 0), 'scipy.sparse.csr')

if (type(import_381071) is not StypyTypeError):

    if (import_381071 != 'pyd_module'):
        __import__(import_381071)
        sys_modules_381072 = sys.modules[import_381071]
        import_from_module(stypy.reporting.localization.Localization(__file__, 229, 0), 'scipy.sparse.csr', sys_modules_381072.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 229, 0), __file__, sys_modules_381072, sys_modules_381072.module_type_store, module_type_store)
    else:
        from scipy.sparse.csr import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 229, 0), 'scipy.sparse.csr', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.csr' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 0), 'scipy.sparse.csr', import_381071)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 230, 0))

# 'from scipy.sparse.csc import ' statement (line 230)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381073 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 230, 0), 'scipy.sparse.csc')

if (type(import_381073) is not StypyTypeError):

    if (import_381073 != 'pyd_module'):
        __import__(import_381073)
        sys_modules_381074 = sys.modules[import_381073]
        import_from_module(stypy.reporting.localization.Localization(__file__, 230, 0), 'scipy.sparse.csc', sys_modules_381074.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 230, 0), __file__, sys_modules_381074, sys_modules_381074.module_type_store, module_type_store)
    else:
        from scipy.sparse.csc import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 230, 0), 'scipy.sparse.csc', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.csc' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'scipy.sparse.csc', import_381073)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 231, 0))

# 'from scipy.sparse.lil import ' statement (line 231)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381075 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 231, 0), 'scipy.sparse.lil')

if (type(import_381075) is not StypyTypeError):

    if (import_381075 != 'pyd_module'):
        __import__(import_381075)
        sys_modules_381076 = sys.modules[import_381075]
        import_from_module(stypy.reporting.localization.Localization(__file__, 231, 0), 'scipy.sparse.lil', sys_modules_381076.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 231, 0), __file__, sys_modules_381076, sys_modules_381076.module_type_store, module_type_store)
    else:
        from scipy.sparse.lil import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 231, 0), 'scipy.sparse.lil', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.lil' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'scipy.sparse.lil', import_381075)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 232, 0))

# 'from scipy.sparse.dok import ' statement (line 232)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381077 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 232, 0), 'scipy.sparse.dok')

if (type(import_381077) is not StypyTypeError):

    if (import_381077 != 'pyd_module'):
        __import__(import_381077)
        sys_modules_381078 = sys.modules[import_381077]
        import_from_module(stypy.reporting.localization.Localization(__file__, 232, 0), 'scipy.sparse.dok', sys_modules_381078.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 232, 0), __file__, sys_modules_381078, sys_modules_381078.module_type_store, module_type_store)
    else:
        from scipy.sparse.dok import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 232, 0), 'scipy.sparse.dok', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.dok' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), 'scipy.sparse.dok', import_381077)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 233, 0))

# 'from scipy.sparse.coo import ' statement (line 233)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381079 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 233, 0), 'scipy.sparse.coo')

if (type(import_381079) is not StypyTypeError):

    if (import_381079 != 'pyd_module'):
        __import__(import_381079)
        sys_modules_381080 = sys.modules[import_381079]
        import_from_module(stypy.reporting.localization.Localization(__file__, 233, 0), 'scipy.sparse.coo', sys_modules_381080.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 233, 0), __file__, sys_modules_381080, sys_modules_381080.module_type_store, module_type_store)
    else:
        from scipy.sparse.coo import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 233, 0), 'scipy.sparse.coo', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.coo' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'scipy.sparse.coo', import_381079)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 234, 0))

# 'from scipy.sparse.dia import ' statement (line 234)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381081 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 234, 0), 'scipy.sparse.dia')

if (type(import_381081) is not StypyTypeError):

    if (import_381081 != 'pyd_module'):
        __import__(import_381081)
        sys_modules_381082 = sys.modules[import_381081]
        import_from_module(stypy.reporting.localization.Localization(__file__, 234, 0), 'scipy.sparse.dia', sys_modules_381082.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 234, 0), __file__, sys_modules_381082, sys_modules_381082.module_type_store, module_type_store)
    else:
        from scipy.sparse.dia import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 234, 0), 'scipy.sparse.dia', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.dia' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 0), 'scipy.sparse.dia', import_381081)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 235, 0))

# 'from scipy.sparse.bsr import ' statement (line 235)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381083 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 235, 0), 'scipy.sparse.bsr')

if (type(import_381083) is not StypyTypeError):

    if (import_381083 != 'pyd_module'):
        __import__(import_381083)
        sys_modules_381084 = sys.modules[import_381083]
        import_from_module(stypy.reporting.localization.Localization(__file__, 235, 0), 'scipy.sparse.bsr', sys_modules_381084.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 235, 0), __file__, sys_modules_381084, sys_modules_381084.module_type_store, module_type_store)
    else:
        from scipy.sparse.bsr import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 235, 0), 'scipy.sparse.bsr', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.bsr' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'scipy.sparse.bsr', import_381083)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 236, 0))

# 'from scipy.sparse.construct import ' statement (line 236)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381085 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 236, 0), 'scipy.sparse.construct')

if (type(import_381085) is not StypyTypeError):

    if (import_381085 != 'pyd_module'):
        __import__(import_381085)
        sys_modules_381086 = sys.modules[import_381085]
        import_from_module(stypy.reporting.localization.Localization(__file__, 236, 0), 'scipy.sparse.construct', sys_modules_381086.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 236, 0), __file__, sys_modules_381086, sys_modules_381086.module_type_store, module_type_store)
    else:
        from scipy.sparse.construct import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 236, 0), 'scipy.sparse.construct', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.construct' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'scipy.sparse.construct', import_381085)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 237, 0))

# 'from scipy.sparse.extract import ' statement (line 237)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381087 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 237, 0), 'scipy.sparse.extract')

if (type(import_381087) is not StypyTypeError):

    if (import_381087 != 'pyd_module'):
        __import__(import_381087)
        sys_modules_381088 = sys.modules[import_381087]
        import_from_module(stypy.reporting.localization.Localization(__file__, 237, 0), 'scipy.sparse.extract', sys_modules_381088.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 237, 0), __file__, sys_modules_381088, sys_modules_381088.module_type_store, module_type_store)
    else:
        from scipy.sparse.extract import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 237, 0), 'scipy.sparse.extract', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.extract' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'scipy.sparse.extract', import_381087)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 238, 0))

# 'from scipy.sparse._matrix_io import ' statement (line 238)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381089 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 238, 0), 'scipy.sparse._matrix_io')

if (type(import_381089) is not StypyTypeError):

    if (import_381089 != 'pyd_module'):
        __import__(import_381089)
        sys_modules_381090 = sys.modules[import_381089]
        import_from_module(stypy.reporting.localization.Localization(__file__, 238, 0), 'scipy.sparse._matrix_io', sys_modules_381090.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 238, 0), __file__, sys_modules_381090, sys_modules_381090.module_type_store, module_type_store)
    else:
        from scipy.sparse._matrix_io import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 238, 0), 'scipy.sparse._matrix_io', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse._matrix_io' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'scipy.sparse._matrix_io', import_381089)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# Assigning a ListComp to a Name (line 240):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 240)
# Processing the call keyword arguments (line 240)
kwargs_381099 = {}
# Getting the type of 'dir' (line 240)
dir_381098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'dir', False)
# Calling dir(args, kwargs) (line 240)
dir_call_result_381100 = invoke(stypy.reporting.localization.Localization(__file__, 240, 22), dir_381098, *[], **kwargs_381099)

comprehension_381101 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 11), dir_call_result_381100)
# Assigning a type to the variable 's' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 's', comprehension_381101)


# Call to startswith(...): (line 240)
# Processing the call arguments (line 240)
str_381094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 48), 'str', '_')
# Processing the call keyword arguments (line 240)
kwargs_381095 = {}
# Getting the type of 's' (line 240)
s_381092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 240)
startswith_381093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 35), s_381092, 'startswith')
# Calling startswith(args, kwargs) (line 240)
startswith_call_result_381096 = invoke(stypy.reporting.localization.Localization(__file__, 240, 35), startswith_381093, *[str_381094], **kwargs_381095)

# Applying the 'not' unary operator (line 240)
result_not__381097 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 31), 'not', startswith_call_result_381096)

# Getting the type of 's' (line 240)
s_381091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 's')
list_381102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 11), list_381102, s_381091)
# Assigning a type to the variable '__all__' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), '__all__', list_381102)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 242, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 242)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_381103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy._lib._testutils')

if (type(import_381103) is not StypyTypeError):

    if (import_381103 != 'pyd_module'):
        __import__(import_381103)
        sys_modules_381104 = sys.modules[import_381103]
        import_from_module(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy._lib._testutils', sys_modules_381104.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 242, 0), __file__, sys_modules_381104, sys_modules_381104.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'scipy._lib._testutils', import_381103)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# Assigning a Call to a Name (line 243):

# Call to PytestTester(...): (line 243)
# Processing the call arguments (line 243)
# Getting the type of '__name__' (line 243)
name___381106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), '__name__', False)
# Processing the call keyword arguments (line 243)
kwargs_381107 = {}
# Getting the type of 'PytestTester' (line 243)
PytestTester_381105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 243)
PytestTester_call_result_381108 = invoke(stypy.reporting.localization.Localization(__file__, 243, 7), PytestTester_381105, *[name___381106], **kwargs_381107)

# Assigning a type to the variable 'test' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'test', PytestTester_call_result_381108)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 244, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
