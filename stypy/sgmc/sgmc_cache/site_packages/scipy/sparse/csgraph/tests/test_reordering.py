
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal
5: from scipy.sparse.csgraph import (reverse_cuthill_mckee,
6:         maximum_bipartite_matching, structural_rank)
7: from scipy.sparse import diags, csc_matrix, csr_matrix, coo_matrix
8: 
9: def test_graph_reverse_cuthill_mckee():
10:     A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
11:                 [0, 1, 1, 0, 0, 1, 0, 1],
12:                 [0, 1, 1, 0, 1, 0, 0, 0],
13:                 [0, 0, 0, 1, 0, 0, 1, 0],
14:                 [1, 0, 1, 0, 1, 0, 0, 0],
15:                 [0, 1, 0, 0, 0, 1, 0, 1],
16:                 [0, 0, 0, 1, 0, 0, 1, 0],
17:                 [0, 1, 0, 0, 0, 1, 0, 1]], dtype=int)
18:     
19:     graph = csr_matrix(A)
20:     perm = reverse_cuthill_mckee(graph)
21:     correct_perm = np.array([6, 3, 7, 5, 1, 2, 4, 0])
22:     assert_equal(perm, correct_perm)
23:     
24:     # Test int64 indices input
25:     graph.indices = graph.indices.astype('int64')
26:     graph.indptr = graph.indptr.astype('int64')
27:     perm = reverse_cuthill_mckee(graph, True)
28:     assert_equal(perm, correct_perm)
29: 
30: 
31: def test_graph_reverse_cuthill_mckee_ordering():
32:     data = np.ones(63,dtype=int)
33:     rows = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 
34:                 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
35:                 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9,
36:                 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 
37:                 12, 12, 12, 13, 13, 13, 13, 14, 14, 14,
38:                 14, 15, 15, 15, 15, 15])
39:     cols = np.array([0, 2, 5, 8, 10, 1, 3, 9, 11, 0, 2,
40:                 7, 10, 1, 3, 11, 4, 6, 12, 14, 0, 7, 13, 
41:                 15, 4, 6, 14, 2, 5, 7, 15, 0, 8, 10, 13,
42:                 1, 9, 11, 0, 2, 8, 10, 15, 1, 3, 9, 11,
43:                 4, 12, 14, 5, 8, 13, 15, 4, 6, 12, 14,
44:                 5, 7, 10, 13, 15])
45:     graph = coo_matrix((data, (rows,cols))).tocsr()
46:     perm = reverse_cuthill_mckee(graph)
47:     correct_perm = np.array([12, 14, 4, 6, 10, 8, 2, 15,
48:                 0, 13, 7, 5, 9, 11, 1, 3])
49:     assert_equal(perm, correct_perm)
50: 
51: 
52: def test_graph_maximum_bipartite_matching():
53:     A = diags(np.ones(25), offsets=0, format='csc')
54:     rand_perm = np.random.permutation(25)
55:     rand_perm2 = np.random.permutation(25)
56: 
57:     Rrow = np.arange(25)
58:     Rcol = rand_perm
59:     Rdata = np.ones(25,dtype=int)
60:     Rmat = coo_matrix((Rdata,(Rrow,Rcol))).tocsc()
61: 
62:     Crow = rand_perm2
63:     Ccol = np.arange(25)
64:     Cdata = np.ones(25,dtype=int)
65:     Cmat = coo_matrix((Cdata,(Crow,Ccol))).tocsc()
66:     # Randomly permute identity matrix
67:     B = Rmat*A*Cmat
68:     
69:     # Row permute
70:     perm = maximum_bipartite_matching(B,perm_type='row')
71:     Rrow = np.arange(25)
72:     Rcol = perm
73:     Rdata = np.ones(25,dtype=int)
74:     Rmat = coo_matrix((Rdata,(Rrow,Rcol))).tocsc()
75:     C1 = Rmat*B
76:     
77:     # Column permute
78:     perm2 = maximum_bipartite_matching(B,perm_type='column')
79:     Crow = perm2
80:     Ccol = np.arange(25)
81:     Cdata = np.ones(25,dtype=int)
82:     Cmat = coo_matrix((Cdata,(Crow,Ccol))).tocsc()
83:     C2 = B*Cmat
84:     
85:     # Should get identity matrix back
86:     assert_equal(any(C1.diagonal() == 0), False)
87:     assert_equal(any(C2.diagonal() == 0), False)
88:     
89:     # Test int64 indices input
90:     B.indices = B.indices.astype('int64')
91:     B.indptr = B.indptr.astype('int64')
92:     perm = maximum_bipartite_matching(B,perm_type='row')
93:     Rrow = np.arange(25)
94:     Rcol = perm
95:     Rdata = np.ones(25,dtype=int)
96:     Rmat = coo_matrix((Rdata,(Rrow,Rcol))).tocsc()
97:     C3 = Rmat*B
98:     assert_equal(any(C3.diagonal() == 0), False)
99: 
100: 
101: def test_graph_structural_rank():
102:     # Test square matrix #1
103:     A = csc_matrix([[1, 1, 0], 
104:                     [1, 0, 1],
105:                     [0, 1, 0]])
106:     assert_equal(structural_rank(A), 3)
107:     
108:     # Test square matrix #2
109:     rows = np.array([0,0,0,0,0,1,1,2,2,3,3,3,3,3,3,4,4,5,5,6,6,7,7])
110:     cols = np.array([0,1,2,3,4,2,5,2,6,0,1,3,5,6,7,4,5,5,6,2,6,2,4])
111:     data = np.ones_like(rows)
112:     B = coo_matrix((data,(rows,cols)), shape=(8,8))
113:     assert_equal(structural_rank(B), 6)
114:     
115:     #Test non-square matrix
116:     C = csc_matrix([[1, 0, 2, 0], 
117:                     [2, 0, 4, 0]])
118:     assert_equal(structural_rank(C), 2)
119:     
120:     #Test tall matrix
121:     assert_equal(structural_rank(C.T), 2)
122: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382962 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_382962) is not StypyTypeError):

    if (import_382962 != 'pyd_module'):
        __import__(import_382962)
        sys_modules_382963 = sys.modules[import_382962]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_382963.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_382962)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382964 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_382964) is not StypyTypeError):

    if (import_382964 != 'pyd_module'):
        __import__(import_382964)
        sys_modules_382965 = sys.modules[import_382964]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_382965.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_382965, sys_modules_382965.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_382964)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse.csgraph import reverse_cuthill_mckee, maximum_bipartite_matching, structural_rank' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382966 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph')

if (type(import_382966) is not StypyTypeError):

    if (import_382966 != 'pyd_module'):
        __import__(import_382966)
        sys_modules_382967 = sys.modules[import_382966]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph', sys_modules_382967.module_type_store, module_type_store, ['reverse_cuthill_mckee', 'maximum_bipartite_matching', 'structural_rank'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_382967, sys_modules_382967.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph import reverse_cuthill_mckee, maximum_bipartite_matching, structural_rank

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph', None, module_type_store, ['reverse_cuthill_mckee', 'maximum_bipartite_matching', 'structural_rank'], [reverse_cuthill_mckee, maximum_bipartite_matching, structural_rank])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph', import_382966)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse import diags, csc_matrix, csr_matrix, coo_matrix' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382968 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse')

if (type(import_382968) is not StypyTypeError):

    if (import_382968 != 'pyd_module'):
        __import__(import_382968)
        sys_modules_382969 = sys.modules[import_382968]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', sys_modules_382969.module_type_store, module_type_store, ['diags', 'csc_matrix', 'csr_matrix', 'coo_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_382969, sys_modules_382969.module_type_store, module_type_store)
    else:
        from scipy.sparse import diags, csc_matrix, csr_matrix, coo_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', None, module_type_store, ['diags', 'csc_matrix', 'csr_matrix', 'coo_matrix'], [diags, csc_matrix, csr_matrix, coo_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', import_382968)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')


@norecursion
def test_graph_reverse_cuthill_mckee(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_reverse_cuthill_mckee'
    module_type_store = module_type_store.open_function_context('test_graph_reverse_cuthill_mckee', 9, 0, False)
    
    # Passed parameters checking function
    test_graph_reverse_cuthill_mckee.stypy_localization = localization
    test_graph_reverse_cuthill_mckee.stypy_type_of_self = None
    test_graph_reverse_cuthill_mckee.stypy_type_store = module_type_store
    test_graph_reverse_cuthill_mckee.stypy_function_name = 'test_graph_reverse_cuthill_mckee'
    test_graph_reverse_cuthill_mckee.stypy_param_names_list = []
    test_graph_reverse_cuthill_mckee.stypy_varargs_param_name = None
    test_graph_reverse_cuthill_mckee.stypy_kwargs_param_name = None
    test_graph_reverse_cuthill_mckee.stypy_call_defaults = defaults
    test_graph_reverse_cuthill_mckee.stypy_call_varargs = varargs
    test_graph_reverse_cuthill_mckee.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_reverse_cuthill_mckee', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_reverse_cuthill_mckee', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_reverse_cuthill_mckee(...)' code ##################

    
    # Assigning a Call to a Name (line 10):
    
    # Call to array(...): (line 10)
    # Processing the call arguments (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_382972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_382973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    int_382974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382974)
    # Adding element type (line 10)
    int_382975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382975)
    # Adding element type (line 10)
    int_382976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382976)
    # Adding element type (line 10)
    int_382977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382977)
    # Adding element type (line 10)
    int_382978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382978)
    # Adding element type (line 10)
    int_382979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382979)
    # Adding element type (line 10)
    int_382980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382980)
    # Adding element type (line 10)
    int_382981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 18), list_382973, int_382981)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_382973)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_382982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    int_382983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382983)
    # Adding element type (line 11)
    int_382984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382984)
    # Adding element type (line 11)
    int_382985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382985)
    # Adding element type (line 11)
    int_382986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382986)
    # Adding element type (line 11)
    int_382987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382987)
    # Adding element type (line 11)
    int_382988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382988)
    # Adding element type (line 11)
    int_382989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382989)
    # Adding element type (line 11)
    int_382990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_382982, int_382990)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_382982)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_382991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    int_382992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382992)
    # Adding element type (line 12)
    int_382993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382993)
    # Adding element type (line 12)
    int_382994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382994)
    # Adding element type (line 12)
    int_382995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382995)
    # Adding element type (line 12)
    int_382996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382996)
    # Adding element type (line 12)
    int_382997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382997)
    # Adding element type (line 12)
    int_382998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382998)
    # Adding element type (line 12)
    int_382999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 16), list_382991, int_382999)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_382991)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_383000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    int_383001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383001)
    # Adding element type (line 13)
    int_383002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383002)
    # Adding element type (line 13)
    int_383003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383003)
    # Adding element type (line 13)
    int_383004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383004)
    # Adding element type (line 13)
    int_383005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383005)
    # Adding element type (line 13)
    int_383006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383006)
    # Adding element type (line 13)
    int_383007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383007)
    # Adding element type (line 13)
    int_383008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_383000, int_383008)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_383000)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_383009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    int_383010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383010)
    # Adding element type (line 14)
    int_383011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383011)
    # Adding element type (line 14)
    int_383012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383012)
    # Adding element type (line 14)
    int_383013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383013)
    # Adding element type (line 14)
    int_383014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383014)
    # Adding element type (line 14)
    int_383015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383015)
    # Adding element type (line 14)
    int_383016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383016)
    # Adding element type (line 14)
    int_383017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_383009, int_383017)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_383009)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_383018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_383019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383019)
    # Adding element type (line 15)
    int_383020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383020)
    # Adding element type (line 15)
    int_383021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383021)
    # Adding element type (line 15)
    int_383022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383022)
    # Adding element type (line 15)
    int_383023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383023)
    # Adding element type (line 15)
    int_383024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383024)
    # Adding element type (line 15)
    int_383025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383025)
    # Adding element type (line 15)
    int_383026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_383018, int_383026)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_383018)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_383027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_383028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383028)
    # Adding element type (line 16)
    int_383029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383029)
    # Adding element type (line 16)
    int_383030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383030)
    # Adding element type (line 16)
    int_383031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383031)
    # Adding element type (line 16)
    int_383032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383032)
    # Adding element type (line 16)
    int_383033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383033)
    # Adding element type (line 16)
    int_383034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383034)
    # Adding element type (line 16)
    int_383035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 16), list_383027, int_383035)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_383027)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_383036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_383037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383037)
    # Adding element type (line 17)
    int_383038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383038)
    # Adding element type (line 17)
    int_383039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383039)
    # Adding element type (line 17)
    int_383040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383040)
    # Adding element type (line 17)
    int_383041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383041)
    # Adding element type (line 17)
    int_383042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383042)
    # Adding element type (line 17)
    int_383043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383043)
    # Adding element type (line 17)
    int_383044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), list_383036, int_383044)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 17), list_382972, list_383036)
    
    # Processing the call keyword arguments (line 10)
    # Getting the type of 'int' (line 17)
    int_383045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 49), 'int', False)
    keyword_383046 = int_383045
    kwargs_383047 = {'dtype': keyword_383046}
    # Getting the type of 'np' (line 10)
    np_382970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 10)
    array_382971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), np_382970, 'array')
    # Calling array(args, kwargs) (line 10)
    array_call_result_383048 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), array_382971, *[list_382972], **kwargs_383047)
    
    # Assigning a type to the variable 'A' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'A', array_call_result_383048)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to csr_matrix(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'A' (line 19)
    A_383050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'A', False)
    # Processing the call keyword arguments (line 19)
    kwargs_383051 = {}
    # Getting the type of 'csr_matrix' (line 19)
    csr_matrix_383049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 19)
    csr_matrix_call_result_383052 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), csr_matrix_383049, *[A_383050], **kwargs_383051)
    
    # Assigning a type to the variable 'graph' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'graph', csr_matrix_call_result_383052)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to reverse_cuthill_mckee(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'graph' (line 20)
    graph_383054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'graph', False)
    # Processing the call keyword arguments (line 20)
    kwargs_383055 = {}
    # Getting the type of 'reverse_cuthill_mckee' (line 20)
    reverse_cuthill_mckee_383053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'reverse_cuthill_mckee', False)
    # Calling reverse_cuthill_mckee(args, kwargs) (line 20)
    reverse_cuthill_mckee_call_result_383056 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), reverse_cuthill_mckee_383053, *[graph_383054], **kwargs_383055)
    
    # Assigning a type to the variable 'perm' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'perm', reverse_cuthill_mckee_call_result_383056)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to array(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_383059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_383060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383060)
    # Adding element type (line 21)
    int_383061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383061)
    # Adding element type (line 21)
    int_383062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383062)
    # Adding element type (line 21)
    int_383063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383063)
    # Adding element type (line 21)
    int_383064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383064)
    # Adding element type (line 21)
    int_383065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383065)
    # Adding element type (line 21)
    int_383066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383066)
    # Adding element type (line 21)
    int_383067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_383059, int_383067)
    
    # Processing the call keyword arguments (line 21)
    kwargs_383068 = {}
    # Getting the type of 'np' (line 21)
    np_383057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'np', False)
    # Obtaining the member 'array' of a type (line 21)
    array_383058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), np_383057, 'array')
    # Calling array(args, kwargs) (line 21)
    array_call_result_383069 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), array_383058, *[list_383059], **kwargs_383068)
    
    # Assigning a type to the variable 'correct_perm' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'correct_perm', array_call_result_383069)
    
    # Call to assert_equal(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'perm' (line 22)
    perm_383071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'perm', False)
    # Getting the type of 'correct_perm' (line 22)
    correct_perm_383072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'correct_perm', False)
    # Processing the call keyword arguments (line 22)
    kwargs_383073 = {}
    # Getting the type of 'assert_equal' (line 22)
    assert_equal_383070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 22)
    assert_equal_call_result_383074 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), assert_equal_383070, *[perm_383071, correct_perm_383072], **kwargs_383073)
    
    
    # Assigning a Call to a Attribute (line 25):
    
    # Call to astype(...): (line 25)
    # Processing the call arguments (line 25)
    str_383078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'str', 'int64')
    # Processing the call keyword arguments (line 25)
    kwargs_383079 = {}
    # Getting the type of 'graph' (line 25)
    graph_383075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'graph', False)
    # Obtaining the member 'indices' of a type (line 25)
    indices_383076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), graph_383075, 'indices')
    # Obtaining the member 'astype' of a type (line 25)
    astype_383077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), indices_383076, 'astype')
    # Calling astype(args, kwargs) (line 25)
    astype_call_result_383080 = invoke(stypy.reporting.localization.Localization(__file__, 25, 20), astype_383077, *[str_383078], **kwargs_383079)
    
    # Getting the type of 'graph' (line 25)
    graph_383081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'graph')
    # Setting the type of the member 'indices' of a type (line 25)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), graph_383081, 'indices', astype_call_result_383080)
    
    # Assigning a Call to a Attribute (line 26):
    
    # Call to astype(...): (line 26)
    # Processing the call arguments (line 26)
    str_383085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'str', 'int64')
    # Processing the call keyword arguments (line 26)
    kwargs_383086 = {}
    # Getting the type of 'graph' (line 26)
    graph_383082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'graph', False)
    # Obtaining the member 'indptr' of a type (line 26)
    indptr_383083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 19), graph_383082, 'indptr')
    # Obtaining the member 'astype' of a type (line 26)
    astype_383084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 19), indptr_383083, 'astype')
    # Calling astype(args, kwargs) (line 26)
    astype_call_result_383087 = invoke(stypy.reporting.localization.Localization(__file__, 26, 19), astype_383084, *[str_383085], **kwargs_383086)
    
    # Getting the type of 'graph' (line 26)
    graph_383088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'graph')
    # Setting the type of the member 'indptr' of a type (line 26)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), graph_383088, 'indptr', astype_call_result_383087)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to reverse_cuthill_mckee(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'graph' (line 27)
    graph_383090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'graph', False)
    # Getting the type of 'True' (line 27)
    True_383091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 40), 'True', False)
    # Processing the call keyword arguments (line 27)
    kwargs_383092 = {}
    # Getting the type of 'reverse_cuthill_mckee' (line 27)
    reverse_cuthill_mckee_383089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'reverse_cuthill_mckee', False)
    # Calling reverse_cuthill_mckee(args, kwargs) (line 27)
    reverse_cuthill_mckee_call_result_383093 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), reverse_cuthill_mckee_383089, *[graph_383090, True_383091], **kwargs_383092)
    
    # Assigning a type to the variable 'perm' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'perm', reverse_cuthill_mckee_call_result_383093)
    
    # Call to assert_equal(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'perm' (line 28)
    perm_383095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'perm', False)
    # Getting the type of 'correct_perm' (line 28)
    correct_perm_383096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'correct_perm', False)
    # Processing the call keyword arguments (line 28)
    kwargs_383097 = {}
    # Getting the type of 'assert_equal' (line 28)
    assert_equal_383094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 28)
    assert_equal_call_result_383098 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), assert_equal_383094, *[perm_383095, correct_perm_383096], **kwargs_383097)
    
    
    # ################# End of 'test_graph_reverse_cuthill_mckee(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_reverse_cuthill_mckee' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_383099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_383099)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_reverse_cuthill_mckee'
    return stypy_return_type_383099

# Assigning a type to the variable 'test_graph_reverse_cuthill_mckee' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_graph_reverse_cuthill_mckee', test_graph_reverse_cuthill_mckee)

@norecursion
def test_graph_reverse_cuthill_mckee_ordering(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_reverse_cuthill_mckee_ordering'
    module_type_store = module_type_store.open_function_context('test_graph_reverse_cuthill_mckee_ordering', 31, 0, False)
    
    # Passed parameters checking function
    test_graph_reverse_cuthill_mckee_ordering.stypy_localization = localization
    test_graph_reverse_cuthill_mckee_ordering.stypy_type_of_self = None
    test_graph_reverse_cuthill_mckee_ordering.stypy_type_store = module_type_store
    test_graph_reverse_cuthill_mckee_ordering.stypy_function_name = 'test_graph_reverse_cuthill_mckee_ordering'
    test_graph_reverse_cuthill_mckee_ordering.stypy_param_names_list = []
    test_graph_reverse_cuthill_mckee_ordering.stypy_varargs_param_name = None
    test_graph_reverse_cuthill_mckee_ordering.stypy_kwargs_param_name = None
    test_graph_reverse_cuthill_mckee_ordering.stypy_call_defaults = defaults
    test_graph_reverse_cuthill_mckee_ordering.stypy_call_varargs = varargs
    test_graph_reverse_cuthill_mckee_ordering.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_reverse_cuthill_mckee_ordering', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_reverse_cuthill_mckee_ordering', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_reverse_cuthill_mckee_ordering(...)' code ##################

    
    # Assigning a Call to a Name (line 32):
    
    # Call to ones(...): (line 32)
    # Processing the call arguments (line 32)
    int_383102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'int')
    # Processing the call keyword arguments (line 32)
    # Getting the type of 'int' (line 32)
    int_383103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'int', False)
    keyword_383104 = int_383103
    kwargs_383105 = {'dtype': keyword_383104}
    # Getting the type of 'np' (line 32)
    np_383100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 32)
    ones_383101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 11), np_383100, 'ones')
    # Calling ones(args, kwargs) (line 32)
    ones_call_result_383106 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), ones_383101, *[int_383102], **kwargs_383105)
    
    # Assigning a type to the variable 'data' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'data', ones_call_result_383106)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to array(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_383109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    int_383110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383110)
    # Adding element type (line 33)
    int_383111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383111)
    # Adding element type (line 33)
    int_383112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383112)
    # Adding element type (line 33)
    int_383113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383113)
    # Adding element type (line 33)
    int_383114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383114)
    # Adding element type (line 33)
    int_383115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383115)
    # Adding element type (line 33)
    int_383116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383116)
    # Adding element type (line 33)
    int_383117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383117)
    # Adding element type (line 33)
    int_383118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383118)
    # Adding element type (line 33)
    int_383119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383119)
    # Adding element type (line 33)
    int_383120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383120)
    # Adding element type (line 33)
    int_383121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383121)
    # Adding element type (line 33)
    int_383122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383122)
    # Adding element type (line 33)
    int_383123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383123)
    # Adding element type (line 33)
    int_383124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383124)
    # Adding element type (line 33)
    int_383125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383125)
    # Adding element type (line 33)
    int_383126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383126)
    # Adding element type (line 33)
    int_383127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383127)
    # Adding element type (line 33)
    int_383128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383128)
    # Adding element type (line 33)
    int_383129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383129)
    # Adding element type (line 33)
    int_383130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383130)
    # Adding element type (line 33)
    int_383131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383131)
    # Adding element type (line 33)
    int_383132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383132)
    # Adding element type (line 33)
    int_383133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383133)
    # Adding element type (line 33)
    int_383134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383134)
    # Adding element type (line 33)
    int_383135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383135)
    # Adding element type (line 33)
    int_383136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383136)
    # Adding element type (line 33)
    int_383137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383137)
    # Adding element type (line 33)
    int_383138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383138)
    # Adding element type (line 33)
    int_383139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383139)
    # Adding element type (line 33)
    int_383140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383140)
    # Adding element type (line 33)
    int_383141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383141)
    # Adding element type (line 33)
    int_383142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383142)
    # Adding element type (line 33)
    int_383143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383143)
    # Adding element type (line 33)
    int_383144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383144)
    # Adding element type (line 33)
    int_383145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383145)
    # Adding element type (line 33)
    int_383146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383146)
    # Adding element type (line 33)
    int_383147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383147)
    # Adding element type (line 33)
    int_383148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383148)
    # Adding element type (line 33)
    int_383149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383149)
    # Adding element type (line 33)
    int_383150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383150)
    # Adding element type (line 33)
    int_383151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383151)
    # Adding element type (line 33)
    int_383152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383152)
    # Adding element type (line 33)
    int_383153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383153)
    # Adding element type (line 33)
    int_383154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383154)
    # Adding element type (line 33)
    int_383155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383155)
    # Adding element type (line 33)
    int_383156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383156)
    # Adding element type (line 33)
    int_383157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383157)
    # Adding element type (line 33)
    int_383158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383158)
    # Adding element type (line 33)
    int_383159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383159)
    # Adding element type (line 33)
    int_383160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383160)
    # Adding element type (line 33)
    int_383161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383161)
    # Adding element type (line 33)
    int_383162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383162)
    # Adding element type (line 33)
    int_383163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383163)
    # Adding element type (line 33)
    int_383164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383164)
    # Adding element type (line 33)
    int_383165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383165)
    # Adding element type (line 33)
    int_383166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383166)
    # Adding element type (line 33)
    int_383167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383167)
    # Adding element type (line 33)
    int_383168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383168)
    # Adding element type (line 33)
    int_383169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383169)
    # Adding element type (line 33)
    int_383170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383170)
    # Adding element type (line 33)
    int_383171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383171)
    # Adding element type (line 33)
    int_383172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 20), list_383109, int_383172)
    
    # Processing the call keyword arguments (line 33)
    kwargs_383173 = {}
    # Getting the type of 'np' (line 33)
    np_383107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 33)
    array_383108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), np_383107, 'array')
    # Calling array(args, kwargs) (line 33)
    array_call_result_383174 = invoke(stypy.reporting.localization.Localization(__file__, 33, 11), array_383108, *[list_383109], **kwargs_383173)
    
    # Assigning a type to the variable 'rows' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'rows', array_call_result_383174)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to array(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_383177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    int_383178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383178)
    # Adding element type (line 39)
    int_383179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383179)
    # Adding element type (line 39)
    int_383180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383180)
    # Adding element type (line 39)
    int_383181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383181)
    # Adding element type (line 39)
    int_383182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383182)
    # Adding element type (line 39)
    int_383183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383183)
    # Adding element type (line 39)
    int_383184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383184)
    # Adding element type (line 39)
    int_383185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383185)
    # Adding element type (line 39)
    int_383186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383186)
    # Adding element type (line 39)
    int_383187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383187)
    # Adding element type (line 39)
    int_383188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383188)
    # Adding element type (line 39)
    int_383189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383189)
    # Adding element type (line 39)
    int_383190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383190)
    # Adding element type (line 39)
    int_383191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383191)
    # Adding element type (line 39)
    int_383192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383192)
    # Adding element type (line 39)
    int_383193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383193)
    # Adding element type (line 39)
    int_383194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383194)
    # Adding element type (line 39)
    int_383195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383195)
    # Adding element type (line 39)
    int_383196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383196)
    # Adding element type (line 39)
    int_383197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383197)
    # Adding element type (line 39)
    int_383198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383198)
    # Adding element type (line 39)
    int_383199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383199)
    # Adding element type (line 39)
    int_383200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383200)
    # Adding element type (line 39)
    int_383201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383201)
    # Adding element type (line 39)
    int_383202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383202)
    # Adding element type (line 39)
    int_383203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383203)
    # Adding element type (line 39)
    int_383204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383204)
    # Adding element type (line 39)
    int_383205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383205)
    # Adding element type (line 39)
    int_383206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383206)
    # Adding element type (line 39)
    int_383207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383207)
    # Adding element type (line 39)
    int_383208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383208)
    # Adding element type (line 39)
    int_383209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383209)
    # Adding element type (line 39)
    int_383210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383210)
    # Adding element type (line 39)
    int_383211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383211)
    # Adding element type (line 39)
    int_383212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383212)
    # Adding element type (line 39)
    int_383213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383213)
    # Adding element type (line 39)
    int_383214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383214)
    # Adding element type (line 39)
    int_383215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383215)
    # Adding element type (line 39)
    int_383216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383216)
    # Adding element type (line 39)
    int_383217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383217)
    # Adding element type (line 39)
    int_383218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383218)
    # Adding element type (line 39)
    int_383219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383219)
    # Adding element type (line 39)
    int_383220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383220)
    # Adding element type (line 39)
    int_383221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383221)
    # Adding element type (line 39)
    int_383222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383222)
    # Adding element type (line 39)
    int_383223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383223)
    # Adding element type (line 39)
    int_383224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383224)
    # Adding element type (line 39)
    int_383225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383225)
    # Adding element type (line 39)
    int_383226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383226)
    # Adding element type (line 39)
    int_383227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383227)
    # Adding element type (line 39)
    int_383228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383228)
    # Adding element type (line 39)
    int_383229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383229)
    # Adding element type (line 39)
    int_383230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383230)
    # Adding element type (line 39)
    int_383231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383231)
    # Adding element type (line 39)
    int_383232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383232)
    # Adding element type (line 39)
    int_383233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383233)
    # Adding element type (line 39)
    int_383234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383234)
    # Adding element type (line 39)
    int_383235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383235)
    # Adding element type (line 39)
    int_383236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383236)
    # Adding element type (line 39)
    int_383237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383237)
    # Adding element type (line 39)
    int_383238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383238)
    # Adding element type (line 39)
    int_383239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383239)
    # Adding element type (line 39)
    int_383240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 20), list_383177, int_383240)
    
    # Processing the call keyword arguments (line 39)
    kwargs_383241 = {}
    # Getting the type of 'np' (line 39)
    np_383175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 39)
    array_383176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 11), np_383175, 'array')
    # Calling array(args, kwargs) (line 39)
    array_call_result_383242 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), array_383176, *[list_383177], **kwargs_383241)
    
    # Assigning a type to the variable 'cols' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'cols', array_call_result_383242)
    
    # Assigning a Call to a Name (line 45):
    
    # Call to tocsr(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_383252 = {}
    
    # Call to coo_matrix(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_383244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    # Getting the type of 'data' (line 45)
    data_383245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 24), tuple_383244, data_383245)
    # Adding element type (line 45)
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_383246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    # Getting the type of 'rows' (line 45)
    rows_383247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'rows', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 31), tuple_383246, rows_383247)
    # Adding element type (line 45)
    # Getting the type of 'cols' (line 45)
    cols_383248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'cols', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 31), tuple_383246, cols_383248)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 24), tuple_383244, tuple_383246)
    
    # Processing the call keyword arguments (line 45)
    kwargs_383249 = {}
    # Getting the type of 'coo_matrix' (line 45)
    coo_matrix_383243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 45)
    coo_matrix_call_result_383250 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), coo_matrix_383243, *[tuple_383244], **kwargs_383249)
    
    # Obtaining the member 'tocsr' of a type (line 45)
    tocsr_383251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), coo_matrix_call_result_383250, 'tocsr')
    # Calling tocsr(args, kwargs) (line 45)
    tocsr_call_result_383253 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), tocsr_383251, *[], **kwargs_383252)
    
    # Assigning a type to the variable 'graph' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'graph', tocsr_call_result_383253)
    
    # Assigning a Call to a Name (line 46):
    
    # Call to reverse_cuthill_mckee(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'graph' (line 46)
    graph_383255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'graph', False)
    # Processing the call keyword arguments (line 46)
    kwargs_383256 = {}
    # Getting the type of 'reverse_cuthill_mckee' (line 46)
    reverse_cuthill_mckee_383254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'reverse_cuthill_mckee', False)
    # Calling reverse_cuthill_mckee(args, kwargs) (line 46)
    reverse_cuthill_mckee_call_result_383257 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), reverse_cuthill_mckee_383254, *[graph_383255], **kwargs_383256)
    
    # Assigning a type to the variable 'perm' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'perm', reverse_cuthill_mckee_call_result_383257)
    
    # Assigning a Call to a Name (line 47):
    
    # Call to array(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_383260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    int_383261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383261)
    # Adding element type (line 47)
    int_383262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383262)
    # Adding element type (line 47)
    int_383263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383263)
    # Adding element type (line 47)
    int_383264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383264)
    # Adding element type (line 47)
    int_383265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383265)
    # Adding element type (line 47)
    int_383266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383266)
    # Adding element type (line 47)
    int_383267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383267)
    # Adding element type (line 47)
    int_383268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383268)
    # Adding element type (line 47)
    int_383269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383269)
    # Adding element type (line 47)
    int_383270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383270)
    # Adding element type (line 47)
    int_383271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383271)
    # Adding element type (line 47)
    int_383272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383272)
    # Adding element type (line 47)
    int_383273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383273)
    # Adding element type (line 47)
    int_383274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383274)
    # Adding element type (line 47)
    int_383275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383275)
    # Adding element type (line 47)
    int_383276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_383260, int_383276)
    
    # Processing the call keyword arguments (line 47)
    kwargs_383277 = {}
    # Getting the type of 'np' (line 47)
    np_383258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'np', False)
    # Obtaining the member 'array' of a type (line 47)
    array_383259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), np_383258, 'array')
    # Calling array(args, kwargs) (line 47)
    array_call_result_383278 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), array_383259, *[list_383260], **kwargs_383277)
    
    # Assigning a type to the variable 'correct_perm' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'correct_perm', array_call_result_383278)
    
    # Call to assert_equal(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'perm' (line 49)
    perm_383280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'perm', False)
    # Getting the type of 'correct_perm' (line 49)
    correct_perm_383281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'correct_perm', False)
    # Processing the call keyword arguments (line 49)
    kwargs_383282 = {}
    # Getting the type of 'assert_equal' (line 49)
    assert_equal_383279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 49)
    assert_equal_call_result_383283 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), assert_equal_383279, *[perm_383280, correct_perm_383281], **kwargs_383282)
    
    
    # ################# End of 'test_graph_reverse_cuthill_mckee_ordering(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_reverse_cuthill_mckee_ordering' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_383284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_383284)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_reverse_cuthill_mckee_ordering'
    return stypy_return_type_383284

# Assigning a type to the variable 'test_graph_reverse_cuthill_mckee_ordering' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'test_graph_reverse_cuthill_mckee_ordering', test_graph_reverse_cuthill_mckee_ordering)

@norecursion
def test_graph_maximum_bipartite_matching(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_maximum_bipartite_matching'
    module_type_store = module_type_store.open_function_context('test_graph_maximum_bipartite_matching', 52, 0, False)
    
    # Passed parameters checking function
    test_graph_maximum_bipartite_matching.stypy_localization = localization
    test_graph_maximum_bipartite_matching.stypy_type_of_self = None
    test_graph_maximum_bipartite_matching.stypy_type_store = module_type_store
    test_graph_maximum_bipartite_matching.stypy_function_name = 'test_graph_maximum_bipartite_matching'
    test_graph_maximum_bipartite_matching.stypy_param_names_list = []
    test_graph_maximum_bipartite_matching.stypy_varargs_param_name = None
    test_graph_maximum_bipartite_matching.stypy_kwargs_param_name = None
    test_graph_maximum_bipartite_matching.stypy_call_defaults = defaults
    test_graph_maximum_bipartite_matching.stypy_call_varargs = varargs
    test_graph_maximum_bipartite_matching.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_maximum_bipartite_matching', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_maximum_bipartite_matching', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_maximum_bipartite_matching(...)' code ##################

    
    # Assigning a Call to a Name (line 53):
    
    # Call to diags(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Call to ones(...): (line 53)
    # Processing the call arguments (line 53)
    int_383288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 22), 'int')
    # Processing the call keyword arguments (line 53)
    kwargs_383289 = {}
    # Getting the type of 'np' (line 53)
    np_383286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'np', False)
    # Obtaining the member 'ones' of a type (line 53)
    ones_383287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), np_383286, 'ones')
    # Calling ones(args, kwargs) (line 53)
    ones_call_result_383290 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), ones_383287, *[int_383288], **kwargs_383289)
    
    # Processing the call keyword arguments (line 53)
    int_383291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'int')
    keyword_383292 = int_383291
    str_383293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 45), 'str', 'csc')
    keyword_383294 = str_383293
    kwargs_383295 = {'format': keyword_383294, 'offsets': keyword_383292}
    # Getting the type of 'diags' (line 53)
    diags_383285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'diags', False)
    # Calling diags(args, kwargs) (line 53)
    diags_call_result_383296 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), diags_383285, *[ones_call_result_383290], **kwargs_383295)
    
    # Assigning a type to the variable 'A' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'A', diags_call_result_383296)
    
    # Assigning a Call to a Name (line 54):
    
    # Call to permutation(...): (line 54)
    # Processing the call arguments (line 54)
    int_383300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_383301 = {}
    # Getting the type of 'np' (line 54)
    np_383297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 54)
    random_383298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), np_383297, 'random')
    # Obtaining the member 'permutation' of a type (line 54)
    permutation_383299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), random_383298, 'permutation')
    # Calling permutation(args, kwargs) (line 54)
    permutation_call_result_383302 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), permutation_383299, *[int_383300], **kwargs_383301)
    
    # Assigning a type to the variable 'rand_perm' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'rand_perm', permutation_call_result_383302)
    
    # Assigning a Call to a Name (line 55):
    
    # Call to permutation(...): (line 55)
    # Processing the call arguments (line 55)
    int_383306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_383307 = {}
    # Getting the type of 'np' (line 55)
    np_383303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'np', False)
    # Obtaining the member 'random' of a type (line 55)
    random_383304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), np_383303, 'random')
    # Obtaining the member 'permutation' of a type (line 55)
    permutation_383305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), random_383304, 'permutation')
    # Calling permutation(args, kwargs) (line 55)
    permutation_call_result_383308 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), permutation_383305, *[int_383306], **kwargs_383307)
    
    # Assigning a type to the variable 'rand_perm2' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'rand_perm2', permutation_call_result_383308)
    
    # Assigning a Call to a Name (line 57):
    
    # Call to arange(...): (line 57)
    # Processing the call arguments (line 57)
    int_383311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'int')
    # Processing the call keyword arguments (line 57)
    kwargs_383312 = {}
    # Getting the type of 'np' (line 57)
    np_383309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 57)
    arange_383310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), np_383309, 'arange')
    # Calling arange(args, kwargs) (line 57)
    arange_call_result_383313 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), arange_383310, *[int_383311], **kwargs_383312)
    
    # Assigning a type to the variable 'Rrow' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'Rrow', arange_call_result_383313)
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'rand_perm' (line 58)
    rand_perm_383314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'rand_perm')
    # Assigning a type to the variable 'Rcol' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'Rcol', rand_perm_383314)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to ones(...): (line 59)
    # Processing the call arguments (line 59)
    int_383317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'int')
    # Processing the call keyword arguments (line 59)
    # Getting the type of 'int' (line 59)
    int_383318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'int', False)
    keyword_383319 = int_383318
    kwargs_383320 = {'dtype': keyword_383319}
    # Getting the type of 'np' (line 59)
    np_383315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 59)
    ones_383316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), np_383315, 'ones')
    # Calling ones(args, kwargs) (line 59)
    ones_call_result_383321 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), ones_383316, *[int_383317], **kwargs_383320)
    
    # Assigning a type to the variable 'Rdata' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'Rdata', ones_call_result_383321)
    
    # Assigning a Call to a Name (line 60):
    
    # Call to tocsc(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_383331 = {}
    
    # Call to coo_matrix(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_383323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'Rdata' (line 60)
    Rdata_383324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'Rdata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_383323, Rdata_383324)
    # Adding element type (line 60)
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_383325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'Rrow' (line 60)
    Rrow_383326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 30), 'Rrow', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 30), tuple_383325, Rrow_383326)
    # Adding element type (line 60)
    # Getting the type of 'Rcol' (line 60)
    Rcol_383327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'Rcol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 30), tuple_383325, Rcol_383327)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_383323, tuple_383325)
    
    # Processing the call keyword arguments (line 60)
    kwargs_383328 = {}
    # Getting the type of 'coo_matrix' (line 60)
    coo_matrix_383322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 60)
    coo_matrix_call_result_383329 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), coo_matrix_383322, *[tuple_383323], **kwargs_383328)
    
    # Obtaining the member 'tocsc' of a type (line 60)
    tocsc_383330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), coo_matrix_call_result_383329, 'tocsc')
    # Calling tocsc(args, kwargs) (line 60)
    tocsc_call_result_383332 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), tocsc_383330, *[], **kwargs_383331)
    
    # Assigning a type to the variable 'Rmat' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'Rmat', tocsc_call_result_383332)
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'rand_perm2' (line 62)
    rand_perm2_383333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'rand_perm2')
    # Assigning a type to the variable 'Crow' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'Crow', rand_perm2_383333)
    
    # Assigning a Call to a Name (line 63):
    
    # Call to arange(...): (line 63)
    # Processing the call arguments (line 63)
    int_383336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
    # Processing the call keyword arguments (line 63)
    kwargs_383337 = {}
    # Getting the type of 'np' (line 63)
    np_383334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 63)
    arange_383335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 11), np_383334, 'arange')
    # Calling arange(args, kwargs) (line 63)
    arange_call_result_383338 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), arange_383335, *[int_383336], **kwargs_383337)
    
    # Assigning a type to the variable 'Ccol' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'Ccol', arange_call_result_383338)
    
    # Assigning a Call to a Name (line 64):
    
    # Call to ones(...): (line 64)
    # Processing the call arguments (line 64)
    int_383341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'int')
    # Processing the call keyword arguments (line 64)
    # Getting the type of 'int' (line 64)
    int_383342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'int', False)
    keyword_383343 = int_383342
    kwargs_383344 = {'dtype': keyword_383343}
    # Getting the type of 'np' (line 64)
    np_383339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 64)
    ones_383340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), np_383339, 'ones')
    # Calling ones(args, kwargs) (line 64)
    ones_call_result_383345 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), ones_383340, *[int_383341], **kwargs_383344)
    
    # Assigning a type to the variable 'Cdata' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'Cdata', ones_call_result_383345)
    
    # Assigning a Call to a Name (line 65):
    
    # Call to tocsc(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_383355 = {}
    
    # Call to coo_matrix(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_383347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    # Getting the type of 'Cdata' (line 65)
    Cdata_383348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'Cdata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 23), tuple_383347, Cdata_383348)
    # Adding element type (line 65)
    
    # Obtaining an instance of the builtin type 'tuple' (line 65)
    tuple_383349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 65)
    # Adding element type (line 65)
    # Getting the type of 'Crow' (line 65)
    Crow_383350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'Crow', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 30), tuple_383349, Crow_383350)
    # Adding element type (line 65)
    # Getting the type of 'Ccol' (line 65)
    Ccol_383351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'Ccol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 30), tuple_383349, Ccol_383351)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 23), tuple_383347, tuple_383349)
    
    # Processing the call keyword arguments (line 65)
    kwargs_383352 = {}
    # Getting the type of 'coo_matrix' (line 65)
    coo_matrix_383346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 65)
    coo_matrix_call_result_383353 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), coo_matrix_383346, *[tuple_383347], **kwargs_383352)
    
    # Obtaining the member 'tocsc' of a type (line 65)
    tocsc_383354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), coo_matrix_call_result_383353, 'tocsc')
    # Calling tocsc(args, kwargs) (line 65)
    tocsc_call_result_383356 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), tocsc_383354, *[], **kwargs_383355)
    
    # Assigning a type to the variable 'Cmat' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'Cmat', tocsc_call_result_383356)
    
    # Assigning a BinOp to a Name (line 67):
    # Getting the type of 'Rmat' (line 67)
    Rmat_383357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'Rmat')
    # Getting the type of 'A' (line 67)
    A_383358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'A')
    # Applying the binary operator '*' (line 67)
    result_mul_383359 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 8), '*', Rmat_383357, A_383358)
    
    # Getting the type of 'Cmat' (line 67)
    Cmat_383360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'Cmat')
    # Applying the binary operator '*' (line 67)
    result_mul_383361 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), '*', result_mul_383359, Cmat_383360)
    
    # Assigning a type to the variable 'B' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'B', result_mul_383361)
    
    # Assigning a Call to a Name (line 70):
    
    # Call to maximum_bipartite_matching(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'B' (line 70)
    B_383363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'B', False)
    # Processing the call keyword arguments (line 70)
    str_383364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 50), 'str', 'row')
    keyword_383365 = str_383364
    kwargs_383366 = {'perm_type': keyword_383365}
    # Getting the type of 'maximum_bipartite_matching' (line 70)
    maximum_bipartite_matching_383362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'maximum_bipartite_matching', False)
    # Calling maximum_bipartite_matching(args, kwargs) (line 70)
    maximum_bipartite_matching_call_result_383367 = invoke(stypy.reporting.localization.Localization(__file__, 70, 11), maximum_bipartite_matching_383362, *[B_383363], **kwargs_383366)
    
    # Assigning a type to the variable 'perm' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'perm', maximum_bipartite_matching_call_result_383367)
    
    # Assigning a Call to a Name (line 71):
    
    # Call to arange(...): (line 71)
    # Processing the call arguments (line 71)
    int_383370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_383371 = {}
    # Getting the type of 'np' (line 71)
    np_383368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 71)
    arange_383369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), np_383368, 'arange')
    # Calling arange(args, kwargs) (line 71)
    arange_call_result_383372 = invoke(stypy.reporting.localization.Localization(__file__, 71, 11), arange_383369, *[int_383370], **kwargs_383371)
    
    # Assigning a type to the variable 'Rrow' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'Rrow', arange_call_result_383372)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'perm' (line 72)
    perm_383373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'perm')
    # Assigning a type to the variable 'Rcol' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'Rcol', perm_383373)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to ones(...): (line 73)
    # Processing the call arguments (line 73)
    int_383376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'int')
    # Processing the call keyword arguments (line 73)
    # Getting the type of 'int' (line 73)
    int_383377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'int', False)
    keyword_383378 = int_383377
    kwargs_383379 = {'dtype': keyword_383378}
    # Getting the type of 'np' (line 73)
    np_383374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 73)
    ones_383375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), np_383374, 'ones')
    # Calling ones(args, kwargs) (line 73)
    ones_call_result_383380 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), ones_383375, *[int_383376], **kwargs_383379)
    
    # Assigning a type to the variable 'Rdata' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'Rdata', ones_call_result_383380)
    
    # Assigning a Call to a Name (line 74):
    
    # Call to tocsc(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_383390 = {}
    
    # Call to coo_matrix(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Obtaining an instance of the builtin type 'tuple' (line 74)
    tuple_383382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 74)
    # Adding element type (line 74)
    # Getting the type of 'Rdata' (line 74)
    Rdata_383383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'Rdata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), tuple_383382, Rdata_383383)
    # Adding element type (line 74)
    
    # Obtaining an instance of the builtin type 'tuple' (line 74)
    tuple_383384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 74)
    # Adding element type (line 74)
    # Getting the type of 'Rrow' (line 74)
    Rrow_383385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'Rrow', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 30), tuple_383384, Rrow_383385)
    # Adding element type (line 74)
    # Getting the type of 'Rcol' (line 74)
    Rcol_383386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'Rcol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 30), tuple_383384, Rcol_383386)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), tuple_383382, tuple_383384)
    
    # Processing the call keyword arguments (line 74)
    kwargs_383387 = {}
    # Getting the type of 'coo_matrix' (line 74)
    coo_matrix_383381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 74)
    coo_matrix_call_result_383388 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), coo_matrix_383381, *[tuple_383382], **kwargs_383387)
    
    # Obtaining the member 'tocsc' of a type (line 74)
    tocsc_383389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), coo_matrix_call_result_383388, 'tocsc')
    # Calling tocsc(args, kwargs) (line 74)
    tocsc_call_result_383391 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), tocsc_383389, *[], **kwargs_383390)
    
    # Assigning a type to the variable 'Rmat' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'Rmat', tocsc_call_result_383391)
    
    # Assigning a BinOp to a Name (line 75):
    # Getting the type of 'Rmat' (line 75)
    Rmat_383392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'Rmat')
    # Getting the type of 'B' (line 75)
    B_383393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'B')
    # Applying the binary operator '*' (line 75)
    result_mul_383394 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 9), '*', Rmat_383392, B_383393)
    
    # Assigning a type to the variable 'C1' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'C1', result_mul_383394)
    
    # Assigning a Call to a Name (line 78):
    
    # Call to maximum_bipartite_matching(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'B' (line 78)
    B_383396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 39), 'B', False)
    # Processing the call keyword arguments (line 78)
    str_383397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 51), 'str', 'column')
    keyword_383398 = str_383397
    kwargs_383399 = {'perm_type': keyword_383398}
    # Getting the type of 'maximum_bipartite_matching' (line 78)
    maximum_bipartite_matching_383395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'maximum_bipartite_matching', False)
    # Calling maximum_bipartite_matching(args, kwargs) (line 78)
    maximum_bipartite_matching_call_result_383400 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), maximum_bipartite_matching_383395, *[B_383396], **kwargs_383399)
    
    # Assigning a type to the variable 'perm2' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'perm2', maximum_bipartite_matching_call_result_383400)
    
    # Assigning a Name to a Name (line 79):
    # Getting the type of 'perm2' (line 79)
    perm2_383401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'perm2')
    # Assigning a type to the variable 'Crow' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'Crow', perm2_383401)
    
    # Assigning a Call to a Name (line 80):
    
    # Call to arange(...): (line 80)
    # Processing the call arguments (line 80)
    int_383404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'int')
    # Processing the call keyword arguments (line 80)
    kwargs_383405 = {}
    # Getting the type of 'np' (line 80)
    np_383402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 80)
    arange_383403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), np_383402, 'arange')
    # Calling arange(args, kwargs) (line 80)
    arange_call_result_383406 = invoke(stypy.reporting.localization.Localization(__file__, 80, 11), arange_383403, *[int_383404], **kwargs_383405)
    
    # Assigning a type to the variable 'Ccol' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'Ccol', arange_call_result_383406)
    
    # Assigning a Call to a Name (line 81):
    
    # Call to ones(...): (line 81)
    # Processing the call arguments (line 81)
    int_383409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'int')
    # Processing the call keyword arguments (line 81)
    # Getting the type of 'int' (line 81)
    int_383410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'int', False)
    keyword_383411 = int_383410
    kwargs_383412 = {'dtype': keyword_383411}
    # Getting the type of 'np' (line 81)
    np_383407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 81)
    ones_383408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), np_383407, 'ones')
    # Calling ones(args, kwargs) (line 81)
    ones_call_result_383413 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), ones_383408, *[int_383409], **kwargs_383412)
    
    # Assigning a type to the variable 'Cdata' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'Cdata', ones_call_result_383413)
    
    # Assigning a Call to a Name (line 82):
    
    # Call to tocsc(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_383423 = {}
    
    # Call to coo_matrix(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_383415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'Cdata' (line 82)
    Cdata_383416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'Cdata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_383415, Cdata_383416)
    # Adding element type (line 82)
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_383417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'Crow' (line 82)
    Crow_383418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'Crow', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 30), tuple_383417, Crow_383418)
    # Adding element type (line 82)
    # Getting the type of 'Ccol' (line 82)
    Ccol_383419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 35), 'Ccol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 30), tuple_383417, Ccol_383419)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_383415, tuple_383417)
    
    # Processing the call keyword arguments (line 82)
    kwargs_383420 = {}
    # Getting the type of 'coo_matrix' (line 82)
    coo_matrix_383414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 82)
    coo_matrix_call_result_383421 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), coo_matrix_383414, *[tuple_383415], **kwargs_383420)
    
    # Obtaining the member 'tocsc' of a type (line 82)
    tocsc_383422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 11), coo_matrix_call_result_383421, 'tocsc')
    # Calling tocsc(args, kwargs) (line 82)
    tocsc_call_result_383424 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), tocsc_383422, *[], **kwargs_383423)
    
    # Assigning a type to the variable 'Cmat' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'Cmat', tocsc_call_result_383424)
    
    # Assigning a BinOp to a Name (line 83):
    # Getting the type of 'B' (line 83)
    B_383425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'B')
    # Getting the type of 'Cmat' (line 83)
    Cmat_383426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'Cmat')
    # Applying the binary operator '*' (line 83)
    result_mul_383427 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 9), '*', B_383425, Cmat_383426)
    
    # Assigning a type to the variable 'C2' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'C2', result_mul_383427)
    
    # Call to assert_equal(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Call to any(...): (line 86)
    # Processing the call arguments (line 86)
    
    
    # Call to diagonal(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_383432 = {}
    # Getting the type of 'C1' (line 86)
    C1_383430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'C1', False)
    # Obtaining the member 'diagonal' of a type (line 86)
    diagonal_383431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 21), C1_383430, 'diagonal')
    # Calling diagonal(args, kwargs) (line 86)
    diagonal_call_result_383433 = invoke(stypy.reporting.localization.Localization(__file__, 86, 21), diagonal_383431, *[], **kwargs_383432)
    
    int_383434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 38), 'int')
    # Applying the binary operator '==' (line 86)
    result_eq_383435 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '==', diagonal_call_result_383433, int_383434)
    
    # Processing the call keyword arguments (line 86)
    kwargs_383436 = {}
    # Getting the type of 'any' (line 86)
    any_383429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 17), 'any', False)
    # Calling any(args, kwargs) (line 86)
    any_call_result_383437 = invoke(stypy.reporting.localization.Localization(__file__, 86, 17), any_383429, *[result_eq_383435], **kwargs_383436)
    
    # Getting the type of 'False' (line 86)
    False_383438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'False', False)
    # Processing the call keyword arguments (line 86)
    kwargs_383439 = {}
    # Getting the type of 'assert_equal' (line 86)
    assert_equal_383428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 86)
    assert_equal_call_result_383440 = invoke(stypy.reporting.localization.Localization(__file__, 86, 4), assert_equal_383428, *[any_call_result_383437, False_383438], **kwargs_383439)
    
    
    # Call to assert_equal(...): (line 87)
    # Processing the call arguments (line 87)
    
    # Call to any(...): (line 87)
    # Processing the call arguments (line 87)
    
    
    # Call to diagonal(...): (line 87)
    # Processing the call keyword arguments (line 87)
    kwargs_383445 = {}
    # Getting the type of 'C2' (line 87)
    C2_383443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'C2', False)
    # Obtaining the member 'diagonal' of a type (line 87)
    diagonal_383444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 21), C2_383443, 'diagonal')
    # Calling diagonal(args, kwargs) (line 87)
    diagonal_call_result_383446 = invoke(stypy.reporting.localization.Localization(__file__, 87, 21), diagonal_383444, *[], **kwargs_383445)
    
    int_383447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 38), 'int')
    # Applying the binary operator '==' (line 87)
    result_eq_383448 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 21), '==', diagonal_call_result_383446, int_383447)
    
    # Processing the call keyword arguments (line 87)
    kwargs_383449 = {}
    # Getting the type of 'any' (line 87)
    any_383442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'any', False)
    # Calling any(args, kwargs) (line 87)
    any_call_result_383450 = invoke(stypy.reporting.localization.Localization(__file__, 87, 17), any_383442, *[result_eq_383448], **kwargs_383449)
    
    # Getting the type of 'False' (line 87)
    False_383451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'False', False)
    # Processing the call keyword arguments (line 87)
    kwargs_383452 = {}
    # Getting the type of 'assert_equal' (line 87)
    assert_equal_383441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 87)
    assert_equal_call_result_383453 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), assert_equal_383441, *[any_call_result_383450, False_383451], **kwargs_383452)
    
    
    # Assigning a Call to a Attribute (line 90):
    
    # Call to astype(...): (line 90)
    # Processing the call arguments (line 90)
    str_383457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 33), 'str', 'int64')
    # Processing the call keyword arguments (line 90)
    kwargs_383458 = {}
    # Getting the type of 'B' (line 90)
    B_383454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'B', False)
    # Obtaining the member 'indices' of a type (line 90)
    indices_383455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 16), B_383454, 'indices')
    # Obtaining the member 'astype' of a type (line 90)
    astype_383456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 16), indices_383455, 'astype')
    # Calling astype(args, kwargs) (line 90)
    astype_call_result_383459 = invoke(stypy.reporting.localization.Localization(__file__, 90, 16), astype_383456, *[str_383457], **kwargs_383458)
    
    # Getting the type of 'B' (line 90)
    B_383460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'B')
    # Setting the type of the member 'indices' of a type (line 90)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), B_383460, 'indices', astype_call_result_383459)
    
    # Assigning a Call to a Attribute (line 91):
    
    # Call to astype(...): (line 91)
    # Processing the call arguments (line 91)
    str_383464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'str', 'int64')
    # Processing the call keyword arguments (line 91)
    kwargs_383465 = {}
    # Getting the type of 'B' (line 91)
    B_383461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'B', False)
    # Obtaining the member 'indptr' of a type (line 91)
    indptr_383462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), B_383461, 'indptr')
    # Obtaining the member 'astype' of a type (line 91)
    astype_383463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), indptr_383462, 'astype')
    # Calling astype(args, kwargs) (line 91)
    astype_call_result_383466 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), astype_383463, *[str_383464], **kwargs_383465)
    
    # Getting the type of 'B' (line 91)
    B_383467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'B')
    # Setting the type of the member 'indptr' of a type (line 91)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), B_383467, 'indptr', astype_call_result_383466)
    
    # Assigning a Call to a Name (line 92):
    
    # Call to maximum_bipartite_matching(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'B' (line 92)
    B_383469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'B', False)
    # Processing the call keyword arguments (line 92)
    str_383470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 50), 'str', 'row')
    keyword_383471 = str_383470
    kwargs_383472 = {'perm_type': keyword_383471}
    # Getting the type of 'maximum_bipartite_matching' (line 92)
    maximum_bipartite_matching_383468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'maximum_bipartite_matching', False)
    # Calling maximum_bipartite_matching(args, kwargs) (line 92)
    maximum_bipartite_matching_call_result_383473 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), maximum_bipartite_matching_383468, *[B_383469], **kwargs_383472)
    
    # Assigning a type to the variable 'perm' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'perm', maximum_bipartite_matching_call_result_383473)
    
    # Assigning a Call to a Name (line 93):
    
    # Call to arange(...): (line 93)
    # Processing the call arguments (line 93)
    int_383476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'int')
    # Processing the call keyword arguments (line 93)
    kwargs_383477 = {}
    # Getting the type of 'np' (line 93)
    np_383474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 93)
    arange_383475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), np_383474, 'arange')
    # Calling arange(args, kwargs) (line 93)
    arange_call_result_383478 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), arange_383475, *[int_383476], **kwargs_383477)
    
    # Assigning a type to the variable 'Rrow' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'Rrow', arange_call_result_383478)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'perm' (line 94)
    perm_383479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'perm')
    # Assigning a type to the variable 'Rcol' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'Rcol', perm_383479)
    
    # Assigning a Call to a Name (line 95):
    
    # Call to ones(...): (line 95)
    # Processing the call arguments (line 95)
    int_383482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'int')
    # Processing the call keyword arguments (line 95)
    # Getting the type of 'int' (line 95)
    int_383483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'int', False)
    keyword_383484 = int_383483
    kwargs_383485 = {'dtype': keyword_383484}
    # Getting the type of 'np' (line 95)
    np_383480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 95)
    ones_383481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), np_383480, 'ones')
    # Calling ones(args, kwargs) (line 95)
    ones_call_result_383486 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), ones_383481, *[int_383482], **kwargs_383485)
    
    # Assigning a type to the variable 'Rdata' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'Rdata', ones_call_result_383486)
    
    # Assigning a Call to a Name (line 96):
    
    # Call to tocsc(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_383496 = {}
    
    # Call to coo_matrix(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Obtaining an instance of the builtin type 'tuple' (line 96)
    tuple_383488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'Rdata' (line 96)
    Rdata_383489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'Rdata', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_383488, Rdata_383489)
    # Adding element type (line 96)
    
    # Obtaining an instance of the builtin type 'tuple' (line 96)
    tuple_383490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'Rrow' (line 96)
    Rrow_383491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'Rrow', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 30), tuple_383490, Rrow_383491)
    # Adding element type (line 96)
    # Getting the type of 'Rcol' (line 96)
    Rcol_383492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 35), 'Rcol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 30), tuple_383490, Rcol_383492)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 23), tuple_383488, tuple_383490)
    
    # Processing the call keyword arguments (line 96)
    kwargs_383493 = {}
    # Getting the type of 'coo_matrix' (line 96)
    coo_matrix_383487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 96)
    coo_matrix_call_result_383494 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), coo_matrix_383487, *[tuple_383488], **kwargs_383493)
    
    # Obtaining the member 'tocsc' of a type (line 96)
    tocsc_383495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), coo_matrix_call_result_383494, 'tocsc')
    # Calling tocsc(args, kwargs) (line 96)
    tocsc_call_result_383497 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), tocsc_383495, *[], **kwargs_383496)
    
    # Assigning a type to the variable 'Rmat' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'Rmat', tocsc_call_result_383497)
    
    # Assigning a BinOp to a Name (line 97):
    # Getting the type of 'Rmat' (line 97)
    Rmat_383498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 9), 'Rmat')
    # Getting the type of 'B' (line 97)
    B_383499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'B')
    # Applying the binary operator '*' (line 97)
    result_mul_383500 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 9), '*', Rmat_383498, B_383499)
    
    # Assigning a type to the variable 'C3' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'C3', result_mul_383500)
    
    # Call to assert_equal(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Call to any(...): (line 98)
    # Processing the call arguments (line 98)
    
    
    # Call to diagonal(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_383505 = {}
    # Getting the type of 'C3' (line 98)
    C3_383503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'C3', False)
    # Obtaining the member 'diagonal' of a type (line 98)
    diagonal_383504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 21), C3_383503, 'diagonal')
    # Calling diagonal(args, kwargs) (line 98)
    diagonal_call_result_383506 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), diagonal_383504, *[], **kwargs_383505)
    
    int_383507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 38), 'int')
    # Applying the binary operator '==' (line 98)
    result_eq_383508 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 21), '==', diagonal_call_result_383506, int_383507)
    
    # Processing the call keyword arguments (line 98)
    kwargs_383509 = {}
    # Getting the type of 'any' (line 98)
    any_383502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'any', False)
    # Calling any(args, kwargs) (line 98)
    any_call_result_383510 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), any_383502, *[result_eq_383508], **kwargs_383509)
    
    # Getting the type of 'False' (line 98)
    False_383511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'False', False)
    # Processing the call keyword arguments (line 98)
    kwargs_383512 = {}
    # Getting the type of 'assert_equal' (line 98)
    assert_equal_383501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 98)
    assert_equal_call_result_383513 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), assert_equal_383501, *[any_call_result_383510, False_383511], **kwargs_383512)
    
    
    # ################# End of 'test_graph_maximum_bipartite_matching(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_maximum_bipartite_matching' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_383514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_383514)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_maximum_bipartite_matching'
    return stypy_return_type_383514

# Assigning a type to the variable 'test_graph_maximum_bipartite_matching' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'test_graph_maximum_bipartite_matching', test_graph_maximum_bipartite_matching)

@norecursion
def test_graph_structural_rank(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_structural_rank'
    module_type_store = module_type_store.open_function_context('test_graph_structural_rank', 101, 0, False)
    
    # Passed parameters checking function
    test_graph_structural_rank.stypy_localization = localization
    test_graph_structural_rank.stypy_type_of_self = None
    test_graph_structural_rank.stypy_type_store = module_type_store
    test_graph_structural_rank.stypy_function_name = 'test_graph_structural_rank'
    test_graph_structural_rank.stypy_param_names_list = []
    test_graph_structural_rank.stypy_varargs_param_name = None
    test_graph_structural_rank.stypy_kwargs_param_name = None
    test_graph_structural_rank.stypy_call_defaults = defaults
    test_graph_structural_rank.stypy_call_varargs = varargs
    test_graph_structural_rank.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_structural_rank', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_structural_rank', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_structural_rank(...)' code ##################

    
    # Assigning a Call to a Name (line 103):
    
    # Call to csc_matrix(...): (line 103)
    # Processing the call arguments (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_383516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_383517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    int_383518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), list_383517, int_383518)
    # Adding element type (line 103)
    int_383519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), list_383517, int_383519)
    # Adding element type (line 103)
    int_383520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), list_383517, int_383520)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), list_383516, list_383517)
    # Adding element type (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_383521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    int_383522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), list_383521, int_383522)
    # Adding element type (line 104)
    int_383523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), list_383521, int_383523)
    # Adding element type (line 104)
    int_383524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 20), list_383521, int_383524)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), list_383516, list_383521)
    # Adding element type (line 103)
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_383525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    int_383526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_383525, int_383526)
    # Adding element type (line 105)
    int_383527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_383525, int_383527)
    # Adding element type (line 105)
    int_383528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_383525, int_383528)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), list_383516, list_383525)
    
    # Processing the call keyword arguments (line 103)
    kwargs_383529 = {}
    # Getting the type of 'csc_matrix' (line 103)
    csc_matrix_383515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 103)
    csc_matrix_call_result_383530 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), csc_matrix_383515, *[list_383516], **kwargs_383529)
    
    # Assigning a type to the variable 'A' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'A', csc_matrix_call_result_383530)
    
    # Call to assert_equal(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Call to structural_rank(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'A' (line 106)
    A_383533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'A', False)
    # Processing the call keyword arguments (line 106)
    kwargs_383534 = {}
    # Getting the type of 'structural_rank' (line 106)
    structural_rank_383532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'structural_rank', False)
    # Calling structural_rank(args, kwargs) (line 106)
    structural_rank_call_result_383535 = invoke(stypy.reporting.localization.Localization(__file__, 106, 17), structural_rank_383532, *[A_383533], **kwargs_383534)
    
    int_383536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'int')
    # Processing the call keyword arguments (line 106)
    kwargs_383537 = {}
    # Getting the type of 'assert_equal' (line 106)
    assert_equal_383531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 106)
    assert_equal_call_result_383538 = invoke(stypy.reporting.localization.Localization(__file__, 106, 4), assert_equal_383531, *[structural_rank_call_result_383535, int_383536], **kwargs_383537)
    
    
    # Assigning a Call to a Name (line 109):
    
    # Call to array(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_383541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    int_383542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383542)
    # Adding element type (line 109)
    int_383543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383543)
    # Adding element type (line 109)
    int_383544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383544)
    # Adding element type (line 109)
    int_383545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383545)
    # Adding element type (line 109)
    int_383546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383546)
    # Adding element type (line 109)
    int_383547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383547)
    # Adding element type (line 109)
    int_383548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383548)
    # Adding element type (line 109)
    int_383549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383549)
    # Adding element type (line 109)
    int_383550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383550)
    # Adding element type (line 109)
    int_383551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383551)
    # Adding element type (line 109)
    int_383552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383552)
    # Adding element type (line 109)
    int_383553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383553)
    # Adding element type (line 109)
    int_383554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383554)
    # Adding element type (line 109)
    int_383555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383555)
    # Adding element type (line 109)
    int_383556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383556)
    # Adding element type (line 109)
    int_383557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383557)
    # Adding element type (line 109)
    int_383558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383558)
    # Adding element type (line 109)
    int_383559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383559)
    # Adding element type (line 109)
    int_383560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383560)
    # Adding element type (line 109)
    int_383561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383561)
    # Adding element type (line 109)
    int_383562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383562)
    # Adding element type (line 109)
    int_383563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 63), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383563)
    # Adding element type (line 109)
    int_383564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 20), list_383541, int_383564)
    
    # Processing the call keyword arguments (line 109)
    kwargs_383565 = {}
    # Getting the type of 'np' (line 109)
    np_383539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 109)
    array_383540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), np_383539, 'array')
    # Calling array(args, kwargs) (line 109)
    array_call_result_383566 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), array_383540, *[list_383541], **kwargs_383565)
    
    # Assigning a type to the variable 'rows' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'rows', array_call_result_383566)
    
    # Assigning a Call to a Name (line 110):
    
    # Call to array(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_383569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    int_383570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383570)
    # Adding element type (line 110)
    int_383571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383571)
    # Adding element type (line 110)
    int_383572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383572)
    # Adding element type (line 110)
    int_383573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383573)
    # Adding element type (line 110)
    int_383574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383574)
    # Adding element type (line 110)
    int_383575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383575)
    # Adding element type (line 110)
    int_383576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383576)
    # Adding element type (line 110)
    int_383577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383577)
    # Adding element type (line 110)
    int_383578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383578)
    # Adding element type (line 110)
    int_383579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383579)
    # Adding element type (line 110)
    int_383580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383580)
    # Adding element type (line 110)
    int_383581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383581)
    # Adding element type (line 110)
    int_383582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383582)
    # Adding element type (line 110)
    int_383583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383583)
    # Adding element type (line 110)
    int_383584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383584)
    # Adding element type (line 110)
    int_383585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383585)
    # Adding element type (line 110)
    int_383586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383586)
    # Adding element type (line 110)
    int_383587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383587)
    # Adding element type (line 110)
    int_383588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383588)
    # Adding element type (line 110)
    int_383589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383589)
    # Adding element type (line 110)
    int_383590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383590)
    # Adding element type (line 110)
    int_383591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 63), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383591)
    # Adding element type (line 110)
    int_383592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 20), list_383569, int_383592)
    
    # Processing the call keyword arguments (line 110)
    kwargs_383593 = {}
    # Getting the type of 'np' (line 110)
    np_383567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 110)
    array_383568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), np_383567, 'array')
    # Calling array(args, kwargs) (line 110)
    array_call_result_383594 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), array_383568, *[list_383569], **kwargs_383593)
    
    # Assigning a type to the variable 'cols' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'cols', array_call_result_383594)
    
    # Assigning a Call to a Name (line 111):
    
    # Call to ones_like(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'rows' (line 111)
    rows_383597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'rows', False)
    # Processing the call keyword arguments (line 111)
    kwargs_383598 = {}
    # Getting the type of 'np' (line 111)
    np_383595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 111)
    ones_like_383596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), np_383595, 'ones_like')
    # Calling ones_like(args, kwargs) (line 111)
    ones_like_call_result_383599 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), ones_like_383596, *[rows_383597], **kwargs_383598)
    
    # Assigning a type to the variable 'data' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'data', ones_like_call_result_383599)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to coo_matrix(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_383601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'data' (line 112)
    data_383602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'data', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), tuple_383601, data_383602)
    # Adding element type (line 112)
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_383603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'rows' (line 112)
    rows_383604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'rows', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 26), tuple_383603, rows_383604)
    # Adding element type (line 112)
    # Getting the type of 'cols' (line 112)
    cols_383605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'cols', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 26), tuple_383603, cols_383605)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), tuple_383601, tuple_383603)
    
    # Processing the call keyword arguments (line 112)
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_383606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    int_383607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 46), tuple_383606, int_383607)
    # Adding element type (line 112)
    int_383608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 46), tuple_383606, int_383608)
    
    keyword_383609 = tuple_383606
    kwargs_383610 = {'shape': keyword_383609}
    # Getting the type of 'coo_matrix' (line 112)
    coo_matrix_383600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'coo_matrix', False)
    # Calling coo_matrix(args, kwargs) (line 112)
    coo_matrix_call_result_383611 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), coo_matrix_383600, *[tuple_383601], **kwargs_383610)
    
    # Assigning a type to the variable 'B' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'B', coo_matrix_call_result_383611)
    
    # Call to assert_equal(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to structural_rank(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'B' (line 113)
    B_383614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'B', False)
    # Processing the call keyword arguments (line 113)
    kwargs_383615 = {}
    # Getting the type of 'structural_rank' (line 113)
    structural_rank_383613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'structural_rank', False)
    # Calling structural_rank(args, kwargs) (line 113)
    structural_rank_call_result_383616 = invoke(stypy.reporting.localization.Localization(__file__, 113, 17), structural_rank_383613, *[B_383614], **kwargs_383615)
    
    int_383617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'int')
    # Processing the call keyword arguments (line 113)
    kwargs_383618 = {}
    # Getting the type of 'assert_equal' (line 113)
    assert_equal_383612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 113)
    assert_equal_call_result_383619 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), assert_equal_383612, *[structural_rank_call_result_383616, int_383617], **kwargs_383618)
    
    
    # Assigning a Call to a Name (line 116):
    
    # Call to csc_matrix(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Obtaining an instance of the builtin type 'list' (line 116)
    list_383621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 116)
    # Adding element type (line 116)
    
    # Obtaining an instance of the builtin type 'list' (line 116)
    list_383622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 116)
    # Adding element type (line 116)
    int_383623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_383622, int_383623)
    # Adding element type (line 116)
    int_383624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_383622, int_383624)
    # Adding element type (line 116)
    int_383625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_383622, int_383625)
    # Adding element type (line 116)
    int_383626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_383622, int_383626)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_383621, list_383622)
    # Adding element type (line 116)
    
    # Obtaining an instance of the builtin type 'list' (line 117)
    list_383627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 117)
    # Adding element type (line 117)
    int_383628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_383627, int_383628)
    # Adding element type (line 117)
    int_383629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_383627, int_383629)
    # Adding element type (line 117)
    int_383630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_383627, int_383630)
    # Adding element type (line 117)
    int_383631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 20), list_383627, int_383631)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_383621, list_383627)
    
    # Processing the call keyword arguments (line 116)
    kwargs_383632 = {}
    # Getting the type of 'csc_matrix' (line 116)
    csc_matrix_383620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 116)
    csc_matrix_call_result_383633 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), csc_matrix_383620, *[list_383621], **kwargs_383632)
    
    # Assigning a type to the variable 'C' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'C', csc_matrix_call_result_383633)
    
    # Call to assert_equal(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to structural_rank(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'C' (line 118)
    C_383636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'C', False)
    # Processing the call keyword arguments (line 118)
    kwargs_383637 = {}
    # Getting the type of 'structural_rank' (line 118)
    structural_rank_383635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'structural_rank', False)
    # Calling structural_rank(args, kwargs) (line 118)
    structural_rank_call_result_383638 = invoke(stypy.reporting.localization.Localization(__file__, 118, 17), structural_rank_383635, *[C_383636], **kwargs_383637)
    
    int_383639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'int')
    # Processing the call keyword arguments (line 118)
    kwargs_383640 = {}
    # Getting the type of 'assert_equal' (line 118)
    assert_equal_383634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 118)
    assert_equal_call_result_383641 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), assert_equal_383634, *[structural_rank_call_result_383638, int_383639], **kwargs_383640)
    
    
    # Call to assert_equal(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Call to structural_rank(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'C' (line 121)
    C_383644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 33), 'C', False)
    # Obtaining the member 'T' of a type (line 121)
    T_383645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 33), C_383644, 'T')
    # Processing the call keyword arguments (line 121)
    kwargs_383646 = {}
    # Getting the type of 'structural_rank' (line 121)
    structural_rank_383643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'structural_rank', False)
    # Calling structural_rank(args, kwargs) (line 121)
    structural_rank_call_result_383647 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), structural_rank_383643, *[T_383645], **kwargs_383646)
    
    int_383648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'int')
    # Processing the call keyword arguments (line 121)
    kwargs_383649 = {}
    # Getting the type of 'assert_equal' (line 121)
    assert_equal_383642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 121)
    assert_equal_call_result_383650 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), assert_equal_383642, *[structural_rank_call_result_383647, int_383648], **kwargs_383649)
    
    
    # ################# End of 'test_graph_structural_rank(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_structural_rank' in the type store
    # Getting the type of 'stypy_return_type' (line 101)
    stypy_return_type_383651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_383651)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_structural_rank'
    return stypy_return_type_383651

# Assigning a type to the variable 'test_graph_structural_rank' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'test_graph_structural_rank', test_graph_structural_rank)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
