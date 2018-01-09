
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: r'''
2: ==============================================================
3: Compressed Sparse Graph Routines (:mod:`scipy.sparse.csgraph`)
4: ==============================================================
5: 
6: .. currentmodule:: scipy.sparse.csgraph
7: 
8: Fast graph algorithms based on sparse matrix representations.
9: 
10: Contents
11: ========
12: 
13: .. autosummary::
14:    :toctree: generated/
15: 
16:    connected_components -- determine connected components of a graph
17:    laplacian -- compute the laplacian of a graph
18:    shortest_path -- compute the shortest path between points on a positive graph
19:    dijkstra -- use Dijkstra's algorithm for shortest path
20:    floyd_warshall -- use the Floyd-Warshall algorithm for shortest path
21:    bellman_ford -- use the Bellman-Ford algorithm for shortest path
22:    johnson -- use Johnson's algorithm for shortest path
23:    breadth_first_order -- compute a breadth-first order of nodes
24:    depth_first_order -- compute a depth-first order of nodes
25:    breadth_first_tree -- construct the breadth-first tree from a given node
26:    depth_first_tree -- construct a depth-first tree from a given node
27:    minimum_spanning_tree -- construct the minimum spanning tree of a graph
28:    reverse_cuthill_mckee -- compute permutation for reverse Cuthill-McKee ordering
29:    maximum_bipartite_matching -- compute permutation to make diagonal zero free
30:    structural_rank -- compute the structural rank of a graph
31:    NegativeCycleError
32: 
33: .. autosummary::
34:    :toctree: generated/
35: 
36:    construct_dist_matrix
37:    csgraph_from_dense
38:    csgraph_from_masked
39:    csgraph_masked_from_dense
40:    csgraph_to_dense
41:    csgraph_to_masked
42:    reconstruct_path
43: 
44: Graph Representations
45: =====================
46: This module uses graphs which are stored in a matrix format.  A
47: graph with N nodes can be represented by an (N x N) adjacency matrix G.
48: If there is a connection from node i to node j, then G[i, j] = w, where
49: w is the weight of the connection.  For nodes i and j which are
50: not connected, the value depends on the representation:
51: 
52: - for dense array representations, non-edges are represented by
53:   G[i, j] = 0, infinity, or NaN.
54: 
55: - for dense masked representations (of type np.ma.MaskedArray), non-edges
56:   are represented by masked values.  This can be useful when graphs with
57:   zero-weight edges are desired.
58: 
59: - for sparse array representations, non-edges are represented by
60:   non-entries in the matrix.  This sort of sparse representation also
61:   allows for edges with zero weights.
62: 
63: As a concrete example, imagine that you would like to represent the following
64: undirected graph::
65: 
66:               G
67: 
68:              (0)
69:             /   \
70:            1     2
71:           /       \
72:         (2)       (1)
73: 
74: This graph has three nodes, where node 0 and 1 are connected by an edge of
75: weight 2, and nodes 0 and 2 are connected by an edge of weight 1.
76: We can construct the dense, masked, and sparse representations as follows,
77: keeping in mind that an undirected graph is represented by a symmetric matrix::
78: 
79:     >>> G_dense = np.array([[0, 2, 1],
80:     ...                     [2, 0, 0],
81:     ...                     [1, 0, 0]])
82:     >>> G_masked = np.ma.masked_values(G_dense, 0)
83:     >>> from scipy.sparse import csr_matrix
84:     >>> G_sparse = csr_matrix(G_dense)
85: 
86: This becomes more difficult when zero edges are significant.  For example,
87: consider the situation when we slightly modify the above graph::
88: 
89:              G2
90: 
91:              (0)
92:             /   \
93:            0     2
94:           /       \
95:         (2)       (1)
96: 
97: This is identical to the previous graph, except nodes 0 and 2 are connected
98: by an edge of zero weight.  In this case, the dense representation above
99: leads to ambiguities: how can non-edges be represented if zero is a meaningful
100: value?  In this case, either a masked or sparse representation must be used
101: to eliminate the ambiguity::
102: 
103:     >>> G2_data = np.array([[np.inf, 2,      0     ],
104:     ...                     [2,      np.inf, np.inf],
105:     ...                     [0,      np.inf, np.inf]])
106:     >>> G2_masked = np.ma.masked_invalid(G2_data)
107:     >>> from scipy.sparse.csgraph import csgraph_from_dense
108:     >>> # G2_sparse = csr_matrix(G2_data) would give the wrong result
109:     >>> G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
110:     >>> G2_sparse.data
111:     array([ 2.,  0.,  2.,  0.])
112: 
113: Here we have used a utility routine from the csgraph submodule in order to
114: convert the dense representation to a sparse representation which can be
115: understood by the algorithms in submodule.  By viewing the data array, we
116: can see that the zero values are explicitly encoded in the graph.
117: 
118: Directed vs. Undirected
119: -----------------------
120: Matrices may represent either directed or undirected graphs.  This is
121: specified throughout the csgraph module by a boolean keyword.  Graphs are
122: assumed to be directed by default. In a directed graph, traversal from node
123: i to node j can be accomplished over the edge G[i, j], but not the edge
124: G[j, i].  In a non-directed graph, traversal from node i to node j can be
125: accomplished over either G[i, j] or G[j, i].  If both edges are not null,
126: and the two have unequal weights, then the smaller of the two is used.
127: Note that a symmetric matrix will represent an undirected graph, regardless
128: of whether the 'directed' keyword is set to True or False.  In this case,
129: using ``directed=True`` generally leads to more efficient computation.
130: 
131: The routines in this module accept as input either scipy.sparse representations
132: (csr, csc, or lil format), masked representations, or dense representations
133: with non-edges indicated by zeros, infinities, and NaN entries.
134: '''
135: 
136: from __future__ import division, print_function, absolute_import
137: 
138: __docformat__ = "restructuredtext en"
139: 
140: __all__ = ['connected_components',
141:            'laplacian',
142:            'shortest_path',
143:            'floyd_warshall',
144:            'dijkstra',
145:            'bellman_ford',
146:            'johnson',
147:            'breadth_first_order',
148:            'depth_first_order',
149:            'breadth_first_tree',
150:            'depth_first_tree',
151:            'minimum_spanning_tree',
152:            'reverse_cuthill_mckee',
153:            'maximum_bipartite_matching',
154:            'structural_rank',
155:            'construct_dist_matrix',
156:            'reconstruct_path',
157:            'csgraph_masked_from_dense',
158:            'csgraph_from_dense',
159:            'csgraph_from_masked',
160:            'csgraph_to_dense',
161:            'csgraph_to_masked',
162:            'NegativeCycleError']
163: 
164: from ._laplacian import laplacian
165: from ._shortest_path import shortest_path, floyd_warshall, dijkstra,\
166:     bellman_ford, johnson, NegativeCycleError
167: from ._traversal import breadth_first_order, depth_first_order, \
168:     breadth_first_tree, depth_first_tree, connected_components
169: from ._min_spanning_tree import minimum_spanning_tree
170: from ._reordering import reverse_cuthill_mckee, maximum_bipartite_matching, \
171:     structural_rank
172: from ._tools import construct_dist_matrix, reconstruct_path,\
173:     csgraph_from_dense, csgraph_to_dense, csgraph_masked_from_dense,\
174:     csgraph_from_masked, csgraph_to_masked
175: 
176: from scipy._lib._testutils import PytestTester
177: test = PytestTester(__name__)
178: del PytestTester
179: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_381638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', "\n==============================================================\nCompressed Sparse Graph Routines (:mod:`scipy.sparse.csgraph`)\n==============================================================\n\n.. currentmodule:: scipy.sparse.csgraph\n\nFast graph algorithms based on sparse matrix representations.\n\nContents\n========\n\n.. autosummary::\n   :toctree: generated/\n\n   connected_components -- determine connected components of a graph\n   laplacian -- compute the laplacian of a graph\n   shortest_path -- compute the shortest path between points on a positive graph\n   dijkstra -- use Dijkstra's algorithm for shortest path\n   floyd_warshall -- use the Floyd-Warshall algorithm for shortest path\n   bellman_ford -- use the Bellman-Ford algorithm for shortest path\n   johnson -- use Johnson's algorithm for shortest path\n   breadth_first_order -- compute a breadth-first order of nodes\n   depth_first_order -- compute a depth-first order of nodes\n   breadth_first_tree -- construct the breadth-first tree from a given node\n   depth_first_tree -- construct a depth-first tree from a given node\n   minimum_spanning_tree -- construct the minimum spanning tree of a graph\n   reverse_cuthill_mckee -- compute permutation for reverse Cuthill-McKee ordering\n   maximum_bipartite_matching -- compute permutation to make diagonal zero free\n   structural_rank -- compute the structural rank of a graph\n   NegativeCycleError\n\n.. autosummary::\n   :toctree: generated/\n\n   construct_dist_matrix\n   csgraph_from_dense\n   csgraph_from_masked\n   csgraph_masked_from_dense\n   csgraph_to_dense\n   csgraph_to_masked\n   reconstruct_path\n\nGraph Representations\n=====================\nThis module uses graphs which are stored in a matrix format.  A\ngraph with N nodes can be represented by an (N x N) adjacency matrix G.\nIf there is a connection from node i to node j, then G[i, j] = w, where\nw is the weight of the connection.  For nodes i and j which are\nnot connected, the value depends on the representation:\n\n- for dense array representations, non-edges are represented by\n  G[i, j] = 0, infinity, or NaN.\n\n- for dense masked representations (of type np.ma.MaskedArray), non-edges\n  are represented by masked values.  This can be useful when graphs with\n  zero-weight edges are desired.\n\n- for sparse array representations, non-edges are represented by\n  non-entries in the matrix.  This sort of sparse representation also\n  allows for edges with zero weights.\n\nAs a concrete example, imagine that you would like to represent the following\nundirected graph::\n\n              G\n\n             (0)\n            /   \\\n           1     2\n          /       \\\n        (2)       (1)\n\nThis graph has three nodes, where node 0 and 1 are connected by an edge of\nweight 2, and nodes 0 and 2 are connected by an edge of weight 1.\nWe can construct the dense, masked, and sparse representations as follows,\nkeeping in mind that an undirected graph is represented by a symmetric matrix::\n\n    >>> G_dense = np.array([[0, 2, 1],\n    ...                     [2, 0, 0],\n    ...                     [1, 0, 0]])\n    >>> G_masked = np.ma.masked_values(G_dense, 0)\n    >>> from scipy.sparse import csr_matrix\n    >>> G_sparse = csr_matrix(G_dense)\n\nThis becomes more difficult when zero edges are significant.  For example,\nconsider the situation when we slightly modify the above graph::\n\n             G2\n\n             (0)\n            /   \\\n           0     2\n          /       \\\n        (2)       (1)\n\nThis is identical to the previous graph, except nodes 0 and 2 are connected\nby an edge of zero weight.  In this case, the dense representation above\nleads to ambiguities: how can non-edges be represented if zero is a meaningful\nvalue?  In this case, either a masked or sparse representation must be used\nto eliminate the ambiguity::\n\n    >>> G2_data = np.array([[np.inf, 2,      0     ],\n    ...                     [2,      np.inf, np.inf],\n    ...                     [0,      np.inf, np.inf]])\n    >>> G2_masked = np.ma.masked_invalid(G2_data)\n    >>> from scipy.sparse.csgraph import csgraph_from_dense\n    >>> # G2_sparse = csr_matrix(G2_data) would give the wrong result\n    >>> G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)\n    >>> G2_sparse.data\n    array([ 2.,  0.,  2.,  0.])\n\nHere we have used a utility routine from the csgraph submodule in order to\nconvert the dense representation to a sparse representation which can be\nunderstood by the algorithms in submodule.  By viewing the data array, we\ncan see that the zero values are explicitly encoded in the graph.\n\nDirected vs. Undirected\n-----------------------\nMatrices may represent either directed or undirected graphs.  This is\nspecified throughout the csgraph module by a boolean keyword.  Graphs are\nassumed to be directed by default. In a directed graph, traversal from node\ni to node j can be accomplished over the edge G[i, j], but not the edge\nG[j, i].  In a non-directed graph, traversal from node i to node j can be\naccomplished over either G[i, j] or G[j, i].  If both edges are not null,\nand the two have unequal weights, then the smaller of the two is used.\nNote that a symmetric matrix will represent an undirected graph, regardless\nof whether the 'directed' keyword is set to True or False.  In this case,\nusing ``directed=True`` generally leads to more efficient computation.\n\nThe routines in this module accept as input either scipy.sparse representations\n(csr, csc, or lil format), masked representations, or dense representations\nwith non-edges indicated by zeros, infinities, and NaN entries.\n")

# Assigning a Str to a Name (line 138):
str_381639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), '__docformat__', str_381639)

# Assigning a List to a Name (line 140):
__all__ = ['connected_components', 'laplacian', 'shortest_path', 'floyd_warshall', 'dijkstra', 'bellman_ford', 'johnson', 'breadth_first_order', 'depth_first_order', 'breadth_first_tree', 'depth_first_tree', 'minimum_spanning_tree', 'reverse_cuthill_mckee', 'maximum_bipartite_matching', 'structural_rank', 'construct_dist_matrix', 'reconstruct_path', 'csgraph_masked_from_dense', 'csgraph_from_dense', 'csgraph_from_masked', 'csgraph_to_dense', 'csgraph_to_masked', 'NegativeCycleError']
module_type_store.set_exportable_members(['connected_components', 'laplacian', 'shortest_path', 'floyd_warshall', 'dijkstra', 'bellman_ford', 'johnson', 'breadth_first_order', 'depth_first_order', 'breadth_first_tree', 'depth_first_tree', 'minimum_spanning_tree', 'reverse_cuthill_mckee', 'maximum_bipartite_matching', 'structural_rank', 'construct_dist_matrix', 'reconstruct_path', 'csgraph_masked_from_dense', 'csgraph_from_dense', 'csgraph_from_masked', 'csgraph_to_dense', 'csgraph_to_masked', 'NegativeCycleError'])

# Obtaining an instance of the builtin type 'list' (line 140)
list_381640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 140)
# Adding element type (line 140)
str_381641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 11), 'str', 'connected_components')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381641)
# Adding element type (line 140)
str_381642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 11), 'str', 'laplacian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381642)
# Adding element type (line 140)
str_381643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 11), 'str', 'shortest_path')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381643)
# Adding element type (line 140)
str_381644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 11), 'str', 'floyd_warshall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381644)
# Adding element type (line 140)
str_381645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 11), 'str', 'dijkstra')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381645)
# Adding element type (line 140)
str_381646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 11), 'str', 'bellman_ford')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381646)
# Adding element type (line 140)
str_381647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 11), 'str', 'johnson')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381647)
# Adding element type (line 140)
str_381648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 11), 'str', 'breadth_first_order')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381648)
# Adding element type (line 140)
str_381649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 11), 'str', 'depth_first_order')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381649)
# Adding element type (line 140)
str_381650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 11), 'str', 'breadth_first_tree')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381650)
# Adding element type (line 140)
str_381651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 11), 'str', 'depth_first_tree')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381651)
# Adding element type (line 140)
str_381652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 11), 'str', 'minimum_spanning_tree')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381652)
# Adding element type (line 140)
str_381653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 11), 'str', 'reverse_cuthill_mckee')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381653)
# Adding element type (line 140)
str_381654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 11), 'str', 'maximum_bipartite_matching')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381654)
# Adding element type (line 140)
str_381655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 11), 'str', 'structural_rank')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381655)
# Adding element type (line 140)
str_381656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 11), 'str', 'construct_dist_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381656)
# Adding element type (line 140)
str_381657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 11), 'str', 'reconstruct_path')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381657)
# Adding element type (line 140)
str_381658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 11), 'str', 'csgraph_masked_from_dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381658)
# Adding element type (line 140)
str_381659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 11), 'str', 'csgraph_from_dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381659)
# Adding element type (line 140)
str_381660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 11), 'str', 'csgraph_from_masked')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381660)
# Adding element type (line 140)
str_381661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 11), 'str', 'csgraph_to_dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381661)
# Adding element type (line 140)
str_381662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 11), 'str', 'csgraph_to_masked')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381662)
# Adding element type (line 140)
str_381663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 11), 'str', 'NegativeCycleError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 10), list_381640, str_381663)

# Assigning a type to the variable '__all__' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), '__all__', list_381640)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 164, 0))

# 'from scipy.sparse.csgraph._laplacian import laplacian' statement (line 164)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381664 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.sparse.csgraph._laplacian')

if (type(import_381664) is not StypyTypeError):

    if (import_381664 != 'pyd_module'):
        __import__(import_381664)
        sys_modules_381665 = sys.modules[import_381664]
        import_from_module(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.sparse.csgraph._laplacian', sys_modules_381665.module_type_store, module_type_store, ['laplacian'])
        nest_module(stypy.reporting.localization.Localization(__file__, 164, 0), __file__, sys_modules_381665, sys_modules_381665.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph._laplacian import laplacian

        import_from_module(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.sparse.csgraph._laplacian', None, module_type_store, ['laplacian'], [laplacian])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph._laplacian' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.sparse.csgraph._laplacian', import_381664)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 165, 0))

# 'from scipy.sparse.csgraph._shortest_path import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, NegativeCycleError' statement (line 165)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.sparse.csgraph._shortest_path')

if (type(import_381666) is not StypyTypeError):

    if (import_381666 != 'pyd_module'):
        __import__(import_381666)
        sys_modules_381667 = sys.modules[import_381666]
        import_from_module(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.sparse.csgraph._shortest_path', sys_modules_381667.module_type_store, module_type_store, ['shortest_path', 'floyd_warshall', 'dijkstra', 'bellman_ford', 'johnson', 'NegativeCycleError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 165, 0), __file__, sys_modules_381667, sys_modules_381667.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph._shortest_path import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, NegativeCycleError

        import_from_module(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.sparse.csgraph._shortest_path', None, module_type_store, ['shortest_path', 'floyd_warshall', 'dijkstra', 'bellman_ford', 'johnson', 'NegativeCycleError'], [shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, NegativeCycleError])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph._shortest_path' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.sparse.csgraph._shortest_path', import_381666)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 167, 0))

# 'from scipy.sparse.csgraph._traversal import breadth_first_order, depth_first_order, breadth_first_tree, depth_first_tree, connected_components' statement (line 167)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 167, 0), 'scipy.sparse.csgraph._traversal')

if (type(import_381668) is not StypyTypeError):

    if (import_381668 != 'pyd_module'):
        __import__(import_381668)
        sys_modules_381669 = sys.modules[import_381668]
        import_from_module(stypy.reporting.localization.Localization(__file__, 167, 0), 'scipy.sparse.csgraph._traversal', sys_modules_381669.module_type_store, module_type_store, ['breadth_first_order', 'depth_first_order', 'breadth_first_tree', 'depth_first_tree', 'connected_components'])
        nest_module(stypy.reporting.localization.Localization(__file__, 167, 0), __file__, sys_modules_381669, sys_modules_381669.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph._traversal import breadth_first_order, depth_first_order, breadth_first_tree, depth_first_tree, connected_components

        import_from_module(stypy.reporting.localization.Localization(__file__, 167, 0), 'scipy.sparse.csgraph._traversal', None, module_type_store, ['breadth_first_order', 'depth_first_order', 'breadth_first_tree', 'depth_first_tree', 'connected_components'], [breadth_first_order, depth_first_order, breadth_first_tree, depth_first_tree, connected_components])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph._traversal' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'scipy.sparse.csgraph._traversal', import_381668)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 169, 0))

# 'from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree' statement (line 169)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 169, 0), 'scipy.sparse.csgraph._min_spanning_tree')

if (type(import_381670) is not StypyTypeError):

    if (import_381670 != 'pyd_module'):
        __import__(import_381670)
        sys_modules_381671 = sys.modules[import_381670]
        import_from_module(stypy.reporting.localization.Localization(__file__, 169, 0), 'scipy.sparse.csgraph._min_spanning_tree', sys_modules_381671.module_type_store, module_type_store, ['minimum_spanning_tree'])
        nest_module(stypy.reporting.localization.Localization(__file__, 169, 0), __file__, sys_modules_381671, sys_modules_381671.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree

        import_from_module(stypy.reporting.localization.Localization(__file__, 169, 0), 'scipy.sparse.csgraph._min_spanning_tree', None, module_type_store, ['minimum_spanning_tree'], [minimum_spanning_tree])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph._min_spanning_tree' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'scipy.sparse.csgraph._min_spanning_tree', import_381670)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 170, 0))

# 'from scipy.sparse.csgraph._reordering import reverse_cuthill_mckee, maximum_bipartite_matching, structural_rank' statement (line 170)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381672 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 170, 0), 'scipy.sparse.csgraph._reordering')

if (type(import_381672) is not StypyTypeError):

    if (import_381672 != 'pyd_module'):
        __import__(import_381672)
        sys_modules_381673 = sys.modules[import_381672]
        import_from_module(stypy.reporting.localization.Localization(__file__, 170, 0), 'scipy.sparse.csgraph._reordering', sys_modules_381673.module_type_store, module_type_store, ['reverse_cuthill_mckee', 'maximum_bipartite_matching', 'structural_rank'])
        nest_module(stypy.reporting.localization.Localization(__file__, 170, 0), __file__, sys_modules_381673, sys_modules_381673.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph._reordering import reverse_cuthill_mckee, maximum_bipartite_matching, structural_rank

        import_from_module(stypy.reporting.localization.Localization(__file__, 170, 0), 'scipy.sparse.csgraph._reordering', None, module_type_store, ['reverse_cuthill_mckee', 'maximum_bipartite_matching', 'structural_rank'], [reverse_cuthill_mckee, maximum_bipartite_matching, structural_rank])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph._reordering' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'scipy.sparse.csgraph._reordering', import_381672)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 172, 0))

# 'from scipy.sparse.csgraph._tools import construct_dist_matrix, reconstruct_path, csgraph_from_dense, csgraph_to_dense, csgraph_masked_from_dense, csgraph_from_masked, csgraph_to_masked' statement (line 172)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy.sparse.csgraph._tools')

if (type(import_381674) is not StypyTypeError):

    if (import_381674 != 'pyd_module'):
        __import__(import_381674)
        sys_modules_381675 = sys.modules[import_381674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy.sparse.csgraph._tools', sys_modules_381675.module_type_store, module_type_store, ['construct_dist_matrix', 'reconstruct_path', 'csgraph_from_dense', 'csgraph_to_dense', 'csgraph_masked_from_dense', 'csgraph_from_masked', 'csgraph_to_masked'])
        nest_module(stypy.reporting.localization.Localization(__file__, 172, 0), __file__, sys_modules_381675, sys_modules_381675.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph._tools import construct_dist_matrix, reconstruct_path, csgraph_from_dense, csgraph_to_dense, csgraph_masked_from_dense, csgraph_from_masked, csgraph_to_masked

        import_from_module(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy.sparse.csgraph._tools', None, module_type_store, ['construct_dist_matrix', 'reconstruct_path', 'csgraph_from_dense', 'csgraph_to_dense', 'csgraph_masked_from_dense', 'csgraph_from_masked', 'csgraph_to_masked'], [construct_dist_matrix, reconstruct_path, csgraph_from_dense, csgraph_to_dense, csgraph_masked_from_dense, csgraph_from_masked, csgraph_to_masked])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph._tools' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy.sparse.csgraph._tools', import_381674)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 176, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 176)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')
import_381676 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy._lib._testutils')

if (type(import_381676) is not StypyTypeError):

    if (import_381676 != 'pyd_module'):
        __import__(import_381676)
        sys_modules_381677 = sys.modules[import_381676]
        import_from_module(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy._lib._testutils', sys_modules_381677.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 176, 0), __file__, sys_modules_381677, sys_modules_381677.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'scipy._lib._testutils', import_381676)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/')


# Assigning a Call to a Name (line 177):

# Call to PytestTester(...): (line 177)
# Processing the call arguments (line 177)
# Getting the type of '__name__' (line 177)
name___381679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), '__name__', False)
# Processing the call keyword arguments (line 177)
kwargs_381680 = {}
# Getting the type of 'PytestTester' (line 177)
PytestTester_381678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 177)
PytestTester_call_result_381681 = invoke(stypy.reporting.localization.Localization(__file__, 177, 7), PytestTester_381678, *[name___381679], **kwargs_381680)

# Assigning a type to the variable 'test' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'test', PytestTester_call_result_381681)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 178, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
