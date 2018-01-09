
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test the minimum spanning tree function'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy as np
5: from numpy.testing import assert_
6: import numpy.testing as npt
7: from scipy.sparse import csr_matrix
8: from scipy.sparse.csgraph import minimum_spanning_tree
9: 
10: 
11: def test_minimum_spanning_tree():
12: 
13:     # Create a graph with two connected components.
14:     graph = [[0,1,0,0,0],
15:              [1,0,0,0,0],
16:              [0,0,0,8,5],
17:              [0,0,8,0,1],
18:              [0,0,5,1,0]]
19:     graph = np.asarray(graph)
20: 
21:     # Create the expected spanning tree.
22:     expected = [[0,1,0,0,0],
23:                 [0,0,0,0,0],
24:                 [0,0,0,0,5],
25:                 [0,0,0,0,1],
26:                 [0,0,0,0,0]]
27:     expected = np.asarray(expected)
28: 
29:     # Ensure minimum spanning tree code gives this expected output.
30:     csgraph = csr_matrix(graph)
31:     mintree = minimum_spanning_tree(csgraph)
32:     npt.assert_array_equal(mintree.todense(), expected,
33:         'Incorrect spanning tree found.')
34: 
35:     # Ensure that the original graph was not modified.
36:     npt.assert_array_equal(csgraph.todense(), graph,
37:         'Original graph was modified.')
38: 
39:     # Now let the algorithm modify the csgraph in place.
40:     mintree = minimum_spanning_tree(csgraph, overwrite=True)
41:     npt.assert_array_equal(mintree.todense(), expected,
42:         'Graph was not properly modified to contain MST.')
43: 
44:     np.random.seed(1234)
45:     for N in (5, 10, 15, 20):
46: 
47:         # Create a random graph.
48:         graph = 3 + np.random.random((N, N))
49:         csgraph = csr_matrix(graph)
50: 
51:         # The spanning tree has at most N - 1 edges.
52:         mintree = minimum_spanning_tree(csgraph)
53:         assert_(mintree.nnz < N)
54: 
55:         # Set the sub diagonal to 1 to create a known spanning tree.
56:         idx = np.arange(N-1)
57:         graph[idx,idx+1] = 1
58:         csgraph = csr_matrix(graph)
59:         mintree = minimum_spanning_tree(csgraph)
60: 
61:         # We expect to see this pattern in the spanning tree and otherwise
62:         # have this zero.
63:         expected = np.zeros((N, N))
64:         expected[idx, idx+1] = 1
65: 
66:         npt.assert_array_equal(mintree.todense(), expected,
67:             'Incorrect spanning tree found.')
68: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_384420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Test the minimum spanning tree function')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384421 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_384421) is not StypyTypeError):

    if (import_384421 != 'pyd_module'):
        __import__(import_384421)
        sys_modules_384422 = sys.modules[import_384421]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_384422.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_384421)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384423 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_384423) is not StypyTypeError):

    if (import_384423 != 'pyd_module'):
        __import__(import_384423)
        sys_modules_384424 = sys.modules[import_384423]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_384424.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_384424, sys_modules_384424.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_384423)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy.testing' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384425 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_384425) is not StypyTypeError):

    if (import_384425 != 'pyd_module'):
        __import__(import_384425)
        sys_modules_384426 = sys.modules[import_384425]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'npt', sys_modules_384426.module_type_store, module_type_store)
    else:
        import numpy.testing as npt

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'npt', numpy.testing, module_type_store)

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_384425)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.sparse import csr_matrix' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384427 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse')

if (type(import_384427) is not StypyTypeError):

    if (import_384427 != 'pyd_module'):
        __import__(import_384427)
        sys_modules_384428 = sys.modules[import_384427]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', sys_modules_384428.module_type_store, module_type_store, ['csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_384428, sys_modules_384428.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix'], [csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.sparse', import_384427)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.sparse.csgraph import minimum_spanning_tree' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384429 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csgraph')

if (type(import_384429) is not StypyTypeError):

    if (import_384429 != 'pyd_module'):
        __import__(import_384429)
        sys_modules_384430 = sys.modules[import_384429]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csgraph', sys_modules_384430.module_type_store, module_type_store, ['minimum_spanning_tree'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_384430, sys_modules_384430.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph import minimum_spanning_tree

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csgraph', None, module_type_store, ['minimum_spanning_tree'], [minimum_spanning_tree])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse.csgraph', import_384429)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')


@norecursion
def test_minimum_spanning_tree(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_minimum_spanning_tree'
    module_type_store = module_type_store.open_function_context('test_minimum_spanning_tree', 11, 0, False)
    
    # Passed parameters checking function
    test_minimum_spanning_tree.stypy_localization = localization
    test_minimum_spanning_tree.stypy_type_of_self = None
    test_minimum_spanning_tree.stypy_type_store = module_type_store
    test_minimum_spanning_tree.stypy_function_name = 'test_minimum_spanning_tree'
    test_minimum_spanning_tree.stypy_param_names_list = []
    test_minimum_spanning_tree.stypy_varargs_param_name = None
    test_minimum_spanning_tree.stypy_kwargs_param_name = None
    test_minimum_spanning_tree.stypy_call_defaults = defaults
    test_minimum_spanning_tree.stypy_call_varargs = varargs
    test_minimum_spanning_tree.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_minimum_spanning_tree', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_minimum_spanning_tree', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_minimum_spanning_tree(...)' code ##################

    
    # Assigning a List to a Name (line 14):
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_384431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_384432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    int_384433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_384432, int_384433)
    # Adding element type (line 14)
    int_384434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_384432, int_384434)
    # Adding element type (line 14)
    int_384435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_384432, int_384435)
    # Adding element type (line 14)
    int_384436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_384432, int_384436)
    # Adding element type (line 14)
    int_384437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 13), list_384432, int_384437)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), list_384431, list_384432)
    # Adding element type (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_384438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_384439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_384438, int_384439)
    # Adding element type (line 15)
    int_384440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_384438, int_384440)
    # Adding element type (line 15)
    int_384441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_384438, int_384441)
    # Adding element type (line 15)
    int_384442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_384438, int_384442)
    # Adding element type (line 15)
    int_384443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), list_384438, int_384443)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), list_384431, list_384438)
    # Adding element type (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_384444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_384445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_384444, int_384445)
    # Adding element type (line 16)
    int_384446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_384444, int_384446)
    # Adding element type (line 16)
    int_384447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_384444, int_384447)
    # Adding element type (line 16)
    int_384448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_384444, int_384448)
    # Adding element type (line 16)
    int_384449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 13), list_384444, int_384449)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), list_384431, list_384444)
    # Adding element type (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_384450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_384451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_384450, int_384451)
    # Adding element type (line 17)
    int_384452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_384450, int_384452)
    # Adding element type (line 17)
    int_384453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_384450, int_384453)
    # Adding element type (line 17)
    int_384454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_384450, int_384454)
    # Adding element type (line 17)
    int_384455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_384450, int_384455)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), list_384431, list_384450)
    # Adding element type (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_384456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_384457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_384456, int_384457)
    # Adding element type (line 18)
    int_384458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_384456, int_384458)
    # Adding element type (line 18)
    int_384459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_384456, int_384459)
    # Adding element type (line 18)
    int_384460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_384456, int_384460)
    # Adding element type (line 18)
    int_384461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), list_384456, int_384461)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), list_384431, list_384456)
    
    # Assigning a type to the variable 'graph' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'graph', list_384431)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to asarray(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'graph' (line 19)
    graph_384464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'graph', False)
    # Processing the call keyword arguments (line 19)
    kwargs_384465 = {}
    # Getting the type of 'np' (line 19)
    np_384462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 19)
    asarray_384463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), np_384462, 'asarray')
    # Calling asarray(args, kwargs) (line 19)
    asarray_call_result_384466 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), asarray_384463, *[graph_384464], **kwargs_384465)
    
    # Assigning a type to the variable 'graph' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'graph', asarray_call_result_384466)
    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_384467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_384468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_384469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_384468, int_384469)
    # Adding element type (line 22)
    int_384470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_384468, int_384470)
    # Adding element type (line 22)
    int_384471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_384468, int_384471)
    # Adding element type (line 22)
    int_384472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_384468, int_384472)
    # Adding element type (line 22)
    int_384473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_384468, int_384473)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_384467, list_384468)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_384474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    int_384475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_384474, int_384475)
    # Adding element type (line 23)
    int_384476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_384474, int_384476)
    # Adding element type (line 23)
    int_384477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_384474, int_384477)
    # Adding element type (line 23)
    int_384478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_384474, int_384478)
    # Adding element type (line 23)
    int_384479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_384474, int_384479)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_384467, list_384474)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_384480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    int_384481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), list_384480, int_384481)
    # Adding element type (line 24)
    int_384482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), list_384480, int_384482)
    # Adding element type (line 24)
    int_384483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), list_384480, int_384483)
    # Adding element type (line 24)
    int_384484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), list_384480, int_384484)
    # Adding element type (line 24)
    int_384485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), list_384480, int_384485)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_384467, list_384480)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_384486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_384487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_384486, int_384487)
    # Adding element type (line 25)
    int_384488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_384486, int_384488)
    # Adding element type (line 25)
    int_384489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_384486, int_384489)
    # Adding element type (line 25)
    int_384490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_384486, int_384490)
    # Adding element type (line 25)
    int_384491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_384486, int_384491)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_384467, list_384486)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_384492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    int_384493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_384492, int_384493)
    # Adding element type (line 26)
    int_384494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_384492, int_384494)
    # Adding element type (line 26)
    int_384495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_384492, int_384495)
    # Adding element type (line 26)
    int_384496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_384492, int_384496)
    # Adding element type (line 26)
    int_384497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), list_384492, int_384497)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 15), list_384467, list_384492)
    
    # Assigning a type to the variable 'expected' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'expected', list_384467)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to asarray(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'expected' (line 27)
    expected_384500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'expected', False)
    # Processing the call keyword arguments (line 27)
    kwargs_384501 = {}
    # Getting the type of 'np' (line 27)
    np_384498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 27)
    asarray_384499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), np_384498, 'asarray')
    # Calling asarray(args, kwargs) (line 27)
    asarray_call_result_384502 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), asarray_384499, *[expected_384500], **kwargs_384501)
    
    # Assigning a type to the variable 'expected' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'expected', asarray_call_result_384502)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to csr_matrix(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'graph' (line 30)
    graph_384504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'graph', False)
    # Processing the call keyword arguments (line 30)
    kwargs_384505 = {}
    # Getting the type of 'csr_matrix' (line 30)
    csr_matrix_384503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 30)
    csr_matrix_call_result_384506 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), csr_matrix_384503, *[graph_384504], **kwargs_384505)
    
    # Assigning a type to the variable 'csgraph' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'csgraph', csr_matrix_call_result_384506)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to minimum_spanning_tree(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'csgraph' (line 31)
    csgraph_384508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 36), 'csgraph', False)
    # Processing the call keyword arguments (line 31)
    kwargs_384509 = {}
    # Getting the type of 'minimum_spanning_tree' (line 31)
    minimum_spanning_tree_384507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'minimum_spanning_tree', False)
    # Calling minimum_spanning_tree(args, kwargs) (line 31)
    minimum_spanning_tree_call_result_384510 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), minimum_spanning_tree_384507, *[csgraph_384508], **kwargs_384509)
    
    # Assigning a type to the variable 'mintree' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'mintree', minimum_spanning_tree_call_result_384510)
    
    # Call to assert_array_equal(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to todense(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_384515 = {}
    # Getting the type of 'mintree' (line 32)
    mintree_384513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'mintree', False)
    # Obtaining the member 'todense' of a type (line 32)
    todense_384514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 27), mintree_384513, 'todense')
    # Calling todense(args, kwargs) (line 32)
    todense_call_result_384516 = invoke(stypy.reporting.localization.Localization(__file__, 32, 27), todense_384514, *[], **kwargs_384515)
    
    # Getting the type of 'expected' (line 32)
    expected_384517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 46), 'expected', False)
    str_384518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'str', 'Incorrect spanning tree found.')
    # Processing the call keyword arguments (line 32)
    kwargs_384519 = {}
    # Getting the type of 'npt' (line 32)
    npt_384511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'npt', False)
    # Obtaining the member 'assert_array_equal' of a type (line 32)
    assert_array_equal_384512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), npt_384511, 'assert_array_equal')
    # Calling assert_array_equal(args, kwargs) (line 32)
    assert_array_equal_call_result_384520 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert_array_equal_384512, *[todense_call_result_384516, expected_384517, str_384518], **kwargs_384519)
    
    
    # Call to assert_array_equal(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to todense(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_384525 = {}
    # Getting the type of 'csgraph' (line 36)
    csgraph_384523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'csgraph', False)
    # Obtaining the member 'todense' of a type (line 36)
    todense_384524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), csgraph_384523, 'todense')
    # Calling todense(args, kwargs) (line 36)
    todense_call_result_384526 = invoke(stypy.reporting.localization.Localization(__file__, 36, 27), todense_384524, *[], **kwargs_384525)
    
    # Getting the type of 'graph' (line 36)
    graph_384527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 46), 'graph', False)
    str_384528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'str', 'Original graph was modified.')
    # Processing the call keyword arguments (line 36)
    kwargs_384529 = {}
    # Getting the type of 'npt' (line 36)
    npt_384521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'npt', False)
    # Obtaining the member 'assert_array_equal' of a type (line 36)
    assert_array_equal_384522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), npt_384521, 'assert_array_equal')
    # Calling assert_array_equal(args, kwargs) (line 36)
    assert_array_equal_call_result_384530 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), assert_array_equal_384522, *[todense_call_result_384526, graph_384527, str_384528], **kwargs_384529)
    
    
    # Assigning a Call to a Name (line 40):
    
    # Call to minimum_spanning_tree(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'csgraph' (line 40)
    csgraph_384532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'csgraph', False)
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'True' (line 40)
    True_384533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 55), 'True', False)
    keyword_384534 = True_384533
    kwargs_384535 = {'overwrite': keyword_384534}
    # Getting the type of 'minimum_spanning_tree' (line 40)
    minimum_spanning_tree_384531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'minimum_spanning_tree', False)
    # Calling minimum_spanning_tree(args, kwargs) (line 40)
    minimum_spanning_tree_call_result_384536 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), minimum_spanning_tree_384531, *[csgraph_384532], **kwargs_384535)
    
    # Assigning a type to the variable 'mintree' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'mintree', minimum_spanning_tree_call_result_384536)
    
    # Call to assert_array_equal(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to todense(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_384541 = {}
    # Getting the type of 'mintree' (line 41)
    mintree_384539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'mintree', False)
    # Obtaining the member 'todense' of a type (line 41)
    todense_384540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 27), mintree_384539, 'todense')
    # Calling todense(args, kwargs) (line 41)
    todense_call_result_384542 = invoke(stypy.reporting.localization.Localization(__file__, 41, 27), todense_384540, *[], **kwargs_384541)
    
    # Getting the type of 'expected' (line 41)
    expected_384543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 46), 'expected', False)
    str_384544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'str', 'Graph was not properly modified to contain MST.')
    # Processing the call keyword arguments (line 41)
    kwargs_384545 = {}
    # Getting the type of 'npt' (line 41)
    npt_384537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'npt', False)
    # Obtaining the member 'assert_array_equal' of a type (line 41)
    assert_array_equal_384538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), npt_384537, 'assert_array_equal')
    # Calling assert_array_equal(args, kwargs) (line 41)
    assert_array_equal_call_result_384546 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), assert_array_equal_384538, *[todense_call_result_384542, expected_384543, str_384544], **kwargs_384545)
    
    
    # Call to seed(...): (line 44)
    # Processing the call arguments (line 44)
    int_384550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
    # Processing the call keyword arguments (line 44)
    kwargs_384551 = {}
    # Getting the type of 'np' (line 44)
    np_384547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 44)
    random_384548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), np_384547, 'random')
    # Obtaining the member 'seed' of a type (line 44)
    seed_384549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), random_384548, 'seed')
    # Calling seed(args, kwargs) (line 44)
    seed_call_result_384552 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), seed_384549, *[int_384550], **kwargs_384551)
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_384553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    int_384554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 14), tuple_384553, int_384554)
    # Adding element type (line 45)
    int_384555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 14), tuple_384553, int_384555)
    # Adding element type (line 45)
    int_384556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 14), tuple_384553, int_384556)
    # Adding element type (line 45)
    int_384557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 14), tuple_384553, int_384557)
    
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 4), tuple_384553)
    # Getting the type of the for loop variable (line 45)
    for_loop_var_384558 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 4), tuple_384553)
    # Assigning a type to the variable 'N' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'N', for_loop_var_384558)
    # SSA begins for a for statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 48):
    int_384559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'int')
    
    # Call to random(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Obtaining an instance of the builtin type 'tuple' (line 48)
    tuple_384563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 48)
    # Adding element type (line 48)
    # Getting the type of 'N' (line 48)
    N_384564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 38), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 38), tuple_384563, N_384564)
    # Adding element type (line 48)
    # Getting the type of 'N' (line 48)
    N_384565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 41), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 38), tuple_384563, N_384565)
    
    # Processing the call keyword arguments (line 48)
    kwargs_384566 = {}
    # Getting the type of 'np' (line 48)
    np_384560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'np', False)
    # Obtaining the member 'random' of a type (line 48)
    random_384561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 20), np_384560, 'random')
    # Obtaining the member 'random' of a type (line 48)
    random_384562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 20), random_384561, 'random')
    # Calling random(args, kwargs) (line 48)
    random_call_result_384567 = invoke(stypy.reporting.localization.Localization(__file__, 48, 20), random_384562, *[tuple_384563], **kwargs_384566)
    
    # Applying the binary operator '+' (line 48)
    result_add_384568 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 16), '+', int_384559, random_call_result_384567)
    
    # Assigning a type to the variable 'graph' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'graph', result_add_384568)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to csr_matrix(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'graph' (line 49)
    graph_384570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'graph', False)
    # Processing the call keyword arguments (line 49)
    kwargs_384571 = {}
    # Getting the type of 'csr_matrix' (line 49)
    csr_matrix_384569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 49)
    csr_matrix_call_result_384572 = invoke(stypy.reporting.localization.Localization(__file__, 49, 18), csr_matrix_384569, *[graph_384570], **kwargs_384571)
    
    # Assigning a type to the variable 'csgraph' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'csgraph', csr_matrix_call_result_384572)
    
    # Assigning a Call to a Name (line 52):
    
    # Call to minimum_spanning_tree(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'csgraph' (line 52)
    csgraph_384574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 40), 'csgraph', False)
    # Processing the call keyword arguments (line 52)
    kwargs_384575 = {}
    # Getting the type of 'minimum_spanning_tree' (line 52)
    minimum_spanning_tree_384573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'minimum_spanning_tree', False)
    # Calling minimum_spanning_tree(args, kwargs) (line 52)
    minimum_spanning_tree_call_result_384576 = invoke(stypy.reporting.localization.Localization(__file__, 52, 18), minimum_spanning_tree_384573, *[csgraph_384574], **kwargs_384575)
    
    # Assigning a type to the variable 'mintree' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'mintree', minimum_spanning_tree_call_result_384576)
    
    # Call to assert_(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Getting the type of 'mintree' (line 53)
    mintree_384578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'mintree', False)
    # Obtaining the member 'nnz' of a type (line 53)
    nnz_384579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), mintree_384578, 'nnz')
    # Getting the type of 'N' (line 53)
    N_384580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'N', False)
    # Applying the binary operator '<' (line 53)
    result_lt_384581 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 16), '<', nnz_384579, N_384580)
    
    # Processing the call keyword arguments (line 53)
    kwargs_384582 = {}
    # Getting the type of 'assert_' (line 53)
    assert__384577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 53)
    assert__call_result_384583 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert__384577, *[result_lt_384581], **kwargs_384582)
    
    
    # Assigning a Call to a Name (line 56):
    
    # Call to arange(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'N' (line 56)
    N_384586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'N', False)
    int_384587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'int')
    # Applying the binary operator '-' (line 56)
    result_sub_384588 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 24), '-', N_384586, int_384587)
    
    # Processing the call keyword arguments (line 56)
    kwargs_384589 = {}
    # Getting the type of 'np' (line 56)
    np_384584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'np', False)
    # Obtaining the member 'arange' of a type (line 56)
    arange_384585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 14), np_384584, 'arange')
    # Calling arange(args, kwargs) (line 56)
    arange_call_result_384590 = invoke(stypy.reporting.localization.Localization(__file__, 56, 14), arange_384585, *[result_sub_384588], **kwargs_384589)
    
    # Assigning a type to the variable 'idx' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'idx', arange_call_result_384590)
    
    # Assigning a Num to a Subscript (line 57):
    int_384591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'int')
    # Getting the type of 'graph' (line 57)
    graph_384592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'graph')
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_384593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    # Getting the type of 'idx' (line 57)
    idx_384594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 14), tuple_384593, idx_384594)
    # Adding element type (line 57)
    # Getting the type of 'idx' (line 57)
    idx_384595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'idx')
    int_384596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
    # Applying the binary operator '+' (line 57)
    result_add_384597 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 18), '+', idx_384595, int_384596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 14), tuple_384593, result_add_384597)
    
    # Storing an element on a container (line 57)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 8), graph_384592, (tuple_384593, int_384591))
    
    # Assigning a Call to a Name (line 58):
    
    # Call to csr_matrix(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'graph' (line 58)
    graph_384599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 29), 'graph', False)
    # Processing the call keyword arguments (line 58)
    kwargs_384600 = {}
    # Getting the type of 'csr_matrix' (line 58)
    csr_matrix_384598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 58)
    csr_matrix_call_result_384601 = invoke(stypy.reporting.localization.Localization(__file__, 58, 18), csr_matrix_384598, *[graph_384599], **kwargs_384600)
    
    # Assigning a type to the variable 'csgraph' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'csgraph', csr_matrix_call_result_384601)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to minimum_spanning_tree(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'csgraph' (line 59)
    csgraph_384603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'csgraph', False)
    # Processing the call keyword arguments (line 59)
    kwargs_384604 = {}
    # Getting the type of 'minimum_spanning_tree' (line 59)
    minimum_spanning_tree_384602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'minimum_spanning_tree', False)
    # Calling minimum_spanning_tree(args, kwargs) (line 59)
    minimum_spanning_tree_call_result_384605 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), minimum_spanning_tree_384602, *[csgraph_384603], **kwargs_384604)
    
    # Assigning a type to the variable 'mintree' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'mintree', minimum_spanning_tree_call_result_384605)
    
    # Assigning a Call to a Name (line 63):
    
    # Call to zeros(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_384608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'N' (line 63)
    N_384609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), tuple_384608, N_384609)
    # Adding element type (line 63)
    # Getting the type of 'N' (line 63)
    N_384610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 29), tuple_384608, N_384610)
    
    # Processing the call keyword arguments (line 63)
    kwargs_384611 = {}
    # Getting the type of 'np' (line 63)
    np_384606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 63)
    zeros_384607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), np_384606, 'zeros')
    # Calling zeros(args, kwargs) (line 63)
    zeros_call_result_384612 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), zeros_384607, *[tuple_384608], **kwargs_384611)
    
    # Assigning a type to the variable 'expected' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'expected', zeros_call_result_384612)
    
    # Assigning a Num to a Subscript (line 64):
    int_384613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'int')
    # Getting the type of 'expected' (line 64)
    expected_384614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'expected')
    
    # Obtaining an instance of the builtin type 'tuple' (line 64)
    tuple_384615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 64)
    # Adding element type (line 64)
    # Getting the type of 'idx' (line 64)
    idx_384616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 17), tuple_384615, idx_384616)
    # Adding element type (line 64)
    # Getting the type of 'idx' (line 64)
    idx_384617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'idx')
    int_384618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 26), 'int')
    # Applying the binary operator '+' (line 64)
    result_add_384619 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 22), '+', idx_384617, int_384618)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 17), tuple_384615, result_add_384619)
    
    # Storing an element on a container (line 64)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 8), expected_384614, (tuple_384615, int_384613))
    
    # Call to assert_array_equal(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Call to todense(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_384624 = {}
    # Getting the type of 'mintree' (line 66)
    mintree_384622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 31), 'mintree', False)
    # Obtaining the member 'todense' of a type (line 66)
    todense_384623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 31), mintree_384622, 'todense')
    # Calling todense(args, kwargs) (line 66)
    todense_call_result_384625 = invoke(stypy.reporting.localization.Localization(__file__, 66, 31), todense_384623, *[], **kwargs_384624)
    
    # Getting the type of 'expected' (line 66)
    expected_384626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 50), 'expected', False)
    str_384627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 12), 'str', 'Incorrect spanning tree found.')
    # Processing the call keyword arguments (line 66)
    kwargs_384628 = {}
    # Getting the type of 'npt' (line 66)
    npt_384620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'npt', False)
    # Obtaining the member 'assert_array_equal' of a type (line 66)
    assert_array_equal_384621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), npt_384620, 'assert_array_equal')
    # Calling assert_array_equal(args, kwargs) (line 66)
    assert_array_equal_call_result_384629 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_array_equal_384621, *[todense_call_result_384625, expected_384626, str_384627], **kwargs_384628)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_minimum_spanning_tree(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_minimum_spanning_tree' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_384630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384630)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_minimum_spanning_tree'
    return stypy_return_type_384630

# Assigning a type to the variable 'test_minimum_spanning_tree' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'test_minimum_spanning_tree', test_minimum_spanning_tree)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
