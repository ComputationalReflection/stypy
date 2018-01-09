
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_array_almost_equal
5: from scipy.sparse.csgraph import (breadth_first_tree, depth_first_tree,
6:     csgraph_to_dense, csgraph_from_dense)
7: 
8: 
9: def test_graph_breadth_first():
10:     csgraph = np.array([[0, 1, 2, 0, 0],
11:                         [1, 0, 0, 0, 3],
12:                         [2, 0, 0, 7, 0],
13:                         [0, 0, 7, 0, 1],
14:                         [0, 3, 0, 1, 0]])
15:     csgraph = csgraph_from_dense(csgraph, null_value=0)
16: 
17:     bfirst = np.array([[0, 1, 2, 0, 0],
18:                        [0, 0, 0, 0, 3],
19:                        [0, 0, 0, 7, 0],
20:                        [0, 0, 0, 0, 0],
21:                        [0, 0, 0, 0, 0]])
22: 
23:     for directed in [True, False]:
24:         bfirst_test = breadth_first_tree(csgraph, 0, directed)
25:         assert_array_almost_equal(csgraph_to_dense(bfirst_test),
26:                                   bfirst)
27: 
28: 
29: def test_graph_depth_first():
30:     csgraph = np.array([[0, 1, 2, 0, 0],
31:                         [1, 0, 0, 0, 3],
32:                         [2, 0, 0, 7, 0],
33:                         [0, 0, 7, 0, 1],
34:                         [0, 3, 0, 1, 0]])
35:     csgraph = csgraph_from_dense(csgraph, null_value=0)
36: 
37:     dfirst = np.array([[0, 1, 0, 0, 0],
38:                        [0, 0, 0, 0, 3],
39:                        [0, 0, 0, 0, 0],
40:                        [0, 0, 7, 0, 0],
41:                        [0, 0, 0, 1, 0]])
42: 
43:     for directed in [True, False]:
44:         dfirst_test = depth_first_tree(csgraph, 0, directed)
45:         assert_array_almost_equal(csgraph_to_dense(dfirst_test),
46:                                   dfirst)
47: 
48: 
49: def test_graph_breadth_first_trivial_graph():
50:     csgraph = np.array([[0]])
51:     csgraph = csgraph_from_dense(csgraph, null_value=0)
52: 
53:     bfirst = np.array([[0]])
54: 
55:     for directed in [True, False]:
56:         bfirst_test = breadth_first_tree(csgraph, 0, directed)
57:         assert_array_almost_equal(csgraph_to_dense(bfirst_test),
58:                                   bfirst)
59: 
60: 
61: def test_graph_depth_first_trivial_graph():
62:     csgraph = np.array([[0]])
63:     csgraph = csgraph_from_dense(csgraph, null_value=0)
64: 
65:     bfirst = np.array([[0]])
66: 
67:     for directed in [True, False]:
68:         bfirst_test = depth_first_tree(csgraph, 0, directed)
69:         assert_array_almost_equal(csgraph_to_dense(bfirst_test),
70:                                   bfirst)
71: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_384631) is not StypyTypeError):

    if (import_384631 != 'pyd_module'):
        __import__(import_384631)
        sys_modules_384632 = sys.modules[import_384631]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_384632.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_384631)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384633 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_384633) is not StypyTypeError):

    if (import_384633 != 'pyd_module'):
        __import__(import_384633)
        sys_modules_384634 = sys.modules[import_384633]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_384634.module_type_store, module_type_store, ['assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_384634, sys_modules_384634.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal'], [assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_384633)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse.csgraph import breadth_first_tree, depth_first_tree, csgraph_to_dense, csgraph_from_dense' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_384635 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph')

if (type(import_384635) is not StypyTypeError):

    if (import_384635 != 'pyd_module'):
        __import__(import_384635)
        sys_modules_384636 = sys.modules[import_384635]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph', sys_modules_384636.module_type_store, module_type_store, ['breadth_first_tree', 'depth_first_tree', 'csgraph_to_dense', 'csgraph_from_dense'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_384636, sys_modules_384636.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph import breadth_first_tree, depth_first_tree, csgraph_to_dense, csgraph_from_dense

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph', None, module_type_store, ['breadth_first_tree', 'depth_first_tree', 'csgraph_to_dense', 'csgraph_from_dense'], [breadth_first_tree, depth_first_tree, csgraph_to_dense, csgraph_from_dense])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse.csgraph', import_384635)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')


@norecursion
def test_graph_breadth_first(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_breadth_first'
    module_type_store = module_type_store.open_function_context('test_graph_breadth_first', 9, 0, False)
    
    # Passed parameters checking function
    test_graph_breadth_first.stypy_localization = localization
    test_graph_breadth_first.stypy_type_of_self = None
    test_graph_breadth_first.stypy_type_store = module_type_store
    test_graph_breadth_first.stypy_function_name = 'test_graph_breadth_first'
    test_graph_breadth_first.stypy_param_names_list = []
    test_graph_breadth_first.stypy_varargs_param_name = None
    test_graph_breadth_first.stypy_kwargs_param_name = None
    test_graph_breadth_first.stypy_call_defaults = defaults
    test_graph_breadth_first.stypy_call_varargs = varargs
    test_graph_breadth_first.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_breadth_first', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_breadth_first', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_breadth_first(...)' code ##################

    
    # Assigning a Call to a Name (line 10):
    
    # Call to array(...): (line 10)
    # Processing the call arguments (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_384639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_384640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    int_384641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 24), list_384640, int_384641)
    # Adding element type (line 10)
    int_384642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 24), list_384640, int_384642)
    # Adding element type (line 10)
    int_384643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 24), list_384640, int_384643)
    # Adding element type (line 10)
    int_384644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 24), list_384640, int_384644)
    # Adding element type (line 10)
    int_384645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 24), list_384640, int_384645)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_384639, list_384640)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_384646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    int_384647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), list_384646, int_384647)
    # Adding element type (line 11)
    int_384648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), list_384646, int_384648)
    # Adding element type (line 11)
    int_384649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), list_384646, int_384649)
    # Adding element type (line 11)
    int_384650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), list_384646, int_384650)
    # Adding element type (line 11)
    int_384651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 24), list_384646, int_384651)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_384639, list_384646)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_384652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    int_384653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 24), list_384652, int_384653)
    # Adding element type (line 12)
    int_384654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 24), list_384652, int_384654)
    # Adding element type (line 12)
    int_384655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 24), list_384652, int_384655)
    # Adding element type (line 12)
    int_384656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 24), list_384652, int_384656)
    # Adding element type (line 12)
    int_384657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 24), list_384652, int_384657)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_384639, list_384652)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_384658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    int_384659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 24), list_384658, int_384659)
    # Adding element type (line 13)
    int_384660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 24), list_384658, int_384660)
    # Adding element type (line 13)
    int_384661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 24), list_384658, int_384661)
    # Adding element type (line 13)
    int_384662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 24), list_384658, int_384662)
    # Adding element type (line 13)
    int_384663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 24), list_384658, int_384663)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_384639, list_384658)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_384664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    int_384665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_384664, int_384665)
    # Adding element type (line 14)
    int_384666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_384664, int_384666)
    # Adding element type (line 14)
    int_384667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_384664, int_384667)
    # Adding element type (line 14)
    int_384668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_384664, int_384668)
    # Adding element type (line 14)
    int_384669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_384664, int_384669)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 23), list_384639, list_384664)
    
    # Processing the call keyword arguments (line 10)
    kwargs_384670 = {}
    # Getting the type of 'np' (line 10)
    np_384637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 10)
    array_384638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 14), np_384637, 'array')
    # Calling array(args, kwargs) (line 10)
    array_call_result_384671 = invoke(stypy.reporting.localization.Localization(__file__, 10, 14), array_384638, *[list_384639], **kwargs_384670)
    
    # Assigning a type to the variable 'csgraph' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'csgraph', array_call_result_384671)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to csgraph_from_dense(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'csgraph' (line 15)
    csgraph_384673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 33), 'csgraph', False)
    # Processing the call keyword arguments (line 15)
    int_384674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 53), 'int')
    keyword_384675 = int_384674
    kwargs_384676 = {'null_value': keyword_384675}
    # Getting the type of 'csgraph_from_dense' (line 15)
    csgraph_from_dense_384672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 15)
    csgraph_from_dense_call_result_384677 = invoke(stypy.reporting.localization.Localization(__file__, 15, 14), csgraph_from_dense_384672, *[csgraph_384673], **kwargs_384676)
    
    # Assigning a type to the variable 'csgraph' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'csgraph', csgraph_from_dense_call_result_384677)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to array(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_384680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_384681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_384682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), list_384681, int_384682)
    # Adding element type (line 17)
    int_384683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), list_384681, int_384683)
    # Adding element type (line 17)
    int_384684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), list_384681, int_384684)
    # Adding element type (line 17)
    int_384685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), list_384681, int_384685)
    # Adding element type (line 17)
    int_384686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), list_384681, int_384686)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_384680, list_384681)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_384687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_384688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_384687, int_384688)
    # Adding element type (line 18)
    int_384689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_384687, int_384689)
    # Adding element type (line 18)
    int_384690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_384687, int_384690)
    # Adding element type (line 18)
    int_384691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_384687, int_384691)
    # Adding element type (line 18)
    int_384692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 23), list_384687, int_384692)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_384680, list_384687)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_384693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    int_384694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_384693, int_384694)
    # Adding element type (line 19)
    int_384695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_384693, int_384695)
    # Adding element type (line 19)
    int_384696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_384693, int_384696)
    # Adding element type (line 19)
    int_384697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_384693, int_384697)
    # Adding element type (line 19)
    int_384698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_384693, int_384698)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_384680, list_384693)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_384699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    int_384700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_384699, int_384700)
    # Adding element type (line 20)
    int_384701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_384699, int_384701)
    # Adding element type (line 20)
    int_384702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_384699, int_384702)
    # Adding element type (line 20)
    int_384703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_384699, int_384703)
    # Adding element type (line 20)
    int_384704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_384699, int_384704)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_384680, list_384699)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_384705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_384706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_384705, int_384706)
    # Adding element type (line 21)
    int_384707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_384705, int_384707)
    # Adding element type (line 21)
    int_384708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_384705, int_384708)
    # Adding element type (line 21)
    int_384709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_384705, int_384709)
    # Adding element type (line 21)
    int_384710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_384705, int_384710)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_384680, list_384705)
    
    # Processing the call keyword arguments (line 17)
    kwargs_384711 = {}
    # Getting the type of 'np' (line 17)
    np_384678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 17)
    array_384679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 13), np_384678, 'array')
    # Calling array(args, kwargs) (line 17)
    array_call_result_384712 = invoke(stypy.reporting.localization.Localization(__file__, 17, 13), array_384679, *[list_384680], **kwargs_384711)
    
    # Assigning a type to the variable 'bfirst' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'bfirst', array_call_result_384712)
    
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_384713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 'True' (line 23)
    True_384714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 20), list_384713, True_384714)
    # Adding element type (line 23)
    # Getting the type of 'False' (line 23)
    False_384715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 20), list_384713, False_384715)
    
    # Testing the type of a for loop iterable (line 23)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 4), list_384713)
    # Getting the type of the for loop variable (line 23)
    for_loop_var_384716 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 4), list_384713)
    # Assigning a type to the variable 'directed' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'directed', for_loop_var_384716)
    # SSA begins for a for statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 24):
    
    # Call to breadth_first_tree(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'csgraph' (line 24)
    csgraph_384718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'csgraph', False)
    int_384719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 50), 'int')
    # Getting the type of 'directed' (line 24)
    directed_384720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 53), 'directed', False)
    # Processing the call keyword arguments (line 24)
    kwargs_384721 = {}
    # Getting the type of 'breadth_first_tree' (line 24)
    breadth_first_tree_384717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'breadth_first_tree', False)
    # Calling breadth_first_tree(args, kwargs) (line 24)
    breadth_first_tree_call_result_384722 = invoke(stypy.reporting.localization.Localization(__file__, 24, 22), breadth_first_tree_384717, *[csgraph_384718, int_384719, directed_384720], **kwargs_384721)
    
    # Assigning a type to the variable 'bfirst_test' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'bfirst_test', breadth_first_tree_call_result_384722)
    
    # Call to assert_array_almost_equal(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to csgraph_to_dense(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'bfirst_test' (line 25)
    bfirst_test_384725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 51), 'bfirst_test', False)
    # Processing the call keyword arguments (line 25)
    kwargs_384726 = {}
    # Getting the type of 'csgraph_to_dense' (line 25)
    csgraph_to_dense_384724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'csgraph_to_dense', False)
    # Calling csgraph_to_dense(args, kwargs) (line 25)
    csgraph_to_dense_call_result_384727 = invoke(stypy.reporting.localization.Localization(__file__, 25, 34), csgraph_to_dense_384724, *[bfirst_test_384725], **kwargs_384726)
    
    # Getting the type of 'bfirst' (line 26)
    bfirst_384728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 34), 'bfirst', False)
    # Processing the call keyword arguments (line 25)
    kwargs_384729 = {}
    # Getting the type of 'assert_array_almost_equal' (line 25)
    assert_array_almost_equal_384723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 25)
    assert_array_almost_equal_call_result_384730 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assert_array_almost_equal_384723, *[csgraph_to_dense_call_result_384727, bfirst_384728], **kwargs_384729)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_graph_breadth_first(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_breadth_first' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_384731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384731)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_breadth_first'
    return stypy_return_type_384731

# Assigning a type to the variable 'test_graph_breadth_first' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_graph_breadth_first', test_graph_breadth_first)

@norecursion
def test_graph_depth_first(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_depth_first'
    module_type_store = module_type_store.open_function_context('test_graph_depth_first', 29, 0, False)
    
    # Passed parameters checking function
    test_graph_depth_first.stypy_localization = localization
    test_graph_depth_first.stypy_type_of_self = None
    test_graph_depth_first.stypy_type_store = module_type_store
    test_graph_depth_first.stypy_function_name = 'test_graph_depth_first'
    test_graph_depth_first.stypy_param_names_list = []
    test_graph_depth_first.stypy_varargs_param_name = None
    test_graph_depth_first.stypy_kwargs_param_name = None
    test_graph_depth_first.stypy_call_defaults = defaults
    test_graph_depth_first.stypy_call_varargs = varargs
    test_graph_depth_first.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_depth_first', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_depth_first', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_depth_first(...)' code ##################

    
    # Assigning a Call to a Name (line 30):
    
    # Call to array(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_384734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_384735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_384736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), list_384735, int_384736)
    # Adding element type (line 30)
    int_384737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), list_384735, int_384737)
    # Adding element type (line 30)
    int_384738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), list_384735, int_384738)
    # Adding element type (line 30)
    int_384739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), list_384735, int_384739)
    # Adding element type (line 30)
    int_384740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), list_384735, int_384740)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), list_384734, list_384735)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_384741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    int_384742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_384741, int_384742)
    # Adding element type (line 31)
    int_384743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_384741, int_384743)
    # Adding element type (line 31)
    int_384744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_384741, int_384744)
    # Adding element type (line 31)
    int_384745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_384741, int_384745)
    # Adding element type (line 31)
    int_384746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 24), list_384741, int_384746)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), list_384734, list_384741)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_384747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    # Adding element type (line 32)
    int_384748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 24), list_384747, int_384748)
    # Adding element type (line 32)
    int_384749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 24), list_384747, int_384749)
    # Adding element type (line 32)
    int_384750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 24), list_384747, int_384750)
    # Adding element type (line 32)
    int_384751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 24), list_384747, int_384751)
    # Adding element type (line 32)
    int_384752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 24), list_384747, int_384752)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), list_384734, list_384747)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_384753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    int_384754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 24), list_384753, int_384754)
    # Adding element type (line 33)
    int_384755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 24), list_384753, int_384755)
    # Adding element type (line 33)
    int_384756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 24), list_384753, int_384756)
    # Adding element type (line 33)
    int_384757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 24), list_384753, int_384757)
    # Adding element type (line 33)
    int_384758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 24), list_384753, int_384758)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), list_384734, list_384753)
    # Adding element type (line 30)
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_384759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    int_384760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_384759, int_384760)
    # Adding element type (line 34)
    int_384761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_384759, int_384761)
    # Adding element type (line 34)
    int_384762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_384759, int_384762)
    # Adding element type (line 34)
    int_384763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_384759, int_384763)
    # Adding element type (line 34)
    int_384764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_384759, int_384764)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), list_384734, list_384759)
    
    # Processing the call keyword arguments (line 30)
    kwargs_384765 = {}
    # Getting the type of 'np' (line 30)
    np_384732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 30)
    array_384733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 14), np_384732, 'array')
    # Calling array(args, kwargs) (line 30)
    array_call_result_384766 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), array_384733, *[list_384734], **kwargs_384765)
    
    # Assigning a type to the variable 'csgraph' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'csgraph', array_call_result_384766)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to csgraph_from_dense(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'csgraph' (line 35)
    csgraph_384768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'csgraph', False)
    # Processing the call keyword arguments (line 35)
    int_384769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 53), 'int')
    keyword_384770 = int_384769
    kwargs_384771 = {'null_value': keyword_384770}
    # Getting the type of 'csgraph_from_dense' (line 35)
    csgraph_from_dense_384767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 35)
    csgraph_from_dense_call_result_384772 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), csgraph_from_dense_384767, *[csgraph_384768], **kwargs_384771)
    
    # Assigning a type to the variable 'csgraph' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'csgraph', csgraph_from_dense_call_result_384772)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to array(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_384775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_384776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_384777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), list_384776, int_384777)
    # Adding element type (line 37)
    int_384778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), list_384776, int_384778)
    # Adding element type (line 37)
    int_384779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), list_384776, int_384779)
    # Adding element type (line 37)
    int_384780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), list_384776, int_384780)
    # Adding element type (line 37)
    int_384781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), list_384776, int_384781)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_384775, list_384776)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_384782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    int_384783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 23), list_384782, int_384783)
    # Adding element type (line 38)
    int_384784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 23), list_384782, int_384784)
    # Adding element type (line 38)
    int_384785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 23), list_384782, int_384785)
    # Adding element type (line 38)
    int_384786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 23), list_384782, int_384786)
    # Adding element type (line 38)
    int_384787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 23), list_384782, int_384787)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_384775, list_384782)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_384788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    int_384789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), list_384788, int_384789)
    # Adding element type (line 39)
    int_384790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), list_384788, int_384790)
    # Adding element type (line 39)
    int_384791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), list_384788, int_384791)
    # Adding element type (line 39)
    int_384792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), list_384788, int_384792)
    # Adding element type (line 39)
    int_384793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), list_384788, int_384793)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_384775, list_384788)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_384794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    int_384795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), list_384794, int_384795)
    # Adding element type (line 40)
    int_384796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), list_384794, int_384796)
    # Adding element type (line 40)
    int_384797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), list_384794, int_384797)
    # Adding element type (line 40)
    int_384798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), list_384794, int_384798)
    # Adding element type (line 40)
    int_384799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), list_384794, int_384799)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_384775, list_384794)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_384800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_384801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 23), list_384800, int_384801)
    # Adding element type (line 41)
    int_384802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 23), list_384800, int_384802)
    # Adding element type (line 41)
    int_384803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 23), list_384800, int_384803)
    # Adding element type (line 41)
    int_384804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 23), list_384800, int_384804)
    # Adding element type (line 41)
    int_384805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 23), list_384800, int_384805)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_384775, list_384800)
    
    # Processing the call keyword arguments (line 37)
    kwargs_384806 = {}
    # Getting the type of 'np' (line 37)
    np_384773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 37)
    array_384774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), np_384773, 'array')
    # Calling array(args, kwargs) (line 37)
    array_call_result_384807 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), array_384774, *[list_384775], **kwargs_384806)
    
    # Assigning a type to the variable 'dfirst' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'dfirst', array_call_result_384807)
    
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_384808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    # Getting the type of 'True' (line 43)
    True_384809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), list_384808, True_384809)
    # Adding element type (line 43)
    # Getting the type of 'False' (line 43)
    False_384810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), list_384808, False_384810)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 4), list_384808)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_384811 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 4), list_384808)
    # Assigning a type to the variable 'directed' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'directed', for_loop_var_384811)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 44):
    
    # Call to depth_first_tree(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'csgraph' (line 44)
    csgraph_384813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'csgraph', False)
    int_384814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 48), 'int')
    # Getting the type of 'directed' (line 44)
    directed_384815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 51), 'directed', False)
    # Processing the call keyword arguments (line 44)
    kwargs_384816 = {}
    # Getting the type of 'depth_first_tree' (line 44)
    depth_first_tree_384812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'depth_first_tree', False)
    # Calling depth_first_tree(args, kwargs) (line 44)
    depth_first_tree_call_result_384817 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), depth_first_tree_384812, *[csgraph_384813, int_384814, directed_384815], **kwargs_384816)
    
    # Assigning a type to the variable 'dfirst_test' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'dfirst_test', depth_first_tree_call_result_384817)
    
    # Call to assert_array_almost_equal(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to csgraph_to_dense(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'dfirst_test' (line 45)
    dfirst_test_384820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 51), 'dfirst_test', False)
    # Processing the call keyword arguments (line 45)
    kwargs_384821 = {}
    # Getting the type of 'csgraph_to_dense' (line 45)
    csgraph_to_dense_384819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'csgraph_to_dense', False)
    # Calling csgraph_to_dense(args, kwargs) (line 45)
    csgraph_to_dense_call_result_384822 = invoke(stypy.reporting.localization.Localization(__file__, 45, 34), csgraph_to_dense_384819, *[dfirst_test_384820], **kwargs_384821)
    
    # Getting the type of 'dfirst' (line 46)
    dfirst_384823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'dfirst', False)
    # Processing the call keyword arguments (line 45)
    kwargs_384824 = {}
    # Getting the type of 'assert_array_almost_equal' (line 45)
    assert_array_almost_equal_384818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 45)
    assert_array_almost_equal_call_result_384825 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_array_almost_equal_384818, *[csgraph_to_dense_call_result_384822, dfirst_384823], **kwargs_384824)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_graph_depth_first(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_depth_first' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_384826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384826)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_depth_first'
    return stypy_return_type_384826

# Assigning a type to the variable 'test_graph_depth_first' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'test_graph_depth_first', test_graph_depth_first)

@norecursion
def test_graph_breadth_first_trivial_graph(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_breadth_first_trivial_graph'
    module_type_store = module_type_store.open_function_context('test_graph_breadth_first_trivial_graph', 49, 0, False)
    
    # Passed parameters checking function
    test_graph_breadth_first_trivial_graph.stypy_localization = localization
    test_graph_breadth_first_trivial_graph.stypy_type_of_self = None
    test_graph_breadth_first_trivial_graph.stypy_type_store = module_type_store
    test_graph_breadth_first_trivial_graph.stypy_function_name = 'test_graph_breadth_first_trivial_graph'
    test_graph_breadth_first_trivial_graph.stypy_param_names_list = []
    test_graph_breadth_first_trivial_graph.stypy_varargs_param_name = None
    test_graph_breadth_first_trivial_graph.stypy_kwargs_param_name = None
    test_graph_breadth_first_trivial_graph.stypy_call_defaults = defaults
    test_graph_breadth_first_trivial_graph.stypy_call_varargs = varargs
    test_graph_breadth_first_trivial_graph.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_breadth_first_trivial_graph', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_breadth_first_trivial_graph', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_breadth_first_trivial_graph(...)' code ##################

    
    # Assigning a Call to a Name (line 50):
    
    # Call to array(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_384829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_384830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    int_384831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 24), list_384830, int_384831)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 23), list_384829, list_384830)
    
    # Processing the call keyword arguments (line 50)
    kwargs_384832 = {}
    # Getting the type of 'np' (line 50)
    np_384827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 50)
    array_384828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 14), np_384827, 'array')
    # Calling array(args, kwargs) (line 50)
    array_call_result_384833 = invoke(stypy.reporting.localization.Localization(__file__, 50, 14), array_384828, *[list_384829], **kwargs_384832)
    
    # Assigning a type to the variable 'csgraph' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'csgraph', array_call_result_384833)
    
    # Assigning a Call to a Name (line 51):
    
    # Call to csgraph_from_dense(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'csgraph' (line 51)
    csgraph_384835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'csgraph', False)
    # Processing the call keyword arguments (line 51)
    int_384836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 53), 'int')
    keyword_384837 = int_384836
    kwargs_384838 = {'null_value': keyword_384837}
    # Getting the type of 'csgraph_from_dense' (line 51)
    csgraph_from_dense_384834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 51)
    csgraph_from_dense_call_result_384839 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), csgraph_from_dense_384834, *[csgraph_384835], **kwargs_384838)
    
    # Assigning a type to the variable 'csgraph' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'csgraph', csgraph_from_dense_call_result_384839)
    
    # Assigning a Call to a Name (line 53):
    
    # Call to array(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_384842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_384843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    int_384844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 23), list_384843, int_384844)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 22), list_384842, list_384843)
    
    # Processing the call keyword arguments (line 53)
    kwargs_384845 = {}
    # Getting the type of 'np' (line 53)
    np_384840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 53)
    array_384841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 13), np_384840, 'array')
    # Calling array(args, kwargs) (line 53)
    array_call_result_384846 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), array_384841, *[list_384842], **kwargs_384845)
    
    # Assigning a type to the variable 'bfirst' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'bfirst', array_call_result_384846)
    
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_384847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    # Getting the type of 'True' (line 55)
    True_384848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_384847, True_384848)
    # Adding element type (line 55)
    # Getting the type of 'False' (line 55)
    False_384849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 20), list_384847, False_384849)
    
    # Testing the type of a for loop iterable (line 55)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 4), list_384847)
    # Getting the type of the for loop variable (line 55)
    for_loop_var_384850 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 4), list_384847)
    # Assigning a type to the variable 'directed' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'directed', for_loop_var_384850)
    # SSA begins for a for statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 56):
    
    # Call to breadth_first_tree(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'csgraph' (line 56)
    csgraph_384852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'csgraph', False)
    int_384853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 50), 'int')
    # Getting the type of 'directed' (line 56)
    directed_384854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 53), 'directed', False)
    # Processing the call keyword arguments (line 56)
    kwargs_384855 = {}
    # Getting the type of 'breadth_first_tree' (line 56)
    breadth_first_tree_384851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'breadth_first_tree', False)
    # Calling breadth_first_tree(args, kwargs) (line 56)
    breadth_first_tree_call_result_384856 = invoke(stypy.reporting.localization.Localization(__file__, 56, 22), breadth_first_tree_384851, *[csgraph_384852, int_384853, directed_384854], **kwargs_384855)
    
    # Assigning a type to the variable 'bfirst_test' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'bfirst_test', breadth_first_tree_call_result_384856)
    
    # Call to assert_array_almost_equal(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Call to csgraph_to_dense(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'bfirst_test' (line 57)
    bfirst_test_384859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 51), 'bfirst_test', False)
    # Processing the call keyword arguments (line 57)
    kwargs_384860 = {}
    # Getting the type of 'csgraph_to_dense' (line 57)
    csgraph_to_dense_384858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'csgraph_to_dense', False)
    # Calling csgraph_to_dense(args, kwargs) (line 57)
    csgraph_to_dense_call_result_384861 = invoke(stypy.reporting.localization.Localization(__file__, 57, 34), csgraph_to_dense_384858, *[bfirst_test_384859], **kwargs_384860)
    
    # Getting the type of 'bfirst' (line 58)
    bfirst_384862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'bfirst', False)
    # Processing the call keyword arguments (line 57)
    kwargs_384863 = {}
    # Getting the type of 'assert_array_almost_equal' (line 57)
    assert_array_almost_equal_384857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 57)
    assert_array_almost_equal_call_result_384864 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assert_array_almost_equal_384857, *[csgraph_to_dense_call_result_384861, bfirst_384862], **kwargs_384863)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_graph_breadth_first_trivial_graph(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_breadth_first_trivial_graph' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_384865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384865)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_breadth_first_trivial_graph'
    return stypy_return_type_384865

# Assigning a type to the variable 'test_graph_breadth_first_trivial_graph' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'test_graph_breadth_first_trivial_graph', test_graph_breadth_first_trivial_graph)

@norecursion
def test_graph_depth_first_trivial_graph(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_graph_depth_first_trivial_graph'
    module_type_store = module_type_store.open_function_context('test_graph_depth_first_trivial_graph', 61, 0, False)
    
    # Passed parameters checking function
    test_graph_depth_first_trivial_graph.stypy_localization = localization
    test_graph_depth_first_trivial_graph.stypy_type_of_self = None
    test_graph_depth_first_trivial_graph.stypy_type_store = module_type_store
    test_graph_depth_first_trivial_graph.stypy_function_name = 'test_graph_depth_first_trivial_graph'
    test_graph_depth_first_trivial_graph.stypy_param_names_list = []
    test_graph_depth_first_trivial_graph.stypy_varargs_param_name = None
    test_graph_depth_first_trivial_graph.stypy_kwargs_param_name = None
    test_graph_depth_first_trivial_graph.stypy_call_defaults = defaults
    test_graph_depth_first_trivial_graph.stypy_call_varargs = varargs
    test_graph_depth_first_trivial_graph.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_graph_depth_first_trivial_graph', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_graph_depth_first_trivial_graph', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_graph_depth_first_trivial_graph(...)' code ##################

    
    # Assigning a Call to a Name (line 62):
    
    # Call to array(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_384868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    # Adding element type (line 62)
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_384869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    # Adding element type (line 62)
    int_384870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 24), list_384869, int_384870)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 23), list_384868, list_384869)
    
    # Processing the call keyword arguments (line 62)
    kwargs_384871 = {}
    # Getting the type of 'np' (line 62)
    np_384866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 62)
    array_384867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 14), np_384866, 'array')
    # Calling array(args, kwargs) (line 62)
    array_call_result_384872 = invoke(stypy.reporting.localization.Localization(__file__, 62, 14), array_384867, *[list_384868], **kwargs_384871)
    
    # Assigning a type to the variable 'csgraph' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'csgraph', array_call_result_384872)
    
    # Assigning a Call to a Name (line 63):
    
    # Call to csgraph_from_dense(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'csgraph' (line 63)
    csgraph_384874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'csgraph', False)
    # Processing the call keyword arguments (line 63)
    int_384875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 53), 'int')
    keyword_384876 = int_384875
    kwargs_384877 = {'null_value': keyword_384876}
    # Getting the type of 'csgraph_from_dense' (line 63)
    csgraph_from_dense_384873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 63)
    csgraph_from_dense_call_result_384878 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), csgraph_from_dense_384873, *[csgraph_384874], **kwargs_384877)
    
    # Assigning a type to the variable 'csgraph' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'csgraph', csgraph_from_dense_call_result_384878)
    
    # Assigning a Call to a Name (line 65):
    
    # Call to array(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_384881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_384882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    int_384883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 23), list_384882, int_384883)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_384881, list_384882)
    
    # Processing the call keyword arguments (line 65)
    kwargs_384884 = {}
    # Getting the type of 'np' (line 65)
    np_384879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 65)
    array_384880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), np_384879, 'array')
    # Calling array(args, kwargs) (line 65)
    array_call_result_384885 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), array_384880, *[list_384881], **kwargs_384884)
    
    # Assigning a type to the variable 'bfirst' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'bfirst', array_call_result_384885)
    
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_384886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    # Getting the type of 'True' (line 67)
    True_384887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), list_384886, True_384887)
    # Adding element type (line 67)
    # Getting the type of 'False' (line 67)
    False_384888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), list_384886, False_384888)
    
    # Testing the type of a for loop iterable (line 67)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 4), list_384886)
    # Getting the type of the for loop variable (line 67)
    for_loop_var_384889 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 4), list_384886)
    # Assigning a type to the variable 'directed' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'directed', for_loop_var_384889)
    # SSA begins for a for statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 68):
    
    # Call to depth_first_tree(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'csgraph' (line 68)
    csgraph_384891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'csgraph', False)
    int_384892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'int')
    # Getting the type of 'directed' (line 68)
    directed_384893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 51), 'directed', False)
    # Processing the call keyword arguments (line 68)
    kwargs_384894 = {}
    # Getting the type of 'depth_first_tree' (line 68)
    depth_first_tree_384890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'depth_first_tree', False)
    # Calling depth_first_tree(args, kwargs) (line 68)
    depth_first_tree_call_result_384895 = invoke(stypy.reporting.localization.Localization(__file__, 68, 22), depth_first_tree_384890, *[csgraph_384891, int_384892, directed_384893], **kwargs_384894)
    
    # Assigning a type to the variable 'bfirst_test' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'bfirst_test', depth_first_tree_call_result_384895)
    
    # Call to assert_array_almost_equal(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to csgraph_to_dense(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'bfirst_test' (line 69)
    bfirst_test_384898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 51), 'bfirst_test', False)
    # Processing the call keyword arguments (line 69)
    kwargs_384899 = {}
    # Getting the type of 'csgraph_to_dense' (line 69)
    csgraph_to_dense_384897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'csgraph_to_dense', False)
    # Calling csgraph_to_dense(args, kwargs) (line 69)
    csgraph_to_dense_call_result_384900 = invoke(stypy.reporting.localization.Localization(__file__, 69, 34), csgraph_to_dense_384897, *[bfirst_test_384898], **kwargs_384899)
    
    # Getting the type of 'bfirst' (line 70)
    bfirst_384901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 34), 'bfirst', False)
    # Processing the call keyword arguments (line 69)
    kwargs_384902 = {}
    # Getting the type of 'assert_array_almost_equal' (line 69)
    assert_array_almost_equal_384896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 69)
    assert_array_almost_equal_call_result_384903 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assert_array_almost_equal_384896, *[csgraph_to_dense_call_result_384900, bfirst_384901], **kwargs_384902)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_graph_depth_first_trivial_graph(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_graph_depth_first_trivial_graph' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_384904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_384904)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_graph_depth_first_trivial_graph'
    return stypy_return_type_384904

# Assigning a type to the variable 'test_graph_depth_first_trivial_graph' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'test_graph_depth_first_trivial_graph', test_graph_depth_first_trivial_graph)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
