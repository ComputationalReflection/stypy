
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal, assert_array_almost_equal
5: from scipy.sparse import csgraph
6: 
7: 
8: def test_weak_connections():
9:     Xde = np.array([[0, 1, 0],
10:                     [0, 0, 0],
11:                     [0, 0, 0]])
12: 
13:     Xsp = csgraph.csgraph_from_dense(Xde, null_value=0)
14: 
15:     for X in Xsp, Xde:
16:         n_components, labels =\
17:             csgraph.connected_components(X, directed=True,
18:                                          connection='weak')
19: 
20:         assert_equal(n_components, 2)
21:         assert_array_almost_equal(labels, [0, 0, 1])
22: 
23: 
24: def test_strong_connections():
25:     X1de = np.array([[0, 1, 0],
26:                      [0, 0, 0],
27:                      [0, 0, 0]])
28:     X2de = X1de + X1de.T
29: 
30:     X1sp = csgraph.csgraph_from_dense(X1de, null_value=0)
31:     X2sp = csgraph.csgraph_from_dense(X2de, null_value=0)
32: 
33:     for X in X1sp, X1de:
34:         n_components, labels =\
35:             csgraph.connected_components(X, directed=True,
36:                                          connection='strong')
37: 
38:         assert_equal(n_components, 3)
39:         labels.sort()
40:         assert_array_almost_equal(labels, [0, 1, 2])
41: 
42:     for X in X2sp, X2de:
43:         n_components, labels =\
44:             csgraph.connected_components(X, directed=True,
45:                                          connection='strong')
46: 
47:         assert_equal(n_components, 2)
48:         labels.sort()
49:         assert_array_almost_equal(labels, [0, 0, 1])
50: 
51: 
52: def test_strong_connections2():
53:     X = np.array([[0, 0, 0, 0, 0, 0],
54:                   [1, 0, 1, 0, 0, 0],
55:                   [0, 0, 0, 1, 0, 0],
56:                   [0, 0, 1, 0, 1, 0],
57:                   [0, 0, 0, 0, 0, 0],
58:                   [0, 0, 0, 0, 1, 0]])
59:     n_components, labels =\
60:         csgraph.connected_components(X, directed=True,
61:                                      connection='strong')
62:     assert_equal(n_components, 5)
63:     labels.sort()
64:     assert_array_almost_equal(labels, [0, 1, 2, 2, 3, 4])
65: 
66: 
67: def test_weak_connections2():
68:     X = np.array([[0, 0, 0, 0, 0, 0],
69:                   [1, 0, 0, 0, 0, 0],
70:                   [0, 0, 0, 1, 0, 0],
71:                   [0, 0, 1, 0, 1, 0],
72:                   [0, 0, 0, 0, 0, 0],
73:                   [0, 0, 0, 0, 1, 0]])
74:     n_components, labels =\
75:         csgraph.connected_components(X, directed=True,
76:                                      connection='weak')
77:     assert_equal(n_components, 2)
78:     labels.sort()
79:     assert_array_almost_equal(labels, [0, 0, 1, 1, 1, 1])
80: 
81: 
82: def test_ticket1876():
83:     # Regression test: this failed in the original implementation
84:     # There should be two strongly-connected components; previously gave one
85:     g = np.array([[0, 1, 1, 0],
86:                   [1, 0, 0, 1],
87:                   [0, 0, 0, 1],
88:                   [0, 0, 1, 0]])
89:     n_components, labels = csgraph.connected_components(g, connection='strong')
90: 
91:     assert_equal(n_components, 2)
92:     assert_equal(labels[0], labels[1])
93:     assert_equal(labels[2], labels[3])
94: 
95: 
96: def test_fully_connected_graph():
97:     # Fully connected dense matrices raised an exception.
98:     # https://github.com/scipy/scipy/issues/3818
99:     g = np.ones((4, 4))
100:     n_components, labels = csgraph.connected_components(g)
101:     assert_equal(n_components, 1)
102: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_381696 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_381696) is not StypyTypeError):

    if (import_381696 != 'pyd_module'):
        __import__(import_381696)
        sys_modules_381697 = sys.modules[import_381696]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_381697.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_381696)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_array_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_381698 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_381698) is not StypyTypeError):

    if (import_381698 != 'pyd_module'):
        __import__(import_381698)
        sys_modules_381699 = sys.modules[import_381698]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_381699.module_type_store, module_type_store, ['assert_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_381699, sys_modules_381699.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_almost_equal'], [assert_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_381698)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse import csgraph' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_381700 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse')

if (type(import_381700) is not StypyTypeError):

    if (import_381700 != 'pyd_module'):
        __import__(import_381700)
        sys_modules_381701 = sys.modules[import_381700]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', sys_modules_381701.module_type_store, module_type_store, ['csgraph'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_381701, sys_modules_381701.module_type_store, module_type_store)
    else:
        from scipy.sparse import csgraph

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', None, module_type_store, ['csgraph'], [csgraph])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', import_381700)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')


@norecursion
def test_weak_connections(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_weak_connections'
    module_type_store = module_type_store.open_function_context('test_weak_connections', 8, 0, False)
    
    # Passed parameters checking function
    test_weak_connections.stypy_localization = localization
    test_weak_connections.stypy_type_of_self = None
    test_weak_connections.stypy_type_store = module_type_store
    test_weak_connections.stypy_function_name = 'test_weak_connections'
    test_weak_connections.stypy_param_names_list = []
    test_weak_connections.stypy_varargs_param_name = None
    test_weak_connections.stypy_kwargs_param_name = None
    test_weak_connections.stypy_call_defaults = defaults
    test_weak_connections.stypy_call_varargs = varargs
    test_weak_connections.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_weak_connections', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_weak_connections', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_weak_connections(...)' code ##################

    
    # Assigning a Call to a Name (line 9):
    
    # Assigning a Call to a Name (line 9):
    
    # Call to array(...): (line 9)
    # Processing the call arguments (line 9)
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_381704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_381705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    int_381706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 20), list_381705, int_381706)
    # Adding element type (line 9)
    int_381707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 20), list_381705, int_381707)
    # Adding element type (line 9)
    int_381708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 20), list_381705, int_381708)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 19), list_381704, list_381705)
    # Adding element type (line 9)
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_381709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    int_381710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 20), list_381709, int_381710)
    # Adding element type (line 10)
    int_381711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 20), list_381709, int_381711)
    # Adding element type (line 10)
    int_381712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 20), list_381709, int_381712)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 19), list_381704, list_381709)
    # Adding element type (line 9)
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_381713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    int_381714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 20), list_381713, int_381714)
    # Adding element type (line 11)
    int_381715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 20), list_381713, int_381715)
    # Adding element type (line 11)
    int_381716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 20), list_381713, int_381716)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 19), list_381704, list_381713)
    
    # Processing the call keyword arguments (line 9)
    kwargs_381717 = {}
    # Getting the type of 'np' (line 9)
    np_381702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'np', False)
    # Obtaining the member 'array' of a type (line 9)
    array_381703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 10), np_381702, 'array')
    # Calling array(args, kwargs) (line 9)
    array_call_result_381718 = invoke(stypy.reporting.localization.Localization(__file__, 9, 10), array_381703, *[list_381704], **kwargs_381717)
    
    # Assigning a type to the variable 'Xde' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'Xde', array_call_result_381718)
    
    # Assigning a Call to a Name (line 13):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to csgraph_from_dense(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'Xde' (line 13)
    Xde_381721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 37), 'Xde', False)
    # Processing the call keyword arguments (line 13)
    int_381722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 53), 'int')
    keyword_381723 = int_381722
    kwargs_381724 = {'null_value': keyword_381723}
    # Getting the type of 'csgraph' (line 13)
    csgraph_381719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'csgraph', False)
    # Obtaining the member 'csgraph_from_dense' of a type (line 13)
    csgraph_from_dense_381720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 10), csgraph_381719, 'csgraph_from_dense')
    # Calling csgraph_from_dense(args, kwargs) (line 13)
    csgraph_from_dense_call_result_381725 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), csgraph_from_dense_381720, *[Xde_381721], **kwargs_381724)
    
    # Assigning a type to the variable 'Xsp' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'Xsp', csgraph_from_dense_call_result_381725)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 15)
    tuple_381726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 15)
    # Adding element type (line 15)
    # Getting the type of 'Xsp' (line 15)
    Xsp_381727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'Xsp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), tuple_381726, Xsp_381727)
    # Adding element type (line 15)
    # Getting the type of 'Xde' (line 15)
    Xde_381728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'Xde')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 13), tuple_381726, Xde_381728)
    
    # Testing the type of a for loop iterable (line 15)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 4), tuple_381726)
    # Getting the type of the for loop variable (line 15)
    for_loop_var_381729 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 4), tuple_381726)
    # Assigning a type to the variable 'X' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'X', for_loop_var_381729)
    # SSA begins for a for statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_381730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
    
    # Call to connected_components(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'X' (line 17)
    X_381733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'X', False)
    # Processing the call keyword arguments (line 17)
    # Getting the type of 'True' (line 17)
    True_381734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 53), 'True', False)
    keyword_381735 = True_381734
    str_381736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 52), 'str', 'weak')
    keyword_381737 = str_381736
    kwargs_381738 = {'directed': keyword_381735, 'connection': keyword_381737}
    # Getting the type of 'csgraph' (line 17)
    csgraph_381731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 17)
    connected_components_381732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), csgraph_381731, 'connected_components')
    # Calling connected_components(args, kwargs) (line 17)
    connected_components_call_result_381739 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), connected_components_381732, *[X_381733], **kwargs_381738)
    
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___381740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), connected_components_call_result_381739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_381741 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___381740, int_381730)
    
    # Assigning a type to the variable 'tuple_var_assignment_381682' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_381682', subscript_call_result_381741)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_381742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
    
    # Call to connected_components(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'X' (line 17)
    X_381745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 41), 'X', False)
    # Processing the call keyword arguments (line 17)
    # Getting the type of 'True' (line 17)
    True_381746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 53), 'True', False)
    keyword_381747 = True_381746
    str_381748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 52), 'str', 'weak')
    keyword_381749 = str_381748
    kwargs_381750 = {'directed': keyword_381747, 'connection': keyword_381749}
    # Getting the type of 'csgraph' (line 17)
    csgraph_381743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 17)
    connected_components_381744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), csgraph_381743, 'connected_components')
    # Calling connected_components(args, kwargs) (line 17)
    connected_components_call_result_381751 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), connected_components_381744, *[X_381745], **kwargs_381750)
    
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___381752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), connected_components_call_result_381751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_381753 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___381752, int_381742)
    
    # Assigning a type to the variable 'tuple_var_assignment_381683' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_381683', subscript_call_result_381753)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_381682' (line 16)
    tuple_var_assignment_381682_381754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_381682')
    # Assigning a type to the variable 'n_components' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'n_components', tuple_var_assignment_381682_381754)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_381683' (line 16)
    tuple_var_assignment_381683_381755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_381683')
    # Assigning a type to the variable 'labels' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'labels', tuple_var_assignment_381683_381755)
    
    # Call to assert_equal(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'n_components' (line 20)
    n_components_381757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'n_components', False)
    int_381758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 35), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_381759 = {}
    # Getting the type of 'assert_equal' (line 20)
    assert_equal_381756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 20)
    assert_equal_call_result_381760 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), assert_equal_381756, *[n_components_381757, int_381758], **kwargs_381759)
    
    
    # Call to assert_array_almost_equal(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'labels' (line 21)
    labels_381762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 34), 'labels', False)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_381763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_381764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 42), list_381763, int_381764)
    # Adding element type (line 21)
    int_381765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 42), list_381763, int_381765)
    # Adding element type (line 21)
    int_381766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 42), list_381763, int_381766)
    
    # Processing the call keyword arguments (line 21)
    kwargs_381767 = {}
    # Getting the type of 'assert_array_almost_equal' (line 21)
    assert_array_almost_equal_381761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 21)
    assert_array_almost_equal_call_result_381768 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_array_almost_equal_381761, *[labels_381762, list_381763], **kwargs_381767)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_weak_connections(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_weak_connections' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_381769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381769)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_weak_connections'
    return stypy_return_type_381769

# Assigning a type to the variable 'test_weak_connections' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test_weak_connections', test_weak_connections)

@norecursion
def test_strong_connections(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_strong_connections'
    module_type_store = module_type_store.open_function_context('test_strong_connections', 24, 0, False)
    
    # Passed parameters checking function
    test_strong_connections.stypy_localization = localization
    test_strong_connections.stypy_type_of_self = None
    test_strong_connections.stypy_type_store = module_type_store
    test_strong_connections.stypy_function_name = 'test_strong_connections'
    test_strong_connections.stypy_param_names_list = []
    test_strong_connections.stypy_varargs_param_name = None
    test_strong_connections.stypy_kwargs_param_name = None
    test_strong_connections.stypy_call_defaults = defaults
    test_strong_connections.stypy_call_varargs = varargs
    test_strong_connections.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_strong_connections', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_strong_connections', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_strong_connections(...)' code ##################

    
    # Assigning a Call to a Name (line 25):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to array(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_381772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_381773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_381774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), list_381773, int_381774)
    # Adding element type (line 25)
    int_381775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), list_381773, int_381775)
    # Adding element type (line 25)
    int_381776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), list_381773, int_381776)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), list_381772, list_381773)
    # Adding element type (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_381777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    int_381778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_381777, int_381778)
    # Adding element type (line 26)
    int_381779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_381777, int_381779)
    # Adding element type (line 26)
    int_381780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_381777, int_381780)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), list_381772, list_381777)
    # Adding element type (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_381781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    int_381782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_381781, int_381782)
    # Adding element type (line 27)
    int_381783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_381781, int_381783)
    # Adding element type (line 27)
    int_381784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), list_381781, int_381784)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), list_381772, list_381781)
    
    # Processing the call keyword arguments (line 25)
    kwargs_381785 = {}
    # Getting the type of 'np' (line 25)
    np_381770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 25)
    array_381771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), np_381770, 'array')
    # Calling array(args, kwargs) (line 25)
    array_call_result_381786 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), array_381771, *[list_381772], **kwargs_381785)
    
    # Assigning a type to the variable 'X1de' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'X1de', array_call_result_381786)
    
    # Assigning a BinOp to a Name (line 28):
    
    # Assigning a BinOp to a Name (line 28):
    # Getting the type of 'X1de' (line 28)
    X1de_381787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'X1de')
    # Getting the type of 'X1de' (line 28)
    X1de_381788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'X1de')
    # Obtaining the member 'T' of a type (line 28)
    T_381789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 18), X1de_381788, 'T')
    # Applying the binary operator '+' (line 28)
    result_add_381790 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), '+', X1de_381787, T_381789)
    
    # Assigning a type to the variable 'X2de' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'X2de', result_add_381790)
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to csgraph_from_dense(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'X1de' (line 30)
    X1de_381793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 38), 'X1de', False)
    # Processing the call keyword arguments (line 30)
    int_381794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 55), 'int')
    keyword_381795 = int_381794
    kwargs_381796 = {'null_value': keyword_381795}
    # Getting the type of 'csgraph' (line 30)
    csgraph_381791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'csgraph', False)
    # Obtaining the member 'csgraph_from_dense' of a type (line 30)
    csgraph_from_dense_381792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), csgraph_381791, 'csgraph_from_dense')
    # Calling csgraph_from_dense(args, kwargs) (line 30)
    csgraph_from_dense_call_result_381797 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), csgraph_from_dense_381792, *[X1de_381793], **kwargs_381796)
    
    # Assigning a type to the variable 'X1sp' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'X1sp', csgraph_from_dense_call_result_381797)
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to csgraph_from_dense(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'X2de' (line 31)
    X2de_381800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 38), 'X2de', False)
    # Processing the call keyword arguments (line 31)
    int_381801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 55), 'int')
    keyword_381802 = int_381801
    kwargs_381803 = {'null_value': keyword_381802}
    # Getting the type of 'csgraph' (line 31)
    csgraph_381798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'csgraph', False)
    # Obtaining the member 'csgraph_from_dense' of a type (line 31)
    csgraph_from_dense_381799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), csgraph_381798, 'csgraph_from_dense')
    # Calling csgraph_from_dense(args, kwargs) (line 31)
    csgraph_from_dense_call_result_381804 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), csgraph_from_dense_381799, *[X2de_381800], **kwargs_381803)
    
    # Assigning a type to the variable 'X2sp' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'X2sp', csgraph_from_dense_call_result_381804)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 33)
    tuple_381805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 33)
    # Adding element type (line 33)
    # Getting the type of 'X1sp' (line 33)
    X1sp_381806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'X1sp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 13), tuple_381805, X1sp_381806)
    # Adding element type (line 33)
    # Getting the type of 'X1de' (line 33)
    X1de_381807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'X1de')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 13), tuple_381805, X1de_381807)
    
    # Testing the type of a for loop iterable (line 33)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 33, 4), tuple_381805)
    # Getting the type of the for loop variable (line 33)
    for_loop_var_381808 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 33, 4), tuple_381805)
    # Assigning a type to the variable 'X' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'X', for_loop_var_381808)
    # SSA begins for a for statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 34):
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_381809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'int')
    
    # Call to connected_components(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'X' (line 35)
    X_381812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 41), 'X', False)
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'True' (line 35)
    True_381813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 53), 'True', False)
    keyword_381814 = True_381813
    str_381815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 52), 'str', 'strong')
    keyword_381816 = str_381815
    kwargs_381817 = {'directed': keyword_381814, 'connection': keyword_381816}
    # Getting the type of 'csgraph' (line 35)
    csgraph_381810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 35)
    connected_components_381811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), csgraph_381810, 'connected_components')
    # Calling connected_components(args, kwargs) (line 35)
    connected_components_call_result_381818 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), connected_components_381811, *[X_381812], **kwargs_381817)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___381819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), connected_components_call_result_381818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_381820 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), getitem___381819, int_381809)
    
    # Assigning a type to the variable 'tuple_var_assignment_381684' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_381684', subscript_call_result_381820)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    int_381821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'int')
    
    # Call to connected_components(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'X' (line 35)
    X_381824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 41), 'X', False)
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'True' (line 35)
    True_381825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 53), 'True', False)
    keyword_381826 = True_381825
    str_381827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 52), 'str', 'strong')
    keyword_381828 = str_381827
    kwargs_381829 = {'directed': keyword_381826, 'connection': keyword_381828}
    # Getting the type of 'csgraph' (line 35)
    csgraph_381822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 35)
    connected_components_381823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), csgraph_381822, 'connected_components')
    # Calling connected_components(args, kwargs) (line 35)
    connected_components_call_result_381830 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), connected_components_381823, *[X_381824], **kwargs_381829)
    
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___381831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), connected_components_call_result_381830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_381832 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), getitem___381831, int_381821)
    
    # Assigning a type to the variable 'tuple_var_assignment_381685' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_381685', subscript_call_result_381832)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_381684' (line 34)
    tuple_var_assignment_381684_381833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_381684')
    # Assigning a type to the variable 'n_components' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'n_components', tuple_var_assignment_381684_381833)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_var_assignment_381685' (line 34)
    tuple_var_assignment_381685_381834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_var_assignment_381685')
    # Assigning a type to the variable 'labels' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'labels', tuple_var_assignment_381685_381834)
    
    # Call to assert_equal(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'n_components' (line 38)
    n_components_381836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'n_components', False)
    int_381837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_381838 = {}
    # Getting the type of 'assert_equal' (line 38)
    assert_equal_381835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 38)
    assert_equal_call_result_381839 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assert_equal_381835, *[n_components_381836, int_381837], **kwargs_381838)
    
    
    # Call to sort(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_381842 = {}
    # Getting the type of 'labels' (line 39)
    labels_381840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'labels', False)
    # Obtaining the member 'sort' of a type (line 39)
    sort_381841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), labels_381840, 'sort')
    # Calling sort(args, kwargs) (line 39)
    sort_call_result_381843 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), sort_381841, *[], **kwargs_381842)
    
    
    # Call to assert_array_almost_equal(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'labels' (line 40)
    labels_381845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'labels', False)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_381846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    int_381847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 42), list_381846, int_381847)
    # Adding element type (line 40)
    int_381848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 42), list_381846, int_381848)
    # Adding element type (line 40)
    int_381849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 42), list_381846, int_381849)
    
    # Processing the call keyword arguments (line 40)
    kwargs_381850 = {}
    # Getting the type of 'assert_array_almost_equal' (line 40)
    assert_array_almost_equal_381844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 40)
    assert_array_almost_equal_call_result_381851 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert_array_almost_equal_381844, *[labels_381845, list_381846], **kwargs_381850)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_381852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'X2sp' (line 42)
    X2sp_381853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'X2sp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 13), tuple_381852, X2sp_381853)
    # Adding element type (line 42)
    # Getting the type of 'X2de' (line 42)
    X2de_381854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'X2de')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 13), tuple_381852, X2de_381854)
    
    # Testing the type of a for loop iterable (line 42)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 4), tuple_381852)
    # Getting the type of the for loop variable (line 42)
    for_loop_var_381855 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 4), tuple_381852)
    # Assigning a type to the variable 'X' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'X', for_loop_var_381855)
    # SSA begins for a for statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 43):
    
    # Assigning a Subscript to a Name (line 43):
    
    # Obtaining the type of the subscript
    int_381856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'int')
    
    # Call to connected_components(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'X' (line 44)
    X_381859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 41), 'X', False)
    # Processing the call keyword arguments (line 44)
    # Getting the type of 'True' (line 44)
    True_381860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 53), 'True', False)
    keyword_381861 = True_381860
    str_381862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 52), 'str', 'strong')
    keyword_381863 = str_381862
    kwargs_381864 = {'directed': keyword_381861, 'connection': keyword_381863}
    # Getting the type of 'csgraph' (line 44)
    csgraph_381857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 44)
    connected_components_381858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), csgraph_381857, 'connected_components')
    # Calling connected_components(args, kwargs) (line 44)
    connected_components_call_result_381865 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), connected_components_381858, *[X_381859], **kwargs_381864)
    
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___381866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), connected_components_call_result_381865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_381867 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), getitem___381866, int_381856)
    
    # Assigning a type to the variable 'tuple_var_assignment_381686' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'tuple_var_assignment_381686', subscript_call_result_381867)
    
    # Assigning a Subscript to a Name (line 43):
    
    # Obtaining the type of the subscript
    int_381868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'int')
    
    # Call to connected_components(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'X' (line 44)
    X_381871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 41), 'X', False)
    # Processing the call keyword arguments (line 44)
    # Getting the type of 'True' (line 44)
    True_381872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 53), 'True', False)
    keyword_381873 = True_381872
    str_381874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 52), 'str', 'strong')
    keyword_381875 = str_381874
    kwargs_381876 = {'directed': keyword_381873, 'connection': keyword_381875}
    # Getting the type of 'csgraph' (line 44)
    csgraph_381869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 44)
    connected_components_381870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), csgraph_381869, 'connected_components')
    # Calling connected_components(args, kwargs) (line 44)
    connected_components_call_result_381877 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), connected_components_381870, *[X_381871], **kwargs_381876)
    
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___381878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), connected_components_call_result_381877, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_381879 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), getitem___381878, int_381868)
    
    # Assigning a type to the variable 'tuple_var_assignment_381687' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'tuple_var_assignment_381687', subscript_call_result_381879)
    
    # Assigning a Name to a Name (line 43):
    # Getting the type of 'tuple_var_assignment_381686' (line 43)
    tuple_var_assignment_381686_381880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'tuple_var_assignment_381686')
    # Assigning a type to the variable 'n_components' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'n_components', tuple_var_assignment_381686_381880)
    
    # Assigning a Name to a Name (line 43):
    # Getting the type of 'tuple_var_assignment_381687' (line 43)
    tuple_var_assignment_381687_381881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'tuple_var_assignment_381687')
    # Assigning a type to the variable 'labels' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'labels', tuple_var_assignment_381687_381881)
    
    # Call to assert_equal(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'n_components' (line 47)
    n_components_381883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'n_components', False)
    int_381884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'int')
    # Processing the call keyword arguments (line 47)
    kwargs_381885 = {}
    # Getting the type of 'assert_equal' (line 47)
    assert_equal_381882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 47)
    assert_equal_call_result_381886 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_equal_381882, *[n_components_381883, int_381884], **kwargs_381885)
    
    
    # Call to sort(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_381889 = {}
    # Getting the type of 'labels' (line 48)
    labels_381887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'labels', False)
    # Obtaining the member 'sort' of a type (line 48)
    sort_381888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), labels_381887, 'sort')
    # Calling sort(args, kwargs) (line 48)
    sort_call_result_381890 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), sort_381888, *[], **kwargs_381889)
    
    
    # Call to assert_array_almost_equal(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'labels' (line 49)
    labels_381892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'labels', False)
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_381893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    int_381894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 42), list_381893, int_381894)
    # Adding element type (line 49)
    int_381895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 42), list_381893, int_381895)
    # Adding element type (line 49)
    int_381896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 42), list_381893, int_381896)
    
    # Processing the call keyword arguments (line 49)
    kwargs_381897 = {}
    # Getting the type of 'assert_array_almost_equal' (line 49)
    assert_array_almost_equal_381891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 49)
    assert_array_almost_equal_call_result_381898 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assert_array_almost_equal_381891, *[labels_381892, list_381893], **kwargs_381897)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_strong_connections(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_strong_connections' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_381899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_strong_connections'
    return stypy_return_type_381899

# Assigning a type to the variable 'test_strong_connections' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_strong_connections', test_strong_connections)

@norecursion
def test_strong_connections2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_strong_connections2'
    module_type_store = module_type_store.open_function_context('test_strong_connections2', 52, 0, False)
    
    # Passed parameters checking function
    test_strong_connections2.stypy_localization = localization
    test_strong_connections2.stypy_type_of_self = None
    test_strong_connections2.stypy_type_store = module_type_store
    test_strong_connections2.stypy_function_name = 'test_strong_connections2'
    test_strong_connections2.stypy_param_names_list = []
    test_strong_connections2.stypy_varargs_param_name = None
    test_strong_connections2.stypy_kwargs_param_name = None
    test_strong_connections2.stypy_call_defaults = defaults
    test_strong_connections2.stypy_call_varargs = varargs
    test_strong_connections2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_strong_connections2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_strong_connections2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_strong_connections2(...)' code ##################

    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to array(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_381902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_381903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    int_381904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_381903, int_381904)
    # Adding element type (line 53)
    int_381905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_381903, int_381905)
    # Adding element type (line 53)
    int_381906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_381903, int_381906)
    # Adding element type (line 53)
    int_381907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_381903, int_381907)
    # Adding element type (line 53)
    int_381908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_381903, int_381908)
    # Adding element type (line 53)
    int_381909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 18), list_381903, int_381909)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_381902, list_381903)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_381910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    int_381911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), list_381910, int_381911)
    # Adding element type (line 54)
    int_381912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), list_381910, int_381912)
    # Adding element type (line 54)
    int_381913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), list_381910, int_381913)
    # Adding element type (line 54)
    int_381914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), list_381910, int_381914)
    # Adding element type (line 54)
    int_381915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), list_381910, int_381915)
    # Adding element type (line 54)
    int_381916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), list_381910, int_381916)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_381902, list_381910)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_381917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    int_381918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_381917, int_381918)
    # Adding element type (line 55)
    int_381919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_381917, int_381919)
    # Adding element type (line 55)
    int_381920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_381917, int_381920)
    # Adding element type (line 55)
    int_381921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_381917, int_381921)
    # Adding element type (line 55)
    int_381922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_381917, int_381922)
    # Adding element type (line 55)
    int_381923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_381917, int_381923)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_381902, list_381917)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_381924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_381925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_381924, int_381925)
    # Adding element type (line 56)
    int_381926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_381924, int_381926)
    # Adding element type (line 56)
    int_381927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_381924, int_381927)
    # Adding element type (line 56)
    int_381928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_381924, int_381928)
    # Adding element type (line 56)
    int_381929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_381924, int_381929)
    # Adding element type (line 56)
    int_381930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 18), list_381924, int_381930)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_381902, list_381924)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 57)
    list_381931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 57)
    # Adding element type (line 57)
    int_381932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 18), list_381931, int_381932)
    # Adding element type (line 57)
    int_381933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 18), list_381931, int_381933)
    # Adding element type (line 57)
    int_381934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 18), list_381931, int_381934)
    # Adding element type (line 57)
    int_381935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 18), list_381931, int_381935)
    # Adding element type (line 57)
    int_381936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 18), list_381931, int_381936)
    # Adding element type (line 57)
    int_381937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 18), list_381931, int_381937)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_381902, list_381931)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_381938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    int_381939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), list_381938, int_381939)
    # Adding element type (line 58)
    int_381940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), list_381938, int_381940)
    # Adding element type (line 58)
    int_381941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), list_381938, int_381941)
    # Adding element type (line 58)
    int_381942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), list_381938, int_381942)
    # Adding element type (line 58)
    int_381943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), list_381938, int_381943)
    # Adding element type (line 58)
    int_381944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), list_381938, int_381944)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 17), list_381902, list_381938)
    
    # Processing the call keyword arguments (line 53)
    kwargs_381945 = {}
    # Getting the type of 'np' (line 53)
    np_381900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 53)
    array_381901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), np_381900, 'array')
    # Calling array(args, kwargs) (line 53)
    array_call_result_381946 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), array_381901, *[list_381902], **kwargs_381945)
    
    # Assigning a type to the variable 'X' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'X', array_call_result_381946)
    
    # Assigning a Call to a Tuple (line 59):
    
    # Assigning a Subscript to a Name (line 59):
    
    # Obtaining the type of the subscript
    int_381947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'int')
    
    # Call to connected_components(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'X' (line 60)
    X_381950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'X', False)
    # Processing the call keyword arguments (line 60)
    # Getting the type of 'True' (line 60)
    True_381951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'True', False)
    keyword_381952 = True_381951
    str_381953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 48), 'str', 'strong')
    keyword_381954 = str_381953
    kwargs_381955 = {'directed': keyword_381952, 'connection': keyword_381954}
    # Getting the type of 'csgraph' (line 60)
    csgraph_381948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 60)
    connected_components_381949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), csgraph_381948, 'connected_components')
    # Calling connected_components(args, kwargs) (line 60)
    connected_components_call_result_381956 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), connected_components_381949, *[X_381950], **kwargs_381955)
    
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___381957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), connected_components_call_result_381956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_381958 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), getitem___381957, int_381947)
    
    # Assigning a type to the variable 'tuple_var_assignment_381688' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_381688', subscript_call_result_381958)
    
    # Assigning a Subscript to a Name (line 59):
    
    # Obtaining the type of the subscript
    int_381959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'int')
    
    # Call to connected_components(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'X' (line 60)
    X_381962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'X', False)
    # Processing the call keyword arguments (line 60)
    # Getting the type of 'True' (line 60)
    True_381963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'True', False)
    keyword_381964 = True_381963
    str_381965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 48), 'str', 'strong')
    keyword_381966 = str_381965
    kwargs_381967 = {'directed': keyword_381964, 'connection': keyword_381966}
    # Getting the type of 'csgraph' (line 60)
    csgraph_381960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 60)
    connected_components_381961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), csgraph_381960, 'connected_components')
    # Calling connected_components(args, kwargs) (line 60)
    connected_components_call_result_381968 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), connected_components_381961, *[X_381962], **kwargs_381967)
    
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___381969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), connected_components_call_result_381968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_381970 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), getitem___381969, int_381959)
    
    # Assigning a type to the variable 'tuple_var_assignment_381689' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_381689', subscript_call_result_381970)
    
    # Assigning a Name to a Name (line 59):
    # Getting the type of 'tuple_var_assignment_381688' (line 59)
    tuple_var_assignment_381688_381971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_381688')
    # Assigning a type to the variable 'n_components' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'n_components', tuple_var_assignment_381688_381971)
    
    # Assigning a Name to a Name (line 59):
    # Getting the type of 'tuple_var_assignment_381689' (line 59)
    tuple_var_assignment_381689_381972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'tuple_var_assignment_381689')
    # Assigning a type to the variable 'labels' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'labels', tuple_var_assignment_381689_381972)
    
    # Call to assert_equal(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'n_components' (line 62)
    n_components_381974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'n_components', False)
    int_381975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 31), 'int')
    # Processing the call keyword arguments (line 62)
    kwargs_381976 = {}
    # Getting the type of 'assert_equal' (line 62)
    assert_equal_381973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 62)
    assert_equal_call_result_381977 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), assert_equal_381973, *[n_components_381974, int_381975], **kwargs_381976)
    
    
    # Call to sort(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_381980 = {}
    # Getting the type of 'labels' (line 63)
    labels_381978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'labels', False)
    # Obtaining the member 'sort' of a type (line 63)
    sort_381979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), labels_381978, 'sort')
    # Calling sort(args, kwargs) (line 63)
    sort_call_result_381981 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), sort_381979, *[], **kwargs_381980)
    
    
    # Call to assert_array_almost_equal(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'labels' (line 64)
    labels_381983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'labels', False)
    
    # Obtaining an instance of the builtin type 'list' (line 64)
    list_381984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 64)
    # Adding element type (line 64)
    int_381985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 38), list_381984, int_381985)
    # Adding element type (line 64)
    int_381986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 38), list_381984, int_381986)
    # Adding element type (line 64)
    int_381987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 38), list_381984, int_381987)
    # Adding element type (line 64)
    int_381988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 38), list_381984, int_381988)
    # Adding element type (line 64)
    int_381989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 38), list_381984, int_381989)
    # Adding element type (line 64)
    int_381990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 38), list_381984, int_381990)
    
    # Processing the call keyword arguments (line 64)
    kwargs_381991 = {}
    # Getting the type of 'assert_array_almost_equal' (line 64)
    assert_array_almost_equal_381982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 64)
    assert_array_almost_equal_call_result_381992 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), assert_array_almost_equal_381982, *[labels_381983, list_381984], **kwargs_381991)
    
    
    # ################# End of 'test_strong_connections2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_strong_connections2' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_381993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381993)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_strong_connections2'
    return stypy_return_type_381993

# Assigning a type to the variable 'test_strong_connections2' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'test_strong_connections2', test_strong_connections2)

@norecursion
def test_weak_connections2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_weak_connections2'
    module_type_store = module_type_store.open_function_context('test_weak_connections2', 67, 0, False)
    
    # Passed parameters checking function
    test_weak_connections2.stypy_localization = localization
    test_weak_connections2.stypy_type_of_self = None
    test_weak_connections2.stypy_type_store = module_type_store
    test_weak_connections2.stypy_function_name = 'test_weak_connections2'
    test_weak_connections2.stypy_param_names_list = []
    test_weak_connections2.stypy_varargs_param_name = None
    test_weak_connections2.stypy_kwargs_param_name = None
    test_weak_connections2.stypy_call_defaults = defaults
    test_weak_connections2.stypy_call_varargs = varargs
    test_weak_connections2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_weak_connections2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_weak_connections2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_weak_connections2(...)' code ##################

    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to array(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_381996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_381997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    int_381998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_381997, int_381998)
    # Adding element type (line 68)
    int_381999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_381997, int_381999)
    # Adding element type (line 68)
    int_382000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_381997, int_382000)
    # Adding element type (line 68)
    int_382001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_381997, int_382001)
    # Adding element type (line 68)
    int_382002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_381997, int_382002)
    # Adding element type (line 68)
    int_382003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_381997, int_382003)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 17), list_381996, list_381997)
    # Adding element type (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_382004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    # Adding element type (line 69)
    int_382005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_382004, int_382005)
    # Adding element type (line 69)
    int_382006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_382004, int_382006)
    # Adding element type (line 69)
    int_382007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_382004, int_382007)
    # Adding element type (line 69)
    int_382008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_382004, int_382008)
    # Adding element type (line 69)
    int_382009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_382004, int_382009)
    # Adding element type (line 69)
    int_382010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_382004, int_382010)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 17), list_381996, list_382004)
    # Adding element type (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_382011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    # Adding element type (line 70)
    int_382012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), list_382011, int_382012)
    # Adding element type (line 70)
    int_382013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), list_382011, int_382013)
    # Adding element type (line 70)
    int_382014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), list_382011, int_382014)
    # Adding element type (line 70)
    int_382015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), list_382011, int_382015)
    # Adding element type (line 70)
    int_382016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), list_382011, int_382016)
    # Adding element type (line 70)
    int_382017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), list_382011, int_382017)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 17), list_381996, list_382011)
    # Adding element type (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_382018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    int_382019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_382018, int_382019)
    # Adding element type (line 71)
    int_382020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_382018, int_382020)
    # Adding element type (line 71)
    int_382021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_382018, int_382021)
    # Adding element type (line 71)
    int_382022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_382018, int_382022)
    # Adding element type (line 71)
    int_382023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_382018, int_382023)
    # Adding element type (line 71)
    int_382024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), list_382018, int_382024)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 17), list_381996, list_382018)
    # Adding element type (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 72)
    list_382025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 72)
    # Adding element type (line 72)
    int_382026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_382025, int_382026)
    # Adding element type (line 72)
    int_382027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_382025, int_382027)
    # Adding element type (line 72)
    int_382028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_382025, int_382028)
    # Adding element type (line 72)
    int_382029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_382025, int_382029)
    # Adding element type (line 72)
    int_382030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_382025, int_382030)
    # Adding element type (line 72)
    int_382031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_382025, int_382031)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 17), list_381996, list_382025)
    # Adding element type (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 73)
    list_382032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 73)
    # Adding element type (line 73)
    int_382033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), list_382032, int_382033)
    # Adding element type (line 73)
    int_382034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), list_382032, int_382034)
    # Adding element type (line 73)
    int_382035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), list_382032, int_382035)
    # Adding element type (line 73)
    int_382036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), list_382032, int_382036)
    # Adding element type (line 73)
    int_382037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), list_382032, int_382037)
    # Adding element type (line 73)
    int_382038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), list_382032, int_382038)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 17), list_381996, list_382032)
    
    # Processing the call keyword arguments (line 68)
    kwargs_382039 = {}
    # Getting the type of 'np' (line 68)
    np_381994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 68)
    array_381995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), np_381994, 'array')
    # Calling array(args, kwargs) (line 68)
    array_call_result_382040 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), array_381995, *[list_381996], **kwargs_382039)
    
    # Assigning a type to the variable 'X' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'X', array_call_result_382040)
    
    # Assigning a Call to a Tuple (line 74):
    
    # Assigning a Subscript to a Name (line 74):
    
    # Obtaining the type of the subscript
    int_382041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'int')
    
    # Call to connected_components(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'X' (line 75)
    X_382044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'X', False)
    # Processing the call keyword arguments (line 75)
    # Getting the type of 'True' (line 75)
    True_382045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'True', False)
    keyword_382046 = True_382045
    str_382047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 48), 'str', 'weak')
    keyword_382048 = str_382047
    kwargs_382049 = {'directed': keyword_382046, 'connection': keyword_382048}
    # Getting the type of 'csgraph' (line 75)
    csgraph_382042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 75)
    connected_components_382043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), csgraph_382042, 'connected_components')
    # Calling connected_components(args, kwargs) (line 75)
    connected_components_call_result_382050 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), connected_components_382043, *[X_382044], **kwargs_382049)
    
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___382051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), connected_components_call_result_382050, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_382052 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), getitem___382051, int_382041)
    
    # Assigning a type to the variable 'tuple_var_assignment_381690' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_381690', subscript_call_result_382052)
    
    # Assigning a Subscript to a Name (line 74):
    
    # Obtaining the type of the subscript
    int_382053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'int')
    
    # Call to connected_components(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'X' (line 75)
    X_382056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'X', False)
    # Processing the call keyword arguments (line 75)
    # Getting the type of 'True' (line 75)
    True_382057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 49), 'True', False)
    keyword_382058 = True_382057
    str_382059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 48), 'str', 'weak')
    keyword_382060 = str_382059
    kwargs_382061 = {'directed': keyword_382058, 'connection': keyword_382060}
    # Getting the type of 'csgraph' (line 75)
    csgraph_382054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 75)
    connected_components_382055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), csgraph_382054, 'connected_components')
    # Calling connected_components(args, kwargs) (line 75)
    connected_components_call_result_382062 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), connected_components_382055, *[X_382056], **kwargs_382061)
    
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___382063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), connected_components_call_result_382062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_382064 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), getitem___382063, int_382053)
    
    # Assigning a type to the variable 'tuple_var_assignment_381691' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_381691', subscript_call_result_382064)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_var_assignment_381690' (line 74)
    tuple_var_assignment_381690_382065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_381690')
    # Assigning a type to the variable 'n_components' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'n_components', tuple_var_assignment_381690_382065)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_var_assignment_381691' (line 74)
    tuple_var_assignment_381691_382066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_381691')
    # Assigning a type to the variable 'labels' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'labels', tuple_var_assignment_381691_382066)
    
    # Call to assert_equal(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'n_components' (line 77)
    n_components_382068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'n_components', False)
    int_382069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 31), 'int')
    # Processing the call keyword arguments (line 77)
    kwargs_382070 = {}
    # Getting the type of 'assert_equal' (line 77)
    assert_equal_382067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 77)
    assert_equal_call_result_382071 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), assert_equal_382067, *[n_components_382068, int_382069], **kwargs_382070)
    
    
    # Call to sort(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_382074 = {}
    # Getting the type of 'labels' (line 78)
    labels_382072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'labels', False)
    # Obtaining the member 'sort' of a type (line 78)
    sort_382073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 4), labels_382072, 'sort')
    # Calling sort(args, kwargs) (line 78)
    sort_call_result_382075 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), sort_382073, *[], **kwargs_382074)
    
    
    # Call to assert_array_almost_equal(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'labels' (line 79)
    labels_382077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'labels', False)
    
    # Obtaining an instance of the builtin type 'list' (line 79)
    list_382078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 79)
    # Adding element type (line 79)
    int_382079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 38), list_382078, int_382079)
    # Adding element type (line 79)
    int_382080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 38), list_382078, int_382080)
    # Adding element type (line 79)
    int_382081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 38), list_382078, int_382081)
    # Adding element type (line 79)
    int_382082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 38), list_382078, int_382082)
    # Adding element type (line 79)
    int_382083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 38), list_382078, int_382083)
    # Adding element type (line 79)
    int_382084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 38), list_382078, int_382084)
    
    # Processing the call keyword arguments (line 79)
    kwargs_382085 = {}
    # Getting the type of 'assert_array_almost_equal' (line 79)
    assert_array_almost_equal_382076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 79)
    assert_array_almost_equal_call_result_382086 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), assert_array_almost_equal_382076, *[labels_382077, list_382078], **kwargs_382085)
    
    
    # ################# End of 'test_weak_connections2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_weak_connections2' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_382087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382087)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_weak_connections2'
    return stypy_return_type_382087

# Assigning a type to the variable 'test_weak_connections2' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'test_weak_connections2', test_weak_connections2)

@norecursion
def test_ticket1876(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ticket1876'
    module_type_store = module_type_store.open_function_context('test_ticket1876', 82, 0, False)
    
    # Passed parameters checking function
    test_ticket1876.stypy_localization = localization
    test_ticket1876.stypy_type_of_self = None
    test_ticket1876.stypy_type_store = module_type_store
    test_ticket1876.stypy_function_name = 'test_ticket1876'
    test_ticket1876.stypy_param_names_list = []
    test_ticket1876.stypy_varargs_param_name = None
    test_ticket1876.stypy_kwargs_param_name = None
    test_ticket1876.stypy_call_defaults = defaults
    test_ticket1876.stypy_call_varargs = varargs
    test_ticket1876.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ticket1876', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ticket1876', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ticket1876(...)' code ##################

    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to array(...): (line 85)
    # Processing the call arguments (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_382090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_382091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    int_382092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), list_382091, int_382092)
    # Adding element type (line 85)
    int_382093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), list_382091, int_382093)
    # Adding element type (line 85)
    int_382094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), list_382091, int_382094)
    # Adding element type (line 85)
    int_382095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 18), list_382091, int_382095)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), list_382090, list_382091)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_382096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    # Adding element type (line 86)
    int_382097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), list_382096, int_382097)
    # Adding element type (line 86)
    int_382098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), list_382096, int_382098)
    # Adding element type (line 86)
    int_382099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), list_382096, int_382099)
    # Adding element type (line 86)
    int_382100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 18), list_382096, int_382100)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), list_382090, list_382096)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_382101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    # Adding element type (line 87)
    int_382102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_382101, int_382102)
    # Adding element type (line 87)
    int_382103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_382101, int_382103)
    # Adding element type (line 87)
    int_382104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_382101, int_382104)
    # Adding element type (line 87)
    int_382105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 18), list_382101, int_382105)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), list_382090, list_382101)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_382106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    # Adding element type (line 88)
    int_382107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 18), list_382106, int_382107)
    # Adding element type (line 88)
    int_382108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 18), list_382106, int_382108)
    # Adding element type (line 88)
    int_382109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 18), list_382106, int_382109)
    # Adding element type (line 88)
    int_382110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 18), list_382106, int_382110)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 17), list_382090, list_382106)
    
    # Processing the call keyword arguments (line 85)
    kwargs_382111 = {}
    # Getting the type of 'np' (line 85)
    np_382088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 85)
    array_382089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), np_382088, 'array')
    # Calling array(args, kwargs) (line 85)
    array_call_result_382112 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), array_382089, *[list_382090], **kwargs_382111)
    
    # Assigning a type to the variable 'g' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'g', array_call_result_382112)
    
    # Assigning a Call to a Tuple (line 89):
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_382113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to connected_components(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'g' (line 89)
    g_382116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 56), 'g', False)
    # Processing the call keyword arguments (line 89)
    str_382117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 70), 'str', 'strong')
    keyword_382118 = str_382117
    kwargs_382119 = {'connection': keyword_382118}
    # Getting the type of 'csgraph' (line 89)
    csgraph_382114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 89)
    connected_components_382115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), csgraph_382114, 'connected_components')
    # Calling connected_components(args, kwargs) (line 89)
    connected_components_call_result_382120 = invoke(stypy.reporting.localization.Localization(__file__, 89, 27), connected_components_382115, *[g_382116], **kwargs_382119)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___382121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), connected_components_call_result_382120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_382122 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___382121, int_382113)
    
    # Assigning a type to the variable 'tuple_var_assignment_381692' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_381692', subscript_call_result_382122)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    int_382123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'int')
    
    # Call to connected_components(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'g' (line 89)
    g_382126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 56), 'g', False)
    # Processing the call keyword arguments (line 89)
    str_382127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 70), 'str', 'strong')
    keyword_382128 = str_382127
    kwargs_382129 = {'connection': keyword_382128}
    # Getting the type of 'csgraph' (line 89)
    csgraph_382124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 89)
    connected_components_382125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), csgraph_382124, 'connected_components')
    # Calling connected_components(args, kwargs) (line 89)
    connected_components_call_result_382130 = invoke(stypy.reporting.localization.Localization(__file__, 89, 27), connected_components_382125, *[g_382126], **kwargs_382129)
    
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___382131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), connected_components_call_result_382130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_382132 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), getitem___382131, int_382123)
    
    # Assigning a type to the variable 'tuple_var_assignment_381693' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_381693', subscript_call_result_382132)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_381692' (line 89)
    tuple_var_assignment_381692_382133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_381692')
    # Assigning a type to the variable 'n_components' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'n_components', tuple_var_assignment_381692_382133)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_var_assignment_381693' (line 89)
    tuple_var_assignment_381693_382134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'tuple_var_assignment_381693')
    # Assigning a type to the variable 'labels' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'labels', tuple_var_assignment_381693_382134)
    
    # Call to assert_equal(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'n_components' (line 91)
    n_components_382136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 'n_components', False)
    int_382137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'int')
    # Processing the call keyword arguments (line 91)
    kwargs_382138 = {}
    # Getting the type of 'assert_equal' (line 91)
    assert_equal_382135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 91)
    assert_equal_call_result_382139 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), assert_equal_382135, *[n_components_382136, int_382137], **kwargs_382138)
    
    
    # Call to assert_equal(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Obtaining the type of the subscript
    int_382141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'int')
    # Getting the type of 'labels' (line 92)
    labels_382142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'labels', False)
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___382143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), labels_382142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_382144 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), getitem___382143, int_382141)
    
    
    # Obtaining the type of the subscript
    int_382145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 35), 'int')
    # Getting the type of 'labels' (line 92)
    labels_382146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'labels', False)
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___382147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), labels_382146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_382148 = invoke(stypy.reporting.localization.Localization(__file__, 92, 28), getitem___382147, int_382145)
    
    # Processing the call keyword arguments (line 92)
    kwargs_382149 = {}
    # Getting the type of 'assert_equal' (line 92)
    assert_equal_382140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 92)
    assert_equal_call_result_382150 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), assert_equal_382140, *[subscript_call_result_382144, subscript_call_result_382148], **kwargs_382149)
    
    
    # Call to assert_equal(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Obtaining the type of the subscript
    int_382152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 24), 'int')
    # Getting the type of 'labels' (line 93)
    labels_382153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'labels', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___382154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), labels_382153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_382155 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), getitem___382154, int_382152)
    
    
    # Obtaining the type of the subscript
    int_382156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 35), 'int')
    # Getting the type of 'labels' (line 93)
    labels_382157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'labels', False)
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___382158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), labels_382157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_382159 = invoke(stypy.reporting.localization.Localization(__file__, 93, 28), getitem___382158, int_382156)
    
    # Processing the call keyword arguments (line 93)
    kwargs_382160 = {}
    # Getting the type of 'assert_equal' (line 93)
    assert_equal_382151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 93)
    assert_equal_call_result_382161 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), assert_equal_382151, *[subscript_call_result_382155, subscript_call_result_382159], **kwargs_382160)
    
    
    # ################# End of 'test_ticket1876(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ticket1876' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_382162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382162)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ticket1876'
    return stypy_return_type_382162

# Assigning a type to the variable 'test_ticket1876' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'test_ticket1876', test_ticket1876)

@norecursion
def test_fully_connected_graph(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fully_connected_graph'
    module_type_store = module_type_store.open_function_context('test_fully_connected_graph', 96, 0, False)
    
    # Passed parameters checking function
    test_fully_connected_graph.stypy_localization = localization
    test_fully_connected_graph.stypy_type_of_self = None
    test_fully_connected_graph.stypy_type_store = module_type_store
    test_fully_connected_graph.stypy_function_name = 'test_fully_connected_graph'
    test_fully_connected_graph.stypy_param_names_list = []
    test_fully_connected_graph.stypy_varargs_param_name = None
    test_fully_connected_graph.stypy_kwargs_param_name = None
    test_fully_connected_graph.stypy_call_defaults = defaults
    test_fully_connected_graph.stypy_call_varargs = varargs
    test_fully_connected_graph.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fully_connected_graph', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fully_connected_graph', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fully_connected_graph(...)' code ##################

    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to ones(...): (line 99)
    # Processing the call arguments (line 99)
    
    # Obtaining an instance of the builtin type 'tuple' (line 99)
    tuple_382165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 99)
    # Adding element type (line 99)
    int_382166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 17), tuple_382165, int_382166)
    # Adding element type (line 99)
    int_382167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 17), tuple_382165, int_382167)
    
    # Processing the call keyword arguments (line 99)
    kwargs_382168 = {}
    # Getting the type of 'np' (line 99)
    np_382163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 99)
    ones_382164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), np_382163, 'ones')
    # Calling ones(args, kwargs) (line 99)
    ones_call_result_382169 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), ones_382164, *[tuple_382165], **kwargs_382168)
    
    # Assigning a type to the variable 'g' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'g', ones_call_result_382169)
    
    # Assigning a Call to a Tuple (line 100):
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_382170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'int')
    
    # Call to connected_components(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'g' (line 100)
    g_382173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 56), 'g', False)
    # Processing the call keyword arguments (line 100)
    kwargs_382174 = {}
    # Getting the type of 'csgraph' (line 100)
    csgraph_382171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 100)
    connected_components_382172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), csgraph_382171, 'connected_components')
    # Calling connected_components(args, kwargs) (line 100)
    connected_components_call_result_382175 = invoke(stypy.reporting.localization.Localization(__file__, 100, 27), connected_components_382172, *[g_382173], **kwargs_382174)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___382176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 4), connected_components_call_result_382175, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_382177 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), getitem___382176, int_382170)
    
    # Assigning a type to the variable 'tuple_var_assignment_381694' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_381694', subscript_call_result_382177)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_382178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'int')
    
    # Call to connected_components(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'g' (line 100)
    g_382181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 56), 'g', False)
    # Processing the call keyword arguments (line 100)
    kwargs_382182 = {}
    # Getting the type of 'csgraph' (line 100)
    csgraph_382179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'csgraph', False)
    # Obtaining the member 'connected_components' of a type (line 100)
    connected_components_382180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), csgraph_382179, 'connected_components')
    # Calling connected_components(args, kwargs) (line 100)
    connected_components_call_result_382183 = invoke(stypy.reporting.localization.Localization(__file__, 100, 27), connected_components_382180, *[g_382181], **kwargs_382182)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___382184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 4), connected_components_call_result_382183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_382185 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), getitem___382184, int_382178)
    
    # Assigning a type to the variable 'tuple_var_assignment_381695' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_381695', subscript_call_result_382185)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_381694' (line 100)
    tuple_var_assignment_381694_382186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_381694')
    # Assigning a type to the variable 'n_components' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'n_components', tuple_var_assignment_381694_382186)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_381695' (line 100)
    tuple_var_assignment_381695_382187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_381695')
    # Assigning a type to the variable 'labels' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'labels', tuple_var_assignment_381695_382187)
    
    # Call to assert_equal(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'n_components' (line 101)
    n_components_382189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'n_components', False)
    int_382190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 31), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_382191 = {}
    # Getting the type of 'assert_equal' (line 101)
    assert_equal_382188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 101)
    assert_equal_call_result_382192 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), assert_equal_382188, *[n_components_382189, int_382190], **kwargs_382191)
    
    
    # ################# End of 'test_fully_connected_graph(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fully_connected_graph' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_382193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382193)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fully_connected_graph'
    return stypy_return_type_382193

# Assigning a type to the variable 'test_fully_connected_graph' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'test_fully_connected_graph', test_fully_connected_graph)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
