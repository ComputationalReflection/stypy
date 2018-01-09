
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Sketching-based Matrix Computations '''
2: 
3: # Author: Jordi Montes <jomsdev@gmail.com>
4: # August 28, 2017
5: 
6: from __future__ import division, print_function, absolute_import
7: 
8: import numpy as np
9: 
10: from scipy._lib._util import check_random_state
11: 
12: __all__ = ['clarkson_woodruff_transform']
13: 
14: 
15: def cwt_matrix(n_rows, n_columns, seed=None):
16:     r'''"
17:     Generate a matrix S for the Clarkson-Woodruff sketch.
18: 
19:     Given the desired size of matrix, the method returns a matrix S of size
20:     (n_rows, n_columns) where each column has all the entries set to 0 less one
21:     position which has been randomly set to +1 or -1 with equal probability.
22: 
23:     Parameters
24:     ----------
25:     n_rows: int
26:         Number of rows of S
27:     n_columns: int
28:         Number of columns of S
29:     seed : None or int or `numpy.random.RandomState` instance, optional
30:         This parameter defines the ``RandomState`` object to use for drawing
31:         random variates.
32:         If None (or ``np.random``), the global ``np.random`` state is used.
33:         If integer, it is used to seed the local ``RandomState`` instance.
34:         Default is None.
35: 
36:     Returns
37:     -------
38:     S : (n_rows, n_columns) array_like
39: 
40:     Notes
41:     -----
42:     Given a matrix A, with probability at least 9/10,
43:     .. math:: ||SA|| == (1 \pm \epsilon)||A||
44:     Where epsilon is related to the size of S
45:     '''
46:     S = np.zeros((n_rows, n_columns))
47:     nz_positions = np.random.randint(0, n_rows, n_columns)
48:     rng = check_random_state(seed)
49:     values = rng.choice([1, -1], n_columns)
50:     for i in range(n_columns):
51:         S[nz_positions[i]][i] = values[i]
52: 
53:     return S
54: 
55: 
56: def clarkson_woodruff_transform(input_matrix, sketch_size, seed=None):
57:     r'''"
58:     Find low-rank matrix approximation via the Clarkson-Woodruff Transform.
59: 
60:     Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
61:     size (sketch_size, d) which holds:
62: 
63:     .. math:: ||Ax|| = (1 \pm \epsilon)||A'x||
64: 
65:     with high probability.
66: 
67:     The error is related to the number of rows of the sketch and it is bounded
68: 
69:     .. math:: poly(r(\epsilon^{-1}))
70: 
71:     Parameters
72:     ----------
73:     input_matrix: array_like
74:         Input matrix, of shape ``(n, d)``.
75:     sketch_size: int
76:         Number of rows for the sketch.
77:     seed : None or int or `numpy.random.RandomState` instance, optional
78:         This parameter defines the ``RandomState`` object to use for drawing
79:         random variates.
80:         If None (or ``np.random``), the global ``np.random`` state is used.
81:         If integer, it is used to seed the local ``RandomState`` instance.
82:         Default is None.
83: 
84:     Returns
85:     -------
86:     A' : array_like
87:         Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.
88: 
89:     Notes
90:     -----
91:     This is an implementation of the Clarkson-Woodruff Transform (CountSketch).
92:     ``A'`` can be computed in principle in ``O(nnz(A))`` (with ``nnz`` meaning
93:     the number of nonzero entries), however we don't take advantage of sparse
94:     matrices in this implementation.
95: 
96:     Examples
97:     --------
98:     Given a big dense matrix ``A``:
99: 
100:     >>> from scipy import linalg
101:     >>> n_rows, n_columns, sketch_n_rows = (2000, 100, 100)
102:     >>> threshold = 0.1
103:     >>> tmp = np.random.normal(0, 0.1, n_rows*n_columns)
104:     >>> A = np.reshape(tmp, (n_rows, n_columns))
105:     >>> sketch = linalg.clarkson_woodruff_transform(A, sketch_n_rows)
106:     >>> sketch.shape
107:     (100, 100)
108:     >>> normA = linalg.norm(A)
109:     >>> norm_sketch = linalg.norm(sketch)
110: 
111:     Now with high probability, the condition ``abs(normA-normSketch) <
112:     threshold`` holds.
113: 
114:     References
115:     ----------
116:     .. [1] Kenneth L. Clarkson and David P. Woodruff. Low rank approximation and
117:            regression in input sparsity time. In STOC, 2013.
118: 
119:     '''
120:     S = cwt_matrix(sketch_size, input_matrix.shape[0], seed)
121:     return np.dot(S, input_matrix)
122: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Sketching-based Matrix Computations ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35797 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_35797) is not StypyTypeError):

    if (import_35797 != 'pyd_module'):
        __import__(import_35797)
        sys_modules_35798 = sys.modules[import_35797]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_35798.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_35797)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._util import check_random_state' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35799 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util')

if (type(import_35799) is not StypyTypeError):

    if (import_35799 != 'pyd_module'):
        __import__(import_35799)
        sys_modules_35800 = sys.modules[import_35799]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', sys_modules_35800.module_type_store, module_type_store, ['check_random_state'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_35800, sys_modules_35800.module_type_store, module_type_store)
    else:
        from scipy._lib._util import check_random_state

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', None, module_type_store, ['check_random_state'], [check_random_state])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._util', import_35799)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 12):
__all__ = ['clarkson_woodruff_transform']
module_type_store.set_exportable_members(['clarkson_woodruff_transform'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_35801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_35802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'clarkson_woodruff_transform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_35801, str_35802)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_35801)

@norecursion
def cwt_matrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 15)
    None_35803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 39), 'None')
    defaults = [None_35803]
    # Create a new context for function 'cwt_matrix'
    module_type_store = module_type_store.open_function_context('cwt_matrix', 15, 0, False)
    
    # Passed parameters checking function
    cwt_matrix.stypy_localization = localization
    cwt_matrix.stypy_type_of_self = None
    cwt_matrix.stypy_type_store = module_type_store
    cwt_matrix.stypy_function_name = 'cwt_matrix'
    cwt_matrix.stypy_param_names_list = ['n_rows', 'n_columns', 'seed']
    cwt_matrix.stypy_varargs_param_name = None
    cwt_matrix.stypy_kwargs_param_name = None
    cwt_matrix.stypy_call_defaults = defaults
    cwt_matrix.stypy_call_varargs = varargs
    cwt_matrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cwt_matrix', ['n_rows', 'n_columns', 'seed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cwt_matrix', localization, ['n_rows', 'n_columns', 'seed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cwt_matrix(...)' code ##################

    str_35804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', '"\n    Generate a matrix S for the Clarkson-Woodruff sketch.\n\n    Given the desired size of matrix, the method returns a matrix S of size\n    (n_rows, n_columns) where each column has all the entries set to 0 less one\n    position which has been randomly set to +1 or -1 with equal probability.\n\n    Parameters\n    ----------\n    n_rows: int\n        Number of rows of S\n    n_columns: int\n        Number of columns of S\n    seed : None or int or `numpy.random.RandomState` instance, optional\n        This parameter defines the ``RandomState`` object to use for drawing\n        random variates.\n        If None (or ``np.random``), the global ``np.random`` state is used.\n        If integer, it is used to seed the local ``RandomState`` instance.\n        Default is None.\n\n    Returns\n    -------\n    S : (n_rows, n_columns) array_like\n\n    Notes\n    -----\n    Given a matrix A, with probability at least 9/10,\n    .. math:: ||SA|| == (1 \\pm \\epsilon)||A||\n    Where epsilon is related to the size of S\n    ')
    
    # Assigning a Call to a Name (line 46):
    
    # Call to zeros(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_35807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'n_rows' (line 46)
    n_rows_35808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'n_rows', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 18), tuple_35807, n_rows_35808)
    # Adding element type (line 46)
    # Getting the type of 'n_columns' (line 46)
    n_columns_35809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'n_columns', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 18), tuple_35807, n_columns_35809)
    
    # Processing the call keyword arguments (line 46)
    kwargs_35810 = {}
    # Getting the type of 'np' (line 46)
    np_35805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 46)
    zeros_35806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), np_35805, 'zeros')
    # Calling zeros(args, kwargs) (line 46)
    zeros_call_result_35811 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), zeros_35806, *[tuple_35807], **kwargs_35810)
    
    # Assigning a type to the variable 'S' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'S', zeros_call_result_35811)
    
    # Assigning a Call to a Name (line 47):
    
    # Call to randint(...): (line 47)
    # Processing the call arguments (line 47)
    int_35815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'int')
    # Getting the type of 'n_rows' (line 47)
    n_rows_35816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'n_rows', False)
    # Getting the type of 'n_columns' (line 47)
    n_columns_35817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 48), 'n_columns', False)
    # Processing the call keyword arguments (line 47)
    kwargs_35818 = {}
    # Getting the type of 'np' (line 47)
    np_35812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'np', False)
    # Obtaining the member 'random' of a type (line 47)
    random_35813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), np_35812, 'random')
    # Obtaining the member 'randint' of a type (line 47)
    randint_35814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), random_35813, 'randint')
    # Calling randint(args, kwargs) (line 47)
    randint_call_result_35819 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), randint_35814, *[int_35815, n_rows_35816, n_columns_35817], **kwargs_35818)
    
    # Assigning a type to the variable 'nz_positions' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'nz_positions', randint_call_result_35819)
    
    # Assigning a Call to a Name (line 48):
    
    # Call to check_random_state(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'seed' (line 48)
    seed_35821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'seed', False)
    # Processing the call keyword arguments (line 48)
    kwargs_35822 = {}
    # Getting the type of 'check_random_state' (line 48)
    check_random_state_35820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'check_random_state', False)
    # Calling check_random_state(args, kwargs) (line 48)
    check_random_state_call_result_35823 = invoke(stypy.reporting.localization.Localization(__file__, 48, 10), check_random_state_35820, *[seed_35821], **kwargs_35822)
    
    # Assigning a type to the variable 'rng' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'rng', check_random_state_call_result_35823)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to choice(...): (line 49)
    # Processing the call arguments (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_35826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    int_35827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), list_35826, int_35827)
    # Adding element type (line 49)
    int_35828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), list_35826, int_35828)
    
    # Getting the type of 'n_columns' (line 49)
    n_columns_35829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'n_columns', False)
    # Processing the call keyword arguments (line 49)
    kwargs_35830 = {}
    # Getting the type of 'rng' (line 49)
    rng_35824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'rng', False)
    # Obtaining the member 'choice' of a type (line 49)
    choice_35825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 13), rng_35824, 'choice')
    # Calling choice(args, kwargs) (line 49)
    choice_call_result_35831 = invoke(stypy.reporting.localization.Localization(__file__, 49, 13), choice_35825, *[list_35826, n_columns_35829], **kwargs_35830)
    
    # Assigning a type to the variable 'values' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'values', choice_call_result_35831)
    
    
    # Call to range(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'n_columns' (line 50)
    n_columns_35833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'n_columns', False)
    # Processing the call keyword arguments (line 50)
    kwargs_35834 = {}
    # Getting the type of 'range' (line 50)
    range_35832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'range', False)
    # Calling range(args, kwargs) (line 50)
    range_call_result_35835 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), range_35832, *[n_columns_35833], **kwargs_35834)
    
    # Testing the type of a for loop iterable (line 50)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 4), range_call_result_35835)
    # Getting the type of the for loop variable (line 50)
    for_loop_var_35836 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 4), range_call_result_35835)
    # Assigning a type to the variable 'i' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'i', for_loop_var_35836)
    # SSA begins for a for statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 51):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 51)
    i_35837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'i')
    # Getting the type of 'values' (line 51)
    values_35838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'values')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___35839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 32), values_35838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_35840 = invoke(stypy.reporting.localization.Localization(__file__, 51, 32), getitem___35839, i_35837)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 51)
    i_35841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'i')
    # Getting the type of 'nz_positions' (line 51)
    nz_positions_35842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'nz_positions')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___35843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 10), nz_positions_35842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_35844 = invoke(stypy.reporting.localization.Localization(__file__, 51, 10), getitem___35843, i_35841)
    
    # Getting the type of 'S' (line 51)
    S_35845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'S')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___35846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), S_35845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_35847 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), getitem___35846, subscript_call_result_35844)
    
    # Getting the type of 'i' (line 51)
    i_35848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'i')
    # Storing an element on a container (line 51)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), subscript_call_result_35847, (i_35848, subscript_call_result_35840))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'S' (line 53)
    S_35849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'S')
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', S_35849)
    
    # ################# End of 'cwt_matrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cwt_matrix' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_35850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35850)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cwt_matrix'
    return stypy_return_type_35850

# Assigning a type to the variable 'cwt_matrix' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'cwt_matrix', cwt_matrix)

@norecursion
def clarkson_woodruff_transform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 56)
    None_35851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 64), 'None')
    defaults = [None_35851]
    # Create a new context for function 'clarkson_woodruff_transform'
    module_type_store = module_type_store.open_function_context('clarkson_woodruff_transform', 56, 0, False)
    
    # Passed parameters checking function
    clarkson_woodruff_transform.stypy_localization = localization
    clarkson_woodruff_transform.stypy_type_of_self = None
    clarkson_woodruff_transform.stypy_type_store = module_type_store
    clarkson_woodruff_transform.stypy_function_name = 'clarkson_woodruff_transform'
    clarkson_woodruff_transform.stypy_param_names_list = ['input_matrix', 'sketch_size', 'seed']
    clarkson_woodruff_transform.stypy_varargs_param_name = None
    clarkson_woodruff_transform.stypy_kwargs_param_name = None
    clarkson_woodruff_transform.stypy_call_defaults = defaults
    clarkson_woodruff_transform.stypy_call_varargs = varargs
    clarkson_woodruff_transform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clarkson_woodruff_transform', ['input_matrix', 'sketch_size', 'seed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clarkson_woodruff_transform', localization, ['input_matrix', 'sketch_size', 'seed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clarkson_woodruff_transform(...)' code ##################

    str_35852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, (-1)), 'str', '"\n    Find low-rank matrix approximation via the Clarkson-Woodruff Transform.\n\n    Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A\'`` of\n    size (sketch_size, d) which holds:\n\n    .. math:: ||Ax|| = (1 \\pm \\epsilon)||A\'x||\n\n    with high probability.\n\n    The error is related to the number of rows of the sketch and it is bounded\n\n    .. math:: poly(r(\\epsilon^{-1}))\n\n    Parameters\n    ----------\n    input_matrix: array_like\n        Input matrix, of shape ``(n, d)``.\n    sketch_size: int\n        Number of rows for the sketch.\n    seed : None or int or `numpy.random.RandomState` instance, optional\n        This parameter defines the ``RandomState`` object to use for drawing\n        random variates.\n        If None (or ``np.random``), the global ``np.random`` state is used.\n        If integer, it is used to seed the local ``RandomState`` instance.\n        Default is None.\n\n    Returns\n    -------\n    A\' : array_like\n        Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.\n\n    Notes\n    -----\n    This is an implementation of the Clarkson-Woodruff Transform (CountSketch).\n    ``A\'`` can be computed in principle in ``O(nnz(A))`` (with ``nnz`` meaning\n    the number of nonzero entries), however we don\'t take advantage of sparse\n    matrices in this implementation.\n\n    Examples\n    --------\n    Given a big dense matrix ``A``:\n\n    >>> from scipy import linalg\n    >>> n_rows, n_columns, sketch_n_rows = (2000, 100, 100)\n    >>> threshold = 0.1\n    >>> tmp = np.random.normal(0, 0.1, n_rows*n_columns)\n    >>> A = np.reshape(tmp, (n_rows, n_columns))\n    >>> sketch = linalg.clarkson_woodruff_transform(A, sketch_n_rows)\n    >>> sketch.shape\n    (100, 100)\n    >>> normA = linalg.norm(A)\n    >>> norm_sketch = linalg.norm(sketch)\n\n    Now with high probability, the condition ``abs(normA-normSketch) <\n    threshold`` holds.\n\n    References\n    ----------\n    .. [1] Kenneth L. Clarkson and David P. Woodruff. Low rank approximation and\n           regression in input sparsity time. In STOC, 2013.\n\n    ')
    
    # Assigning a Call to a Name (line 120):
    
    # Call to cwt_matrix(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'sketch_size' (line 120)
    sketch_size_35854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'sketch_size', False)
    
    # Obtaining the type of the subscript
    int_35855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 51), 'int')
    # Getting the type of 'input_matrix' (line 120)
    input_matrix_35856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'input_matrix', False)
    # Obtaining the member 'shape' of a type (line 120)
    shape_35857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), input_matrix_35856, 'shape')
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___35858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), shape_35857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_35859 = invoke(stypy.reporting.localization.Localization(__file__, 120, 32), getitem___35858, int_35855)
    
    # Getting the type of 'seed' (line 120)
    seed_35860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 55), 'seed', False)
    # Processing the call keyword arguments (line 120)
    kwargs_35861 = {}
    # Getting the type of 'cwt_matrix' (line 120)
    cwt_matrix_35853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'cwt_matrix', False)
    # Calling cwt_matrix(args, kwargs) (line 120)
    cwt_matrix_call_result_35862 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), cwt_matrix_35853, *[sketch_size_35854, subscript_call_result_35859, seed_35860], **kwargs_35861)
    
    # Assigning a type to the variable 'S' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'S', cwt_matrix_call_result_35862)
    
    # Call to dot(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'S' (line 121)
    S_35865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'S', False)
    # Getting the type of 'input_matrix' (line 121)
    input_matrix_35866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'input_matrix', False)
    # Processing the call keyword arguments (line 121)
    kwargs_35867 = {}
    # Getting the type of 'np' (line 121)
    np_35863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'np', False)
    # Obtaining the member 'dot' of a type (line 121)
    dot_35864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), np_35863, 'dot')
    # Calling dot(args, kwargs) (line 121)
    dot_call_result_35868 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), dot_35864, *[S_35865, input_matrix_35866], **kwargs_35867)
    
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type', dot_call_result_35868)
    
    # ################# End of 'clarkson_woodruff_transform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clarkson_woodruff_transform' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_35869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35869)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clarkson_woodruff_transform'
    return stypy_return_type_35869

# Assigning a type to the variable 'clarkson_woodruff_transform' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'clarkson_woodruff_transform', clarkson_woodruff_transform)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
