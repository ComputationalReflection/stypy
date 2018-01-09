
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_array_almost_equal
5: from scipy.sparse import csr_matrix
6: from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense
7: 
8: 
9: def test_csgraph_from_dense():
10:     np.random.seed(1234)
11:     G = np.random.random((10, 10))
12:     some_nulls = (G < 0.4)
13:     all_nulls = (G < 0.8)
14: 
15:     for null_value in [0, np.nan, np.inf]:
16:         G[all_nulls] = null_value
17:         olderr = np.seterr(invalid="ignore")
18:         try:
19:             G_csr = csgraph_from_dense(G, null_value=0)
20:         finally:
21:             np.seterr(**olderr)
22: 
23:         G[all_nulls] = 0
24:         assert_array_almost_equal(G, G_csr.toarray())
25: 
26:     for null_value in [np.nan, np.inf]:
27:         G[all_nulls] = 0
28:         G[some_nulls] = null_value
29:         olderr = np.seterr(invalid="ignore")
30:         try:
31:             G_csr = csgraph_from_dense(G, null_value=0)
32:         finally:
33:             np.seterr(**olderr)
34: 
35:         G[all_nulls] = 0
36:         assert_array_almost_equal(G, G_csr.toarray())
37: 
38: 
39: def test_csgraph_to_dense():
40:     np.random.seed(1234)
41:     G = np.random.random((10, 10))
42:     nulls = (G < 0.8)
43:     G[nulls] = np.inf
44: 
45:     G_csr = csgraph_from_dense(G)
46: 
47:     for null_value in [0, 10, -np.inf, np.inf]:
48:         G[nulls] = null_value
49:         assert_array_almost_equal(G, csgraph_to_dense(G_csr, null_value))
50: 
51: 
52: def test_multiple_edges():
53:     # create a random sqare matrix with an even number of elements
54:     np.random.seed(1234)
55:     X = np.random.random((10, 10))
56:     Xcsr = csr_matrix(X)
57: 
58:     # now double-up every other column
59:     Xcsr.indices[::2] = Xcsr.indices[1::2]
60: 
61:     # normal sparse toarray() will sum the duplicated edges
62:     Xdense = Xcsr.toarray()
63:     assert_array_almost_equal(Xdense[:, 1::2],
64:                               X[:, ::2] + X[:, 1::2])
65: 
66:     # csgraph_to_dense chooses the minimum of each duplicated edge
67:     Xdense = csgraph_to_dense(Xcsr)
68:     assert_array_almost_equal(Xdense[:, 1::2],
69:                               np.minimum(X[:, ::2], X[:, 1::2]))
70: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382194 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_382194) is not StypyTypeError):

    if (import_382194 != 'pyd_module'):
        __import__(import_382194)
        sys_modules_382195 = sys.modules[import_382194]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_382195.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_382194)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_almost_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382196 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_382196) is not StypyTypeError):

    if (import_382196 != 'pyd_module'):
        __import__(import_382196)
        sys_modules_382197 = sys.modules[import_382196]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_382197.module_type_store, module_type_store, ['assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_382197, sys_modules_382197.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal'], [assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_382196)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.sparse import csr_matrix' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382198 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse')

if (type(import_382198) is not StypyTypeError):

    if (import_382198 != 'pyd_module'):
        __import__(import_382198)
        sys_modules_382199 = sys.modules[import_382198]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', sys_modules_382199.module_type_store, module_type_store, ['csr_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_382199, sys_modules_382199.module_type_store, module_type_store)
    else:
        from scipy.sparse import csr_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', None, module_type_store, ['csr_matrix'], [csr_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.sparse', import_382198)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')
import_382200 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph')

if (type(import_382200) is not StypyTypeError):

    if (import_382200 != 'pyd_module'):
        __import__(import_382200)
        sys_modules_382201 = sys.modules[import_382200]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph', sys_modules_382201.module_type_store, module_type_store, ['csgraph_from_dense', 'csgraph_to_dense'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_382201, sys_modules_382201.module_type_store, module_type_store)
    else:
        from scipy.sparse.csgraph import csgraph_from_dense, csgraph_to_dense

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph', None, module_type_store, ['csgraph_from_dense', 'csgraph_to_dense'], [csgraph_from_dense, csgraph_to_dense])

else:
    # Assigning a type to the variable 'scipy.sparse.csgraph' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.sparse.csgraph', import_382200)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/csgraph/tests/')


@norecursion
def test_csgraph_from_dense(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csgraph_from_dense'
    module_type_store = module_type_store.open_function_context('test_csgraph_from_dense', 9, 0, False)
    
    # Passed parameters checking function
    test_csgraph_from_dense.stypy_localization = localization
    test_csgraph_from_dense.stypy_type_of_self = None
    test_csgraph_from_dense.stypy_type_store = module_type_store
    test_csgraph_from_dense.stypy_function_name = 'test_csgraph_from_dense'
    test_csgraph_from_dense.stypy_param_names_list = []
    test_csgraph_from_dense.stypy_varargs_param_name = None
    test_csgraph_from_dense.stypy_kwargs_param_name = None
    test_csgraph_from_dense.stypy_call_defaults = defaults
    test_csgraph_from_dense.stypy_call_varargs = varargs
    test_csgraph_from_dense.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csgraph_from_dense', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csgraph_from_dense', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csgraph_from_dense(...)' code ##################

    
    # Call to seed(...): (line 10)
    # Processing the call arguments (line 10)
    int_382205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
    # Processing the call keyword arguments (line 10)
    kwargs_382206 = {}
    # Getting the type of 'np' (line 10)
    np_382202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 10)
    random_382203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), np_382202, 'random')
    # Obtaining the member 'seed' of a type (line 10)
    seed_382204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), random_382203, 'seed')
    # Calling seed(args, kwargs) (line 10)
    seed_call_result_382207 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), seed_382204, *[int_382205], **kwargs_382206)
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to random(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining an instance of the builtin type 'tuple' (line 11)
    tuple_382211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 11)
    # Adding element type (line 11)
    int_382212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 26), tuple_382211, int_382212)
    # Adding element type (line 11)
    int_382213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 26), tuple_382211, int_382213)
    
    # Processing the call keyword arguments (line 11)
    kwargs_382214 = {}
    # Getting the type of 'np' (line 11)
    np_382208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 11)
    random_382209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), np_382208, 'random')
    # Obtaining the member 'random' of a type (line 11)
    random_382210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), random_382209, 'random')
    # Calling random(args, kwargs) (line 11)
    random_call_result_382215 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), random_382210, *[tuple_382211], **kwargs_382214)
    
    # Assigning a type to the variable 'G' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'G', random_call_result_382215)
    
    # Assigning a Compare to a Name (line 12):
    
    # Getting the type of 'G' (line 12)
    G_382216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'G')
    float_382217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'float')
    # Applying the binary operator '<' (line 12)
    result_lt_382218 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 18), '<', G_382216, float_382217)
    
    # Assigning a type to the variable 'some_nulls' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'some_nulls', result_lt_382218)
    
    # Assigning a Compare to a Name (line 13):
    
    # Getting the type of 'G' (line 13)
    G_382219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'G')
    float_382220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'float')
    # Applying the binary operator '<' (line 13)
    result_lt_382221 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 17), '<', G_382219, float_382220)
    
    # Assigning a type to the variable 'all_nulls' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'all_nulls', result_lt_382221)
    
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_382222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_382223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_382222, int_382223)
    # Adding element type (line 15)
    # Getting the type of 'np' (line 15)
    np_382224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 26), 'np')
    # Obtaining the member 'nan' of a type (line 15)
    nan_382225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 26), np_382224, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_382222, nan_382225)
    # Adding element type (line 15)
    # Getting the type of 'np' (line 15)
    np_382226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'np')
    # Obtaining the member 'inf' of a type (line 15)
    inf_382227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 34), np_382226, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 22), list_382222, inf_382227)
    
    # Testing the type of a for loop iterable (line 15)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 4), list_382222)
    # Getting the type of the for loop variable (line 15)
    for_loop_var_382228 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 4), list_382222)
    # Assigning a type to the variable 'null_value' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'null_value', for_loop_var_382228)
    # SSA begins for a for statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 16):
    # Getting the type of 'null_value' (line 16)
    null_value_382229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'null_value')
    # Getting the type of 'G' (line 16)
    G_382230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'G')
    # Getting the type of 'all_nulls' (line 16)
    all_nulls_382231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'all_nulls')
    # Storing an element on a container (line 16)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 8), G_382230, (all_nulls_382231, null_value_382229))
    
    # Assigning a Call to a Name (line 17):
    
    # Call to seterr(...): (line 17)
    # Processing the call keyword arguments (line 17)
    str_382234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'str', 'ignore')
    keyword_382235 = str_382234
    kwargs_382236 = {'invalid': keyword_382235}
    # Getting the type of 'np' (line 17)
    np_382232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'np', False)
    # Obtaining the member 'seterr' of a type (line 17)
    seterr_382233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 17), np_382232, 'seterr')
    # Calling seterr(args, kwargs) (line 17)
    seterr_call_result_382237 = invoke(stypy.reporting.localization.Localization(__file__, 17, 17), seterr_382233, *[], **kwargs_382236)
    
    # Assigning a type to the variable 'olderr' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'olderr', seterr_call_result_382237)
    
    # Try-finally block (line 18)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to csgraph_from_dense(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'G' (line 19)
    G_382239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 39), 'G', False)
    # Processing the call keyword arguments (line 19)
    int_382240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 53), 'int')
    keyword_382241 = int_382240
    kwargs_382242 = {'null_value': keyword_382241}
    # Getting the type of 'csgraph_from_dense' (line 19)
    csgraph_from_dense_382238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 19)
    csgraph_from_dense_call_result_382243 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), csgraph_from_dense_382238, *[G_382239], **kwargs_382242)
    
    # Assigning a type to the variable 'G_csr' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'G_csr', csgraph_from_dense_call_result_382243)
    
    # finally branch of the try-finally block (line 18)
    
    # Call to seterr(...): (line 21)
    # Processing the call keyword arguments (line 21)
    # Getting the type of 'olderr' (line 21)
    olderr_382246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'olderr', False)
    kwargs_382247 = {'olderr_382246': olderr_382246}
    # Getting the type of 'np' (line 21)
    np_382244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'np', False)
    # Obtaining the member 'seterr' of a type (line 21)
    seterr_382245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), np_382244, 'seterr')
    # Calling seterr(args, kwargs) (line 21)
    seterr_call_result_382248 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), seterr_382245, *[], **kwargs_382247)
    
    
    
    # Assigning a Num to a Subscript (line 23):
    int_382249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'int')
    # Getting the type of 'G' (line 23)
    G_382250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'G')
    # Getting the type of 'all_nulls' (line 23)
    all_nulls_382251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'all_nulls')
    # Storing an element on a container (line 23)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), G_382250, (all_nulls_382251, int_382249))
    
    # Call to assert_array_almost_equal(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'G' (line 24)
    G_382253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'G', False)
    
    # Call to toarray(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_382256 = {}
    # Getting the type of 'G_csr' (line 24)
    G_csr_382254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'G_csr', False)
    # Obtaining the member 'toarray' of a type (line 24)
    toarray_382255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 37), G_csr_382254, 'toarray')
    # Calling toarray(args, kwargs) (line 24)
    toarray_call_result_382257 = invoke(stypy.reporting.localization.Localization(__file__, 24, 37), toarray_382255, *[], **kwargs_382256)
    
    # Processing the call keyword arguments (line 24)
    kwargs_382258 = {}
    # Getting the type of 'assert_array_almost_equal' (line 24)
    assert_array_almost_equal_382252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 24)
    assert_array_almost_equal_call_result_382259 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), assert_array_almost_equal_382252, *[G_382253, toarray_call_result_382257], **kwargs_382258)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_382260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'np' (line 26)
    np_382261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'np')
    # Obtaining the member 'nan' of a type (line 26)
    nan_382262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), np_382261, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_382260, nan_382262)
    # Adding element type (line 26)
    # Getting the type of 'np' (line 26)
    np_382263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'np')
    # Obtaining the member 'inf' of a type (line 26)
    inf_382264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 31), np_382263, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_382260, inf_382264)
    
    # Testing the type of a for loop iterable (line 26)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 4), list_382260)
    # Getting the type of the for loop variable (line 26)
    for_loop_var_382265 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 4), list_382260)
    # Assigning a type to the variable 'null_value' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'null_value', for_loop_var_382265)
    # SSA begins for a for statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Subscript (line 27):
    int_382266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'int')
    # Getting the type of 'G' (line 27)
    G_382267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'G')
    # Getting the type of 'all_nulls' (line 27)
    all_nulls_382268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'all_nulls')
    # Storing an element on a container (line 27)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 8), G_382267, (all_nulls_382268, int_382266))
    
    # Assigning a Name to a Subscript (line 28):
    # Getting the type of 'null_value' (line 28)
    null_value_382269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'null_value')
    # Getting the type of 'G' (line 28)
    G_382270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'G')
    # Getting the type of 'some_nulls' (line 28)
    some_nulls_382271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'some_nulls')
    # Storing an element on a container (line 28)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), G_382270, (some_nulls_382271, null_value_382269))
    
    # Assigning a Call to a Name (line 29):
    
    # Call to seterr(...): (line 29)
    # Processing the call keyword arguments (line 29)
    str_382274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'str', 'ignore')
    keyword_382275 = str_382274
    kwargs_382276 = {'invalid': keyword_382275}
    # Getting the type of 'np' (line 29)
    np_382272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'np', False)
    # Obtaining the member 'seterr' of a type (line 29)
    seterr_382273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 17), np_382272, 'seterr')
    # Calling seterr(args, kwargs) (line 29)
    seterr_call_result_382277 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), seterr_382273, *[], **kwargs_382276)
    
    # Assigning a type to the variable 'olderr' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'olderr', seterr_call_result_382277)
    
    # Try-finally block (line 30)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to csgraph_from_dense(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'G' (line 31)
    G_382279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 39), 'G', False)
    # Processing the call keyword arguments (line 31)
    int_382280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 53), 'int')
    keyword_382281 = int_382280
    kwargs_382282 = {'null_value': keyword_382281}
    # Getting the type of 'csgraph_from_dense' (line 31)
    csgraph_from_dense_382278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 31)
    csgraph_from_dense_call_result_382283 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), csgraph_from_dense_382278, *[G_382279], **kwargs_382282)
    
    # Assigning a type to the variable 'G_csr' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'G_csr', csgraph_from_dense_call_result_382283)
    
    # finally branch of the try-finally block (line 30)
    
    # Call to seterr(...): (line 33)
    # Processing the call keyword arguments (line 33)
    # Getting the type of 'olderr' (line 33)
    olderr_382286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'olderr', False)
    kwargs_382287 = {'olderr_382286': olderr_382286}
    # Getting the type of 'np' (line 33)
    np_382284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'np', False)
    # Obtaining the member 'seterr' of a type (line 33)
    seterr_382285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), np_382284, 'seterr')
    # Calling seterr(args, kwargs) (line 33)
    seterr_call_result_382288 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), seterr_382285, *[], **kwargs_382287)
    
    
    
    # Assigning a Num to a Subscript (line 35):
    int_382289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'int')
    # Getting the type of 'G' (line 35)
    G_382290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'G')
    # Getting the type of 'all_nulls' (line 35)
    all_nulls_382291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 10), 'all_nulls')
    # Storing an element on a container (line 35)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), G_382290, (all_nulls_382291, int_382289))
    
    # Call to assert_array_almost_equal(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'G' (line 36)
    G_382293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'G', False)
    
    # Call to toarray(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_382296 = {}
    # Getting the type of 'G_csr' (line 36)
    G_csr_382294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'G_csr', False)
    # Obtaining the member 'toarray' of a type (line 36)
    toarray_382295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 37), G_csr_382294, 'toarray')
    # Calling toarray(args, kwargs) (line 36)
    toarray_call_result_382297 = invoke(stypy.reporting.localization.Localization(__file__, 36, 37), toarray_382295, *[], **kwargs_382296)
    
    # Processing the call keyword arguments (line 36)
    kwargs_382298 = {}
    # Getting the type of 'assert_array_almost_equal' (line 36)
    assert_array_almost_equal_382292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 36)
    assert_array_almost_equal_call_result_382299 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_array_almost_equal_382292, *[G_382293, toarray_call_result_382297], **kwargs_382298)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_csgraph_from_dense(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csgraph_from_dense' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_382300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382300)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csgraph_from_dense'
    return stypy_return_type_382300

# Assigning a type to the variable 'test_csgraph_from_dense' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_csgraph_from_dense', test_csgraph_from_dense)

@norecursion
def test_csgraph_to_dense(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_csgraph_to_dense'
    module_type_store = module_type_store.open_function_context('test_csgraph_to_dense', 39, 0, False)
    
    # Passed parameters checking function
    test_csgraph_to_dense.stypy_localization = localization
    test_csgraph_to_dense.stypy_type_of_self = None
    test_csgraph_to_dense.stypy_type_store = module_type_store
    test_csgraph_to_dense.stypy_function_name = 'test_csgraph_to_dense'
    test_csgraph_to_dense.stypy_param_names_list = []
    test_csgraph_to_dense.stypy_varargs_param_name = None
    test_csgraph_to_dense.stypy_kwargs_param_name = None
    test_csgraph_to_dense.stypy_call_defaults = defaults
    test_csgraph_to_dense.stypy_call_varargs = varargs
    test_csgraph_to_dense.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_csgraph_to_dense', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_csgraph_to_dense', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_csgraph_to_dense(...)' code ##################

    
    # Call to seed(...): (line 40)
    # Processing the call arguments (line 40)
    int_382304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'int')
    # Processing the call keyword arguments (line 40)
    kwargs_382305 = {}
    # Getting the type of 'np' (line 40)
    np_382301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 40)
    random_382302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), np_382301, 'random')
    # Obtaining the member 'seed' of a type (line 40)
    seed_382303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), random_382302, 'seed')
    # Calling seed(args, kwargs) (line 40)
    seed_call_result_382306 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), seed_382303, *[int_382304], **kwargs_382305)
    
    
    # Assigning a Call to a Name (line 41):
    
    # Call to random(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_382310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    int_382311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), tuple_382310, int_382311)
    # Adding element type (line 41)
    int_382312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), tuple_382310, int_382312)
    
    # Processing the call keyword arguments (line 41)
    kwargs_382313 = {}
    # Getting the type of 'np' (line 41)
    np_382307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 41)
    random_382308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), np_382307, 'random')
    # Obtaining the member 'random' of a type (line 41)
    random_382309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), random_382308, 'random')
    # Calling random(args, kwargs) (line 41)
    random_call_result_382314 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), random_382309, *[tuple_382310], **kwargs_382313)
    
    # Assigning a type to the variable 'G' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'G', random_call_result_382314)
    
    # Assigning a Compare to a Name (line 42):
    
    # Getting the type of 'G' (line 42)
    G_382315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'G')
    float_382316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'float')
    # Applying the binary operator '<' (line 42)
    result_lt_382317 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 13), '<', G_382315, float_382316)
    
    # Assigning a type to the variable 'nulls' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'nulls', result_lt_382317)
    
    # Assigning a Attribute to a Subscript (line 43):
    # Getting the type of 'np' (line 43)
    np_382318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'np')
    # Obtaining the member 'inf' of a type (line 43)
    inf_382319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 15), np_382318, 'inf')
    # Getting the type of 'G' (line 43)
    G_382320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'G')
    # Getting the type of 'nulls' (line 43)
    nulls_382321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 6), 'nulls')
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), G_382320, (nulls_382321, inf_382319))
    
    # Assigning a Call to a Name (line 45):
    
    # Call to csgraph_from_dense(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'G' (line 45)
    G_382323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'G', False)
    # Processing the call keyword arguments (line 45)
    kwargs_382324 = {}
    # Getting the type of 'csgraph_from_dense' (line 45)
    csgraph_from_dense_382322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'csgraph_from_dense', False)
    # Calling csgraph_from_dense(args, kwargs) (line 45)
    csgraph_from_dense_call_result_382325 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), csgraph_from_dense_382322, *[G_382323], **kwargs_382324)
    
    # Assigning a type to the variable 'G_csr' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'G_csr', csgraph_from_dense_call_result_382325)
    
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_382326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    int_382327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 22), list_382326, int_382327)
    # Adding element type (line 47)
    int_382328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 22), list_382326, int_382328)
    # Adding element type (line 47)
    
    # Getting the type of 'np' (line 47)
    np_382329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'np')
    # Obtaining the member 'inf' of a type (line 47)
    inf_382330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 31), np_382329, 'inf')
    # Applying the 'usub' unary operator (line 47)
    result___neg___382331 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 30), 'usub', inf_382330)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 22), list_382326, result___neg___382331)
    # Adding element type (line 47)
    # Getting the type of 'np' (line 47)
    np_382332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 39), 'np')
    # Obtaining the member 'inf' of a type (line 47)
    inf_382333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 39), np_382332, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 22), list_382326, inf_382333)
    
    # Testing the type of a for loop iterable (line 47)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 4), list_382326)
    # Getting the type of the for loop variable (line 47)
    for_loop_var_382334 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 4), list_382326)
    # Assigning a type to the variable 'null_value' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'null_value', for_loop_var_382334)
    # SSA begins for a for statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 48):
    # Getting the type of 'null_value' (line 48)
    null_value_382335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'null_value')
    # Getting the type of 'G' (line 48)
    G_382336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'G')
    # Getting the type of 'nulls' (line 48)
    nulls_382337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'nulls')
    # Storing an element on a container (line 48)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 8), G_382336, (nulls_382337, null_value_382335))
    
    # Call to assert_array_almost_equal(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'G' (line 49)
    G_382339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'G', False)
    
    # Call to csgraph_to_dense(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'G_csr' (line 49)
    G_csr_382341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 54), 'G_csr', False)
    # Getting the type of 'null_value' (line 49)
    null_value_382342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 61), 'null_value', False)
    # Processing the call keyword arguments (line 49)
    kwargs_382343 = {}
    # Getting the type of 'csgraph_to_dense' (line 49)
    csgraph_to_dense_382340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 37), 'csgraph_to_dense', False)
    # Calling csgraph_to_dense(args, kwargs) (line 49)
    csgraph_to_dense_call_result_382344 = invoke(stypy.reporting.localization.Localization(__file__, 49, 37), csgraph_to_dense_382340, *[G_csr_382341, null_value_382342], **kwargs_382343)
    
    # Processing the call keyword arguments (line 49)
    kwargs_382345 = {}
    # Getting the type of 'assert_array_almost_equal' (line 49)
    assert_array_almost_equal_382338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 49)
    assert_array_almost_equal_call_result_382346 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assert_array_almost_equal_382338, *[G_382339, csgraph_to_dense_call_result_382344], **kwargs_382345)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_csgraph_to_dense(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_csgraph_to_dense' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_382347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382347)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_csgraph_to_dense'
    return stypy_return_type_382347

# Assigning a type to the variable 'test_csgraph_to_dense' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'test_csgraph_to_dense', test_csgraph_to_dense)

@norecursion
def test_multiple_edges(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_edges'
    module_type_store = module_type_store.open_function_context('test_multiple_edges', 52, 0, False)
    
    # Passed parameters checking function
    test_multiple_edges.stypy_localization = localization
    test_multiple_edges.stypy_type_of_self = None
    test_multiple_edges.stypy_type_store = module_type_store
    test_multiple_edges.stypy_function_name = 'test_multiple_edges'
    test_multiple_edges.stypy_param_names_list = []
    test_multiple_edges.stypy_varargs_param_name = None
    test_multiple_edges.stypy_kwargs_param_name = None
    test_multiple_edges.stypy_call_defaults = defaults
    test_multiple_edges.stypy_call_varargs = varargs
    test_multiple_edges.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_edges', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_edges', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_edges(...)' code ##################

    
    # Call to seed(...): (line 54)
    # Processing the call arguments (line 54)
    int_382351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_382352 = {}
    # Getting the type of 'np' (line 54)
    np_382348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 54)
    random_382349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), np_382348, 'random')
    # Obtaining the member 'seed' of a type (line 54)
    seed_382350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), random_382349, 'seed')
    # Calling seed(args, kwargs) (line 54)
    seed_call_result_382353 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), seed_382350, *[int_382351], **kwargs_382352)
    
    
    # Assigning a Call to a Name (line 55):
    
    # Call to random(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_382357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    int_382358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 26), tuple_382357, int_382358)
    # Adding element type (line 55)
    int_382359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 26), tuple_382357, int_382359)
    
    # Processing the call keyword arguments (line 55)
    kwargs_382360 = {}
    # Getting the type of 'np' (line 55)
    np_382354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 55)
    random_382355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), np_382354, 'random')
    # Obtaining the member 'random' of a type (line 55)
    random_382356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), random_382355, 'random')
    # Calling random(args, kwargs) (line 55)
    random_call_result_382361 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), random_382356, *[tuple_382357], **kwargs_382360)
    
    # Assigning a type to the variable 'X' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'X', random_call_result_382361)
    
    # Assigning a Call to a Name (line 56):
    
    # Call to csr_matrix(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'X' (line 56)
    X_382363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'X', False)
    # Processing the call keyword arguments (line 56)
    kwargs_382364 = {}
    # Getting the type of 'csr_matrix' (line 56)
    csr_matrix_382362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'csr_matrix', False)
    # Calling csr_matrix(args, kwargs) (line 56)
    csr_matrix_call_result_382365 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), csr_matrix_382362, *[X_382363], **kwargs_382364)
    
    # Assigning a type to the variable 'Xcsr' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'Xcsr', csr_matrix_call_result_382365)
    
    # Assigning a Subscript to a Subscript (line 59):
    
    # Obtaining the type of the subscript
    int_382366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 37), 'int')
    int_382367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 40), 'int')
    slice_382368 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 24), int_382366, None, int_382367)
    # Getting the type of 'Xcsr' (line 59)
    Xcsr_382369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'Xcsr')
    # Obtaining the member 'indices' of a type (line 59)
    indices_382370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), Xcsr_382369, 'indices')
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___382371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 24), indices_382370, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_382372 = invoke(stypy.reporting.localization.Localization(__file__, 59, 24), getitem___382371, slice_382368)
    
    # Getting the type of 'Xcsr' (line 59)
    Xcsr_382373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'Xcsr')
    # Obtaining the member 'indices' of a type (line 59)
    indices_382374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), Xcsr_382373, 'indices')
    int_382375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'int')
    slice_382376 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 4), None, None, int_382375)
    # Storing an element on a container (line 59)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 4), indices_382374, (slice_382376, subscript_call_result_382372))
    
    # Assigning a Call to a Name (line 62):
    
    # Call to toarray(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_382379 = {}
    # Getting the type of 'Xcsr' (line 62)
    Xcsr_382377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'Xcsr', False)
    # Obtaining the member 'toarray' of a type (line 62)
    toarray_382378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 13), Xcsr_382377, 'toarray')
    # Calling toarray(args, kwargs) (line 62)
    toarray_call_result_382380 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), toarray_382378, *[], **kwargs_382379)
    
    # Assigning a type to the variable 'Xdense' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'Xdense', toarray_call_result_382380)
    
    # Call to assert_array_almost_equal(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Obtaining the type of the subscript
    slice_382382 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 63, 30), None, None, None)
    int_382383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'int')
    int_382384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'int')
    slice_382385 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 63, 30), int_382383, None, int_382384)
    # Getting the type of 'Xdense' (line 63)
    Xdense_382386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'Xdense', False)
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___382387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 30), Xdense_382386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_382388 = invoke(stypy.reporting.localization.Localization(__file__, 63, 30), getitem___382387, (slice_382382, slice_382385))
    
    
    # Obtaining the type of the subscript
    slice_382389 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 30), None, None, None)
    int_382390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'int')
    slice_382391 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 30), None, None, int_382390)
    # Getting the type of 'X' (line 64)
    X_382392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___382393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), X_382392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_382394 = invoke(stypy.reporting.localization.Localization(__file__, 64, 30), getitem___382393, (slice_382389, slice_382391))
    
    
    # Obtaining the type of the subscript
    slice_382395 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 42), None, None, None)
    int_382396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 47), 'int')
    int_382397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 50), 'int')
    slice_382398 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 42), int_382396, None, int_382397)
    # Getting the type of 'X' (line 64)
    X_382399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___382400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 42), X_382399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_382401 = invoke(stypy.reporting.localization.Localization(__file__, 64, 42), getitem___382400, (slice_382395, slice_382398))
    
    # Applying the binary operator '+' (line 64)
    result_add_382402 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 30), '+', subscript_call_result_382394, subscript_call_result_382401)
    
    # Processing the call keyword arguments (line 63)
    kwargs_382403 = {}
    # Getting the type of 'assert_array_almost_equal' (line 63)
    assert_array_almost_equal_382381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 63)
    assert_array_almost_equal_call_result_382404 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), assert_array_almost_equal_382381, *[subscript_call_result_382388, result_add_382402], **kwargs_382403)
    
    
    # Assigning a Call to a Name (line 67):
    
    # Call to csgraph_to_dense(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'Xcsr' (line 67)
    Xcsr_382406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'Xcsr', False)
    # Processing the call keyword arguments (line 67)
    kwargs_382407 = {}
    # Getting the type of 'csgraph_to_dense' (line 67)
    csgraph_to_dense_382405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'csgraph_to_dense', False)
    # Calling csgraph_to_dense(args, kwargs) (line 67)
    csgraph_to_dense_call_result_382408 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), csgraph_to_dense_382405, *[Xcsr_382406], **kwargs_382407)
    
    # Assigning a type to the variable 'Xdense' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'Xdense', csgraph_to_dense_call_result_382408)
    
    # Call to assert_array_almost_equal(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining the type of the subscript
    slice_382410 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 30), None, None, None)
    int_382411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 40), 'int')
    int_382412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 43), 'int')
    slice_382413 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 30), int_382411, None, int_382412)
    # Getting the type of 'Xdense' (line 68)
    Xdense_382414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'Xdense', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___382415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), Xdense_382414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_382416 = invoke(stypy.reporting.localization.Localization(__file__, 68, 30), getitem___382415, (slice_382410, slice_382413))
    
    
    # Call to minimum(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Obtaining the type of the subscript
    slice_382419 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 41), None, None, None)
    int_382420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 48), 'int')
    slice_382421 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 41), None, None, int_382420)
    # Getting the type of 'X' (line 69)
    X_382422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 41), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___382423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 41), X_382422, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_382424 = invoke(stypy.reporting.localization.Localization(__file__, 69, 41), getitem___382423, (slice_382419, slice_382421))
    
    
    # Obtaining the type of the subscript
    slice_382425 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 52), None, None, None)
    int_382426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 57), 'int')
    int_382427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 60), 'int')
    slice_382428 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 52), int_382426, None, int_382427)
    # Getting the type of 'X' (line 69)
    X_382429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 52), 'X', False)
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___382430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 52), X_382429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_382431 = invoke(stypy.reporting.localization.Localization(__file__, 69, 52), getitem___382430, (slice_382425, slice_382428))
    
    # Processing the call keyword arguments (line 69)
    kwargs_382432 = {}
    # Getting the type of 'np' (line 69)
    np_382417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'np', False)
    # Obtaining the member 'minimum' of a type (line 69)
    minimum_382418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 30), np_382417, 'minimum')
    # Calling minimum(args, kwargs) (line 69)
    minimum_call_result_382433 = invoke(stypy.reporting.localization.Localization(__file__, 69, 30), minimum_382418, *[subscript_call_result_382424, subscript_call_result_382431], **kwargs_382432)
    
    # Processing the call keyword arguments (line 68)
    kwargs_382434 = {}
    # Getting the type of 'assert_array_almost_equal' (line 68)
    assert_array_almost_equal_382409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 68)
    assert_array_almost_equal_call_result_382435 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), assert_array_almost_equal_382409, *[subscript_call_result_382416, minimum_call_result_382433], **kwargs_382434)
    
    
    # ################# End of 'test_multiple_edges(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_edges' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_382436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_382436)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_edges'
    return stypy_return_type_382436

# Assigning a type to the variable 'test_multiple_edges' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'test_multiple_edges', test_multiple_edges)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
